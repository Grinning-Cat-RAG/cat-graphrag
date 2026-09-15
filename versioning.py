"""Versioned-schema decorators for the GraphRAG handler (todo 11, P4).

Import-safe: no side effects at import time (the Cat plugin loader imports
every ``.py`` in the plugin folder).

Two decorators implement the converged detect-and-rerun-once design:

- ``@ensure_version`` — single-query methods: probe the generation token,
  rebuild the versioned names on drift, then run the query ONCE (no retry,
  no ``WHERE`` guard, no canary, no transactions).
- ``@retry_on_generation_change`` — multi-query methods: probe before entry,
  run the whole function under that snapshot, probe after; if the generation
  changed mid-run, re-initialise the versioned names and re-run the whole
  function once (max 2 attempts total; a second mismatch raises
  ``RuntimeError``). No transactions.
"""

import functools
from typing import Any, Callable


def ensure_version(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Single-query decorator: probe generation -> rebuild on drift -> run once.

    The decorated method must be an async method of a handler exposing
    ``_read_generation`` and ``_rebuild_for_generation`` (see
    ``epoch.EpochMixin``). The data query is executed exactly once — never
    retried — and the versioned names are re-suffixed only when the probed
    generation differs from the cached one.
    """

    @functools.wraps(fn)
    async def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        gen = await self._read_generation()
        if gen != getattr(self, "_generation", None):
            self._rebuild_for_generation(gen)
        return await fn(self, *args, **kwargs)

    return wrapper


def retry_on_generation_change(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Multi-query decorator: before/after probe -> whole-fn re-run once.

    ``before = probe()`` at entry; the whole function runs under that
    snapshot; ``after = probe()``; if ``before != after`` the versioned names
    are re-initialised for the new generation and the whole function is
    re-run once (max 2 attempts total). A second mismatch raises
    ``RuntimeError``. No transactions.
    """

    @functools.wraps(fn)
    async def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        before = await self._read_generation()
        if before != getattr(self, "_generation", None):
            self._rebuild_for_generation(before)

        result = await fn(self, *args, **kwargs)

        after = await self._read_generation()
        if after == before:
            return result

        # Generation changed mid-run -> re-init and re-run the whole fn once.
        self._rebuild_for_generation(after)
        result = await fn(self, *args, **kwargs)

        final = await self._read_generation()
        if final != after:
            raise RuntimeError(
                f"Generation changed again during retry "
                f"({after} -> {final}); giving up after 2 attempts"
            )
        return result

    return wrapper