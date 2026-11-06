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

``with_transient_retry`` retries a write operation ONLY on Neo4j
``TransientError`` (deadlock / lock / leader / termination), with bounded
exponential backoff and jitter.
"""

import asyncio
import functools
import random
from typing import Any, Awaitable, Callable

from cat.log import log

# Only transient failures are retried. ``TransientError`` covers the lock /
# leader / termination family (``DeadlockDetected``, ``LockClientStopped``,
# ``LeaderSwitch``, ``Termination``); anything else must propagate untouched.
# The guarded alias keeps this module importable by the standalone harnesses
# whose neo4j stub only defines ``Neo4jError``; the empty-tuple fallback
# matches nothing, so an unresolved class never causes retry-everything.
_TRANSIENT_ERROR_TYPES: tuple[type[BaseException], ...]
try:  # pragma: no cover - the fallback only matters for those test stubs
    from neo4j.exceptions import TransientError as _TransientError
    _TRANSIENT_ERROR_TYPES = (_TransientError,)
except ImportError:  # pragma: no cover
    _TRANSIENT_ERROR_TYPES = ()

# Upper bound for a single backoff sleep, independent of the attempt number.
_MAX_RETRY_DELAY = 1.0


def _retry_delay(attempt: int, base_delay: float, max_delay: float) -> float:
    """Exponential backoff for ``attempt`` (1-based) with jitter, capped.

    The cap is applied AFTER jitter so the returned delay is always
    ``0 < delay <= max_delay`` regardless of attempt or jitter draw.
    """
    raw = base_delay * (2 ** (attempt - 1))
    return min(max_delay, raw * (0.5 + random.random()))


async def with_transient_retry(
    coro_factory: Callable[[], Awaitable[Any]],
    *,
    attempts: int = 4,
    base_delay: float = 0.05,
    max_delay: float = _MAX_RETRY_DELAY,
    operation: str = "Neo4j write",
) -> Any:
    """Run ``coro_factory()`` retrying ONLY Neo4j transient errors.

    A deadlock/lock/leader/termination failure is transient: retry with
    exponential backoff + jitter up to ``attempts`` total tries, logging one
    WARNING per retry. When the attempts are exhausted, or for any
    non-transient error, the ORIGINAL exception is re-raised (never swallowed).

    ``coro_factory`` must be a zero-argument callable returning a FRESH
    awaitable, so every attempt starts a new coroutine.
    """
    if attempts < 1:
        attempts = 1

    for attempt in range(1, attempts + 1):
        try:
            return await coro_factory()
        except _TRANSIENT_ERROR_TYPES as exc:
            if attempt >= attempts:
                raise
            delay = _retry_delay(attempt, base_delay, max_delay)
            log.warning(
                f"[GraphRAG] Transient Neo4j error during {operation} "
                f"(attempt {attempt}/{attempts}): {exc}; retrying in {delay:.3f}s"
            )
            await asyncio.sleep(delay)


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
