"""Tenant-scoped Neo4j write lock — eliminates concurrent-write deadlocks.

Problem
-------
Two concurrent sources (e.g. the embedding phase writing ``:Document`` nodes
while another source's graph phase writes entities/relations) issue overlapping
Neo4j write transactions for the SAME tenant. Neo4j detects the circular wait
and raises ``Neo4j.TransientError.Transaction.DeadlockDetected``. The existing
``with_transient_retry`` only masks the short cases; a long-running write can
stay stuck for minutes (live evidence: an embedding phase stuck 7+ minutes with
the log frozen).

Fix
---
Serialise ALL Neo4j writes for ONE tenant behind a single distributed lock::

    key      : ``graph-write:<tenant_id>``
    primitive: ``cat.db.crud.distributed_lock`` (Redis-backed, cross-worker)
    timeout  : ``CAT_GRAPH_WRITE_LOCK_TIMEOUT_SECONDS`` (default 300 s), used as
               both the lock's auto-release TTL and its acquisition blocking
               timeout.

No raw Redis is used: the sanctioned ``cat.db.crud.distributed_lock`` helper is
the only locking primitive. The key embeds the tenant id, so different tenants
never share a lock (no global lock across tenants).

Lock ordering (MANDATORY)
-------------------------
This lock is taken AFTER the per-source ingestion lock. The core RabbitHole
already holds the lock of the source (``ingestion-source:...``) when its phases
invoke the handler, so the handler must never acquire a source lock itself. The global order is::

    source lock  ->  tenant write lock

Taking it in the opposite order would let a worker hold the write lock while
waiting for a source lock and deadlock against a worker holding the source lock
while waiting for the write lock. The handler never takes the source lock.

Re-entrancy (MANDATORY)
-----------------------
An inner helper already running under the write lock MUST NOT re-acquire it: a
second acquisition on the same key from the same worker is a deadlock. The set
of tenant ids currently locked in the CURRENT task context is tracked in a
``ContextVar``. This is context-local state — NOT coroutine-identity detection —
so it survives arbitrarily deep awaited call chains while still letting
concurrent tasks each take the lock.

Background tasks spawned while the lock is held would inherit the ContextVar and
silently skip acquisition; :func:`detached_write_task` clears the inherited
state so the task takes the lock on its own instead of writing UNLOCKED after
the parent releases.

Failure policy
--------------
On acquisition timeout the helper logs a WARNING naming the tenant and the
operation and RE-RAISES. A write never proceeds unlocked; the phase surfaces the
failure/retry.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import os
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Dict, FrozenSet, Optional

from cat.log import log

try:  # pragma: no cover - always present in the Cat runtime
    from cat.db.crud import distributed_lock
except ImportError:  # standalone test harness without cat.db.crud
    distributed_lock = None  # type: ignore[assignment]

try:  # ``cat.db.crud`` re-exports LockError from redis.exceptions
    from cat.db.crud import LockError as _DistributedLockError
except ImportError:  # pragma: no cover - defensive alias for standalone runs

    class _DistributedLockError(Exception):  # type: ignore[no-redef]
        """Placeholder so ``except`` clauses never fail at import time."""


#: Default timeout (seconds) for the tenant write lock. Generous because a
#: write batch / graph phase body can run for minutes. Env-overridable through
#: ``CAT_GRAPH_WRITE_LOCK_TIMEOUT_SECONDS``.
GRAPH_WRITE_LOCK_TIMEOUT_DEFAULT = 300.0

#: Tenant ids whose write lock is held in the CURRENT task context. Nested
#: calls see their tenant here and skip re-acquisition (re-entrancy).
GRAPH_WRITE_LOCK_HELD: contextvars.ContextVar[FrozenSet[str]] = contextvars.ContextVar(
    "cat_graphrag_graph_write_lock_held", default=frozenset()
)


class TenantWriteLockTimeout(RuntimeError):
    """The tenant write lock could not be acquired within the configured timeout."""


def graph_write_lock_timeout_seconds() -> float:
    """Timeout for the tenant write lock (env-overridable).

    ``CAT_GRAPH_WRITE_LOCK_TIMEOUT_SECONDS`` defaults to 300 s and is used as
    both the lock's auto-release TTL and its acquisition blocking timeout.
    """
    raw = os.environ.get("CAT_GRAPH_WRITE_LOCK_TIMEOUT_SECONDS")
    if raw is None:
        return GRAPH_WRITE_LOCK_TIMEOUT_DEFAULT
    try:
        value = float(raw)
    except (TypeError, ValueError):
        log.warning(
            f"[GraphRAG] Invalid CAT_GRAPH_WRITE_LOCK_TIMEOUT_SECONDS={raw!r}; "
            f"using default {GRAPH_WRITE_LOCK_TIMEOUT_DEFAULT:.0f}s"
        )
        return GRAPH_WRITE_LOCK_TIMEOUT_DEFAULT
    if value <= 0:
        log.warning(
            f"[GraphRAG] Non-positive CAT_GRAPH_WRITE_LOCK_TIMEOUT_SECONDS={value}; "
            f"using default {GRAPH_WRITE_LOCK_TIMEOUT_DEFAULT:.0f}s"
        )
        return GRAPH_WRITE_LOCK_TIMEOUT_DEFAULT
    return value


def _tenant_id_of(handler: Any, tenant_id: Optional[str]) -> str:
    """Resolve the tenant id for a handler call (explicit arg wins).

    ``agent_id`` is always set on a live handler; the ``"unknown"`` fallback
    keeps the key well-formed if a caller passes neither (never expected).
    """
    resolved = tenant_id or getattr(handler, "agent_id", None)
    return str(resolved) if resolved else "unknown"


@asynccontextmanager
async def acquire_tenant_write_lock(
    handler: Any, tenant_id: Optional[str], operation: str
):
    """Acquire — or re-enter — the tenant-scoped Neo4j write lock.

    ``handler`` only needs ``agent_id`` and (for the in-process fallback) a
    ``_local_write_locks`` dict. ``operation`` is a human-readable label used in
    the timeout WARNING.
    """
    tenant_id = _tenant_id_of(handler, tenant_id)
    key = f"graph-write:{tenant_id}"
    timeout = graph_write_lock_timeout_seconds()

    held = GRAPH_WRITE_LOCK_HELD.get()
    if tenant_id in held:
        # Re-entrant call from an already-locked method in this task context.
        yield
        return

    if distributed_lock is None:
        # In-process fallback (standalone harness / Redis unavailable).
        # Lock ordering: taken AFTER the per-source lock held by the caller
        # (the core RabbitHole holds the lock of the source before its phases
        # invoke the handler); never the reverse. See the module docstring.
        # ``is None``, not truthiness: the handler initialises an EMPTY dict,
        # which must be reused — a fresh dict per call would never serialise.
        locks: Optional[Dict[str, asyncio.Lock]] = getattr(handler, "_local_write_locks", None)
        if locks is None:
            locks = {}
            try:
                handler._local_write_locks = locks
            except Exception:  # pragma: no cover - exotic handler objects
                pass
        lock = locks.setdefault(tenant_id, asyncio.Lock())
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            log.warning(
                f"[GraphRAG] Timed out after {timeout:.0f}s acquiring tenant "
                f"write lock '{key}' for operation '{operation}' "
                f"(tenant {tenant_id!r}); aborting the write."
            )
            raise TenantWriteLockTimeout(
                f"Could not acquire tenant write lock {key} for {operation}"
            ) from exc
        token = GRAPH_WRITE_LOCK_HELD.set(held | {tenant_id})
        try:
            yield
        finally:
            GRAPH_WRITE_LOCK_HELD.reset(token)
            lock.release()
        return

    # Lock ordering: taken AFTER the source lock held by the caller (the core
    # RabbitHole holds the lock of the source before its phases invoke the
    # handler); never the reverse. See the module docstring.
    acquired = False
    try:
        async with distributed_lock(key, timeout=timeout, blocking_timeout=timeout):
            acquired = True
            token = GRAPH_WRITE_LOCK_HELD.set(held | {tenant_id})
            try:
                yield
            finally:
                GRAPH_WRITE_LOCK_HELD.reset(token)
    except _DistributedLockError as exc:
        if acquired:
            # The locked body itself raised a lock error: propagate untouched.
            raise
        log.warning(
            f"[GraphRAG] Timed out after {timeout:.0f}s acquiring tenant "
            f"write lock '{key}' for operation '{operation}' "
            f"(tenant {tenant_id!r}); aborting the write. ({exc})"
        )
        raise TenantWriteLockTimeout(
            f"Could not acquire tenant write lock {key} for {operation}"
        ) from exc


def tenant_write_locked(operation: str | None = None):
    """Decorator: serialise an async write method behind the tenant write lock.

    Re-entrancy is handled by the ContextVar in :func:`acquire_tenant_write_lock`;
    nested calls from an already-locked method run without re-acquiring.

    The tenant id is resolved from a ``tenant_id`` argument when the wrapped
    method declares one (positionally or as a kwarg), otherwise from
    ``self.agent_id`` — the handler's own tenant.
    """
    def decorator(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        op = operation or getattr(fn, "__name__", "graph write")
        try:
            params = list(inspect.signature(fn).parameters.values())
            tenant_index = next(
                (i - 1 for i, p in enumerate(params) if p.name == "tenant_id"),
                None,
            )
        except (TypeError, ValueError):  # pragma: no cover - C callables
            tenant_index = None

        @functools.wraps(fn)
        async def wrapper(self, *args, **kwargs):
            tenant_id = kwargs.get("tenant_id")
            if tenant_id is None and tenant_index is not None and len(args) > tenant_index:
                tenant_id = args[tenant_index]
            async with acquire_tenant_write_lock(self, tenant_id, op):
                return await fn(self, *args, **kwargs)

        return wrapper

    return decorator


def detached_write_task(coro: Awaitable[Any]) -> Awaitable[Any]:
    """Run ``coro`` with a CLEARED write-lock context.

    Use for background tasks spawned while the write lock is held: without this,
    ``asyncio.create_task`` copies the parent's ContextVar and the child would
    skip the lock, then write UNLOCKED after the parent releases.
    """
    async def _runner():
        # ContextVars are copied per task; setting the empty set here only
        # affects this child task, so it takes the lock on its own.
        GRAPH_WRITE_LOCK_HELD.set(frozenset())
        return await coro

    return _runner()
