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
               timeout. A heartbeat renews the TTL every third of it while the
               write runs, so a write longer than the timeout keeps the lock; a
               writer that dies releases it within the timeout.

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

Handover
--------
``distributed_lock`` (redis-py) polls a busy lock every 100 ms. The writers of
one process queue on a local gate per tenant (FIFO) and only the head of the
queue waits on Redis: a handover inside the process is immediate, the polling
is left to the contention between pods. A streak of local handovers pauses
once a second for longer than a poll, so another pod is not starved. The
timeout bounds the wait on both.

Failure policy
--------------
On acquisition timeout the helper logs a WARNING naming the tenant and the
operation and RE-RAISES. A write never proceeds unlocked; the phase surfaces the
failure/retry.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import inspect
import os
import weakref
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


#: A streak of local handovers longer than this pauses once (``_YIELD_SECONDS``), so another pod gets its turn.
_YIELD_AFTER_SECONDS = 1.0
#: Longer than the polling interval of ``distributed_lock`` (redis-py: 0.1 s): another pod waiting finds the lock free.
_YIELD_SECONDS = 0.15


class _Gate:
    """The local queue of the writers of one tenant in this process (FIFO, immediate handover): only its head waits on
    the distributed lock. ``distributed_lock`` (redis-py) polls a busy lock every 100 ms: without the gate, every
    handover between two writers of the same process left the lock idle until the next poll. With it, the lock is
    free only for an instant between two local writers, so another pod's poll would hardly ever find it free: a streak
    of local handovers (writers waiting all along) pauses once every ``_YIELD_AFTER_SECONDS``."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.waiting = 0
        self.streak_since: Optional[float] = None


#: The gates of each tenant, per event loop (an ``asyncio.Lock`` belongs to one loop).
_GATES: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Dict[str, _Gate]]" = weakref.WeakKeyDictionary()


def _gate(tenant_id: str) -> _Gate:
    gates = _GATES.setdefault(asyncio.get_running_loop(), {})
    return gates.setdefault(tenant_id, _Gate())


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
    # The local gate first (the writers of this process hand over at once), then the distributed lock (against the
    # other pods); the wait on both is bounded by ``timeout`` as a whole.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    gate = _gate(tenant_id)
    gate.waiting += 1
    try:
        await asyncio.wait_for(gate.lock.acquire(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        log.warning(
            f"[GraphRAG] Timed out after {timeout:.0f}s acquiring tenant "
            f"write lock '{key}' for operation '{operation}' "
            f"(tenant {tenant_id!r}); aborting the write."
        )
        raise TenantWriteLockTimeout(
            f"Could not acquire tenant write lock {key} for {operation}"
        ) from exc
    finally:
        gate.waiting -= 1
    acquired = False
    try:
        now = loop.time()
        if gate.streak_since is None:
            gate.streak_since = now
        elif now - gate.streak_since >= _YIELD_AFTER_SECONDS:
            # a long streak of local handovers: the lock stays free for longer than a poll, for another pod
            await asyncio.sleep(min(_YIELD_SECONDS, max(0.0, deadline - now)))
            gate.streak_since = loop.time()
        # no time left after the queue: a single try
        async with distributed_lock(key, timeout=timeout, blocking_timeout=max(0.0, deadline - loop.time())) as lock:
            acquired = True
            token = GRAPH_WRITE_LOCK_HELD.set(held | {tenant_id})
            heartbeat = asyncio.create_task(_renew_while_held(lock, key, timeout))
            try:
                yield
            finally:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat
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
    finally:
        if not gate.waiting:  # nobody behind: the streak ends
            gate.streak_since = None
        gate.lock.release()


async def _renew_while_held(lock: Any, key: str, timeout: float) -> None:
    """The heartbeat of a held lock: its TTL is ``timeout``, so a write longer than that (e.g. the removal of a large
    old graph) would lose it and let another pod in. It is renewed every third of the timeout until the body ends. A
    failed renewal is retried at the next beat; a lock already lost (e.g. Redis restarted) is logged, and not
    renewed anymore."""
    while True:
        await asyncio.sleep(timeout / 3)
        try:
            await lock.extend(timeout, replace_ttl=True)
        except _DistributedLockError as exc:  # LockNotOwnedError: another writer may hold it now
            log.warning(f"[GraphRAG] Tenant write lock '{key}' lost while held ({exc}); not renewed anymore")
            return
        except Exception as exc:  # noqa: BLE001 - retried at the next beat
            log.warning(f"[GraphRAG] Renewal of the tenant write lock '{key}' failed: {exc}")


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
