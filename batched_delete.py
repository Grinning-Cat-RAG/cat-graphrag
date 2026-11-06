"""Bounded, sequential batching for large Neo4j deletes.

Why this module exists
----------------------
Deleting every chunk/point of a large source in ONE transaction can exceed
Neo4j's transaction memory limit: a live re-embed of two 37 MB sources staged
529 points each and the single ``DETACH DELETE`` transaction failed with
``Neo.TransientError.General.MemoryPoolOutOfMemoryError`` ("the allocation of an
extra 2.0 MiB would use more than the limit 716.8 MiB", total transaction max
716.8 MiB). The estimate from that failure is ~1.35 MB of transaction memory per
point-with-edges, so 529 points ≈ 714 MiB — right at the limit.

The fix is the same shape already used for graph WRITES: resolve the node ids
with a cheap read, then delete them in a SEQUENCE of small transactions — one
awaited after another, never concurrent.

``DELETE_BATCH_MAX_ITEMS``
--------------------------
A *memory guard*, not a tuning knob. At 100 points-with-edges a batch stays at
~135 MB of transaction memory, comfortably under the 716.8 MiB ceiling even with
the tenant's other work in flight. It is a module constant with NO environment
override: raising it without bound would re-create the OOM this module exists to
prevent. ``delete_in_batches`` rejects a larger ``max_items`` for that reason;
callers may only lower it.

This module is import-safe (stdlib only) so the plugin host can import it and
the standalone harness can load it by path.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

#: Maximum ids deleted in one transaction. A memory guard: 100 points-with-edges
#: ≈ 135 MB of transaction memory, well under the ~716.8 MiB Neo4j limit that a
#: 529-point single-transaction delete exceeded. Never raise without measuring.
DELETE_BATCH_MAX_ITEMS = 100

T = TypeVar("T")


def chunk_ids(ids: Iterable[T], max_items: int = DELETE_BATCH_MAX_ITEMS) -> list[list[T]]:
    """Split ``ids`` into sequential chunks of at most ``max_items``."""
    if max_items < 1:
        raise ValueError("max_items must be >= 1")
    if max_items > DELETE_BATCH_MAX_ITEMS:
        message = (
            f"max_items ({max_items}) exceeds the DELETE_BATCH_MAX_ITEMS "
            f"memory guard ({DELETE_BATCH_MAX_ITEMS})"
        )
        raise ValueError(message)
    materialized = list(ids)
    return [
        materialized[start : start + max_items]
        for start in range(0, len(materialized), max_items)
    ]


async def delete_in_batches(
    ids: Iterable[T],
    delete_batch: Callable[[list[T]], Awaitable[None]],
    max_items: int = DELETE_BATCH_MAX_ITEMS,
) -> int:
    """Await ``delete_batch`` once per chunk, strictly sequentially.

    Returns the number of transactions executed (1 for any list that fits in one
    batch, 0 for an empty list). Idempotency is the caller's responsibility:
    deleting ids that no longer exist is a no-op in Neo4j.
    """
    batches = chunk_ids(ids, max_items)
    for batch in batches:
        await delete_batch(batch)
    return len(batches)
