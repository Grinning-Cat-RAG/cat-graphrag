"""Per-source Neo4j graph write planning.

Why this module exists
----------------------
spaCy extraction for every chunk of a source is independent and CPU-bound: it
takes no Neo4j locks and can run in parallel. The WRITES, however, all touch
shared ``:Entity`` nodes, so one write transaction per chunk lets concurrent
transactions acquire the same node locks in different orders and deadlock
(live evidence: transactions 2660<->2661 on NODE 46416 and 2662<->2663 on
NODE 45973, both formula entities of the same source).

The fix is to accumulate the extraction output for a whole source, deduplicate
it, and write it in ONE transaction. A single transaction per source removes
the concurrent-writer set entirely, so there is no lock ordering to get wrong.

``GRAPH_WRITE_BATCH_MAX_ITEMS``
-------------------------------
A *memory guard*, not a tuning knob. It caps how many entities+relations ride
in one ``UNWIND`` payload so a very large source cannot materialise an
unbounded transaction. Above it the writes are split into a SEQUENCE of
transactions — one awaited after another, never concurrent. Raising this value
never enables concurrency; lowering it only splits further.

This module is import-safe (stdlib only) so the plugin host can import it and
the standalone harness can load it by path.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

#: Maximum entities+relations in one write transaction. A memory guard: above
#: it the plan is split into sequential transactions (never concurrent).
GRAPH_WRITE_BATCH_MAX_ITEMS = 5000

Entity = Dict[str, Any]
Mention = Dict[str, Any]
Relation = Dict[str, Any]
WritePlan = Dict[str, List[Dict[str, Any]]]


def _metadata_dict(raw: Any) -> Dict[str, Any]:
    """Parse an entity-metadata JSON string into a dict (tolerant)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _merge_entity_metadata(raws: Sequence[Any]) -> str:
    """Union ``source_documents`` and take the max ``confidence`` across chunks."""
    documents: List[str] = []
    confidence: Optional[float] = None
    for raw in raws:
        data = _metadata_dict(raw)
        for doc_id in data.get("source_documents") or []:
            if doc_id not in documents:
                documents.append(doc_id)
        value = data.get("confidence")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            confidence = float(value) if confidence is None else max(confidence, float(value))
    return json.dumps({"source_documents": documents, "confidence": confidence})


def _merge_source_files(left: Any, right: Any) -> Optional[List[str]]:
    if left is None:
        return list(right) if right is not None else None
    if right is None:
        return list(left)
    merged = list(left)
    for item in right:
        if item not in merged:
            merged.append(item)
    return merged


def _dedupe_entities(entities: Sequence[Entity]) -> List[Entity]:
    """One entry per entity id, metadata unioned across every chunk."""
    grouped: Dict[str, Dict[str, Any]] = {}
    metadata: Dict[str, List[Any]] = {}
    for entity in entities:
        entity_id = entity["id"]
        metadata.setdefault(entity_id, []).append(entity.get("metadata"))
        current = grouped.get(entity_id)
        if current is None:
            grouped[entity_id] = dict(entity)
            continue
        if current.get("embedding") is None and entity.get("embedding") is not None:
            current["embedding"] = entity["embedding"]
        if not current.get("name") and entity.get("name"):
            current["name"] = entity["name"]
        if not current.get("type") and entity.get("type"):
            current["type"] = entity["type"]

    for entity_id, payload in grouped.items():
        payload["metadata"] = _merge_entity_metadata(metadata.get(entity_id, []))

    return sorted(grouped.values(), key=lambda item: item["id"])


def _dedupe_mentions(mentions: Sequence[Mention]) -> List[Mention]:
    """One ``(document, entity)`` pair, keeping the highest confidence."""
    grouped: Dict[Tuple[str, str], Mention] = {}
    for mention in mentions:
        key = (mention["doc_id"], mention["entity_id"])
        current = grouped.get(key)
        if current is None:
            grouped[key] = dict(mention)
            continue
        current["confidence"] = max(
            float(current.get("confidence") or 0.0),
            float(mention.get("confidence") or 0.0),
        )
    return sorted(grouped.values(), key=lambda item: (item["doc_id"], item["entity_id"]))


def _dedupe_relations(relations: Sequence[Relation]) -> List[Relation]:
    """One edge per ``(source_id, target_id, rel_type)``.

    Matching the previous per-document behaviour, a duplicate weight is
    averaged and ``source_files`` is unioned.
    """
    grouped: Dict[Tuple[str, str, str], Relation] = {}
    for relation in relations:
        key = (
            relation["source_id"],
            relation["target_id"],
            relation["rel_type"],
        )
        current = grouped.get(key)
        if current is None:
            grouped[key] = dict(relation)
            continue
        current["weight"] = (
            float(current.get("weight") or 0.0) + float(relation.get("weight") or 0.0)
        ) / 2.0
        current["source_files"] = _merge_source_files(
            current.get("source_files"), relation.get("source_files")
        )
    return sorted(
        grouped.values(),
        key=lambda item: (item["source_id"], item["target_id"], item["rel_type"]),
    )


def prepare_source_graph(
    entities: Sequence[Entity],
    mentions: Sequence[Mention],
    relations: Sequence[Relation],
) -> tuple[List[Entity], List[Mention], List[Relation]]:
    """Deduplicate the accumulated extraction output of one source."""
    return (
        _dedupe_entities(entities),
        _dedupe_mentions(mentions),
        _dedupe_relations(relations),
    )


def plan_write_transactions(
    entities: Sequence[Entity],
    mentions: Sequence[Mention],
    relations: Sequence[Relation],
    max_items: int = GRAPH_WRITE_BATCH_MAX_ITEMS,
) -> List[WritePlan]:
    """Plan the transaction bodies for one source.

    Below the memory guard this returns exactly ONE plan (one transaction).
    Above it, entities (with their mentions) are written first in sequential
    batches, then relations — so every relation endpoint already exists when
    its batch runs. The plans are always executed one at a time by the caller.
    """
    if max_items < 1:
        raise ValueError("max_items must be >= 1")

    if len(entities) + len(relations) <= max_items:
        return [
            {
                "entities": list(entities),
                "mentions": list(mentions),
                "relations": list(relations),
            }
        ]

    plans: List[WritePlan] = []
    mentions_by_entity: Dict[str, List[Mention]] = {}
    for mention in mentions:
        mentions_by_entity.setdefault(mention["entity_id"], []).append(mention)

    for start in range(0, len(entities), max_items):
        entity_chunk = list(entities[start : start + max_items])
        entity_ids = {entity["id"] for entity in entity_chunk}
        chunk_mentions = [
            mention
            for entity_id in entity_ids
            for mention in mentions_by_entity.get(entity_id, [])
        ]
        plans.append(
            {
                "entities": entity_chunk,
                "mentions": chunk_mentions,
                "relations": [],
            }
        )

    for start in range(0, len(relations), max_items):
        plans.append(
            {
                "entities": [],
                "mentions": [],
                "relations": list(relations[start : start + max_items]),
            }
        )

    return plans


async def write_source_graph(
    entities: Sequence[Entity],
    mentions: Sequence[Mention],
    relations: Sequence[Relation],
    execute_transaction: Callable[[WritePlan], Awaitable[Any]],
    max_items: int = GRAPH_WRITE_BATCH_MAX_ITEMS,
) -> int:
    """Deduplicate, plan and execute the writes for one source.

    ``execute_transaction`` is awaited once per plan, strictly sequentially
    (the next plan starts only after the previous one returns). Returns the
    number of transactions executed (1 for a source below the guard).
    """
    entities, mentions, relations = prepare_source_graph(
        entities, mentions, relations
    )
    plans = plan_write_transactions(entities, mentions, relations, max_items)
    for plan in plans:
        await execute_transaction(plan)
    return len(plans)
