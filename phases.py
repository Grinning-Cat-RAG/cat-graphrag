"""Class-based ingestion phases for the Neo4j GraphRAG plugin.

The two graph phases (``graph_spacy_extraction`` / ``graph_llm_extraction``)
are exposed as :class:`IngestionPhase` instances through the
``ingestion_phase_objects`` getter hook. Their bodies delegate to the SAME
graph work the legacy hooks execute (``run_spacy_extraction_phase`` /
``run_llm_extraction_phase``): this module adds the ownership contract
(applicability, cleanup, tag), it never re-implements extraction.

Import-safe: the EffING contract is imported at module top inside a
``try/except`` (the layer-3 provider contract), with a local fallback that
keeps the module importable when EffING is absent. ``graphrag_handler`` is
imported lazily (FX-9: the plugin loader reloads ``main.py`` before
``graphrag_handler.py``, so a module-level binding would compare against the
PRE-reload class and every ``isinstance`` guard would fail).

``delete`` is source-scoped and idempotent: it only removes the graph
artefacts derived from the given source (``purge_source_graph_artifacts``),
never the chunk Documents, other sources or other tenants.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

try:
    from cat.plugins.cat_efficient_ingestion.phases import (
        IngestionPhase,
        PhaseResult,
    )
    from cat.plugins.cat_efficient_ingestion.phase_tag_resolver import own_tag
except Exception:  # noqa: BLE001 - EffING may be absent; keep the module import-safe
    from abc import ABC, abstractmethod
    from dataclasses import dataclass

    def _canonical_json(value: Any) -> str:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )

    def own_tag(kind: str, relevant_settings: Any) -> str:
        return hashlib.sha256(
            f"{kind}:{_canonical_json(relevant_settings)}".encode("utf-8")
        ).hexdigest()

    @dataclass(frozen=True)
    class PhaseResult:
        status: str
        reason: str | None = None
        processed: int | None = None
        expected: int | None = None
        warnings: list[dict] | None = None

    class IngestionPhase(ABC):
        """Minimal stand-in mirroring the EffING ABC."""

        id: str = ""
        depends_on: tuple[str, ...] = ()
        settings_category: str | None = None
        settings_check: Any = None

        @property
        def name(self) -> str:
            return self.id

        @abstractmethod
        async def applies_to(self, cat, deps_state) -> tuple[bool, str | None]:
            ...

        @abstractmethod
        async def process(self, source: str, cat) -> PhaseResult:
            ...

        @abstractmethod
        async def delete(self, source: str, cat) -> None:
            ...

        async def tag(self, settings: Any = None) -> str:
            return own_tag(self.id, {} if settings is None else settings)


_STATUSES = ("done", "not_ready", "error", "skipped")


def _as_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _to_result(phase_id: str, raw: Any) -> PhaseResult:
    """Normalize a legacy status dict into :class:`PhaseResult`."""
    if isinstance(raw, PhaseResult):
        return raw
    if isinstance(raw, dict):
        status = raw.get("status")
        if status in _STATUSES:
            return PhaseResult(
                status=status,
                reason=raw.get("reason"),
                processed=_as_count(raw.get("processed")),
                expected=_as_count(raw.get("expected")),
                warnings=raw.get("warnings") or None,
            )
        return PhaseResult(
            status="error",
            reason=f"phase '{phase_id}' returned invalid status {status!r}",
        )
    return PhaseResult(
        status="error",
        reason=f"phase '{phase_id}' returned {raw!r} (expected a status dict)",
    )


def graphrag_handler(cat):
    """Resolve the active GraphRAG handler for ``cat``, or ``None``.

    The presence of a :class:`GraphRAGHandler` as ``cat.vector_memory_handler``
    IS the "graph backend active for this agent" condition: the factory only
    instantiates it when the Neo4j backend is the agent's selected vector
    memory. A pure-vector agent has a different handler, so both graph phases
    report themselves inapplicable (``skipped``) instead of ``pending``.

    Lazy import (FX-9) + the current class keeps the guard correct across the
    plugin loader's file-by-file reload.
    """
    from .graphrag_handler import GraphRAGHandler

    handler = getattr(cat, "vector_memory_handler", None)
    if isinstance(handler, GraphRAGHandler):
        return handler
    return None


class _GraphPhase(IngestionPhase):
    """Shared applicability/cleanup/tag contract for both graph phases."""

    depends_on: tuple[str, ...] = ("parsing_chunking", "embedding")
    settings_category: str | None = None

    def __init__(self, cat=None):
        # ``cat`` is captured because ``tag(settings)`` only receives the
        # resolved settings slice, while the fingerprint is read from the live
        # handler.
        self._cat = cat

    async def applies_to(self, cat, deps_state) -> tuple[bool, str | None]:
        handler = graphrag_handler(cat if cat is not None else self._cat)
        if handler is None:
            return False, "graph backend is not active for this agent"
        return True, None

    async def delete(self, source: str, cat) -> None:
        """Remove ONLY this source's graph artefacts (idempotent).

        Delegates to ``purge_source_graph_artifacts``: PROVENANCE/MENTIONS and
        SIMILAR_TO edges of the source's Documents, the source's contribution
        to RELATED_TO ``source_files`` and entities left orphaned by that
        removal. Chunk Documents, the document-structure graph (owned by the
        embedding phase), other sources and other tenants are untouched.
        Re-running it is a no-op.

        Both graph phases write ONE shared graph (entities, PROVENANCE,
        RELATED_TO), so the purge also removes the SIBLING phase's product.
        EffING's recovery re-runs only the abandoned phase and the later ones,
        so the sibling's diary entry is dropped here as well: both phases then
        re-run instead of the sibling staying ``complete`` over a deleted
        graph. A failure to do so raises, so EffING retries the delete.
        """
        cat = cat if cat is not None else self._cat
        handler = graphrag_handler(cat)
        if handler is None:
            return
        await handler.purge_source_graph_artifacts(source)
        await _requeue_sibling_phase(self.id, handler, source, cat)

    async def _fingerprint(self, handler) -> str:
        raise NotImplementedError

    async def _execute(self, handler, source: str, cat) -> Any:
        """Delegate the phase body to the existing ingestion code."""
        raise NotImplementedError

    async def _run(self, source: str, cat) -> Any:
        """Raw status dict of the phase body, or ``None`` when inactive."""
        handler = graphrag_handler(cat if cat is not None else self._cat)
        if handler is None:
            return None
        return await self._execute(handler, source, cat)

    async def tag(self, settings: Any = None) -> str:
        """Stable tag of the phase's OWN material configuration.

        Reuses the existing fingerprints (configuration only, NO LLM call) and
        wraps them with EffING's ``own_tag`` so the class path is byte-for-byte
        parity with the legacy marker -> ``HookPhase`` tag.
        """
        handler = graphrag_handler(self._cat)
        if handler is None:
            return await super().tag(settings)
        marker = await self._fingerprint(handler)
        return own_tag(self.id, marker)


async def _requeue_sibling_phase(phase_id: str, handler, source: str, cat) -> None:
    """Drop the OTHER graph phase's diary entry for ``source`` (see ``delete``).

    The scope is the one EffING's delete orchestrator publishes; without it the
    same fallback EffING uses applies (a caller with an ``id`` is chat-scoped,
    otherwise agent-scoped). Without EffING there is no diary to fix.
    """
    sibling = next(pid for pid in _PHASE_CLASSES if pid != phase_id)
    try:
        from cat.plugins.cat_efficient_ingestion.registry import reset_phase_entries
    except ImportError:
        return
    scope = None
    try:
        from cat.plugins.cat_efficient_ingestion.phases import _DELETE_SCOPE

        scope = _DELETE_SCOPE.get()
    except ImportError:
        pass
    if scope is None:
        caller_id = getattr(cat, "id", None)
        scope = str(caller_id) if caller_id else "agent"
    await reset_phase_entries(handler.agent_id, scope, source, [sibling])


class GraphSpacyPhase(_GraphPhase):
    """spaCy extraction phase (NER + link + similarity)."""

    id = "graph_spacy_extraction"

    async def process(self, source: str, cat) -> PhaseResult:
        raw = await self._run(source, cat)
        if raw is None:
            return PhaseResult(status="skipped", reason="graph backend inactive")
        return _to_result(self.id, raw)

    async def _execute(self, handler, source: str, cat) -> Any:
        return await handler.run_spacy_extraction_phase(source, cat=cat)

    async def _fingerprint(self, handler) -> str:
        return handler._spacy_fingerprint()


class GraphConceptPhase(_GraphPhase):
    """LLM concept-relations phase."""

    id = "graph_llm_extraction"
    # An LLM-backed phase must not hold a source for the generic hour.
    not_ready_budget_seconds = 300.0
    not_ready_budget_env = "CAT_INGESTION_LLM_NOT_READY_MAX_SECONDS"

    async def process(self, source: str, cat) -> PhaseResult:
        raw = await self._run(source, cat)
        if raw is None:
            return PhaseResult(status="skipped", reason="graph backend inactive")
        return _to_result(self.id, raw)

    async def _execute(self, handler, source: str, cat) -> Any:
        return await handler.run_llm_extraction_phase(source, cat=cat)

    async def _fingerprint(self, handler) -> str:
        return await handler._concept_fingerprint()


_PHASE_CLASSES: dict[str, type[_GraphPhase]] = {
    GraphSpacyPhase.id: GraphSpacyPhase,
    GraphConceptPhase.id: GraphConceptPhase,
}


def _phase_id(value: Any) -> str | None:
    """Normalize a phase id, mapping/spec or object with ``.id`` to its id.

    The legacy hooks still receive the EffING PhaseSpec payload (a dict), while
    the class API passes the string id; both must resolve the same class.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        candidate = value.get("id")
        return candidate if isinstance(candidate, str) else None
    candidate = getattr(value, "id", None)
    return candidate if isinstance(candidate, str) else None


def phase_for(phase_id: Any, cat=None) -> _GraphPhase | None:
    """The phase instance for ``phase_id``, or ``None`` for an unknown id.

    ``phase_id`` may be a string id, a mapping carrying ``"id"`` (the legacy
    PhaseSpec payload) or any object exposing ``.id``.
    """
    cls = _PHASE_CLASSES.get(_phase_id(phase_id))
    return None if cls is None else cls(cat)


async def phase_fingerprint(phase_id: str, cat) -> Any:
    """Raw configuration fingerprint for ``phase_id`` (no LLM call)."""
    phase = phase_for(phase_id, cat)
    if phase is None:
        return None
    handler = graphrag_handler(cat if cat is not None else phase._cat)
    if handler is None:
        return None
    return await phase._fingerprint(handler)


async def run_phase(phase_id: str, source: str, cat) -> Any:
    """Run the phase body: the legacy status dict, or ``None`` when inactive."""
    phase = phase_for(phase_id, cat)
    if phase is None:
        return None
    return await phase._run(source, cat)


async def delete_phase(phase_id: str, source: str, cat) -> None:
    """Remove ``source``'s artifacts for ``phase_id`` (idempotent no-op if inactive)."""
    phase = phase_for(phase_id, cat)
    if phase is None:
        return
    await phase.delete(source, cat)


__all__ = [
    "GraphSpacyPhase",
    "GraphConceptPhase",
    "IngestionPhase",
    "PhaseResult",
    "delete_phase",
    "phase_fingerprint",
    "phase_for",
    "run_phase",
]
