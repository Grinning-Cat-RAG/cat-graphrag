"""The two graph phases the plugin adds to the ingestion of the sources (``cat.rabbit_hole.phases``).

Registered through the ``rabbithole_ingestion_phases`` hook (``main.py``) where the vector memory of the agent is
GraphRAG; the RabbitHole instantiates them without arguments for every run, so they keep no state: everything comes
from ``cat`` (the CheshireCat of the agent) and ``source``.

- ``graph_spacy_extraction`` (after ``embedding``): the NER, entity linking and similarity graph of the source;
- ``graph_llm_extraction`` (after the spaCy phase): the concepts and relations the agent's LLM extracts.

Both write the SAME graph (entities, PROVENANCE, RELATED_TO): the spaCy phase replaces the whole graph of the source
(it purges it before writing), so the LLM phase depends on it and runs again whenever it runs.

The tag of a phase (``settings``) holds only what no dedicated job owns: the spaCy models and the two toggles (the
concept toggle is in the spaCy tag too: turning the concepts off must re-run the purge that removes them). The
technology patterns are refreshed on save (``refresh_technology_entities``), and the concept definitions and the
prompt switch generation atomically (the concept recompute in ``main.py``): putting them in the tags would run the
same work twice.

``graphrag_handler`` is imported lazily (FX-9: the plugin loader reloads ``main.py`` before ``graphrag_handler.py``,
so a module-level binding would compare against the PRE-reload class and every ``isinstance`` guard would fail).
"""
from __future__ import annotations

from typing import Any, Tuple

from cat.rabbit_hole.phases import PHASE_EMBEDDING, IngestionPhase, IngestionSource, PhaseResult
from cat.services.memory.models import VectorMemoryType

GRAPH_SPACY_PHASE = "graph_spacy_extraction"
GRAPH_LLM_PHASE = "graph_llm_extraction"

_INACTIVE = "graph backend is not active for this agent"
_STATUSES = ("done", "not_ready", "skipped", "error")


def graphrag_handler(cat):
    """The GraphRAG handler of the agent, or ``None`` when its vector memory is another one."""
    from .graphrag_handler import GraphRAGHandler

    handler = getattr(cat, "vector_memory_handler", None)
    return handler if isinstance(handler, GraphRAGHandler) else None


def collection_of(source: IngestionSource) -> str:
    """The collection holding the points of the source: the knowledge base, or the episodic memory of its chat."""
    return str(VectorMemoryType.DECLARATIVE if source.chat_id is None else VectorMemoryType.EPISODIC)


def _as_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def to_result(phase_id: str, raw: Any) -> PhaseResult:
    """The handler's status dict as a ``PhaseResult``; anything else is an error, never a silent success."""
    if not isinstance(raw, dict) or raw.get("status") not in _STATUSES:
        return PhaseResult(status="error", reason=f"phase '{phase_id}' returned {raw!r}")
    retry_after = raw.get("retry_after")
    return PhaseResult(
        status=raw["status"],
        reason=raw.get("reason"),
        retry_after=float(retry_after) if isinstance(retry_after, (int, float)) else None,
        processed=_as_count(raw.get("processed")),
        expected=_as_count(raw.get("expected")),
        warnings=raw.get("warnings") or None,
    )


class _GraphPhase(IngestionPhase):
    async def applies_to(self, source: IngestionSource, cat) -> Tuple[bool, str | None]:
        if graphrag_handler(cat) is None:
            return False, _INACTIVE
        return True, None

    async def delete(self, source: IngestionSource, cat) -> None:
        """Remove the graph the source contributed (both phases write one graph): its documents' entity edges and
        similarity edges, its RELATED_TO provenance and the entities left orphaned, plus its pending extraction.
        Idempotent; the documents themselves go with the points of the source (``embedding``)."""
        handler = graphrag_handler(cat)
        if handler is None:
            return
        await handler.purge_source_graph_artifacts(source.name, collection_of(source), source.chat_id)


class GraphSpacyPhase(_GraphPhase):
    id = GRAPH_SPACY_PHASE
    label = "Extracting the entities"
    depends_on = (PHASE_EMBEDDING,)

    async def settings(self, cat) -> Any:
        handler = graphrag_handler(cat)
        return None if handler is None else handler.spacy_phase_settings()

    async def process(self, source: IngestionSource, cat) -> PhaseResult:
        handler = graphrag_handler(cat)
        if handler is None:
            return PhaseResult(status="skipped", reason=_INACTIVE)
        raw = await handler.run_spacy_extraction_phase(source.name, collection_of(source), source.chat_id)
        return to_result(self.id, raw)


class GraphConceptPhase(_GraphPhase):
    id = GRAPH_LLM_PHASE
    label = "Extracting the concepts"
    depends_on = (GRAPH_SPACY_PHASE,)
    # an LLM-backed phase must not hold its source for the generic hour of ``not_ready``
    not_ready_budget_seconds = 300.0

    async def settings(self, cat) -> Any:
        handler = graphrag_handler(cat)
        return None if handler is None else handler.concept_phase_settings()

    async def process(self, source: IngestionSource, cat) -> PhaseResult:
        handler = graphrag_handler(cat)
        if handler is None:
            return PhaseResult(status="skipped", reason=_INACTIVE)
        raw = await handler.run_llm_extraction_phase(source.name, collection_of(source), source.chat_id, cat)
        return to_result(self.id, raw)


__all__ = ["GRAPH_LLM_PHASE", "GRAPH_SPACY_PHASE", "GraphConceptPhase", "GraphSpacyPhase", "collection_of", "to_result"]
