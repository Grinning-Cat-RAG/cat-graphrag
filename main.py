from typing import List, Dict, Any
import asyncio
from langchain_core.documents import Document

from cat import hook, RecallSettings, VectorDatabaseSettings
from cat.log import log
from cat.looking_glass.stray_cat import StrayCat
from cat.services.memory.models import PointStruct

from .graphrag_handler import Neo4jGraphRAGConfig, GraphRAGHandler
from .entity_extractor import EntityExtractor

# Concept-recompute guard state (todo 12, concurrency lifecycle). Module-level
# (not per-handler) because the handler is re-instantiated across settings
# saves — the PERSISTED ``concept_gen_active`` marker is the real source of
# truth for single-flight; the lock only serializes access to the pending map
# and the dict holds the LATEST-WINS target generation per tenant. Both are
# harmless at import time (no side effects) and therefore import-safe.
_concept_recompute_lock = asyncio.Lock()
_concept_recompute_pending: Dict[str, str] = {}


@hook(priority=10)
def factory_allowed_vector_databases(allowed: List[VectorDatabaseSettings], cat) -> List:
    allowed.append(Neo4jGraphRAGConfig)
    return allowed


@hook(priority=10)
async def after_cheshire_cat_creation(cat) -> None:
    """
    Boot-time provenance migration.

    Every agent's graph that predates the PROVENANCE / ``source_files``
    tracking (nodes and relations created before this feature) is reconciled
    in the background, so the file-deletion cascade works for historical data
    too. ``recompute_provenance`` self-skips tenants that are already
    reconciled (a single ``provenance_reconciled`` marker count), so this is a
    cheap no-op on every boot after the first migration.
    """
    # Lazy import (FX-9): the plugin loader reloads main.py BEFORE
    # graphrag_handler.py (glob order), so a module-level import binds the
    # PRE-reload class and isinstance() below ALWAYS fails (the factory
    # instantiates the handler from the CURRENT, post-reload class). Import at
    # call time, when graphrag_handler is the current module.
    from .graphrag_handler import GraphRAGHandler as _GH
    handler = getattr(cat, "vector_memory_handler", None)
    if not isinstance(handler, _GH):
        return
    task = asyncio.create_task(handler.recompute_provenance())
    handler._pending_entity_tasks.append(task)
    log.info(
        f"[GraphRAG] Scheduled provenance reconciliation for "
        f"{getattr(handler, 'agent_id', 'unknown')}"
    )


@hook(priority=10)
async def before_cat_recalls_memories(config: RecallSettings, cat: StrayCat) -> RecallSettings:
    """
    Injects the current user message and embedder into the GraphRAGHandler
    before any memory retrieval takes place.

    - `user_message` lets the handler extract named entities from the raw query
      and perform direct graph lookups (Phase A② and A③).
    - `embedder` enables entity vector search (Phase A④) and allows entity
      embeddings to be stored during background ingestion tasks.

    Priority 10 ensures this hook runs before the default (priority 0).
    """
    if hasattr(cat.vector_memory_handler, "user_message"):
        cat.vector_memory_handler.user_message = cat.working_memory.user_message.text

    if hasattr(cat.vector_memory_handler, "embedder"):
        cat.vector_memory_handler.embedder = await cat.embedder()
        if hasattr(cat.vector_memory_handler, "_align_embedder_lazy"):
            await cat.vector_memory_handler._align_embedder_lazy()

    return config


@hook(priority=10)
async def before_rabbithole_stores_documents(docs: List[Document], cat) -> List[Document]:
    if hasattr(cat.vector_memory_handler, "embedder"):
        cat.vector_memory_handler.embedder = await cat.embedder()
        if hasattr(cat.vector_memory_handler, "_align_embedder_lazy"):
            await cat.vector_memory_handler._align_embedder_lazy()

    # Lazy import (FX-9): same reload-order rationale as
    # after_cheshire_cat_creation — a module-level binding would be the
    # PRE-reload class and this guard would ALWAYS fail.
    from .graphrag_handler import GraphRAGHandler as _GH
    if isinstance(cat.vector_memory_handler, _GH):
        handler = cat.vector_memory_handler
        if handler.entity_extractor:
            await handler.entity_extractor.ensure_initialized()
        for i, doc in enumerate(docs):
            doc.metadata.setdefault("chunk_index", i)

    return docs


@hook
async def after_rabbithole_stored_documents(source: str, stored_points: List[PointStruct], cat) -> None:
    # Lazy import (FX-7): the plugin loader reloads each file in glob order and
    # main.py comes BEFORE graphrag_handler.py, so a module-level import binds
    # the PRE-reload class and isinstance() below ALWAYS fails (the factory
    # instantiates the handler from the CURRENT, post-reload class). Import at
    # call time, when graphrag_handler is the current module, so the check
    # matches the class the factory actually created.
    from .graphrag_handler import GraphRAGHandler as _GH
    handler = cat.vector_memory_handler
    if not isinstance(handler, _GH):
        return
    # Single source of truth: the handler's config (vector-DB settings), NOT the
    # plugin settings store (which is empty unless explicitly saved — a stale
    # default there silently disabled the whole derived-graph + LLM path, FX-1).
    if not getattr(handler, "_enable_derived_graph", False):
        return
    await handler.create_derived_graph_for_source(source, stored_points, cat)


# ========== INGESTION PHASE MACHINE (T21-T24) ==========
#
# The core dispatcher (feat/ingestion-phase-machine) drives the phase-aware
# ingestion lifecycle through three no-op hooks declared in
# ``cat/core_plugins/base_plugin/hooks/ingestion.py`` plus the phase executor
# ``ingestion_phase_run``. Dispatch is by hook name, so these handlers work
# even before the core patch lands (they simply never fire).


def _graphrag_handler(cat):
    """Resolve the active GraphRAG handler, or None.

    Same guard as every other hook in this module: lazy import (FX-9, the
    plugin loader reloads main.py BEFORE graphrag_handler.py, so a module-level
    binding would compare against the PRE-reload class and always fail) +
    isinstance against the current class.
    """
    from .graphrag_handler import GraphRAGHandler as _GH
    handler = getattr(cat, "vector_memory_handler", None)
    if isinstance(handler, _GH):
        return handler
    return None


@hook(priority=10)
async def ingestion_phase_pending(pending, source, completed_phases, cat) -> list:
    """Probe hook (T21): report ``graph_building`` as stale work.

    ACCUMULATOR convention: the core dispatcher deep-copies the first
    positional arg (``pending``) and threads it through every registrant;
    each registrant receives the current accumulator and its non-None return
    REPLACES it. This hook appends ``{"phase": "graph_building", "gen":
    <recomputed>}`` only when BOTH dependency phases (``parsing_chunking``
    and ``embedding``) are present in ``completed_phases`` AND the recorded
    generation for ``graph_building`` differs from the recomputed one
    (settings fingerprint + concept-relations fingerprint + dependency
    generations). It ALWAYS returns ``pending`` (never ``[]`` — that would
    discard other plugins' entries). Entries are dicts ``{"phase", "gen"}``,
    not bare strings.
    """
    handler = _graphrag_handler(cat)
    if handler is None:
        return pending
    if not getattr(handler, "_enable_derived_graph", False):
        return pending

    dep_ids = {"parsing_chunking", "embedding"}
    completed_ids = {p.get("phase") for p in completed_phases if isinstance(p, dict)}
    if not dep_ids.issubset(completed_ids):
        return pending

    # Core helpers (feat/ingestion-phase-machine). If they are not deployed
    # yet, the phase machine is not running: no work to report.
    try:
        from cat.core_plugins.ingestion_status.fingerprints import (
            build_graphrag_fingerprint,
            phase_generation,
        )
    except Exception:  # noqa: BLE001
        return pending

    # Recorded generation of graph_building (from the full diary), and the
    # dependency generations used to recompute it. graph_building itself is
    # EXCLUDED from dep_gens: the recorded generation is computed from the
    # DEPENDENCY phases only, so including it here would make the recomputed
    # hash self-referential and the phase permanently stale.
    recorded = next(
        (
            p.get("gen")
            for p in completed_phases
            if isinstance(p, dict) and p.get("phase") == "graph_building"
        ),
        None,
    )
    dep_gens = {
        p["phase"]: p.get("gen")
        for p in completed_phases
        if isinstance(p, dict)
        and p.get("phase") is not None
        and p["phase"] != "graph_building"
    }
    fp = {
        "settings": await build_graphrag_fingerprint(cat.agent_key),
        "concept": handler._concept_fingerprint(),
    }
    recomputed = phase_generation("graph_building", fp, dep_gens)
    if recorded == recomputed:
        return pending
    pending.append({"phase": "graph_building", "gen": recomputed})
    return pending


@hook(priority=10)
async def ingestion_phase_run(phase, source, completed_phases, cat):
    """Phase executor (T22): run the graph-building phase inline and awaited.

    Only handles ``phase == "graph_building"`` (returns None otherwise). The
    graph work for ``source`` runs AWAITED (not fire-and-forget): the source's
    background NER tasks are joined, similarity relationships are re-run for
    the source's stored points (needs valid vectors — that is why the
    embedding phase must precede), and the LLM concept-relations step runs
    inline. Returns ``{"status": "done"}`` on success and
    ``{"status": "not_ready", "retry_after": <s>}`` when the source's points
    are not yet in the store. Exceptions propagate (the core maps raise ->
    status=error, fail-hard).
    """
    if phase != "graph_building":
        return None
    handler = _graphrag_handler(cat)
    if handler is None:
        return None
    if not getattr(handler, "_enable_derived_graph", False):
        return None
    return await handler.run_graph_building_phase(source, cat=cat)


@hook(priority=10)
async def before_ingestion_status_completed(source, cat) -> None:
    """Completion gate (T23): fail-hard when graph work is still pending.

    Raises when live background tasks remain for the source or when the
    ``graph_building`` generation was not recorded in the status doc, so the
    core writes status=error. No-op when graphrag is not the active handler.
    """
    handler = _graphrag_handler(cat)
    if handler is None:
        return
    if not getattr(handler, "_enable_derived_graph", False):
        return

    if handler.has_pending_source_tasks(source):
        raise RuntimeError(
            f"[GraphRAG] Pending background graph tasks for source '{source}'"
        )

    # The graph phase's generation must have been recorded in the status doc's
    # completed-phases diary. Registry unavailable -> skip the doc check (the
    # in-memory pending-task check above already passed).
    try:
        from cat.core_plugins.ingestion_status.registry import get_status
    except Exception:  # noqa: BLE001
        return
    scope = cat.id if hasattr(cat, "id") else "agent"
    doc = await get_status(cat.agent_key, scope, source)
    if doc is not None:
        completed = doc.get("completed_phases") or []
        if not any(
            isinstance(e, dict) and e.get("phase") == "graph_building"
            for e in completed
        ):
            raise RuntimeError(
                f"[GraphRAG] graph_building generation not recorded for source '{source}'"
            )


@hook(priority=10)
async def after_vector_memory_transfer_on_agent(success, old_handler_name, cat) -> None:
    """Old-store cleanup on vector-memory switch (T24).

    When the transfer SUCCEEDED and the old handler was the GraphRAG one
    (graphrag -> qdrant switch), wipe THIS agent's Neo4j tenant data
    (Document/Entity/relations, strictly tenant-scoped). When the new handler
    is graphrag (qdrant -> graphrag) nothing is done here — the revalidation
    sweep + the ``graph_building`` phase handle graph computation. NEVER wipes
    when ``success`` is False.
    """
    if not success or old_handler_name != "Neo4jGraphRAGConfig":
        return
    from .graphrag_handler import wipe_tenant_graph_data
    await wipe_tenant_graph_data(cat.agent_key)


@hook(priority=10)
async def after_plugin_settings_update(plugin_id: str, settings: Dict[str, Any], cat) -> None:
    # Lazy import (FX-9): same reload-order rationale as
    # after_cheshire_cat_creation — a module-level binding would be the
    # PRE-reload class and this guard would ALWAYS fail.
    from .graphrag_handler import GraphRAGHandler as _GH
    if isinstance(cat.vector_memory_handler, _GH) and cat.vector_memory_handler.entity_extractor:
        await cat.vector_memory_handler.entity_extractor.ensure_downloaded()


@hook(priority=10)
async def after_vector_database_settings_update(
    vector_database_name: str,
    previous_config: Dict[str, Any],
    new_config: Dict[str, Any],
    cat,
) -> None:
    """
    Reacts to ``Neo4jGraphRAGConfig`` saves with two INDEPENDENT refresh paths
    (neither sits behind the other's early return):

    1. Technology terminology: when ``extra_technology_patterns`` changed,
       rebuild the handler's EntityExtractor with the new patterns and re-run
       the pure-regex technology extraction over the stored Documents.
    2. Concept relations: when ``concept_definitions`` / ``relation_definitions``
       / ``concept_relations_prompt`` changed, compute the new config
       fingerprint and launch a single-flight background recompute that
       re-extracts concept relations from EVERY source of this agent tagging
       with the new generation, then atomically flips the tenant's
       ``concept_gen_active`` marker (``GraphRAGHandler._flip_concept_gen``).
       Retrieval (_recall_entity_related) gates on that marker, so old
       concepts disappear exactly when the new ones are in place.

    Idempotence/single-flight comes from the persisted marker + the latest-
    wins pending map + the gen-guard inside the flip — NOT from any in-memory
    handler state (handlers are re-instantiated on every save).
    """
    if vector_database_name != "Neo4jGraphRAGConfig":
        return

    # Lazy import (FX-9): the plugin loader reloads main.py BEFORE
    # graphrag_handler.py (glob order), so a module-level import binds the
    # PRE-reload class and isinstance() below ALWAYS fails (the factory
    # instantiates the handler from the CURRENT, post-reload class). Import at
    # call time, when graphrag_handler is the current module.
    from .graphrag_handler import GraphRAGHandler as _GH
    handler = cat.vector_memory_handler
    if not isinstance(handler, _GH):
        return

    # 1 — Technology terminology refresh (unchanged semantics).
    if new_config.get("extra_technology_patterns") != previous_config.get("extra_technology_patterns"):
        if handler.entity_extractor:
            handler._entity_extractor = EntityExtractor(
                models=handler._spacy_models,
                extra_technology_patterns=new_config.get("extra_technology_patterns") or None,
            )
            await handler.refresh_technology_entities(tenant_id=cat.agent_key)

    # 2 — Concept-relation generation flip (todos 9-13).
    concept_changed = any(
        new_config.get(k) != previous_config.get(k)
        for k in ("concept_definitions", "relation_definitions", "concept_relations_prompt")
    )
    if not concept_changed:
        return

    new_gen = handler._concept_fingerprint()
    active = await handler._read_concept_gen(cat.agent_key)
    if active == new_gen:
        return  # the marker already enforces this generation: no-op

    async with _concept_recompute_lock:
        _concept_recompute_pending[cat.agent_key] = new_gen

    task = asyncio.create_task(
        _concept_recompute_job(handler, cat, cat.agent_key, new_gen)
    )
    # Tracked so the handler's close() can await it during agent shutdown
    # (pattern from after_cheshire_cat_creation) and so it is not GC'd.
    handler._pending_entity_tasks.append(task)
    log.info(
        f"[GraphRAG] Concept-relation config changed for {cat.agent_key}: "
        f"scheduled recompute to generation {new_gen[:12]}"
    )


async def _concept_recompute_job(
    handler: GraphRAGHandler, cat: StrayCat, tenant: str, new_gen: str
) -> None:
    """
    Single-flight concept recompute for one tenant.

    Latest-wins: every pass re-reads ``_concept_recompute_pending`` and the
    persisted marker. The in-process lock only serializes pending-map access;
    single-flight ACROSS handler re-instantiations holds because a losing flip
    (gen-guard inside ``_flip_concept_gen``) aborts and this loop re-derives
    from the marker read and the newest pending generation. A flip that commits
    is followed by the deferred GC of stale old-generation concept nodes.
    """
    attempts = 0
    while attempts < 4:
        attempts += 1
        async with _concept_recompute_lock:
            pending = _concept_recompute_pending.get(tenant)
        if pending is None:
            return  # superseded work: a newer job took over
        active = await handler._read_concept_gen(tenant)
        if active == pending:
            async with _concept_recompute_lock:
                if _concept_recompute_pending.get(tenant) == pending:
                    _concept_recompute_pending.pop(tenant, None)
            return

        try:
            await handler.recompute_concept_relations(cat, gen=pending)
            flipped = await handler._flip_concept_gen(tenant, active, pending)
        except Exception as e:  # noqa: BLE001
            log.error(f"[GraphRAG] Concept recompute failed for {tenant}: {e}")
            return  # keep pending: the next settings save re-triggers it

        if flipped:
            await handler._gc_stale_concept_nodes(tenant)
            async with _concept_recompute_lock:
                if _concept_recompute_pending.get(tenant) == pending:
                    _concept_recompute_pending.pop(tenant, None)
            return

        # Gen-guard aborted: a newer save won the race -> loop re-reads the
        # latest pending generation (latest-wins).
    log.error(
        f"[GraphRAG] Concept recompute for {tenant} gave up after repeated "
        "gen-guard aborts (newer saves keep winning)"
    )
