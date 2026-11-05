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


async def _inject_embedder(cat) -> None:
    """Inject the agent's resolved embedder into the active handler (idempotent).

    ``embedder`` is one of the two SYSTEM-level services, so the instance the
    handler must use is what ``cat.embedder()`` resolves for this agent. Owning
    the same instance again is a no-op and ``_align_embedder_lazy`` is guarded
    to run at most once per handler, so repeated injection is harmless.
    """
    if hasattr(cat.vector_memory_handler, "embedder"):
        cat.vector_memory_handler.embedder = await cat.embedder()
        if hasattr(cat.vector_memory_handler, "_align_embedder_lazy"):
            await cat.vector_memory_handler._align_embedder_lazy()


@hook(priority=10)
async def rabbithole_ingestion_processing(source, cat) -> None:
    """Inject the embedder on the hook BOTH ingestion engines emit.

    The classic RabbitHole fires this right before parsing/embedding; the
    efficient engine (EffING, ``cat/core_plugins/cat_efficient_ingestion``)
    fires it too and NEVER fires ``before_rabbithole_stores_documents``. Without
    this hook the handler would run its extraction phases with no embedder under
    EffING, silently skipping entity embeddings. Idempotent (see
    ``_inject_embedder``), so on the classic flow — which emits both hooks — the
    second injection changes nothing.
    """
    await _inject_embedder(cat)


@hook(priority=10)
async def before_rabbithole_stores_documents(docs: List[Document], cat) -> List[Document]:
    await _inject_embedder(cat)

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
    # The document-structure graph is ALWAYS built.
    # The classic callback is the ONLY graph writer when the phase path is
    # inactive; when it is active the registered phases own the work, so this
    # fire-and-forget must be a NO-OP (otherwise the graph is built twice).
    if await _phase_path_active(cat):
        return
    await handler.create_derived_graph_for_source(source, stored_points, cat)


# ========== INGESTION PHASE MACHINE (EffING registration contract) ==========
#
# The EffING phase machine (deployed as
# ``cat.core_plugins.cat_efficient_ingestion``)
# drives the phase-aware ingestion lifecycle. Plugins REGISTER their phases
# (``ingestion_phase_specs``), provide the material marker for each
# (``ingestion_phase_settings_marker``), and execute them
# (``ingestion_phase_run``). EffING computes staleness for ALL merged phases
# (deps by upstream marker identity + marker comparison) — providers do NOT
# implement their own pending-probe hook. Clock-free: no timestamps
# for correctness, no full settings fingerprints, no raw Redis.


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


_EFFICIENT_INGESTION_CONFIG_NAME = "EfficientIngestionConfiguration"


async def _resolved_system_ingestion_config_name(cat) -> "str | None":
    """Effective SYSTEM-level ingestion engine name.

    The ``ingestion`` category is one of the two system-level services (the
    other is ``embedder``), so its selection is read against
    ``DEFAULT_SYSTEM_KEY`` — never the agent key. ``resolved_config_name`` is
    the canonical resolver (signature: ``resolved_config_name(lizard) -> str``);
    a host that predates it is handled by reading the same single system-level
    category entry directly, so both paths return the same value.
    """
    try:
        from cat.services.factory.ingestion import resolved_config_name
    except ImportError:
        resolved_config_name = None

    if resolved_config_name is not None:
        return await resolved_config_name(cat)

    from cat.db.cruds import settings as crud_settings
    from cat.db.database import DEFAULT_SYSTEM_KEY

    saved = await crud_settings.get_settings_by_category(
        DEFAULT_SYSTEM_KEY, "ingestion"
    )
    return (saved or {}).get("name")


async def _phase_path_active(cat) -> bool:
    """Whether the EffING phase path owns graph building for this ingest.

    Single source of truth for the two ingestion modes. The phase path is
    active **iff**:

    ==========================  ======================  =============
    EffING installed/loadable   SYSTEM ``ingestion``    phase path
    ==========================  ======================  =============
    no                          any                     inactive
    yes                         core engine             inactive
    yes                         efficient engine        active
    ==========================  ======================  =============

    In every other case the classic path runs (``after_rabbithole_stored_documents``
    -> ``create_derived_graph_for_source``). The phase hooks are still REGISTERED
    whenever EffING is importable; this helper only decides which mechanism
    writes the graph, so the two can never run together and neither is dead.
    """
    try:
        import cat.core_plugins.cat_efficient_ingestion  # noqa: F401
    except Exception:  # noqa: BLE001
        return False

    try:
        name = await _resolved_system_ingestion_config_name(cat)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            f"[GraphRAG] Could not resolve the system ingestion engine "
            f"({exc!r}); using the classic path."
        )
        return False

    return name == _EFFICIENT_INGESTION_CONFIG_NAME


@hook(priority=10)
def ingestion_phase_specs(specs, cat) -> list:
    """Register the two graph phases with the EffING phase machine (T5).

    ACCUMULATOR convention: EffING threads ``specs`` through every registrant
    and merges the returned list into its PHASES. ``settings_category=None``:
    the material marker IS the settings fingerprint (``_spacy_fingerprint`` /
    ``_concept_fingerprint``), so the settings fast-path is skipped. Both
    phases depend on ``parsing_chunking`` + ``embedding`` (the graph work
    needs the stored chunks AND valid vectors). Lazy import (FX-9 style) in a
    try/except keeps this module import-safe when EffING is not deployed.
    """
    try:
        from cat.core_plugins.cat_efficient_ingestion.phases import PhaseSpec
    except ImportError:
        PhaseSpec = None
    if PhaseSpec is None:
        return specs
    specs.append(
        PhaseSpec(
            "graph_spacy_extraction",
            settings_category=None,
            depends_on=("parsing_chunking", "embedding"),
        )
    )
    specs.append(
        PhaseSpec(
            "graph_llm_extraction",
            settings_category=None,
            depends_on=("parsing_chunking", "embedding"),
        )
    )
    return specs


@hook(priority=10)
async def ingestion_phase_settings_marker(phase, cat) -> Any:
    """Material marker for the graph phases (T5).

    The marker IS the settings fingerprint: ``_spacy_fingerprint`` for
    ``graph_spacy_extraction`` (spaCy models + technology terminology +
    ``enable_entity_extraction``) and ``await _concept_fingerprint`` for
    ``graph_llm_extraction`` (concept/relation definitions + prompt +
    agent-level LLM identity + ``enable_concept_relations``). EffING compares
    markers across runs to decide staleness — no timestamps, no full settings
    dumps.
    """
    handler = _graphrag_handler(cat)
    if handler is None:
        return None
    if phase == "graph_spacy_extraction":
        return handler._spacy_fingerprint()
    if phase == "graph_llm_extraction":
        return await handler._concept_fingerprint()
    return None


@hook(priority=10)
async def ingestion_phase_run(phase, source, completed_phases, cat):
    """Phase executor (T5): dispatch the two graph phases inline and awaited.

    ``graph_spacy_extraction`` -> ``run_spacy_extraction_phase`` (NER + link +
    similarity, awaited); ``graph_llm_extraction`` ->
    ``run_llm_extraction_phase`` (LLM concept relations). Returns the
    documented tri-state status dict:

    - ``{"status": "done"}`` when there is nothing to do (the active handler is
      not ``GraphRAGHandler``) — a registered-but-inactive graph phase must NOT
      turn a successful ingestion into an ERROR;
    - whatever the phase executor returns (``done`` / ``not_ready``) when the
      handler is active;
    - ``None`` for an UNKNOWN phase (fail-hard: the core maps ``None`` ->
      status=error, so an unhandled phase never silently succeeds).

    Exceptions propagate (the core maps them to status=error).
    """
    handler = _graphrag_handler(cat)
    if handler is None:
        return {"status": "done"}
    if phase == "graph_spacy_extraction":
        return await handler.run_spacy_extraction_phase(source, cat=cat)
    if phase == "graph_llm_extraction":
        return await handler.run_llm_extraction_phase(source, cat=cat)
    return None


@hook(priority=10)
async def before_ingestion_status_completed(source, cat) -> None:
    """Completion gate (T5): fail-hard when graph work is still pending.

    Raises when live background tasks remain for the source, so the core
    writes status=error. The diary check is gone: EffING owns the completed-
    phases diary and computes staleness for the registered phases itself.
    No-op when graphrag is not the active handler.
    """
    handler = _graphrag_handler(cat)
    if handler is None:
        return

    if handler.has_pending_source_tasks(source):
        raise RuntimeError(
            f"[GraphRAG] Pending background graph tasks for source '{source}'"
        )


@hook(priority=10)
async def after_vector_memory_transfer_on_agent(success: bool, cat) -> None:
    """Old-store cleanup on vector-memory switch (T24).

    Host contract: ``after_vector_memory_transfer_on_agent(success: bool, cat)``
    (``cat.core_plugins.base_plugin.hooks.flow``; invoked by
    ``CheshireCat.transfer_vector_points_from``). The host does NOT pass the
    old handler name, and by the time this hook runs ``cat.vector_memory_handler``
    already points at the NEW handler, so the old name is not reachable from
    the hook args or from the (already updated) settings.

    The switch direction is therefore inferred from the CURRENT handler: the
    only switch that leaves a stale GraphRAG store behind is
    GraphRAG -> non-GraphRAG, and in that case the current handler is not a
    ``GraphRAGHandler``. When the new handler IS GraphRAG
    (non-graphrag -> graphrag) the revalidation sweep + the ``graph_building``
    phase own graph computation, so nothing is wiped here. NEVER wipes when
    ``success`` is False.

    A non-GraphRAG -> non-GraphRAG switch can also reach the wipe branch, but
    ``wipe_tenant_graph_data`` is a strict no-op without a recorded Neo4j
    config and tenant-scoped otherwise, so it stays safe.
    """
    if not success:
        return
    if _graphrag_handler(cat) is not None:
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

    new_gen = await handler._concept_fingerprint()
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
