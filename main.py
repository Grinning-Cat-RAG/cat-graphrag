from typing import List, Dict, Any
import asyncio
import math
import time
from langchain_core.documents import Document

from cat import hook, RecallSettings, VectorDatabaseSettings
from cat.log import log
from cat.looking_glass.stray_cat import StrayCat

from . import background_jobs
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
# tenant -> the fingerprint whose visible concepts this process checked: a change of the agent's LLM fires no hook, so
# the plugin checks before a recall or an ingestion, once per change (not per message).
_concept_checked: Dict[str, str] = {}
# A recompute holds a Redis lock per tenant and generation, so that two pods noticing the same change do not both pay
# the LLM pass; it expires on its own if the pod dies (a recompute longer than this may start twice: the flip guard
# keeps it correct).
_CONCEPT_RECOMPUTE_LOCK_SECONDS = 2 * 60 * 60
# tenant -> when this process checks again a change being recomputed by another pod (its lock was taken): the holder
# may die, its lock then expires and the next check here starts the work.
_concept_recheck_at: Dict[str, float] = {}
_CONCEPT_RECHECK_SECONDS = 10 * 60


@hook(priority=10)
def factory_allowed_vector_databases(allowed: List[VectorDatabaseSettings], cat) -> List:
    allowed.append(Neo4jGraphRAGConfig)
    return allowed


@hook(priority=10)
async def after_cheshire_cat_creation(cat) -> None:
    """
    Provenance migration of an agent being CREATED.

    The core fires this hook only when an agent is created
    (``BillTheLizard.create_cheshire_cat``), not when an existing agent is
    loaded: an agent's graph that predates the PROVENANCE / ``source_files``
    tracking is reconciled here in the background. ``recompute_provenance``
    self-skips tenants that are already reconciled (a single
    ``provenance_reconciled`` marker count).
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

    await _check_concept_generation(cat)
    return config


async def _check_concept_generation(cat) -> None:
    """``_ensure_concept_generation``, never failing the recall or the ingestion that runs it."""
    try:
        await _ensure_concept_generation(cat)
    except Exception as e:  # noqa: BLE001 - a check, never a reason to fail the caller
        log.warning(f"[GraphRAG] Concept generation check failed for {getattr(cat, 'agent_key', '?')}: {e}")


async def _ensure_concept_generation(cat) -> None:
    """The visible concepts are those of the current concept settings AND of the agent's LLM.

    A save of the plugin settings recomputes them (``after_vector_database_settings_update``); a change of the
    agent's LLM fires no hook. So, once per tenant and fingerprint in this process: if concepts of another generation
    are still visible, the same recompute starts. Not "marker != fingerprint": an agent never recomputed has no marker
    and its concepts are those of the current fingerprint (no LLM pass on the day of the release).
    """
    # Lazy import (FX-9): same reload-order rationale as after_cheshire_cat_creation.
    from .graphrag_handler import GraphRAGHandler as _GH
    handler = cat.vector_memory_handler
    if not isinstance(handler, _GH) or not handler._enable_concept_relations:
        return
    tenant = cat.agent_key
    fingerprint = await handler._concept_fingerprint()
    if _concept_checked.get(tenant) == fingerprint and time.monotonic() < _concept_recheck_at.get(tenant, math.inf):
        return  # checked (or recomputing elsewhere: checked again after an interval)
    _concept_checked[tenant] = fingerprint
    _concept_recheck_at.pop(tenant, None)
    try:
        stale = await handler.has_stale_visible_concepts(fingerprint)
    except Exception:
        _concept_checked.pop(tenant, None)  # the next recall checks again
        raise
    if not stale:
        return
    async with _concept_recompute_lock:
        if _concept_recompute_pending.get(tenant) == fingerprint:
            return  # already scheduled in this process
        _concept_recompute_pending[tenant] = fingerprint
    _schedule_concept_recompute(tenant, fingerprint, "concepts of another generation are visible")


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
async def before_rabbithole_stores_documents(docs: List[Document], cat) -> List[Document]:
    await _inject_embedder(cat)
    await _check_concept_generation(cat)

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


@hook(priority=10)
def rabbithole_ingestion_phases(phases: List, cat) -> List:
    """Add the two graph phases to the ingestion of the sources, where the vector memory is GraphRAG.

    ``graph_spacy_extraction`` runs after ``embedding``, ``graph_llm_extraction`` after it (see ``phases.py``).
    Classes, not instances: the RabbitHole instantiates them for every run. Lazy import (FX-9): the current
    ``phases`` module is the one reloaded with the current ``graphrag_handler``.
    """
    from .phases import GraphConceptPhase, GraphSpacyPhase
    if _graphrag_handler(cat) is not None:
        phases.extend([GraphSpacyPhase, GraphConceptPhase])
    return phases


def _graphrag_handler(cat):
    """Resolve the active GraphRAG handler, or None.

    Lazy import (FX-9, the plugin loader reloads main.py BEFORE graphrag_handler.py, so a module-level binding would
    compare against the PRE-reload class and always fail) + isinstance against the current class.
    """
    from .graphrag_handler import GraphRAGHandler as _GH
    handler = getattr(cat, "vector_memory_handler", None)
    if isinstance(handler, _GH):
        return handler
    return None


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
    (non-graphrag -> graphrag) nothing is wiped: the graph of every source is
    rebuilt instead (``_rebuild_graph_after_transfer``). Nothing happens when
    ``success`` is False.

    A non-GraphRAG -> non-GraphRAG switch can also reach the wipe branch, but
    ``wipe_tenant_graph_data`` is a strict no-op without a recorded Neo4j
    config and tenant-scoped otherwise, so it stays safe.
    """
    if not success:
        return
    if _graphrag_handler(cat) is not None:
        await _rebuild_graph_after_transfer(cat)
        return
    from .graphrag_handler import wipe_tenant_graph_data
    await wipe_tenant_graph_data(cat.agent_key)


async def _rebuild_graph_after_transfer(cat) -> None:
    """Bring the graph of every source of the agent up to date, once its points are in GraphRAG.

    The transfer writes the points outside any ingestion: each write already made the graph phases of its sources
    due (``GraphRAGHandler._graph_routes``) and queued their rerun. This pass, under the lock of each source, is the
    safety net once the whole transfer is done (e.g. the resume ran a graph phase on a store still being filled): a
    source already rebuilt has nothing to do. No invalidation, so no source is rebuilt twice. A failing source does
    not stop the others.
    """
    from cat.db.cruds import ingestions as crud_ingestions

    for row in await crud_ingestions.list_statuses(cat.agent_key):
        try:
            await cat.rabbit_hole.rerun_source(cat, row["source"], row["chat_id"])
        except Exception as e:  # noqa: BLE001 - the other sources still need their graph
            log.error(
                f"[GraphRAG] Graph rebuild of '{row['source']}' after the transfer to GraphRAG failed: {e}"
            )


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

    _schedule_concept_recompute(cat.agent_key, new_gen, "concept-relation config changed")


def _schedule_concept_recompute(tenant: str, new_gen: str, why: str) -> None:
    """Start the recompute of the tenant's concepts to ``new_gen`` in background. The task belongs to no request (a
    request's agent shuts its handler down, cancelling the handler's tasks, as soon as the request ends): it is held
    here until it ends, and it loads an agent of its own."""
    background_jobs.start(_concept_recompute_job(tenant, new_gen))
    log.info(f"[GraphRAG] {why} for {tenant}: scheduled recompute to generation {new_gen[:12]}")


async def _concept_recompute_job(tenant: str, new_gen: str) -> None:
    """The recompute of a tenant's concepts to ``new_gen`` (``_concept_recompute_passes``), on an agent of its own,
    holding the Redis lock of that tenant and generation.

    - The lock is taken by another pod: the work is theirs; this pod checks again after an interval (if the holder
      died, its lock expires and a later check here starts the work).
    - Any other end without the generation in place (a failure, a cancellation, the agent gone): nothing pending and
      nothing checked here, so the next recall checks again.
    """
    from redis.exceptions import LockError

    from cat.db.crud import distributed_lock

    from .graphrag_handler import GraphRAGHandler as _GH

    held = done = elsewhere = False
    try:
        async with distributed_lock(
            f"graphrag-concept-recompute:{tenant}:{new_gen}",
            timeout=_CONCEPT_RECOMPUTE_LOCK_SECONDS,
            blocking_timeout=0.1,
        ):
            held = True
            ccat = await background_jobs.load_agent(tenant)
            handler = getattr(ccat, "vector_memory_handler", None)
            if not isinstance(handler, _GH):
                log.info(f"[GraphRAG] Concept recompute of {tenant}: the agent is gone or not on GraphRAG anymore")
                return
            done = await _concept_recompute_passes(handler, ccat, tenant, new_gen)
    except LockError as e:
        if held:  # a lock of the work itself (e.g. the tenant write lock): a failure of the recompute
            log.error(f"[GraphRAG] Concept recompute failed for {tenant}: {e}")
        else:
            elsewhere = True
            log.info(f"[GraphRAG] Concept recompute of {tenant} to {new_gen[:12]} is running elsewhere")
    except Exception as e:  # noqa: BLE001 - a background job: logged, then retried by a later check
        log.error(f"[GraphRAG] Concept recompute failed for {tenant}: {e}")
    finally:
        if elsewhere:
            await _drop_pending(tenant, new_gen)
            _concept_recheck_at[tenant] = time.monotonic() + _CONCEPT_RECHECK_SECONDS
        elif not done:
            await _forget_concept_work(tenant, new_gen)


async def _concept_recompute_passes(
    handler: GraphRAGHandler, cat, tenant: str, new_gen: str
) -> bool:
    """
    Single-flight concept recompute for one tenant. Returns whether the tenant ends on the pending generation.

    Latest-wins: every pass re-reads ``_concept_recompute_pending`` and the
    persisted marker. The in-process lock only serializes pending-map access;
    single-flight ACROSS handler re-instantiations holds because a losing flip
    (gen-guard inside ``_flip_concept_gen``) aborts and this loop re-derives
    from the marker read and the newest pending generation. A flip that commits
    is followed by the deferred GC of stale old-generation concept nodes.
    Exceptions propagate (the job forgets the work: a later check retries).
    """
    attempts = 0
    while attempts < 4:
        attempts += 1
        async with _concept_recompute_lock:
            pending = _concept_recompute_pending.get(tenant)
        if pending is None:
            return False  # superseded work: a newer job took over
        active = await handler._read_concept_gen(tenant)
        if active == pending:
            await _drop_pending(tenant, pending)
            return True

        await handler.recompute_concept_relations(cat, gen=pending)
        flipped = await handler._flip_concept_gen(tenant, active, pending)
        if flipped:
            await handler._gc_stale_concept_nodes(tenant)
            await _drop_pending(tenant, pending)
            return True

        # Gen-guard aborted: a newer save won the race -> loop re-reads the
        # latest pending generation (latest-wins).
    log.error(
        f"[GraphRAG] Concept recompute for {tenant} gave up after repeated "
        "gen-guard aborts (newer saves keep winning)"
    )
    return False


async def _drop_pending(tenant: str, gen: str) -> None:
    """Nothing pending for the tenant anymore, if what is pending is still ``gen`` (a newer generation stays)."""
    async with _concept_recompute_lock:
        if _concept_recompute_pending.get(tenant) == gen:
            _concept_recompute_pending.pop(tenant, None)


async def _forget_concept_work(tenant: str, gen: str) -> None:
    """A recompute to ``gen`` that will not complete here: nothing pending for it, nothing checked, so the next recall
    of this process checks the concepts again and starts it if they are still stale."""
    await _drop_pending(tenant, gen)
    if _concept_checked.get(tenant) == gen:
        _concept_checked.pop(tenant, None)
    _concept_recheck_at.pop(tenant, None)
