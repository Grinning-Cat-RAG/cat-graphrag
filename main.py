from typing import List, Dict, Any
import asyncio
from langchain_core.documents import Document

from cat import hook, RecallSettings, VectorDatabaseSettings
from cat.log import log
from cat.looking_glass.stray_cat import StrayCat
from cat.services.memory.models import PointStruct

from .graphrag_handler import Neo4jGraphRAGConfig, GraphRAGHandler
from .entity_extractor import EntityExtractor


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
    handler = getattr(cat, "vector_memory_handler", None)
    if not isinstance(handler, GraphRAGHandler):
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

    if isinstance(cat.vector_memory_handler, GraphRAGHandler):
        handler = cat.vector_memory_handler
        if handler.entity_extractor:
            await handler.entity_extractor.ensure_initialized()
        for i, doc in enumerate(docs):
            doc.metadata.setdefault("chunk_index", i)

    return docs


@hook
async def after_rabbithole_stored_documents(source: str, stored_points: List[PointStruct], cat) -> None:
    handler = cat.vector_memory_handler
    if not isinstance(handler, GraphRAGHandler):
        return
    settings = await cat.mad_hatter.get_plugin().load_settings()
    if not settings.get("enable_derived_graph", True):
        return
    await handler.create_derived_graph_for_source(source, stored_points, cat)


@hook(priority=10)
async def after_plugin_settings_update(plugin_id: str, settings: Dict[str, Any], cat) -> None:
    if isinstance(cat.vector_memory_handler, GraphRAGHandler) and cat.vector_memory_handler.entity_extractor:
        await cat.vector_memory_handler.entity_extractor.ensure_downloaded()


@hook(priority=10)
async def after_vector_database_settings_update(
    vector_database_name: str,
    previous_config: Dict[str, Any],
    new_config: Dict[str, Any],
    cat,
) -> None:
    """
    Refreshes the Technology-entity subgraph for the given agent when its
    ``extra_technology_patterns`` change.

    Fired by the core's ``upsert_vector_database_setting`` with the
    vector-DB config name and the previous/new config payloads. Only reacts to
    ``Neo4jGraphRAGConfig`` and only when the technology patterns actually
    changed. Rebuilds the handler's EntityExtractor with the new patterns and
    re-runs the pure-regex technology extraction over the existing stored
    Documents of this agent — no re-ingest, no re-embed, no full wipe, and no
    other agent's graph is touched.
    """
    if vector_database_name != "Neo4jGraphRAGConfig":
        return
    if new_config.get("extra_technology_patterns") == previous_config.get("extra_technology_patterns"):
        return

    handler = cat.vector_memory_handler
    if not isinstance(handler, GraphRAGHandler):
        return
    if not handler.entity_extractor:
        return

    # Rebuild the extractor with the new patterns (mirrors __init__:87-90).
    handler._entity_extractor = EntityExtractor(
        models=handler._spacy_models,
        extra_technology_patterns=new_config.get("extra_technology_patterns") or None,
    )

    await handler.refresh_technology_entities(tenant_id=cat.agent_key)
