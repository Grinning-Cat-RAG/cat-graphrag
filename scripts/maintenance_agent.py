#!/usr/bin/env python3
"""
Maintenance agent CLI for the Neo4j GraphRAG plugin.

Standalone, import-safe maintenance script that drives Grinning Cat's
two-phase ingestion engine and the GraphRAG graph rebuild from INSIDE the
Cat container.

Usage (inside the Cat container):
    # Re-embed only (chunk reuse) for one agent
    docker exec cheshire_cat_core python /app/maintenance_agent.py --agent <id> --reembed --yes

    # Re-ingest from scratch (points-first wipe, then re-parse + re-embed) for all agents
    docker exec cheshire_cat_core python /app/maintenance_agent.py --all --reingest --yes

    # Rebuild the GraphRAG graph (wipe-then-rebuild) for one agent
    docker exec cheshire_cat_core python /app/maintenance_agent.py --agent <id> --graph --yes

    # Combine ops and target the episodic collection
    docker exec cheshire_cat_core python /app/maintenance_agent.py --agent <id> --reingest --graph --collection episodic --yes

    # Show the plan without touching anything (exits 2)
    docker exec cheshire_cat_core python /app/maintenance_agent.py --all --graph --dry-run

    # Run the acceptance checks after the ops (read-only)
    docker exec cheshire_cat_core python /app/maintenance_agent.py --agent <id> --reembed --yes --verify

Exit codes:
    0 = all agents/ops ok (and verify ok when --verify)
    1 = partial failure (some agent/op failed, or a --verify check failed)
    2 = aborted (dry-run plan printed, or --yes missing for a destructive step)

Every op requires --yes: --reembed deletes the status docs per source,
--reingest additionally deletes the source's points and re-ingests, and
--graph wipes the tenant's graph edges + orphan entities (A0 Cypher).

IMPORT-SAFETY: this file lives in the plugin folder, which the Cat plugin
loader imports recursively at activation. It must have ZERO top-level side
effects: only stdlib imports at module level; every ``cat`` / ``neo4j``
import happens inside functions.
"""

import argparse
import asyncio
import importlib.util
import json
import sys
import types  # noqa: F401  (used by the --graph LLM part in a later todo)
from typing import Any, NotRequired, TypedDict

# Destructive steps per op — what `--yes` confirms. `--reembed` deletes the
# status docs per source (forces the embedding phase, Metis #9); `--reingest`
# deletes the source's points FIRST (points-first wipe) and re-ingests;
# `--graph` wipes the tenant's graph edges + orphan entities via the A0
# Cypher. Every op therefore requires `--yes`; the missing-`--yes` message
# lists these steps per agent+op.
_DESTRUCTIVE_STEPS: dict[str, list[str]] = {
    "reingest": [
        "delete_tenant_points per source (points-first wipe)",
        "delete_status per source",
        "reembed_sources re-ingest (re-parse + re-embed)",
    ],
    "graph": [
        "A0 wipe Cypher (5 tenant-filtered statements: RELATED_TO, MENTIONS, "
        "PROVENANCE, SIMILAR_TO_<gen>, orphan entities)",
    ],
    "reembed": [
        "delete_status per source (forces the embedding phase)",
    ],
}

# Batch size for the --graph fixed-part walk: after each batch the Epoch
# token is re-read (Metis #16 drift re-check) and the batch re-run on drift.
_GRAPH_WALK_BATCH_SIZE = 50
# Upper bound for per-batch re-runs after a generation drift; beyond it the
# batch is logged as unstable and the walk continues with the new generation.
_GRAPH_WALK_MAX_RERUNS = 3

# The admin's default agent (cat.db.database.DEFAULT_SYSTEM_KEY) is always
# skipped by the enumeration; the legacy `default` agent is kept with a
# one-time warning (module-level flag: warn once per process, not per call).
_default_agent_warned = False

# The Cat lizard's plugin manager is bootstrapped lazily ONCE per process
# (see _ensure_lizard_booted).
_lizard_booted = False


async def _ensure_lizard_booted() -> None:
    """Bootstrap the Cat lizard's plugin manager in this process.

    ``CheshireCat.create`` resolves non-system agents' plugins through
    ``BillTheLizard().plugin_manager.plugins``; in a fresh ``docker exec``
    process that lizard is never bootstrapped (the server does it in the
    uvicorn lifespan), so the registry is empty and ``base_plugin`` is
    missing. Discover the plugins once, lazily, before creating any Cat.
    """
    global _lizard_booted
    if _lizard_booted:
        return
    from cat.looking_glass.bill_the_lizard import BillTheLizard
    await BillTheLizard().plugin_manager.discover_plugins()
    _lizard_booted = True


class _PlanEntry(TypedDict):
    """One per-agent plan entry: target, ops, collection, optional skip."""

    agent_id: str
    ops: list[str]
    collection: str
    # Present when the agent is excluded from the run (e.g. the GraphRAG
    # plugin is not active for it while `--graph` was requested).
    skip_reason: NotRequired[str]


def _parse_args() -> argparse.Namespace:
    """Parse the CLI: scope (--agent|--all), ops, collection, safety flags."""
    parser = argparse.ArgumentParser(
        prog="maintenance_agent",
        description=(
            "Agent maintenance for the Neo4j GraphRAG plugin: re-ingest, "
            "re-embed or rebuild the graph for one agent or all agents. "
            "Must run inside the Cat container."
        ),
    )

    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--agent",
        metavar="ID",
        help="Target a single agent by id.",
    )
    scope.add_argument(
        "--all",
        action="store_true",
        help="Target all agents (the system agent is always skipped).",
    )

    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Re-ingest from scratch: delete points per source, then re-parse "
             "and re-embed. DESTRUCTIVE (requires --yes).",
    )
    parser.add_argument(
        "--reembed",
        action="store_true",
        help="Re-embed only: reuse stored chunks, recompute vectors. Deletes "
             "status docs per source (requires --yes).",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Rebuild the GraphRAG graph (wipe-then-rebuild: fixed NER/"
             "similarity/derived part + optional LLM concept relations). "
             "DESTRUCTIVE (requires --yes).",
    )

    parser.add_argument(
        "--collection",
        choices=["declarative", "episodic"],
        default="declarative",
        help="Memory collection to operate on (default: declarative).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan (per agent+op steps) and exit 2 without any write.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive steps (required for every op: status/point "
             "deletes, the graph wipe, re-ingest).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run acceptance assertions after the ops (read-only; never "
             "against the system agent).",
    )

    args = parser.parse_args()

    # At least one op required; combinable.
    args.ops = [op for op in ("reingest", "reembed", "graph") if getattr(args, op)]
    if not args.ops:
        parser.error("at least one operation is required: --reingest, --reembed, --graph")

    return args


def _runtime_guard() -> bool:
    """Check we are running inside the Cat runtime.

    The script drives the Cat ingestion engine and the GraphRAG handler, so
    it must run inside the container (``docker exec cheshire_cat_core``).
    Returns True when ``cat`` / ``cat.db.database`` are importable; otherwise
    prints the required invocation and returns False.
    """
    try:
        if importlib.util.find_spec("cat") is None:
            raise ImportError("cat not importable")
        if importlib.util.find_spec("cat.db.database") is None:
            raise ImportError("cat.db.database not importable")
    except (ImportError, ValueError, AttributeError):
        print(
            "ERROR: the Cat runtime is not importable — this script must run "
            "inside the Cat container.\n"
            "Run it as:\n"
            "    docker exec cheshire_cat_core python /app/maintenance_agent.py "
            "--agent <id> --reembed --yes"
        )
        return False
    return True


async def _list_agents() -> list[str]:
    """Enumerate agent ids from Redis (read-only).

    ``get_agents_main_keys()`` scans ``agents:*`` and returns the unique
    second key segment. The ``system`` agent (the admin's default,
    ``cat.db.database.DEFAULT_SYSTEM_KEY``) is always skipped; the legacy
    ``default`` agent is kept with a one-time warning.
    """
    global _default_agent_warned

    from cat.db.cruds import settings as crud_settings

    ids = await crud_settings.get_agents_main_keys()

    agents: list[str] = []
    for agent_id in ids:
        if agent_id == "system":
            continue
        if agent_id == "default" and not _default_agent_warned:
            print(
                "[maintenance] warn: agent 'default' is the legacy default "
                "agent — included in the plan"
            )
            _default_agent_warned = True
        agents.append(agent_id)
    return agents


def _resolve_plugin_id() -> str | None:
    """Resolve the deployed GraphRAG plugin folder name at runtime.

    Scans the Cat plugins dir (``cat/plugins``) plus the optional
    ``CAT_PLUGINS_DIR`` override for a folder containing a
    ``graphrag_handler.py`` that defines ``GraphRAGHandler``. Returns the
    folder name (the deployed plugin id, e.g. ``cat_graphrag``) or None when
    the plugin is not deployed. Never hardcoded: the repo folder is
    ``cat-graphrag.my`` but the deployed id is ``cat_graphrag``.
    """
    import os

    from cat.env import get_env
    from cat.utils import get_plugins_path

    candidates = []
    try:
        candidates.append(get_plugins_path())
    except Exception:  # noqa: BLE001 - a broken cat.utils must not kill the scan
        pass
    env_dir = get_env("CAT_PLUGINS_DIR")
    if env_dir:
        candidates.append(env_dir)

    for plugins_dir in candidates:
        if not os.path.isdir(plugins_dir):
            continue
        for folder in sorted(os.listdir(plugins_dir)):
            handler_path = os.path.join(plugins_dir, folder, "graphrag_handler.py")
            if not os.path.isfile(handler_path):
                continue
            try:
                with open(handler_path, encoding="utf-8", errors="ignore") as fh:
                    if "class GraphRAGHandler" in fh.read():
                        return folder
            except OSError:
                continue
    return None


async def _build_ops_plan(args: argparse.Namespace) -> list[_PlanEntry]:
    """Build the per-agent ops plan.

    ``--agent <id>`` validates the id against the live agent list (aborts
    with exit 2 when unknown); ``--all`` expands to every agent minus
    ``system``. When ``--graph`` is requested, the agent set is intersected
    with the agents that have the GraphRAG plugin active; the others get a
    ``skip_reason="plugin-not-active"`` entry.
    """
    agents = await _list_agents()

    if args.agent:
        if args.agent not in agents:
            print(f"ABORTED: unknown agent {args.agent}")
            sys.exit(2)
        agent_ids = [args.agent]
    else:
        agent_ids = agents

    active: set[str] = set()
    if "graph" in args.ops:
        plugin_id = _resolve_plugin_id()
        if plugin_id is None:
            print(
                "ABORTED: GraphRAG plugin not found in the plugins dir "
                "(--graph requires it)"
            )
            sys.exit(2)
        from cat.db.cruds import plugins as crud_plugins

        active = set(await crud_plugins.get_agents_plugin_keys(plugin_id))

    plan: list[_PlanEntry] = []
    for agent_id in agent_ids:
        entry: _PlanEntry = {
            "agent_id": agent_id,
            "ops": list(args.ops),
            "collection": args.collection,
        }
        if "graph" in args.ops and agent_id not in active:
            entry["skip_reason"] = "plugin-not-active"
        plan.append(entry)
    return plan


def _print_plan(plan: list[_PlanEntry]) -> None:
    """Print the dry-run plan: per agent+op the concrete steps.

    Pure plan (Metis #26): NO bootstrap, NO writes, NO ``CheshireCat.create``.
    The A0 wipe statements are printed verbatim (with a ``<gen>`` placeholder
    for the run-time generation); the per-source wipe/status-delete steps are
    printed as templates — the actual source names are only known after
    bootstrap, which dry-run must NOT do.
    """
    print("DRY-RUN PLAN (no writes performed)")
    for entry in plan:
        agent_id = entry["agent_id"]
        if entry.get("skip_reason"):
            print(f"  agent={agent_id} SKIPPED (reason={entry.get('skip_reason')})")
            continue
        collection = entry["collection"]
        for op in entry["ops"]:
            destructive = "yes" if op in _DESTRUCTIVE_STEPS else "no"
            print(
                f"  agent={agent_id} op={op} collection={collection} "
                f"destructive={destructive}"
            )
            if op == "reingest":
                print("    steps:")
                print(
                    "      1. for each source <name>: delete_tenant_points("
                    f"'{collection}', metadata={{'source': <name>}})"
                )
                print(
                    "      2. for each source <name>: delete_status("
                    "agent, 'agent', <name>)"
                )
                print(
                    "      3. reembed_sources(ccat, collection, sources)  "
                    "# re-parse + re-embed (PHASE_PARSING_CHUNKING)"
                )
            elif op == "reembed":
                print("    steps:")
                print(
                    "      1. for each source <name>: delete_status("
                    "agent, 'agent', <name>)"
                )
                print(
                    "      2. reembed_sources(ccat, collection, sources)  "
                    "# embedding phase, chunk reuse (PHASE_EMBEDDING)"
                )
            elif op == "graph":
                print("    steps:")
                print("      1. A0 wipe Cypher (5 tenant-filtered statements):")
                for stmt in _graph_wipe_statements("<gen>"):
                    print(f"         {stmt}")
                print(
                    "      2. re-walk documents (NER + similarity + derived)"
                )
                print("      3. LLM concept relations (if enabled)")
    print("(dry-run: exiting 2, nothing was written)")


async def _bootstrap_agent(agent_id: str) -> tuple[object, object | None, str | None]:
    """Bootstrap one agent for the maintenance ops.

    Creates the agent's ``CheshireCat`` (NOT ``BillTheLizard`` — the lizard
    fires the resume-sweep hooks we want to skip), checks that the vector
    memory handler is the plugin's ``GraphRAGHandler``, connects it, and runs
    ``initialize`` (required for the vector index; on an embedder change it
    launches the seamless shadow-swap ``reembed_tenant``, which flips the
    Epoch token — the flip is logged). Snapshots the current generation on
    the handler as ``_walk_gen`` (consumed by the ``--graph`` A-helper) and
    records the GraphRAG detection (``_graphrag_detected`` /
    ``_concept_relations_enabled``) from the ``Neo4jGraphRAGConfig``
    vector-db setting entry.

    Returns ``(ccat, handler, None)`` on success, or
    ``(ccat, None, "handler-not-graphrag")`` when the agent's vector memory
    handler is not a ``GraphRAGHandler``.
    """
    from cat.looking_glass.cheshire_cat import CheshireCat

    # In a fresh `docker exec` process the lizard is never bootstrapped (the
    # server does it in the uvicorn lifespan), so its plugin registry is empty
    # and CheshireCat.create would KeyError on 'base_plugin'. Bootstrap once.
    await _ensure_lizard_booted()

    ccat = await CheshireCat.create(agent_id)
    handler = ccat.vector_memory_handler

    # The deployed plugin id is resolved at runtime (never hardcoded); the
    # handler class is imported from the plugin module so isinstance checks
    # the deployed code, not this script's copy.
    plugin_id = _resolve_plugin_id()
    if plugin_id is None:
        return (ccat, None, "handler-not-graphrag")
    graphrag_module = importlib.import_module(f"cat.plugins.{plugin_id}.graphrag_handler")
    if not isinstance(handler, graphrag_module.GraphRAGHandler):
        return (ccat, None, "handler-not-graphrag")

    # The versioned decorators probe _get_session() BEFORE the decorated body
    # runs, so the driver must be connected first (AGENTS.md known pitfall).
    await handler._ensure_connected()

    # Generation baseline: initialize() may detect an embedder change and run
    # the shadow-swap reembed_tenant, which flips the Epoch token — the flip
    # is the observable side effect we log (Metis #3).
    gen_before = await handler._read_generation(agent_id)

    embedder = await ccat.embedder()
    await handler.initialize(embedder.name, embedder.size)

    gen = await handler._read_generation(agent_id)
    handler._walk_gen = gen
    if gen != gen_before:
        print(
            f"[maintenance] agent={agent_id} detail=initialize-triggered-reembed "
            f"gen={gen_before}->{gen}"
        )

    # GraphRAG detection: the vector-db setting entry named after the config
    # class (same semantics as main.py:137). The --graph op additionally
    # requires enable_knowledge_graph AND enable_concept_relations (D1: the
    # knowledge-graph flag is the master switch for LLM concept extraction);
    # the per-op skip is decided in _run_agent.
    from cat.db.cruds import settings as crud_settings

    entry = await crud_settings.get_setting_by_name(agent_id, "Neo4jGraphRAGConfig")
    handler._graphrag_detected = entry is not None
    kg = bool((entry or {}).get("value", {}).get("enable_knowledge_graph", False))
    cr = bool((entry or {}).get("value", {}).get("enable_concept_relations", False))
    handler._concept_relations_enabled = bool(kg and cr)

    return (ccat, handler, None)


async def _resolve_ingestion_engine(ccat, op: str = "reembed") -> str | None:
    """Resolve the active ingestion engine; refuse anything but the efficient one.

    Reads the ``ingestion`` settings category (the saved config name) and
    resolves it against the allowed classes via ``cat.services.factory.ingestion``
    (``resolved_config_name``: saved entry, else first plugin class, else the
    base). Only the efficient engine (``EfficientIngestionConfiguration`` /
    ``EfficientIngestionEngine``) is supported by ``--reembed``/``--reingest``:
    the base engine's re-embed path (``embed_all_in_cheshire_cats``) is
    destructive and out of scope, so it is refused with a message naming the
    active engine (Metis #23).

    Returns the active config name when the efficient engine is active, else
    None (the caller marks the op failed → exit 1).
    """
    from cat.services.factory.ingestion import resolved_config_name

    agent_id = getattr(ccat, "agent_key", None) or getattr(ccat, "_id", None)
    name = await resolved_config_name(ccat)
    if name != "EfficientIngestionConfiguration":
        print(
            f"[maintenance] agent={agent_id} op={op} result=fail "
            f"detail=engine-not-efficient active={name}"
        )
        return None
    return name


async def _op_reembed(ccat, handler, collection: str) -> bool:
    """Re-embed only: delete the status docs, then run the embedding phase.

    ``reembed_sources`` decides the phase per source from the status doc: no
    doc + points exist → ``PHASE_EMBEDDING`` (chunk reuse, Metis #9). Deleting
    the status doc per source therefore forces the embedding phase when the
    source's points exist — chunks are reused, nothing is re-parsed. Status
    docs whose source is absent from the enumeration are warned
    (``detail=file-missing``, Metis #10). ``handler`` is accepted for signature
    symmetry with the other ops (todo 5+).
    """
    from cat.core_plugins.efficient_ingestion.reembed import reembed_sources
    from cat.core_plugins.ingestion_status.registry import delete_status, list_statuses
    from cat.services.memory.models import VectorMemoryType

    agent_id = getattr(ccat, "agent_key", None) or getattr(ccat, "_id", None)

    all_sources = await ccat.get_stored_sources_with_metadata()
    sources = all_sources.get(VectorMemoryType(collection), [])

    # Metis #10: status docs whose source is absent from the enumeration.
    statuses = await list_statuses(agent_id)
    known = {s.name for s in sources}
    for doc in statuses:
        src = doc.get("source")
        if src and src not in known:
            print(
                f"[maintenance] agent={agent_id} op=reembed result=warn "
                f"detail=file-missing source={src}"
            )

    # Metis #9: no doc + points exist → PHASE_EMBEDDING (chunk reuse).
    for source in sources:
        await delete_status(agent_id, "agent", source.name)

    await reembed_sources(ccat, VectorMemoryType(collection), sources)
    return True


async def _op_reingest(ccat, handler, collection: str) -> bool:
    """Re-ingest from scratch: delete points per source, then re-parse + re-embed.

    ``reembed_sources`` decides the phase per source from the status doc: no
    doc + no points → ``PHASE_PARSING_CHUNKING`` (clean re-parse, Metis #9).
    Deleting the status doc ALONE would leave the source's points in place and
    silently chunk-reuse (``PHASE_EMBEDDING``) — so the points MUST be deleted
    FIRST, per source, via ``handler.delete_tenant_points(str(collection),
    metadata={"source": name})`` (which also triggers the provenance-cascade
    graph cleanup), and only THEN the status doc. URL sources are passed
    through as-is: the engine re-downloads them (the ``content=None`` path in
    ``_source_from_entry``), so their points are NOT deleted by source name —
    each pass-through is logged (``detail=url-pass-through``).
    """
    from cat.core_plugins.efficient_ingestion.reembed import reembed_sources
    from cat.core_plugins.ingestion_status.registry import delete_status
    from cat.services.memory.models import VectorMemoryType
    from cat.utils import is_url

    agent_id = getattr(ccat, "agent_key", None) or getattr(ccat, "_id", None)

    all_sources = await ccat.get_stored_sources_with_metadata()
    sources = all_sources.get(VectorMemoryType(collection), [])

    # Metis #9 points-first: delete the source's points, THEN the status doc,
    # so reembed_sources sees no doc + no points → PHASE_PARSING_CHUNKING.
    for source in sources:
        if is_url(source.name):
            print(
                f"[maintenance] agent={agent_id} op=reingest result=warn "
                f"detail=url-pass-through source={source.name}"
            )
            continue
        await handler.delete_tenant_points(
            str(collection), metadata={"source": source.name}
        )
        await delete_status(agent_id, "agent", source.name)

    await reembed_sources(ccat, VectorMemoryType(collection), sources)
    return True


def _graph_wipe_statements(gen) -> list[str]:
    """The 5 tenant-filtered A0 wipe statements (single source of truth).

    Shared by ``_graph_wipe`` (execution) and the dry-run plan printer
    (``_print_plan``, which passes the literal ``"<gen>"`` placeholder — the
    real generation is only known after bootstrap, which dry-run must NOT
    do, Metis #26). Every statement is tenant-filtered; no ``MATCH
    (e:Entity)`` without ``{tenant_id}``.
    """
    similar_rel = f"SIMILAR_TO_{gen}"
    return [
        # 1. RELATED_TO edges — undirected match catches both directions.
        "MATCH (:Entity {tenant_id: $tenant_id})-[r:RELATED_TO]-() DELETE r",
        # 2. MENTIONS edges (Document -> Entity).
        "MATCH (:Document {tenant_id: $tenant_id})-[r:MENTIONS]->() DELETE r",
        # 3. PROVENANCE edges (Document -> Entity).
        "MATCH (:Document {tenant_id: $tenant_id})-[r:PROVENANCE]->() DELETE r",
        # 4. SIMILAR_TO edges of the walk generation (Document <-> Document).
        f"MATCH (:Document {{tenant_id: $tenant_id}})-[r:{similar_rel}]->"
        f"(:Document {{tenant_id: $tenant_id}}) DELETE r",
        # 5. Orphan entities: once the edges above are gone, entities with NO
        #    edges at all are orphans — including stale ones (PROVENANCE was
        #    deleted in step 3, so the cascade-prune condition matches,
        #    Metis #7).
        "MATCH (e:Entity {tenant_id: $tenant_id}) WHERE NOT (e)--() DELETE e",
    ]


async def _graph_wipe(handler, tenant_id, gen) -> None:
    """Phase A0: wipe the tenant's fixed-graph edges + orphan entities.

    The fixed part is additive-only (``_extract_and_link_entities`` MERGEs,
    never deletes stale edges), so a plain re-walk would accumulate ghost
    entities and stale relations (Metis #1). No handler helper exists for
    this — the Cypher is defined HERE (``_graph_wipe_statements``),
    tenant-filtered on every statement. The Document/SourceFile/Collection
    structure is kept: only edges and orphan entities are deleted. The
    SIMILAR_TO relation name is the versioned one of the walk generation
    (``SIMILAR_TO_<gen>``).
    """
    async with handler._get_session() as session:
        for stmt in _graph_wipe_statements(gen):
            await session.run(stmt, tenant_id=tenant_id)


async def _graph_fetch_docs(handler, tenant_id, gen) -> list[dict[str, Any]]:
    """Phase A1: fetch the tenant's stored Documents with their embeddings.

    Same fetch pattern as ``recompute_concept_relations``
    (graphrag_handler.py:3157-3164) plus the versioned embedding property
    ``embedding_<gen>`` of the walk generation. Metadata is stored as a JSON
    string in Neo4j and parsed back to a dict here (same handling as the
    handler's own walk).
    """
    embedding_prop = f"embedding_{gen}"
    query = (
        "MATCH (d:Document {tenant_id: $tenant_id}) "
        f"RETURN d.id AS id, d.content AS content, d.metadata AS metadata, "
        f"d.{embedding_prop} AS embedding"
    )
    async with handler._get_session() as session:
        result = await session.run(query, tenant_id=tenant_id)
        rows = [record async for record in result]

    docs: list[dict[str, Any]] = []
    for row in rows:
        raw_meta = row["metadata"]
        try:
            meta = json.loads(raw_meta) if isinstance(raw_meta, str) else (raw_meta or {})
        except (TypeError, ValueError):
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        docs.append(
            {
                "id": row["id"],
                "content": row["content"] or "",
                "metadata": meta,
                "embedding": row["embedding"],
            }
        )
    return docs


async def _graph_walk_batch(handler, batch: list[dict[str, Any]], collection: str) -> None:
    """Phase A2: re-walk one batch of documents INLINE.

    Per doc: ``_extract_and_link_entities`` then
    ``_create_similarity_relationships`` — both awaited synchronously (Metis
    #8: NEVER ``add_point_to_tenant`` — its Document CREATE is not
    idempotent — and NEVER ``create_task`` fire-and-forget). The vector for
    similarity is the doc's ``embedding_<gen>`` property; docs without one
    are skipped (the handler's own guard rejects zero/non-finite vectors,
    but a missing property must not even reach it).
    """
    for doc in batch:
        doc_id = doc["id"]
        await handler._extract_and_link_entities(doc_id, doc["content"], doc["metadata"])
        vector = doc["embedding"]
        if vector:
            await handler._create_similarity_relationships(doc_id, vector, collection)


async def _op_graph_part_a(ccat, handler, collection: str) -> bool:
    """Rebuild the FIXED graph part: wipe-then-re-walk (NER + similarity + derived).

    Phases:
      A0 wipe — tenant-filtered Cypher defined here (no handler helper
         exists, Metis #1): RELATED_TO / MENTIONS / PROVENANCE /
         SIMILAR_TO_<gen> edges + orphan entities;
      A1 fetch — stored Documents with their ``embedding_<gen>`` (same fetch
         as ``recompute_concept_relations``), grouped by ``metadata.source``;
      A2 re-walk — per doc, ``_extract_and_link_entities`` +
         ``_create_similarity_relationships`` INLINE (Metis #8);
      A3 derived — per source, ``create_derived_graph_for_source`` WITHOUT
         ``stray_cat`` (Metis #6: passing it would double-run the LLM part B
         and inflate ``RELATED_TO.weight`` by +0.5 per run); warns when a
         point lacks ``chunk_index`` (derived structure not rebuildable,
         Metis #5);
      A4 gen re-check — per batch, re-read the Epoch token; on drift from
         ``handler._walk_gen``, ``_rebuild_for_generation`` + re-run the
         batch (bounded: max 3 re-runs per batch, then log error, Metis #16).

    Part B (LLM concept relations) is ``_op_graph_part_b`` — this op never
    touches it.
    """
    tenant_id = getattr(ccat, "agent_key", None) or getattr(ccat, "_id", None)

    gen = getattr(handler, "_walk_gen", None)
    if gen is None:
        gen = await handler._read_generation(tenant_id)
        handler._walk_gen = gen

    # ── Phase A0: wipe ────────────────────────────────────────────────────
    await _graph_wipe(handler, tenant_id, gen)

    # ── Phase A1: fetch ───────────────────────────────────────────────────
    docs = await _graph_fetch_docs(handler, tenant_id, gen)

    # ── Phase A2 + A4: walk in batches with per-batch generation re-check ─
    for start in range(0, len(docs), _GRAPH_WALK_BATCH_SIZE):
        batch = docs[start : start + _GRAPH_WALK_BATCH_SIZE]
        await _graph_walk_batch(handler, batch, collection)

        gen_now = await handler._read_generation(tenant_id)
        if gen_now == gen:
            continue

        print(
            f"[maintenance] agent={tenant_id} op=graph result=warn "
            f"detail=generation-drift gen={gen}->{gen_now}"
        )
        handler._rebuild_for_generation(gen_now)
        gen = gen_now
        handler._walk_gen = gen
        docs = await _graph_fetch_docs(handler, tenant_id, gen)
        batch_ids = {d["id"] for d in batch}
        rerun_docs = [d for d in docs if d["id"] in batch_ids]
        for _attempt in range(1, _GRAPH_WALK_MAX_RERUNS + 1):
            await _graph_walk_batch(handler, rerun_docs, collection)
            gen_now = await handler._read_generation(tenant_id)
            if gen_now == gen:
                break
            handler._rebuild_for_generation(gen_now)
            gen = gen_now
            handler._walk_gen = gen
            docs = await _graph_fetch_docs(handler, tenant_id, gen)
            rerun_docs = [d for d in docs if d["id"] in batch_ids]
        else:
            print(
                f"[maintenance] agent={tenant_id} op=graph result=error "
                f"detail=generation-drift-unstable gen={gen}"
            )

    # ── Phase A3: derived structure per source (no stray_cat — Metis #6) ──
    from cat.services.memory.models import PointStruct

    by_source: dict[str, list[dict[str, Any]]] = {}
    for doc in docs:
        source = str(doc["metadata"].get("source") or "unknown")
        by_source.setdefault(source, []).append(doc)

    for source in sorted(by_source):
        src_docs = by_source[source]
        missing_ci = [d for d in src_docs if d["metadata"].get("chunk_index") is None]
        if missing_ci:
            print(
                f"[maintenance] agent={tenant_id} op=graph result=warn "
                f"detail=missing-chunk-index source={source} docs={len(missing_ci)}"
            )
        points = [
            PointStruct(
                id=d["id"],
                payload={
                    "id": d["id"],
                    "page_content": d["content"],
                    "metadata": d["metadata"],
                },
                vector=[],
            )
            for d in src_docs
        ]
        await handler.create_derived_graph_for_source(source, points)

    return True


async def _op_graph_part_b(ccat, handler) -> bool:
    """Rebuild the LLM concept-relations part with a latest-wins flip loop.

    Runs ONLY when GraphRAG is detected AND ``enable_concept_relations``
    (enforced in ``_run_agent``). The LLM re-extraction needs only
    ``ccat.large_language_model`` — a ``SimpleNamespace`` duck suffices (Metis
    #19, verified ``_llm_extract_relations`` graphrag_handler.py:2830 touches
    only that attr).

    Latest-wins loop (Metis #14): the in-process single-flight in main.py does
    NOT coordinate with this separate process — the persisted
    ``concept_gen_active`` marker IS the cross-process guard. Each pass:
    fingerprint -> read the active marker -> recompute with the new gen ->
    ``_flip_concept_gen(expected_prev_gen=active)``; a False flip means a newer
    save won the race, so the loop re-reads the marker and retries (max 3
    attempts). After a successful flip, the deferred GC of stale
    old-generation concept nodes runs. Returns True on a committed flip,
    False when the gen-guard aborted on every attempt.
    """
    tenant_id = getattr(ccat, "agent_key", None) or getattr(ccat, "_id", None)
    stray_duck = types.SimpleNamespace(large_language_model=ccat.large_language_model)

    for attempt in range(1, 4):
        gen = handler._concept_fingerprint()
        active = await handler._read_concept_gen(tenant_id)
        await handler.recompute_concept_relations(stray_duck, gen=gen)
        flipped = await handler._flip_concept_gen(
            tenant_id, expected_prev_gen=active, new_gen=gen
        )
        if flipped:
            await handler._gc_stale_concept_nodes(tenant_id)
            return True
        print(
            f"[maintenance] agent={tenant_id} op=graph result=warn "
            f"detail=flip-conflict retry={attempt}"
        )

    print(
        f"[maintenance] agent={tenant_id} op=graph result=fail "
        f"detail=flip-conflict-exhausted"
    )
    return False


async def _run_agent(agent_id: str, ops: list[str], collection: str = "declarative") -> bool:
    """Run the requested ops for one agent.

    Bootstraps the agent (``CheshireCat.create`` + handler checks +
    ``initialize``) and dispatches each op. ``--reembed`` and ``--reingest``
    resolve the active ingestion engine first (refusing the base engine, Metis
    #23) and then run the embedding phase via ``_op_reembed`` / the points-
    first wipe + clean re-parse via ``_op_reingest``; ``--graph`` runs Part A
    (fixed graph wipe-then-rebuild, ``_op_graph_part_a``) then Part B (LLM
    concept relations with the latest-wins flip, ``_op_graph_part_b``). Skip
    reasons: ``handler-not-graphrag`` (no GraphRAG handler) and
    ``graphrag-not-enabled`` (the ``--graph`` op requires the
    ``Neo4jGraphRAGConfig`` setting with ``enable_knowledge_graph`` AND
    ``enable_concept_relations``).
    Returns True when every op succeeded or was skipped, False otherwise.
    """
    ccat, handler, reason = await _bootstrap_agent(agent_id)
    if reason is not None:
        for op in ops:
            print(f"[maintenance] agent={agent_id} op={op} result=skip reason={reason}")
        return True

    ok = True
    for op in ops:
        if op == "graph" and not (
            getattr(handler, "_graphrag_detected", False)
            and getattr(handler, "_concept_relations_enabled", False)
        ):
            print(
                f"[maintenance] agent={agent_id} op={op} result=skip "
                f"reason=graphrag-not-enabled"
            )
            continue
        if op in ("reembed", "reingest"):
            engine = await _resolve_ingestion_engine(ccat, op=op)
            if engine is None:
                ok = False
                continue
        if op == "reembed":
            try:
                await _op_reembed(ccat, handler, collection)
            except Exception as exc:  # noqa: BLE001 - per-op isolation
                print(f"[maintenance] agent={agent_id} op={op} result=fail detail={exc}")
                ok = False
                continue
            print(f"[maintenance] agent={agent_id} op={op} result=ok detail=reembedded")
            continue
        if op == "reingest":
            try:
                await _op_reingest(ccat, handler, collection)
            except Exception as exc:  # noqa: BLE001 - per-op isolation
                print(f"[maintenance] agent={agent_id} op={op} result=fail detail={exc}")
                ok = False
                continue
            print(f"[maintenance] agent={agent_id} op={op} result=ok detail=reingested")
            continue
        if op == "graph":
            try:
                await _op_graph_part_a(ccat, handler, collection)
                part_b_ok = await _op_graph_part_b(ccat, handler)
            except Exception as exc:  # noqa: BLE001 - per-op isolation
                print(f"[maintenance] agent={agent_id} op={op} result=fail detail={exc}")
                ok = False
                continue
            if not part_b_ok:
                # Part B already logged result=fail detail=flip-conflict-exhausted.
                ok = False
                continue
            print(f"[maintenance] agent={agent_id} op={op} result=ok detail=graph")
            continue
    return ok


async def _verify_agent(ccat, handler, collection: str, gen: str | None = None) -> list[tuple[str, bool]]:
    """Run the acceptance checks (plan Success criteria, Metis #25).

    Read-only assertions against the agent's status docs and the Neo4j
    graph (via ``handler._get_session()``):

      1. status docs completed with the active embedder/chunker;
      2. zero ``Document.embedding_<gen> IS NULL``;
      3. ``MENTIONS`` edge count > 0 and ``SIMILAR_TO_<gen>`` edge count > 0;
      4. ``Collection.concept_gen_active == handler._concept_fingerprint()``
         (same read as ``_read_concept_gen``, graphrag_handler.py:2995);
      5. no orphan entities (no MENTIONS/PROVENANCE) — count == 0;
      6. ``SourceFile`` count == source count.

    Returns a list of ``(check_name, ok)`` tuples; the caller prints the
    ``result=verify-ok|verify-fail`` summary. Never destructive.
    """
    from cat.core_plugins.ingestion_status.registry import list_statuses
    from cat.services.memory.models import VectorMemoryType

    tenant_id = getattr(ccat, "agent_key", None) or getattr(ccat, "_id", None)
    if gen is None:
        gen = getattr(handler, "_walk_gen", None) or await handler._read_generation(
            tenant_id
        )

    checks: list[tuple[str, bool]] = []

    # 1. status docs completed with the active embedder/chunker.
    embedder = await ccat.embedder()
    chunker = ccat.chunker
    statuses = await list_statuses(tenant_id)
    checks.append(
        (
            "status-completed-active-embedder-chunker",
            bool(statuses)
            and all(
                doc.get("status") == "completed"
                and doc.get("embedder_name") == embedder.name
                and doc.get("chunker_name") == chunker.name
                for doc in statuses
            ),
        )
    )

    async def _scalar(query: str) -> Any:
        async with handler._get_session() as session:
            result = await session.run(query, tenant_id=tenant_id)
            record = await result.single()
            return record["n"] if record is not None else None

    embedding_prop = f"embedding_{gen}"
    similar_rel = f"SIMILAR_TO_{gen}"

    # 2. zero Document.embedding_<gen> IS NULL.
    null_emb = await _scalar(
        f"MATCH (d:Document {{tenant_id: $tenant_id}}) "
        f"WHERE d.{embedding_prop} IS NULL RETURN count(d) AS n"
    )
    checks.append(("no-null-embeddings", null_emb == 0))

    # 3. MENTIONS > 0 and SIMILAR_TO_<gen> > 0.
    mentions = await _scalar(
        "MATCH (:Document {tenant_id: $tenant_id})-[r:MENTIONS]->() "
        "RETURN count(r) AS n"
    )
    similar = await _scalar(
        f"MATCH (:Document {{tenant_id: $tenant_id}})-[r:{similar_rel}]-"
        f">(:Document {{tenant_id: $tenant_id}}) RETURN count(r) AS n"
    )
    checks.append(("mentions-edges", mentions > 0))
    checks.append(("similar-to-edges", similar > 0))

    # 4. Collection.concept_gen_active == fingerprint.
    active_gen = await _scalar(
        "MATCH (c:Collection {tenant_id: $tenant_id}) "
        "RETURN c.concept_gen_active AS n LIMIT 1"
    )
    checks.append(("concept-gen-active", active_gen == handler._concept_fingerprint()))

    # 5. no orphan entities (no MENTIONS/PROVENANCE) — count == 0.
    orphans = await _scalar(
        "MATCH (e:Entity {tenant_id: $tenant_id}) "
        "WHERE NOT (e)-[:MENTIONS]-() AND NOT (e)-[:PROVENANCE]-() "
        "RETURN count(e) AS n"
    )
    checks.append(("no-orphan-entities", orphans == 0))

    # 6. SourceFile count == source count.
    all_sources = await ccat.get_stored_sources_with_metadata()
    sources = all_sources.get(VectorMemoryType(collection), [])
    source_files = await _scalar(
        "MATCH (s:SourceFile {tenant_id: $tenant_id}) RETURN count(s) AS n"
    )
    checks.append(("sourcefile-count", source_files == len(sources)))

    return checks


async def main() -> None:
    """CLI entrypoint: parse -> runtime guard -> plan -> dry-run/yes gates -> run."""
    args = _parse_args()

    if not _runtime_guard():
        sys.exit(2)

    plan = await _build_ops_plan(args)

    if args.dry_run:
        _print_plan(plan)
        sys.exit(2)

    # Skipped agents (e.g. plugin-not-active) are reported, not run.
    runnable = []
    for entry in plan:
        if entry.get("skip_reason"):
            print(
                f"[maintenance] agent={entry['agent_id']} result=skip "
                f"reason={entry.get('skip_reason')}"
            )
        else:
            runnable.append(entry)

    if not args.yes:
        destructive = [
            (entry["agent_id"], op)
            for entry in runnable
            for op in entry["ops"]
            if op in _DESTRUCTIVE_STEPS
        ]
        if destructive:
            print("ABORTED: destructive operations require --yes:")
            for agent_id, op in destructive:
                print(f"  agent={agent_id} op={op} steps:")
                for step in _DESTRUCTIVE_STEPS[op]:
                    print(f"    - {step}")
            print("Re-run with --yes to confirm (or --dry-run to preview).")
            sys.exit(2)

    results = []
    for entry in runnable:
        # Sequential per-agent execution (Metis #18): agents are processed one
        # at a time, so no two ops ever run concurrently on the same agent.
        print(f"[maintenance] agent={entry['agent_id']} progress=start")
        try:
            ok = await _run_agent(
                entry["agent_id"], entry["ops"], entry["collection"]
            )
        except Exception as exc:  # noqa: BLE001 - per-agent isolation
            print(f"[maintenance] agent={entry['agent_id']} result=fail detail={exc}")
            ok = False
        print(f"[maintenance] agent={entry['agent_id']} progress=done")
        results.append(ok)

    # --verify: acceptance checks per agent (read-only; the system agent is
    # never in the plan, so it can never be verified either).
    verify_ok = True
    if args.verify:
        for entry in runnable:
            try:
                ccat, handler, reason = await _bootstrap_agent(entry["agent_id"])
                if reason is not None:
                    print(
                        f"[maintenance] agent={entry['agent_id']} "
                        f"result=verify-skip reason={reason}"
                    )
                    continue
                checks = await _verify_agent(ccat, handler, entry["collection"])
            except Exception as exc:  # noqa: BLE001 - per-agent isolation
                print(
                    f"[maintenance] agent={entry['agent_id']} "
                    f"result=verify-fail detail={exc}"
                )
                verify_ok = False
                continue
            failed = [name for name, ok_ in checks if not ok_]
            if failed:
                verify_ok = False
                print(
                    f"[maintenance] agent={entry['agent_id']} result=verify-fail "
                    f"checks={len(checks) - len(failed)}/{len(checks)} "
                    f"failed={','.join(failed)}"
                )
            else:
                print(
                    f"[maintenance] agent={entry['agent_id']} result=verify-ok "
                    f"checks={len(checks)}/{len(checks)}"
                )
        print(f"[maintenance] result=verify-{'ok' if verify_ok else 'fail'}")

    if all(results) and verify_ok:
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())