#!/usr/bin/env python3
"""
Maintenance agent CLI for the Neo4j GraphRAG plugin.

Standalone, import-safe maintenance script that drives Grinning Cat's
two-phase ingestion engine and the GraphRAG graph rebuild from INSIDE the
Cat container.

Usage (inside the Cat container):
    # Re-embed only (chunk reuse) for one agent
    docker exec cheshire_cat_core python /app/cat/plugins/cat_graphrag/scripts/maintenance_agent.py --agent <id> --reembed --yes

    # Re-ingest from scratch (points-first wipe, then re-parse + re-embed) for all agents
    docker exec cheshire_cat_core python /app/cat/plugins/cat_graphrag/scripts/maintenance_agent.py --all --reingest --yes

    # Rebuild the GraphRAG graph (wipe-then-rebuild) for one agent
    docker exec cheshire_cat_core python /app/cat/plugins/cat_graphrag/scripts/maintenance_agent.py --agent <id> --graph --yes

    # Combine ops and target the episodic collection
    docker exec cheshire_cat_core python /app/cat/plugins/cat_graphrag/scripts/maintenance_agent.py --agent <id> --reingest --graph --collection episodic --yes

    # Show the plan without touching anything (exits 2)
    docker exec cheshire_cat_core python /app/cat/plugins/cat_graphrag/scripts/maintenance_agent.py --all --graph --dry-run

    # Resume ONLY the failed phases (no wipe, no re-parse, no re-embed)
    docker exec cheshire_cat_core python /app/cat/plugins/cat_graphrag/scripts/maintenance_agent.py --agent <id> --resume-failed --yes

    # Run the acceptance checks after the ops (read-only)
    docker exec cheshire_cat_core python /app/cat/plugins/cat_graphrag/scripts/maintenance_agent.py --agent <id> --reembed --yes --verify

Exit codes:
    0 = all agents/ops ok (and verify ok when --verify)
    1 = partial failure (some agent/op failed, or a --verify check failed)
    2 = aborted (dry-run plan printed, --yes missing for a destructive step,
        or an op needs the efficient ingestion plugin EffING and it is not
        importable as ``cat.plugins.cat_efficient_ingestion``)

``--reembed``, ``--reingest``, ``--resume-failed`` and ``--verify`` require the
extracted EffING plugin (``reembed``/``registry`` modules). Its imports are
lazy and guarded: when it is absent the CLI still starts (``--help`` works)
and those ops abort with a clear message and exit code 2 instead of raising a
traceback.

Every op requires --yes: --reembed deletes the status docs per source,
--reingest additionally deletes the source's points and re-ingests,
--resume-failed rewrites the failed diary/status and re-dispatches only the
failing phases (never purging, re-parsing or re-embedding a completed phase),
and --graph wipes the tenant's graph edges + orphan entities (A0 Cypher).

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
from typing import Any, NoReturn, NotRequired, TypedDict

_EFFING_MODULE = "cat.plugins.cat_efficient_ingestion"


# Per-agent maintenance marker shared with the EffING resume sweep. The sweep
# skips an agent while this marker is present, so the CLI and the periodic
# app sweep never process the same agent's sources concurrently. Written
# through ``cat.db.crud`` (never a raw Redis client). A generous TTL makes a
# crashed CLI fail safe: the marker expires on its own and the sweep resumes.
_MAINTENANCE_MARKER_TTL_SECONDS = 2 * 60 * 60


def _maintenance_marker_key(agent_id: str) -> str:
    """Redis key of the CLI maintenance marker for one agent."""
    return f"agents:{agent_id}:ingestion:maintenance"


async def _set_maintenance_marker(agent_id: str) -> None:
    """Write/refresh the per-agent maintenance marker with a 2h TTL.

    Best-effort: a marker write failure is reported but never aborts the
    maintenance op (the TTL then simply is not renewed).
    """
    from cat.db import crud
    from cat.db.models import generate_timestamp

    try:
        await crud.store(
            _maintenance_marker_key(agent_id),
            {"owner": "maintenance", "started_at": generate_timestamp()},
            expire=_MAINTENANCE_MARKER_TTL_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 - coordination is best-effort
        print(
            f"[maintenance] agent={agent_id} result=warn "
            f"detail=maintenance-marker-write-failed error={exc}"
        )


async def _clear_maintenance_marker(agent_id: str) -> None:
    """Delete the per-agent maintenance marker (also on failure).

    A failed delete is only logged: the marker's TTL guarantees it cannot
    block the resume sweep forever.
    """
    from cat.db import crud

    try:
        await crud.delete(_maintenance_marker_key(agent_id))
    except Exception as exc:  # noqa: BLE001 - TTL is the fail-safe
        print(
            f"[maintenance] agent={agent_id} result=warn "
            f"detail=maintenance-marker-delete-failed error={exc}"
        )


def _abort_missing_effing(exc: ImportError) -> NoReturn:
    """Report the absent EffING dependency and abort (exit code 2, no traceback).

    EffING was extracted to ``cat.plugins.cat_efficient_ingestion``; the
    ops that drive its phase machine cannot run without it. Callers wrap their
    lazy ``cat.plugins.cat_efficient_ingestion.*`` imports in
    ``try/except ImportError`` and delegate here, so the module itself stays
    import-safe and ``--help`` keeps working when the plugin is absent.
    """
    print(
        "[maintenance] ABORTED: required optional dependency "
        f"'{_EFFING_MODULE}' is not importable ({exc}). "
        "--reembed, --reingest, --resume-failed and --verify need the efficient ingestion "
        "plugin (EffING); install/enable it and retry.",
        file=sys.stderr,
    )
    sys.exit(2)


# Destructive steps per op — what `--yes` confirms. `--reembed` deletes the
# status docs per source (forces the embedding phase, Metis #9); `--reingest`
# deletes the source's points FIRST (points-first wipe) and re-ingests;
# `--graph` wipes the tenant's graph edges + orphan entities via the A0
# Cypher. Every op therefore requires `--yes`; the missing-`--yes` message
# lists these steps per agent+op.
_DESTRUCTIVE_STEPS: dict[str, list[str]] = {
    "reingest": [
        "delete_tenant_points per source (points-first wipe)",
        "clear_completed_phases per source (drop the lying diary)",
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
    "resume-failed": [
        "reset the ERROR diary entries per failed source (keeps complete ones)",
        "set status=processing, error=None per failed source",
        "reembed_sources re-dispatch of ONLY the failing phases "
        "(no purge, no re-parse, no re-embed of healthy phases)",
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
    # Resolved error-status sources for `--resume-failed` (dry-run plan).
    resume_sources: NotRequired[list[dict[str, Any]]]


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
        action="append",
        help="Target one or more agents by id (repeatable). Explicitly naming "
             "a protected/system agent allows it.",
    )
    scope.add_argument(
        "--all",
        action="store_true",
        help="Target all agents (system and __*__ protected agents skipped).",
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
        "--resume-failed",
        action="store_true",
        help="Resume ONLY the phases whose diary entry is in the error state "
             "(keeps complete phases with their tag/started_at, writes "
             "status=processing, error=None, and re-dispatches only those "
             "phases). No purge, no re-parse, no re-embed. Requires --yes.",
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
             "deletes, the graph wipe, re-ingest, failed-phase resume).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run acceptance assertions after the ops (read-only; never "
             "against the system agent).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the resolved plan and the final summary as JSON.",
    )

    args = parser.parse_args()

    # Combinable ops. --resume-failed is declarative-only: it operates on the
    # agent's stored sources, which live in the declarative collection.
    ops = ("reingest", "reembed", "resume-failed", "graph")
    ops_attr = {
        "reingest": args.reingest,
        "reembed": args.reembed,
        "resume-failed": args.resume_failed,
        "graph": args.graph,
    }
    args.ops = [op for op in ops if ops_attr[op]]
    if not args.ops:
        parser.error(
            "at least one operation is required: --reingest, --reembed, "
            "--resume-failed, --graph"
        )
    if "resume-failed" in args.ops and args.collection != "declarative":
        parser.error("--resume-failed only supports --collection declarative")

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
            "    docker exec cheshire_cat_core python /app/cat/plugins/cat_graphrag/scripts/maintenance_agent.py "
            "--agent <id> --reembed --yes"
        )
        return False
    return True


async def _list_agents(include_protected: bool = False) -> list[str]:
    """Enumerate agent ids from Redis (read-only).

    ``get_agents_main_keys()`` scans ``agents:*`` and returns the unique
    second key segment. With ``include_protected=False`` (``--all``), the
    ``system`` agent (the admin's default, ``cat.db.database.DEFAULT_SYSTEM_KEY``)
    and the ``__*__`` system agents are always skipped. An explicit
    ``--agent`` validates against the protected-inclusive list, so naming one
    explicitly is the only way to target it. The legacy ``default`` agent is
    kept with a one-time warning.
    """
    global _default_agent_warned

    from cat.db.cruds import settings as crud_settings

    ids = await crud_settings.get_agents_main_keys()

    agents: list[str] = []
    for agent_id in ids:
        if not include_protected:
            if agent_id == "system":
                continue
            if agent_id.startswith("__") and agent_id.endswith("__"):
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
    ``cat-graphrag`` but the deployed id is ``cat_graphrag``.
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


async def _agent_uses_graphrag_handler(agent_id: str) -> bool:
    """Read-only: is the agent's saved vector DB the GraphRAG one?

    Mirrors ``_bootstrap_agent``'s ``isinstance`` check WITHOUT bootstrapping
    the agent (the dry-run plan must not create a ``CheshireCat``). The saved
    ``vector_database`` setting names the active handler class; GraphRAG is
    ``Neo4jGraphRAGConfig``. Returns False on any read error (fail-safe: the
    graph op is then reported as ``handler-not-graphrag``).
    """
    from cat.db.cruds import settings as crud_settings

    try:
        entry = await crud_settings.get_settings_by_category(
            agent_id, "vector_database"
        )
    except Exception:  # noqa: BLE001 - a read error must not crash the plan
        return False
    return (entry or {}).get("name") == "Neo4jGraphRAGConfig"


async def _resolve_resume_failed_plan(agent_id: str) -> list[dict[str, Any]]:
    """Read-only resolution of the sources `--resume-failed` will act on.

    Enumerates the agent-scope status docs (``list_statuses``), keeps only the
    rows whose status is ``error`` and re-reads each one with
    ``registry.get_status`` (the same API the op uses). For every row it
    reports the diary phases whose explicit state is ``error`` (including the
    collateral ``dependency <x> is error`` entries) and the count of phases it
    will keep untouched (``complete`` / ``skipped`` / ``processing``).

    Never bootstraps the agent and never writes: safe for the dry-run plan.
    """
    try:
        from cat.plugins.cat_efficient_ingestion.registry import (
            PHASE_STATE_ERROR,
            get_status,
            list_statuses,
            phase_state,
        )
    except ImportError as exc:
        _abort_missing_effing(exc)

    resolved: list[dict[str, Any]] = []
    for listed in await list_statuses(agent_id):
        source = listed.get("source")
        if not source or listed.get("status") != "error":
            continue
        scope = str(listed.get("scope") or "agent")
        doc = await get_status(agent_id, scope, source)
        if not isinstance(doc, dict) or doc.get("status") != "error":
            continue
        diary = doc.get("completed_phases")
        diary = diary if isinstance(diary, dict) else {}
        failed = [
            phase
            for phase, entry in diary.items()
            if phase_state(entry) == PHASE_STATE_ERROR
        ]
        kept = [
            phase
            for phase, entry in diary.items()
            if phase_state(entry) != PHASE_STATE_ERROR
        ]
        resolved.append(
            {
                "source": source,
                "scope": scope,
                "type": doc.get("type") or "file",
                "failed_phases": failed,
                "kept_phases": kept,
            }
        )
    return resolved


async def _build_ops_plan(args: argparse.Namespace) -> list[_PlanEntry]:
    """Build the per-agent ops plan.

    ``--agent <id>`` (repeatable) validates every id against the live agent
    list (aborts with exit 2 when unknown) and is the ONLY way to target a
    protected/system agent; ``--all`` expands to every agent minus ``system``
    and ``__*__``. When ``--graph`` is the ONLY requested op, an agent is
    skipped with ``skip_reason="plugin-not-active"`` (plugin off) or
    ``skip_reason="handler-not-graphrag"`` (plugin on, vector backend not
    GraphRAG). With a mixed plan the engine ops still run, so the graph op is
    skipped per-op in ``_run_agent`` instead.

    ``--resume-failed`` also resolves the error-status sources for every
    targeted agent (read-only: status docs only, no bootstrap) so the dry-run
    can print agent, source, failed phases and the kept-phase count.
    """
    raw_agents = args.agent or []
    if isinstance(raw_agents, str):
        raw_agents = [raw_agents]
    explicit = list(dict.fromkeys(raw_agents))
    if explicit:
        all_agents = await _list_agents(include_protected=True)
        unknown = [agent_id for agent_id in explicit if agent_id not in all_agents]
        if unknown:
            print(f"ABORTED: unknown agent {', '.join(unknown)}")
            sys.exit(2)
        agent_ids = explicit
    else:
        agent_ids = await _list_agents()

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

    # A whole-agent plan-level skip is only correct when EVERY requested op is
    # graph-level. With a mixed plan (e.g. --reingest --graph) the engine op
    # must still run, so the graph op is skipped per-op in _run_agent instead.
    graph_ops_only = args.ops == ["graph"]

    plan: list[_PlanEntry] = []
    for agent_id in agent_ids:
        entry: _PlanEntry = {
            "agent_id": agent_id,
            "ops": list(args.ops),
            "collection": args.collection,
        }
        if graph_ops_only and agent_id not in active:
            entry["skip_reason"] = "plugin-not-active"
        elif graph_ops_only and not await _agent_uses_graphrag_handler(agent_id):
            # GraphRAG plugin loaded but the agent's vector backend is not the
            # GraphRAG handler (e.g. Qdrant): the graph op cannot run.
            entry["skip_reason"] = "handler-not-graphrag"
        if "resume-failed" in args.ops:
            entry["resume_sources"] = await _resolve_resume_failed_plan(agent_id)
        plan.append(entry)
    return plan


def _print_plan(plan: list[_PlanEntry], as_json: bool = False) -> None:
    """Print the dry-run plan: per agent+op the concrete steps.

    Pure plan (Metis #26): NO bootstrap, NO writes, NO ``CheshireCat.create``.
    The A0 wipe statements are printed verbatim (with a ``<gen>`` placeholder
    for the run-time generation); the per-source wipe/status-delete steps are
    printed as templates — the actual source names are only known after
    bootstrap, which dry-run must NOT do. ``--resume-failed`` is the exception:
    its sources are resolved from the status docs (read-only) and printed
    explicitly with the failed and kept phases.

    With ``as_json`` the whole plan is emitted as one JSON document instead of
    the human-readable lines.
    """
    if as_json:
        print(json.dumps({"dry_run": True, "plan": plan}, ensure_ascii=False, default=str))
        return

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
            elif op == "resume-failed":
                print("    steps:")
                print(
                    "      1. delete_status is NOT called; clear only the "
                    "ERROR entries in the status diary"
                )
                print(
                    "      2. reembed_sources(ccat, declarative, [source], "
                    "stale_after=0.0)  # failing phases only"
                )
                for item in entry.get("resume_sources", []):
                    failed = item.get("failed_phases") or []
                    kept = item.get("kept_phases") or []
                    if not failed:
                        print(
                            f"    source={item['source']} nothing-to-resume "
                            f"(kept_phases={len(kept)})"
                        )
                    else:
                        print(
                            f"    source={item['source']} "
                            f"failed_phases={','.join(failed)} "
                            f"kept_phases={len(kept)}"
                        )
            elif op == "graph":
                print("    steps:")
                print(
                    "      1. A0 wipe Cypher (5 tenant-filtered statements):"
                )
                for stmt in _graph_wipe_statements("<gen>"):
                    print(f"         {stmt}")
                print(
                    "      2. reset the graph phases' diary entries for every "
                    "source (data deleted first, state last)"
                )
                print(
                    "      3. re-walk documents (NER + similarity + derived)"
                )
                print("      4. LLM concept relations (if enabled)")
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
    ``(ccat, None, "handler-not-graphrag")`` when the agent's vector
    memory handler is not a ``GraphRAGHandler``. That reason only gates the
    GRAPH-level ops (``--graph``); the ENGINE-level ops (``--reembed`` /
    ``--reingest``) run anyway and resolve the vector handler from ``ccat``
    themselves when ``handler`` is None.
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
    # requires enable_concept_relations (the only concept-write gate;
    # enable_knowledge_graph is display-only); the per-op skip is decided in
    # _run_agent.
    from cat.db.cruds import settings as crud_settings

    entry = await crud_settings.get_setting_by_name(agent_id, "Neo4jGraphRAGConfig")
    handler._graphrag_detected = entry is not None
    cr = bool((entry or {}).get("value", {}).get("enable_concept_relations", False))
    handler._concept_relations_enabled = cr

    return (ccat, handler, None)


async def _resolve_ingestion_engine(ccat, op: str = "reembed") -> str | None:
    """Resolve the active ingestion engine; refuse anything but the efficient one.

    Mirrors ``main.py``: the current core does not expose
    ``resolved_config_name``; when a host provides it the saved entry is
    resolved against the allowed classes via
    ``cat.services.factory.ingestion``, otherwise the same SYSTEM-level
    ``ingestion`` category entry (``DEFAULT_SYSTEM_KEY``) is read directly, so
    both paths return the same value. Only the efficient engine
    (``EfficientIngestionConfiguration`` / ``EfficientIngestionEngine``) is
    supported by ``--reembed``/``--reingest``: the base engine's re-embed path
    (``embed_all_in_cheshire_cats``) is destructive and out of scope, so it is
    refused with a message naming the active engine (Metis #23).

    Returns the active config name when the efficient engine is active, else
    None (the caller marks the op failed → exit 1).
    """
    from cat.log import log

    agent_id = getattr(ccat, "agent_key", None) or getattr(ccat, "_id", None)

    try:
        from cat.services.factory.ingestion import resolved_config_name
    except ImportError:
        resolved_config_name = None

    if resolved_config_name is not None:
        log.debug("[maintenance] ingestion engine resolved via resolved_config_name")
        name = await resolved_config_name(ccat)
    else:
        from cat.db.cruds import settings as crud_settings
        from cat.db.database import DEFAULT_SYSTEM_KEY

        log.debug("[maintenance] ingestion engine resolved via system settings fallback")
        saved = await crud_settings.get_settings_by_category(
            DEFAULT_SYSTEM_KEY, "ingestion"
        )
        name = (saved or {}).get("name")

    if name != "EfficientIngestionConfiguration":
        print(
            f"[maintenance] agent={agent_id} op={op} result=fail "
            f"detail=engine-not-efficient active={name}"
        )
        return None
    return name


async def _read_ingestion_status(agent_id: str, source_name: str):
    """Read one source's ingestion-status doc (lazy EffING import)."""
    try:
        from cat.plugins.cat_efficient_ingestion.registry import get_status
    except ImportError as exc:
        _abort_missing_effing(exc)
    return await get_status(agent_id, "agent", source_name)


async def _report_ingestion_outcome(
    agent_id: str, op: str, sources: list
) -> bool:
    """Report the REAL per-source phase outcome after handing sources to EffING.

    ``reembed_sources`` runs its serial dispatch loop synchronously but never
    raises on a failed phase: a phase error only lives in the source's status
    doc (``status=error`` plus the ``error`` entry in ``completed_phases``).
    Printing ``result=ok`` right after the loop therefore claims success for
    work the engine did not finish. This reads every processed source's status
    doc back and owns the final summary line:

      - all ``completed`` -> ``result=ok detail=reingested|reembedded``;
      - any ``error`` -> ``result=fail detail=phase-failed`` with the verbatim
        reason of each failing phase;
      - anything not terminal (``processing`` / missing doc) -> ``result=fail
        detail=still-processing``.

    Returns True only when every processed source is ``completed``.
    """
    try:
        from cat.plugins.cat_efficient_ingestion.registry import (
            PHASE_STATE_ERROR,
            phase_state,
        )
    except ImportError as exc:
        _abort_missing_effing(exc)

    failures: dict[str, dict[str, Any]] = {}
    still_processing: list[str] = []
    for source in sources:
        name = source.name
        doc = await _read_ingestion_status(agent_id, name)
        if not isinstance(doc, dict):
            still_processing.append(name)
            continue
        diary = doc.get("completed_phases")
        phase_errors: dict[str, Any] = {}
        if isinstance(diary, dict):
            for phase_id, entry in diary.items():
                if phase_state(entry) == PHASE_STATE_ERROR:
                    phase_errors[phase_id] = entry.get("reason")
        status = doc.get("status")
        if status == "error" or phase_errors:
            if not phase_errors:
                phase_errors[doc.get("phase") or "unknown"] = doc.get("error")
            failures[name] = phase_errors
        elif status != "completed":
            still_processing.append(name)

    if failures:
        print(
            f"[maintenance] agent={agent_id} op={op} result=fail "
            f"detail=phase-failed failures="
            f"{json.dumps(failures, ensure_ascii=False, default=str)}"
        )
        return False
    if still_processing:
        print(
            f"[maintenance] agent={agent_id} op={op} result=fail "
            f"detail=still-processing sources={','.join(still_processing)}"
        )
        return False

    detail = "reembedded" if op == "reembed" else "reingested"
    print(f"[maintenance] agent={agent_id} op={op} result=ok detail={detail}")
    return True


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
    try:
        from cat.plugins.cat_efficient_ingestion.reembed import reembed_sources
        from cat.plugins.cat_efficient_ingestion.registry import (
            delete_status,
            list_statuses,
        )
    except ImportError as exc:
        _abort_missing_effing(exc)
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
        # Refresh the maintenance marker per source: a long run must stay
        # protected even if it outlives the initial TTL.
        await _set_maintenance_marker(agent_id)
        await delete_status(agent_id, "agent", source.name)

    await reembed_sources(ccat, VectorMemoryType(collection), sources)
    return await _report_ingestion_outcome(agent_id, "reembed", sources)


async def _op_resume_failed(
    ccat, handler, collection: str, json_output: bool = False
) -> bool:
    """Resume ONLY the diary phases that are in the ``error`` state.

    For every agent-scope source whose status is ``error``:

    1. read the status doc with ``registry.get_status``;
    2. remove ONLY the diary entries whose ``phase_state`` is ``error``
       through ``registry.reset_phase_entries`` — the collateral
       ``dependency <x> is error`` entries are error entries too, while every
       ``complete`` entry keeps its ``tag`` / ``started_at`` untouched;
    3. write ``status=processing`` and ``error=None`` (``set_status``);
    4. hand only that source to ``reembed_sources(..., stale_after=0.0)``: the
       probe sees the remaining complete phases as fresh, so only the cleared
       phases re-run. There is NO purge, NO point wipe, NO re-parse and NO
       re-embed of a completed phase;
    5. hold the per-agent maintenance marker for the whole pass (refreshed per
       source) so the EffING sweep cannot steal the rows.

    A source with no failed phase is reported ``nothing-to-resume`` and left
    untouched. ``graph_llm_extraction`` is never forced: when concept relations
    are disabled its body records ``skipped`` (never ``error``). Per-source
    failures are isolated: one bad source never aborts the others, and the
    final ``result=ok|fail`` is the same shape as ``--reingest``.

    Returns True only when every attempted source ended ``completed`` (or had
    nothing to resume).
    """
    try:
        from cat.plugins.cat_efficient_ingestion.reembed import reembed_sources
        from cat.plugins.cat_efficient_ingestion.registry import (
            PHASE_STATE_ERROR,
            IngestionStatus,
            get_status,
            phase_state,
            reset_phase_entries,
            set_status,
        )
        from cat.plugins.cat_efficient_ingestion.resume import _source_from_entry
    except ImportError as exc:
        _abort_missing_effing(exc)

    from cat.services.memory.models import VectorMemoryType

    agent_id = getattr(ccat, "agent_key", None) or getattr(ccat, "_id", None)
    plan = await _resolve_resume_failed_plan(agent_id)

    retried: dict[str, list[str]] = {}
    skipped: list[str] = []
    failures: dict[str, Any] = {}

    for item in plan:
        source_name = item["source"]
        scope = item["scope"]
        type_ = item["type"]

        # A long pass must stay protected even if it outlives the initial TTL.
        await _set_maintenance_marker(agent_id)

        if not item["failed_phases"]:
            skipped.append(source_name)
            if not json_output:
                print(
                    f"[maintenance] agent={agent_id} op=resume-failed "
                    f"source={source_name} result=skip detail=nothing-to-resume"
                )
            continue

        try:
            # Build the source object BEFORE any write: a missing file must
            # never leave the row ``processing`` with its error diary dropped.
            source_obj = _source_from_entry(
                ccat, agent_id, scope, source_name, type_
            )
            if source_obj is None:
                failures[source_name] = {"source": "source-unavailable"}
                if not json_output:
                    print(
                        f"[maintenance] agent={agent_id} op=resume-failed "
                        f"source={source_name} result=fail detail=source-unavailable"
                    )
                continue

            # Re-read just before writing: the row may have changed between
            # the plan resolution and this pass.
            doc = await get_status(agent_id, scope, source_name)
            diary = (doc or {}).get("completed_phases")
            diary = diary if isinstance(diary, dict) else {}
            failed_phases = [
                phase
                for phase, entry in diary.items()
                if phase_state(entry) == PHASE_STATE_ERROR
            ]
            if not failed_phases:
                skipped.append(source_name)
                if not json_output:
                    print(
                        f"[maintenance] agent={agent_id} op=resume-failed "
                        f"source={source_name} result=skip detail=nothing-to-resume"
                    )
                continue

            # STATE ONLY: drop the failed diary entries (complete entries and
            # their tag/started_at are preserved), then mark the source
            # processing with no error. No points are touched.
            await reset_phase_entries(agent_id, scope, source_name, failed_phases)
            await set_status(
                agent_id,
                scope,
                source_name,
                type_=type_,
                status=IngestionStatus.PROCESSING,
                chat_id=None if scope == "agent" else scope,
                error=None,
            )

            # Re-dispatch: the probe sees the kept phases as fresh, so only
            # the cleared (failed) phases run. stale_after=0.0 lets the claim
            # take the row we just wrote.
            await reembed_sources(
                ccat,
                VectorMemoryType(collection),
                [source_obj],
                stale_after=0.0,
            )

            after = await get_status(agent_id, scope, source_name) or {}
            after_diary = after.get("completed_phases")
            after_diary = after_diary if isinstance(after_diary, dict) else {}
            still_failed = {
                phase: entry.get("reason")
                for phase, entry in after_diary.items()
                if phase_state(entry) == PHASE_STATE_ERROR
            }
            status = after.get("status")
            if status == "error" or still_failed:
                failures[source_name] = {
                    "phases": still_failed
                    or {after.get("phase") or "unknown": after.get("error")}
                }
                if not json_output:
                    print(
                        f"[maintenance] agent={agent_id} op=resume-failed "
                        f"source={source_name} result=fail detail=phase-failed "
                        f"phases={json.dumps(still_failed, ensure_ascii=False, default=str)}"
                    )
                continue
            if status != "completed":
                failures[source_name] = {"phases": {"still-processing": status}}
                if not json_output:
                    print(
                        f"[maintenance] agent={agent_id} op=resume-failed "
                        f"source={source_name} result=fail "
                        f"detail=still-processing status={status!r}"
                    )
                continue

            retried[source_name] = failed_phases
            if not json_output:
                print(
                    f"[maintenance] agent={agent_id} op=resume-failed "
                    f"source={source_name} result=ok detail=resumed "
                    f"phases={','.join(failed_phases)}"
                )
        except Exception as exc:  # noqa: BLE001 - per-source isolation
            failures[source_name] = {"error": f"{type(exc).__name__}: {exc}"}
            if not json_output:
                print(
                    f"[maintenance] agent={agent_id} op=resume-failed "
                    f"source={source_name} result=fail "
                    f"detail={type(exc).__name__}: {exc}"
                )

    if json_output:
        print(
            json.dumps(
                {
                    "agent": agent_id,
                    "op": "resume-failed",
                    "result": "fail" if failures else "ok",
                    "retried": retried,
                    "skipped": skipped,
                    "failures": failures,
                },
                ensure_ascii=False,
                default=str,
            )
        )
    elif failures:
        print(
            f"[maintenance] agent={agent_id} op=resume-failed result=fail "
            f"detail=phase-failed failures="
            f"{json.dumps(failures, ensure_ascii=False, default=str)} "
            f"retried={json.dumps(retried, ensure_ascii=False, default=str)}"
        )
    else:
        print(
            f"[maintenance] agent={agent_id} op=resume-failed result=ok "
            f"detail=resumed retried="
            f"{json.dumps(retried, ensure_ascii=False, default=str)} "
            f"skipped={json.dumps(skipped, ensure_ascii=False, default=str)}"
        )

    return not failures


async def _preflight_reembed_path(ccat) -> str | None:
    """Return why the re-embed path cannot rebuild, or None when it can.

    Runs BEFORE any destructive wipe: ``_op_reingest`` deletes the source
    points first, so a parse/embed path that cannot run must abort before
    touching storage (incident: a ``None`` RabbitHole wiped the points and
    then failed at ``rabbit_hole.cat = ccat``, leaving nothing to rebuild).
    """
    try:
        from cat.plugins.cat_efficient_ingestion.reembed import (
            _resolve_rabbit_hole,
        )
    except ImportError as exc:
        _abort_missing_effing(exc)

    try:
        rabbit_hole = await _resolve_rabbit_hole(ccat)
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return f"rabbit-hole-unavailable: {exc}"
    if rabbit_hole is None or getattr(rabbit_hole, "cat", None) is None:
        return "rabbit-hole-unbound"
    if getattr(ccat, "chunker", None) is None:
        return "chunker-unavailable"

    try:
        handlers = await ccat.file_handlers()
        embedder = await ccat.embedder()
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return f"parsing-engine-unavailable: {exc}"
    if not handlers:
        return "no-file-handlers"
    if embedder is None:
        return "embedder-unavailable"
    return None


def _union_with_stored_files(ccat, agent_id: str | None, sources: list) -> list:
    """Union the vector-store sources with the agent's stored files.

    ``get_stored_sources_with_metadata`` reads the VECTOR STORE, so a source
    whose points were already wiped by a failed re-ingest is invisible and can
    never be recovered. The bytes still live in the file manager, so every
    stored file missing from the vector enumeration is appended, built through
    EffING's existing ``_source_from_entry`` loader (the same ``BytesIO`` shape
    the vector-store enumeration produces). Dedup is by ``name``.

    Derived image artifacts (``<stem>_img_<index>_<digest>.<ext>``, produced by
    cat-multimodal-ingestion) are NOT sources: they are artifacts of their
    parent file. They are skipped explicitly and logged; re-ingesting them
    would mint duplicate image sources next to the parent PDF.

    Read-only and fail-open: a missing/unusable file manager or an
    unreadable file only logs, leaving the vector-store sources untouched.
    """
    try:
        from cat.plugins.cat_efficient_ingestion.resume import _source_from_entry
    except ImportError as exc:
        _abort_missing_effing(exc)

    if not agent_id:
        return sources
    file_manager = getattr(ccat, "file_manager", None)
    if file_manager is None:
        return sources

    try:
        stored_files = file_manager.list_files(agent_id)
    except Exception as exc:  # noqa: BLE001 - enumeration must never abort reingest
        print(
            f"[maintenance] agent={agent_id} op=reingest result=warn "
            f"detail=file-manager-list-failed error={exc!r}"
        )
        return sources

    merged = list(sources)
    seen = {source.name for source in merged}
    for entry in stored_files:
        name = getattr(entry, "name", None)
        if not name or name in seen:
            continue
        if "_img_" in name:
            print(
                f"[maintenance] agent={agent_id} op=reingest result=warn "
                f"detail=derived-image-skip source={name}"
            )
            continue
        source = _source_from_entry(ccat, agent_id, "agent", name, "file")
        if source is None:
            print(
                f"[maintenance] agent={agent_id} op=reingest result=warn "
                f"detail=stored-file-unreadable source={name}"
            )
            continue
        merged.append(source)
        seen.add(name)
    return merged


async def _op_reingest(ccat, handler, collection: str) -> bool:
    """Re-ingest from scratch: delete points per source, then re-parse + re-embed.

    ``reembed_sources`` decides the phase per source from the status doc: no
    doc + no points → ``PHASE_PARSING_CHUNKING`` (clean re-parse, Metis #9).
    Deleting the status doc ALONE would leave the source's points in place and
    silently chunk-reuse (``PHASE_EMBEDDING``) — so the points MUST be deleted
    FIRST, per source, via the active vector handler's
    ``delete_tenant_points(str(collection), metadata={"source": name})`` (which
    also triggers the provenance-cascade graph cleanup), THEN clear the
    completed-phases diary (``clear_completed_phases``), and only THEN the
    status doc. The diary is bookkeeping, not proof the points exist: clearing
    it before ``delete_status`` guarantees the phases re-run even if the status
    row survives a failed delete. ``handler`` may be None (non-GraphRAG agent):
    the live vector handler is resolved from ``ccat`` in that case. URL sources
    are passed through as-is: the engine re-downloads them (the
    ``content=None`` path in ``_source_from_entry``), so their points are NOT
    deleted by source name — each pass-through is logged
    (``detail=url-pass-through``).

    Enumeration is the UNION of the vector-store sources and the agent's
    stored files (``ccat.file_manager.list_files(agent_id)``): a source whose
    points were already deleted by a failed re-ingest is invisible to the
    vector store, so a points-only enumeration could never recover it. Stored
    files without points are added through the existing EffING
    ``_source_from_entry`` loader (no second content loader); derived image
    artifacts (``*_img_*.jpg``-style names) are skipped and logged, since they
    belong to their parent PDF.
    """
    try:
        from cat.plugins.cat_efficient_ingestion.reembed import reembed_sources
        from cat.plugins.cat_efficient_ingestion.registry import (
            purge_source_for_reingest,
        )
    except ImportError as exc:
        _abort_missing_effing(exc)
    from cat.services.memory.models import VectorMemoryType
    from cat.utils import is_url

    agent_id = getattr(ccat, "agent_key", None) or getattr(ccat, "_id", None)

    # The points-first wipe only needs the active VECTOR handler's
    # delete_tenant_points, which every vector backend exposes. ``handler`` is
    # the GraphRAG handler only for GraphRAG agents; for any other backend the
    # caller passes None, so resolve the live vector handler here. The graph
    # phases themselves are engine-level and no-op when GraphRAG is inactive.
    vector_handler = handler if handler is not None else ccat.vector_memory_handler

    all_sources = await ccat.get_stored_sources_with_metadata()
    sources = all_sources.get(VectorMemoryType(collection), [])
    sources = _union_with_stored_files(ccat, agent_id, sources)

    # PREFLIGHT: never wipe when the re-embed path cannot rebuild. This runs
    # before the points-first delete loop below, so a broken path aborts the
    # agent with everything still in place.
    preflight_reason = await _preflight_reembed_path(ccat)
    if preflight_reason is not None:
        print(
            f"[maintenance] agent={agent_id} op=reingest result=fail "
            f"detail=preflight-reembed-path-unusable reason={preflight_reason}"
        )
        return False

    # Metis #9 points-first: delete the source's points, THEN the status doc,
    # so reembed_sources sees no doc + no points → PHASE_PARSING_CHUNKING.
    for source in sources:
        # Refresh the maintenance marker per source (long runs outlive the TTL).
        await _set_maintenance_marker(agent_id)
        if is_url(source.name):
            print(
                f"[maintenance] agent={agent_id} op=reingest result=warn "
                f"detail=url-pass-through source={source.name}"
            )
            continue
        # DATA FIRST, STATE LAST — through the shared EffING helper so the two
        # sides can never be reordered: the points are deleted, THEN the diary
        # is cleared (every phase, not a hardcoded pair), THEN the status row.
        # If the vector delete raises, neither the diary nor the row is
        # touched, and the source re-runs from its current checkpoint.
        await purge_source_for_reingest(
            agent_id,
            "agent",
            source.name,
            lambda: vector_handler.delete_tenant_points(
                str(collection), metadata={"source": source.name}
            ),
        )

    await reembed_sources(ccat, VectorMemoryType(collection), sources)
    return await _report_ingestion_outcome(agent_id, "reingest", sources)


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


async def _graph_phase_ids(ccat) -> list[str]:
    """The graph phases THIS plugin registers with EffING (never hardcoded).

    Reads the plugin's own ``graph_phase_specs()`` (the declarations behind
    the ``ingestion_phase_specs`` hook), so adding/renaming a graph phase
    updates the ``--graph`` invalidation automatically. Unlike the hook, it is
    not gated on EffING being active for the agent: entries written while it
    was active must be dropped even if it is off now. Returns ``[]`` when the
    plugin module (or EffING) is unavailable.
    """
    plugin_id = _resolve_plugin_id()
    if plugin_id is None:
        return []
    try:
        plugin_main = importlib.import_module(f"cat.plugins.{plugin_id}.main")
        specs = plugin_main.graph_phase_specs()
    except Exception:  # noqa: BLE001 - invalidation must not abort the wipe
        return []
    return [spec.id for spec in specs if getattr(spec, "id", None)]


async def _reset_graph_phase_entries(ccat, tenant_id: str) -> int:
    """Drop the graph phases' diary entries after the graph data was wiped.

    **DATA FIRST, STATE LAST**: called only after ``_graph_wipe`` succeeded.
    The tenant-wide wipe removes every source's graph edges/entities, so EVERY
    status doc of this agent (all scopes) must stop recording the graph phases
    as ``complete`` — otherwise the next ingestion trusts a diary whose data is
    gone and skips the rebuild. EffING enumerates the docs; this helper only
    supplies the phase ids it owns.
    """
    try:
        from cat.plugins.cat_efficient_ingestion.registry import (
            reset_phase_entries_for_agent,
        )
    except ImportError as exc:
        _abort_missing_effing(exc)

    phase_ids = await _graph_phase_ids(ccat)
    if not phase_ids:
        return 0
    changed = await reset_phase_entries_for_agent(tenant_id, phase_ids)
    print(
        f"[maintenance] agent={tenant_id} op=graph detail=phase-state-reset "
        f"phases={phase_ids} docs={changed}"
    )
    return changed


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
      A3 derived — per source, ``create_derived_graph_for_source`` with
         ``run_concepts=False``: the LLM concept step is part B's, running it
         here too would call the LLM twice per source; warns when a
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
    # The graph data is gone: immediately drop the graph phases' diary entries
    # (state LAST). A later crash in the rebuild then leaves the phases
    # pending, so the next ingestion re-runs them instead of trusting a
    # ``complete`` entry over deleted graph artifacts.
    await _reset_graph_phase_entries(ccat, tenant_id)

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
        # Structure only: the LLM concept step is part B's.
        await handler.create_derived_graph_for_source(
            source, points, run_concepts=False
        )

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
        gen = await handler._concept_fingerprint()
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


async def _run_agent(
    agent_id: str,
    ops: list[str],
    collection: str = "declarative",
    json_output: bool = False,
) -> bool:
    """Run the requested ops for one agent.

    Bootstraps the agent (``CheshireCat.create`` + handler checks +
    ``initialize``) and dispatches each op. ``--reembed``, ``--reingest`` and
    ``--resume-failed`` resolve the active ingestion engine first (refusing
    the base engine, Metis #23) and then run the embedding phase via
    ``_op_reembed`` / the points-first wipe + clean re-parse via
    ``_op_reingest`` / the failed-phases-only resume via ``_op_resume_failed``;
    ``--graph`` runs Part A (fixed graph wipe-then-rebuild,
    ``_op_graph_part_a``) then Part B (LLM concept relations with the
    latest-wins flip, ``_op_graph_part_b``).

    The skip reason from ``_bootstrap_agent`` only gates the GRAPH ops:
    ``handler-not-graphrag`` (no GraphRAG handler) skips ``--graph`` but NOT
    ``--reembed`` / ``--reingest`` / ``--resume-failed``, whose phases are
    engine-level and record graph phases as ``skipped`` when inactive.
    Returns True when every op succeeded or was skipped, False otherwise.
    """
    ccat, handler, reason = await _bootstrap_agent(agent_id)

    ok = True
    for op in ops:
        if op == "graph" and reason is not None:
            print(
                f"[maintenance] agent={agent_id} op={op} result=skip reason={reason}"
            )
            continue
        if op == "graph" and not (
            getattr(handler, "_graphrag_detected", False)
            and getattr(handler, "_concept_relations_enabled", False)
        ):
            print(
                f"[maintenance] agent={agent_id} op={op} result=skip "
                f"reason=graphrag-not-enabled"
            )
            continue
        if op in ("reembed", "reingest", "resume-failed"):
            engine = await _resolve_ingestion_engine(ccat, op=op)
            if engine is None:
                ok = False
                continue
        if op == "reembed":
            try:
                reembed_ok = await _op_reembed(ccat, handler, collection)
            except Exception as exc:  # noqa: BLE001 - per-op isolation
                print(f"[maintenance] agent={agent_id} op={op} result=fail detail={exc}")
                ok = False
                continue
            if not reembed_ok:
                # _op_reembed already logged the per-source outcome summary.
                ok = False
            continue
        if op == "resume-failed":
            try:
                resume_ok = await _op_resume_failed(
                    ccat, handler, collection, json_output=json_output
                )
            except Exception as exc:  # noqa: BLE001 - per-op isolation
                print(f"[maintenance] agent={agent_id} op={op} result=fail detail={exc}")
                ok = False
                continue
            if not resume_ok:
                # _op_resume_failed already logged the per-source summary.
                ok = False
            continue
        if op == "reingest":
            try:
                reingest_ok = await _op_reingest(ccat, handler, collection)
            except Exception as exc:  # noqa: BLE001 - per-op isolation
                print(f"[maintenance] agent={agent_id} op={op} result=fail detail={exc}")
                ok = False
                continue
            if not reingest_ok:
                # _op_reingest already logged the failure: preflight, or the
                # per-source phase-outcome summary. Nothing to add here.
                ok = False
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
    try:
        from cat.plugins.cat_efficient_ingestion.registry import list_statuses
    except ImportError as exc:
        _abort_missing_effing(exc)
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
    checks.append(("concept-gen-active", active_gen == await handler._concept_fingerprint()))

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
        _print_plan(plan, as_json=args.json)
        sys.exit(2)

    # Skipped agents (e.g. plugin-not-active) are reported, not run.
    runnable = []
    for entry in plan:
        if entry.get("skip_reason"):
            if not args.json:
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
            # The resolved plan is the dry-run view (default mode): print it
            # before aborting, so the operator sees exactly what --yes would do.
            _print_plan(plan, as_json=args.json)
            if not args.json:
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
        if not args.json:
            print(f"[maintenance] agent={entry['agent_id']} progress=start")
        # Coordinate with the EffING periodic sweep: while this marker exists
        # the sweep skips the agent, so it can never steal/retry the rows this
        # CLI is processing. Deleted at the end, on success AND on failure.
        await _set_maintenance_marker(entry["agent_id"])
        try:
            ok = await _run_agent(
                entry["agent_id"],
                entry["ops"],
                entry["collection"],
                json_output=args.json,
            )
        except Exception as exc:  # noqa: BLE001 - per-agent isolation
            print(f"[maintenance] agent={entry['agent_id']} result=fail detail={exc}")
            ok = False
        finally:
            await _clear_maintenance_marker(entry["agent_id"])
        if not args.json:
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

    if args.json:
        print(
            json.dumps(
                {
                    "result": "ok" if all(results) and verify_ok else "fail",
                    "agents": [
                        {"agent": entry["agent_id"], "ok": ok}
                        for entry, ok in zip(runnable, results)
                    ],
                },
                ensure_ascii=False,
                default=str,
            )
        )

    if all(results) and verify_ok:
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())