#!/usr/bin/env python3
"""
Maintenance agent CLI for the Neo4j GraphRAG plugin.

Standalone, import-safe maintenance script run INSIDE the Cat container, as a process of its own. It drives the
public ingestion API of the core (``RabbitHole.reingest_source`` / ``rerun_source`` and the maintenance marker of
``cat.db.cruds.ingestions``): every source is processed by the same phase engine as an upload, under the lock of the
source, with its state row updated, so the server and this script never process a source at the same time.

Usage (inside the Cat container):
    # Re-embed only (the stored chunks are reused) for one agent
    docker exec cheshire_cat_core python /app/cat/plugins/cat_graphrag/scripts/maintenance_agent.py --agent <id> --reembed --yes

    # Re-ingest from scratch (parse + embed + graph) for all agents
    docker exec cheshire_cat_core python /app/cat/plugins/cat_graphrag/scripts/maintenance_agent.py --all --reingest --yes

    # Rebuild the GraphRAG graph for one agent
    docker exec cheshire_cat_core python /app/cat/plugins/cat_graphrag/scripts/maintenance_agent.py --agent <id> --graph --yes

    # The sources uploaded in the chats (episodic collection)
    docker exec cheshire_cat_core python /app/cat/plugins/cat_graphrag/scripts/maintenance_agent.py --agent <id> --reingest --collection episodic --yes

    # Show the plan without touching anything (exits 2)
    docker exec cheshire_cat_core python /app/cat/plugins/cat_graphrag/scripts/maintenance_agent.py --all --graph --dry-run

    # Run again ONLY the failed phases of the sources in error
    docker exec cheshire_cat_core python /app/cat/plugins/cat_graphrag/scripts/maintenance_agent.py --agent <id> --resume-failed --yes

    # Run the acceptance checks after the ops (read-only)
    docker exec cheshire_cat_core python /app/cat/plugins/cat_graphrag/scripts/maintenance_agent.py --agent <id> --reembed --yes --verify

Operations (combinable; they run in this order):
    --reingest       every source of the collection runs every phase again, from the bytes kept for it;
    --reembed        every source runs its embedding again on the stored chunks (and the phases depending on it);
    --resume-failed  every source in error runs its failed phases again (with the phases depending on them), as
                     the core's POST /ingestion/retry; a source that failed before its phases needs a new upload;
    --graph          GraphRAG agents: the graph of the tenant is wiped, then the graph phases of every source (both
                     collections) run again and the concept generation marker is aligned to the current settings.
A source ingested before the ingestion states existed (points in the vector memory, no state) is reingested by
--reingest, --reembed and --graph: it has no stored chunks, and that gives it a state.

Exit codes:
    0 = all agents/ops ok (and verify ok when --verify)
    1 = partial failure (a source or an op failed, or a --verify check failed)
    2 = aborted (dry-run plan printed, --yes missing, unknown agent, or --graph without the plugin deployed)

While an agent is processed it is in maintenance (``crud_ingestions.set_maintenance``): the automatic resume of the
server leaves it alone. The marker expires on its own (2h, refreshed per source) if this process dies.

IMPORT-SAFETY: this file lives in the plugin folder, which the Cat plugin loader imports recursively at activation.
It must have ZERO top-level side effects: only stdlib imports at module level; every ``cat`` / ``neo4j`` import
happens inside functions.
"""

import argparse
import asyncio
import copy
import importlib
import importlib.util
import json
import sys
from typing import Any, NotRequired, TypedDict

# The maintenance marker suspends the automatic resume of the agent; refreshed before every source, so a long run
# stays protected, and it expires on its own when this process dies.
_MAINTENANCE_TTL_SECONDS = 2 * 60 * 60

# What ``--yes`` confirms, per op (printed when it is missing).
_DESTRUCTIVE_STEPS: dict[str, list[str]] = {
    "reingest": [
        "every source runs every phase again from its kept bytes (its points and graph are replaced)",
    ],
    "reembed": [
        "every source runs its embedding again (its points are replaced) and the phases depending on it",
    ],
    "resume-failed": [
        "every source in error runs its failed phases again (a source that failed before them is reported)",
    ],
    "graph": [
        "wipe of the tenant graph (entity, similarity and RELATED_TO edges, entities left without edges)",
        "the graph phases of every source run again; the concept marker is aligned",
    ],
}

# The admin's default agent (cat.db.database.DEFAULT_SYSTEM_KEY) is always skipped by the enumeration; the legacy
# `default` agent is kept with a one-time warning (module-level flag: warn once per process, not per call).
_default_agent_warned = False

# The runtime of this process (plugins discovered, RabbitHole created) is prepared once (see _ensure_runtime).
_runtime_ready = False


class _PlanEntry(TypedDict):
    """One per-agent plan entry: target, ops, collection, optional skip."""

    agent_id: str
    ops: list[str]
    collection: str
    # Present when the agent is excluded from the run (the GraphRAG plugin is not active for it, or its vector
    # database is not the GraphRAG one, while `--graph` is the only op).
    skip_reason: NotRequired[str]
    # The sources in error `--resume-failed` will act on (dry-run plan).
    resume_sources: NotRequired[list[dict[str, Any]]]


def _parse_args() -> argparse.Namespace:
    """Parse the CLI: scope (--agent|--all), ops, collection, safety flags."""
    parser = argparse.ArgumentParser(
        prog="maintenance_agent",
        description=(
            "Agent maintenance for the Neo4j GraphRAG plugin: re-ingest, re-embed, resume the failed phases or "
            "rebuild the graph for one agent or all agents. Must run inside the Cat container."
        ),
    )

    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--agent",
        metavar="ID",
        action="append",
        help="Target one or more agents by id (repeatable). Explicitly naming a protected/system agent allows it.",
    )
    scope.add_argument(
        "--all",
        action="store_true",
        help="Target all agents (system and __*__ protected agents skipped).",
    )

    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Re-ingest from scratch: every phase of every source runs again from its kept bytes. Requires --yes.",
    )
    parser.add_argument(
        "--reembed",
        action="store_true",
        help="Re-embed only: the embedding runs again on the stored chunks (and the phases depending on it). "
             "Requires --yes.",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Rebuild the GraphRAG graph: wipe the tenant graph, run the graph phases of every source (both "
             "collections) again, align the concept marker. Requires --yes.",
    )
    parser.add_argument(
        "--resume-failed",
        action="store_true",
        help="Run again ONLY the failed phases of the sources in error (the completed phases are kept). "
             "Requires --yes.",
    )

    parser.add_argument(
        "--collection",
        choices=["declarative", "episodic"],
        default="declarative",
        help="The sources to operate on: the knowledge base (declarative, default) or the chats (episodic). "
             "--graph always covers both.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan (per agent+op steps) and exit 2 without any write.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the operations (every op rewrites points, graph or states).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run acceptance checks after the ops (read-only; never against the system agent).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the plan, the outcome of every op and the final summary as JSON (one document per line).",
    )

    args = parser.parse_args()

    ops = ("reingest", "reembed", "resume-failed", "graph")
    ops_attr = {
        "reingest": args.reingest,
        "reembed": args.reembed,
        "resume-failed": args.resume_failed,
        "graph": args.graph,
    }
    args.ops = [op for op in ops if ops_attr[op]]
    if not args.ops:
        parser.error("at least one operation is required: --reingest, --reembed, --resume-failed, --graph")
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
                "agent — included in the plan",
                file=sys.stderr,
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


def _plugin_module(name: str):
    """A module of the deployed GraphRAG plugin (``graphrag_handler``, ``phases``), or None when not deployed."""
    plugin_id = _resolve_plugin_id()
    if plugin_id is None:
        return None
    return importlib.import_module(f"cat.plugins.{plugin_id}.{name}")


def _scoped(rows: list[dict], collection: str | None) -> list[dict]:
    """The rows of a collection: the knowledge base (declarative) or the chats (episodic); None = every row."""
    if collection is None:
        return rows
    in_chats = collection == "episodic"
    return [row for row in rows if (row.get("chat_id") is not None) == in_chats]


def _failed_phases(row: dict) -> list[str]:
    """The phases of the diary in the error state (the dependents of a failed phase are among them)."""
    diary = row.get("completed_phases") or {}
    return [phase_id for phase_id, entry in diary.items() if (entry or {}).get("state") == "error"]


async def _resolve_resume_failed_plan(agent_id: str, collection: str) -> list[dict[str, Any]]:
    """Read-only: the sources in error of the collection, with their failed phases (dry-run plan)."""
    from cat.db.cruds import ingestions as crud_ingestions

    rows = _scoped(await crud_ingestions.list_statuses(agent_id), collection)
    return [
        {"source": row["source"], "chat_id": row.get("chat_id"), "failed_phases": _failed_phases(row)}
        for row in rows
        if row.get("status") == "error"
    ]


async def _build_ops_plan(args: argparse.Namespace) -> list[_PlanEntry]:
    """Build the per-agent ops plan.

    ``--agent <id>`` (repeatable) validates every id against the live agent list (aborts with exit 2 when unknown)
    and is the ONLY way to target a protected/system agent; ``--all`` expands to every agent minus ``system`` and
    ``__*__``. When ``--graph`` is the ONLY requested op, an agent is skipped with ``skip_reason="plugin-not-active"``
    (plugin off) or ``skip_reason="handler-not-graphrag"`` (plugin on, vector backend not GraphRAG). With a mixed
    plan the other ops still run, so the graph op is skipped per-op in ``_run_agent`` instead.

    Read-only (Redis only, no agent is loaded): safe for the dry-run.
    """
    explicit = list(dict.fromkeys(args.agent or []))
    if explicit:
        all_agents = await _list_agents(include_protected=True)
        unknown = [agent_id for agent_id in explicit if agent_id not in all_agents]
        if unknown:
            print(f"ABORTED: unknown agent {', '.join(unknown)}", file=_out(args.json))
            sys.exit(2)
        agent_ids = explicit
    else:
        agent_ids = await _list_agents()

    active: set[str] = set()
    if "graph" in args.ops:
        plugin_id = _resolve_plugin_id()
        if plugin_id is None:
            print("ABORTED: GraphRAG plugin not found in the plugins dir (--graph requires it)", file=_out(args.json))
            sys.exit(2)
        from cat.db.cruds import plugins as crud_plugins

        active = set(await crud_plugins.get_agents_plugin_keys(plugin_id))

    graph_ops_only = args.ops == ["graph"]

    plan: list[_PlanEntry] = []
    for agent_id in agent_ids:
        entry: _PlanEntry = {"agent_id": agent_id, "ops": list(args.ops), "collection": args.collection}
        if graph_ops_only and agent_id not in active:
            entry["skip_reason"] = "plugin-not-active"
        elif graph_ops_only and not await _agent_uses_graphrag_handler(agent_id):
            entry["skip_reason"] = "handler-not-graphrag"
        if "resume-failed" in args.ops:
            entry["resume_sources"] = await _resolve_resume_failed_plan(agent_id, args.collection)
        plan.append(entry)
    return plan


def _print_plan(plan: list[_PlanEntry], as_json: bool = False) -> None:
    """Print the dry-run plan: per agent+op the steps (and, for --resume-failed, the sources in error)."""
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
            scope = "declarative+episodic" if op == "graph" else collection
            print(f"  agent={agent_id} op={op} collection={scope}")
            for step in _DESTRUCTIVE_STEPS[op]:
                print(f"    - {step}")
            if op == "resume-failed":
                for item in entry.get("resume_sources", []):
                    print(
                        f"    source={item['source']} chat={item['chat_id']} "
                        f"failed_phases={','.join(item['failed_phases']) or '-'}"
                    )
    print("(dry-run: exiting 2, nothing was written)")


async def _ensure_runtime() -> None:
    """Prepare the core runtime of this process, once: the plugins and the RabbitHole of the lizard.

    In a fresh ``docker exec`` process the lizard is never bootstrapped (the server does it in the uvicorn
    lifespan): its plugin registry is empty, it has no RabbitHole and no websocket manager. Its full ``bootstrap``
    is NOT run: it would start the automatic resume of the ingestions, the websocket listener and the endpoints in
    this process.
    """
    global _runtime_ready
    if _runtime_ready:
        return
    from cat.looking_glass.bill_the_lizard import BillTheLizard
    from cat.rabbit_hole import RabbitHole

    from cat.services.websocket_manager import WebSocketManager

    lizard = BillTheLizard()
    await lizard.plugin_manager.discover_plugins()
    if lizard.rabbit_hole is None:
        lizard.rabbit_hole = RabbitHole()
    if lizard.websocket_manager is None:
        # the ingestion of a chat source asks it whether the chat is connected to this process: it never is (not
        # started, no connection, nothing sent)
        lizard.websocket_manager = WebSocketManager()
    _runtime_ready = True


async def _load_agent(agent_id: str):
    """The CheshireCat of the agent, as the server loads it."""
    await _ensure_runtime()
    from cat.looking_glass.bill_the_lizard import BillTheLizard

    ccat = await BillTheLizard().get_cheshire_cat(agent_id)
    if ccat is None:
        raise RuntimeError(f"the agent {agent_id} cannot be loaded (no settings)")
    return ccat


def _graphrag_handler(ccat):
    """The GraphRAG handler of the agent, or None when its vector memory is another backend."""
    module = _plugin_module("graphrag_handler")
    handler = ccat.vector_memory_handler
    if module is not None and isinstance(handler, module.GraphRAGHandler):
        return handler
    return None


async def _set_marker(agent_id: str) -> None:
    """Put (or keep) the agent in maintenance. Best-effort: a failed write is reported, never fatal."""
    from cat.db.cruds import ingestions as crud_ingestions

    try:
        await crud_ingestions.set_maintenance(agent_id, ttl=_MAINTENANCE_TTL_SECONDS)
    except Exception as exc:  # noqa: BLE001 - coordination is best-effort
        print(f"[maintenance] agent={agent_id} result=warn detail=maintenance-marker-write-failed error={exc}",
              file=sys.stderr)


async def _clear_marker(agent_id: str) -> None:
    """Take the agent out of maintenance (also on failure). A failed delete expires with the TTL."""
    from cat.db.cruds import ingestions as crud_ingestions

    try:
        await crud_ingestions.clear_maintenance(agent_id)
    except Exception as exc:  # noqa: BLE001 - the TTL is the fail-safe
        print(f"[maintenance] agent={agent_id} result=warn detail=maintenance-marker-delete-failed error={exc}",
              file=sys.stderr)


class _OpReport:
    """The outcome of one op on one agent: sources completed, gone (deleted meanwhile), failed."""

    def __init__(self, agent_id: str, op: str, json_output: bool):
        self.agent_id = agent_id
        self.op = op
        self.json_output = json_output
        self.completed = 0
        self.gone = 0
        self.failures: list[dict[str, Any]] = []

    def _fail(self, name: str | None, chat_id: str | None, failure: dict[str, Any], detail: str) -> None:
        self.failures.append({"source": name, "chat_id": chat_id, **failure})
        if not self.json_output:
            where = f" source={name} chat={chat_id}" if name is not None else ""
            print(f"[maintenance] agent={self.agent_id} op={self.op}{where} result=fail detail={detail}")

    def row(self, name: str, chat_id: str | None, row: dict | None) -> None:
        """The final row of a source: completed, gone (None), or a failure naming its failed phases."""
        if row is None:
            self.gone += 1
            return
        if row.get("status") == "completed":
            self.completed += 1
            return
        diary = row.get("completed_phases") or {}
        phases = {p: (diary[p] or {}).get("reason") for p in _failed_phases(row)}
        if not phases:
            phases = {row.get("phase") or "status": row.get("error") or row.get("status")}
        self._fail(name, chat_id, {"phases": phases},
                   f"phase-failed phases={json.dumps(phases, ensure_ascii=False, default=str)}")

    def error(self, name: str | None, chat_id: str | None, error: str) -> None:
        """A source (or the op itself, ``name=None``) that could not be processed."""
        self._fail(name, chat_id, {"error": error}, error)

    def finish(self) -> bool:
        ok = not self.failures
        if self.json_output:
            print(json.dumps({
                "agent": self.agent_id, "op": self.op, "result": "ok" if ok else "fail",
                "completed": self.completed, "gone": self.gone, "failures": self.failures,
            }, ensure_ascii=False, default=str))
        else:
            outcome = "ok" if ok else f"fail failed={len(self.failures)}"
            print(f"[maintenance] agent={self.agent_id} op={self.op} result={outcome} "
                  f"completed={self.completed} gone={self.gone}")
        return ok


async def _each_source(report: _OpReport, items: list[tuple[str, str | None]], call) -> None:
    """Run ``call(name, chat_id)`` on every source, the maintenance marker refreshed before each one; a failing
    source never stops the others."""
    from cat.rabbit_hole import SourceNotKept

    for name, chat_id in items:
        await _set_marker(report.agent_id)
        try:
            row = await call(name, chat_id)
        except SourceNotKept:
            report.error(name, chat_id, "source-not-kept")
            continue
        except Exception as exc:  # noqa: BLE001 - per-source isolation
            report.error(name, chat_id, f"{type(exc).__name__}: {exc}")
            continue
        report.row(name, chat_id, row)


async def _rows(agent_id: str, collection: str | None) -> list[tuple[str, str | None]]:
    from cat.db.cruds import ingestions as crud_ingestions

    return [(row["source"], row.get("chat_id")) for row in _scoped(await crud_ingestions.list_statuses(agent_id),
                                                                   collection)]


async def _sources_without_state(ccat, collections: list[str]) -> list[tuple[str, str | None]]:
    """The sources of the collections found in the vector memory without a state (ingested before the states
    existed), read from their points. Their bytes are not checked here: ``reingest_source`` reports a source whose
    file was not kept (``SourceNotKept``), so it is listed among the failures instead of being skipped."""
    from cat.db.cruds import ingestions as crud_ingestions

    with_state = {(row["source"], row.get("chat_id")) for row in await crud_ingestions.list_statuses(ccat.agent_key)}
    found: list[tuple[str, str | None]] = []
    for collection in collections:
        points, _ = await ccat.vector_memory_handler.get_all_tenant_points(collection, with_vectors=False)
        for point in points:
            metadata = (point.payload or {}).get("metadata") or {}
            key = (metadata.get("source"), metadata.get("chat_id"))
            if key[0] and key not in with_state and key not in found:
                found.append(key)
    return found


async def _op_reingest(ccat, report: _OpReport, collection: str) -> None:
    rabbit_hole = ccat.rabbit_hole
    items = await _rows(ccat.agent_key, collection) + await _sources_without_state(ccat, [collection])
    await _each_source(report, items, lambda name, chat_id: rabbit_hole.reingest_source(ccat, name, chat_id))


async def _with_dependents(ccat, phase_ids: list[str]) -> list[str]:
    """The phases and every phase depending on them, directly or not.

    ``rerun_source`` resets the diary entries of the phases it is given only: their dependents run again because the
    engine remembers, in memory, that a dependency ran. If this process dies after the new embedding is stored, the
    dependents look up to date to the automatic resume (e.g. the graph, purged by its phase and not written yet, is
    never rebuilt). Invalidating them too leaves them to do whatever happens.
    """
    from cat.rabbit_hole import engine

    rabbit_hole = copy.copy(ccat.rabbit_hole)
    await rabbit_hole.setup(ccat)
    phases = await engine.collect_phases(rabbit_hole, ccat)  # in execution order: a dependency comes first
    wanted = list(phase_ids)
    for phase in phases:
        if phase.id not in wanted and any(dep in wanted for dep in phase.depends_on):
            wanted.append(phase.id)
    return wanted


async def _op_reembed(ccat, report: _OpReport, collection: str) -> None:
    from cat.rabbit_hole.phases import PHASE_EMBEDDING

    rabbit_hole = ccat.rabbit_hole
    invalidate = await _with_dependents(ccat, [PHASE_EMBEDDING])
    await _each_source(
        report, await _rows(ccat.agent_key, collection),
        lambda name, chat_id: rabbit_hole.rerun_source(ccat, name, chat_id, invalidate=invalidate),
    )
    await _each_source(
        report, await _sources_without_state(ccat, [collection]),
        lambda name, chat_id: rabbit_hole.reingest_source(ccat, name, chat_id),
    )


async def _op_resume_failed(ccat, report: _OpReport, collection: str) -> None:
    """The rule of the core's retry (POST /ingestion/retry): the failed phases of a source in error run again, with
    the phases depending on them. A source in error with no failed phase left but phases already done (e.g. a
    ``--graph`` that reset the graph phases of every diary and died before rerunning it) has phases still to do: they
    run. A source that failed before any phase (e.g. a download: empty diary) has nothing to retry and is reported:
    it needs a new upload."""
    from cat.db.cruds import ingestions as crud_ingestions

    rabbit_hole = ccat.rabbit_hole
    failed = {}
    for row in _scoped(await crud_ingestions.list_statuses(ccat.agent_key), collection):
        if row.get("status") != "error":
            continue
        key = (row["source"], row.get("chat_id"))
        if phases := _failed_phases(row):
            failed[key] = phases
        elif row.get("completed_phases"):
            failed[key] = None  # no invalidation: the phases still to do run
        else:
            report.error(*key, "failed-before-its-phases: upload it again")
    await _each_source(
        report, list(failed),
        lambda name, chat_id: rabbit_hole.rerun_source(ccat, name, chat_id, invalidate=failed[(name, chat_id)]),
    )


async def _align_concept_marker(handler, agent_id: str, marker_at_start: str | None) -> bool:
    """Make the concepts of the rebuild the visible ones: flip the concept marker from the one read when the op
    started to the fingerprint of the settings the rebuild ran with (the generation its graph phases tagged), then
    collect the superseded concept nodes.

    A save of the concept settings during the rebuild flips the marker on its own (the recompute of the server, on
    the new settings): the flip then fails on purpose, never bringing back the older generation, and the conflict
    is reported (the concepts are those of the newer settings; ``--graph`` again aligns the rest)."""
    fingerprint = await handler._concept_fingerprint()
    if marker_at_start == fingerprint:
        return await handler._read_concept_gen(agent_id) == fingerprint
    if not await handler._flip_concept_gen(agent_id, marker_at_start, fingerprint):
        return False
    await handler._gc_stale_concept_nodes(agent_id)
    return True


async def _op_graph(ccat, handler, report: _OpReport) -> None:
    """Wipe the tenant graph, then run the graph phases of every source again.

    The graph entries of the diaries are reset BEFORE the wipe: whenever this process dies or the wipe fails, the
    sources are left stale and the automatic resume runs their graph phases again. Every source is then rerun with
    its graph phases invalidated under its lock (an ingestion of the server finishing meanwhile may have recorded
    them after the reset). The sources without a state are reingested: it is the only way to give them a graph
    again; the ones whose file was not kept cannot be rebuilt and are reported.
    """
    from cat.db.cruds import ingestions as crud_ingestions

    phases = _plugin_module("phases")
    graph_phases = [phases.GRAPH_SPACY_PHASE, phases.GRAPH_LLM_PHASE]
    agent_id = ccat.agent_key
    rabbit_hole = ccat.rabbit_hole

    await _set_marker(agent_id)
    concept_marker = await handler._read_concept_gen(agent_id) if handler._enable_concept_relations else None
    await crud_ingestions.reset_phase_entries_for_agent(agent_id, graph_phases)
    await handler.wipe_tenant_graph()
    await _each_source(
        report, await _rows(agent_id, None),
        lambda name, chat_id: rabbit_hole.rerun_source(ccat, name, chat_id, invalidate=graph_phases),
    )
    await _each_source(
        report, await _sources_without_state(ccat, ["declarative", "episodic"]),
        lambda name, chat_id: rabbit_hole.reingest_source(ccat, name, chat_id),
    )
    if handler._enable_concept_relations and not await _align_concept_marker(handler, agent_id, concept_marker):
        report.error(None, None, "concept-marker-conflict")


async def _run_agent(agent_id: str, ops: list[str], collection: str = "declarative",
                     json_output: bool = False) -> bool:
    """Run the requested ops for one agent, in order. ``--graph`` needs the GraphRAG handler: without it the op is
    skipped (the others run on any vector backend). Returns True when every op succeeded or was skipped."""
    ccat = await _load_agent(agent_id)
    handler = _graphrag_handler(ccat)

    ok = True
    for op in ops:
        if op == "graph" and handler is None:
            if not json_output:
                print(f"[maintenance] agent={agent_id} op={op} result=skip reason=handler-not-graphrag")
            continue
        report = _OpReport(agent_id, op, json_output)
        try:
            if op == "reingest":
                await _op_reingest(ccat, report, collection)
            elif op == "reembed":
                await _op_reembed(ccat, report, collection)
            elif op == "resume-failed":
                await _op_resume_failed(ccat, report, collection)
            else:
                await _op_graph(ccat, handler, report)
        except Exception as exc:  # noqa: BLE001 - per-op isolation
            report.error(None, None, f"{type(exc).__name__}: {exc}")
        ok = report.finish() and ok
    return ok


async def _graph_checks(handler, agent_id: str) -> list[tuple[str, bool]]:
    """Read-only checks of the tenant graph: every Document has the embedding of the current generation, no entity
    is left without edges, the visible concept generation is the one of the current settings."""
    await handler._ensure_connected()
    gen = await handler._read_generation(agent_id)

    async def scalar(query: str) -> Any:
        async with handler._get_session() as session:
            result = await session.run(query, tenant_id=agent_id)
            record = await result.single()
            return record["n"] if record is not None else None

    null_embeddings = await scalar(
        f"MATCH (d:Document {{tenant_id: $tenant_id}}) WHERE d.embedding_{gen} IS NULL RETURN count(d) AS n"
    )
    orphans = await scalar("MATCH (e:Entity {tenant_id: $tenant_id}) WHERE NOT (e)--() RETURN count(e) AS n")
    checks = [("no-null-embeddings", null_embeddings == 0), ("no-orphan-entities", orphans == 0)]
    if handler._enable_concept_relations:
        fingerprint = await handler._concept_fingerprint()
        # no marker yet: the concepts of the current fingerprint are the visible ones
        active = await handler._read_concept_gen(agent_id) or fingerprint
        checks.append(("concept-gen-active", active == fingerprint))
    return checks


async def _verify_agent(ccat, handler, collection: str) -> list[tuple[str, bool]]:
    """Read-only acceptance checks: every source of the collection completed with the current tag of every phase
    (nothing left for the automatic resume), then the graph checks for a GraphRAG agent."""
    from cat.db.cruds import ingestions as crud_ingestions
    from cat.rabbit_hole import engine
    from cat.rabbit_hole.phases import compute_tags

    rabbit_hole = copy.copy(ccat.rabbit_hole)
    await rabbit_hole.setup(ccat)
    tags = await compute_tags(await engine.collect_phases(rabbit_hole, ccat), ccat)
    rows = _scoped(await crud_ingestions.list_statuses(ccat.agent_key), collection)
    current = all(
        row.get("status") == "completed"
        and all(((row.get("completed_phases") or {}).get(p) or {}).get("tag") == tag for p, tag in tags.items())
        for row in rows
    )
    checks = [("rows-completed", current)]
    if handler is not None:
        checks += await _graph_checks(handler, ccat.agent_key)
    return checks


def _logs_to_stderr() -> None:
    """With ``--json`` stdout carries only the JSON documents: the logs of the core (and of the plugins) go to
    stderr."""
    from loguru import logger

    from cat.log import log

    logger.remove()
    logger.add(sys.stderr, level=log.LOG_LEVEL, filter=log.show_log_level)


def _out(json_output: bool):
    """Where a plain message goes: stdout, or stderr when stdout carries the JSON documents."""
    return sys.stderr if json_output else sys.stdout


async def main() -> None:
    """CLI entrypoint: parse -> runtime guard -> plan -> dry-run/yes gates -> run -> verify."""
    args = _parse_args()

    if not _runtime_guard():
        sys.exit(2)
    if args.json:
        _logs_to_stderr()

    plan = await _build_ops_plan(args)

    if args.dry_run:
        _print_plan(plan, as_json=args.json)
        sys.exit(2)

    runnable = []
    for entry in plan:
        if entry.get("skip_reason"):
            if not args.json:
                print(f"[maintenance] agent={entry['agent_id']} result=skip reason={entry.get('skip_reason')}")
        else:
            runnable.append(entry)

    if not args.yes and runnable:
        _print_plan(plan, as_json=args.json)
        if not args.json:
            print("ABORTED: the operations require --yes:")
            for entry in runnable:
                for op in entry["ops"]:
                    print(f"  agent={entry['agent_id']} op={op} steps:")
                    for step in _DESTRUCTIVE_STEPS[op]:
                        print(f"    - {step}")
            print("Re-run with --yes to confirm (or --dry-run to preview).")
        sys.exit(2)

    results = []
    for entry in runnable:
        # One agent at a time; while it is processed the automatic resume of the server leaves it alone.
        agent_id = entry["agent_id"]
        if not args.json:
            print(f"[maintenance] agent={agent_id} progress=start")
        await _set_marker(agent_id)
        try:
            ok = await _run_agent(agent_id, entry["ops"], entry["collection"], json_output=args.json)
        except Exception as exc:  # noqa: BLE001 - per-agent isolation
            print(f"[maintenance] agent={agent_id} result=fail detail={type(exc).__name__}: {exc}",
                  file=_out(args.json))
            ok = False
        finally:
            await _clear_marker(agent_id)
        if not args.json:
            print(f"[maintenance] agent={agent_id} progress=done")
        results.append(ok)

    verify_ok = True
    if args.verify:
        for entry in runnable:
            agent_id = entry["agent_id"]
            try:
                ccat = await _load_agent(agent_id)
                checks = await _verify_agent(ccat, _graphrag_handler(ccat), entry["collection"])
            except Exception as exc:  # noqa: BLE001 - per-agent isolation
                checks = [(f"error: {type(exc).__name__}: {exc}", False)]
            failed = [name for name, ok_ in checks if not ok_]
            verify_ok = verify_ok and not failed
            if args.json:
                print(json.dumps({"agent": agent_id, "verify": dict(checks)}, ensure_ascii=False, default=str))
            elif failed:
                print(f"[maintenance] agent={agent_id} result=verify-fail checks={len(checks) - len(failed)}/"
                      f"{len(checks)} failed={','.join(failed)}")
            else:
                print(f"[maintenance] agent={agent_id} result=verify-ok checks={len(checks)}/{len(checks)}")
        if not args.json:
            print(f"[maintenance] result=verify-{'ok' if verify_ok else 'fail'}")

    if args.json:
        print(json.dumps({
            "result": "ok" if all(results) and verify_ok else "fail",
            "agents": [{"agent": entry["agent_id"], "ok": ok} for entry, ok in zip(runnable, results)],
        }, ensure_ascii=False, default=str))

    sys.exit(0 if all(results) and verify_ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
