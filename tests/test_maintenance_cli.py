"""The maintenance CLI (``scripts/maintenance_agent.py``) on the REAL ingestion API of the core.

The CLI runs as a separate process and drives the core's public API: ``RabbitHole.reingest_source`` /
``rerun_source`` (the REAL methods: locks, rows, phase engine, the graph phases of this plugin) and the maintenance
marker of ``cat.db.cruds.ingestions``. Only the storage is faked: Neo4j (``FakeGraphStore``), the parsing and the
embedding (``core_rabbit_hole``), the uploaded files (``FakeFileManager``).

Covered, through the command line (exit code and output):

- ``--reingest``: every source of the collection is processed again from scratch, the sources without a state
  (ingested before the states existed) get one; a source whose bytes were not kept fails without being touched;
- ``--reembed``: the embedding runs again on the stored chunks (no parsing), the sources without a state are
  reingested;
- ``--resume-failed``: only the failed phases of the sources in error run again, in either collection;
- ``--graph``: the graph of every source (both collections, with or without a state) is rebuilt after the wipe and
  the concept marker is aligned; it runs whatever the concept toggle; a concurrent concept flip is retried;
- the maintenance marker holds during an operation and is gone after it (also on failure);
- per-source isolation, sources deleted meanwhile, dry-run and missing ``--yes`` (no write), ``--json``,
  ``--verify``, agent scoping, the runtime of the separate process.

Standalone script (plugin rules: the body runs only as ``__main__``). Needs the core checkout and a Redis Stack on
localhost:6379 (database 14 is used and cleaned):

    cd ../grinning-cat-core
    .venv/bin/python /path/to/cat-graphrag/tests/test_maintenance_cli.py
"""

if __name__ == "__main__":
    import asyncio
    import contextlib
    import importlib.util
    import io
    import json
    import os
    import sys
    import traceback
    import uuid

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import graph_fakes as gf

    main_mod, handler_mod, phases_mod = gf.bootstrap("_graphrag_cli")

    from cat.db.cruds import ingestions as crud_ingestions
    from cat.db.cruds import settings as crud_settings
    from cat.db.database import get_async_db
    from cat.rabbit_hole import engine
    from cat.services.memory.models import PointStruct

    ma = gf.load_cli(handler_mod, phases_mod)

    def types_namespace(**kw):
        import types

        return types.SimpleNamespace(**kw)

    WORLDS = {}
    real_load_agent = ma._load_agent

    async def load_agent(agent_id):
        return WORLDS[agent_id].cat

    ma._load_agent = load_agent

    async def new_world(concepts=True, graphrag=True):
        agent = f"cli_{uuid.uuid4().hex[:10]}"
        await gf.create_agent(agent, graphrag=graphrag)
        world = gf.CliWorld(agent, main_mod, handler_mod, PointStruct, concepts=concepts)
        WORLDS[agent] = world
        return world

    async def cli(*argv):
        return await gf.run_cli(ma, *argv)

    def lines_with(lines, *parts):
        return [line for line in lines if all(p in line for p in parts)]

    # ── --reingest ──────────────────────────────────────────────────────────
    async def test_reingest_processes_every_source_of_the_collection_and_adopts_legacy():
        w = await new_world()
        await w.upload("a.pdf", ["Alice met Bob"])
        await w.upload("c.pdf", ["Carol met Dave"], chat="chat-1")
        await w.legacy("old.pdf", ["Erin met Frank", "Grace"], custom={"author": "Erin"})
        chat_before = await w.row("c.pdf", "chat-1")
        w.files[(None, "a.pdf")] = ["Grace met Alice", "Bob"]  # the file changed: a reingest parses it again
        code, lines = await cli("--agent", w.agent, "--reingest", "--yes")
        assert code == 0, lines
        for name in ("a.pdf", "old.pdf"):
            row = await w.row(name)
            assert row is not None and row["status"] == "completed", (name, row)
            assert w.graph_ok(name), name
        assert await w.row("c.pdf", "chat-1") == chat_before, "the episodic collection was touched"
        adopted = (await w.row("old.pdf"))["metadata"]
        assert adopted == {"author": "Erin"}, f"the upload metadata of an old source are not its own: {adopted}"
        assert lines_with(lines, "op=reingest", "result=ok"), lines
        assert not await crud_ingestions.is_in_maintenance(w.agent)

    async def test_reingest_source_not_kept_fails_untouched():
        w = await new_world()
        await w.upload("a.pdf", ["Alice met Bob"])
        await w.upload("lost.pdf", ["Carol met Dave"], keep=False)
        before = await w.row("lost.pdf")
        docs_before = w.store.scoped_doc_ids("lost.pdf", gf.DECLARATIVE, None)
        code, lines = await cli("--agent", w.agent, "--reingest", "--yes")
        assert code == 1, lines
        assert lines_with(lines, "source=lost.pdf", "detail=source-not-kept"), lines
        assert lines_with(lines, "op=reingest", "result=fail"), lines
        assert await w.row("lost.pdf") == before, "a source that cannot be reingested was touched"
        assert w.store.scoped_doc_ids("lost.pdf", gf.DECLARATIVE, None) == docs_before
        assert (await w.row("a.pdf"))["status"] == "completed"
        assert not await crud_ingestions.is_in_maintenance(w.agent)

    async def test_reingest_episodic_collection():
        w = await new_world()
        await w.upload("a.pdf", ["Alice met Bob"])
        await w.upload("c.pdf", ["Carol met Dave"], chat="chat-1")
        kb_before = await w.row("a.pdf")
        w.files[("chat-1", "c.pdf")] = ["Erin met Carol"]
        code, lines = await cli("--agent", w.agent, "--reingest", "--collection", "episodic", "--yes")
        assert code == 0, lines
        assert w.graph_ok("c.pdf", "chat-1")
        assert await w.row("a.pdf") == kb_before

    # ── --reembed ───────────────────────────────────────────────────────────
    async def test_reembed_runs_the_embedding_only_and_reingests_legacy():
        w = await new_world()
        await w.upload("a.pdf", ["Alice met Bob", "Carol"])
        await w.legacy("old.pdf", ["Erin met Frank"])
        parsing_before = (await w.row("a.pdf"))["completed_phases"]["parsing_chunking"]
        ids_before = w.store.scoped_doc_ids("a.pdf", gf.DECLARATIVE, None)
        code, lines = await cli("--agent", w.agent, "--reembed", "--yes")
        assert code == 0, lines
        row = await w.row("a.pdf")
        assert row["status"] == "completed"
        assert row["completed_phases"]["parsing_chunking"] == parsing_before, "reembed parsed again"
        assert row["completed_phases"]["embedding"]["finished_at"] >= parsing_before["finished_at"]
        assert not ids_before & w.store.scoped_doc_ids("a.pdf", gf.DECLARATIVE, None), "the points were not replaced"
        assert w.graph_ok("a.pdf")
        legacy = await w.row("old.pdf")
        assert legacy is not None and legacy["status"] == "completed" and w.graph_ok("old.pdf")
        assert lines_with(lines, "op=reembed", "result=ok"), lines

    async def test_reembed_dying_in_a_dependent_phase_is_resumed():
        """--reembed invalidates the embedding AND the phases depending on it: if the process dies after the new
        points are stored (the graph phase purged the old graph and dies writing the new one), the automatic resume
        finds the graph phases still to do and rebuilds the graph."""
        import types as pytypes

        from cat.rabbit_hole import recovery

        w = await new_world()
        await w.upload("contratto.pdf", ["Alice met Bob"])
        write = w.handler._write_source_graph

        async def dies(*a, **k):
            raise asyncio.CancelledError()  # the container restarts

        w.handler._write_source_graph = dies
        try:
            await cli("--agent", w.agent, "--reembed", "--yes")
        except asyncio.CancelledError:
            pass
        finally:
            w.handler._write_source_graph = write

        async def get_cat(agent_id):
            return w.cat

        await crud_ingestions.clear_maintenance(w.agent)  # the marker of the dead process expired
        await recovery.resume_agent(pytypes.SimpleNamespace(rabbit_hole=w.rh, get_cheshire_cat=get_cat), w.agent)
        row = await w.row("contratto.pdf")
        assert row["status"] == "completed" and w.graph_ok("contratto.pdf"), (row["status"], w.store.entities)

    # ── --resume-failed ─────────────────────────────────────────────────────
    async def test_resume_failed_reruns_only_the_failed_phases():
        w = await new_world()
        await w.upload("ok.pdf", ["Alice met Bob"])
        w.store.fail_writes = 1
        row = await w.upload("bad.pdf", ["Carol met Dave"])
        w.store.fail_writes = 0
        assert row["status"] == "error", row
        ok_before = await w.row("ok.pdf")
        embedding_before = row["completed_phases"]["embedding"]
        code, lines = await cli("--agent", w.agent, "--resume-failed", "--yes")
        assert code == 0, lines
        bad = await w.row("bad.pdf")
        assert bad["status"] == "completed" and w.graph_ok("bad.pdf"), bad
        assert bad["completed_phases"]["embedding"] == embedding_before, "a healthy phase ran again"
        assert await w.row("ok.pdf") == ok_before, "a completed source was touched"
        assert lines_with(lines, "op=resume-failed", "result=ok"), lines

    async def test_resume_failed_failing_again_reports_phase_and_reason():
        w = await new_world()
        w.store.fail_writes = 1
        await w.upload("bad.pdf", ["Carol met Dave"])
        w.store.fail_writes = 1  # Neo4j still down
        code, lines = await cli("--agent", w.agent, "--resume-failed", "--yes")
        assert code == 1, lines
        failed = lines_with(lines, "source=bad.pdf", "detail=phase-failed")
        assert failed and "graph_spacy_extraction" in failed[0] and "neo4j unavailable" in failed[0], lines
        assert (await w.row("bad.pdf"))["status"] == "error"
        assert not await crud_ingestions.is_in_maintenance(w.agent)

    async def test_resume_failed_source_failed_before_its_phases():
        """Like the core's retry: a source that failed before any phase (e.g. its download) needs a new upload."""
        w = await new_world()
        await crud_ingestions.create_status(w.agent, None, "https://x.test/doc", type_="url")
        await crud_ingestions.update_status(w.agent, None, "https://x.test/doc", status="error", error="404")
        before = await w.row("https://x.test/doc")
        code, lines = await cli("--agent", w.agent, "--resume-failed", "--yes")
        assert code == 1, lines
        assert lines_with(lines, "source=https://x.test/doc", "failed-before-its-phases"), lines
        assert await w.row("https://x.test/doc") == before, "the source was run again"

    async def test_resume_failed_after_a_graph_that_died():
        """A source in error, then a --graph that reset the graph phases of every diary and died before rerunning it:
        the source is in error with no failed phase left, but with phases still to do. --resume-failed brings it up
        to date (a source that failed before ANY phase, e.g. its download, is still left to a new upload)."""
        w = await new_world()
        w.store.fail_writes = 1
        row = await w.upload("b.pdf", ["Carol met Dave"])
        w.store.fail_writes = 0
        assert row["status"] == "error", row
        await crud_ingestions.reset_phase_entries_for_agent(w.agent, [phases_mod.GRAPH_SPACY_PHASE, phases_mod.GRAPH_LLM_PHASE])
        await w.handler.wipe_tenant_graph()  # what the dead --graph did

        code, lines = await cli("--agent", w.agent, "--resume-failed", "--yes")
        assert code == 0, lines
        assert (await w.row("b.pdf"))["status"] == "completed" and w.graph_ok("b.pdf")

    async def test_resume_failed_in_a_chat():
        w = await new_world()
        w.store.fail_writes = 1
        await w.upload("c.pdf", ["Carol met Dave"], chat="chat-1")
        w.store.fail_writes = 0
        code, lines = await cli("--agent", w.agent, "--resume-failed", "--collection", "episodic", "--yes")
        assert code == 0, lines
        assert (await w.row("c.pdf", "chat-1"))["status"] == "completed" and w.graph_ok("c.pdf", "chat-1")

    # ── --graph ─────────────────────────────────────────────────────────────
    async def test_graph_rebuilds_every_source_and_aligns_the_concept_marker():
        w = await new_world()
        await w.upload("a.pdf", ["Alice met Bob"])
        await w.upload("c.pdf", ["Carol met Dave"], chat="chat-1")
        await w.legacy("old.pdf", ["Erin met Frank"])
        w.store.entities["Ghost"] = {"tracked": False}  # a leftover the per-source purge never reaches
        w.store.concept_marker = "fp-old"
        code, lines = await cli("--agent", w.agent, "--graph", "--yes")
        assert code == 0, lines
        assert w.store.wipes == 1
        assert "Ghost" not in w.store.entities, "the wipe left a stale entity"
        assert w.graph_ok("a.pdf") and w.graph_ok("c.pdf", "chat-1") and w.graph_ok("old.pdf")
        assert (await w.row("old.pdf"))["status"] == "completed", "the source without a state got no graph"
        assert w.store.concept_marker == "fp-1" and w.store.concept_gcs == 1
        assert lines_with(lines, "op=graph", "result=ok"), lines

    async def test_graph_runs_with_concepts_disabled():
        w = await new_world(concepts=False)
        await w.upload("a.pdf", ["Alice met Bob"])
        code, lines = await cli("--agent", w.agent, "--graph", "--yes")
        assert code == 0, lines
        assert w.store.wipes == 1 and w.graph_ok("a.pdf", concepts=False)
        assert (await w.row("a.pdf"))["completed_phases"]["graph_llm_extraction"]["state"] == "skipped"
        assert w.store.concept_marker is None, "the marker moved with the concepts off"

    async def test_graph_concept_settings_saved_meanwhile_are_not_reverted():
        w = await new_world()
        await w.upload("a.pdf", ["Alice met Bob"])
        w.store.concept_marker = "fp-0"
        wipe = w.handler.wipe_tenant_graph

        async def wipe_then_save():
            await wipe()
            w.store.concept_marker = "fp-new"  # an admin saves new concept definitions: the server flips

        w.handler.wipe_tenant_graph = wipe_then_save
        try:
            code, lines = await cli("--agent", w.agent, "--graph", "--yes")
        finally:
            w.handler.wipe_tenant_graph = wipe
        assert code == 1, lines
        assert lines_with(lines, "op=graph", "detail=concept-marker-conflict"), lines
        assert w.store.concept_marker == "fp-new" and w.store.concept_gcs == 0, "the newer generation was reverted"
        w.store.concept_marker = "fp-1"
        code, lines = await cli("--agent", w.agent, "--graph", "--yes")  # already aligned, nobody flips meanwhile
        assert code == 0, lines

    async def test_graph_reports_legacy_sources_it_cannot_rebuild():
        w = await new_world()
        await w.upload("a.pdf", ["Alice met Bob"])
        await w.legacy("cloud.pdf", ["Carol met Dave"])
        w.kept.discard((None, "cloud.pdf"))  # ingested without storing its file
        code, lines = await cli("--agent", w.agent, "--graph", "--yes")
        assert code == 1, lines
        assert lines_with(lines, "source=cloud.pdf", "detail=source-not-kept"), lines
        assert w.store.wipes == 1 and w.graph_ok("a.pdf")
        assert await w.row("cloud.pdf") is None
        assert w.store.scoped_doc_ids("cloud.pdf", gf.DECLARATIVE, None), "its points must stay"
        code, lines = await cli("--agent", w.agent, "--reembed", "--yes")
        assert code == 1 and lines_with(lines, "source=cloud.pdf", "detail=source-not-kept"), lines

    async def test_graph_reset_precedes_the_wipe():
        """A --graph dying in its wipe leaves the sources stale: the automatic resume rebuilds them."""
        w = await new_world()
        await w.upload("a.pdf", ["Alice met Bob"])
        wipe = w.handler.wipe_tenant_graph

        async def dying_wipe():
            await wipe()
            raise asyncio.CancelledError()

        w.handler.wipe_tenant_graph = dying_wipe
        try:
            await cli("--agent", w.agent, "--graph", "--yes")
        except asyncio.CancelledError:
            pass
        finally:
            w.handler.wipe_tenant_graph = wipe
        tags = await gf.current_tags(w.rh, w.cat)
        diary = (await w.row("a.pdf"))["completed_phases"]
        assert any((diary.get(p) or {}).get("tag") != t for p, t in tags.items()), "the wiped source looks up to date"

    async def test_json_logs_go_to_stderr():
        from cat.log import log

        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                ma._logs_to_stderr()
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                log.warning("probe-line")
            assert "probe-line" in err.getvalue() and "probe-line" not in out.getvalue()
        finally:
            log.default_log()

    async def test_graph_already_aligned_marker_is_left_alone():
        w = await new_world()
        await w.upload("a.pdf", ["Alice met Bob"])
        w.store.concept_marker = "fp-1"
        code, _ = await cli("--agent", w.agent, "--graph", "--yes")
        assert code == 0 and w.store.concept_gcs == 0

    async def test_graph_skipped_when_handler_not_graphrag_other_ops_run():
        w = await new_world()
        await w.upload("a.pdf", ["Alice met Bob"])
        original = ma._graphrag_handler  # e.g. Qdrant: the engine ops still run, the graph op cannot
        ma._graphrag_handler = lambda ccat: None
        try:
            code, lines = await cli("--agent", w.agent, "--reingest", "--graph", "--yes")
        finally:
            ma._graphrag_handler = original
        assert code == 0, lines
        assert lines_with(lines, "op=graph", "result=skip", "reason=handler-not-graphrag"), lines
        assert lines_with(lines, "op=reingest", "result=ok"), lines
        assert w.store.wipes == 0

    async def test_graph_only_plan_skips_agents_without_graphrag():
        w = await new_world(graphrag=False)
        code, lines = await cli("--agent", w.agent, "--graph", "--yes")
        assert code == 0, lines
        assert lines_with(lines, w.agent, "result=skip", "reason=plugin-not-active"), lines

    # ── lifecycle ───────────────────────────────────────────────────────────
    async def test_marker_held_during_the_op_and_cleared_on_failure():
        w = await new_world()
        await w.upload("a.pdf", ["Alice met Bob"])
        await w.upload("b.pdf", ["Carol met Dave"])
        seen = []
        original = w.rh.rerun_source

        async def rerun(cat, name, chat_id=None, blocking=True, invalidate=None):
            seen.append(await crud_ingestions.is_in_maintenance(w.agent))
            if name == "a.pdf":
                raise RuntimeError("redis hiccup")
            return await original(cat, name, chat_id, blocking=blocking, invalidate=invalidate)

        w.rh.rerun_source = rerun
        code, lines = await cli("--agent", w.agent, "--reembed", "--yes")
        assert seen == [True, True], seen
        assert code == 1, lines
        assert lines_with(lines, "source=a.pdf", "result=fail", "redis hiccup"), lines
        assert (await w.row("b.pdf"))["status"] == "completed", "one failing source stopped the others"
        assert not await crud_ingestions.is_in_maintenance(w.agent)

    async def test_agent_failure_clears_the_marker():
        w = await new_world()

        async def broken(agent_id):
            raise RuntimeError("agent cannot load")

        ma._load_agent = broken
        try:
            code, lines = await cli("--agent", w.agent, "--reembed", "--yes")
        finally:
            ma._load_agent = load_agent
        assert code == 1 and lines_with(lines, w.agent, "result=fail", "agent cannot load"), lines
        assert not await crud_ingestions.is_in_maintenance(w.agent)

    async def test_source_deleted_meanwhile_is_gone_not_failed():
        w = await new_world()
        await w.upload("a.pdf", ["Alice met Bob"])
        await w.upload("b.pdf", ["Carol met Dave"])
        original = w.rh.rerun_source

        async def rerun(cat, name, chat_id=None, blocking=True, invalidate=None):
            if name == "b.pdf":  # the user deletes it while the maintenance runs
                await gf.delete(engine, w.rh, w.cat, None, "b.pdf")
            return await original(cat, name, chat_id, blocking=blocking, invalidate=invalidate)

        w.rh.rerun_source = rerun
        code, lines = await cli("--agent", w.agent, "--reembed", "--yes")
        assert code == 0, lines
        assert lines_with(lines, "op=reembed", "result=ok", "gone=1"), lines
        assert await w.row("b.pdf") is None

    async def test_dry_run_and_missing_yes_write_nothing():
        w = await new_world()
        await w.upload("a.pdf", ["Alice met Bob"])
        w.store.fail_writes = 1
        await w.upload("bad.pdf", ["Carol met Dave"])
        rows_before = await crud_ingestions.list_statuses(w.agent)
        docs_before = dict(w.store.docs)

        async def must_not_load(agent_id):
            raise AssertionError("the dry-run bootstrapped the agent")

        ma._load_agent = must_not_load
        try:
            code, lines = await cli("--agent", w.agent, "--resume-failed", "--graph", "--dry-run")
            assert code == 2, lines
            assert lines_with(lines, "source=bad.pdf", "failed_phases=graph_spacy_extraction"), lines
            assert lines_with(lines, "op=graph"), lines
            code, lines = await cli("--agent", w.agent, "--reingest")
            assert code == 2 and lines_with(lines, "ABORTED", "--yes"), lines
            code, lines = await cli("--agent", w.agent, "--reingest", "--dry-run", "--json")
            assert code == 2 and json.loads(lines[-1])["dry_run"] is True, lines
            code, lines = await cli("--agent", w.agent, "--reingest", "--json")  # no --yes: the plan, JSON only
            assert code == 2 and [json.loads(line)["dry_run"] for line in lines] == [True], lines
        finally:
            ma._load_agent = load_agent
        assert await crud_ingestions.list_statuses(w.agent) == rows_before
        assert w.store.docs == docs_before and w.store.wipes == 0
        assert not await crud_ingestions.is_in_maintenance(w.agent)

    async def test_json_output():
        w = await new_world()
        await w.upload("a.pdf", ["Alice met Bob"])
        await w.upload("lost.pdf", ["Carol"], keep=False)
        code, lines = await cli("--agent", w.agent, "--reingest", "--yes", "--json")
        assert code == 1, lines
        docs = [json.loads(line) for line in lines]  # every line is JSON
        op = next(d for d in docs if d.get("op") == "reingest")
        assert op["result"] == "fail" and op["completed"] == 1, op
        assert op["failures"] == [{"source": "lost.pdf", "chat_id": None, "error": "source-not-kept"}], op
        assert docs[-1]["result"] == "fail" and docs[-1]["agents"] == [{"agent": w.agent, "ok": False}], docs[-1]

    async def test_verify():
        w = await new_world()
        await w.upload("a.pdf", ["Alice met Bob"])
        checks_seen = []

        async def graph_checks(handler, agent_id):
            checks_seen.append(agent_id)
            return [("no-orphan-entities", True)]

        original = ma._graph_checks
        ma._graph_checks = graph_checks
        try:
            code, lines = await cli("--agent", w.agent, "--reembed", "--yes", "--verify")
            assert code == 0 and lines_with(lines, w.agent, "result=verify-ok"), lines
            assert checks_seen == [w.agent]
            # a source in error: the verification fails and names the check
            w.store.fail_writes = 1
            await w.upload("bad.pdf", ["Carol met Dave"])
            w.store.fail_writes = 1
            code, lines = await cli("--agent", w.agent, "--resume-failed", "--yes", "--verify")
            assert code == 1 and lines_with(lines, "result=verify-fail", "rows-completed"), lines
            # a phase whose settings changed and was not rerun yet is not current
            w.store.fail_writes = 0
            await cli("--agent", w.agent, "--resume-failed", "--yes")
            w.handler._spacy_models = {"en": "en_core_web_lg"}
            code, lines = await cli("--agent", w.agent, "--resume-failed", "--yes", "--verify")
            assert code == 1 and lines_with(lines, "result=verify-fail", "rows-completed"), lines
        finally:
            ma._graph_checks = original
            w.handler._spacy_models = {"en": "en_core_web_sm"}

    async def test_verify_without_graphrag_checks_rows_only():
        w = await new_world()
        await w.upload("a.pdf", ["Alice met Bob"])
        original = ma._graphrag_handler
        ma._graphrag_handler = lambda ccat: None
        try:
            code, lines = await cli("--agent", w.agent, "--reembed", "--yes", "--verify")
        finally:
            ma._graphrag_handler = original
        assert code == 0 and lines_with(lines, w.agent, "result=verify-ok", "checks=1/1"), lines

    async def test_scoping():
        w = await new_world()
        protected = f"__cli_{uuid.uuid4().hex[:6]}__"
        await gf.create_agent(protected)
        try:
            code, lines = await cli("--all", "--reembed", "--dry-run")
            assert code == 2
            assert lines_with(lines, w.agent) and not lines_with(lines, protected), lines
            code, lines = await cli("--agent", protected, "--reembed", "--dry-run")
            assert code == 2 and lines_with(lines, protected), lines
            code, lines = await cli("--agent", "nobody_here", "--reembed", "--yes")
            assert code == 2 and lines_with(lines, "unknown agent"), lines
        finally:
            await gf.destroy_agent(protected)

    async def test_runtime_of_the_separate_process():
        from cat.looking_glass.bill_the_lizard import BillTheLizard
        from cat.rabbit_hole import RabbitHole

        lizard = BillTheLizard()
        calls = []

        async def discover():
            calls.append(1)

        saved = lizard.plugin_manager.discover_plugins, lizard.rabbit_hole, lizard.websocket_manager
        lizard.plugin_manager.discover_plugins = discover
        lizard.rabbit_hole = None
        lizard.websocket_manager = None
        ma._runtime_ready = False
        try:
            await ma._ensure_runtime()
            await ma._ensure_runtime()
            assert calls == [1], calls
            assert isinstance(lizard.rabbit_hole, RabbitHole)
            # a chat source asks whether its chat is connected to this process: never, and no error
            assert lizard.websocket_manager is not None and not lizard.websocket_manager.is_connected("chat-1")
            assert lizard._ingestion_resume is None, "the resume loop must not run in the CLI process"
        finally:
            lizard.plugin_manager.discover_plugins, lizard.rabbit_hole, lizard.websocket_manager = saved
            ma._runtime_ready = False

    async def test_load_agent_uses_the_lizard():
        from cat.looking_glass.bill_the_lizard import BillTheLizard

        lizard = BillTheLizard()
        got = []

        async def get_cheshire_cat(agent_id):
            got.append(agent_id)
            return "ccat"

        async def ready():
            return None

        saved = ma._ensure_runtime
        lizard.get_cheshire_cat = get_cheshire_cat
        ma._ensure_runtime = ready
        try:
            assert await real_load_agent("x") == "ccat" and got == ["x"]

            async def missing(agent_id):
                return None

            lizard.get_cheshire_cat = missing
            try:
                await real_load_agent("x")
                raise AssertionError("an agent without settings was loaded")
            except RuntimeError as exc:
                assert "cannot be loaded" in str(exc), exc
        finally:
            del lizard.get_cheshire_cat
            ma._ensure_runtime = saved

    async def test_graphrag_handler_detection():
        w = await new_world()
        assert ma._graphrag_handler(w.cat) is w.handler
        assert ma._graphrag_handler(types_namespace(vector_memory_handler=object())) is None

    # ── the edges of the command line ───────────────────────────────────────
    async def test_no_operation_and_no_runtime_abort():
        code, _ = await cli("--agent", "x")
        assert code == 2  # argparse: an operation is required
        import importlib.util as iu

        saved = iu.find_spec
        for missing in ("cat", "cat.db.database"):
            iu.find_spec = lambda name, *a, _m=missing, **k: None if name == _m else saved(name, *a, **k)
            try:
                code, lines = await cli("--all", "--reembed", "--yes")
            finally:
                iu.find_spec = saved
            assert code == 2 and lines_with(lines, "Cat runtime is not importable"), (missing, lines)

    async def test_graph_without_the_plugin_deployed_aborts():
        saved = ma._resolve_plugin_id
        ma._resolve_plugin_id = lambda: None
        try:
            code, lines = await cli("--all", "--graph", "--yes")
            assert code == 2 and lines_with(lines, "GraphRAG plugin not found"), lines
            code, lines = await cli("--all", "--graph", "--yes", "--json")
            assert code == 2 and not lines, lines  # the message goes to stderr: stdout is JSON only
        finally:
            ma._resolve_plugin_id = saved

    async def test_plan_skips_and_protected_agents():
        from cat.db import crud

        legacy_default = "default"
        plain = f"cli_{uuid.uuid4().hex[:10]}"
        # the plugin is active but the vector database is another backend
        await crud.store(crud_settings.format_key(plain), [{"name": "active_plugins", "value": ["cat_graphrag"]}])
        await gf.create_agent(legacy_default)
        ma._default_agent_warned = False
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                code, lines = await cli("--all", "--graph", "--dry-run")
            assert code == 2, lines
            assert lines_with(lines, plain, "SKIPPED", "handler-not-graphrag"), lines
            assert "legacy default agent" in err.getvalue(), err.getvalue()
            code, lines = await cli("--agent", plain, "--graph", "--yes", "--json")
            assert code == 0 and json.loads(lines[-1]) == {"result": "ok", "agents": []}, lines
            code, lines = await cli("--agent", plain, "--graph", "--json")  # nothing runnable: no --yes needed
            assert code == 0, lines
        finally:
            for agent_id in (plain, legacy_default):
                await gf.destroy_agent(agent_id)

    async def test_vector_database_read_error_is_not_graphrag():
        saved = crud_settings.get_settings_by_category

        async def broken(*a, **k):
            raise RuntimeError("redis down")

        crud_settings.get_settings_by_category = broken
        try:
            assert await ma._agent_uses_graphrag_handler("anyone") is False
        finally:
            crud_settings.get_settings_by_category = saved

    async def test_resolve_and_import_the_deployed_plugin():
        import tempfile
        import types as pytypes

        spec = importlib.util.spec_from_file_location(
            "maintenance_agent_plain", os.path.join(gf.PLUGIN_DIR, "scripts", "maintenance_agent.py"),
        )
        plain = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plain)  # the real _resolve_plugin_id / _plugin_module, not the test bindings
        import cat.utils as cat_utils

        saved_path, saved_env = cat_utils.get_plugins_path, os.environ.get("CAT_PLUGINS_DIR")
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "a_other"))
            with open(os.path.join(root, "a_other", "graphrag_handler.py"), "w") as fh:
                fh.write("# not the handler")
            os.makedirs(os.path.join(root, "b_graphrag"))
            with open(os.path.join(root, "b_graphrag", "graphrag_handler.py"), "w") as fh:
                fh.write("class GraphRAGHandler: ...")
            os.makedirs(os.path.join(root, "0_empty"))  # no handler file
            os.makedirs(os.path.join(root, "1_unreadable"))
            unreadable = os.path.join(root, "1_unreadable", "graphrag_handler.py")
            with open(unreadable, "w") as fh:
                fh.write("class GraphRAGHandler: ...")
            os.chmod(unreadable, 0)
            only_other = os.path.join(root, "only_other")
            os.makedirs(os.path.join(only_other, "x"))
            with open(os.path.join(only_other, "x", "graphrag_handler.py"), "w") as fh:
                fh.write("# not the handler")

            def broken():
                raise RuntimeError("no plugins path")

            cat_utils.get_plugins_path = broken
            os.environ["CAT_PLUGINS_DIR"] = root
            try:
                assert plain._resolve_plugin_id() == "b_graphrag"
                os.environ["CAT_PLUGINS_DIR"] = os.path.join(root, "missing")
                assert plain._resolve_plugin_id() is None
                os.environ["CAT_PLUGINS_DIR"] = only_other
                assert plain._resolve_plugin_id() is None
                os.environ.pop("CAT_PLUGINS_DIR")
                cat_utils.get_plugins_path = lambda: root
                assert plain._resolve_plugin_id() == "b_graphrag"
                cat_utils.get_plugins_path = broken
                assert plain._plugin_module("phases") is None
                os.environ["CAT_PLUGINS_DIR"] = root
                fake = pytypes.ModuleType("cat.plugins.b_graphrag.phases")
                sys.modules["cat.plugins.b_graphrag.phases"] = fake
                assert plain._plugin_module("phases") is fake
            finally:
                sys.modules.pop("cat.plugins.b_graphrag.phases", None)
                cat_utils.get_plugins_path = saved_path
                if saved_env is None:
                    os.environ.pop("CAT_PLUGINS_DIR", None)
                else:
                    os.environ["CAT_PLUGINS_DIR"] = saved_env

    async def test_marker_failures_are_warnings():
        w = await new_world()
        await w.upload("a.pdf", ["Alice met Bob"])
        saved = crud_ingestions.set_maintenance, crud_ingestions.clear_maintenance

        async def broken(*a, **k):
            raise RuntimeError("redis down")

        crud_ingestions.set_maintenance = crud_ingestions.clear_maintenance = broken
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                code, lines = await cli("--agent", w.agent, "--reembed", "--yes")
        finally:
            crud_ingestions.set_maintenance, crud_ingestions.clear_maintenance = saved
        assert code == 0, lines  # the op ran: the marker only coordinates with the resume
        assert "maintenance-marker-write-failed" in err.getvalue() and "maintenance-marker-delete-failed" in err.getvalue()

    async def test_op_failure_and_row_without_failed_phases():
        w = await new_world()
        await w.upload("a.pdf", ["Alice met Bob"])
        wipe = w.handler.wipe_tenant_graph

        async def broken():
            raise RuntimeError("neo4j down")

        w.handler.wipe_tenant_graph = broken
        try:
            code, lines = await cli("--agent", w.agent, "--graph", "--yes")
        finally:
            w.handler.wipe_tenant_graph = wipe
        assert code == 1 and lines_with(lines, "op=graph", "result=fail", "neo4j down"), lines
        report = ma._OpReport(w.agent, "reembed", json_output=True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            report.row("x.pdf", None, {"status": "error", "error": "Invalid phase graph", "completed_phases": {}})
            assert report.finish() is False
        assert json.loads(out.getvalue())["failures"] == [
            {"source": "x.pdf", "chat_id": None, "phases": {"status": "Invalid phase graph"}}
        ]

    async def test_json_skip_and_verify():
        w = await new_world()
        await w.upload("a.pdf", ["Alice met Bob"])
        original = ma._graphrag_handler
        ma._graphrag_handler = lambda ccat: None
        try:
            code, lines = await cli("--agent", w.agent, "--graph", "--reembed", "--yes", "--json", "--verify")
        finally:
            ma._graphrag_handler = original
        docs = [json.loads(line) for line in lines]
        assert code == 0, docs
        assert [d.get("op") for d in docs if "op" in d] == ["reembed"], docs  # the graph op is skipped
        assert {"agent": w.agent, "verify": {"rows-completed": True}} in docs, docs

    async def test_verify_error_is_a_failed_check():
        w = await new_world()
        saved = ma._verify_agent

        async def broken(*a, **k):
            raise RuntimeError("neo4j down")

        ma._verify_agent = broken
        try:
            code, lines = await cli("--agent", w.agent, "--reembed", "--yes", "--verify")
        finally:
            ma._verify_agent = saved
        assert code == 1 and lines_with(lines, "result=verify-fail", "neo4j down"), lines

    async def test_runtime_keeps_what_the_lizard_has():
        from cat.looking_glass.bill_the_lizard import BillTheLizard

        lizard = BillTheLizard()
        saved = lizard.plugin_manager.discover_plugins, lizard.rabbit_hole, lizard.websocket_manager
        rabbit_hole, websocket_manager = object(), object()

        async def discover():
            return None

        lizard.plugin_manager.discover_plugins = discover
        lizard.rabbit_hole, lizard.websocket_manager = rabbit_hole, websocket_manager
        ma._runtime_ready = False
        try:
            await ma._ensure_runtime()
            assert lizard.rabbit_hole is rabbit_hole and lizard.websocket_manager is websocket_manager
        finally:
            lizard.plugin_manager.discover_plugins, lizard.rabbit_hole, lizard.websocket_manager = saved
            ma._runtime_ready = False

    TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_")]

    async def sweep():
        for agent_id in await crud_settings.get_agents_main_keys():
            if agent_id.startswith("cli_"):
                await gf.destroy_agent(agent_id)

    async def run_all():
        failed = 0
        try:
            for test in TESTS:
                try:
                    await test()
                    print(f"PASS {test.__name__}")
                except Exception:  # noqa: BLE001 - report every failure
                    failed += 1
                    print(f"FAIL {test.__name__}")
                    traceback.print_exc()
        finally:
            await sweep()
            leftovers = [k async for k in get_async_db().scan_iter("agents:cli_*")]
            if leftovers:
                failed += 1
                print(f"FAIL leftovers in Redis: {leftovers[:5]}")
        print(f"\ntest_maintenance_cli: {len(TESTS) - failed}/{len(TESTS)} passed")
        return failed

    sys.exit(1 if asyncio.run(run_all()) else 0)
