"""The two graph phases on the REAL ingestion engine of the core (``cat.rabbit_hole``).

The plugin registers ``graph_spacy_extraction`` (after ``embedding``) and ``graph_llm_extraction`` (after the spaCy
phase) through the ``rabbithole_ingestion_phases`` hook. These checks run the core's own engine (``run_phases``,
``delete_source``, the lock of the source, the diary in Redis) with the plugin's phases, its real handler
orchestration and an in-memory graph at the Neo4j boundary (``graph_fakes.FakeGraphStore``):

- registration only where the vector memory is GraphRAG, in dependency order;
- tags move only with the settings that no dedicated job owns (spaCy models, the two toggles);
- a completed source holds the graph of its CURRENT content only (re-upload, settings change, restart);
- points read and purge are scoped to the collection/chat of the source;
- outcome mapping (done / error / skipped / not_ready) and the dependency of the LLM phase on the spaCy one;
- no extraction task outlives the phase or the deletion of its source;
- the system embedder is refreshed before the extraction tasks start;
- every EffING leftover is gone.

Standalone script (plugin rules: the body runs only as ``__main__``). Needs the core checkout and a Redis Stack on
localhost:6379 (database 14 is used and cleaned). Run from the core checkout, with the plugin's dependencies
importable (see tests/README.md):

    cd ../grinning-cat-core
    .venv/bin/python /path/to/cat-graphrag/tests/test_graph_phases.py
"""

if __name__ == "__main__":
    import asyncio
    import json
    import os
    import sys
    import traceback
    import uuid

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import graph_fakes as gf

    main, handler_mod, phases_mod = gf.bootstrap("_graphrag_phases_test")

    from cat.db.cruds import ingestions as crud_ingestions
    from cat.rabbit_hole import engine
    from cat.rabbit_hole.phases import IngestionSource, compute_tags
    from cat.services.memory.models import PointStruct

    SPACY = "graph_spacy_extraction"
    LLM = "graph_llm_extraction"

    class Env:
        """One agent with its store, handler, cat and rabbit hole; cleaned on exit."""

        def __init__(self, **handler_overrides):
            self.agent = f"graphtest_{uuid.uuid4().hex[:10]}"
            self.store = gf.FakeGraphStore()
            self.embedder = object()
            self.handler = self.store.install(
                gf.make_handler(handler_mod, self.agent, **handler_overrides), system_embedder=self.embedder,
            )
            self.cat = gf.FakeCat(self.agent, self.handler, main)
            self.files = {}
            self.rh = gf.FakeRabbitHole(self.files, PointStruct)

        async def __aenter__(self):
            await gf.create_agent(self.agent)
            return self

        async def __aexit__(self, *exc):
            await gf.destroy_agent(self.agent)
            return False

        async def upload(self, name, chunks, chat=None):
            self.files[(chat, name)] = list(chunks)
            return await gf.ingest(engine, self.rh, self.cat, chat, name)

        def doc_ids(self, name, chat=None):
            return self.store.scoped_doc_ids(name, gf.FakeRabbitHole.collection(chat), chat)

        def assert_current_graph(self, name, chat=None, concepts=True):
            ids = self.doc_ids(name, chat)
            expected = self.store.expected_for(ids, concepts)
            actual = self.store.actual_for(ids, name, chat)
            assert actual == expected, f"graph of {name!r} ({chat}) is not the one of its content:\n{actual}\n!=\n{expected}"
            assert not self.handler._pending_source_tasks.get((name, chat)), f"tasks left for {name!r}"

    def diary(row):
        return {pid: entry["state"] for pid, entry in (row.get("completed_phases") or {}).items()}

    # ── registration ────────────────────────────────────────────────────────
    async def test_phases_registered_only_for_a_graphrag_agent():
        async with Env() as env:
            ordered = await engine.collect_phases(env.rh, env.cat)
            assert [p.id for p in ordered] == ["parsing_chunking", "embedding", SPACY, LLM], [p.id for p in ordered]
            by_id = {p.id: p for p in ordered}
            assert by_id[SPACY].depends_on == ("embedding",)
            assert by_id[LLM].depends_on == (SPACY,)
            assert by_id[LLM].not_ready_budget_seconds == 300
            env.cat.vector_memory_handler = object()
            ordered = await engine.collect_phases(env.rh, env.cat)
            assert [p.id for p in ordered] == ["parsing_chunking", "embedding"], [p.id for p in ordered]

    async def test_tags_move_only_with_unowned_settings():
        async with Env(extra_technology_patterns=["Kafka"]) as env:
            ordered = await engine.collect_phases(env.rh, env.cat)
            base = await compute_tags(ordered, env.cat)
            h = env.handler

            # owned by dedicated jobs: the technology refresh and the concept-generation flip
            h._extra_technology_patterns = ["Kafka", "Flink"]
            h._concept_definitions = {"NEW": "concept"}
            h._relation_definitions = {"NEW": "relation"}
            h._concept_relations_prompt = "another prompt"
            h._llm_identity = "OtherLLM:model"
            assert await compute_tags(ordered, env.cat) == base, "a setting owned by a dedicated job moved a tag"

            models = h._spacy_models
            h._spacy_models = {"en": "en_core_web_lg"}
            moved = await compute_tags(ordered, env.cat)
            assert moved[SPACY] != base[SPACY] and moved[LLM] != base[LLM], "spaCy models must move spaCy and LLM"
            assert moved["embedding"] == base["embedding"]

            h._spacy_models = models
            h._enable_entity_extraction = False
            assert (await compute_tags(ordered, env.cat))[SPACY] != base[SPACY], "entity toggle must move spaCy"

            h._enable_entity_extraction = True
            h._enable_concept_relations = False
            moved = await compute_tags(ordered, env.cat)
            # the spaCy phase owns the purge of the concepts: turning them off must re-run it
            assert moved[LLM] != base[LLM] and moved[SPACY] != base[SPACY], "concept toggle must move both phases"

    # ── lifecycle on the engine ──────────────────────────────────────────────
    async def test_first_ingestion_builds_the_graph_once():
        async with Env() as env:
            row = await env.upload("a.pdf", ["Alice met Bob", "Carol saw Dave"])
            assert row["status"] == "completed", row
            assert diary(row) == {"parsing_chunking": "complete", "embedding": "complete", SPACY: "complete", LLM: "complete"}
            assert len([w for w in env.store.writes if "a.pdf" in w[0]]) == 1, env.store.writes
            assert env.store.similar == env.doc_ids("a.pdf"), "similarity not rebuilt for every point"
            env.assert_current_graph("a.pdf")

    async def test_reupload_replaces_the_graph_of_the_source():
        async with Env() as env:
            await env.upload("a.pdf", ["Alice met Bob"])
            await env.upload("b.pdf", ["Alice met Bob"])
            row = await env.upload("a.pdf", ["Erin met Frank"])
            assert row["status"] == "completed", row
            env.assert_current_graph("a.pdf")
            env.assert_current_graph("b.pdf")
            assert ("Alice", "Bob", "NEXT_TO") not in env.store.related_of("a.pdf"), "stale relation kept"

    async def test_spacy_settings_change_reruns_both_phases_and_replaces():
        async with Env() as env:
            await env.upload("a.pdf", ["Alice met Bob"])
            before = env.store.purges[:]
            # a relation the old models produced and the current content does not
            env.store.related.setdefault(("Ghost", "Alice", "NEXT_TO"), set()).add("a.pdf")
            env.handler._spacy_models = {"en": "en_core_web_lg"}
            row = await gf.rerun(engine, env.rh, env.cat, None, "a.pdf")
            assert row["status"] == "completed", row
            assert len(env.store.purges) == len(before) + 1, "the rerun must purge before re-extracting"
            assert diary(row)[LLM] == "complete"
            env.assert_current_graph("a.pdf")

    async def test_restart_mid_phase_resumes_and_replaces():
        async with Env() as env:
            env.files[(None, "a.pdf")] = ["Alice met Bob"]
            original = env.handler._write_source_graph

            async def crash(*args, **kwargs):
                raise asyncio.CancelledError()

            env.handler._write_source_graph = crash
            try:
                await gf.ingest(engine, env.rh, env.cat, None, "a.pdf")
                raise AssertionError("the crash did not stop the worker")
            except asyncio.CancelledError:
                pass
            row = await crud_ingestions.get_status(env.agent, None, "a.pdf")
            assert row["status"] == "processing" and row["phase"] == SPACY, row
            # the process restarts: a new handler on the same graph, no task in memory
            fresh = env.store.install(gf.make_handler(handler_mod, env.agent), system_embedder=env.embedder)
            fresh._write_source_graph = original
            env.handler = env.cat.vector_memory_handler = fresh
            row = await gf.rerun(engine, env.rh, env.cat, None, "a.pdf")
            assert row["status"] == "completed", row
            env.assert_current_graph("a.pdf")

    async def test_embedder_change_rebuilds_the_graph_on_the_new_points():
        async with Env() as env:
            await env.upload("a.pdf", ["Alice met Bob", "Carol saw Dave"])
            row = await gf.rerun(engine, env.rh, env.cat, None, "a.pdf", invalidate=["embedding"])
            assert row["status"] == "completed", row
            env.assert_current_graph("a.pdf")

    # ── scoping ──────────────────────────────────────────────────────────────
    async def test_same_name_in_kb_and_chat_are_independent_documents():
        async with Env() as env:
            await env.upload("x.pdf", ["Alice met Bob"])
            await env.upload("x.pdf", ["Carol saw Dave"], chat="chat-1")
            kb_ids, chat_ids = env.doc_ids("x.pdf"), env.doc_ids("x.pdf", "chat-1")
            assert kb_ids and chat_ids and not kb_ids & chat_ids
            kb_mentions = {(d, e) for d, e in env.store.mentions if d in kb_ids}
            assert kb_mentions == {(d, n) for d in kb_ids for n in ("Alice", "Bob")}, "KB graph built from chat docs"
            await gf.delete(engine, env.rh, env.cat, "chat-1", "x.pdf")
            assert not env.doc_ids("x.pdf", "chat-1")
            assert {(d, e) for d, e in env.store.mentions if d in kb_ids} == kb_mentions, "chat delete touched KB docs"
            assert all(p[1] == gf.EPISODIC and p[2] == "chat-1" for p in env.store.purges[-2:]), env.store.purges

    async def test_points_are_read_page_by_page():
        async with Env() as env:
            for i in range(2300):
                env.store.docs[f"d{i:05d}"] = {
                    "collection": gf.DECLARATIVE, "content": f"Item{i} here", "metadata": {"source": "big.pdf"},
                    "vector": [1.0, 1.0],
                }
            points = await env.handler.get_source_points(gf.DECLARATIVE, "big.pdf", None)
            assert len(points) == 2300, len(points)

    # ── outcomes ─────────────────────────────────────────────────────────────
    async def test_spacy_failure_fails_the_llm_phase_by_dependency():
        async with Env() as env:
            env.store.fail_writes = 1
            row = await env.upload("a.pdf", ["Alice met Bob"])
            assert row["status"] == "error", row
            assert diary(row)[SPACY] == "error" and diary(row)[LLM] == "error", diary(row)
            assert "graph_spacy_extraction" in row["completed_phases"][LLM]["reason"], row["completed_phases"][LLM]
            assert not env.handler._pending_source_tasks.get(("a.pdf", None)), "tasks left after a failed phase"

    async def test_llm_transient_is_retried_then_errors_are_recorded():
        async with Env() as env:
            env.store.concept_outcomes = ["not_ready", "done"]
            row = await env.upload("a.pdf", ["Alice met Bob"])
            assert row["status"] == "completed" and diary(row)[LLM] == "complete", row
            env.store.concept_outcomes = ["error"]
            row = await env.upload("b.pdf", ["Carol saw Dave"])
            assert row["status"] == "error" and diary(row)[LLM] == "error" and diary(row)[SPACY] == "complete", row

    async def test_turning_concepts_off_removes_them():
        async with Env() as env:
            await env.upload("a.pdf", ["Alice met Bob"])
            env.handler._enable_concept_relations = False
            row = await gf.rerun(engine, env.rh, env.cat, None, "a.pdf")
            assert row["status"] == "completed" and diary(row)[LLM] == "skipped", row
            env.assert_current_graph("a.pdf", concepts=False)

    async def test_llm_disabled_is_skipped_and_the_source_completes():
        async with Env(enable_concept_relations=False) as env:
            row = await env.upload("a.pdf", ["Alice met Bob"])
            assert row["status"] == "completed", row
            assert diary(row)[LLM] == "skipped", diary(row)
            env.assert_current_graph("a.pdf", concepts=False)

    async def test_missing_points_are_an_error_not_a_wait():
        async with Env() as env:
            source = IngestionSource(env.agent, None, "ghost.pdf")
            for phase_class in (phases_mod.GraphSpacyPhase, phases_mod.GraphConceptPhase):
                result = await phase_class().process(source, env.cat)
                assert result.status == "error" and "ghost.pdf" in (result.reason or ""), result

    async def test_inactive_backend_is_skipped():
        async with Env() as env:
            env.cat.vector_memory_handler = object()
            source = IngestionSource(env.agent, None, "a.pdf")
            for phase_class in (phases_mod.GraphSpacyPhase, phases_mod.GraphConceptPhase):
                phase = phase_class()
                applies, reason = await phase.applies_to(source, env.cat)
                assert applies is False and reason, (applies, reason)
                assert (await phase.process(source, env.cat)).status == "skipped"
                assert await phase.settings(env.cat) is None
                await phase.delete(source, env.cat)  # nothing to remove, no error

    async def test_delete_is_idempotent_and_drops_pending_tasks():
        async with Env() as env:
            await env.upload("a.pdf", ["Alice met Bob"])
            pending = asyncio.get_running_loop().create_future()
            env.handler._track_source_task({"source": "a.pdf"}, pending)
            assert env.handler._pending_source_tasks.get(("a.pdf", None))
            source = IngestionSource(env.agent, None, "a.pdf")
            for phase_class in (phases_mod.GraphConceptPhase, phases_mod.GraphSpacyPhase):
                await phase_class().delete(source, env.cat)
                await phase_class().delete(source, env.cat)
            assert ("a.pdf", None) not in env.handler._pending_source_tasks
            assert not {(d, e) for d, e in env.store.mentions if d in env.doc_ids("a.pdf")}
            pending.cancel()

    # ── embedder ─────────────────────────────────────────────────────────────
    async def test_system_embedder_refreshed_before_extraction():
        async with Env(enable_entity_embeddings=True) as env:
            env.handler.embedder = object()  # injected before the embedder changed
            env.files[(None, "a.pdf")] = ["Alice met Bob"]
            await gf.ingest(engine, env.rh, env.cat, None, "a.pdf")
            assert env.handler.embedder is env.embedder, "extraction ran with a stale embedder"

    async def test_rerun_without_embedding_uses_the_system_embedder():
        """A spaCy re-run without a new embedding (settings change, resume) on a fresh handler — the resume builds a
        new agent, whose handler has no injected embedder — must still embed the entities with the system one."""
        async with Env(enable_entity_embeddings=True) as env:
            await env.upload("a.pdf", ["Alice met Bob"])
            fresh = env.store.install(
                gf.make_handler(handler_mod, env.agent, enable_entity_embeddings=True, spacy_models={"en": "x"}),
                system_embedder=env.embedder,
            )
            env.handler = env.cat.vector_memory_handler = fresh
            env.store.embedders_used.clear()
            row = await gf.rerun(engine, env.rh, env.cat, None, "a.pdf")
            assert row["status"] == "completed", row
            assert env.store.embedders_used and all(e is env.embedder for e in env.store.embedders_used), (
                env.store.embedders_used
            )

    async def test_transfer_to_graphrag_makes_every_graph_due():
        """Once the points are transferred to GraphRAG, the hook makes the graph phases of every source due (knowledge
        base and chats) under the lock of the source, and runs nothing: the automatic resume builds them after the
        transfer (test_outside_writes.py). A failed transfer touches nothing (the core embeds everything again)."""
        async with Env() as env:
            await env.upload("a.pdf", ["Alice met Bob"])
            await env.upload("c.pdf", ["Carol saw Dave"], chat="chat-1")
            calls = []

            class Hole:
                """The core's invalidate_source (run for real in test_outside_writes.py): the phases and their
                dependents (the LLM phase depends on the spaCy one) left to run again."""

                async def invalidate_source(self, cat, name, chat_id=None, phase_ids=()):
                    calls.append((name, chat_id, tuple(phase_ids)))
                    return await crud_ingestions.reset_phase_entries(
                        cat.agent_key, chat_id, name, [*phase_ids, phases_mod.GRAPH_LLM_PHASE],
                    )

                async def rerun_source(self, *args, **kwargs):
                    calls.append(("rerun", args, kwargs))

            env.cat.rabbit_hole = Hole()
            writes = len(env.store.writes)
            hook = getattr(main.after_vector_memory_transfer_on_agent, "function", main.after_vector_memory_transfer_on_agent)
            await hook(True, env.cat)
            spacy = phases_mod.GRAPH_SPACY_PHASE
            assert sorted(calls) == sorted([("a.pdf", None, (spacy,)), ("c.pdf", "chat-1", (spacy,))]), calls
            assert len(env.store.writes) == writes, "the hook ran a graph phase within the transfer"
            for name, chat in (("a.pdf", None), ("c.pdf", "chat-1")):
                diary = (await crud_ingestions.get_status(env.cat.agent_key, chat, name))["completed_phases"]
                assert not {spacy, phases_mod.GRAPH_LLM_PHASE} & set(diary), (name, sorted(diary))

            calls.clear()
            await hook(False, env.cat)  # a failed transfer touches nothing
            assert calls == []

    async def test_no_system_embedder_keeps_the_injected_one():
        async with Env(enable_entity_embeddings=True) as env:
            injected = object()
            env.handler.embedder = injected
            env.store.install(env.handler, system_embedder=None)
            env.files[(None, "a.pdf")] = ["Alice met Bob"]
            await gf.ingest(engine, env.rh, env.cat, None, "a.pdf")
            assert env.handler.embedder is injected

    # ── edges of the touched code ────────────────────────────────────────────
    async def test_system_embedder_resolution():
        from unittest.mock import patch

        handler = gf.make_handler(handler_mod, "agent_x")
        system = object()

        class Lizard:
            async def embedder(self):
                return system

        class BrokenLizard:
            async def embedder(self):
                raise RuntimeError("no lizard")

        with patch("cat.looking_glass.bill_the_lizard.BillTheLizard", Lizard):
            assert await handler._system_embedder() is system
            await handler._refresh_system_embedder()
            assert handler.embedder is system
        injected = object()
        handler.embedder = injected
        with patch("cat.looking_glass.bill_the_lizard.BillTheLizard", BrokenLizard):
            assert await handler._system_embedder() is None
            await handler._refresh_system_embedder()
            assert handler.embedder is injected, "a failed resolution must keep the injected embedder"

    async def test_points_without_source_or_vector():
        async with Env() as env:
            # a payload-only point (zero vector) of an ingestion still starts the extraction, without a similarity task
            from cat.db.cruds import ingestions as crud_ingestions

            await crud_ingestions.create_status(env.handler.agent_id, None, "z.pdf", type_="file")
            await crud_ingestions.update_status(env.handler.agent_id, None, "z.pdf", status="processing", phase="embedding")
            await env.handler.add_points_to_tenant(gf.DECLARATIVE, [
                PointStruct(id="z0", payload={"page_content": "Alice met Bob", "metadata": {"source": "z.pdf"}},
                            vector=[0.0, 0.0]),
                # a point without a source is stored: no graph is built for it (no extraction), only its similarity
                PointStruct(id="n0", payload={"page_content": "Carol", "metadata": {}}, vector=[1.0, 1.0]),
            ])
            tasks = env.handler._pending_source_tasks.get(("z.pdf", None)) or []
            assert len(tasks) == 1, tasks  # the extraction only: no similarity for a zero vector
            assert set(env.handler._pending_source_tasks) == {("z.pdf", None)}, env.handler._pending_source_tasks
            env.handler._embedder = None
            await env.handler.add_points_to_tenant(gf.DECLARATIVE, [])  # nothing to store: no embedder resolved
            assert env.handler._embedder is None
            env.handler._drop_source_tasks(None)  # no source: nothing to drop
            assert ("z.pdf", None) in env.handler._pending_source_tasks
            await asyncio.gather(*env.handler._pending_entity_tasks, return_exceptions=True)

    async def test_document_structure_edges():
        from unittest.mock import AsyncMock

        handler = gf.make_handler(handler_mod, "agent_x")
        handler._ensure_connected = AsyncMock()
        handler._get_session = lambda: (_ for _ in ()).throw(AssertionError("no write expected"))
        await handler.create_derived_graph_for_source("e.pdf", [])  # nothing stored: nothing to build
        card = PointStruct(id="c", payload={"page_content": "x", "metadata": {"is_catalogue_card": True}}, vector=[])
        await handler.create_derived_graph_for_source("e.pdf", [card])  # only the card: no structure

    async def test_invalid_phase_outcome_is_an_error():
        for raw in (None, {"status": "weird"}, "done"):
            result = phases_mod.to_result("graph_spacy_extraction", raw)
            assert result.status == "error" and "graph_spacy_extraction" in result.reason, (raw, result)
        result = phases_mod.to_result("x", {"status": "not_ready", "retry_after": 2, "processed": True})
        assert result.retry_after == 2.0 and result.processed is None, result

    async def test_stored_metadata_read_back():
        """The metadata of a Document read back are the caller's: parsed, never the handler's tenant_id."""
        read = handler_mod._stored_metadata
        assert read(json.dumps({"source": "a.pdf", "tenant_id": "t"})) == {"source": "a.pdf"}
        assert read({"source": "a.pdf", "tenant_id": "t"}) == {"source": "a.pdf"}
        assert read("not json") == {} and read(None) == {} and read(json.dumps([1])) == {}

    async def test_the_stub_of_the_core_enum_matches_the_core():
        """The harnesses that stub the core share one copy of ``VectorMemoryType``: it must be the core's."""
        from cat.services.memory.models import VectorMemoryType

        stub = {k: v for k, v in vars(gf.StubVectorMemoryType).items() if k.isupper()}
        assert stub == {m.name: m.value for m in VectorMemoryType}, stub

    # ── EffING leftovers ─────────────────────────────────────────────────────
    async def test_no_effing_leftovers():
        for name in (
            "ingestion_phase_specs", "ingestion_phase_objects", "ingestion_phase_settings_marker",
            "ingestion_phase_run", "ingestion_phase_purge", "before_ingestion_status_completed",
            "rabbithole_ingestion_processing", "after_rabbithole_stored_documents", "_phase_path_active",
            "graph_phase_specs",
        ):
            assert not hasattr(main, name), f"main.{name} is still defined"
        for name in ("_is_empty_mapping", "_is_empty_vector_staging_pass", "get_source_points_by_collection"):
            assert not hasattr(handler_mod.GraphRAGHandler, name), f"GraphRAGHandler.{name} is still defined"
        with open(os.path.join(gf.PLUGIN_DIR, "plugin.json"), encoding="utf-8") as fh:
            assert "cat_efficient_ingestion" not in json.load(fh).get("dependencies", [])
        this_file = os.path.abspath(__file__)
        for root, _, files in os.walk(gf.PLUGIN_DIR):
            if any(part in root for part in (".git", ".venv", "__pycache__")):
                continue
            for file in files:
                path = os.path.join(root, file)
                if file.endswith((".py", ".json", ".md")) and os.path.abspath(path) != this_file:
                    with open(path, encoding="utf-8") as fh:
                        text = fh.read()
                    for word in ("cat_efficient_ingestion", "EffING", "EFFING_DIR", "reembed_sources"):
                        assert word not in text, f"{word!r} still referenced in {os.path.join(root, file)}"

    TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_") and asyncio.iscoroutinefunction(v)]

    async def run_all():
        failed = 0
        for test in TESTS:
            try:
                await test()
                print(f"PASS {test.__name__}")
            except Exception:  # noqa: BLE001 - report every failure
                failed += 1
                print(f"FAIL {test.__name__}")
                traceback.print_exc()
        print(f"\ntest_graph_phases: {len(TESTS) - failed}/{len(TESTS)} passed")
        return failed

    sys.exit(1 if asyncio.run(run_all()) else 0)
