"""Points written outside an ingestion get their graph.

The graph of a source is written by its graph phases, during its ingestion. Other paths of the core write points
without running them: the clone of an agent, the transfer to another vector database, the import of a memory dump,
``store_documents`` (plugins), a point added through the memory API, the procedures. ``add_points_to_tenant`` tells
the writes apart by the source's state row:

- a write of the source's ingestion (row ``processing`` on the embedding phase): the extraction tasks the spaCy
  phase joins, as before;
- a source with a state row (e.g. a cloned agent: rows copied ``completed``): its graph phase entries are reset, so
  the core's automatic resume rebuilds the graph (locked, crash-safe, multi-pod); no task is started;
- a source without a row (e.g. a memory dump): a background job of the plugin, on an agent of its own (the request's
  handler is closed when the request ends), runs the graph phases' work for that source;
- the procedures: no extraction (they are not knowledge).

Standalone script (plugin rules: the body runs only as ``__main__``). Needs the core checkout and a Redis Stack on
localhost:6379 (database 14):

    cd ../grinning-cat-core
    .venv/bin/python /path/to/cat-graphrag/tests/test_outside_writes.py
"""

if __name__ == "__main__":
    import asyncio
    import importlib
    import os
    import sys
    import traceback
    import types
    import uuid

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import graph_fakes as gf

    main_mod, handler_mod, phases_mod = gf.bootstrap("_graphrag_outside")
    jobs_mod = importlib.import_module("_graphrag_outside.background_jobs")

    from cat.db.cruds import ingestions as crud_ingestions
    from cat.rabbit_hole import recovery
    from cat.services.memory.models import PointStruct

    GRAPH_PHASES = {phases_mod.GRAPH_SPACY_PHASE, phases_mod.GRAPH_LLM_PHASE}
    AGENTS = {}  # agent -> the world whose agent a job loads

    async def load_agent(tenant):
        world = AGENTS.get(tenant)
        return None if world is None else world.load_agent()

    jobs_mod.load_agent = load_agent

    async def settle():
        jobs = list(jobs_mod.JOBS)
        if jobs:
            await asyncio.wait(jobs, timeout=10)

    def points(name, texts, chat=None):
        meta = {"source": name} | ({"chat_id": chat} if chat else {})
        return [
            PointStruct(id=uuid.uuid4().hex, payload={"page_content": t, "metadata": meta | {"chunk_index": i}},
                        vector=[1.0, float(i + 1)])
            for i, t in enumerate(texts)
        ]

    class World(gf.CliWorld):
        """An agent: its graph (the fake store), its files, the core RabbitHole on it."""

        def load_agent(self):
            """The agent a job loads: a new handler on the same graph (the request's handler may be closed)."""
            handler = self.store.install(gf.make_handler(handler_mod, self.agent, enable_concept_relations=True))
            return gf.FakeAgentCat(self.agent, handler, main_mod, self.rh, self.kept)

    async def new_world(agent=None, store=None):
        agent = agent or f"outside_{uuid.uuid4().hex[:8]}"
        await gf.create_agent(agent, graphrag=True)
        world = World(agent, main_mod, handler_mod, PointStruct)
        if store is not None:
            world.store = store
            world.handler = store.install(gf.make_handler(handler_mod, agent, enable_concept_relations=True))
            world.cat.vector_memory_handler = world.handler
        AGENTS[agent] = world
        return world

    def tasks_of(handler):
        return {key: list(tasks) for key, tasks in handler._pending_source_tasks.items() if tasks}

    async def test_a_write_of_the_ingestion_keeps_its_tasks():
        """The core's embedding phase writes while the row is processing on the embedding: the spaCy phase joins
        the extraction tasks started by the write (unchanged)."""
        w = await new_world()
        await crud_ingestions.create_status(w.agent, None, "a.pdf", type_="file")
        await crud_ingestions.update_status(w.agent, None, "a.pdf", status="processing", phase="embedding")
        await w.handler.add_points_to_tenant(gf.DECLARATIVE, points("a.pdf", ["Alice met Bob"]))
        assert ("a.pdf", None) in tasks_of(w.handler), "the ingestion lost its extraction tasks"
        assert not jobs_mod.JOBS
        await gf.destroy_agent(w.agent)

    async def test_a_cloned_source_is_rebuilt():
        """The clone copies the rows (completed, graph phases done) and then the points: the graph phase entries are
        reset at once (state first), then the worker reruns them through the core. If the pod dies before the worker
        runs, the core's automatic resume finds the phases due and rebuilds the graph."""
        for pod_dies in (False, True):
            source = await new_world()
            await source.upload("a.pdf", ["Alice met Bob"])
            clone = await new_world()
            clone.files.update(source.files)
            clone.kept.update(source.kept)
            await crud_ingestions.clone_knowledge_base(source.agent, clone.agent)
            copied = [PointStruct(id=uuid.uuid4().hex, payload=p.payload, vector=p.vector)
                      for p in (await source.handler.get_all_tenant_points(gf.DECLARATIVE))[0]]
            await clone.handler.add_points_to_tenant(gf.DECLARATIVE, copied)
            if pod_dies:  # the pod dies right after the write: the worker never runs (nothing awaited in between)
                for job in list(jobs_mod.JOBS):
                    job.cancel()
            assert not tasks_of(clone.handler), "an outside write left extraction work on the request's handler"
            diary = (await clone.row("a.pdf"))["completed_phases"]
            assert not GRAPH_PHASES & set(diary), f"the cloned row still claims a graph: {sorted(diary)}"
            if pod_dies:
                await asyncio.gather(*list(jobs_mod.JOBS), return_exceptions=True)
                assert not clone.store.mentions

                async def get_cat(agent_id):
                    return clone.cat

                await recovery.resume_agent(
                    types.SimpleNamespace(rabbit_hole=clone.rh, get_cheshire_cat=get_cat), clone.agent,
                )
            else:
                await settle()
            row = await clone.row("a.pdf")
            assert row["status"] == "completed" and clone.graph_ok("a.pdf"), (pod_dies, row["status"], sorted(row["completed_phases"]))
            for w in (source, clone):
                await gf.destroy_agent(w.agent)

    async def test_a_source_without_a_row_gets_its_graph_from_a_job():
        """A memory dump (no row): a job on an agent of its own writes the graph of the source, even though the
        request's handler is closed at the end of the request."""
        w = await new_world()
        w.files[(None, "dump.json")] = ["Carol met Dave", "Erin"]
        await w.handler.add_points_to_tenant(gf.DECLARATIVE, points("dump.json", ["Carol met Dave", "Erin"]))
        assert not tasks_of(w.handler), "the request's handler holds the work"
        await w.handler.close()  # the request ends
        await settle()
        assert w.graph_ok("dump.json"), w.store.mentions
        assert await w.row("dump.json") is None, "the job invented a state row"
        await gf.destroy_agent(w.agent)

    async def test_a_point_added_through_the_memory_api_gets_its_graph():
        w = await new_world()
        w.files[(None, "user-1")] = ["Frank met Grace"]
        await w.handler.add_point_to_tenant(
            gf.DECLARATIVE, "Frank met Grace", [1.0, 2.0], {"source": "user-1", "chunk_index": 0}, uuid.uuid4().hex,
        )
        await settle()
        assert w.graph_ok("user-1"), w.store.mentions
        await gf.destroy_agent(w.agent)

    async def test_a_transfer_rebuilds_every_source_once():
        """A transfer to GraphRAG: the points of the old store are written into the new one outside any ingestion.
        Every source with a row (completed, or in error after a failure during the transfer window) is rebuilt by the
        worker through the core; the transfer hook afterwards finds nothing left to do."""
        old = await new_world()
        await old.upload("a.pdf", ["Alice met Bob"])
        old.store.fail_writes = 1
        await old.upload("b.pdf", ["Erin met Frank"])  # its graph phase failed
        old.store.fail_writes = 0
        assert (await old.row("b.pdf"))["status"] == "error"
        new_store = gf.FakeGraphStore()
        target = World(old.agent, main_mod, handler_mod, PointStruct)  # the agent, now on the new store
        target.store = new_store
        target.files, target.kept = old.files, old.kept
        target.rh = gf.core_rabbit_hole(target.files, PointStruct)
        target.handler = new_store.install(gf.make_handler(handler_mod, old.agent, enable_concept_relations=True))
        target.cat = gf.FakeAgentCat(old.agent, target.handler, main_mod, target.rh, target.kept)
        AGENTS[old.agent] = target
        transferred = (await old.handler.get_all_tenant_points(gf.DECLARATIVE))[0]
        await target.handler.add_points_to_tenant(gf.DECLARATIVE, [
            PointStruct(id=p.id, payload=p.payload, vector=p.vector) for p in transferred
        ])
        await settle()
        for name in ("a.pdf", "b.pdf"):
            row = await target.row(name)
            assert row["status"] == "completed" and target.graph_ok(name), (name, row["status"], new_store.mentions)
        writes = len(new_store.writes)
        hook = getattr(main_mod.after_vector_memory_transfer_on_agent, "function",
                       main_mod.after_vector_memory_transfer_on_agent)
        await hook(True, target.cat)
        assert len(new_store.writes) == writes, "the transfer rebuilt a source twice"
        await gf.destroy_agent(old.agent)

    async def test_a_burst_of_writes_to_a_source_without_a_row_is_built_once_and_whole():
        """The memory API in a loop (its source is the user by default): the writes queue the source once while the
        worker runs, and its graph ends with every point."""
        w = await new_world()
        builds = []
        real = handler_mod.GraphRAGHandler.run_spacy_extraction_phase

        async def counting(self, source, collection_name, chat_id=None):
            builds.append(source)
            await asyncio.sleep(0.05)  # a slow extraction: the next writes arrive meanwhile
            return await real(self, source, collection_name, chat_id)

        handler_mod.GraphRAGHandler.run_spacy_extraction_phase = counting
        try:
            texts = ["Alice met Bob", "Carol met Dave", "Erin met Frank", "Grace met Alice"]
            w.files[(None, "user-1")] = texts
            for i, text in enumerate(texts):
                await w.handler.add_point_to_tenant(
                    gf.DECLARATIVE, text, [1.0, float(i + 1)], {"source": "user-1", "chunk_index": i}, uuid.uuid4().hex,
                )
            await settle()
        finally:
            handler_mod.GraphRAGHandler.run_spacy_extraction_phase = real
        assert w.graph_ok("user-1"), w.store.mentions
        assert len(builds) <= 2, f"the source was rebuilt at every write: {builds}"
        await gf.destroy_agent(w.agent)

    async def test_procedures_start_nothing():
        w = await new_world()
        await w.handler.add_points_to_tenant(gf.PROCEDURAL, points("tool-x", ["Alice uses the Calculator"]))
        await settle()
        assert not tasks_of(w.handler) and not w.store.mentions and not jobs_mod.JOBS
        await gf.destroy_agent(w.agent)

    async def test_a_job_whose_agent_is_gone_does_nothing():
        w = await new_world()
        AGENTS.pop(w.agent)  # the agent is deleted before the job loads it
        await w.handler.add_points_to_tenant(gf.DECLARATIVE, points("dump.json", ["Carol met Dave"]))
        await settle()
        assert not w.store.mentions
        await gf.destroy_agent(w.agent)

    TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_")]

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
            await settle()
        print(f"\ntest_outside_writes: {len(TESTS) - failed}/{len(TESTS)} passed")
        return failed

    sys.exit(1 if asyncio.run(run_all()) else 0)
