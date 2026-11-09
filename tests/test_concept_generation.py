"""The visible concepts follow the current concept settings AND the agent's LLM.

Concepts are tagged with a generation (``_concept_fingerprint``: definitions, prompt, LLM identity, toggle); recall
shows the active one. A save of the plugin settings recomputes them, but no hook fires when the agent's LLM changes:
the plugin finds out itself, before a recall or an ingestion, at most once per agent and fingerprint in a process.

- an LLM change: visible concepts of another generation exist -> ONE recompute to the new fingerprint, flipped;
- nothing changed: no further check, no recompute;
- concepts of the current fingerprint and no marker (an agent never recomputed): no recompute;
- concepts disabled, or another vector database: nothing;
- the recompute runs on an agent of its own: the end of the request that noticed the change (its handler closed)
  does not stop it;
- two pods noticing the same change: one recompute (a lock per agent and generation); the other does not check
  again at every message, only after an interval (e.g. the holder died and its lock expired);
- a recompute that fails, is cancelled or finds its agent gone: the next recall checks again;
- the hooks call it and a failure never breaks a recall or an ingestion.

Standalone script (plugin rules: the body runs only as ``__main__``). Needs the core checkout and a Redis Stack on
localhost:6379 (database 14):

    cd ../grinning-cat-core
    .venv/bin/python /path/to/cat-graphrag/tests/test_concept_generation.py
"""

if __name__ == "__main__":
    import asyncio
    import os
    import sys
    import traceback
    import types
    import uuid

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import graph_fakes as gf

    main_mod, handler_mod, _ = gf.bootstrap("_graphrag_concepts")
    import importlib

    jobs_mod = importlib.import_module("_graphrag_concepts.background_jobs")

    from cat.db.database import get_async_db

    class Pod:
        """One process of the Cat on an agent: the agents it loads share the agent's graph (the fake store)."""

        def __init__(self, agent, store, concepts=True):
            self.agent, self.store, self.concepts = agent, store, concepts
            self.checks = []
            self.recomputes = []
            self.stale = False
            self.outcome = None  # None: done; an exception to raise from the LLM pass
            self.hold = None  # an Event the LLM pass waits for (a long recompute)
            self.gone = False  # the agent was deleted before the job loaded it
            self.cat = self.load()  # the agent of the request that runs the check

        def load(self):
            handler = self.store.install(
                gf.make_handler(handler_mod, self.agent, enable_concept_relations=self.concepts),
            )
            pod = self

            async def has_stale_visible_concepts(fingerprint):
                pod.checks.append(fingerprint)
                return pod.stale

            async def recompute_concept_relations(stray_cat, source_group=None, gen=None):
                assert handler is not pod.cat.vector_memory_handler, "the job used the handler of the request"
                pod.recomputes.append(gen)
                if pod.hold is not None:
                    await pod.hold.wait()
                if pod.outcome is not None:
                    raise pod.outcome

            handler.has_stale_visible_concepts = has_stale_visible_concepts
            handler.recompute_concept_relations = recompute_concept_relations
            return types.SimpleNamespace(agent_key=self.agent, vector_memory_handler=handler, large_language_model="llm")

        async def agent_for_job(self, tenant):
            return None if self.gone else self.load()

    PODS = {}  # agent -> the pod whose job loads the agent

    async def agent_for_recompute(tenant):
        return await PODS[tenant].agent_for_job(tenant)

    jobs_mod.load_agent = agent_for_recompute

    HELD = []  # every Event a test holds a recompute on: released whatever the test's outcome

    async def settle():
        """Wait for the background recomputes, bounded: a test failing while a job waits must not hang the run."""
        for event in HELD:
            event.set()
        jobs = list(jobs_mod.JOBS)
        if jobs:
            await asyncio.wait(jobs, timeout=10)

    def new_process():
        """A process starting: nothing checked yet."""
        main_mod._concept_checked.clear()
        main_mod._concept_recheck_at.clear()
        main_mod._concept_recompute_pending.clear()

    def new_pod(store=None, concepts=True, agent=None):
        agent = agent or f"concepts_{uuid.uuid4().hex[:8]}"
        pod = Pod(agent, store or gf.FakeGraphStore(), concepts)
        PODS[agent] = pod
        return pod

    async def test_an_llm_change_recomputes_once():
        new_process()
        pod = new_pod()
        pod.store.concept_marker = "fp-1"
        await main_mod._ensure_concept_generation(pod.cat)  # nothing changed
        await settle()
        assert pod.checks == ["fp-1"] and pod.recomputes == [], (pod.checks, pod.recomputes)

        pod.store.concept_fingerprint = "fp-2"  # the agent's LLM changed
        pod.stale = True
        await main_mod._ensure_concept_generation(pod.cat)
        await main_mod._ensure_concept_generation(pod.cat)  # the next messages: no further check
        await settle()
        assert pod.checks == ["fp-1", "fp-2"], pod.checks
        assert pod.recomputes == ["fp-2"] and pod.store.concept_marker == "fp-2" and pod.store.concept_gcs == 1

    async def test_the_end_of_the_request_does_not_stop_the_recompute():
        new_process()
        pod = new_pod()
        pod.store.concept_marker, pod.store.concept_fingerprint = "fp-1", "fp-2"
        pod.stale = True
        pod.hold = asyncio.Event()
        HELD.append(pod.hold)
        await main_mod._ensure_concept_generation(pod.cat)
        await asyncio.sleep(0.2)  # the LLM pass is running
        await pod.cat.vector_memory_handler.close()  # the request ends: its CheshireCat shuts its handler down
        pod.hold.set()
        await settle()
        assert pod.recomputes == ["fp-2"] and pod.store.concept_marker == "fp-2", pod.recomputes

    async def test_an_agent_never_recomputed_is_left_alone():
        """No marker and concepts of the current fingerprint (the release day on every agent): no LLM pass."""
        new_process()
        pod = new_pod()
        await main_mod._ensure_concept_generation(pod.cat)
        await settle()
        assert pod.checks == ["fp-1"] and pod.recomputes == [] and pod.store.concept_marker is None

    async def test_disabled_concepts_or_another_database_do_nothing():
        new_process()
        pod = new_pod(concepts=False)
        pod.stale = True
        await main_mod._ensure_concept_generation(pod.cat)
        assert pod.checks == [], pod.checks
        other = types.SimpleNamespace(agent_key="x", vector_memory_handler=object())
        await main_mod._ensure_concept_generation(other)  # e.g. Qdrant: no error

    async def test_two_pods_recompute_once():
        store = gf.FakeGraphStore()
        store.concept_marker, store.concept_fingerprint = "fp-1", "fp-2"
        first = new_pod(store)
        second = Pod(first.agent, store)
        first.stale = second.stale = True
        first.hold = asyncio.Event()
        HELD.append(first.hold)

        new_process()
        await main_mod._ensure_concept_generation(first.cat)
        await asyncio.sleep(0.2)  # the first pod's job holds the lock, its LLM pass running
        new_process()  # the second pod is another process
        PODS[first.agent] = second
        await main_mod._ensure_concept_generation(second.cat)
        await asyncio.sleep(0.2)
        assert second.recomputes == [], "both pods paid the LLM pass"
        await main_mod._ensure_concept_generation(second.cat)  # the next message on the second pod
        assert second.checks == ["fp-2"], "the second pod checks again at every message"

        main_mod._concept_recheck_at[first.agent] = 0  # the interval elapsed (e.g. the holder died, its lock expired)
        await main_mod._ensure_concept_generation(second.cat)  # still held: it backs off again
        await asyncio.sleep(0.2)
        assert second.checks == ["fp-2", "fp-2"] and second.recomputes == [], (second.checks, second.recomputes)

        PODS[first.agent] = first
        first.hold.set()
        await settle()
        assert first.recomputes == ["fp-2"] and store.concept_marker == "fp-2"
        locks = [k async for k in get_async_db().scan_iter(f"lock:*{first.agent}*")]
        assert not locks, f"lock left behind: {locks}"

    async def test_a_recompute_that_does_not_end_is_retried_by_the_next_recall():
        for outcome in (RuntimeError("llm down"), asyncio.CancelledError(), "gone"):
            new_process()
            pod = new_pod()
            pod.store.concept_marker, pod.store.concept_fingerprint = "fp-1", "fp-2"
            pod.stale = True
            if outcome == "gone":
                pod.gone = True
            else:
                pod.outcome = outcome
            await main_mod._ensure_concept_generation(pod.cat)
            await settle()
            assert pod.store.concept_marker == "fp-1", outcome

            pod.outcome, pod.gone = None, False
            await main_mod._ensure_concept_generation(pod.cat)
            await settle()
            assert pod.store.concept_marker == "fp-2", (outcome, pod.recomputes)

    async def test_a_settings_save_runs_on_an_agent_of_its_own():
        new_process()
        pod = new_pod()
        pod.store.concept_marker, pod.store.concept_fingerprint = "fp-1", "fp-2"
        hook = getattr(main_mod.after_vector_database_settings_update, "function",
                       main_mod.after_vector_database_settings_update)
        await hook("Neo4jGraphRAGConfig", {"concept_definitions": "A: a"}, {"concept_definitions": "B: b"}, pod.cat)
        await settle()
        assert pod.recomputes == ["fp-2"] and pod.store.concept_marker == "fp-2", pod.recomputes

    async def test_the_hooks_check_and_never_break():
        calls = []
        saved = main_mod._ensure_concept_generation

        async def recording(cat):
            calls.append(cat.agent_key)
            raise RuntimeError("neo4j down")

        main_mod._ensure_concept_generation = recording
        pod = new_pod()

        async def embedder():
            return None

        handler = pod.cat.vector_memory_handler
        stray = types.SimpleNamespace(
            agent_key=pod.agent, vector_memory_handler=handler, large_language_model="llm",
            working_memory=types.SimpleNamespace(user_message=types.SimpleNamespace(text="who is Alice?")),
            embedder=embedder,
        )
        handler._align_embedder_lazy = lambda: asyncio.sleep(0)
        handler._entity_extractor = None  # no spaCy model to load here
        try:
            recall = getattr(main_mod.before_cat_recalls_memories, "function", main_mod.before_cat_recalls_memories)
            config = object()
            assert await recall(config, stray) is config
            store_docs = getattr(
                main_mod.before_rabbithole_stores_documents, "function", main_mod.before_rabbithole_stores_documents,
            )
            from langchain_core.documents import Document

            docs = [Document(page_content="Alice wrote")]
            assert await store_docs(docs, stray) == docs
        finally:
            main_mod._ensure_concept_generation = saved
        assert calls == [stray.agent_key, stray.agent_key], calls

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
            HELD.clear()
        leftovers = [k async for k in get_async_db().scan_iter("lock:*concepts_*")]
        if leftovers:
            failed += 1
            print(f"FAIL locks left: {leftovers}")
            await get_async_db().delete(*leftovers)
        print(f"\ntest_concept_generation: {len(TESTS) - failed}/{len(TESTS)} passed")
        return failed

    sys.exit(1 if asyncio.run(run_all()) else 0)
