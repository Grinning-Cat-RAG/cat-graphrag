"""Stateful property test of the graph phases on the REAL ingestion engine of the core.

A generator (hypothesis ``RuleBasedStateMachine``) explores sequences of operations on the sources of one agent —
upload / re-upload with new content, deletion, an embedder change (embedding invalidated), a change of the spaCy
models, the concept toggle, the automatic resume — with injected faults: a Neo4j write failure, a transient LLM
outcome, a crash of the worker in the middle of the spaCy phase followed by a restart (new handler, no task in
memory). After every step it checks the business rules:

1. a source whose row is ``completed`` holds exactly the graph of its CURRENT content (entities, provenance and
   RELATED_TO from its current documents only; concepts iff enabled): nothing stale, nothing missing;
2. the row status is the one the operations imply (completed / error / processing after a crash / gone);
3. a deleted source leaves no document, edge or RELATED_TO provenance behind, and the others are untouched (1);
4. no extraction task outlives the phase that owns it, a failure, or the deletion of its source.

Sources with the same name in the knowledge base and in a chat are out of the generator on purpose: RELATED_TO
records the provenance by name only (a known limitation, see the README).

``--mutate llm-dependency`` (the LLM phase no longer depends on the spaCy one) and ``--mutate no-purge`` (the spaCy
phase no longer purges before writing) inject a wrong rule: the run must then FAIL, which proves the rules above
catch it.

Standalone script (plugin rules: the body runs only as ``__main__``). Needs the core checkout and a Redis Stack on
localhost:6379 (database 14 is used and cleaned):

    cd ../grinning-cat-core
    .venv/bin/python /path/to/cat-graphrag/tests/test_graph_phases_stateful.py [--mutate ...] [--examples N]
"""

if __name__ == "__main__":
    import argparse
    import asyncio
    import os
    import sys
    import uuid

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import graph_fakes as gf

    parser = argparse.ArgumentParser()
    parser.add_argument("--mutate", choices=["llm-dependency", "no-purge"], default=None)
    parser.add_argument("--examples", type=int, default=60)
    args = parser.parse_args()

    main, handler_mod, phases_mod = gf.bootstrap("_graphrag_stateful")

    from hypothesis import HealthCheck, settings, strategies as st
    from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

    from cat.db.cruds import ingestions as crud_ingestions
    from cat.rabbit_hole import engine
    from cat.services.memory.models import PointStruct

    if args.mutate == "llm-dependency":
        phases_mod.GraphConceptPhase.depends_on = ("embedding",)

    LOOP = asyncio.new_event_loop()

    def run(coro):
        return LOOP.run_until_complete(coro)

    SOURCES = [("a.pdf", None), ("b.pdf", None), ("c.pdf", "chat-1")]
    WORDS = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace"]
    contents = st.lists(
        st.lists(st.sampled_from(WORDS), min_size=0, max_size=3).map(lambda ws: " met ".join(ws) or "nothing here"),
        min_size=1, max_size=3,
    )

    class GraphPhases(RuleBasedStateMachine):
        @initialize()
        def setup(self):
            self.agent = f"stateful_{uuid.uuid4().hex[:10]}"
            self.store = gf.FakeGraphStore()
            self.files = {}
            self.rh = gf.FakeRabbitHole(self.files, PointStruct)
            self.concepts = True
            self.models = {"en": "en_core_web_sm"}
            self.expected = {}  # (name, chat) -> "completed" | "error" | "processing"
            self.crash_next = False
            self.cat = gf.FakeCat(self.agent, self._handler(), main)
            run(gf.create_agent(self.agent))

        def _handler(self):
            handler = self.store.install(
                gf.make_handler(handler_mod, self.agent, enable_concept_relations=self.concepts, spacy_models=dict(self.models)),
            )
            if args.mutate == "no-purge":
                async def no_purge(*a, **k):
                    return 0

                handler.purge_source_graph_artifacts = no_purge
            original = handler._write_source_graph

            async def write(*a, **k):
                if self.crash_next:
                    self.crash_next = False
                    raise asyncio.CancelledError()  # the worker dies: not an error state
                return await original(*a, **k)

            handler._write_source_graph = write
            return handler

        def teardown(self):
            if hasattr(self, "agent"):
                run(gf.destroy_agent(self.agent))

        # ── operations ──────────────────────────────────────────────────────
        def _outcome(self, key, coro):
            """Run an operation and record the status the injected faults imply: a consumed write failure is an
            error, a consumed crash leaves the source in flight, anything else completes. A fault the operation
            did not reach (e.g. content without entities: nothing to write) is withdrawn."""
            failures = self.store.fail_writes
            crashing = self.crash_next
            try:
                row = run(coro)
            except asyncio.CancelledError:
                assert crashing, "a worker died without an injected crash"
                self.expected[key] = "processing"
                return
            finally:
                failed = self.store.fail_writes < failures
                self.store.fail_writes = 0
                self.crash_next = False
            if row is None:
                self.expected.pop(key, None)
                return
            self.expected[key] = "error" if failed else "completed"

        @rule(source=st.sampled_from(SOURCES), chunks=contents, fail=st.booleans(), crash=st.booleans())
        def upload(self, source, chunks, fail, crash):
            name, chat = source
            self.files[(chat, name)] = chunks
            self.store.fail_writes = 1 if fail else 0
            self.crash_next = crash and not fail
            self._outcome(source, gf.ingest(engine, self.rh, self.cat, chat, name))

        @rule(source=st.sampled_from(SOURCES))
        def delete(self, source):
            if source not in self.expected:
                return
            name, chat = source
            run(gf.delete(engine, self.rh, self.cat, chat, name))
            self.expected.pop(source)
            self.files.pop((chat, name), None)

        @rule(source=st.sampled_from(SOURCES))
        def embedder_change(self, source):
            if self.expected.get(source) != "completed":
                return
            name, chat = source
            self._outcome(source, gf.rerun(engine, self.rh, self.cat, chat, name, invalidate=["embedding"]))

        @rule()
        def spacy_models_change(self):
            self.models = {"en": "en_core_web_lg" if self.models["en"] == "en_core_web_sm" else "en_core_web_sm"}
            self.cat.vector_memory_handler._spacy_models = dict(self.models)
            self._resume()

        @rule()
        def concept_toggle(self):
            self.concepts = not self.concepts
            self.cat.vector_memory_handler._enable_concept_relations = self.concepts
            self._resume()

        @rule()
        def llm_transient(self):
            self.store.concept_outcomes = ["not_ready"]

        @rule()
        def restart(self):
            """The process restarts: a new handler on the same graph, nothing in memory; then the resume."""
            self.crash_next = False
            self.cat.vector_memory_handler = self._handler()
            self._resume()

        def _resume(self):
            """The automatic resume: sources in flight, and completed sources whose tags changed."""
            tags = run(engine.current_tags(self.rh, self.cat))
            for source, status in list(self.expected.items()):
                row = run(crud_ingestions.get_status(self.agent, source[1], source[0]))
                stale = any((row["completed_phases"].get(p) or {}).get("tag") != t for p, t in tags.items())
                if status == "processing" or (status == "completed" and stale):
                    self._outcome(source, gf.rerun(engine, self.rh, self.cat, source[1], source[0]))

        # ── rules ───────────────────────────────────────────────────────────
        @invariant()
        def rows_match_the_operations(self):
            if not hasattr(self, "agent"):
                return
            for (name, chat), status in self.expected.items():
                row = run(crud_ingestions.get_status(self.agent, chat, name))
                assert row is not None and row["status"] == status, (name, chat, status, row and row["status"])
            for name, chat in SOURCES:
                if (name, chat) not in self.expected:
                    assert run(crud_ingestions.get_status(self.agent, chat, name)) is None, (name, chat)

        @invariant()
        def completed_sources_hold_the_graph_of_their_content(self):
            if not hasattr(self, "agent"):
                return
            for (name, chat), status in self.expected.items():
                if status != "completed":
                    continue
                ids = self.store.scoped_doc_ids(name, gf.FakeRabbitHole.collection(chat), chat)
                assert len(ids) == len(self.files[(chat, name)]), (name, len(ids))
                actual = self.store.actual_for(ids, name)
                expected = self.store.expected_for(ids, self.concepts)
                assert actual == expected, f"{name}: graph is not the one of its content\n{actual}\n!=\n{expected}"

        @invariant()
        def deleted_sources_leave_nothing(self):
            if not hasattr(self, "agent"):
                return
            for name, chat in SOURCES:
                if (name, chat) in self.expected:
                    continue
                assert not self.store.scoped_doc_ids(name, gf.FakeRabbitHole.collection(chat), chat), name
                assert not self.store.related_of(name), (name, self.store.related_of(name))

        @invariant()
        def no_orphan_tasks(self):
            if not hasattr(self, "agent"):
                return
            handler = self.cat.vector_memory_handler
            for key, tasks in handler._pending_source_tasks.items():
                assert not tasks or self.expected.get(key) == "processing", f"tasks left for {key}"

    GraphPhases.TestCase.settings = settings(
        max_examples=args.examples,
        stateful_step_count=12,
        deadline=None,
        suppress_health_check=list(HealthCheck),
    )

    async def destroy_all_test_agents():
        """Every agent this harness created: a failing or shrunk example may not reach its teardown."""
        from cat.db.cruds import settings as crud_settings

        for agent_id in await crud_settings.get_agents_main_keys():
            if agent_id.startswith("stateful_"):
                await gf.destroy_agent(agent_id)

    try:
        GraphPhases.TestCase("runTest").runTest()
        outcome = "PASS"
    except Exception as exc:  # noqa: BLE001 - the verdict depends on the mutation
        outcome = "FAIL"
        failure = exc
    finally:
        run(destroy_all_test_agents())
        LOOP.close()

    if args.mutate:
        if outcome == "FAIL":
            print(f"PASS mutation '{args.mutate}' is caught: {str(failure).splitlines()[0][:200]}")
            sys.exit(0)
        print(f"FAIL mutation '{args.mutate}' was NOT caught by the rules")
        sys.exit(1)
    if outcome == "FAIL":
        raise failure
    print(f"PASS test_graph_phases_stateful ({args.examples} examples)")
