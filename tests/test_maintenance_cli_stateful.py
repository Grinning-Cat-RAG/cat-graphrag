"""Stateful property test of the maintenance CLI on the REAL core ingestion API (rows, locks, phase engine, resume).

A generator (hypothesis ``RuleBasedStateMachine``) explores sequences on the sources of one agent: uploads through
the server (bytes kept or not, a Neo4j failure), sources ingested before the states existed, deletions, the CLI ops
(``--reingest`` / ``--reembed`` / ``--resume-failed`` / ``--graph``, on either collection) with injected faults (a
Neo4j write failure, the CLI process dying in the middle of a source or right after the wipe of ``--graph``), on an agent that may start with
sources ingested before the states existed, the expiry of the maintenance marker and the
automatic resume of the server (the REAL ``recovery.resume_agent``). After every step it checks the business rules:

1. a source whose row is ``completed`` with the current tag of every phase (nothing left for the automatic resume)
   holds exactly the documents of its content and the graph derived from them;
2. the CLI says ``ok`` (exit 0) iff every source it processed ended completed (or was deleted meanwhile);
3. after a CLI op with no fault, every source in its scope is completed and has a state, but the sources whose bytes
   were not kept under ``--reingest`` (left untouched); after ``--graph`` the whole tenant graph is the one of the
   sources' contents (nothing stale, nothing missing);
4. a CLI op that ends (well or badly) leaves the agent out of maintenance; a dead CLI leaves it in maintenance until
   the marker expires, and the automatic resume never touches the agent meanwhile;
5. no extraction task outlives a completed source.

``--mutate no-reset`` (``--graph`` does not reset the graph entries of the diaries before the wipe: a CLI dying
after it leaves sources completed and up to date on paper, without a graph) and ``--mutate no-legacy`` (the sources
without a state are not reingested) inject a wrong rule: the run must FAIL.

Standalone script (plugin rules: the body runs only as ``__main__``). Needs the core checkout and a Redis Stack on
localhost:6379 (database 14 is used and cleaned):

    cd ../grinning-cat-core
    .venv/bin/python /path/to/cat-graphrag/tests/test_maintenance_cli_stateful.py [--mutate ...] [--examples N]
"""

if __name__ == "__main__":
    import argparse
    import asyncio
    import os
    import sys
    import types
    import uuid

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import graph_fakes as gf

    parser = argparse.ArgumentParser()
    parser.add_argument("--mutate", choices=["no-reset", "no-legacy"], default=None)
    parser.add_argument("--examples", type=int, default=120)
    args = parser.parse_args()

    main_mod, handler_mod, phases_mod = gf.bootstrap("_graphrag_cli_stateful")

    from hypothesis import HealthCheck, settings, strategies as st
    from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

    from cat.db.cruds import ingestions as crud_ingestions
    from cat.db.cruds import settings as crud_settings
    from cat.rabbit_hole import recovery
    from cat.services.memory.models import PointStruct

    ma = gf.load_cli(handler_mod, phases_mod)

    if args.mutate == "no-reset":
        async def no_reset(agent_id, phase_ids):
            return 0

        crud_ingestions.reset_phase_entries_for_agent = no_reset

    if args.mutate == "no-legacy":
        async def no_legacy(ccat, collections):
            return []

        ma._sources_without_state = no_legacy

    LOOP = asyncio.new_event_loop()

    def run(coro):
        return LOOP.run_until_complete(coro)

    SOURCES = [("a.pdf", None), ("b.pdf", None), ("c.pdf", "chat-1")]
    WORDS = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank"]
    # a fault in one operation out of four: rare enough that most operations complete and interleave
    RARE = st.sampled_from([False, False, False, True])
    contents = st.lists(
        st.lists(st.sampled_from(WORDS), min_size=0, max_size=3).map(lambda ws: " met ".join(ws) or "nothing here"),
        min_size=1, max_size=3,
    )
    OPS = ["reingest", "reembed", "resume-failed", "graph"]

    class CliMachine(RuleBasedStateMachine):
        @initialize(
            old_sources=st.lists(st.tuples(st.sampled_from(SOURCES), contents), max_size=3, unique_by=lambda t: t[0]),
        )
        def setup(self, old_sources):
            """An agent, possibly born before the ingestion states: some sources already in its memory, without a
            state (the case the maintenance meets on a real installation)."""
            self.agent = f"clistate_{uuid.uuid4().hex[:10]}"
            run(gf.create_agent(self.agent, graphrag=True))
            self.world = gf.CliWorld(self.agent, main_mod, handler_mod, PointStruct)
            self.crash_next = False
            self.crash_wipe = False
            self.dead_cli = False  # a CLI died: its marker is still there
            self.last = None  # (op, collection, fault, exit code, in-scope sources before the op)
            self._wire()

            async def load(agent_id):
                return self.world.cat

            ma._load_agent = load
            for (name, chat), chunks in old_sources:
                run(self.world.legacy(name, chunks, chat))

        def _wire(self):
            handler = self.world.handler
            write = handler._write_source_graph

            async def crashing_write(*a, **k):
                if self.crash_next:
                    self.crash_next = False
                    raise asyncio.CancelledError()  # the CLI process dies here
                return await write(*a, **k)

            handler._write_source_graph = crashing_write
            wipe = handler.wipe_tenant_graph

            async def crashing_wipe():
                await wipe()
                if self.crash_wipe:
                    self.crash_wipe = False
                    raise asyncio.CancelledError()  # the CLI process dies right after its wipe

            handler.wipe_tenant_graph = crashing_wipe

        def teardown(self):
            if hasattr(self, "agent"):
                run(gf.destroy_agent(self.agent))

        def _has_docs(self, name, chat):
            return bool(self.world.store.scoped_doc_ids(name, gf.FakeRabbitHole.collection(chat), chat))

        # ── server-side operations ──────────────────────────────────────────
        @rule(source=st.sampled_from(SOURCES), chunks=contents, keep=st.booleans(), fail=RARE)
        def upload(self, source, chunks, keep, fail):
            if self.dead_cli:  # a CLI died: only its marker expiring or the resume happen (checked here, not as a
                return  # precondition: whether a crash happened depends on background work hypothesis cannot replay)
            name, chat = source
            self.world.store.fail_writes = 1 if fail else 0
            run(self.world.upload(name, chunks, chat, keep=keep))
            self.world.store.fail_writes = 0
            self.last = None

        @rule(source=st.sampled_from(SOURCES), chunks=contents)
        def legacy(self, source, chunks):
            if self.dead_cli:
                return
            name, chat = source
            if run(self.world.row(name, chat)) is not None or self._has_docs(name, chat):
                return
            run(self.world.legacy(name, chunks, chat))
            self.last = None

        @rule(source=st.sampled_from(SOURCES))
        def delete(self, source):
            if self.dead_cli:
                return
            name, chat = source
            if run(self.world.row(name, chat)) is None:
                return
            run(self.world.delete(name, chat))
            self.last = None

        # ── the CLI ─────────────────────────────────────────────────────────
        @rule(op=st.sampled_from(OPS), episodic=st.booleans(), fault=st.sampled_from([None, None, "neo4j", "crash"]))
        def cli(self, op, episodic, fault):
            if self.dead_cli:
                return
            collection = "episodic" if episodic else "declarative"
            scope = self._scope(op, collection)
            not_kept = {s for s in scope if (s[1], s[0]) not in self.world.kept}
            self.world.store.fail_writes = 1 if fault == "neo4j" else 0
            self.crash_next = fault == "crash"
            self.crash_wipe = fault == "crash-wipe"
            argv = ["--agent", self.agent, f"--{op}", "--collection", collection, "--yes"]
            try:
                code, _ = run(gf.run_cli(ma, *argv))
            except asyncio.CancelledError:
                # the process died: its marker stays (``finally`` never runs on a kill), the lock of the source
                # expires; what it was processing stays in flight
                run(crud_ingestions.set_maintenance(self.agent, ttl=3600))
                self.dead_cli = True
                self.last = None
                return
            finally:
                # a fault the op did not reach (e.g. nothing to write) is withdrawn
                faulted = fault == "neo4j" and self.world.store.fail_writes == 0
                self.world.store.fail_writes = 0
                self.crash_next = False
                self.crash_wipe = False
            self.last = (op, collection, faulted, code, scope, not_kept)

        def _a_completed_source_has_a_graph(self):
            for row in run(crud_ingestions.list_statuses(self.agent)):
                ids = self.world.store.scoped_doc_ids(row["source"], gf.FakeRabbitHole.collection(row["chat_id"]), row["chat_id"])
                if row["status"] == "completed" and any(d in ids for d, _ in self.world.store.mentions):
                    return True
            return False

        @rule()
        def graph_dies_after_its_wipe(self):
            """A restart right after the wipe of ``--graph`` (the longest step on a large tenant), when there is a graph
            to lose: what the resume must rebuild. Checked inside the rule, not as a precondition: the graph is written
            by background work, and hypothesis requires preconditions that replay identically."""
            if self.dead_cli:
                return
            if not self._a_completed_source_has_a_graph():
                self.last = None
                return
            self.cli("graph", False, "crash-wipe")

        def _scope(self, op, collection):
            """The sources the op processes: the rows (and the sources without a state, but for --resume-failed)
            of its collection; --graph covers both."""
            rows = run(crud_ingestions.list_statuses(self.agent))
            wanted = (lambda chat: True) if op == "graph" else (lambda chat: (chat is not None) == (collection == "episodic"))
            scope = {(r["source"], r["chat_id"]) for r in rows if wanted(r["chat_id"])}
            if op == "resume-failed":
                return {(r["source"], r["chat_id"]) for r in rows if wanted(r["chat_id"]) and r["status"] == "error"}
            for (chat, name) in self.world.files:
                if wanted(chat) and self._has_docs(name, chat) and (chat, name) in self.world.kept:
                    scope.add((name, chat))
            return scope

        @rule()
        def marker_expires(self):
            if not self.dead_cli:
                return
            run(crud_ingestions.clear_maintenance(self.agent))
            self.dead_cli = False

        @rule()
        def automatic_resume(self):
            """One pass of the server's automatic resume (skips the agent while it is in maintenance)."""
            before = run(crud_ingestions.list_statuses(self.agent))

            async def get_cheshire_cat(agent_id):
                return self.world.cat

            lizard = types.SimpleNamespace(rabbit_hole=self.world.rh, get_cheshire_cat=get_cheshire_cat)
            run(recovery.resume_agent(lizard, self.agent))
            if self.dead_cli:
                assert run(crud_ingestions.list_statuses(self.agent)) == before, "the resume touched an agent in maintenance"
            self.last = None

        # ── rules ───────────────────────────────────────────────────────────
        @invariant()
        def completed_sources_hold_their_content(self):
            if not hasattr(self, "agent"):
                return
            tags = run(gf.current_tags(self.world.rh, self.world.cat))
            for row in run(crud_ingestions.list_statuses(self.agent)):
                diary = row.get("completed_phases") or {}
                current = all((diary.get(p) or {}).get("tag") == t for p, t in tags.items())
                if row["status"] == "completed" and current:
                    assert self.world.graph_ok(row["source"], row["chat_id"]), f"{row['source']}: stale or missing graph"

        @invariant()
        def cli_outcome(self):
            if not hasattr(self, "agent") or self.last is None:
                return
            op, collection, faulted, code, scope, not_kept = self.last
            assert not run(crud_ingestions.is_in_maintenance(self.agent)), "the CLI left the agent in maintenance"
            rows = {(r["source"], r["chat_id"]): r for r in run(crud_ingestions.list_statuses(self.agent))}
            processed = scope - not_kept if op == "reingest" else scope
            all_completed = all(rows.get(s, {}).get("status") == "completed" for s in processed)
            reingest_blocked = op == "reingest" and bool(not_kept)
            assert (code == 0) == (all_completed and not reingest_blocked), (op, code, all_completed, not_kept)
            if not faulted:
                assert all_completed, (op, collection, {s: rows.get(s, {}).get("status") for s in processed})
                if op == "graph" and set(rows) >= {(n, c) for (c, n) in self.world.files if self._has_docs(n, c)}:
                    every = set(self.world.store.docs)
                    expected = self.world.store.expected_for(every, True)
                    actual = (self.world.store.mentions, self.world.store.provenance,
                              {k for k in self.world.store.related})
                    assert actual == expected, "--graph left the tenant graph different from the sources' content"

        @invariant()
        def no_orphan_tasks(self):
            if not hasattr(self, "agent"):
                return
            statuses = {(r["source"], r["chat_id"]): r["status"] for r in run(crud_ingestions.list_statuses(self.agent))}
            for key, tasks in self.world.handler._pending_source_tasks.items():
                if tasks:
                    assert statuses.get(key) != "completed", f"tasks left for the completed source {key}"

    CliMachine.TestCase.settings = settings(
        max_examples=args.examples,
        stateful_step_count=12,
        deadline=None,
        suppress_health_check=list(HealthCheck),
    )

    async def destroy_all_test_agents():
        for agent_id in await crud_settings.get_agents_main_keys():
            if agent_id.startswith("clistate_"):
                await gf.destroy_agent(agent_id)

    try:
        CliMachine.TestCase("runTest").runTest()
        outcome = "PASS"
    except Exception as exc:  # noqa: BLE001 - the verdict depends on the mutation
        outcome = "FAIL"
        failure = exc
    finally:
        run(destroy_all_test_agents())
        LOOP.close()

    if args.mutate:
        if outcome == "FAIL":
            reasons = [str(e).splitlines()[0][:160] for e in getattr(failure, "exceptions", [failure])]
            print(f"PASS mutation '{args.mutate}' is caught: {' | '.join(reasons)}")
            sys.exit(0)
        print(f"FAIL mutation '{args.mutate}' was NOT caught by the rules")
        sys.exit(1)
    if outcome == "FAIL":
        raise failure
    print(f"PASS test_maintenance_cli_stateful ({args.examples} examples)")
