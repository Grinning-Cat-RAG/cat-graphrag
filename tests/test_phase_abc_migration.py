"""Standalone, import-safe tests for the class-based GraphRAG phases.

Covers the phase-contract v2 migration of the two graph phases:

  (a) a pure-vector agent (no ``GraphRAGHandler``) makes BOTH phases
      inapplicable, so the dispatcher records them as ``skipped``;
  (b) ``delete()`` removes only the given source's artefacts and is
      idempotent (called twice, second call is a no-op);
  (c) ``tag()`` is the existing configuration fingerprint: stable while the
      config is unchanged, different when the spaCy / concept config changes,
      and it never calls the LLM;
  (d) ``process()`` DELEGATES to the existing
      ``run_spacy_extraction_phase`` / ``run_llm_extraction_phase`` path
      instead of re-implementing extraction;
  (e) the module owns no diary/state/lock/retry protocol logic;
  (f) ``phase_for()`` resolves the same class from a string id, a mapping
      carrying ``"id"`` (legacy PhaseSpec payload) or an object with ``.id``.

The Cat plugin loader imports every ``.py`` under a plugin folder at
activation, so the whole file is inert on import: every import, class and test
lives inside ``if __name__ == "__main__":``. Run it as a plain script:

    python tests/test_phase_abc_migration.py
"""

if __name__ == "__main__":
    import asyncio
    import sys
    import types
    from abc import ABC, abstractmethod
    from dataclasses import dataclass
    from unittest.mock import AsyncMock

    # ------------------------------------------------------------------
    # Import-safe stubs
    # ------------------------------------------------------------------

    def _install_cat_stub():
        cat_mod = types.ModuleType("cat")
        cat_mod.__path__ = []

        def _hook(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda f: f

        cat_mod.hook = _hook
        sys.modules["cat"] = cat_mod

    def _install_effing_stub():
        """A faithful minimal EffING phase contract (ABC + resolver)."""

        for name in (
            "cat.plugins",
            "cat.plugins.cat_efficient_ingestion",
        ):
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod

        phases = types.ModuleType("cat.plugins.cat_efficient_ingestion.phases")

        @dataclass(frozen=True)
        class PhaseResult:
            status: str
            reason: str | None = None
            processed: int | None = None
            expected: int | None = None
            warnings: list[dict] | None = None

        class IngestionPhase(ABC):
            id = ""
            depends_on = ()
            settings_category = None
            settings_check = None
            not_ready_budget_seconds = None
            not_ready_budget_env = None

            def not_ready_budget(self):
                """Resolve the phase's own budget (same order as EffING)."""
                import os

                if self.not_ready_budget_env:
                    try:
                        value = float(os.environ[self.not_ready_budget_env])
                    except (KeyError, TypeError, ValueError):
                        value = None
                    if value is not None and value >= 0:
                        return value
                if self.not_ready_budget_seconds is not None:
                    return float(self.not_ready_budget_seconds)
                return None

            @property
            def name(self):
                return self.id

            @abstractmethod
            async def applies_to(self, cat, deps_state):
                ...

            @abstractmethod
            async def process(self, source, cat):
                ...

            @abstractmethod
            async def delete(self, source, cat):
                ...

            async def tag(self, settings=None):
                from cat.plugins.cat_efficient_ingestion.phase_tag_resolver import (
                    own_tag,
                )

                return own_tag(self.id, {} if settings is None else settings)

        phases.IngestionPhase = IngestionPhase
        phases.PhaseResult = PhaseResult
        # Same contract as EffING's phases._DELETE_SCOPE: the delete
        # orchestrator publishes the source's scope ("agent" or a chat id).
        import contextvars

        phases._DELETE_SCOPE = contextvars.ContextVar(
            "effing_delete_scope", default=None
        )
        sys.modules[phases.__name__] = phases

        # registry.reset_phase_entries: drops diary entries -> phases re-run.
        registry = types.ModuleType("cat.plugins.cat_efficient_ingestion.registry")
        registry.reset_calls = []
        registry.fail_with = None

        async def reset_phase_entries(agent_id, scope, source, phases_to_reset):
            if registry.fail_with is not None:
                raise registry.fail_with
            registry.reset_calls.append(
                (agent_id, scope, source, sorted(phases_to_reset))
            )
            return {}

        registry.reset_phase_entries = reset_phase_entries
        sys.modules[registry.__name__] = registry

        import hashlib
        import json

        resolver = types.ModuleType(
            "cat.plugins.cat_efficient_ingestion.phase_tag_resolver"
        )

        def own_tag(kind, value):
            payload = json.dumps(
                value, sort_keys=True, separators=(",", ":"), default=str
            )
            return hashlib.sha256(f"{kind}:{payload}".encode("utf-8")).hexdigest()

        resolver.own_tag = own_tag
        sys.modules[resolver.__name__] = resolver

        return IngestionPhase, PhaseResult, own_tag

    class FakeGraphHandler:
        """A GraphRAG handler stand-in: only the phase-facing surface."""

        def __init__(self, spacy_marker="spacy-v1", concept_marker="concept-v1"):
            self.agent_id = "agent-x"
            self.spacy_marker = spacy_marker
            self.concept_marker = concept_marker
            self.artifacts = {"file.pdf": {"entity:A"}, "other.pdf": {"entity:B"}}
            self.llm_calls = 0
            self.run_spacy_extraction_phase = AsyncMock(
                return_value={"status": "done"}
            )
            self.run_llm_extraction_phase = AsyncMock(return_value={"status": "done"})
            self.purge_source_graph_artifacts = AsyncMock(side_effect=self._purge)
            self._extract_and_link_entities = AsyncMock()
            self._extract_concept_relations = AsyncMock()

        def _purge(self, source):
            self.artifacts.pop(source, None)
            return 0

        def _spacy_fingerprint(self):
            return self.spacy_marker

        async def _concept_fingerprint(self):
            return self.concept_marker

        async def _resolve_agent_llm(self):  # must never run inside tag()
            self.llm_calls += 1
            raise AssertionError("tag() must never resolve or call the LLM")

    class FakeVectorHandler:
        """A pure-vector handler: not GraphRAG, so the graph is not active."""

    class FakeCat:
        def __init__(self, handler):
            self.vector_memory_handler = handler

    _install_cat_stub()
    _IngestionPhase, _PhaseResult, _own_tag = _install_effing_stub()

    _pkg = types.ModuleType("catgraphrag_phase")
    _pkg.__path__ = [__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))]
    sys.modules["catgraphrag_phase"] = _pkg

    from catgraphrag_phase import phases

    # The tests exercise the contract with a fake handler; production resolves
    # the live ``GraphRAGHandler`` through ``isinstance`` (same guard).
    _ORIGINAL_RESOLVER = phases.graphrag_handler
    phases.graphrag_handler = lambda cat: (
        cat.vector_memory_handler
        if isinstance(cat.vector_memory_handler, FakeGraphHandler)
        else None
    )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    async def test_a_vector_only_agent_is_skipped():
        cat = FakeCat(FakeVectorHandler())
        for phase in (phases.GraphSpacyPhase(cat), phases.GraphConceptPhase(cat)):
            applies, reason = await phase.applies_to(cat, {})
            assert applies is False, (phase.id, applies)
            assert "not active" in (reason or ""), (phase.id, reason)
        print("PASS (a) vector-only agent: both graph phases are skipped")

    async def test_b_delete_source_scoped_and_idempotent():
        handler = FakeGraphHandler()
        cat = FakeCat(handler)
        phase = phases.GraphSpacyPhase(cat)

        await phase.delete("file.pdf", cat)
        assert handler.artifacts == {"other.pdf": {"entity:B"}}, handler.artifacts
        handler.purge_source_graph_artifacts.assert_awaited_once_with("file.pdf")

        # second call is a no-op (idempotent) and still only targets file.pdf
        await phase.delete("file.pdf", cat)
        assert handler.artifacts == {"other.pdf": {"entity:B"}}, handler.artifacts
        assert handler.purge_source_graph_artifacts.await_count == 2
        assert handler.purge_source_graph_artifacts.await_args_list[1].args == (
            "file.pdf",
        )

        # the other phase shares the same graph cleanup contract
        concept = phases.GraphConceptPhase(cat)
        await concept.delete("file.pdf", cat)
        assert handler.artifacts == {"other.pdf": {"entity:B"}}, handler.artifacts
        print("PASS (b) delete() is source-scoped and idempotent")

    async def test_c_tag_follows_config_only():
        handler = FakeGraphHandler()
        cat = FakeCat(handler)
        spacy = phases.GraphSpacyPhase(cat)
        concept = phases.GraphConceptPhase(cat)

        spacy_first = await spacy.tag({})
        spacy_second = await spacy.tag({})
        assert spacy_first == spacy_second, (spacy_first, spacy_second)
        assert spacy_first == _own_tag(spacy.id, handler.spacy_marker), spacy_first

        concept_first = await concept.tag({})
        concept_second = await concept.tag({})
        assert concept_first == concept_second, (concept_first, concept_second)
        assert concept_first == _own_tag(concept.id, handler.concept_marker)

        handler.spacy_marker = "spacy-v2"
        assert await spacy.tag({}) != spacy_first
        handler.concept_marker = "concept-v2"
        assert await concept.tag({}) != concept_first
        assert handler.llm_calls == 0, "tag() must not call the LLM"
        print("PASS (c) tag() is stable per config and changes with the config")

    async def test_d_process_delegates_to_existing_path():
        handler = FakeGraphHandler()
        cat = FakeCat(handler)

        spacy = phases.GraphSpacyPhase(cat)
        spacy_result = await spacy.process("file.pdf", cat)
        assert spacy_result.status == "done", spacy_result
        handler.run_spacy_extraction_phase.assert_awaited_once_with(
            "file.pdf", cat=cat
        )
        handler._extract_and_link_entities.assert_not_awaited()

        concept = phases.GraphConceptPhase(cat)
        concept_result = await concept.process("file.pdf", cat)
        assert concept_result.status == "done", concept_result
        handler.run_llm_extraction_phase.assert_awaited_once_with(
            "file.pdf", cat=cat
        )
        handler._extract_concept_relations.assert_not_awaited()
        print("PASS (d) process() delegates to the existing ingestion path")

    async def test_e_phase_module_owns_no_protocol_logic():
        """The plugin must NOT re-implement the phase-ingestion protocol.

        Diary writes, staleness, locks, retries and status derivation all
        belong to EffING; the plugin only declares + executes + cleans up.
        Guard the module source so a future edit cannot reintroduce them.
        """
        import inspect

        source = inspect.getsource(phases)
        forbidden = (
            "record_phase_",
            "should_abandon",
            "distributed_lock",
            "derive_source_status",
        )
        found = [name for name in forbidden if name in source]
        assert not found, f"phase module re-implements protocol logic: {found}"
        print("PASS (e) phase module contains no diary/state/lock/retry protocol")

    async def test_f_phase_for_accepts_spec_or_id():
        """``phase_for`` resolves the same class from an id, a dict or an object.

        The legacy hooks receive the PhaseSpec payload (a dict), while the
        class API passes the string id; both forms must resolve identically.
        """
        cat = FakeCat(FakeGraphHandler())

        class SpecWithId:
            id = "graph_spacy_extraction"

        from_id = phases.phase_for("graph_spacy_extraction", cat)
        from_dict = phases.phase_for({"id": "graph_spacy_extraction"}, cat)
        from_obj = phases.phase_for(SpecWithId(), cat)

        assert from_id is not None and from_dict is not None and from_obj is not None
        assert type(from_id) is phases.GraphSpacyPhase
        assert type(from_dict) is type(from_id)
        assert type(from_obj) is type(from_id)

        # the unknown forms keep the None contract
        assert phases.phase_for("does_not_exist", cat) is None
        assert phases.phase_for({"id": "does_not_exist"}, cat) is None
        assert phases.phase_for({}, cat) is None
        assert phases.phase_for(None, cat) is None
        print("PASS (f) phase_for() accepts a string id, a dict or an object")

    async def test_g_concept_phase_declares_llm_budget():
        """The concept phase declares its own 300s LLM not-ready budget."""
        import os

        env = "CAT_INGESTION_LLM_NOT_READY_MAX_SECONDS"
        concept = phases.GraphConceptPhase(FakeCat(FakeGraphHandler()))

        assert concept.not_ready_budget_seconds == 300.0, (
            concept.not_ready_budget_seconds
        )
        assert concept.not_ready_budget_env == env, concept.not_ready_budget_env

        saved = os.environ.pop(env, None)
        try:
            # no override: the declared value wins over the generic 3600s hour
            assert concept.not_ready_budget() == 300.0, concept.not_ready_budget()

            # a declared env var overrides the declared value when set
            os.environ[env] = "120"
            assert concept.not_ready_budget() == 120.0, concept.not_ready_budget()

            # a negative/invalid override falls back to the declared value
            os.environ[env] = "-5"
            assert concept.not_ready_budget() == 300.0, concept.not_ready_budget()
            os.environ[env] = "not-a-number"
            assert concept.not_ready_budget() == 300.0, concept.not_ready_budget()
        finally:
            os.environ.pop(env, None)
            if saved is not None:
                os.environ[env] = saved
        print("PASS (g) concept phase declares the 300s LLM not-ready budget")

    async def test_h_warning_phase_is_complete_not_retried():
        """A ``done`` result carrying warnings is COMPLETE.

        Warnings are telemetry attached to ``done``; they never change the
        status. EffING retries only ``not_ready`` (and blocks dependents on
        ``error``/``skipped``), so a warning phase is never re-run and never
        blocks a dependent.
        """
        handler = FakeGraphHandler()
        warning = {
            "code": "empty-extraction",
            "message": "no entities found in 42 chars of 1 chunks for 'file.pdf'",
        }
        handler.run_spacy_extraction_phase = AsyncMock(
            return_value={
                "status": "done",
                "warnings": [warning],
                "processed": 0,
            }
        )
        cat = FakeCat(handler)
        phase = phases.GraphSpacyPhase(cat)

        result = await phase.process("file.pdf", cat)

        assert result.status == "done", result
        assert result.processed == 0, result
        assert result.warnings == [warning], result
        # The machine's retry rule: only not_ready is retried; error blocks
        # dependents; done (with or without warnings) is terminal-complete.
        assert result.status not in ("not_ready", "error"), (
            "a phase with warnings must never be retried or treated as failed"
        )
        print("PASS (h) a phase with warnings is complete and not retried")

    async def test_i_delete_requeues_the_sibling_graph_phase():
        """Both graph phases write ONE shared graph (entities, PROVENANCE,
        RELATED_TO), so purging it for one phase destroys the other's product.
        EffING's recovery re-runs only the abandoned phase and the LATER ones
        (reembed._recover_abandoned_phases): the sibling's diary entry must be
        dropped too, or it stays ``complete`` over a deleted graph."""
        registry = sys.modules["cat.plugins.cat_efficient_ingestion.registry"]
        effing_phases = sys.modules["cat.plugins.cat_efficient_ingestion.phases"]
        handler = FakeGraphHandler()
        cat = FakeCat(handler)

        registry.reset_calls.clear()
        token = effing_phases._DELETE_SCOPE.set("agent")
        try:
            await phases.GraphConceptPhase(cat).delete("file.pdf", cat)
            await phases.GraphSpacyPhase(cat).delete("file.pdf", cat)
        finally:
            effing_phases._DELETE_SCOPE.reset(token)
        assert registry.reset_calls == [
            ("agent-x", "agent", "file.pdf", ["graph_spacy_extraction"]),
            ("agent-x", "agent", "file.pdf", ["graph_llm_extraction"]),
        ], registry.reset_calls

        # Without a published scope: a chat-scoped caller (StrayCat.id) is
        # chat-scoped, otherwise agent-scoped (EffING's own fallback).
        registry.reset_calls.clear()
        chat_cat = FakeCat(handler)
        chat_cat.id = "chat-7"
        await phases.GraphConceptPhase(chat_cat).delete("file.pdf", chat_cat)
        assert registry.reset_calls[-1][1] == "chat-7", registry.reset_calls
        await phases.GraphConceptPhase(cat).delete("file.pdf", cat)
        assert registry.reset_calls[-1][1] == "agent", registry.reset_calls
        print("PASS (i) a graph-phase delete re-queues the sibling graph phase")

    async def test_j_requeue_failure_fails_the_delete():
        """If the sibling cannot be re-queued, delete() must raise: EffING then
        leaves the entry processing and retries next pass, instead of
        resetting only this phase over a graph the sibling still claims."""
        registry = sys.modules["cat.plugins.cat_efficient_ingestion.registry"]
        handler = FakeGraphHandler()
        cat = FakeCat(handler)
        registry.fail_with = RuntimeError("redis down")
        try:
            try:
                await phases.GraphSpacyPhase(cat).delete("file.pdf", cat)
            except RuntimeError:
                pass
            else:
                raise AssertionError("delete() swallowed the re-queue failure")
        finally:
            registry.fail_with = None
        print("PASS (j) a failed sibling re-queue fails the delete")

    async def test_k_delete_without_effing_registry_still_purges():
        """No EffING registry importable means no diary to fix: the purge
        still runs and delete() does not raise."""
        saved = sys.modules.pop("cat.plugins.cat_efficient_ingestion.registry")
        try:
            handler = FakeGraphHandler()
            cat = FakeCat(handler)
            await phases.GraphSpacyPhase(cat).delete("file.pdf", cat)
            handler.purge_source_graph_artifacts.assert_awaited_once_with("file.pdf")
        finally:
            sys.modules["cat.plugins.cat_efficient_ingestion.registry"] = saved
        print("PASS (k) delete without the EffING registry still purges")

    async def _main():
        await test_a_vector_only_agent_is_skipped()
        await test_b_delete_source_scoped_and_idempotent()
        await test_c_tag_follows_config_only()
        await test_d_process_delegates_to_existing_path()
        await test_e_phase_module_owns_no_protocol_logic()
        await test_f_phase_for_accepts_spec_or_id()
        await test_g_concept_phase_declares_llm_budget()
        await test_h_warning_phase_is_complete_not_retried()
        await test_i_delete_requeues_the_sibling_graph_phase()
        await test_j_requeue_failure_fails_the_delete()
        await test_k_delete_without_effing_registry_still_purges()

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    print("OK phase ABC migration (eleven tests)")
