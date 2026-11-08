"""The tags of the graph phases and the concept generation must track what actually changes their output.

Loads the real plugin ``graphrag_handler`` (relative imports resolved through a
synthetic package) and pins:

- ``spacy_phase_settings`` (the tag of the spaCy phase): the spaCy models,
  ``enable_entity_extraction`` and ``enable_concept_relations`` (the phase
  purges the concepts too); the technology patterns are refreshed by their
  own job on save, so they must NOT move it; model order must not move it;
- ``concept_phase_settings`` (the tag of the LLM phase): only
  ``enable_concept_relations`` (definitions and prompt belong to the
  generation flip);
- ``_concept_fingerprint`` (the concept generation): concept/relation
  definitions + prompt + agent-level LLM identity + ``enable_concept_relations``.

The frontend-visualisation toggle (``enable_student_knowledge_graph``) and the
display-only ``enable_knowledge_graph`` must NOT move the fingerprint. Cosmetic
edits (key order, trailing whitespace) must not move it either.

Standalone script (plugin rules): the host imports every ``.py`` under the
plugin folder, so the ENTIRE body lives inside ``if __name__ == "__main__":``
and importing this file executes zero code. Run it as a plain script from the
host checkout:

    cd ../grinning-cat-core
    .venv/bin/python /path/to/cat-graphrag/tests/test_phase_fingerprint.py
"""


if __name__ == "__main__":
    import os

    # the real core refuses to load without these secrets: test-only values, never a running Cat's
    os.environ.setdefault("CAT_CRYPTO_KEY", "test_only_crypto_key_0123456789abcdef")
    os.environ.setdefault("CAT_CRYPTO_SALT", "test_only_crypto_salt_0123456789")
    os.environ.setdefault("CAT_JWT_SECRET", "test_only_jwt_secret_0123456789_abcdefghij")

    import asyncio
    import importlib
    import sys
    import traceback
    import types
    from pathlib import Path
    from unittest.mock import patch

    PLUGIN_DIR = Path(__file__).resolve().parent.parent
    PACKAGE = "_mygraph_fingerprint"

    def _load_handler_module():
        module = sys.modules.get(PACKAGE)
        if module is None:
            package = types.ModuleType(PACKAGE)
            package.__path__ = [str(PLUGIN_DIR)]
            sys.modules[PACKAGE] = package
        return importlib.import_module(PACKAGE + ".graphrag_handler")

    def _handler(**overrides):
        handler_mod = _load_handler_module()
        kwargs = dict(
            neo4j_uri="bolt://fake",
            neo4j_user="u",
            neo4j_password="p",
            enable_entity_extraction=True,
            enable_concept_relations=True,
            concept_definitions="ALPHA: first\nBETA: second",
            relation_definitions="LINKS: connects",
            concept_relations_prompt="Extract the concepts.",
        )
        kwargs.update(overrides)
        handler = handler_mod.GraphRAGHandler(**kwargs)
        handler.agent_id = "agent_test"
        return handler

    def _fixed_llm(handler, identity):
        async def _identity():
            return identity

        handler._resolve_agent_llm_identity = _identity
        return handler

    def test_spacy_phase_settings_sensitive_to_entity_extraction_toggle():
        on = _handler(enable_entity_extraction=True)
        off = _handler(enable_entity_extraction=False)
        assert on.spacy_phase_settings() != off.spacy_phase_settings()

    def test_spacy_phase_settings_ignore_patterns_and_model_order():
        a = _handler(
            spacy_models={"en": "en_core_web_sm", "it": "it_core_news_sm"},
            extra_technology_patterns=["Python", "  Docker  "],
        )
        b = _handler(
            spacy_models={"it": "it_core_news_sm", "en": "en_core_web_sm"},
            extra_technology_patterns=["Kafka"],
        )
        assert a.spacy_phase_settings() == b.spacy_phase_settings()

    def test_spacy_phase_settings_follow_the_concept_toggle():
        # the spaCy phase purges the concepts too: turning them off must re-run it
        assert _handler(enable_concept_relations=False).spacy_phase_settings() != _handler().spacy_phase_settings()

    def test_concept_phase_settings_hold_only_the_toggle():
        a = _handler(concept_definitions="ALPHA: first", concept_relations_prompt="one")
        b = _handler(concept_definitions="GAMMA: third", concept_relations_prompt="two")
        assert a.concept_phase_settings() == b.concept_phase_settings()
        assert _handler(enable_concept_relations=False).concept_phase_settings() != a.concept_phase_settings()

    def test_concept_fingerprint_moves_with_llm_identity():
        async def _run():
            a = _fixed_llm(_handler(), "OpenRouterConfig:qwen/qwen3")
            b = _fixed_llm(_handler(), "OpenRouterConfig:llama/llama-3")
            assert await a._concept_fingerprint() != await b._concept_fingerprint()

        asyncio.run(_run())

    def test_concept_fingerprint_moves_with_definitions_and_prompt():
        async def _run():
            base = _fixed_llm(_handler(), "LLM:model")
            other_defs = _fixed_llm(
                _handler(concept_definitions="ALPHA: first\nGAMMA: third"), "LLM:model"
            )
            other_rel = _fixed_llm(
                _handler(relation_definitions="LINKS: connects\nCAUSES: causes"), "LLM:model"
            )
            other_prompt = _fixed_llm(
                _handler(concept_relations_prompt="Extract the concepts differently."),
                "LLM:model",
            )
            base_fp = await base._concept_fingerprint()
            assert await other_defs._concept_fingerprint() != base_fp
            assert await other_rel._concept_fingerprint() != base_fp
            assert await other_prompt._concept_fingerprint() != base_fp

        asyncio.run(_run())

    def test_concept_fingerprint_moves_with_concept_relations_toggle():
        async def _run():
            on = _fixed_llm(_handler(enable_concept_relations=True), "LLM:model")
            off = _fixed_llm(_handler(enable_concept_relations=False), "LLM:model")
            assert await on._concept_fingerprint() != await off._concept_fingerprint()

        asyncio.run(_run())

    def test_concept_fingerprint_ignores_display_only_toggles():
        async def _run():
            base = _fixed_llm(_handler(), "LLM:model")
            student = _fixed_llm(
                _handler(enable_student_knowledge_graph=True), "LLM:model"
            )
            knowledge = _fixed_llm(_handler(enable_knowledge_graph=True), "LLM:model")
            base_fp = await base._concept_fingerprint()
            assert await student._concept_fingerprint() == base_fp
            assert await knowledge._concept_fingerprint() == base_fp

        asyncio.run(_run())

    def test_concept_fingerprint_cosmetic_edits_are_stable():
        async def _run():
            a = _fixed_llm(
                _handler(
                    concept_definitions="ALPHA: first\nBETA: second",
                    relation_definitions="LINKS: connects",
                    concept_relations_prompt="Extract the concepts.   \n",
                ),
                "LLM:model",
            )
            b = _fixed_llm(
                _handler(
                    concept_definitions="BETA: second\nALPHA: first",
                    relation_definitions="LINKS: connects",
                    concept_relations_prompt="  Extract the concepts.",
                ),
                "LLM:model",
            )
            assert await a._concept_fingerprint() == await b._concept_fingerprint()

        asyncio.run(_run())

    def test_llm_identity_is_config_name_and_model_never_secrets():
        handler_mod = _load_handler_module()
        handler = handler_mod.GraphRAGHandler(
            neo4j_uri="bolt://fake",
            neo4j_user="u",
            neo4j_password="p",
        )
        handler.agent_id = "agent_test"

        async def fake_settings(agent_key, category):
            assert category == "llm"
            return {
                "name": "OpenRouterConfig",
                "value": {"model": "qwen/qwen3", "api_key": "SECRET-DO-NOT-LEAK"},
            }

        async def _run():
            with patch("cat.db.cruds.settings.get_settings_by_category", fake_settings):
                return await handler._resolve_agent_llm_identity()

        identity = asyncio.run(_run())
        assert identity == "OpenRouterConfig:qwen/qwen3"
        assert "SECRET" not in identity

    _tests = {
        name: fn
        for name, fn in list(globals().items())
        if name.startswith("test_") and callable(fn)
    }
    _failed = 0
    for _name, _fn in _tests.items():
        try:
            _fn()
            print(f"PASS {_name}")
        except Exception as _exc:  # noqa: BLE001 - report and continue
            _failed += 1
            traceback.print_exc()
            print(f"FAIL {_name}: {_exc}")

    print(f"\ntest_phase_fingerprint: {len(_tests) - _failed}/{len(_tests)} passed")
    raise SystemExit(1 if _failed else 0)
