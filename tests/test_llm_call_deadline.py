"""Standalone, import-safe tests for the wall-clock LLM-call deadline.

The graph concept phase awaits a structured LLM call. A streaming response has
no overall deadline, so without an explicit wall clock a stalled socket keeps
the phase ``processing`` forever. These cases lock the fix:

  (a) an LLM that never returns -> ``asyncio.wait_for`` fires -> ``not_ready``
      on attempt 1 (NOT ``error``, NOT a hang);
  (b) after the retry budget is exhausted -> ``error`` naming the timeout;
  (c) a normal fast LLM still returns ``done`` with the product;
  (d) the deadline is bounded by ``CAT_INGESTION_LLM_CALL_TIMEOUT_SECONDS``
      (a small value is set in the test);
  (e) streaming is disabled AND reasoning is turned off for the structured
      call (the view passed to ``StructuredLLM`` carries a falsy ``stream`` kwarg
      and the OpenRouter ``reasoning`` override), while the original LLM object
      is left unmodified.

The Cat plugin loader imports every ``.py`` under a plugin folder at
activation, so this whole file is inert on import: every import, class and
test lives inside ``if __name__ == "__main__":``. External dependencies are
stubbed in ``sys.modules`` before the plugin module is loaded. Run it as a
plain script:

    python tests/test_llm_call_deadline.py
"""

if __name__ == "__main__":
    import asyncio
    import importlib
    import json
    import os
    import sys
    import time
    import types
    from unittest.mock import patch

    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PKG = "catgraphrag_deadlinetest"
    ENV = "CAT_INGESTION_LLM_CALL_TIMEOUT_SECONDS"

    # ------------------------------------------------------------------
    # Minimal external stubs (installed BEFORE importing the plugin)
    # ------------------------------------------------------------------

    class _StubLog:
        def info(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    def _install_cat_stub():
        cat_mod = types.ModuleType("cat")
        cat_mod.__path__ = []
        cat_mod.hook = lambda *a, **kwargs: (lambda f: f)
        cat_mod.RecallSettings = type("RecallSettings", (), {})
        cat_mod.VectorDatabaseSettings = type("VectorDatabaseSettings", (), {})
        cat_mod.Embeddings = type("Embeddings", (), {})
        cat_mod.AgenticWorkflowTask = type("AgenticWorkflowTask", (), {})

        class _BaseHandler:
            def __init__(self, **kwargs):
                pass

        cat_mod.BaseVectorDatabaseHandler = _BaseHandler
        cat_mod.log = _StubLog()
        sys.modules["cat"] = cat_mod

        log_mod = types.ModuleType("cat.log")
        log_mod.log = _StubLog()
        sys.modules["cat.log"] = log_mod

        looking = types.ModuleType("cat.looking_glass")
        looking.__path__ = []
        sys.modules["cat.looking_glass"] = looking

        stray = types.ModuleType("cat.looking_glass.stray_cat")
        stray.StrayCat = type("StrayCat", (), {})
        sys.modules["cat.looking_glass.stray_cat"] = stray

        services = types.ModuleType("cat.services")
        services.__path__ = []
        sys.modules["cat.services"] = services

        memory = types.ModuleType("cat.services.memory")
        memory.__path__ = []
        sys.modules["cat.services.memory"] = memory

        models_mod = types.ModuleType("cat.services.memory.models")

        class _PointStruct:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        models_mod.PointStruct = _PointStruct
        models_mod.DocumentRecall = type("DocumentRecall", (), {})
        models_mod.Record = type("Record", (), {})
        models_mod.ScoredPoint = type("ScoredPoint", (), {})
        models_mod.UpdateResult = type("UpdateResult", (), {})
        # the core's enum: one copy for every harness (graph_fakes.py), checked against the core
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from graph_fakes import StubVectorMemoryType

        models_mod.VectorMemoryType = StubVectorMemoryType
        sys.modules["cat.services.memory.models"] = models_mod

    def _install_neo4j_stub():
        neo4j_mod = types.ModuleType("neo4j")
        neo4j_mod.AsyncGraphDatabase = type("AsyncGraphDatabase", (), {})
        neo4j_mod.AsyncDriver = type("AsyncDriver", (), {})
        neo4j_mod.AsyncSession = type("AsyncSession", (), {})
        sys.modules["neo4j"] = neo4j_mod

        exceptions_mod = types.ModuleType("neo4j.exceptions")
        exceptions_mod.Neo4jError = type("Neo4jError", (), {})
        sys.modules["neo4j.exceptions"] = exceptions_mod

    def _install_langchain_stub():
        lc = types.ModuleType("langchain_core")
        lc.__path__ = []
        sys.modules["langchain_core"] = lc

        docs = types.ModuleType("langchain_core.documents")
        docs.Document = type("Document", (), {})
        sys.modules["langchain_core.documents"] = docs

        runnables = types.ModuleType("langchain_core.runnables")
        runnables.RunnableConfig = type("RunnableConfig", (), {})
        sys.modules["langchain_core.runnables"] = runnables

    def _install_langdetect_stub():
        langdetect_mod = types.ModuleType("langdetect")

        class DetectorFactory:
            seed = 0

        setattr(langdetect_mod, "DetectorFactory", DetectorFactory)
        setattr(langdetect_mod, "detect_langs", lambda text: [])
        sys.modules["langdetect"] = langdetect_mod

    def _install_spacy_stubs():
        def _forbidden(*args, **kwargs):
            raise AssertionError("real spaCy must not be loaded inside the test")

        spacy_mod = types.ModuleType("spacy")
        spacy_mod.__path__ = []
        setattr(spacy_mod, "load", _forbidden)
        sys.modules["spacy"] = spacy_mod

        util_mod = types.ModuleType("spacy.util")
        setattr(util_mod, "is_package", lambda name: True)
        sys.modules["spacy.util"] = util_mod

        cli_mod = types.ModuleType("spacy.cli")
        cli_mod.__path__ = []
        sys.modules["spacy.cli"] = cli_mod

        download_mod = types.ModuleType("spacy.cli.download")
        setattr(download_mod, "download", _forbidden)
        sys.modules["spacy.cli.download"] = download_mod

        language_mod = types.ModuleType("spacy.language")
        setattr(language_mod, "Language", type("Language", (), {}))
        sys.modules["spacy.language"] = language_mod

        tokens_mod = types.ModuleType("spacy.tokens")
        setattr(tokens_mod, "Doc", type("Doc", (), {}))
        sys.modules["spacy.tokens"] = tokens_mod

    _install_cat_stub()
    _install_neo4j_stub()
    _install_langchain_stub()
    _install_langdetect_stub()
    _install_spacy_stubs()

    _pkg = types.ModuleType(PKG)
    _pkg.__path__ = [REPO_ROOT]
    sys.modules[PKG] = _pkg

    graphrag_handler = importlib.import_module(f"{PKG}.graphrag_handler")

    # ------------------------------------------------------------------
    # Fakes / helpers
    # ------------------------------------------------------------------

    class _Result:
        """Stand-in for ``StructuredLLMResult`` (only ``.model``/``.text``)."""

        def __init__(self, model, text):
            self.model = model
            self.text = text

    _PRODUCT = {
        "concepts": [{"type": "CONCEPT", "text": "Algebra"}],
        "relations": [
            {
                "type": "RELATED_TO",
                "origin": "Algebra",
                "destination": "Calculus",
                "text": "Algebra precedes Calculus",
            }
        ],
    }

    class _HangingStructuredLLM:
        calls = 0

        async def run(self, prompt, llm, output_model):
            type(self).calls += 1
            await asyncio.Event().wait()

    class _FastStructuredLLM:
        async def run(self, prompt, llm, output_model):
            return _Result(_PRODUCT, json.dumps(_PRODUCT))

    class _CapturingStructuredLLM:
        captured = None

        async def run(self, prompt, llm, output_model):
            type(self).captured = llm
            llm.with_structured_output(output_model)
            return _Result(_PRODUCT, json.dumps(_PRODUCT))

    class _StreamingLLM:
        """ChatOpenAI-like LLM recording how the structured call is invoked."""

        streaming = True
        extra_body = None
        reasoning = None
        # ChatOpenAI exposes ``extra_body`` as a pydantic field; the plugin only
        # sends the OpenRouter reasoning override to models that accept it.
        model_fields = {"extra_body": None, "reasoning": None}
        openai_api_base = "https://openrouter.ai/api/v1"

        def __init__(self):
            self.structured_kwargs = None
            self.bind_kwargs = None

        def with_structured_output(self, output_model, **kwargs):
            self.structured_kwargs = kwargs
            return self

        def bind(self, **kwargs):
            self.bind_kwargs = kwargs
            clone = _StreamingLLM()
            clone.streaming = kwargs.get("stream", kwargs.get("streaming", True))
            return clone

    class _NonNativeLLM:
        """LLM without native structured output: exercises the fallback path."""

        streaming = True
        extra_body = None
        reasoning = None
        model_fields = {"extra_body": None, "reasoning": None}
        openai_api_base = "https://openrouter.ai/api/v1"

        def __init__(self):
            self.bind_kwargs = None

        def bind(self, **kwargs):
            self.bind_kwargs = kwargs
            return self

        async def ainvoke(self, *args, **kwargs):
            return _Result(_PRODUCT, json.dumps(_PRODUCT))

    def _make_handler():
        handler = graphrag_handler.GraphRAGHandler(
            neo4j_uri="bolt://fake",
            neo4j_user="u",
            neo4j_password="p",
            enable_concept_relations=True,
        )
        handler.agent_id = "agent_test"
        return handler

    def _set_env(value):
        if value is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = value

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    async def test_a_never_returning_llm_is_not_ready():
        _HangingStructuredLLM.calls = 0
        handler = _make_handler()
        _set_env("0.05")
        started = time.perf_counter()
        try:
            with patch.object(
                graphrag_handler, "StructuredLLM", _HangingStructuredLLM
            ):
                outcome = await handler._llm_extract_relations(
                    "some document text", llm=object(), attempt=1
                )
        finally:
            _set_env(None)
        elapsed = time.perf_counter() - started

        assert outcome["status"] == "not_ready", outcome
        assert outcome["status"] != "error", outcome
        assert "timed out" in (outcome.get("reason") or "").lower(), outcome
        assert outcome.get("retry_after") == 5, outcome
        assert elapsed < 1.0, elapsed
        assert _HangingStructuredLLM.calls == 1, _HangingStructuredLLM.calls
        print("PASS (a) never-returning LLM -> not_ready, bounded (no hang)")

    async def test_b_budget_exhausted_is_error_naming_timeout():
        handler = _make_handler()
        _set_env("0.05")
        try:
            with patch.object(
                graphrag_handler, "StructuredLLM", _HangingStructuredLLM
            ):
                first = await handler._llm_extract_relations(
                    "text", llm=object(), attempt=1
                )
                second = await handler._llm_extract_relations(
                    "text", llm=object(), attempt=2
                )
        finally:
            _set_env(None)

        assert first["status"] == "not_ready", first
        assert second["status"] == "error", second
        reason = (second.get("reason") or "").lower()
        assert "timed out" in reason, second
        assert "2 attempt" in reason, second
        print("PASS (b) budget exhausted -> error naming the timeout")

    async def test_c_fast_llm_still_done_with_product():
        handler = _make_handler()
        _set_env("5")
        try:
            with patch.object(graphrag_handler, "StructuredLLM", _FastStructuredLLM):
                outcome = await handler._llm_extract_relations(
                    "text", llm=object(), attempt=1
                )
        finally:
            _set_env(None)

        assert outcome["status"] == "done", outcome
        assert outcome["concepts"] == _PRODUCT["concepts"], outcome
        assert outcome["relations"] == _PRODUCT["relations"], outcome
        print("PASS (c) fast LLM -> done with the product")

    async def test_d_deadline_bounded_by_env():
        handler = _make_handler()
        _set_env("0.10")
        started = time.perf_counter()
        try:
            with patch.object(
                graphrag_handler, "StructuredLLM", _HangingStructuredLLM
            ):
                outcome = await handler._llm_extract_relations(
                    "text", llm=object(), attempt=1
                )
        finally:
            _set_env(None)
        elapsed = time.perf_counter() - started

        assert outcome["status"] == "not_ready", outcome
        assert elapsed < 1.0, elapsed
        assert graphrag_handler.llm_call_timeout_seconds() == 180.0, (
            graphrag_handler.llm_call_timeout_seconds()
        )
        for bad in ("not-a-number", "-5", "inf", "nan", "0"):
            _set_env(bad)
            try:
                assert graphrag_handler.llm_call_timeout_seconds() == 180.0, bad
            finally:
                _set_env(None)
        print("PASS (d) deadline bounded by the env value; invalid values -> 180s")

    async def test_e_structured_call_disables_streaming_and_reasoning():
        handler = _make_handler()
        llm = _StreamingLLM()
        _set_env("5")
        try:
            with patch.object(
                graphrag_handler, "StructuredLLM", _CapturingStructuredLLM
            ):
                outcome = await handler._llm_extract_relations(
                    "text", llm=llm, attempt=1
                )
        finally:
            _set_env(None)

        assert outcome["status"] == "done", outcome
        captured = _CapturingStructuredLLM.captured
        assert captured is not None, "StructuredLLM was not called"
        # (a) the bound structured request carries stream + reasoning override.
        assert getattr(captured, "streaming", None) is False, captured
        assert llm.structured_kwargs is not None, "structured call not invoked"
        assert llm.structured_kwargs.get("stream") is False, llm.structured_kwargs
        assert llm.structured_kwargs.get("extra_body") == {
            "reasoning": {"enabled": False}
        }, llm.structured_kwargs
        # (c) the original LLM object is untouched by the per-call view.
        assert getattr(llm, "streaming", None) is True, "global LLM was mutated"
        assert llm.extra_body is None, "global LLM extra_body was mutated"
        assert llm.reasoning is None, "global LLM reasoning was mutated"
        print("PASS (e) structured call disables streaming + reasoning, LLM untouched")

    async def test_f_fallback_call_disables_reasoning():
        llm = _NonNativeLLM()
        view = graphrag_handler._without_streaming(llm)
        assert view.bind_kwargs.get("stream") is False, view.bind_kwargs
        assert view.bind_kwargs.get("extra_body") == {
            "reasoning": {"enabled": False}
        }, view.bind_kwargs
        assert llm.extra_body is None, "global LLM extra_body was mutated"
        print("PASS (f) fallback call disables streaming + reasoning, LLM untouched")

    async def test_g_reasoning_override_only_for_openrouter():
        """The ``reasoning`` body field is OpenRouter's: a plain OpenAI-compatible
        endpoint rejects unknown request arguments, so it must not receive it.
        Streaming stays disabled for every provider."""
        for base in ("https://api.openai.com/v1", None):
            llm = _StreamingLLM()
            llm.openai_api_base = base
            view = graphrag_handler._without_streaming(llm)
            view.with_structured_output(object)
            assert llm.structured_kwargs.get("stream") is False, llm.structured_kwargs
            assert "extra_body" not in llm.structured_kwargs, (base, llm.structured_kwargs)
            plain = _NonNativeLLM()
            plain.openai_api_base = base
            bound = graphrag_handler._without_streaming(plain)
            assert "extra_body" not in bound.bind_kwargs, (base, bound.bind_kwargs)
        print("PASS (g) reasoning override sent only to OpenRouter endpoints")

    async def _main():
        await test_a_never_returning_llm_is_not_ready()
        await test_b_budget_exhausted_is_error_naming_timeout()
        await test_c_fast_llm_still_done_with_product()
        await test_d_deadline_bounded_by_env()
        await test_e_structured_call_disables_streaming_and_reasoning()
        await test_f_fallback_call_disables_reasoning()
        await test_g_reasoning_override_only_for_openrouter()

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    print("OK LLM call deadline (seven tests)")
