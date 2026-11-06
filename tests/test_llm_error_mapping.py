"""Standalone, import-safe tests for the LLM-failure -> phase-state mapping.

Covers the graph concept phase contract (``_extract_concept_relations`` /
``_llm_extract_relations``): a failed LLM call must never become a silent
``done``. The cases:

  (a) ``StructuredLLMError`` with unparsable raw -> ``not_ready`` on the first
      attempt, ``error`` afterwards (never ``done``);
  (b) no usable LLM -> ``error`` with the configuration reason;
  (c) a VALID LLM call returning empty concepts AND relations on non-empty
      text -> ``done`` + an ``empty-extraction`` warning (never ``error``);
  (d) a clean extraction still yields ``done`` with the product;
  (e) a partial lenient parse (items dropped) -> ``error`` with ``dropped=<n>``;
  (f) NATIVE structured output, schema-valid but effectively empty product
      (whitespace-only text/endpoints) -> ``done`` + warning, never ``error``;
  (g) NATIVE empty strings: the output model's ``min_length=1`` rejects them
      (never ``done``);
  (h) a non-list ``concepts``/``relations`` container counts as dropped items,
      not as zero candidates (MAJOR-1 MINOR-4);
  (i) ``enable_concept_relations=False`` -> ``skipped`` with a reason, never
      ``done`` (MAJOR-1 MINOR-5), and the fingerprint covers the toggle;
  (j) a clean NATIVE product still yields ``done`` with the product;
  (k) a non-list ``concepts``/``relations`` container (``null``/``0``) is
      coerced to zero parsed items, counted as dropped, and never iterated
      into a ``TypeError``;
  (l) an empty relation ``text`` is allowed by the output model but dropped
      and counted, never a whole-payload schema failure and never ``done``;
  (m) a NATIVE payload with SOME items dropped -> ``error`` with
      ``dropped>=1``, never ``done``;
  (n) a timeout -> ``not_ready`` then ``error``, never a warning;
  (o) the spaCy phase ran and produced zero entities -> ``done`` with an
      ``empty-extraction`` warning and ``processed=0``;
  (p) the spaCy phase with no stored points -> ``not_ready`` (unchanged).

The Cat plugin loader imports every ``.py`` under a plugin folder at
activation, so this whole file is inert on import: every import, class and
test lives inside ``if __name__ == "__main__":``. External dependencies are
stubbed in ``sys.modules`` before the plugin module is loaded. Run it as a
plain script:

    python tests/test_llm_error_mapping.py
"""

if __name__ == "__main__":
    import asyncio
    import importlib
    import json
    import os
    import sys
    import types
    from unittest.mock import AsyncMock, patch

    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PKG = "catgraphrag_llmtest"

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

        # The NATIVE structured-output branch imports RunnableConfig lazily;
        # a plain callable class is enough for the harness.
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

    def _install_pydantic_stub():
        """Keep real pydantic when importable (real field validation).

        Returns ``True`` when the real package is in use. The minimal stub is
        only a fallback for harnesses that run without pydantic installed; it
        does NOT enforce field constraints, so the ``min_length=1`` assertion
        is gated on the returned flag.
        """
        try:
            import pydantic  # noqa: F401 - real validation when available
        except ImportError:
            pass
        else:
            return True

        pydantic_mod = types.ModuleType("pydantic")

        class Field:
            def __init__(self, default=None, *, default_factory=None, **kwargs):
                self.default = default
                self.default_factory = default_factory

        class ConfigDict(dict):
            pass

        class BaseModel:
            def __init__(self, **kwargs):
                for name, value in kwargs.items():
                    setattr(self, name, value)

        def BeforeValidator(fn):
            return fn

        def create_model(name, **kwargs):
            return type(name, (BaseModel,), {})

        setattr(pydantic_mod, "Field", Field)
        setattr(pydantic_mod, "ConfigDict", ConfigDict)
        setattr(pydantic_mod, "BaseModel", BaseModel)
        setattr(pydantic_mod, "BeforeValidator", BeforeValidator)
        setattr(pydantic_mod, "create_model", create_model)
        sys.modules["pydantic"] = pydantic_mod
        return False

    _install_cat_stub()
    _install_neo4j_stub()
    _install_langchain_stub()
    _install_langdetect_stub()
    _install_spacy_stubs()
    _REAL_PYDANTIC = _install_pydantic_stub()

    _pkg = types.ModuleType(PKG)
    _pkg.__path__ = [REPO_ROOT]
    sys.modules[PKG] = _pkg

    graphrag_handler = importlib.import_module(f"{PKG}.graphrag_handler")
    structured_llm = importlib.import_module(f"{PKG}.structured_llm")

    StructuredLLMError = structured_llm.StructuredLLMError

    # ------------------------------------------------------------------
    # Fakes / helpers
    # ------------------------------------------------------------------

    class _Point:
        def __init__(self, pid, text, metadata=None):
            self.id = pid
            self.payload = {
                "page_content": text,
                "metadata": metadata or {},
            }

    def _points(text="Some document text with concepts."):
        return [_Point("doc1", text)]

    def _meta_point(pid, text, **metadata):
        return _Point(pid, text, metadata)

    _BINARY_SEGMENT = "%PDF-1.5\n\x00\x01\x02 binary catalogue-card bytes \ufffd\ufffd"

    class _Result:
        """Stand-in for ``StructuredLLMResult`` (only ``.model``/``.text``)."""

        def __init__(self, model, text):
            self.model = model
            self.text = text

    class _FakeStructuredLLM:
        behavior = None
        calls = 0
        prompts = []

        async def run(self, prompt, llm, output_model):
            type(self).calls += 1
            type(self).prompts.append(prompt)
            return type(self).behavior()

    def _make_handler():
        handler = graphrag_handler.GraphRAGHandler(
            neo4j_uri="bolt://fake",
            neo4j_user="u",
            neo4j_password="p",
            enable_concept_relations=True,
        )
        handler.agent_id = "agent_test"
        handler._store_concept_relations = AsyncMock()
        handler._concept_fingerprint = AsyncMock(return_value="gen-test")
        return handler

    def _reset_fake():
        _FakeStructuredLLM.calls = 0
        _FakeStructuredLLM.prompts = []
        _FakeStructuredLLM.behavior = None

    class _NativeSeamLLM:
        """LLM exposing ``with_structured_output``: the NATIVE seam.

        ``StructuredLLM`` therefore takes its native branch and its own
        validation runs; only the bound runnable's ``ainvoke`` is faked.
        """

        def __init__(self, payload):
            self._payload = payload

        def with_structured_output(self, output_model):
            payload = self._payload

            class _BoundRunnable:
                async def ainvoke(self, prompt, config=None):
                    return payload

            return _BoundRunnable()

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    async def test_a_structured_error_is_never_done():
        _reset_fake()
        _FakeStructuredLLM.behavior = lambda: (_ for _ in ()).throw(
            StructuredLLMError("unparsable", "not json at all")
        )
        handler = _make_handler()

        with patch.object(graphrag_handler, "StructuredLLM", _FakeStructuredLLM):
            first = await handler._extract_concept_relations(
                "src-a", _points(), llm=object()
            )
            second = await handler._extract_concept_relations(
                "src-a", _points(), llm=object()
            )

        assert first["status"] == "not_ready", first
        assert first["status"] != "done", first
        assert "not json at all" in (first.get("reason") or ""), first
        assert second["status"] == "error", second
        assert "not json at all" in (second.get("reason") or ""), second
        assert _FakeStructuredLLM.calls == 2, _FakeStructuredLLM.calls
        handler._store_concept_relations.assert_not_awaited()
        print("PASS (a) StructuredLLMError -> not_ready then error, never done")

    async def test_b_no_llm_is_configuration_error():
        handler = _make_handler()
        handler._resolve_agent_llm = AsyncMock(return_value=None)

        outcome = await handler._extract_concept_relations(
            "src-b", _points(), llm=None, stray_cat=None
        )

        assert outcome["status"] == "error", outcome
        assert outcome["reason"] == (
            "no LLM configured for graph concept extraction"
        ), outcome
        handler._store_concept_relations.assert_not_awaited()
        print("PASS (b) no LLM -> error with the configuration reason")

    async def test_c_valid_empty_is_done_with_warning():
        """A VALID LLM call returning empty concepts AND relations on non-empty
        text is ``done`` with an ``empty-extraction`` warning (never ``error``):
        the call ran, the content simply has no concepts."""
        _reset_fake()
        raw = '{"concepts": [], "relations": []}'
        _FakeStructuredLLM.behavior = lambda: _Result(
            {"concepts": [], "relations": []}, raw
        )
        handler = _make_handler()

        with patch.object(graphrag_handler, "StructuredLLM", _FakeStructuredLLM):
            first = await handler._extract_concept_relations(
                "src-c", _points(), llm=object()
            )
            # A second run stays ``done``: no not_ready -> error transition.
            second = await handler._extract_concept_relations(
                "src-c", _points(), llm=object()
            )

        for outcome in (first, second):
            assert outcome["status"] == "done", outcome
            assert outcome["status"] != "error", outcome
            assert outcome["processed"] == 0, outcome
            warnings = outcome.get("warnings") or []
            assert len(warnings) == 1, outcome
            assert warnings[0]["code"] == "empty-extraction", outcome
            assert "chars" in warnings[0]["message"], outcome
            assert "src-c" in warnings[0]["message"], outcome
        handler._store_concept_relations.assert_not_awaited()
        print("PASS (c) valid empty extraction -> done + empty-extraction warning")

    async def test_d_clean_extraction_is_done_with_product():
        _reset_fake()
        product = {
            "concepts": [
                {"type": "CONCEPT", "text": "Algebra"},
                {"type": "CONCEPT", "text": "Calculus"},
            ],
            "relations": [
                {
                    "type": "PREREQUISITE_FOR",
                    "origin": "Algebra",
                    "destination": "Calculus",
                    "text": "Algebra is a prerequisite for Calculus",
                }
            ],
        }
        _FakeStructuredLLM.behavior = lambda: _Result(product, json.dumps(product))
        handler = _make_handler()

        with patch.object(graphrag_handler, "StructuredLLM", _FakeStructuredLLM):
            outcome = await handler._extract_concept_relations(
                "src-d", _points(), llm=object()
            )

        assert outcome["status"] == "done", outcome
        assert outcome["processed"] == 1, outcome
        handler._store_concept_relations.assert_awaited_once()
        stored = handler._store_concept_relations.await_args.args[1]
        assert len(stored["concepts"]) == 2, stored
        assert len(stored["relations"]) == 1, stored
        print("PASS (d) clean extraction -> done with the product")

    async def test_e_partial_extraction_reports_dropped_count():
        _reset_fake()
        raw = json.dumps(
            {
                "concepts": [
                    {"type": "CONCEPT", "text": "A"},
                    {"type": "CONCEPT", "text": "B"},
                ],
                "relations": [
                    {"type": "IS_A", "origin": "A", "destination": "B", "text": ""},
                    {"type": "NOT_A_TYPE", "origin": "A", "destination": "B", "text": ""},
                    {"type": "IS_A", "origin": "", "destination": "B", "text": ""},
                ],
            }
        )
        _FakeStructuredLLM.behavior = lambda: (_ for _ in ()).throw(
            StructuredLLMError("schema validation failed", raw)
        )
        handler = _make_handler()

        with patch.object(graphrag_handler, "StructuredLLM", _FakeStructuredLLM):
            outcome = await handler._extract_concept_relations(
                "src-e", _points(), llm=object()
            )

        assert outcome["status"] == "error", outcome
        assert "dropped=2" in (outcome.get("reason") or ""), outcome
        handler._store_concept_relations.assert_not_awaited()
        print("PASS (e) partial extraction -> error with dropped=2")

    # ------------------------------------------------------------------
    # MAJOR-1: the NATIVE structured LLM path must not report an empty
    # product as ``done`` either.
    # ------------------------------------------------------------------

    async def test_f_native_empty_product_is_done_with_warning():
        """Schema-valid but effectively empty native product -> done + warning.
        NOT patching StructuredLLM: its native branch + validation run."""
        _reset_fake()
        payload = {
            "concepts": [{"type": "DEFINITION", "text": "   "}],
            "relations": [
                {"type": "IS_A", "origin": " ", "destination": " ", "text": " "}
            ],
        }
        handler = _make_handler()

        outcome = await handler._extract_concept_relations(
            "src-f", _points(), llm=_NativeSeamLLM(payload)
        )

        assert outcome["status"] == "done", outcome
        assert outcome["processed"] == 0, outcome
        warnings = outcome.get("warnings") or []
        assert len(warnings) == 1, outcome
        assert warnings[0]["code"] == "empty-extraction", outcome
        handler._store_concept_relations.assert_not_awaited()
        print("PASS (f) native empty product -> done + empty-extraction warning")

    async def test_g_native_empty_strings_are_rejected():
        """The output model declares ``min_length=1``: empty strings must not
        pass validation (so the native path can never store them)."""
        _reset_fake()
        payload = {"concepts": [{"type": "DEFINITION", "text": ""}], "relations": []}
        handler = _make_handler()

        outcome = await handler._extract_concept_relations(
            "src-g", _points(), llm=_NativeSeamLLM(payload)
        )

        assert outcome["status"] != "done", outcome
        handler._store_concept_relations.assert_not_awaited()
        if _REAL_PYDANTIC:
            output_model = handler._build_llm_output_model()
            try:
                output_model(
                    concepts=[{"type": "DEFINITION", "text": ""}], relations=[]
                )
            except Exception:
                pass
            else:
                raise AssertionError(
                    "the output model must reject an empty concept text"
                )
        print("PASS (g) native empty strings are rejected, never done")

    async def test_h_non_list_items_count_as_dropped():
        """MINOR-4: a non-list concepts/relations container is a malformed
        payload whose items were ALL dropped — never 0 candidates."""
        handler = _make_handler()
        raw_bad_concepts = json.dumps(
            {
                "concepts": {"Algebra": "a branch of mathematics"},
                "relations": [
                    {"type": "IS_A", "origin": "A", "destination": "B", "text": "A is a B"}
                ],
            }
        )
        raw_bad_relations = json.dumps(
            {
                "concepts": [{"type": "DEFINITION", "text": "A"}],
                "relations": {"A": "B"},
            }
        )
        for raw in (raw_bad_concepts, raw_bad_relations):
            _reset_fake()
            _FakeStructuredLLM.behavior = lambda raw=raw: (_ for _ in ()).throw(
                StructuredLLMError("schema validation failed", raw)
            )
            handler._store_concept_relations.reset_mock()
            with patch.object(graphrag_handler, "StructuredLLM", _FakeStructuredLLM):
                outcome = await handler._extract_concept_relations(
                    "src-h", _points(), llm=object()
                )
            assert outcome["status"] == "error", (raw, outcome)
            assert "dropped=1" in (outcome.get("reason") or ""), (raw, outcome)
            handler._store_concept_relations.assert_not_awaited()
        print("PASS (h) non-list payload containers count as dropped")

    async def test_i_disabled_feature_is_skipped_not_done():
        """MINOR-5: with the feature off there is no product, so the phase is
        ``skipped`` with a reason. The fingerprint still covers the toggle, so
        enabling it later re-runs the phase."""
        disabled = graphrag_handler.GraphRAGHandler(
            neo4j_uri="bolt://fake",
            neo4j_user="u",
            neo4j_password="p",
            enable_concept_relations=False,
        )
        disabled.agent_id = "agent_test"

        with patch.object(type(disabled), "_drain_source_tasks_counted", new=AsyncMock()), \
             patch.object(
                 type(disabled),
                 "get_source_points_by_collection",
                 new=AsyncMock(return_value={"declarative": _points()}),
             ), \
             patch.object(
                 type(disabled), "_extract_concept_relations", new=AsyncMock()
             ) as extract:
            outcome = await disabled.run_llm_extraction_phase("src-i")

        assert outcome["status"] == "skipped", outcome
        assert outcome["status"] != "done", outcome
        assert "disabled" in (outcome.get("reason") or ""), outcome
        extract.assert_not_awaited()

        enabled = graphrag_handler.GraphRAGHandler(
            neo4j_uri="bolt://fake",
            neo4j_user="u",
            neo4j_password="p",
            enable_concept_relations=True,
        )
        enabled.agent_id = "agent_test"
        assert await disabled._concept_fingerprint() != await enabled._concept_fingerprint(), (
            "the concept fingerprint must include the enable toggle"
        )
        print("PASS (i) disabled feature -> skipped with a reason; toggle in fingerprint")

    async def test_j_native_clean_product_is_done():
        """A clean NATIVE product still yields ``done`` with the product."""
        _reset_fake()
        payload = {
            "concepts": [{"type": "DEFINITION", "text": "Algebra"}],
            "relations": [
                {
                    "type": "PREREQUISITE_FOR",
                    "origin": "Algebra",
                    "destination": "Calculus",
                    "text": "Algebra precedes Calculus",
                }
            ],
        }
        handler = _make_handler()

        outcome = await handler._extract_concept_relations(
            "src-j", _points(), llm=_NativeSeamLLM(payload)
        )

        assert outcome["status"] == "done", outcome
        assert outcome["processed"] == 1, outcome
        handler._store_concept_relations.assert_awaited_once()
        stored = handler._store_concept_relations.await_args.args[1]
        assert len(stored["concepts"]) == 1, stored
        assert len(stored["relations"]) == 1, stored
        print("PASS (j) clean native extraction -> done with the product")

    # ------------------------------------------------------------------
    # F2 MINOR/INFO items: malformed containers, empty relation text and
    # the NATIVE partial-drop gap.
    # ------------------------------------------------------------------

    async def test_k_non_iterable_container_is_dropped_not_raised():
        """MINOR (a): a ``concepts: null`` (or ``concepts: 0``) payload must
        not raise while being iterated. It is a malformed container whose
        items were all dropped, so the result is ``error`` with
        ``dropped>=1`` and the raw prefix — never an exception, never ``done``."""
        for bad in (None, 0):
            _reset_fake()
            raw = json.dumps(
                {
                    "concepts": bad,
                    "relations": [
                        {
                            "type": "IS_A",
                            "origin": "A",
                            "destination": "B",
                            "text": "A is a B",
                        }
                    ],
                }
            )
            _FakeStructuredLLM.behavior = lambda raw=raw: (_ for _ in ()).throw(
                StructuredLLMError("schema validation failed", raw)
            )
            handler = _make_handler()

            with patch.object(graphrag_handler, "StructuredLLM", _FakeStructuredLLM):
                outcome = await handler._extract_concept_relations(
                    "src-k", _points(), llm=object()
                )

            assert outcome["status"] == "error", (bad, outcome)
            assert outcome["status"] != "done", (bad, outcome)
            assert "dropped=1" in (outcome.get("reason") or ""), (bad, outcome)
            assert raw[:120] in (outcome.get("reason") or ""), (bad, outcome)
            handler._store_concept_relations.assert_not_awaited()
        print("PASS (k) non-iterable concepts container -> error, dropped>=1")

    async def test_l_empty_relation_text_is_dropped_not_schema_failure():
        """MINOR (b): the model allows an empty relation ``text`` (the store
        tolerates it), but such an item is DROPPED and counted, so one empty
        text yields ``error`` with ``dropped>=1`` instead of rejecting the
        whole payload at validation time or silently passing as ``done``."""
        _reset_fake()
        payload = {
            "concepts": [{"type": "DEFINITION", "text": "Algebra"}],
            "relations": [
                {
                    "type": "IS_A",
                    "origin": "Algebra",
                    "destination": "Mathematics",
                    "text": "",
                }
            ],
        }
        handler = _make_handler()

        if _REAL_PYDANTIC:
            model = handler._build_llm_output_model()
            instance = model(**payload)  # must NOT raise
            assert instance.relations[0].text == "", instance
            try:
                model(
                    concepts=[{"type": "DEFINITION", "text": ""}], relations=[]
                )
            except Exception:
                pass
            else:
                raise AssertionError(
                    "concept text must stay constrained by min_length=1"
                )

        outcome = await handler._extract_concept_relations(
            "src-l", _points(), llm=_NativeSeamLLM(payload)
        )

        assert outcome["status"] == "error", outcome
        assert outcome["status"] != "done", outcome
        assert "dropped=1" in (outcome.get("reason") or ""), outcome
        assert json.dumps(payload)[:120] in (outcome.get("reason") or ""), outcome
        handler._store_concept_relations.assert_not_awaited()
        print(
            "PASS (l) empty relation text -> dropped=1, never a schema failure"
        )

    async def test_m_native_partial_drop_is_error_not_done():
        """MINOR (e): a NATIVE payload with SOME items dropped (one storable
        concept + one whitespace-only concept) must be ``error`` with
        ``dropped>=1`` and the raw prefix, never ``done``."""
        _reset_fake()
        payload = {
            "concepts": [
                {"type": "DEFINITION", "text": "Algebra"},
                {"type": "DEFINITION", "text": "   "},
            ],
            "relations": [
                {
                    "type": "PREREQUISITE_FOR",
                    "origin": "Algebra",
                    "destination": "Calculus",
                    "text": "Algebra precedes Calculus",
                }
            ],
        }
        handler = _make_handler()

        outcome = await handler._extract_concept_relations(
            "src-m", _points(), llm=_NativeSeamLLM(payload)
        )

        assert outcome["status"] == "error", outcome
        assert outcome["status"] != "done", outcome
        assert "dropped=1" in (outcome.get("reason") or ""), outcome
        assert json.dumps(payload)[:120] in (outcome.get("reason") or ""), outcome
        handler._store_concept_relations.assert_not_awaited()
        print("PASS (m) native partial drop -> error with dropped=1, never done")

    async def test_n_timeout_is_not_ready_then_error():
        """A timeout is a "could not run" failure: ``not_ready`` on the first
        attempt, ``error`` afterwards — never the empty-extraction warning."""
        _reset_fake()
        _FakeStructuredLLM.behavior = lambda: (_ for _ in ()).throw(
            TimeoutError("deadline exceeded")
        )
        handler = _make_handler()

        with patch.object(graphrag_handler, "StructuredLLM", _FakeStructuredLLM):
            first = await handler._extract_concept_relations(
                "src-n", _points(), llm=object()
            )
            second = await handler._extract_concept_relations(
                "src-n", _points(), llm=object()
            )

        assert first["status"] == "not_ready", first
        assert first.get("warnings") is None, first
        assert second["status"] == "error", second
        assert "timed out" in (second.get("reason") or ""), second
        assert second.get("warnings") is None, second
        handler._store_concept_relations.assert_not_awaited()
        print("PASS (n) timeout -> not_ready then error, never a warning")

    async def test_o_spacy_zero_entities_is_done_with_warning():
        """The spaCy phase RAN and produced zero entities -> ``done`` with an
        ``empty-extraction`` warning and ``processed=0`` (never ``error``)."""
        handler = _make_handler()
        point = _Point("doc1", "Formula-only content with no named entities.")

        with patch.object(
            type(handler),
            "_drain_source_tasks_counted",
            new=AsyncMock(return_value=None),
        ), patch.object(
            type(handler),
            "get_source_points_by_collection",
            new=AsyncMock(return_value={"declarative": [point]}),
        ), patch.object(
            type(handler),
            "_create_similarity_relationships",
            new=AsyncMock(),
        ), patch.object(
            type(handler),
            "_extract_document_graph",
            new=AsyncMock(return_value=None),
        ):
            outcome = await handler.run_spacy_extraction_phase("src-spacy")

        assert outcome["status"] == "done", outcome
        assert outcome["status"] != "error", outcome
        assert outcome["processed"] == 0, outcome
        warnings = outcome.get("warnings") or []
        assert len(warnings) == 1, outcome
        assert warnings[0]["code"] == "empty-extraction", outcome
        message = warnings[0]["message"]
        assert message.startswith("no entities found in "), message
        assert "chars of 1 chunks" in message, message
        assert "'src-spacy'" in message, message
        print("PASS (o) spaCy zero entities -> done + empty-extraction warning")

    async def test_p_spacy_no_points_stays_not_ready():
        """The "could not run" case is unchanged: no stored points -> retry."""
        handler = _make_handler()

        with patch.object(
            type(handler),
            "_drain_source_tasks_counted",
            new=AsyncMock(return_value=None),
        ), patch.object(
            type(handler),
            "get_source_points_by_collection",
            new=AsyncMock(return_value={}),
        ):
            outcome = await handler.run_spacy_extraction_phase("src-empty")

        assert outcome == {"status": "not_ready", "retry_after": 5}, outcome
        print("PASS (p) spaCy no points -> not_ready (unchanged)")

    async def test_q_card_binary_segments_alone_are_skipped_input():
        """(a) A source whose store order puts binary CATALOG card segments
        first and one prose chunk last must extract from the PROSE only: the
        binary bytes never reach the LLM and the result is a real product."""
        _reset_fake()
        product = {
            "concepts": [{"type": "CONCEPT", "text": "Algebra"}],
            "relations": [
                {
                    "type": "IS_A",
                    "origin": "Algebra",
                    "destination": "Mathematics",
                    "text": "Algebra is a branch of mathematics",
                }
            ],
        }
        _FakeStructuredLLM.behavior = lambda: _Result(product, json.dumps(product))
        handler = _make_handler()
        card = _meta_point(
            "card", _BINARY_SEGMENT,
            is_catalogue_card=True,
            is_catalogue_card_segment=True,
            chunk_index=0,
        )
        prose = _meta_point(
            "prose",
            "Algebra is a branch of mathematics that studies symbols.",
            is_catalogue_card=False,
            chunk_index=1,
        )

        with patch.object(
            type(handler), "_drain_source_tasks_counted", new=AsyncMock()
        ), patch.object(
            type(handler),
            "get_source_points_by_collection",
            new=AsyncMock(return_value={"declarative": [card, prose]}),
        ), patch.object(
            type(handler),
            "_resolve_agent_llm",
            new=AsyncMock(return_value=object()),
        ), patch.object(graphrag_handler, "StructuredLLM", _FakeStructuredLLM):
            outcome = await handler.run_llm_extraction_phase("src-q")

        assert outcome["status"] == "done", outcome
        assert outcome["processed"] == 1, outcome
        prompt = _FakeStructuredLLM.prompts[0]
        assert "Algebra is a branch of mathematics that studies symbols." in prompt
        assert "%PDF" not in prompt, prompt
        assert "catalogue-card bytes" not in prompt, prompt
        handler._store_concept_relations.assert_awaited_once()
        stored = handler._store_concept_relations.await_args.args[1]
        assert len(stored["concepts"]) == 1, stored
        print("PASS (q) binary card segments never reach the extractor")

    async def test_r_only_binary_points_is_done_with_empty_warning():
        """(b) A source made ONLY of binary/card points stays a completed
        extraction with an ``empty-extraction`` warning: the LLM is never
        called on unreadable bytes, and it is not a retryable failure."""
        _reset_fake()
        handler = _make_handler()
        points = [
            _meta_point(
                "card1", _BINARY_SEGMENT,
                is_catalogue_card=True,
                is_catalogue_card_segment=True,
                chunk_index=0,
            ),
            _meta_point(
                "card2", "%PDF-1.5\n\x00\x00 more binary",
                is_catalogue_card=True,
                chunk_index=1,
            ),
        ]

        with patch.object(
            type(handler), "_drain_source_tasks_counted", new=AsyncMock()
        ), patch.object(
            type(handler),
            "get_source_points_by_collection",
            new=AsyncMock(return_value={"declarative": points}),
        ), patch.object(
            type(handler),
            "_resolve_agent_llm",
            new=AsyncMock(return_value=object()),
        ), patch.object(graphrag_handler, "StructuredLLM", _FakeStructuredLLM):
            outcome = await handler.run_llm_extraction_phase("src-r")

        assert outcome["status"] == "done", outcome
        assert outcome["processed"] == 0, outcome
        assert _FakeStructuredLLM.calls == 0, _FakeStructuredLLM.calls
        warnings = outcome.get("warnings") or []
        assert len(warnings) == 1, outcome
        assert warnings[0]["code"] == "empty-extraction", outcome
        assert "src-r" in warnings[0]["message"], warnings[0]
        handler._store_concept_relations.assert_not_awaited()
        print("PASS (r) only binary points -> done + empty-extraction warning")

    async def test_s_extraction_input_is_ordered_by_start_index():
        """(c) Survivors are concatenated in ``start_index`` ascending order;
        ``start_index`` wins over ``chunk_index`` when both are present."""
        _reset_fake()
        product = {
            "concepts": [{"type": "CONCEPT", "text": "Algebra"}],
            "relations": [
                {
                    "type": "IS_A",
                    "origin": "Algebra",
                    "destination": "Mathematics",
                    "text": "Algebra is a branch of mathematics",
                }
            ],
        }
        _FakeStructuredLLM.behavior = lambda: _Result(product, json.dumps(product))
        handler = _make_handler()
        points = [
            _meta_point("p30", "THIRTY comes last.", start_index=30, chunk_index=0),
            _meta_point("p5", "FIVE comes first.", start_index=5, chunk_index=999),
            _meta_point("p20", "TWENTY comes second.", start_index=20, chunk_index=1),
            _meta_point("p10", "TEN comes third.", start_index=10, chunk_index=2),
        ]

        with patch.object(graphrag_handler, "StructuredLLM", _FakeStructuredLLM):
            outcome = await handler._extract_concept_relations(
                "src-s", points, llm=object()
            )

        assert outcome["status"] == "done", outcome
        prompt = _FakeStructuredLLM.prompts[0]
        assert (
            prompt.index("FIVE comes first.")
            < prompt.index("TEN comes third.")
            < prompt.index("TWENTY comes second.")
            < prompt.index("THIRTY comes last.")
        ), prompt
        print("PASS (s) extraction input ordered by start_index")

    async def test_t_card_flagged_textual_point_is_still_used():
        """(d) The CATALOG metadata leak marks a real prose chunk as a card:
        content wins, so a clearly textual card-flagged point is still input."""
        _reset_fake()
        product = {
            "concepts": [{"type": "CONCEPT", "text": "Algebra"}],
            "relations": [
                {
                    "type": "IS_A",
                    "origin": "Algebra",
                    "destination": "Mathematics",
                    "text": "Algebra is a branch of mathematics",
                }
            ],
        }
        _FakeStructuredLLM.behavior = lambda: _Result(product, json.dumps(product))
        handler = _make_handler()
        leaked = _meta_point(
            "leaked",
            "Real prose about algebra that drives the concept graph.",
            is_catalogue_card=True,
            is_catalogue_card_segment=True,
            start_index=0,
        )

        with patch.object(graphrag_handler, "StructuredLLM", _FakeStructuredLLM):
            outcome = await handler._extract_concept_relations(
                "src-t", [leaked], llm=object()
            )

        assert outcome["status"] == "done", outcome
        prompt = _FakeStructuredLLM.prompts[0]
        assert "Real prose about algebra" in prompt, prompt
        handler._store_concept_relations.assert_awaited_once()
        print("PASS (t) card-flagged but clearly textual point is still used")

    async def test_u_readable_prose_with_leaked_full_card_is_kept():
        """(u/a) A readable prose chunk carrying leaked ``full_card`` +
        ``is_catalogue_card`` metadata IS kept: flags never veto content."""
        handler = _make_handler()
        leaked = _meta_point(
            "leaked",
            "Questa e una frase di prosa italiana leggibile, non una card.",
            full_card="# CATALOGUE CARD\n## Initial page\nvecchia card",
            is_catalogue_card=True,
            is_catalogue_card_segment=True,
        )

        selected = handler._select_extraction_text_points([leaked])

        assert [p.id for p in selected] == ["leaked"], [p.id for p in selected]
        print("PASS (u) readable prose with leaked full_card is kept")

    async def test_v_card_header_content_is_dropped():
        """(b) A real card segment whose content starts with the catalogue-card
        header is dropped, even without any flag."""
        handler = _make_handler()
        card = _meta_point(
            "card", "# CATALOGUE CARD\n## Initial page\nCARD BODY"
        )
        prose = _meta_point("prose", "Real prose survives the filter.")

        selected = handler._select_extraction_text_points([card, prose])

        assert [p.id for p in selected] == ["prose"], [p.id for p in selected]
        print("PASS (v) card-header content is dropped")

    async def test_w_binary_point_is_dropped():
        """(c) A binary-looking point is dropped regardless of its flags."""
        handler = _make_handler()
        binary = _meta_point(
            "binary",
            "%PDF-1.5\n\x00\x01 binary",
            is_catalogue_card=False,
        )
        prose = _meta_point("prose", "Real prose survives the filter.")

        selected = handler._select_extraction_text_points([binary, prose])

        assert [p.id for p in selected] == ["prose"], [p.id for p in selected]
        print("PASS (w) binary point is dropped")

    async def test_x_image_point_is_dropped():
        """(d) An image point is dropped: its content is only a placeholder."""
        handler = _make_handler()
        image = _meta_point("img", "[Image] scan.pdf", image=True)
        prose = _meta_point("prose", "Real prose survives the filter.")

        selected = handler._select_extraction_text_points([image, prose])

        assert [p.id for p in selected] == ["prose"], [p.id for p in selected]
        print("PASS (x) image point is dropped")

    async def test_y_empty_after_drops_is_done_with_empty_extraction():
        """(e) When every point is unusable the extraction is still ``done``
        with an ``empty-extraction`` warning (not a retryable failure)."""
        _reset_fake()
        handler = _make_handler()
        points = [
            _meta_point("img", "[Image] scan.pdf", image=True),
            _meta_point("binary", "%PDF-1.5\n\x00\x01 binary"),
            _meta_point("card", "# CATALOGUE CARD\nCARD BODY"),
        ]

        with patch.object(graphrag_handler, "StructuredLLM", _FakeStructuredLLM):
            outcome = await handler._extract_concept_relations(
                "src-empty", points, llm=object()
            )

        assert outcome["status"] == "done", outcome
        assert outcome["processed"] == 0, outcome
        assert _FakeStructuredLLM.calls == 0, _FakeStructuredLLM.calls
        warnings = outcome.get("warnings") or []
        assert len(warnings) == 1, outcome
        assert warnings[0]["code"] == "empty-extraction", outcome
        print("PASS (y) empty-after-drops -> done + empty-extraction")

    async def _main():
        await test_a_structured_error_is_never_done()
        await test_b_no_llm_is_configuration_error()
        await test_c_valid_empty_is_done_with_warning()
        await test_d_clean_extraction_is_done_with_product()
        await test_e_partial_extraction_reports_dropped_count()
        await test_f_native_empty_product_is_done_with_warning()
        await test_g_native_empty_strings_are_rejected()
        await test_h_non_list_items_count_as_dropped()
        await test_i_disabled_feature_is_skipped_not_done()
        await test_j_native_clean_product_is_done()
        await test_k_non_iterable_container_is_dropped_not_raised()
        await test_l_empty_relation_text_is_dropped_not_schema_failure()
        await test_m_native_partial_drop_is_error_not_done()
        await test_n_timeout_is_not_ready_then_error()
        await test_o_spacy_zero_entities_is_done_with_warning()
        await test_p_spacy_no_points_stays_not_ready()
        await test_q_card_binary_segments_alone_are_skipped_input()
        await test_r_only_binary_points_is_done_with_empty_warning()
        await test_s_extraction_input_is_ordered_by_start_index()
        await test_t_card_flagged_textual_point_is_still_used()
        await test_u_readable_prose_with_leaked_full_card_is_kept()
        await test_v_card_header_content_is_dropped()
        await test_w_binary_point_is_dropped()
        await test_x_image_point_is_dropped()
        await test_y_empty_after_drops_is_done_with_empty_extraction()

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    print("OK LLM error mapping (twenty-five tests, native path included)")
