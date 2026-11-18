"""The graph never ingests card/image text, and the legacy staging pass builds no graph.

Standalone, import-safe script (the host imports every ``.py`` under the plugin): the ENTIRE body lives inside
``if __name__ == "__main__":`` so importing this file executes nothing. It stubs the external services the plugin
imports (no real Cat/Neo4j/spaCy is loaded) and prints ``PASS <test>`` lines.

Verified here:
- ``_is_empty_mapping`` recognises ONLY the legacy ``vector={}`` empty mapping (a zero-norm list is not one);
- ``_is_empty_vector_staging_pass`` requires BOTH ``vector={}`` AND the recorded ``parsing_chunking`` phase;
- ``_select_extraction_text_points`` drops blank, image, binary and catalogue-card-header points while KEEPING
  real prose that merely inherited catalogue-card metadata (content wins).
"""


if __name__ == "__main__":
    import importlib
    import os
    import sys
    import types

    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PKG = "catgraphrag_content_gating"

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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

        class _BaseHandler:
            def __init__(self, **kwargs):
                pass

        cat_mod.BaseVectorDatabaseHandler = _BaseHandler
        cat_mod.log = _StubLog()
        sys.modules["cat"] = cat_mod

        log_mod = types.ModuleType("cat.log")
        log_mod.log = _StubLog()
        sys.modules["cat.log"] = log_mod

        services = types.ModuleType("cat.services")
        services.__path__ = []
        sys.modules["cat.services"] = services

        memory = types.ModuleType("cat.services.memory")
        memory.__path__ = []
        sys.modules["cat.services.memory"] = memory

        models_mod = types.ModuleType("cat.services.memory.models")
        models_mod.PointStruct = type("PointStruct", (), {})
        models_mod.DocumentRecall = type("DocumentRecall", (), {})
        models_mod.Record = type("Record", (), {})
        models_mod.ScoredPoint = type("ScoredPoint", (), {})
        models_mod.UpdateResult = type("UpdateResult", (), {})
        from graph_fakes import StubVectorMemoryType

        models_mod.VectorMemoryType = StubVectorMemoryType
        sys.modules["cat.services.memory.models"] = models_mod

        from graph_fakes import install_ingestion_registry_stub

        install_ingestion_registry_stub()


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


    def _install_langdetect_stub():
        mod = types.ModuleType("langdetect")

        class DetectorFactory:
            seed = 0

        mod.DetectorFactory = DetectorFactory
        mod.detect_langs = lambda text: []
        sys.modules["langdetect"] = mod


    def _install_spacy_stubs():
        def _forbidden(*args, **kwargs):
            raise AssertionError("real spaCy must not be loaded inside the standalone test")

        spacy_mod = types.ModuleType("spacy")
        spacy_mod.__path__ = []
        spacy_mod.load = _forbidden
        sys.modules["spacy"] = spacy_mod
        util_mod = types.ModuleType("spacy.util")
        util_mod.is_package = lambda name: True
        sys.modules["spacy.util"] = util_mod
        cli_mod = types.ModuleType("spacy.cli")
        cli_mod.__path__ = []
        sys.modules["spacy.cli"] = cli_mod
        download_mod = types.ModuleType("spacy.cli.download")
        download_mod.download = _forbidden
        sys.modules["spacy.cli.download"] = download_mod
        language_mod = types.ModuleType("spacy.language")
        language_mod.Language = type("Language", (), {})
        sys.modules["spacy.language"] = language_mod
        tokens_mod = types.ModuleType("spacy.tokens")
        tokens_mod.Doc = type("Doc", (), {})
        sys.modules["spacy.tokens"] = tokens_mod


    def _install_pydantic_stub():
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

        pydantic_mod.Field = Field
        pydantic_mod.ConfigDict = ConfigDict
        pydantic_mod.BaseModel = BaseModel
        pydantic_mod.BeforeValidator = BeforeValidator
        pydantic_mod.create_model = create_model
        sys.modules["pydantic"] = pydantic_mod


    def _install_stubs():
        _install_cat_stub()
        _install_neo4j_stub()
        _install_langchain_stub()
        _install_langdetect_stub()
        _install_spacy_stubs()
        _install_pydantic_stub()


    def _load_handler_module():
        module = sys.modules.get(PKG)
        if module is None:
            package = types.ModuleType(PKG)
            package.__path__ = [REPO_ROOT]
            sys.modules[PKG] = package
        return importlib.import_module(PKG + ".graphrag_handler")


    class _Point:
        def __init__(self, pid, content, metadata=None):
            self.id = pid
            self.payload = {"page_content": content, "metadata": metadata or {}}


    def _handler():
        handler_mod = _load_handler_module()
        handler = handler_mod.GraphRAGHandler(
            neo4j_uri="bolt://fake",
            neo4j_user="u",
            neo4j_password="p",
            enable_entity_extraction=True,
            enable_concept_relations=True,
        )
        handler.agent_id = "agent_test"
        return handler


    def test_empty_mapping_is_only_the_empty_dict():
        handler = _handler()
        assert handler._is_empty_mapping({}) is True
        assert handler._is_empty_mapping([]) is False, "a zero-norm list is not the legacy empty mapping"
        assert handler._is_empty_mapping([0.0, 0.0]) is False
        assert handler._is_empty_mapping(None) is False
        print("PASS empty mapping is only the empty dict")


    def test_staging_pass_requires_the_recorded_parsing_phase():
        import asyncio

        handler = _handler()
        ingestions = sys.modules["cat.db.cruds.ingestions"]

        async def _parsing(agent_id, chat_id, source):
            return {"phase": "parsing_chunking"}

        async def _embedding(agent_id, chat_id, source):
            return {"phase": "embedding"}

        async def _no_row(agent_id, chat_id, source):
            return None

        ingestions.get_status = _parsing
        assert asyncio.run(handler._is_empty_vector_staging_pass("f.pdf", {"chat_id": None}, {})) is True

        ingestions.get_status = _embedding
        assert asyncio.run(handler._is_empty_vector_staging_pass("f.pdf", {"chat_id": None}, {})) is False

        ingestions.get_status = _no_row
        assert asyncio.run(handler._is_empty_vector_staging_pass("f.pdf", {"chat_id": None}, {})) is False

        ingestions.get_status = _parsing
        assert asyncio.run(
            handler._is_empty_vector_staging_pass("f.pdf", {"chat_id": None}, [0.1, 0.2])
        ) is False, "a real vector is never a staging pass, even on the parsing phase"
        print("PASS staging pass requires empty mapping AND parsing_chunking")


    def test_extraction_selection_drops_cards_images_and_blank_text():
        handler = _handler()
        points = [
            _Point("prose", "Real course content about algorithms."),
            _Point("card", "# CATALOGUE CARD\nTitle: x"),
            _Point("image", "[Image] a diagram", {"image": True}),
            _Point("leaked-flag", "Still real prose, despite leaked metadata.", {"is_catalogue_card": True}),
            _Point("blank", "   "),
            _Point("binary", "%PDF-1.7\x00\x00"),
        ]
        kept = handler._select_extraction_text_points(points)
        kept_ids = [p.id for p in kept]
        assert "card" not in kept_ids, kept_ids
        assert "image" not in kept_ids, kept_ids
        assert "blank" not in kept_ids, kept_ids
        assert "binary" not in kept_ids, kept_ids
        assert "prose" in kept_ids, kept_ids
        assert "leaked-flag" in kept_ids, "the card FLAG alone must never veto real prose"
        print("PASS extraction selection drops cards/images/blank/binary, keeps real prose")


    def test_card_header_detection():
        handler = _handler()
        assert handler._looks_like_catalogue_card_content("# CATALOGUE CARD\nx") is True
        assert handler._looks_like_catalogue_card_content("  # CATALOGUE CARD\nx") is True
        assert handler._looks_like_catalogue_card_content("A normal paragraph") is False
        print("PASS catalogue-card header detection")


    def main():
        _install_stubs()
        test_empty_mapping_is_only_the_empty_dict()
        test_staging_pass_requires_the_recorded_parsing_phase()
        test_extraction_selection_drops_cards_images_and_blank_text()
        test_card_header_detection()
        print("ALL CONTENT-GATING TESTS PASSED")


    main()
