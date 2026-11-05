"""Standalone verification that ``enable_knowledge_graph`` is display-only.

Runnable with:  python tests/test_kg_master_switch.py

Pure-stdlib, plain asserts, no pytest and no model loading. The plugin module
(``graphrag_handler``) is imported through an in-memory package whose
``__path__`` points at this repository root (bypassing the plugin
``__init__.py`` side effects), while external dependencies (cat, langdetect,
spacy, pydantic, neo4j, langchain_core) are stubbed in ``sys.modules`` *before*
the import. Neo4j is mocked with a permissive in-memory fake session that
accepts every query (the tests only care about the LLM-extraction gate, not
about the derived-graph Cypher).

What is verified (``enable_knowledge_graph`` no longer gates concept writes):
- ``create_derived_graph_for_source`` step 8 DOES call
  ``_extract_concept_relations`` when ``enable_knowledge_graph=False`` but
  ``enable_concept_relations=True`` (the old gate wrote nothing);
- the same holds on the vector-db path (no ``stray_cat``): the agent LLM is
  resolved via ``_resolve_agent_llm`` and the extraction is called;
- ``concept_relations=False`` writes nothing, regardless of
  ``enable_knowledge_graph``;
- ``recompute_concept_relations`` runs with ``enable_knowledge_graph=False`` and
  returns early only when ``enable_concept_relations=False``;
- ``enable_student_knowledge_graph`` changes nothing in either direction;
- points WITHOUT ``chunk_index`` (e.g. the efficient_ingestion engine) fall
  back to arrival order, so step 8 still reaches the LLM call; the CATALOG card
  stays excluded.
"""

import asyncio
import os
import sys
import types
from unittest.mock import AsyncMock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The plugin folder name contains a dot/hyphen, so the scripts register the
# repo root under this synthetic package and import the plugin modules from
# it: a bare ``python tests/<file>.py`` run then resolves the same modules the
# Cat loader would, without a relative import from ``__main__``.
PKG = "catgraphrag_kgtest"


def _plugin(name):
    import importlib

    return importlib.import_module(f"{PKG}.{name}")



# ---------------------------------------------------------------------------
# Minimal stubs for the external packages imported by the plugin modules.
# Installed into sys.modules BEFORE importing the plugin modules.
# ---------------------------------------------------------------------------


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
    cat_mod.__path__ = []  # look like a package root for submodule imports

    # @hook decorator: returns the function unchanged.
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


def _install_langdetect_stub():
    langdetect_mod = types.ModuleType("langdetect")

    class DetectorFactory:
        seed = 0

    setattr(langdetect_mod, "DetectorFactory", DetectorFactory)
    setattr(langdetect_mod, "detect_langs", lambda text: [])
    sys.modules["langdetect"] = langdetect_mod


def _install_spacy_stubs():
    def _forbidden(*args, **kwargs):
        raise AssertionError("real spaCy must not be loaded inside the standalone test")

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
        # Identity is enough for the stub: the real one wraps a validator fn.
        return fn

    def create_model(name, **kwargs):
        return type(name, (BaseModel,), {})

    setattr(pydantic_mod, "Field", Field)
    setattr(pydantic_mod, "ConfigDict", ConfigDict)
    setattr(pydantic_mod, "BaseModel", BaseModel)
    setattr(pydantic_mod, "BeforeValidator", BeforeValidator)
    setattr(pydantic_mod, "create_model", create_model)
    sys.modules["pydantic"] = pydantic_mod


def _install_stubs():
    _install_cat_stub()
    _install_neo4j_stub()
    _install_langchain_stub()
    _install_langdetect_stub()
    _install_spacy_stubs()
    _install_pydantic_stub()


# ---------------------------------------------------------------------------
# Permissive in-memory fake Neo4j: accepts every query, returns empty results.
# The tests only assert on the LLM-extraction gate, never on Cypher output.
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, records):
        self._records = list(records)

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for r in self._records:
            yield r


class _FakeSession:
    def __init__(self, records=()):
        self.queries = []
        self._records = list(records)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run(self, query, **params):
        self.queries.append((query, params))
        return _FakeResult(self._records)


class _FakeDriver:
    def session(self, database=None):
        return _FakeSession()


class _Point:
    """Minimal stand-in for PointStruct: id + payload with metadata."""

    def __init__(self, pid, chunk_index=0, chunk_level=None, parent_id=None,
                 has_formula=False, is_catalogue_card=False):
        self.id = pid
        self.payload = {
            "metadata": {
                "chunk_index": chunk_index,
                "chunk_level": chunk_level,
                "parent_id": parent_id,
                "has_formula": has_formula,
                "is_catalogue_card": is_catalogue_card,
            }
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(kg: bool, cr: bool, student: bool = False):
    """Build a GraphRAGHandler with the given concept-extraction flags."""
    graphrag_handler = _plugin("graphrag_handler")

    handler = graphrag_handler.GraphRAGHandler(
        neo4j_uri="bolt://fake",
        neo4j_user="u",
        neo4j_password="p",
        enable_knowledge_graph=kg,
        enable_concept_relations=cr,
        enable_student_knowledge_graph=student,
    )
    handler._driver = _FakeDriver()
    handler.agent_id = "agent_test"
    return handler


def _stored_points():
    return [
        _Point("doc1", chunk_index=0, chunk_level="section"),
        _Point("doc2", chunk_index=1, chunk_level="paragraph"),
    ]


def _stored_points_no_chunk_index():
    """Points as produced by efficient_ingestion: no chunk_index in metadata."""
    return [
        _Point("doc1", chunk_index=None, chunk_level="section"),
        _Point("doc2", chunk_index=None, chunk_level="paragraph"),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_derived_graph_calls_llm_when_kg_off_cr_on():
    """Display flag off + CR on: step 8 MUST call _extract_concept_relations."""
    handler = _make_handler(kg=False, cr=True)

    with patch.object(
        type(handler), "_extract_concept_relations", new=AsyncMock()
    ) as extract:
        asyncio.run(handler.create_derived_graph_for_source(
            "file.pdf", _stored_points(), stray_cat=object()
        ))

    extract.assert_awaited_once()


def test_derived_graph_calls_llm_when_kg_and_cr_on():
    """Display flag on + CR on: step 8 MUST call _extract_concept_relations."""
    handler = _make_handler(kg=True, cr=True)

    with patch.object(
        type(handler), "_extract_concept_relations", new=AsyncMock()
    ) as extract:
        asyncio.run(handler.create_derived_graph_for_source(
            "file.pdf", _stored_points(), stray_cat=object()
        ))

    extract.assert_awaited_once()


def test_derived_graph_skips_llm_when_cr_off():
    """CR off: no extraction, regardless of the display flag."""
    for kg in (False, True):
        handler = _make_handler(kg=kg, cr=False)

        with patch.object(
            type(handler), "_extract_concept_relations", new=AsyncMock()
        ) as extract:
            asyncio.run(handler.create_derived_graph_for_source(
                "file.pdf", _stored_points(), stray_cat=object()
            ))

        extract.assert_not_awaited()


def test_derived_graph_calls_llm_without_chunk_index():
    """No chunk_index (efficient_ingestion) + CR on (display flag off): the
    arrival-order fallback must still reach step 8 and call
    _extract_concept_relations; the CATALOG card stays excluded."""
    handler = _make_handler(kg=False, cr=True)
    session = _FakeSession()
    handler._get_session = lambda: session

    points = [
        _Point("doc1", chunk_index=None, chunk_level="section"),
        _Point("doc2", chunk_index=None, chunk_level="paragraph"),
        _Point("card1", chunk_index=None, is_catalogue_card=True),
    ]

    with patch.object(
        type(handler), "_extract_concept_relations", new=AsyncMock()
    ) as extract:
        asyncio.run(handler.create_derived_graph_for_source(
            "file.pdf", points, stray_cat=object()
        ))

    extract.assert_awaited_once()

    # The CATALOG card must not be part of the derived graph (PART_OF links).
    part_of_params = [params for q, params in session.queries if "PART_OF" in q]
    assert part_of_params, "expected a PART_OF query"
    assert "card1" not in part_of_params[0]["point_ids"]


def test_derived_graph_skips_llm_without_chunk_index_when_cr_off():
    """No chunk_index + CR off: the fallback must NOT call
    _extract_concept_relations."""
    handler = _make_handler(kg=True, cr=False)

    with patch.object(
        type(handler), "_extract_concept_relations", new=AsyncMock()
    ) as extract:
        asyncio.run(handler.create_derived_graph_for_source(
            "file.pdf", _stored_points_no_chunk_index(), stray_cat=object()
        ))

    extract.assert_not_awaited()


def test_recompute_runs_when_kg_off():
    """Display flag off + CR on: recompute_concept_relations must NOT return
    early; it extracts from the tenant's stored documents."""
    handler = _make_handler(kg=False, cr=True)
    doc = {
        "id": "doc1",
        "content": "Some course content about algorithms.",
        "metadata": '{"source": "file.pdf"}',
    }
    handler._get_session = lambda: _FakeSession([doc])

    with patch.object(
        type(handler), "_extract_concept_relations", new=AsyncMock()
    ) as extract:
        result = asyncio.run(handler.recompute_concept_relations(stray_cat=object()))

    assert result is None
    extract.assert_awaited_once()


def test_recompute_returns_early_when_cr_off():
    """CR off: recompute_concept_relations must return before any LLM call."""
    for kg in (False, True):
        handler = _make_handler(kg=kg, cr=False)
        handler._get_session = lambda: _FakeSession([
            {"id": "doc1", "content": "x", "metadata": '{"source": "file.pdf"}'}
        ])

        with patch.object(
            type(handler), "_extract_concept_relations", new=AsyncMock()
        ) as extract:
            result = asyncio.run(handler.recompute_concept_relations(stray_cat=object()))

        assert result is None
        extract.assert_not_awaited()


class _FakeLLM:
    """Minimal stand-in for the agent LLM returned by _resolve_agent_llm."""

    def __init__(self, name="fake-llm"):
        self.name = name


def test_derived_graph_calls_llm_without_stray_cat():
    """Display flag off + CR on, NO stray_cat (vector-db path, e.g.
    add_points_to_tenant): step 8 must resolve the agent LLM via
    _resolve_agent_llm and call _extract_concept_relations."""
    handler = _make_handler(kg=False, cr=True)
    fake_llm = _FakeLLM()

    with patch.object(
        type(handler), "_extract_concept_relations", new=AsyncMock()
    ) as extract, patch.object(
        type(handler), "_resolve_agent_llm", new=AsyncMock(return_value=fake_llm)
    ) as resolve:
        asyncio.run(handler.create_derived_graph_for_source(
            "file.pdf", _stored_points()
        ))

    resolve.assert_awaited_once()
    extract.assert_awaited_once()
    # the resolved LLM must flow into the extraction call
    _, kwargs = extract.await_args
    assert kwargs.get("llm") is fake_llm


def test_derived_graph_skips_llm_without_stray_cat_when_cr_off():
    """CR off, NO stray_cat: step 8 must NOT resolve the LLM and must NOT call
    _extract_concept_relations."""
    handler = _make_handler(kg=True, cr=False)

    with patch.object(
        type(handler), "_extract_concept_relations", new=AsyncMock()
    ) as extract, patch.object(
        type(handler), "_resolve_agent_llm", new=AsyncMock(return_value=_FakeLLM())
    ) as resolve:
        asyncio.run(handler.create_derived_graph_for_source(
            "file.pdf", _stored_points()
        ))

    resolve.assert_not_awaited()
    extract.assert_not_awaited()


def test_student_flag_changes_nothing():
    """enable_student_knowledge_graph is display-only: it must not affect the
    concept-write gate in either direction."""
    for student in (False, True):
        # CR on: extraction still runs with either student value.
        handler = _make_handler(kg=False, cr=True, student=student)
        with patch.object(
            type(handler), "_extract_concept_relations", new=AsyncMock()
        ) as extract:
            asyncio.run(handler.create_derived_graph_for_source(
                "file.pdf", _stored_points(), stray_cat=object()
            ))
        extract.assert_awaited_once()

        # CR off: extraction still skipped with either student value.
        handler = _make_handler(kg=True, cr=False, student=student)
        with patch.object(
            type(handler), "_extract_concept_relations", new=AsyncMock()
        ) as extract:
            asyncio.run(handler.create_derived_graph_for_source(
                "file.pdf", _stored_points(), stray_cat=object()
            ))
        extract.assert_not_awaited()


def main():
    _install_stubs()

    # In-memory package pointing at the repo root: imports the plugin modules
    # directly (relative imports resolved via __path__) without executing the
    # plugin __init__.py side effects.
    _pkg = types.ModuleType("catgraphrag_kgtest")
    _pkg.__path__ = [REPO_ROOT]
    sys.modules["catgraphrag_kgtest"] = _pkg

    global graphrag_handler
    graphrag_handler = _plugin("graphrag_handler")  # noqa: E402

    tests = [
        test_derived_graph_calls_llm_when_kg_off_cr_on,
        test_derived_graph_calls_llm_when_kg_and_cr_on,
        test_derived_graph_skips_llm_when_cr_off,
        test_derived_graph_calls_llm_without_chunk_index,
        test_derived_graph_skips_llm_without_chunk_index_when_cr_off,
        test_recompute_runs_when_kg_off,
        test_recompute_returns_early_when_cr_off,
        test_derived_graph_calls_llm_without_stray_cat,
        test_derived_graph_skips_llm_without_stray_cat_when_cr_off,
        test_student_flag_changes_nothing,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()