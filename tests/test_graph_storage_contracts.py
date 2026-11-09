"""Standalone verification of the graph storage contracts that outlive any ingestion engine.

- ``after_vector_memory_transfer_on_agent`` wipes the tenant's Neo4j data
  ONLY on success when the NEW handler is not GraphRAG (the host does not pass
  the old handler name), rebuilds the graph of every source when it IS
  GraphRAG, and does nothing on failure;
- ``wipe_tenant_graph_data`` skips without a recorded config and wipes the
  tenant (tenant-scoped) when one exists;
- re-running a write (concepts, similarity, derived structure) issues the
  same idempotent MERGE queries;
- the spaCy entity batch write keeps ``REMOVE e.concept_gen,
  e.concept_gen_active`` — the shared Entity-node contract stays internal
  to the plugin.

The graph phases themselves run on the core's engine in
``test_graph_phases.py``.

Runnable:  python tests/test_graph_storage_contracts.py
Pure-stdlib + unittest.mock, plain asserts, no pytest, no model loading. All
external packages (cat, langchain_core, spacy, langdetect, pydantic, neo4j)
are stubbed in sys.modules *before* the plugin modules are imported (same
pattern as test_kg_master_switch.py).
"""

import asyncio
import os
import sys
import types
from unittest.mock import AsyncMock, Mock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



class _StubLog:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    warnings = []

    def warning(self, *args, **kwargs):
        type(self).warnings.append(" ".join(str(a) for a in args))

    def error(self, *args, **kwargs):
        pass


def _install_cat_stub():
    cat_mod = types.ModuleType("cat")
    cat_mod.__path__ = []  # look like a package root for submodule imports

    # @hook decorator: supports both bare ``@hook`` and ``@hook(priority=...)``;
    # returns the function unchanged either way.
    def _hook(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

    cat_mod.hook = _hook
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
    models_mod.PointStruct = type("PointStruct", (), {})
    models_mod.DocumentRecall = type("DocumentRecall", (), {})
    models_mod.Record = type("Record", (), {})
    models_mod.ScoredPoint = type("ScoredPoint", (), {})
    models_mod.UpdateResult = type("UpdateResult", (), {})

    # the core's enum: one copy for every harness (graph_fakes.py), checked against the core
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from graph_fakes import StubVectorMemoryType

    models_mod.VectorMemoryType = StubVectorMemoryType
    sys.modules["cat.services.memory.models"] = models_mod

    factory = types.ModuleType("cat.services.factory")
    factory.__path__ = []
    sys.modules["cat.services.factory"] = factory



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
# Fakes
# ---------------------------------------------------------------------------


class _FakeCat:
    """Minimal stand-in for CheshireCat: agent key and vector handler."""

    def __init__(self, handler):
        self.vector_memory_handler = handler
        self.agent_key = "agent_test"


class _FakePoint:
    def __init__(self, pid, vector=None, payload=None):
        self.id = pid
        self.vector = vector
        self.payload = payload or {}


def _make_handler(kg=True, cr=True):
    """Build a GraphRAGHandler with the given flags (no driver needed)."""
    from catgraphrag_phase import graphrag_handler

    handler = graphrag_handler.GraphRAGHandler(
        neo4j_uri="bolt://fake",
        neo4j_user="u",
        neo4j_password="p",
        enable_knowledge_graph=kg,
        enable_concept_relations=cr,
    )
    handler.agent_id = "agent_test"
    return handler




# ---------------------------------------------------------------------------
# T24 — old-store cleanup on switch
# ---------------------------------------------------------------------------


def test_transfer_cleanup_noop_on_failure():
    from catgraphrag_phase import main

    cat = _FakeCat(_make_handler())
    with patch("catgraphrag_phase.graphrag_handler.wipe_tenant_graph_data", new=AsyncMock()) as wipe:
        asyncio.run(main.after_vector_memory_transfer_on_agent(False, cat))
        wipe.assert_not_awaited()


def test_transfer_to_graphrag_rebuilds_instead_of_wiping():
    from catgraphrag_phase import main

    cat = _FakeCat(_make_handler())
    with patch("catgraphrag_phase.graphrag_handler.wipe_tenant_graph_data", new=AsyncMock()) as wipe, \
         patch.object(main, "_rebuild_graph_after_transfer", new=AsyncMock()) as rebuild:
        asyncio.run(main.after_vector_memory_transfer_on_agent(True, cat))
        wipe.assert_not_awaited()
        rebuild.assert_awaited_once_with(cat)


def test_transfer_cleanup_wipes_on_success_when_new_handler_not_graphrag():
    from catgraphrag_phase import main

    cat = _FakeCat(None)
    with patch("catgraphrag_phase.graphrag_handler.wipe_tenant_graph_data", new=AsyncMock()) as wipe:
        asyncio.run(main.after_vector_memory_transfer_on_agent(True, cat))
        wipe.assert_awaited_once_with("agent_test")


def test_wipe_tenant_graph_data_skips_without_config():
    from catgraphrag_phase import graphrag_handler

    graphrag_handler._last_graphrag_config.pop("agent_test", None)
    with patch.object(graphrag_handler.GraphRAGHandler, "_ensure_connected", new=AsyncMock()) as conn:
        asyncio.run(graphrag_handler.wipe_tenant_graph_data("agent_test"))
        conn.assert_not_awaited()


def test_wipe_tenant_graph_data_wipes_tenant():
    from catgraphrag_phase import graphrag_handler

    graphrag_handler._last_graphrag_config["agent_test"] = {
        "neo4j_uri": "bolt://fake",
        "neo4j_user": "u",
        "neo4j_password": "p",
        "neo4j_database": "neo4j",
        "neo4j_kwargs": {},
    }

    class _FakeSessionCM:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *a):
            return False

    captured = {}

    async def _fake_drop(self, session):
        captured["handler"] = self

    try:
        with patch.object(graphrag_handler.GraphRAGHandler, "_ensure_connected", new=AsyncMock()) as conn, \
             patch.object(graphrag_handler.GraphRAGHandler, "_get_session",
                          new=Mock(return_value=_FakeSessionCM())), \
             patch.object(graphrag_handler.GraphRAGHandler, "_drop_tenant_data_in_session", new=_fake_drop):
                asyncio.run(graphrag_handler.wipe_tenant_graph_data("agent_test"))
        conn.assert_awaited_once()
        # the wipe handler must be scoped to the target agent
        assert captured["handler"].agent_id == "agent_test"
    finally:
        graphrag_handler._last_graphrag_config.pop("agent_test", None)


# ---------------------------------------------------------------------------
# T25 — graph-phase write idempotency (restart-safe re-run)
# ---------------------------------------------------------------------------


class _EmptyResult:
    """Minimal async result: no rows, no single record."""

    async def single(self):
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _RecordingTx:
    def __init__(self, session):
        self._session = session

    async def run(self, query, **params):
        self._session.queries.append(str(query))
        return _EmptyResult()


class _RecordingSession:
    """Captures every Cypher statement sent through run/execute_write."""

    def __init__(self):
        self.queries = []

    async def run(self, query, **params):
        self.queries.append(str(query))
        return _EmptyResult()

    async def execute_write(self, fn):
        await fn(_RecordingTx(self))


class _RecordingSessionCM:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *a):
        return False


def _capture_session(handler, session):
    """Patch the handler's _get_session to yield a recording session."""
    from unittest.mock import Mock

    return patch.object(
        type(handler),
        "_get_session",
        new=Mock(return_value=_RecordingSessionCM(session)),
    )


def test_concept_store_rerun_issues_identical_merge_queries():
    """Re-running the LLM concept step for the same source+gen must issue
    identical MERGE-based statements (no bare CREATE, no weight accumulation)."""

    handler = _make_handler()
    handler.agent_id = "agent_test"

    relations = {
        "concepts": [
            {"type": "CONCEPT", "text": "Overfitting", "_norm": "overfitting"},
            {"type": "CONCEPT", "text": "Generalization", "_norm": "generalization"},
        ],
        "relations": [
            {
                "type": "CAUSES",
                "origin": "Overfitting",
                "destination": "Generalization",
                "text": "Overfitting causes poor generalization",
            },
        ],
    }

    runs = []
    for _ in range(2):
        rec = _RecordingSession()
        with _capture_session(handler, rec):
            asyncio.run(handler._store_concept_relations(
                "agent_test",
                relations,
                source="file.pdf",
                document_ids=["doc1", "doc2"],
                concept_gen="gen_abc",
            ))
        runs.append(rec.queries)

    assert runs[0], "expected captured Cypher statements"
    # Re-running the same source+gen must issue the SAME statements.
    assert runs[0] == runs[1], "re-run must issue identical queries"

    all_queries = "\n".join(runs[0])
    # No bare CREATE anywhere in the concept-store writes.
    assert "CREATE (" not in all_queries, f"bare CREATE found: {all_queries}"
    # Entity / edge / provenance writes are MERGE-based.
    assert "MERGE (z:Entity" in all_queries
    assert "MERGE (s)-[r:RELATED_TO" in all_queries
    assert "MERGE (d)-[:PROVENANCE]->(s)" in all_queries
    # The weight write is idempotent: no `+ 0.5` accumulation on re-run.
    assert "SET r.weight = coalesce(r.weight, 1.0)" in all_queries
    assert "coalesce(r.weight, 1.0) + 0.5" not in all_queries


def test_similar_rel_query_is_merge_based():
    """The SIMILAR_TO_{gen} write (epoch.py create_similar_rel) must be a
    MERGE in both directions — re-running the graph phase for the same source
    must not duplicate similarity edges."""

    handler = _make_handler()
    query = handler._compile_query("create_similar_rel", "v1")
    assert "MERGE (a)-[r1:SIMILAR_TO_v1]->(b)" in query
    assert "MERGE (b)-[r2:SIMILAR_TO_v1]->(a)" in query
    assert "CREATE (" not in query


def test_derived_graph_writes_are_merge_based():
    """The structural derived graph (SourceFile / PART_OF / NEXT / CHILD_OF /
    HAS_SUMMARY) must issue only MERGE statements for a source re-run."""

    handler = _make_handler()
    handler.agent_id = "agent_test"

    class _FakePoint:
        def __init__(self, pid, chunk_index, parent_id=None):
            self.id = pid
            self.payload = {
                "metadata": {
                    "source": "file.pdf",
                    "chunk_index": chunk_index,
                    "parent_id": parent_id,
                }
            }

    points = [
        _FakePoint("doc1", 0),
        _FakePoint("doc2", 1, parent_id="doc1"),
        _FakePoint("doc3", 2),
    ]

    rec = _RecordingSession()
    with patch.object(type(handler), "_ensure_connected", new=AsyncMock()), \
         _capture_session(handler, rec):
        asyncio.run(handler.create_derived_graph_for_source("file.pdf", points))

    assert rec.queries, "expected captured Cypher statements"
    all_queries = "\n".join(rec.queries)
    # No bare CREATE in the derived-graph writes (Document CREATE lives in
    # add_point_to_tenant and is guarded by the document_id_unique constraint).
    assert "CREATE (" not in all_queries, f"bare CREATE found: {all_queries}"
    assert "MERGE (sf:SourceFile" in all_queries
    assert "MERGE (d)-[:PART_OF]->(sf)" in all_queries
    assert "MERGE (a)-[:NEXT]->(b)" in all_queries
    assert "MERGE (child)-[:CHILD_OF]->(parent)" in all_queries


def test_spacy_batch_query_keeps_concept_gen_removal():
    """T7 shared-schema guard: the spaCy entity batch write must keep
    REMOVE-ing the LLM-owned concept_gen markers. The Entity-node contract
    (concept_gen / concept_gen_active) stays internal to the plugin: the
    spaCy phase must never resurrect stale LLM generations."""
    import inspect
    from catgraphrag_phase import graphrag_handler

    # Every spaCy write (single document and per-source batch) goes through
    # the shared transaction body, so the guard lives there.
    src = inspect.getsource(graphrag_handler.GraphRAGHandler._write_graph_plan_in_tx)
    assert "REMOVE e.concept_gen, e.concept_gen_active" in src, (
        "spaCy batch entity query must REMOVE the LLM-owned concept_gen markers"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    _install_stubs()

    # In-memory package pointing at the repo root: imports the plugin modules
    # directly (relative imports resolved via __path__) without executing the
    # plugin __init__.py side effects.
    _pkg = types.ModuleType("catgraphrag_phase")
    _pkg.__path__ = [REPO_ROOT]
    sys.modules["catgraphrag_phase"] = _pkg


    tests = [
        test_transfer_cleanup_noop_on_failure,
        test_transfer_to_graphrag_rebuilds_instead_of_wiping,
        test_transfer_cleanup_wipes_on_success_when_new_handler_not_graphrag,
        test_wipe_tenant_graph_data_skips_without_config,
        test_wipe_tenant_graph_data_wipes_tenant,
        test_concept_store_rerun_issues_identical_merge_queries,
        test_similar_rel_query_is_merge_based,
        test_derived_graph_writes_are_merge_based,
        test_spacy_batch_query_keeps_concept_gen_removal,
    ]

    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()

    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()