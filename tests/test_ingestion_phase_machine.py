"""Standalone verification for the ingestion phase machine hooks (T21-T24).

Covers the four hooks added to ``main.py`` plus the handler helpers they
drive (``run_graph_building_phase``, ``drain_source_tasks``,
``has_pending_source_tasks``, ``get_source_points_by_collection``,
``wipe_tenant_graph_data``):

- ``ingestion_phase_pending`` (accumulator convention: first positional arg
  ``pending`` is threaded through registrants and its non-None return
  replaces it) returns the accumulator unchanged when the dependency phases
  are missing, when the recorded generation matches the recomputed one, or
  when graphrag is not the active handler; appends
  ``{"phase": "graph_building", "gen": <recomputed>}`` when the deps are
  done and the generation is stale, preserving any prior accumulator
  entries;
- ``ingestion_phase_run`` returns None for other phases, not_ready when the
  source's points are absent, and done (with the LLM concept-relations step
  awaited) when the points are present;
- ``before_ingestion_status_completed`` raises when live background tasks
  remain for the source or when the graph_building generation was not
  recorded, and passes when the work is clean;
- ``after_vector_memory_transfer_on_agent`` wipes the tenant's Neo4j data
  ONLY on success + old handler == Neo4jGraphRAGConfig, and never on failure;
- ``wipe_tenant_graph_data`` skips without a recorded config and wipes the
  tenant (tenant-scoped) when one exists.

Runnable:  python test_ingestion_phase_machine.py
Pure-stdlib + unittest.mock, plain asserts, no pytest, no model loading. All
external packages (cat, langchain_core, spacy, langdetect, pydantic, neo4j)
are stubbed in sys.modules *before* the plugin modules are imported (same
pattern as test_kg_master_switch.py). The core phase-machine helpers
(``cat.core_plugins.ingestion_status.fingerprints`` / ``.registry``) are
stubbed with the real ``phase_generation`` logic and fakes for the Redis
reads.
"""

import asyncio
import hashlib
import json
import os
import sys
import types
from unittest.mock import AsyncMock, Mock, patch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


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
    models_mod.PointStruct = type("PointStruct", (), {})
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


def _phase_generation(phase_id, settings_fp, dep_gens):
    """Mirror of the core's ``phase_generation`` (fingerprints.py)."""
    payload = json.dumps(
        {"phase": phase_id, "fp": settings_fp, "deps": dep_gens},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _install_ingestion_status_stub():
    """Stub the core phase-machine helpers the hooks import lazily.

    ``build_graphrag_fingerprint`` is a fake Redis read (fixed value);
    ``phase_generation`` is the REAL algorithm; ``get_status`` is a fake
    Redis read whose return value tests override per-case.
    """
    core_plugins = types.ModuleType("cat.core_plugins")
    core_plugins.__path__ = []
    sys.modules["cat.core_plugins"] = core_plugins

    ingestion_status = types.ModuleType("cat.core_plugins.ingestion_status")
    ingestion_status.__path__ = []
    sys.modules["cat.core_plugins.ingestion_status"] = ingestion_status

    fingerprints = types.ModuleType("cat.core_plugins.ingestion_status.fingerprints")
    fingerprints.build_graphrag_fingerprint = AsyncMock(
        return_value={"name": "Neo4jGraphRAGConfig", "settings": {"neo4j_uri": "bolt://fake"}}
    )
    fingerprints.phase_generation = _phase_generation
    sys.modules["cat.core_plugins.ingestion_status.fingerprints"] = fingerprints

    registry = types.ModuleType("cat.core_plugins.ingestion_status.registry")
    registry.get_status = AsyncMock(return_value=None)
    sys.modules["cat.core_plugins.ingestion_status.registry"] = registry


def _install_stubs():
    _install_cat_stub()
    _install_neo4j_stub()
    _install_langchain_stub()
    _install_langdetect_stub()
    _install_spacy_stubs()
    _install_pydantic_stub()
    _install_ingestion_status_stub()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeCat:
    """Minimal stand-in for CheshireCat: no ``id`` (agent-scoped), handler set."""

    def __init__(self, handler):
        self.vector_memory_handler = handler
        self.agent_key = "agent_test"


class _FakePoint:
    def __init__(self, pid, vector=None):
        self.id = pid
        self.vector = vector


def _make_handler(kg=True, cr=True, derived=True):
    """Build a GraphRAGHandler with the given flags (no driver needed)."""
    from .. import graphrag_handler

    handler = graphrag_handler.GraphRAGHandler(
        neo4j_uri="bolt://fake",
        neo4j_user="u",
        neo4j_password="p",
        enable_derived_graph=derived,
        enable_knowledge_graph=kg,
        enable_concept_relations=cr,
    )
    handler.agent_id = "agent_test"
    return handler


def _completed(deps, recorded_gen=None):
    entries = [{"phase": pid, "gen": gen} for pid, gen in deps]
    if recorded_gen is not None:
        entries.append({"phase": "graph_building", "gen": recorded_gen})
    return entries


# ---------------------------------------------------------------------------
# T21 — probe
# ---------------------------------------------------------------------------


def test_probe_returns_empty_when_deps_missing():
    from .. import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    pending = asyncio.run(main.ingestion_phase_pending(
        [], "file.pdf", _completed([("parsing_chunking", "g1")]), cat
    ))
    assert pending == [], f"expected [] with only parsing_chunking done, got {pending}"


def test_probe_returns_graph_building_when_deps_done_and_stale():
    from .. import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    pending = asyncio.run(main.ingestion_phase_pending(
        [],
        "file.pdf",
        _completed([("parsing_chunking", "g1"), ("embedding", "g2")]),
        cat,
    ))
    assert len(pending) == 1, f"expected one entry, got {pending}"
    entry = pending[0]
    assert isinstance(entry, dict), f"entry must be a dict, got {type(entry)}"
    assert entry["phase"] == "graph_building", f"got {entry}"
    assert entry["gen"], f"entry must carry a recomputed gen, got {entry}"


def test_probe_preserves_prior_accumulator_entries():
    from .. import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    accumulator = [{"phase": "embedding", "gen": "g2"}]
    pending = asyncio.run(main.ingestion_phase_pending(
        accumulator,
        "file.pdf",
        _completed([("parsing_chunking", "g1"), ("embedding", "g2")]),
        cat,
    ))
    assert pending[0] == {"phase": "embedding", "gen": "g2"}, (
        f"prior accumulator entries must be preserved, got {pending}"
    )
    assert pending[1]["phase"] == "graph_building", f"got {pending}"


def test_probe_returns_empty_when_gen_recorded():
    from .. import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    # The recorded gen is computed from the DEPENDENCY phases only (the probe
    # excludes graph_building itself from dep_gens).
    dep_gens = {"parsing_chunking": "g1", "embedding": "g2"}
    fp = {
        "settings": {"name": "Neo4jGraphRAGConfig", "settings": {"neo4j_uri": "bolt://fake"}},
        "concept": handler._concept_fingerprint(),
    }
    recorded = _phase_generation("graph_building", fp, dep_gens)
    pending = asyncio.run(main.ingestion_phase_pending(
        [],
        "file.pdf",
        _completed([("parsing_chunking", "g1"), ("embedding", "g2")], recorded_gen=recorded),
        cat,
    ))
    assert pending == [], f"expected [] when gen recorded, got {pending}"


def test_probe_returns_empty_when_not_graphrag_handler():
    from .. import main

    class _OtherHandler:
        pass

    cat = _FakeCat(_OtherHandler())
    pending = asyncio.run(main.ingestion_phase_pending(
        [], "file.pdf", _completed([("parsing_chunking", "g1"), ("embedding", "g2")]), cat
    ))
    assert pending == [], f"expected [] for non-graphrag handler, got {pending}"


def test_probe_returns_empty_when_derived_graph_disabled():
    from .. import main

    handler = _make_handler(derived=False)
    cat = _FakeCat(handler)
    pending = asyncio.run(main.ingestion_phase_pending(
        [], "file.pdf", _completed([("parsing_chunking", "g1"), ("embedding", "g2")]), cat
    ))
    assert pending == [], f"expected [] when derived graph disabled, got {pending}"


# ---------------------------------------------------------------------------
# T22 — phase execution
# ---------------------------------------------------------------------------


def test_phase_run_returns_none_for_other_phase():
    from .. import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    result = asyncio.run(main.ingestion_phase_run(
        "embedding", "file.pdf", _completed([]), cat
    ))
    assert result is None, f"expected None for non-graph_building phase, got {result}"


def test_phase_run_not_ready_when_points_absent():
    from .. import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    with patch.object(type(handler), "get_source_points_by_collection", new=AsyncMock(return_value={})), \
         patch.object(type(handler), "drain_source_tasks", new=AsyncMock()) as drain:
        result = asyncio.run(main.ingestion_phase_run(
            "graph_building", "file.pdf", _completed([]), cat
        ))
    assert result == {"status": "not_ready", "retry_after": 5}, f"got {result}"
    drain.assert_awaited_once_with("file.pdf")


def test_phase_run_done_when_points_present():
    from .. import main

    handler = _make_handler(kg=True, cr=True)
    cat = _FakeCat(handler)
    points = [_FakePoint("doc1", vector=[0.1, 0.2, 0.3])]
    with patch.object(type(handler), "get_source_points_by_collection",
                      new=AsyncMock(return_value={"declarative": points})), \
         patch.object(type(handler), "drain_source_tasks", new=AsyncMock()) as drain, \
         patch.object(type(handler), "_create_similarity_relationships", new=AsyncMock()) as sim, \
         patch.object(type(handler), "_extract_concept_relations", new=AsyncMock()) as extract:
        result = asyncio.run(main.ingestion_phase_run(
            "graph_building", "file.pdf", _completed([]), cat
        ))
    assert result == {"status": "done"}, f"got {result}"
    drain.assert_awaited_once_with("file.pdf")
    sim.assert_awaited_once_with("doc1", [0.1, 0.2, 0.3], "declarative")
    extract.assert_awaited_once()
    # the LLM step receives the source's points and the cat (LLM resolution)
    _args = extract.await_args.args
    assert _args[0] == "file.pdf"
    assert _args[1] == points


def test_phase_run_skips_llm_when_kg_off():
    from .. import main

    handler = _make_handler(kg=False, cr=True)
    cat = _FakeCat(handler)
    points = [_FakePoint("doc1", vector=[0.1, 0.2, 0.3])]
    with patch.object(type(handler), "get_source_points_by_collection",
                      new=AsyncMock(return_value={"declarative": points})), \
         patch.object(type(handler), "drain_source_tasks", new=AsyncMock()), \
         patch.object(type(handler), "_create_similarity_relationships", new=AsyncMock()), \
         patch.object(type(handler), "_extract_concept_relations", new=AsyncMock()) as extract:
        result = asyncio.run(main.ingestion_phase_run(
            "graph_building", "file.pdf", _completed([]), cat
        ))
    assert result == {"status": "done"}, f"got {result}"
    extract.assert_not_awaited()


# ---------------------------------------------------------------------------
# T23 — completion gate
# ---------------------------------------------------------------------------


def test_gate_raises_when_pending_tasks():
    from .. import main

    handler = _make_handler()
    cat = _FakeCat(handler)

    async def _run():
        task = asyncio.create_task(asyncio.sleep(30))
        handler._pending_source_tasks["file.pdf"] = [task]
        try:
            await main.before_ingestion_status_completed("file.pdf", cat)
            raise AssertionError("expected RuntimeError for pending tasks")
        except RuntimeError as e:
            assert "Pending background graph tasks" in str(e)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(_run())


def test_gate_raises_when_gen_not_recorded():
    from .. import main
    from cat.plugins.cat_efficient_ingestion import registry

    handler = _make_handler()
    cat = _FakeCat(handler)
    registry.get_status.return_value = {
        "status": "processing",
        "completed_phases": [{"phase": "parsing_chunking", "gen": "g1"}],
    }
    try:
        try:
            asyncio.run(main.before_ingestion_status_completed("file.pdf", cat))
            raise AssertionError("expected RuntimeError when gen not recorded")
        except RuntimeError as e:
            assert "graph_building generation not recorded" in str(e)
    finally:
        registry.get_status.return_value = None


def test_gate_passes_when_clean():
    from .. import main
    from cat.plugins.cat_efficient_ingestion import registry

    handler = _make_handler()
    cat = _FakeCat(handler)
    registry.get_status.return_value = {
        "status": "processing",
        "completed_phases": [
            {"phase": "parsing_chunking", "gen": "g1"},
            {"phase": "embedding", "gen": "g2"},
            {"phase": "graph_building", "gen": "g3"},
        ],
    }
    try:
        asyncio.run(main.before_ingestion_status_completed("file.pdf", cat))
    finally:
        registry.get_status.return_value = None


def test_gate_noop_when_not_graphrag_handler():
    from .. import main

    class _OtherHandler:
        pass

    cat = _FakeCat(_OtherHandler())
    # must not raise even with a "pending" source (no graphrag handler)
    asyncio.run(main.before_ingestion_status_completed("file.pdf", cat))


# ---------------------------------------------------------------------------
# T24 — old-store cleanup on switch
# ---------------------------------------------------------------------------


def test_transfer_cleanup_noop_on_failure():
    from .. import main

    cat = _FakeCat(None)
    with patch("graphrag_handler.wipe_tenant_graph_data", new=AsyncMock()) as wipe:
        asyncio.run(main.after_vector_memory_transfer_on_agent(False, "Neo4jGraphRAGConfig", cat))
        wipe.assert_not_awaited()


def test_transfer_cleanup_noop_when_old_not_neo4j():
    from .. import main

    cat = _FakeCat(None)
    with patch("graphrag_handler.wipe_tenant_graph_data", new=AsyncMock()) as wipe:
        asyncio.run(main.after_vector_memory_transfer_on_agent(True, "QdrantConfig", cat))
        wipe.assert_not_awaited()


def test_transfer_cleanup_wipes_on_success_neo4j():
    from .. import main

    cat = _FakeCat(None)
    with patch("graphrag_handler.wipe_tenant_graph_data", new=AsyncMock()) as wipe:
        asyncio.run(main.after_vector_memory_transfer_on_agent(True, "Neo4jGraphRAGConfig", cat))
        wipe.assert_awaited_once_with("agent_test")


def test_wipe_tenant_graph_data_skips_without_config():
    from .. import graphrag_handler

    graphrag_handler._last_graphrag_config.pop("agent_test", None)
    with patch.object(graphrag_handler.GraphRAGHandler, "_ensure_connected", new=AsyncMock()) as conn:
        asyncio.run(graphrag_handler.wipe_tenant_graph_data("agent_test"))
        conn.assert_not_awaited()


def test_wipe_tenant_graph_data_wipes_tenant():
    from .. import graphrag_handler

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
    from .. import graphrag_handler

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
    from .. import graphrag_handler

    handler = _make_handler()
    query = handler._compile_query("create_similar_rel", "v1")
    assert "MERGE (a)-[r1:SIMILAR_TO_v1]->(b)" in query
    assert "MERGE (b)-[r2:SIMILAR_TO_v1]->(a)" in query
    assert "CREATE (" not in query


def test_derived_graph_writes_are_merge_based():
    """The structural derived graph (SourceFile / PART_OF / NEXT / CHILD_OF /
    HAS_SUMMARY) must issue only MERGE statements for a source re-run."""
    from .. import graphrag_handler

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

    from .. import graphrag_handler  # noqa: E402
    from .. import main as main_mod  # noqa: E402

    tests = [
        test_probe_returns_empty_when_deps_missing,
        test_probe_returns_graph_building_when_deps_done_and_stale,
        test_probe_preserves_prior_accumulator_entries,
        test_probe_returns_empty_when_gen_recorded,
        test_probe_returns_empty_when_not_graphrag_handler,
        test_probe_returns_empty_when_derived_graph_disabled,
        test_phase_run_returns_none_for_other_phase,
        test_phase_run_not_ready_when_points_absent,
        test_phase_run_done_when_points_present,
        test_phase_run_skips_llm_when_kg_off,
        test_gate_raises_when_pending_tasks,
        test_gate_raises_when_gen_not_recorded,
        test_gate_passes_when_clean,
        test_gate_noop_when_not_graphrag_handler,
        test_transfer_cleanup_noop_on_failure,
        test_transfer_cleanup_noop_when_old_not_neo4j,
        test_transfer_cleanup_wipes_on_success_neo4j,
        test_wipe_tenant_graph_data_skips_without_config,
        test_wipe_tenant_graph_data_wipes_tenant,
        test_concept_store_rerun_issues_identical_merge_queries,
        test_similar_rel_query_is_merge_based,
        test_derived_graph_writes_are_merge_based,
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