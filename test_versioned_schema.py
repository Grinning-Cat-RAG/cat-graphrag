"""Standalone verification for the versioned schema machinery (todo 11, P4).

Runnable with:  python test_versioned_schema.py

Pure-stdlib, plain asserts, no pytest and no model loading. The plugin modules
(``graphrag_handler``, ``epoch``, ``versioning``) are imported through an
in-memory package whose ``__path__`` points at this repository root (bypassing
the plugin ``__init__.py`` side effects), while external dependencies (cat,
langdetect, spacy, pydantic, neo4j, langchain_core) are stubbed in
``sys.modules`` *before* the import. Neo4j is mocked with an in-memory fake
graph that understands the Epoch queries and the versioned Cypher patterns
used by the decorated recall/similarity paths.

What is verified:
- ``_read_generation`` creates an ``(:Epoch {generation: 'v1'})`` node when
  absent and always returns a non-empty string;
- ``_set_generation`` atomically writes the new token;
- ``_versioned_names`` resolves ``embedding_{gen}`` / ``document_embeddings_{gen}``
  / ``SIMILAR_TO_{gen}`` correctly;
- ``@ensure_version`` probes the generation, rebuilds on drift and runs the
  query exactly ONCE (no retry);
- ``@retry_on_generation_change`` re-runs the whole function once when the
  generation changed mid-run (first run on the old gen, second on the new one);
- the query cache reuses the compiled cypher for the same generation and
  recompiles on drift (versioned names INLINED, never passed as params);
- the decorated paths use plain ``session.run`` — never ``execute_write``
  (no transactions).
"""

import asyncio
import os
import re
import sys
import types

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


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

    class _StubModel:
        def __init__(self, **kwargs):
            for _name, _value in kwargs.items():
                setattr(self, _name, _value)

        def model_dump(self):
            return dict(vars(self))

    models_mod.PointStruct = _StubModel
    models_mod.DocumentRecall = _StubModel
    models_mod.Record = _StubModel
    models_mod.ScoredPoint = _StubModel
    models_mod.UpdateResult = _StubModel
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

    class _StubLcDocument:
        def __init__(self, **kwargs):
            for _name, _value in kwargs.items():
                setattr(self, _name, _value)

    docs.Document = _StubLcDocument
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

    setattr(pydantic_mod, "Field", Field)
    setattr(pydantic_mod, "ConfigDict", ConfigDict)
    setattr(pydantic_mod, "BaseModel", BaseModel)
    sys.modules["pydantic"] = pydantic_mod


def _install_stubs():
    _install_cat_stub()
    _install_neo4j_stub()
    _install_langchain_stub()
    _install_langdetect_stub()
    _install_spacy_stubs()
    _install_pydantic_stub()


# ---------------------------------------------------------------------------
# In-memory fake Neo4j graph + session that understands the Epoch queries and
# the versioned recall/similarity patterns used by the configured paths.
# ---------------------------------------------------------------------------


class _FakeGraph:
    """Epoch tokens + a query/execute_write recorder for the decorated paths."""

    def __init__(self):
        self.epochs = {}              # tenant_id -> generation token
        self.similar_to = set()       # (tenant_id, doc_a, doc_b)
        self.queries = []             # (query, params) across all sessions
        self.execute_write_calls = 0  # must stay 0 in the decorated paths
        # Stateful additions for the write-path / bootstrap tests:
        self.docs = {}                # doc_id -> {tenant_id, content, collection, embedding_v1, ...}
        self.indexes = set()          # vector index names that exist
        self.collections = set()      # (tenant_id, collection_name)


def _doc_rows(extra=None):
    rows = [
        {"id": "doc1", "content": "alpha", "metadata": "{}", "embedding": [0.1, 0.2], "score": 0.9},
        {"id": "doc2", "content": "beta", "metadata": "{}", "embedding": [0.3, 0.4], "score": 0.8},
    ]
    if extra:
        for r in rows:
            r.update(extra)
    return rows


class _FakeResult:
    def __init__(self, records):
        self._records = list(records)

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for r in self._records:
            yield r

    async def data(self):
        return list(self._records)

    async def single(self):
        return self._records[0] if self._records else None


class _FakeSession:
    def __init__(self, graph):
        self.graph = graph

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def _dispatch(self, query, params):
        self.graph.queries.append((query, params))
        tenant = params.get("tenant_id")
        q = query

        # ── Epoch queries ──────────────────────────────────────────────────
        if "RETURN e.generation AS gen" in q:
            gen = self.graph.epochs.get(tenant)
            if gen is None:
                return _FakeResult([])
            return _FakeResult([{"gen": gen}])

        if "ON CREATE SET e.generation = 'v1'" in q:
            self.graph.epochs.setdefault(tenant, "v1")
            return _FakeResult([])

        if "SET e.generation = $gen" in q:
            self.graph.epochs[tenant] = params["gen"]
            return _FakeResult([])

        # ── initialize() queries ──────────────────────────────────────────
        if "CREATE CONSTRAINT" in q or "CREATE INDEX" in q:
            return _FakeResult([])

        if "SHOW INDEXES" in q:  # dimension probe
            return _FakeResult([])

        if "RETURN c.embedder_name AS embedder_name" in q:  # collection embedder config
            return _FakeResult([])

        if "MERGE (c:Collection {name: $name, tenant_id: $tenant_id})" in q:
            self.graph.collections.add((tenant, params["name"]))
            return _FakeResult([])

        if "CREATE VECTOR INDEX" in q:  # initialize + shadow-build index creation
            name = re.search(r"CREATE VECTOR INDEX (\S+) IF NOT EXISTS", q).group(1)
            self.graph.indexes.add(name)
            return _FakeResult([])

        # ── Backfill: copy legacy unversioned embedding -> embedding_{gen} ─
        if "SET d.embedding_" in q and "d.embedding" in q:
            prop = re.search(r"SET d\.(\w+) = d\.embedding", q).group(1)
            for doc in self.graph.docs.values():
                if doc["tenant_id"] == tenant and "embedding" in doc and prop not in doc:
                    doc[prop] = doc["embedding"]
            return _FakeResult([])

        # ── CREATE Document (add_point_to_tenant) ─────────────────────────
        if "CREATE (d:Document {" in q:
            prop = re.search(
                r"CREATE \(d:Document \{\s*id: \$id,\s*content: \$content,\s*(\w+): \$embedding",
                q,
            ).group(1)
            self.graph.docs[params["id"]] = {
                "id": params["id"],
                "tenant_id": tenant,
                "content": params["content"],
                "collection": params["collection_name"],
                prop: params["embedding"],
            }
            return _FakeResult([{"id": params["id"]}])

        # ── Versioned data queries ─────────────────────────────────────────
        if "db.index.vector.queryNodes" in q:
            # find_similar / recall_by_vector / recall_entity_by_vector / search_in_tenant
            return _FakeResult(self._read_rows(tenant))

        if "matched_count" in q:  # recall_entity_direct
            return _FakeResult(self._read_rows(tenant))

        if "min_hops" in q:  # recall_entity_related
            return _FakeResult(self._read_rows(tenant))

        if "MERGE (a)-[r1:SIMILAR_TO" in q:  # create_similar_rel
            for sim in params.get("similar", []):
                self.graph.similar_to.add((tenant, params["point_id"], sim["id"]))
            return _FakeResult([])

        if "SKIP $skip" in q:  # get_all_tenant_points
            return _FakeResult(self._read_rows(tenant))

        if "WHERE d.id IN $ids" in q:  # retrieve_tenant_points
            return _FakeResult(self._read_rows(tenant))

        if "BELONGS_TO" in q and "RETURN d.id AS id" in q:  # recall_tenant_memory
            return _FakeResult(self._read_rows(tenant))

        raise AssertionError(f"Unhandled query in fake session:\n{q}")

    def _read_rows(self, tenant):
        """Return stored tenant docs as recall rows, or the static fallback.

        The existing decorated-path tests seed no documents, so they fall back
        to ``_doc_rows()`` (unchanged behaviour). The write-path / bootstrap
        tests store documents via ``add_point_to_tenant`` / ``initialize`` and
        get the real stored rows back.
        """
        tenant_docs = [d for d in self.graph.docs.values() if d["tenant_id"] == tenant]
        if not tenant_docs:
            return _doc_rows()
        rows = []
        for doc in tenant_docs:
            emb = None
            for key in ("embedding_v1", "embedding_v2", "embedding"):
                if key in doc:
                    emb = doc[key]
                    break
            rows.append({
                "id": doc["id"],
                "content": doc["content"],
                "metadata": "{}",
                "embedding": emb,
                "score": 0.9,
            })
        return rows

    async def run(self, query, **params):
        return await self._dispatch(query, params)

    async def execute_write(self, fn):
        self.graph.execute_write_calls += 1


class _FakeDriver:
    def __init__(self, graph):
        self.graph = graph

    def session(self, database=None):
        return _FakeSession(self.graph)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(graph):
    """Build a GraphRAGHandler wired to the fake graph."""
    from catgraphrag_verstest import graphrag_handler

    handler = graphrag_handler.GraphRAGHandler(
        neo4j_uri="bolt://fake",
        neo4j_user="u",
        neo4j_password="p",
    )
    handler._driver = _FakeDriver(graph)
    handler.agent_id = "agent_test"
    return handler


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_read_generation_creates_v1_when_absent():
    graph = _FakeGraph()
    handler = _make_handler(graph)

    gen = asyncio.run(handler._read_generation())

    assert gen == "v1", "absent Epoch node must yield generation 'v1'"
    assert graph.epochs.get("agent_test") == "v1", (
        "the Epoch node must be created with generation='v1'"
    )


def test_read_generation_returns_existing_token():
    graph = _FakeGraph()
    graph.epochs["agent_test"] = "v7"
    handler = _make_handler(graph)

    gen = asyncio.run(handler._read_generation())

    assert gen == "v7", "existing token must be returned verbatim"


def test_set_generation_writes_atomically():
    graph = _FakeGraph()
    graph.epochs["agent_test"] = "v1"
    handler = _make_handler(graph)

    asyncio.run(handler._set_generation("agent_test", "v2"))

    assert graph.epochs["agent_test"] == "v2"


def test_versioned_names_suffixes():
    handler = _make_handler(_FakeGraph())

    names = handler._versioned_names("v2")

    assert names == {
        "embedding_prop": "embedding_v2",
        "index": "document_embeddings_v2",
        "relation": "SIMILAR_TO_v2",
    }, "all three schema names must be version-suffixed"


def test_ensure_version_probes_rebuilds_and_runs_once():
    from catgraphrag_verstest import versioning

    graph = _FakeGraph()
    graph.epochs["agent_test"] = "v1"
    handler = _make_handler(graph)
    handler._generation = None  # drift: cached generation unknown

    calls = []

    @versioning.ensure_version
    async def _query(self, *args, **kwargs):
        calls.append((self._generation, self._names.get("index")))
        return "ok"

    result = asyncio.run(_query(handler))

    assert result == "ok"
    assert len(calls) == 1, "query must run exactly once (no retry)"
    assert handler._generation == "v1", "cached generation rebuilt to v1"
    assert handler._names["index"] == "document_embeddings_v1", (
        "versioned index name rebuilt on drift"
    )


def test_retry_on_generation_change_reruns_whole_fn_once():
    from catgraphrag_verstest import versioning

    graph = _FakeGraph()
    graph.epochs["agent_test"] = "v1"
    handler = _make_handler(graph)
    handler._generation = None

    calls = []

    async def _multi(self, *args, **kwargs):
        calls.append(self._generation)
        if len(calls) == 1:
            graph.epochs["agent_test"] = "v2"  # flip mid-run
        return f"run-{len(calls)}"

    wrapped = versioning.retry_on_generation_change(_multi)
    result = asyncio.run(wrapped(handler))

    assert result == "run-2", "second attempt result must win"
    assert len(calls) == 2, "whole fn must be re-run exactly once (max 2 attempts)"
    assert calls == ["v1", "v2"], "first run on v1, second run on new gen v2"
    assert handler._generation == "v2", "handler re-initialised for the new gen"


def test_query_cache_reuses_per_gen_and_recompiles_on_drift():
    graph = _FakeGraph()
    handler = _make_handler(graph)
    handler._rebuild_for_generation("v1")

    q1 = handler._compile_query("recall_by_vector", "v1")
    q2 = handler._compile_query("recall_by_vector", "v1")
    assert q1 is q2, "same generation -> cached cypher reused (no recompile)"
    assert "document_embeddings_v1" in q1, "versioned index inlined"
    assert "doc.embedding_v1" in q1, "versioned embedding property inlined"

    handler._rebuild_for_generation("v2")
    q3 = handler._compile_query("recall_by_vector", "v2")
    assert q3 is not q1, "drift -> recompiled for the new generation"
    assert "document_embeddings_v2" in q3, "new index name inlined"
    assert "doc.embedding_v2" in q3, "new embedding property inlined"
    assert handler._query_cache["recall_by_vector"]["v1"] is q1, (
        "the v1 entry is kept alongside the v2 entry"
    )


def test_decorated_paths_use_plain_session_run_no_execute_write():
    graph = _FakeGraph()
    graph.epochs["agent_test"] = "v1"
    handler = _make_handler(graph)

    # Single-query decorated path.
    asyncio.run(handler._recall_by_vector("declarative", [0.1, 0.2], 10))
    # Multi-query decorated path (read phase + write phase).
    asyncio.run(handler._create_similarity_relationships("doc1", [0.1, 0.2], "declarative"))

    assert graph.execute_write_calls == 0, (
        "decorated paths must use plain session.run, never execute_write"
    )
    assert any("document_embeddings_v1" in q for q, _ in graph.queries), (
        "versioned index name must be inlined in the compiled Cypher"
    )
    assert any("SIMILAR_TO_v1" in q for q, _ in graph.queries), (
        "versioned SIMILAR_TO relation must be inlined"
    )
    # create_similar_rel matches Documents by id only (no tenant_id param),
    # so the fake records the edges under tenant=None.
    assert len(graph.similar_to) == 2, (
        "write phase recorded SIMILAR_TO edges via plain session.run"
    )
    assert any("doc.embedding_v1" in q for q, _ in graph.queries), (
        "versioned embedding property read must be inlined"
    )


def test_recall_tenant_memory_from_embedding_retries_whole_orchestration():
    from catgraphrag_verstest import versioning

    graph = _FakeGraph()
    graph.epochs["agent_test"] = "v1"
    handler = _make_handler(graph)
    handler._enable_entity_expansion = False  # only Phase B active

    calls = []
    original = handler.recall_tenant_memory_from_embedding.__wrapped__

    async def _patched(self, *args, **kwargs):
        calls.append(self._generation)
        if len(calls) == 1:
            graph.epochs["agent_test"] = "v2"
        return await original(self, *args, **kwargs)

    # Plain instance-attribute assignment does NOT bind self; wrap with
    # MethodType so the decorator receives the handler as its first arg.
    handler.recall_tenant_memory_from_embedding = types.MethodType(
        versioning.retry_on_generation_change(_patched), handler
    )
    result = asyncio.run(handler.recall_tenant_memory_from_embedding("declarative", [0.1, 0.2]))

    assert len(calls) == 2, "whole orchestration must be re-run once on flip"
    assert calls == ["v1", "v2"]
    assert isinstance(result, list)


def test_add_point_writes_versioned_embedding_and_reads_find_it():
    graph = _FakeGraph()
    handler = _make_handler(graph)
    handler._enable_entity_extraction = False  # no background spaCy task

    # Fresh install: no Epoch, no docs, no indexes.
    result = asyncio.run(handler.add_point_to_tenant(
        "declarative", "alpha", [0.1, 0.2], {}, "doc1"
    ))

    assert result is not None
    # (a) the write targeted embedding_v1 (the current generation).
    assert graph.docs["doc1"].get("embedding_v1") == [0.1, 0.2], (
        "add_point_to_tenant must write the versioned embedding property"
    )
    assert "embedding" not in graph.docs["doc1"], (
        "the unversioned embedding property must not be written"
    )
    # The Epoch was created with v1.
    assert graph.epochs["agent_test"] == "v1"

    # A versioned read finds the freshly-written document.
    docs = asyncio.run(handler.recall_tenant_memory("declarative"))
    assert any(d.id == "doc1" for d in docs), (
        "a versioned read must find the freshly-written document"
    )


def test_initialize_bootstraps_versioned_index_and_backfills():
    graph = _FakeGraph()
    # Pre-P4 data: docs with the unversioned `embedding` property, no Epoch,
    # no versioned index.
    graph.docs["doc1"] = {
        "id": "doc1",
        "tenant_id": "agent_test",
        "content": "alpha",
        "collection": "declarative",
        "embedding": [0.1, 0.2],
    }
    handler = _make_handler(graph)
    handler._collection_names = ["declarative"]

    async def _noop_connect(self):
        pass

    handler._connect = types.MethodType(_noop_connect, handler)

    asyncio.run(handler.initialize("fake-embedder", 4))

    # (b) the versioned index the reads will use exists.
    assert "document_embeddings_v1" in graph.indexes, (
        "initialize must create the versioned index for the current generation"
    )
    # (c) the legacy unversioned embedding was backfilled into embedding_v1.
    assert graph.docs["doc1"].get("embedding_v1") == [0.1, 0.2], (
        "initialize must backfill the legacy unversioned embedding into embedding_v1"
    )
    # The Epoch was created with v1.
    assert graph.epochs["agent_test"] == "v1"


def main():
    _install_stubs()

    _pkg = types.ModuleType("catgraphrag_verstest")
    _pkg.__path__ = [REPO_ROOT]
    sys.modules["catgraphrag_verstest"] = _pkg

    global graphrag_handler
    from catgraphrag_verstest import graphrag_handler  # noqa: E402
    from catgraphrag_verstest import epoch  # noqa: E402
    from catgraphrag_verstest import versioning  # noqa: E402

    tests = [
        test_read_generation_creates_v1_when_absent,
        test_read_generation_returns_existing_token,
        test_set_generation_writes_atomically,
        test_versioned_names_suffixes,
        test_ensure_version_probes_rebuilds_and_runs_once,
        test_retry_on_generation_change_reruns_whole_fn_once,
        test_query_cache_reuses_per_gen_and_recompiles_on_drift,
        test_decorated_paths_use_plain_session_run_no_execute_write,
        test_recall_tenant_memory_from_embedding_retries_whole_orchestration,
        test_add_point_writes_versioned_embedding_and_reads_find_it,
        test_initialize_bootstraps_versioned_index_and_backfills,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
