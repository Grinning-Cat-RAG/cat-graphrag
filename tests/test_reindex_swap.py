"""Standalone verification for the seamless re-embed swap (todo 12, P4).

Runnable with:  python test_reindex_swap.py

Pure-stdlib, plain asserts, no pytest and no model loading. The plugin modules
(``graphrag_handler``, ``epoch``, ``versioning``) are imported through an
in-memory package whose ``__path__`` points at this repository root (bypassing
the plugin ``__init__.py`` side effects), while external dependencies (cat,
cat.db.crud, langdetect, spacy, pydantic, neo4j, langchain_core) are stubbed in
``sys.modules`` *before* the import. Neo4j is mocked with an in-memory fake
graph that understands the Epoch queries, the shadow-build writes, the
versioned similarity paths and the GC statements used by ``reembed_tenant``.

What is verified:
- (a) shadow-build writes ``embedding_v2`` + creates ``document_embeddings_v2``
  index + ``SIMILAR_TO_v2`` edges while the v1 set is still fully intact;
- (b) the flip sets the Epoch generation token to v2;
- (c) GC drops the v1 index, deletes ``SIMILAR_TO_v1`` edges and removes
  ``embedding_v1`` AFTER the flip (flip-before-GC order asserted);
- (d) the old v1 index is gone after GC (not usable);
- (e) re-running the swap is safe (idempotent-ish: v2 -> v3);
- (f) per-tenant isolation: another tenant's v1 set is untouched;
- the swap is guarded by ``distributed_lock`` (stubbed no-op here);
- the swap uses plain ``session.run`` — never ``execute_write``.
"""

import asyncio
import os
import re
import sys
import types
from contextlib import asynccontextmanager

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


@asynccontextmanager
async def _noop_distributed_lock(*args, **kwargs):
    """No-op stand-in for ``cat.db.crud.distributed_lock``."""
    yield None


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

    # cat.db.crud.distributed_lock — the swap's migration guard. The real
    # package is importable in the plugin runtime; the standalone test stubs
    # it so no Redis connection is attempted.
    db_mod = types.ModuleType("cat.db")
    db_mod.__path__ = []
    sys.modules["cat.db"] = db_mod

    crud_mod = types.ModuleType("cat.db.crud")
    crud_mod.distributed_lock = _noop_distributed_lock
    sys.modules["cat.db.crud"] = crud_mod

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
# In-memory fake Neo4j graph + session that understands the Epoch queries,
# the shadow-build writes, the versioned similarity paths and the GC
# statements used by ``reembed_tenant``.
# ---------------------------------------------------------------------------


class _FakeGraph:
    """Stateful fake: Epoch tokens, Documents, vector indexes, SIMILAR edges."""

    def __init__(self):
        self.epochs = {}          # tenant_id -> generation token
        self.docs = {}            # doc_id -> {tenant_id, content, collection, embedding_v1, embedding_v2}
        self.indexes = set()      # vector index names that exist
        self.similar = set()      # (tenant_id, relation, doc_a, doc_b)
        self.queries = []         # (query, params) across all sessions
        self.execute_write_calls = 0  # must stay 0 (no transactions)

    def seed_tenant(self, tenant_id, gen="v1", doc_ids=("doc1", "doc2")):
        """Seed a tenant with v1 documents, index and SIMILAR_TO_v1 edges."""
        self.epochs[tenant_id] = gen
        for i, doc_id in enumerate(doc_ids):
            self.docs[doc_id] = {
                "tenant_id": tenant_id,
                "content": f"content-{doc_id}",
                "collection": "declarative",
                "embedding_v1": [float(i + 1), 0.0, 0.0, 0.0],
            }
        self.indexes.add(f"document_embeddings_{gen}")
        for a in doc_ids:
            for b in doc_ids:
                if a != b:
                    self.similar.add((tenant_id, f"SIMILAR_TO_{gen}", a, b))


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

        # ── Shadow-build: read all tenant documents (with collection) ──────
        if "RETURN d.id AS id, d.content AS content, c.name AS collection_name" in q:
            rows = [
                {"id": doc_id, "content": doc["content"], "collection_name": doc["collection"]}
                for doc_id, doc in self.graph.docs.items()
                if doc["tenant_id"] == tenant
            ]
            return _FakeResult(rows)

        # ── Shadow-build: write embedding_{gen} (batch UNWIND SET) ─────────
        if "UNWIND $docs AS d" in q and "SET doc.embedding_" in q:
            prop = re.search(r"SET doc\.(\w+) = d\.vector", q).group(1)
            for d in params.get("docs", []):
                doc = self.graph.docs.get(d["id"])
                if doc is not None:
                    doc[prop] = d["vector"]
            return _FakeResult([])

        # ── Shadow-build: create the versioned vector index ────────────────
        if "CREATE VECTOR INDEX" in q:
            name = re.search(r"CREATE VECTOR INDEX (\S+) IF NOT EXISTS", q).group(1)
            self.graph.indexes.add(name)
            return _FakeResult([])

        # ── Shadow-build: find_similar on the versioned index ──────────────
        if "db.index.vector.queryNodes" in q:
            index = re.search(r"queryNodes\('([^']+)'", q).group(1)
            if index not in self.graph.indexes:
                return _FakeResult([])  # index absent -> no matches
            rows = [
                {"id": doc_id, "score": 0.9}
                for doc_id, doc in self.graph.docs.items()
                if doc["tenant_id"] == tenant
                and doc["collection"] == params.get("collection_name")
                and doc_id != params.get("point_id")
            ]
            return _FakeResult(rows)

        # ── Shadow-build: create_similar_rel (SIMILAR_TO_{gen}) ────────────
        if "MERGE (a)-[r1:SIMILAR_TO" in q:
            rel = re.search(r"\[r1:(\w+)\]", q).group(1)
            for sim in params.get("similar", []):
                self.graph.similar.add((tenant, rel, params["point_id"], sim["id"]))
            return _FakeResult([])

        # ── GC: drop the old vector index ──────────────────────────────────
        if "DROP INDEX" in q:
            name = re.search(r"DROP INDEX (\S+) IF EXISTS", q).group(1)
            self.graph.indexes.discard(name)
            return _FakeResult([])

        # ── GC: delete the old SIMILAR_TO edges (tenant-filtered) ──────────
        if "DELETE r" in q and "SIMILAR_TO" in q:
            rel = re.search(r"\[r:(\w+)\]", q).group(1)
            self.graph.similar = {
                e for e in self.graph.similar
                if not (e[0] == tenant and e[1] == rel)
            }
            return _FakeResult([])

        # ── GC: remove the old embedding property ──────────────────────────
        if "REMOVE d.embedding_" in q:
            prop = re.search(r"REMOVE d\.(\w+)", q).group(1)
            for doc in self.graph.docs.values():
                if doc["tenant_id"] == tenant:
                    doc.pop(prop, None)
            return _FakeResult([])

        raise AssertionError(f"Unhandled query in fake session:\n{q}")

    async def run(self, query, **params):
        return await self._dispatch(query, params)

    async def execute_write(self, fn):
        self.graph.execute_write_calls += 1


class _FakeDriver:
    def __init__(self, graph):
        self.graph = graph

    def session(self, database=None):
        return _FakeSession(self.graph)


class _FakeEmbedder:
    """Deterministic fake embedder: vector depends on the content length."""

    def __init__(self, name="fake-embedder", size=4):
        self.name = name
        self.size = size
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0, 0.5, 0.25] for t in texts]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(graph):
    """Build a GraphRAGHandler wired to the fake graph."""
    from catgraphrag_swaptest import graphrag_handler

    handler = graphrag_handler.GraphRAGHandler(
        neo4j_uri="bolt://fake",
        neo4j_user="u",
        neo4j_password="p",
    )
    handler._driver = _FakeDriver(graph)
    handler.agent_id = "agent_test"
    return handler


def _query_index(graph, needle):
    """Index of the first recorded query containing ``needle``, or -1."""
    for i, (q, _) in enumerate(graph.queries):
        if needle in q:
            return i
    return -1


def _probe_flip_state(graph, handler):
    """Wrap ``_set_generation`` to snapshot the graph right before the flip."""
    state = {}
    original = handler._set_generation

    async def _probe(tenant_id, gen):
        state["v1_index_present"] = "document_embeddings_v1" in graph.indexes
        state["v2_index_present"] = "document_embeddings_v2" in graph.indexes
        state["v1_props"] = {
            doc_id: "embedding_v1" in doc
            for doc_id, doc in graph.docs.items()
            if doc["tenant_id"] == tenant_id
        }
        state["v1_edges"] = {
            e for e in graph.similar
            if e[0] == tenant_id and e[1] == "SIMILAR_TO_v1"
        }
        await original(tenant_id, gen)

    handler._set_generation = _probe
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_shadow_build_writes_v2_while_v1_intact():
    graph = _FakeGraph()
    graph.seed_tenant("agent_test", gen="v1")
    handler = _make_handler(graph)
    embedder = _FakeEmbedder()

    flip_state = _probe_flip_state(graph, handler)
    asyncio.run(handler.reembed_tenant("agent_test", embedder))

    # (a) shadow-build: v2 written, v2 index created, SIMILAR_TO_v2 edges.
    assert all("embedding_v2" in doc for doc in graph.docs.values()), (
        "every tenant Document must carry embedding_v2 after the shadow build"
    )
    assert "document_embeddings_v2" in graph.indexes, (
        "the v2 vector index must be created"
    )
    assert any(e[1] == "SIMILAR_TO_v2" for e in graph.similar), (
        "SIMILAR_TO_v2 edges must be computed from the new vectors"
    )

    # v1 was still fully intact at flip time (probe snapshot).
    assert flip_state["v1_index_present"], "v1 index must still exist at flip time"
    assert flip_state["v2_index_present"], "v2 index must exist before the flip"
    assert all(flip_state["v1_props"].values()), (
        "embedding_v1 must still be present on every doc at flip time"
    )
    assert len(flip_state["v1_edges"]) > 0, (
        "SIMILAR_TO_v1 edges must still be present at flip time"
    )

    # (b) flip: generation token is now v2.
    assert graph.epochs["agent_test"] == "v2", "epoch must be flipped to v2"

    # (c) GC after flip: v1 index dropped, SIMILAR_TO_v1 deleted, embedding_v1 removed.
    assert "document_embeddings_v1" not in graph.indexes, (
        "v1 index must be dropped by the GC phase"
    )
    assert not any(e[1] == "SIMILAR_TO_v1" for e in graph.similar), (
        "SIMILAR_TO_v1 edges must be deleted by the GC phase"
    )
    assert all("embedding_v1" not in doc for doc in graph.docs.values()), (
        "embedding_v1 property must be removed by the GC phase"
    )


def test_flip_before_gc_order():
    graph = _FakeGraph()
    graph.seed_tenant("agent_test", gen="v1")
    handler = _make_handler(graph)

    asyncio.run(handler.reembed_tenant("agent_test", _FakeEmbedder()))

    flip_i = _query_index(graph, "SET e.generation = $gen")
    gc_drop_i = _query_index(graph, "DROP INDEX document_embeddings_v1")
    gc_edges_i = _query_index(graph, "SIMILAR_TO_v1")
    gc_remove_i = _query_index(graph, "REMOVE d.embedding_v1")

    assert flip_i != -1, "flip query must be recorded"
    assert gc_drop_i != -1 and gc_edges_i != -1 and gc_remove_i != -1, (
        "all three GC statements must be recorded"
    )
    assert flip_i < gc_drop_i, "flip must happen BEFORE dropping the v1 index"
    assert flip_i < gc_edges_i, "flip must happen BEFORE deleting SIMILAR_TO_v1"
    assert flip_i < gc_remove_i, "flip must happen BEFORE removing embedding_v1"

    # The shadow-build queries all precede the flip.
    shadow_write_i = _query_index(graph, "SET doc.embedding_v2")
    shadow_index_i = _query_index(graph, "CREATE VECTOR INDEX document_embeddings_v2")
    shadow_sim_i = _query_index(graph, "MERGE (a)-[r1:SIMILAR_TO_v2")
    assert shadow_write_i != -1 and shadow_index_i != -1 and shadow_sim_i != -1
    assert shadow_write_i < flip_i and shadow_index_i < flip_i and shadow_sim_i < flip_i


def test_old_v1_not_usable_after_gc():
    graph = _FakeGraph()
    graph.seed_tenant("agent_test", gen="v1")
    handler = _make_handler(graph)

    asyncio.run(handler.reembed_tenant("agent_test", _FakeEmbedder()))

    # (d) the v1 index is gone — no vector search can target it anymore.
    assert "document_embeddings_v1" not in graph.indexes
    assert "document_embeddings_v2" in graph.indexes
    # No v1 embedding data remains anywhere in the tenant.
    assert not any("embedding_v1" in doc for doc in graph.docs.values())


def test_rerun_is_safe_v2_to_v3():
    graph = _FakeGraph()
    graph.seed_tenant("agent_test", gen="v1")
    handler = _make_handler(graph)

    # (e) first swap: v1 -> v2.
    asyncio.run(handler.reembed_tenant("agent_test", _FakeEmbedder()))
    assert graph.epochs["agent_test"] == "v2"
    assert "document_embeddings_v2" in graph.indexes

    # Second swap: v2 -> v3 — must not raise and must GC the v2 set.
    asyncio.run(handler.reembed_tenant("agent_test", _FakeEmbedder()))
    assert graph.epochs["agent_test"] == "v3"
    assert "document_embeddings_v3" in graph.indexes
    assert "document_embeddings_v2" not in graph.indexes, (
        "the v2 index must be GC'd by the second swap"
    )
    assert not any("embedding_v2" in doc for doc in graph.docs.values()), (
        "embedding_v2 must be removed by the second swap's GC"
    )
    assert all("embedding_v3" in doc for doc in graph.docs.values()), (
        "embedding_v3 must be written by the second swap's shadow build"
    )


def test_per_tenant_isolation():
    graph = _FakeGraph()
    graph.seed_tenant("agent_test", gen="v1", doc_ids=("doc1", "doc2"))
    graph.seed_tenant("agent_other", gen="v1", doc_ids=("doc3", "doc4"))
    handler = _make_handler(graph)

    asyncio.run(handler.reembed_tenant("agent_test", _FakeEmbedder()))

    # (f) the other tenant's v1 set is untouched. The v1 vector index is a
    # global Neo4j index (indexes are not tenant-scoped), so the swap's GC
    # drops it unconditionally — but every tenant-scoped piece of data
    # (epoch token, embedding_v1 properties, SIMILAR_TO_v1 edges) survives.
    assert graph.epochs["agent_other"] == "v1", "other tenant's epoch must not flip"
    other_docs = [d for d in graph.docs.values() if d["tenant_id"] == "agent_other"]
    assert all("embedding_v1" in d for d in other_docs), (
        "other tenant's embedding_v1 must survive"
    )
    assert all("embedding_v2" not in d for d in other_docs), (
        "other tenant's docs must not receive embedding_v2"
    )
    assert not any(e[0] == "agent_other" and e[1] == "SIMILAR_TO_v2" for e in graph.similar), (
        "other tenant must not gain SIMILAR_TO_v2 edges"
    )
    assert any(e[0] == "agent_other" and e[1] == "SIMILAR_TO_v1" for e in graph.similar), (
        "other tenant's SIMILAR_TO_v1 edges must survive"
    )
    # The swapped tenant's v2 set is present.
    assert graph.epochs["agent_test"] == "v2"
    assert "document_embeddings_v2" in graph.indexes


def test_no_documents_flips_and_gcs():
    graph = _FakeGraph()
    graph.epochs["agent_test"] = "v1"
    graph.indexes.add("document_embeddings_v1")
    handler = _make_handler(graph)

    asyncio.run(handler.reembed_tenant("agent_test", _FakeEmbedder()))

    assert graph.epochs["agent_test"] == "v2", "empty tenant still flips the epoch"
    assert "document_embeddings_v1" not in graph.indexes, (
        "empty tenant still GCs the old index"
    )


def test_swap_uses_plain_session_run_no_execute_write():
    graph = _FakeGraph()
    graph.seed_tenant("agent_test", gen="v1")
    handler = _make_handler(graph)

    asyncio.run(handler.reembed_tenant("agent_test", _FakeEmbedder()))

    assert graph.execute_write_calls == 0, (
        "the swap must use plain session.run, never execute_write"
    )


def test_none_embedder_skips_swap():
    graph = _FakeGraph()
    graph.seed_tenant("agent_test", gen="v1")
    handler = _make_handler(graph)

    asyncio.run(handler.reembed_tenant("agent_test", None))

    assert graph.epochs["agent_test"] == "v1", "no embedder -> no flip"
    assert "document_embeddings_v1" in graph.indexes, "no embedder -> no GC"


def main():
    _install_stubs()

    _pkg = types.ModuleType("catgraphrag_swaptest")
    _pkg.__path__ = [REPO_ROOT]
    sys.modules["catgraphrag_swaptest"] = _pkg

    global graphrag_handler
    from catgraphrag_swaptest import graphrag_handler  # noqa: E402
    from catgraphrag_swaptest import epoch  # noqa: E402
    from catgraphrag_swaptest import versioning  # noqa: E402

    tests = [
        test_shadow_build_writes_v2_while_v1_intact,
        test_flip_before_gc_order,
        test_old_v1_not_usable_after_gc,
        test_rerun_is_safe_v2_to_v3,
        test_per_tenant_isolation,
        test_no_documents_flips_and_gcs,
        test_swap_uses_plain_session_run_no_execute_write,
        test_none_embedder_skips_swap,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()