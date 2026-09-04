"""Standalone verification for the per-agent terminology refresh (todo 10).

Runnable with:  python test_terminology_refresh.py

Pure-stdlib, plain asserts, no pytest and no model loading. The plugin modules
(``main``, ``graphrag_handler``, ``entity_extractor``, ``models``) are imported
through an in-memory package whose ``__path__`` points at this repository root
(bypassing the plugin ``__init__.py`` side effects), while external dependencies
(cat, langdetect, spacy, pydantic, neo4j, langchain_core) are stubbed in
``sys.modules`` *before* the import. Neo4j is mocked with an in-memory fake
graph that executes the specific Cypher patterns used by
``GraphRAGHandler.refresh_technology_entities``.

What is verified:
- the ``after_vector_database_settings_update`` hook fires only for
  ``Neo4jGraphRAGConfig`` and only when ``extra_technology_patterns`` changed,
  and rebuilds the handler's EntityExtractor with the new patterns;
- ``refresh_technology_entities`` adds Technology entities + MENTIONS edges for
  newly matched terms on existing documents;
- removed terms drop their MENTIONS edges and orphaned Technology entities are
  pruned;
- idempotence — a second run with the same patterns leaves node/edge counts
  unchanged;
- Document / SIMILAR_TO / non-Technology entity data is untouched;
- per-tenant isolation — another tenant's graph is untouched.
"""

import asyncio
import os
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
# In-memory fake Neo4j graph + session that understands the exact Cypher
# patterns used by refresh_technology_entities.
# ---------------------------------------------------------------------------


class _FakeGraph:
    """Minimal in-memory graph: Document/Entity nodes, MENTIONS + SIMILAR_TO."""

    def __init__(self):
        self.documents = {}   # (tenant_id, doc_id) -> content
        self.entities = {}    # (tenant_id, entity_id) -> {"name", "type"}
        self.mentions = set()  # (tenant_id, doc_id, entity_id)
        self.provenance = set()  # (tenant_id, doc_id, entity_id)
        self.similar_to = set()  # (tenant_id, doc_a, doc_b)

    def add_document(self, tenant, doc_id, content):
        self.documents[(tenant, doc_id)] = content

    def add_entity(self, tenant, entity_id, name, etype):
        self.entities[(tenant, entity_id)] = {"name": name, "type": etype}

    def add_mention(self, tenant, doc_id, entity_id):
        self.mentions.add((tenant, doc_id, entity_id))

    def add_similar(self, tenant, doc_a, doc_b):
        self.similar_to.add((tenant, doc_a, doc_b))

    def entity_ids(self, tenant, etype):
        return {
            eid for (t, eid), ent in self.entities.items()
            if t == tenant and ent["type"] == etype
        }

    def mention_count(self, tenant=None):
        if tenant is None:
            return len(self.mentions)
        return sum(1 for m in self.mentions if m[0] == tenant)


class _FakeResult:
    def __init__(self, records):
        self._records = list(records)

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for r in self._records:
            yield r


class _FakeTx:
    def __init__(self, session):
        self._session = session

    async def run(self, query, **params):
        return await self._session._dispatch(query, params)


class _FakeSession:
    def __init__(self, graph):
        self.graph = graph
        self.queries = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def _dispatch(self, query, params):
        self.queries.append((query, params))
        q = query
        tenant = params.get("tenant_id")

        # 1. Fetch stored documents of the tenant.
        if "RETURN d.id AS id, d.content AS content" in q:
            return _FakeResult([
                {"id": doc_id, "content": content}
                for (t, doc_id), content in self.graph.documents.items()
                if t == tenant
            ])

        # 2. MERGE entity nodes.
        if "MERGE (e:Entity {id: ent.id" in q:
            for ent in params.get("entities", []):
                self.graph.entities[(tenant, ent["id"])] = {
                    "name": ent["name"],
                    "type": ent["type"],
                }
            return _FakeResult([])

        # 3. MERGE MENTIONS edges.
        if "MERGE (d)-[r:MENTIONS]->(e)" in q:
            for m in params.get("mentions", []):
                self.graph.mentions.add((tenant, m["doc_id"], m["entity_id"]))
            return _FakeResult([])

        # 3b. MERGE PROVENANCE edges (file-deletion tracking, added to the
        #     same documents/entities as the MENTIONS batch).
        if "MERGE (d)-[:PROVENANCE]->(e)" in q:
            for m in params.get("mentions", []):
                doc_id = m.get("doc_id") or params.get("doc_id")
                self.graph.provenance.add((tenant, doc_id, m["entity_id"]))
            return _FakeResult([])

        # 4. Delete stale MENTIONS edges to Technology entities no longer matched.
        if "DELETE r" in q and "doc_terms" in params:
            for d in params["doc_terms"]:
                doc_id = d["doc_id"]
                valid = set(d["entity_ids"])
                stale = [
                    (tenant, doc_id, eid)
                    for (t, did, eid) in self.graph.mentions
                    if t == tenant and did == doc_id
                    and self.graph.entities.get((tenant, eid), {}).get("type") == "TECHNOLOGY"
                    and eid not in valid
                ]
                for m in stale:
                    self.graph.mentions.discard(m)
            return _FakeResult([])

        # 5. Prune orphaned Technology entities (no remaining MENTIONS).
        if "DETACH DELETE e" in q:
            orphans = [
                (tenant, eid)
                for (t, eid), ent in self.graph.entities.items()
                if t == tenant and ent["type"] == "TECHNOLOGY"
                and not any(m[0] == tenant and m[2] == eid for m in self.graph.mentions)
            ]
            for key in orphans:
                self.graph.entities.pop(key, None)
            return _FakeResult([])

        raise AssertionError(f"Unhandled query in fake session:\n{q}")

    async def run(self, query, **params):
        return await self._dispatch(query, params)

    async def execute_write(self, fn):
        await fn(_FakeTx(self))


class _FakeDriver:
    def __init__(self, graph):
        self.graph = graph

    def session(self, database=None):
        return _FakeSession(self.graph)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(graph, patterns):
    """Build a GraphRAGHandler wired to the fake graph with the given patterns."""
    from catgraphrag_tertest import graphrag_handler

    handler = graphrag_handler.GraphRAGHandler(
        neo4j_uri="bolt://fake",
        neo4j_user="u",
        neo4j_password="p",
        extra_technology_patterns=patterns,
    )
    handler._driver = _FakeDriver(graph)
    handler.agent_id = "agent_test"
    return handler


def _hash(term, tenant="agent_test"):
    """Entity hash for a TECHNOLOGY term (mirrors EntityExtractor.get_entity_hash)."""
    from catgraphrag_tertest import entity_extractor
    from catgraphrag_tertest.models import EntityType

    return entity_extractor.EntityExtractor.get_entity_hash(
        term, EntityType.TECHNOLOGY, tenant
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_hook_ignores_other_vector_databases():
    from catgraphrag_tertest import main as main_mod

    graph = _FakeGraph()
    handler = _make_handler(graph, [r"\b(MyTool)\b"])
    old_extractor = handler._entity_extractor

    class _Cat:
        vector_memory_handler = handler
        agent_key = "agent_test"

    async def _run():
        await main_mod.after_vector_database_settings_update(
            "QdrantConfig", {}, {"extra_technology_patterns": [r"\b(X)\b"]}, _Cat()
        )

    asyncio.run(_run())
    assert handler._entity_extractor is old_extractor, (
        "extractor must not be rebuilt for a non-Neo4j vector database"
    )


def test_hook_ignores_unchanged_patterns():
    from catgraphrag_tertest import main as main_mod

    graph = _FakeGraph()
    handler = _make_handler(graph, [r"\b(MyTool)\b"])
    old_extractor = handler._entity_extractor

    class _Cat:
        vector_memory_handler = handler
        agent_key = "agent_test"

    async def _run():
        await main_mod.after_vector_database_settings_update(
            "Neo4jGraphRAGConfig",
            {"extra_technology_patterns": [r"\b(MyTool)\b"]},
            {"extra_technology_patterns": [r"\b(MyTool)\b"]},
            _Cat(),
        )

    asyncio.run(_run())
    assert handler._entity_extractor is old_extractor, (
        "extractor must not be rebuilt when patterns are unchanged"
    )


def test_hook_rebuilds_extractor_on_pattern_change():
    from catgraphrag_tertest import main as main_mod

    graph = _FakeGraph()
    handler = _make_handler(graph, [])
    old_extractor = handler._entity_extractor

    class _Cat:
        vector_memory_handler = handler
        agent_key = "agent_test"

    async def _run():
        await main_mod.after_vector_database_settings_update(
            "Neo4jGraphRAGConfig",
            {"extra_technology_patterns": []},
            {"extra_technology_patterns": [r"\b(MyTool)\b"]},
            _Cat(),
        )

    asyncio.run(_run())
    assert handler._entity_extractor is not old_extractor, (
        "extractor must be rebuilt when patterns change"
    )
    assert any("MyTool" in p for p in handler._entity_extractor._technology_patterns), (
        "new pattern must be present in the rebuilt extractor"
    )


def test_refresh_adds_technology_entities_and_mentions():
    graph = _FakeGraph()
    graph.add_document("agent_test", "doc1", "We use MyTool and Neo4j here.")
    handler = _make_handler(graph, [r"\b(MyTool)\b"])

    asyncio.run(handler.refresh_technology_entities("agent_test"))

    mytool_id = _hash("mytool")
    assert ("agent_test", mytool_id) in graph.entities, "MyTool entity must be created"
    assert graph.entities[("agent_test", mytool_id)]["type"] == "TECHNOLOGY"
    assert ("agent_test", "doc1", mytool_id) in graph.mentions, (
        "MENTIONS edge doc1->MyTool must exist"
    )


def test_refresh_removes_stale_mentions_and_prunes_orphans():
    graph = _FakeGraph()
    graph.add_document("agent_test", "doc1", "We use MyTool now.")
    mytool_id = _hash("mytool")
    oldtool_id = _hash("oldtool")
    graph.add_entity("agent_test", mytool_id, "mytool", "TECHNOLOGY")
    graph.add_entity("agent_test", oldtool_id, "oldtool", "TECHNOLOGY")
    graph.add_mention("agent_test", "doc1", mytool_id)
    graph.add_mention("agent_test", "doc1", oldtool_id)

    # New patterns only match MyTool → OldTool must lose its MENTIONS and be pruned.
    handler = _make_handler(graph, [r"\b(MyTool)\b"])
    asyncio.run(handler.refresh_technology_entities("agent_test"))

    assert ("agent_test", "doc1", mytool_id) in graph.mentions, "MyTool MENTIONS kept"
    assert ("agent_test", "doc1", oldtool_id) not in graph.mentions, (
        "OldTool MENTIONS removed"
    )
    assert ("agent_test", oldtool_id) not in graph.entities, (
        "orphaned OldTool entity pruned"
    )
    assert ("agent_test", mytool_id) in graph.entities, "MyTool entity kept"


def test_refresh_is_idempotent():
    graph = _FakeGraph()
    graph.add_document("agent_test", "doc1", "We use MyTool and Neo4j here.")
    handler = _make_handler(graph, [r"\b(MyTool)\b"])

    asyncio.run(handler.refresh_technology_entities("agent_test"))
    entities_after_first = len(graph.entities)
    mentions_after_first = graph.mention_count("agent_test")

    asyncio.run(handler.refresh_technology_entities("agent_test"))
    assert len(graph.entities) == entities_after_first, (
        "second run must not add/remove entities"
    )
    assert graph.mention_count("agent_test") == mentions_after_first, (
        "second run must not add/remove MENTIONS"
    )


def test_refresh_leaves_documents_similar_and_other_entities_untouched():
    graph = _FakeGraph()
    graph.add_document("agent_test", "doc1", "Alice uses MyTool.")
    graph.add_document("agent_test", "doc2", "Bob uses MyTool too.")
    graph.add_similar("agent_test", "doc1", "doc2")

    person_id = _hash("alice", tenant="agent_test")
    # A PERSON entity (not TECHNOLOGY) with a MENTIONS edge must survive.
    graph.add_entity("agent_test", person_id, "alice", "PERSON")
    graph.add_mention("agent_test", "doc1", person_id)

    handler = _make_handler(graph, [r"\b(MyTool)\b"])
    asyncio.run(handler.refresh_technology_entities("agent_test"))

    assert ("agent_test", "doc1") in graph.documents
    assert ("agent_test", "doc2") in graph.documents
    assert ("agent_test", "doc1", "doc2") in graph.similar_to, "SIMILAR_TO untouched"
    assert ("agent_test", person_id) in graph.entities, "PERSON entity untouched"
    assert ("agent_test", "doc1", person_id) in graph.mentions, (
        "PERSON MENTIONS untouched"
    )


def test_refresh_is_per_tenant_isolated():
    graph = _FakeGraph()
    graph.add_document("agent_test", "doc1", "We use MyTool here.")
    graph.add_document("other_agent", "docX", "We use MyTool here too.")

    other_id = _hash("mytool", tenant="other_agent")
    graph.add_entity("other_agent", other_id, "mytool", "TECHNOLOGY")
    graph.add_mention("other_agent", "docX", other_id)
    other_entity_count = len(graph.entities)
    other_mention_count = graph.mention_count("other_agent")

    handler = _make_handler(graph, [r"\b(MyTool)\b"])
    asyncio.run(handler.refresh_technology_entities("agent_test"))

    # The other tenant's graph must be byte-for-byte untouched.
    assert len(graph.entities) == other_entity_count + 1, (
        "only agent_test's MyTool entity added"
    )
    assert graph.mention_count("other_agent") == other_mention_count, (
        "other tenant MENTIONS untouched"
    )
    assert ("other_agent", other_id) in graph.entities, "other tenant entity kept"
    assert ("other_agent", "docX", other_id) in graph.mentions, (
        "other tenant MENTIONS kept"
    )
    # agent_test's own doc got its MyTool entity + MENTIONS.
    mytool_id = _hash("mytool", tenant="agent_test")
    assert ("agent_test", mytool_id) in graph.entities
    assert ("agent_test", "doc1", mytool_id) in graph.mentions


def main():
    _install_stubs()

    # In-memory package pointing at the repo root: imports the plugin modules
    # directly (relative imports resolved via __path__) without executing the
    # plugin __init__.py side effects.
    _pkg = types.ModuleType("catgraphrag_tertest")
    _pkg.__path__ = [REPO_ROOT]
    sys.modules["catgraphrag_tertest"] = _pkg

    global graphrag_handler, main, entity_extractor, models
    from catgraphrag_tertest import graphrag_handler  # noqa: E402
    from catgraphrag_tertest import main  # noqa: E402
    from catgraphrag_tertest import entity_extractor  # noqa: E402
    from catgraphrag_tertest import models  # noqa: E402

    tests = [
        test_hook_ignores_other_vector_databases,
        test_hook_ignores_unchanged_patterns,
        test_hook_rebuilds_extractor_on_pattern_change,
        test_refresh_adds_technology_entities_and_mentions,
        test_refresh_removes_stale_mentions_and_prunes_orphans,
        test_refresh_is_idempotent,
        test_refresh_leaves_documents_similar_and_other_entities_untouched,
        test_refresh_is_per_tenant_isolated,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()