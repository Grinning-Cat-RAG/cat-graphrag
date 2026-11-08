"""Test doubles shared by the graph-phase harnesses (``test_graph_phases*.py``).

Import-safe: the Cat plugin loader imports every ``.py`` under the plugin folder, so this module only defines
classes and functions; nothing runs and nothing from ``cat`` is imported at module level.

- ``FakeGraphStore`` stands in for Neo4j at the handler's storage boundary (documents, entity edges, RELATED_TO
  provenance, similarity edges, concepts). It mirrors the semantics of the real Cypher of each method it replaces;
  the Cypher itself is exercised against a live Neo4j by ``test_graph_purge_live.py``.
- ``FakeCat`` / ``FakeRabbitHole`` drive the REAL ``cat.rabbit_hole.engine`` of the core: the built-in embedding
  phase is replaced by one that replaces the points of the source exactly like the core does (new points first,
  then the old ones by id).
"""
import json
import os
import types
import uuid

PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Redis database of the harnesses: never the one of a running Cat (0) nor the one of the core test suite (1).
TEST_REDIS_DB = "14"


def bootstrap(package_name):
    """Prepare the REAL core (env vars, Redis on localhost) and load the plugin as ``package_name``.

    Returns ``(main, graphrag_handler, phases)`` modules. Must run before anything imports ``cat``.
    """
    import importlib
    import sys

    os.environ.setdefault("CAT_REDIS_HOST", "localhost")
    os.environ.setdefault("CAT_REDIS_PORT", "6379")
    os.environ["CAT_REDIS_DB"] = TEST_REDIS_DB
    os.environ.setdefault("CAT_CRYPTO_KEY", "test_only_crypto_key_0123456789abcdef")
    os.environ.setdefault("CAT_CRYPTO_SALT", "test_only_crypto_salt_0123456789")
    os.environ.setdefault("CAT_JWT_SECRET", "test_only_jwt_secret_0123456789_abcdefghij")
    os.environ.setdefault("CAT_INGESTION_RESUME_INTERVAL_SECONDS", "0")
    package = types.ModuleType(package_name)
    package.__path__ = [PLUGIN_DIR]
    sys.modules[package_name] = package
    return (
        importlib.import_module(f"{package_name}.main"),
        importlib.import_module(f"{package_name}.graphrag_handler"),
        importlib.import_module(f"{package_name}.phases"),
    )


async def create_agent(agent_id):
    """An agent the engine considers alive (its settings key exists)."""
    from cat.db import crud
    from cat.db.cruds import settings as crud_settings

    await crud.store(crud_settings.format_key(agent_id), [])


async def destroy_agent(agent_id):
    from cat.db.cruds import ingestions as crud_ingestions
    from cat.db.cruds import settings as crud_settings

    await crud_ingestions.destroy_all(agent_id)
    await crud_settings.destroy_all(agent_id)


def make_handler(handler_module, agent_id, **overrides):
    """A GraphRAGHandler on the given agent, every material switch on unless overridden."""
    kwargs = dict(
        neo4j_uri="bolt://fake",
        neo4j_user="u",
        neo4j_password="p",
        enable_entity_extraction=True,
        enable_concept_relations=True,
        spacy_models={"en": "en_core_web_sm"},
    )
    kwargs.update(overrides)
    handler = handler_module.GraphRAGHandler(**kwargs)
    handler.agent_id = agent_id
    return handler


async def ingest(engine, rabbit_hole, cat, chat_id, name):
    """Upload (or upload again) a source: a fresh row, then its phases, under the lock of the source."""
    from cat.db.cruds import ingestions as crud_ingestions
    from cat.rabbit_hole.phases import IngestionSource

    source = IngestionSource(cat.agent_key, chat_id, name)
    async with engine.source_lock(source):
        await crud_ingestions.create_status(cat.agent_key, chat_id, name, type_="file", metadata={})
        return await engine.run_phases(rabbit_hole, cat, source)


async def rerun(engine, rabbit_hole, cat, chat_id, name, invalidate=None):
    """What the automatic resume / the maintenance does on one source (``RabbitHole.rerun_source``)."""
    from cat.db.cruds import ingestions as crud_ingestions
    from cat.rabbit_hole.phases import IngestionSource

    source = IngestionSource(cat.agent_key, chat_id, name)
    async with engine.source_lock(source):
        if invalidate:
            await crud_ingestions.reset_phase_entries(cat.agent_key, chat_id, name, invalidate)
        return await engine.run_phases(rabbit_hole, cat, source)


async def delete(engine, rabbit_hole, cat, chat_id, name):
    from cat.rabbit_hole.phases import IngestionSource

    source = IngestionSource(cat.agent_key, chat_id, name)
    async with engine.source_lock(source):
        await engine.delete_source(rabbit_hole, cat, source)


def content_entities(content):
    """The entities a chunk mentions: its capitalized words (deterministic stand-in for spaCy)."""
    return [word.strip(".,") for word in content.split() if word[:1].isupper()]


class FakeGraphStore:
    """In-memory graph: the state the handler's Neo4j queries read and write."""

    def __init__(self):
        self.docs = {}  # id -> {"collection", "content", "metadata", "vector"}
        self.mentions = set()  # (doc_id, entity_id)
        self.provenance = set()  # (doc_id, entity_id)
        self.entities = {}  # id -> {"tracked": bool}
        self.related = {}  # (s, t, type) -> set(source names)
        self.similar = set()  # doc ids holding a similarity edge
        self.fail_writes = 0  # the next N graph writes raise (Neo4j unavailable)
        self.concept_outcomes = []  # scripted LLM outcomes ("done", "not_ready", "error"), consumed in order
        self.writes = []  # (operation, n_entities, n_relations) of every successful graph write
        self.purges = []  # (source, collection, chat_id)
        self.embedders_used = []  # the handler's embedder at every chunk extraction

    # ── documents ────────────────────────────────────────────────────────────
    def _matches(self, doc, collection, metadata):
        if doc["collection"] != collection:
            return False
        meta = doc["metadata"]
        return all(meta.get(k) == v for k, v in (metadata or {}).items())

    def scoped_doc_ids(self, source, collection, chat_id):
        wanted = {"source": source} | ({"chat_id": chat_id} if chat_id else {})
        return {d for d, doc in self.docs.items() if self._matches(doc, collection, wanted)}

    def _drop_docs(self, doc_ids):
        for d in doc_ids:
            self.docs.pop(d, None)
        self.mentions = {(d, e) for d, e in self.mentions if d not in doc_ids}
        self.provenance = {(d, e) for d, e in self.provenance if d not in doc_ids}
        self.similar -= set(doc_ids)

    def _strip_related(self, source):
        for key in list(self.related):
            self.related[key].discard(source)
            if not self.related[key]:
                del self.related[key]

    def _prune(self, candidates=None, tracked_only=False):
        linked = {e for _, e in self.mentions} | {e for _, e in self.provenance}
        for e in list(self.entities if candidates is None else candidates):
            if e in linked or e not in self.entities:
                continue
            if tracked_only and not self.entities[e]["tracked"]:
                continue
            del self.entities[e]
            for key in [k for k in self.related if e in (k[0], k[1])]:
                del self.related[key]

    # ── handler boundary ─────────────────────────────────────────────────────
    def install(self, handler, system_embedder=None):
        """Replace the handler's storage methods with this store (the orchestration code stays real)."""
        store = self

        async def noop(*args, **kwargs):
            return None

        async def read_generation(*args, **kwargs):
            return "v1"

        handler._ensure_connected = noop
        handler._read_generation = read_generation
        handler._generation = "v1"
        handler._names = {"embedding_prop": "embedding_v1"}
        handler._get_session = lambda: _FakeSession(store)
        handler.create_derived_graph_for_source = noop
        handler.agent_id = handler.agent_id if getattr(handler, "agent_id", None) else "agent_fake"

        async def get_all_tenant_points(collection_name, limit=None, offset=None, metadata=None, with_vectors=True):
            # the core's contract: ``limit=None`` means every point, a limit means one page
            skip = int(offset) if offset and str(offset).isdigit() else 0
            rows = sorted(d for d, doc in store.docs.items() if store._matches(doc, collection_name, metadata))
            page = limit or max(len(rows), 1)
            chosen = rows[skip:skip + page]
            points = [
                types.SimpleNamespace(
                    id=d,
                    payload={"id": d, "page_content": store.docs[d]["content"], "metadata": dict(store.docs[d]["metadata"])},
                    vector=store.docs[d]["vector"] if with_vectors else None,
                )
                for d in chosen
            ]
            return points, (str(skip + len(chosen)) if limit and len(chosen) == page else None)

        async def create_similarity(point_id, vector, collection_name):
            if point_id in store.docs:
                store.similar.add(point_id)

        async def extract(document_id, content, metadata):
            store.embedders_used.append(handler._embedder)
            names = content_entities(content)
            source = (metadata or {}).get("source")
            return {
                "entities": [{"id": n} for n in names],
                "mentions": [{"document_id": document_id, "entity_id": n} for n in names],
                "relations": [
                    {"source_id": a, "target_id": b, "rel_type": "NEXT_TO", "source_file": source}
                    for a, b in zip(names, names[1:])
                ],
            }

        async def write_source_graph(entities, mentions, relations, operation="write source graph"):
            if store.fail_writes > 0:
                store.fail_writes -= 1
                raise RuntimeError("neo4j unavailable")
            for e in entities:
                store.entities.setdefault(e["id"], {"tracked": True})
            for m in mentions:
                if m["document_id"] in store.docs:
                    store.mentions.add((m["document_id"], m["entity_id"]))
                    store.provenance.add((m["document_id"], m["entity_id"]))
            for r in relations:
                store.related.setdefault((r["source_id"], r["target_id"], r["rel_type"]), set()).add(r["source_file"])
            store.writes.append((operation, len(entities), len(relations)))
            return 1

        async def delete_tenant_points(collection_name, metadata=None):
            handler._drop_source_tasks((metadata or {}).get("source"), (metadata or {}).get("chat_id"))
            ids = {d for d, doc in store.docs.items() if store._matches(doc, collection_name, metadata)}
            store._drop_docs(ids)
            source = (metadata or {}).get("source")
            if source:
                store._strip_related(source)
                store._prune(tracked_only=True)
            else:
                store._prune()

        async def delete_tenant_points_by_ids(collection_name, points_ids):
            ids = {d for d in points_ids if d in store.docs and store.docs[d]["collection"] == collection_name}
            store._drop_docs(ids)
            store._prune()

        async def purge(source, collection_name, chat_id=None):
            handler._drop_source_tasks(source, chat_id)
            store.purges.append((source, collection_name, chat_id))
            ids = store.scoped_doc_ids(source, collection_name, chat_id)
            candidates = {e for d, e in store.mentions | store.provenance if d in ids}
            store.mentions = {(d, e) for d, e in store.mentions if d not in ids}
            store.provenance = {(d, e) for d, e in store.provenance if d not in ids}
            store.similar -= ids
            store._prune(candidates)
            store._strip_related(source)
            return len(ids)

        async def concepts(source, stored_points, stray_cat=None, gen=None, llm=None):
            outcome = store.concept_outcomes.pop(0) if store.concept_outcomes else "done"
            if outcome == "not_ready":
                return {"status": "not_ready", "retry_after": 0, "reason": "llm busy"}
            if outcome == "error":
                return {"status": "error", "reason": "llm refused"}
            if store.fail_writes > 0:
                store.fail_writes -= 1
                raise RuntimeError("neo4j unavailable")
            for p in stored_points:
                words = content_entities(p.payload["page_content"])
                if not words:
                    continue
                concept = f"C:{words[0]}"
                store.entities.setdefault(concept, {"tracked": True})
                store.provenance.add((p.id, concept))
                store.related.setdefault((concept, words[0], "DESCRIBES"), set()).add(source)
            return {"status": "done", "processed": len(stored_points)}

        handler.get_all_tenant_points = get_all_tenant_points
        handler._create_similarity_relationships = create_similarity
        handler._extract_document_graph = extract
        handler._write_source_graph = write_source_graph
        handler.delete_tenant_points = delete_tenant_points
        handler.delete_tenant_points_by_ids = delete_tenant_points_by_ids
        handler.purge_source_graph_artifacts = purge
        handler._extract_concept_relations = concepts

        async def resolve_system_embedder():
            return system_embedder

        handler._system_embedder = resolve_system_embedder
        return handler

    # ── expectations ─────────────────────────────────────────────────────────
    def expected_for(self, doc_ids, concepts_enabled):
        """The graph a source must have once its phases completed, derived ONLY from its current documents:
        ``(mentions, provenance, related)``."""
        mentions, provenance, related = set(), set(), set()
        for d in doc_ids:
            names = content_entities(self.docs[d]["content"])
            mentions |= {(d, n) for n in names}
            provenance |= {(d, n) for n in names}
            related |= {(a, b, "NEXT_TO") for a, b in zip(names, names[1:])}
            if concepts_enabled and names:
                provenance.add((d, f"C:{names[0]}"))
                related.add((f"C:{names[0]}", names[0], "DESCRIBES"))
        return mentions, provenance, related

    def actual_for(self, doc_ids, source):
        return (
            {(d, e) for d, e in self.mentions if d in doc_ids},
            {(d, e) for d, e in self.provenance if d in doc_ids},
            self.related_of(source),
        )

    def related_of(self, source):
        return {key for key, sources in self.related.items() if source in sources}


class _FakeResult:
    def __init__(self, record):
        self._record = record

    async def single(self):
        return self._record

    async def consume(self):
        return None


class _FakeSession:
    """The two statements ``add_point_to_tenant`` runs: the Document CREATE and the conditional embedding SET."""

    def __init__(self, store):
        self.store = store

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, query, **params):
        if "CREATE (d:Document" in query:
            self.store.docs[params["id"]] = {
                "collection": params["collection_name"],
                "content": params["content"],
                "metadata": json.loads(params["metadata"]),
                "vector": None,
            }
            return _FakeResult({"id": params["id"]})
        if "SET d." in query and params.get("id") in self.store.docs:
            self.store.docs[params["id"]]["vector"] = params["embedding"]
        return _FakeResult(None)


class FakePluginManager:
    def __init__(self, main_module):
        self.main = main_module

    async def execute_hook(self, name, *args, caller=None):
        if name == "rabbithole_ingestion_phases":
            hook = self.main.rabbithole_ingestion_phases  # a CatHook under the real ``@hook``
            return getattr(hook, "function", hook)(list(args[0]), caller)
        return args[0] if args else None


class FakeCat:
    """The CheshireCat the engine passes to the phases: agent key, plugin manager, vector handler, LLM."""

    def __init__(self, agent_key, handler, main_module):
        self.agent_key = agent_key
        self.vector_memory_handler = handler
        self.plugin_manager = FakePluginManager(main_module)
        self.large_language_model = types.SimpleNamespace(name="fake-llm")


class FakeRabbitHole:
    """The built-in phases: parsing is a no-op, embedding replaces the points of the source like the core does."""

    def __init__(self, files, point_struct):
        self.files = files  # (chat_id, name) -> list of chunk texts (the current content)
        self.PointStruct = point_struct

    @staticmethod
    def collection(chat_id):
        return "declarative" if chat_id is None else "episodic"

    async def run_parsing_chunking(self, source, cat):
        from cat.rabbit_hole.phases import PhaseResult

        return PhaseResult(status="done", processed=len(self.files[(source.chat_id, source.name)]))

    async def delete_parsing_chunking(self, source, cat):
        return None

    async def run_embedding(self, source, cat):
        from cat.rabbit_hole.phases import PhaseResult

        handler = cat.vector_memory_handler
        collection = self.collection(source.chat_id)
        meta_filter = {"source": source.name} | ({"chat_id": source.chat_id} if source.chat_id else {})
        old, _ = await handler.get_all_tenant_points(collection, metadata=meta_filter, with_vectors=False)
        points = [
            self.PointStruct(
                id=uuid.uuid4().hex,
                payload={
                    "page_content": text,
                    "metadata": {"source": source.name, "chunk_index": i}
                    | ({"chat_id": source.chat_id} if source.chat_id else {}),
                },
                vector=[1.0, float(i + 1)],
            )
            for i, text in enumerate(self.files[(source.chat_id, source.name)])
        ]
        await handler.add_points_to_tenant(collection_name=collection, points=points)
        new_ids = {p.id for p in points}
        old_ids = [p.id for p in old if p.id not in new_ids]
        if old_ids:
            await handler.delete_tenant_points_by_ids(collection, old_ids)
        return PhaseResult(status="done", processed=len(points))

    async def delete_embedding(self, source, cat):
        meta_filter = {"source": source.name} | ({"chat_id": source.chat_id} if source.chat_id else {})
        await cat.vector_memory_handler.delete_tenant_points(self.collection(source.chat_id), meta_filter)
