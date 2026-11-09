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

#: The names of the core's collections (``str(VectorMemoryType.X)``): set by ``bootstrap``, which loads the core (this
#: module imports nothing from ``cat`` at import time).
DECLARATIVE = EPISODIC = PROCEDURAL = None


class StubVectorMemoryType(str):
    """The core's ``VectorMemoryType`` (cat/services/memory/models.py) for the harnesses that stub the core instead of
    loading it: ``str()`` of a member is its value, as in the core. The one copy: ``test_graph_phases.py`` checks it
    against the real enum."""

    EPISODIC = "episodic"
    DECLARATIVE = "declarative"
    PROCEDURAL = "procedural"


def install_ingestion_registry_stub():
    """For the harnesses that stub the core: ``cat.db.cruds.ingestions`` answering that the source's ingestion is
    writing (row ``processing`` on the embedding phase), the case those harnesses exercise: the points of a write
    start the extraction tasks the spaCy phase joins."""
    import sys

    db = sys.modules.setdefault("cat.db", types.ModuleType("cat.db"))
    db.__path__ = getattr(db, "__path__", [])
    cruds = sys.modules.setdefault("cat.db.cruds", types.ModuleType("cat.db.cruds"))
    cruds.__path__ = getattr(cruds, "__path__", [])
    ingestions = types.ModuleType("cat.db.cruds.ingestions")

    async def get_status(agent_id, chat_id, source):
        return {"source": source, "chat_id": chat_id, "status": "processing", "phase": "embedding"}

    async def reset_phase_entries(agent_id, chat_id, source, phase_ids):
        return True

    ingestions.get_status = get_status
    ingestions.reset_phase_entries = reset_phase_entries
    cruds.ingestions = ingestions
    sys.modules["cat.db.cruds.ingestions"] = ingestions


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
    global DECLARATIVE, EPISODIC, PROCEDURAL
    from cat.services.memory.models import VectorMemoryType

    DECLARATIVE = str(VectorMemoryType.DECLARATIVE)
    EPISODIC = str(VectorMemoryType.EPISODIC)
    PROCEDURAL = str(VectorMemoryType.PROCEDURAL)
    return (
        importlib.import_module(f"{package_name}.main"),
        importlib.import_module(f"{package_name}.graphrag_handler"),
        importlib.import_module(f"{package_name}.phases"),
    )


async def create_agent(agent_id, graphrag=False):
    """An agent the engine considers alive (its settings key exists); with ``graphrag``, the plugin ``cat_graphrag``
    is active for it and its vector database is the GraphRAG one (what the maintenance CLI reads)."""
    from cat.db import crud
    from cat.db.cruds import settings as crud_settings

    settings = []
    if graphrag:
        settings = [
            {"name": "active_plugins", "value": ["cat_graphrag"]},
            {"name": "Neo4jGraphRAGConfig", "category": "vector_database", "value": {}},
        ]
    await crud.store(crud_settings.format_key(agent_id), settings)


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
        self.wipes = 0  # tenant-wide graph wipes
        self.concept_fingerprint = "fp-1"  # the fingerprint of the current concept settings
        self.concept_marker = None  # the persisted active concept generation
        self.concept_gcs = 0
        self.key_of = None  # the provenance key of the handler module (set by install)

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

    def _strip_related(self, keep):
        """Remove the provenance keys ``keep`` rejects (a key, or a predicate), and the relations left without one."""
        rejects = keep if callable(keep) else (lambda k: k == keep)
        for key in list(self.related):
            self.related[key] = {k for k in self.related[key] if not rejects(k)}
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
        import sys as _sys

        store = self
        # the provenance key of the REAL handler module: the fake keys relations exactly as the real code does
        key_of = _sys.modules[type(handler).__module__].provenance_key
        store.key_of = key_of

        async def noop(*args, **kwargs):
            return None

        async def read_generation(*args, **kwargs):
            return "v1"

        handler._ensure_connected = noop
        handler._read_generation = read_generation
        handler._generation = "v1"
        handler._names = {"embedding_prop": "embedding_v1"}
        handler._get_session = lambda: _FakeSession(store)
        handler._fake_store = store
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
            source = key_of(source, (metadata or {}).get("chat_id")) if source else None
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
            source, chat_id = (metadata or {}).get("source"), (metadata or {}).get("chat_id")
            if source:
                store._strip_related(key_of(source, chat_id))
                store._prune(tracked_only=True)
            elif metadata and set(metadata) == {"chat_id"}:  # a chat destroyed: every key of the chat
                store._strip_related(lambda k: k.startswith(key_of("", chat_id)))
                store._prune()
            elif not metadata and collection_name == DECLARATIVE:  # the whole knowledge base
                store._strip_related(lambda k: not k.startswith("chat:"))
                store._prune()
            elif not metadata and collection_name == EPISODIC:  # every chat's memory
                store._strip_related(lambda k: k.startswith("chat:"))
                store._prune()
            else:  # another collection (e.g. procedural) or another filter: no provenance to strip
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
            store._strip_related(key_of(source, chat_id))
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
                chat_id = (p.payload.get("metadata") or {}).get("chat_id")
                store.related.setdefault((concept, words[0], "DESCRIBES"), set()).add(key_of(source, chat_id))
            return {"status": "done", "processed": len(stored_points)}

        async def wipe_tenant_graph():
            # the Cypher of ``GraphRAGHandler.wipe_tenant_graph`` (run on a live Neo4j by test_graph_purge_live.py):
            # every entity edge, similarity edge and RELATED_TO goes, then the entities left without edges
            store.mentions.clear()
            store.provenance.clear()
            store.related.clear()
            store.similar.clear()
            store.entities.clear()
            store.wipes += 1

        async def concept_fingerprint():
            return store.concept_fingerprint

        async def read_concept_gen(tenant_id=None):
            return store.concept_marker

        async def flip_concept_gen(tenant_id, expected_prev_gen, new_gen):
            if store.concept_marker != expected_prev_gen:
                return False
            store.concept_marker = new_gen
            return True

        async def gc_stale_concept_nodes(tenant_id):
            store.concept_gcs += 1

        handler.wipe_tenant_graph = wipe_tenant_graph
        handler._concept_fingerprint = concept_fingerprint
        handler._read_concept_gen = read_concept_gen
        handler._flip_concept_gen = flip_concept_gen
        handler._gc_stale_concept_nodes = gc_stale_concept_nodes
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

    def actual_for(self, doc_ids, source, chat_id=None):
        return (
            {(d, e) for d, e in self.mentions if d in doc_ids},
            {(d, e) for d, e in self.provenance if d in doc_ids},
            self.related_of(source, chat_id),
        )

    def related_of(self, source, chat_id=None):
        """The relations whose provenance includes the source (its key: the name in the knowledge base, scoped by
        the chat in a chat)."""
        key = self.key_of(source, chat_id)
        return {k for k, keys in self.related.items() if key in keys}


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
        return DECLARATIVE if chat_id is None else EPISODIC

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


class FakeFileManager:
    """The uploaded files the core reads back: the bytes of a source exist iff ``(chat_id, name)`` is kept."""

    def __init__(self, kept):
        self.kept = kept  # set of (chat_id, name)

    def read_file(self, name, folder):
        parts = folder.split("/", 1)
        chat_id = parts[1] if len(parts) > 1 else None
        return b"bytes" if (chat_id, name) in self.kept else None


def core_rabbit_hole(files, point_struct):
    """The REAL core ``RabbitHole`` (``rerun_source`` / ``reingest_source`` / locks / rows), bound to a fake cat, with
    the built-in parsing and embedding of ``FakeRabbitHole``. A source whose content is gone fails its parsing."""
    from cat.rabbit_hole import RabbitHole
    from cat.rabbit_hole.phases import PhaseResult

    builtin = FakeRabbitHole(files, point_struct)

    class CoreRabbitHole(RabbitHole):
        async def setup(self, _cat):
            self.cat, self.stray = _cat, None

        async def run_parsing_chunking(self, source, cat):
            if (source.chat_id, source.name) not in files:
                return PhaseResult(status="error", reason=f"Cannot parse {source.name}")
            return await builtin.run_parsing_chunking(source, cat)

        async def delete_parsing_chunking(self, source, cat):
            return await builtin.delete_parsing_chunking(source, cat)

        async def run_embedding(self, source, cat):
            return await builtin.run_embedding(source, cat)

        async def delete_embedding(self, source, cat):
            return await builtin.delete_embedding(source, cat)

    return CoreRabbitHole()


class FakeAgentCat(FakeCat):
    """The CheshireCat the maintenance CLI works on: the core RabbitHole and the uploaded files."""

    def __init__(self, agent_key, handler, main_module, rabbit_hole, kept):
        super().__init__(agent_key, handler, main_module)
        self.rabbit_hole = rabbit_hole
        self.file_manager = FakeFileManager(kept)
        self.kept = kept

    async def _find_stray_cat(self, chat_id):
        return None  # the owner of the chat is not needed by the fake phases


def load_cli(handler_module, phases_module):
    """The maintenance CLI module (``scripts/maintenance_agent.py``), bound to the plugin modules under test."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "maintenance_agent", os.path.join(PLUGIN_DIR, "scripts", "maintenance_agent.py"),
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    modules = {"graphrag_handler": handler_module, "phases": phases_module}
    cli._resolve_plugin_id = lambda: "cat_graphrag"
    cli._plugin_module = lambda name: modules[name]
    return cli


async def run_cli(cli, *argv):
    """Run the CLI's ``main`` with ``argv``; returns ``(exit code, stdout lines)``."""
    import contextlib
    import io
    import sys

    out = io.StringIO()
    saved = sys.argv
    sys.argv = ["maintenance_agent", *argv]
    code = None
    try:
        with contextlib.redirect_stdout(out):
            await cli.main()
    except SystemExit as exc:
        code = exc.code
    finally:
        sys.argv = saved
    return code, out.getvalue().splitlines()


class CliWorld:
    """One agent as the maintenance CLI sees it: its graph, the content of its sources, the files kept, its cat."""

    def __init__(self, agent, main_module, handler_module, point_struct, concepts=True):
        self.agent = agent
        self.point_struct = point_struct
        self.store = FakeGraphStore()
        self.files = {}  # (chat_id, name) -> the content of the source (its kept bytes, parsed)
        self.kept = set()  # (chat_id, name) whose bytes are kept
        self.handler = self.store.install(make_handler(handler_module, agent, enable_concept_relations=concepts))
        self.rh = core_rabbit_hole(self.files, point_struct)
        self.cat = FakeAgentCat(agent, self.handler, main_module, self.rh, self.kept)

    async def upload(self, name, chunks, chat=None, keep=True):
        """An upload through the server: a fresh row, then the phases (``keep``: the bytes are stored)."""
        from cat.rabbit_hole import engine

        self.files[(chat, name)] = chunks
        if keep:
            self.kept.add((chat, name))
        else:
            self.kept.discard((chat, name))
        return await ingest(engine, self.rh, self.cat, chat, name)

    async def legacy(self, name, chunks, chat=None, custom=None):
        """A source ingested before the states existed: its points (and graph), its stored file, no row. ``custom``:
        the metadata of its upload, on every chunk."""
        import asyncio

        self.files[(chat, name)] = chunks
        self.kept.add((chat, name))
        meta = (custom or {}) | {"source": name} | ({"chat_id": chat} if chat else {})
        points = [
            self.point_struct(
                id=uuid.uuid4().hex,
                payload={"page_content": text, "metadata": meta | {"chunk_index": i}},
                vector=[1.0, float(i + 1)],
            )
            for i, text in enumerate(chunks)
        ]
        await self.handler.add_points_to_tenant(FakeRabbitHole.collection(chat), points)
        await asyncio.gather(*self.handler._pending_entity_tasks, return_exceptions=True)

    async def delete(self, name, chat=None):
        from cat.rabbit_hole import engine

        await delete(engine, self.rh, self.cat, chat, name)
        self.files.pop((chat, name), None)
        self.kept.discard((chat, name))

    async def row(self, name, chat=None):
        from cat.db.cruds import ingestions as crud_ingestions

        return await crud_ingestions.get_status(self.agent, chat, name)

    def graph_ok(self, name, chat=None, concepts=True):
        """The source holds exactly the documents of its content and the graph derived from them."""
        ids = self.store.scoped_doc_ids(name, FakeRabbitHole.collection(chat), chat)
        if sorted(self.store.docs[d]["content"] for d in ids) != sorted(self.files[(chat, name)]):
            return False
        return self.store.actual_for(ids, name, chat) == self.store.expected_for(ids, concepts)


async def current_tags(rabbit_hole, cat):
    """The current tag of every phase of the agent (what the core's automatic resume compares the diaries with)."""
    from cat.rabbit_hole import engine
    from cat.rabbit_hole.phases import compute_tags

    return await compute_tags(await engine.collect_phases(rabbit_hole, cat), cat)
