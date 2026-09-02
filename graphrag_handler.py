import random
import math
import uuid
import json
import asyncio
from typing import Any, List, Iterable, Dict, Tuple, Optional, AsyncContextManager, cast, LiteralString, Type
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import Neo4jError
from langchain_core.documents import Document as LangChainDocument
from pydantic import Field, ConfigDict

from cat import BaseVectorDatabaseHandler, Embeddings, VectorDatabaseSettings, AgenticWorkflowTask
from cat.services.memory.models import (
    DocumentRecall, PointStruct, Record, ScoredPoint, UpdateResult
)
from cat.log import log

# Migration guard for the seamless re-embed swap. Prefer the
# Cat distributed lock (Redis-backed, cross-worker); fall back to a simple
# in-process asyncio lock when the Cat runtime is unavailable (e.g. the
# standalone test harness stubs ``cat`` without ``cat.db.crud``).
try:
    from cat.db.crud import distributed_lock
except ImportError:
    distributed_lock = None

from .entity_extractor import EntityExtractor
from .epoch import EpochMixin
from .models import EntityType
from .versioning import ensure_version, retry_on_generation_change


class GraphRAGHandler(EpochMixin, BaseVectorDatabaseHandler):
    """
    Advanced GraphRAG handler with:
    - Neo4j 5.23+ vector indexes (HNSW)
    - Entity extraction with spaCy
    - Knowledge graph with entities and semantic relations
    - Hybrid retrieval (vector + graph + entity expansion)
    """
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        neo4j_database: str = "neo4j",
        neo4j_kwargs: Dict = None,
        document_vector_index: str = "document_embeddings",
        entity_vector_index: str = "entity_embeddings",
        vector_similarity_threshold: float = 0.7,
        enable_entity_extraction: bool = True,
        enable_entity_embeddings: bool = False,
        enable_entity_expansion: bool = True,
        spacy_models: Dict[str, str] = None,
        extra_technology_patterns: List[str] | None = None,
        graph_retrieval_depth: int = 2,
        graph_decay_factor: float = 0.5,
        connection_pool_size: int = 50,
        enable_derived_graph: bool = False,
        enable_concept_relations: bool = False,
        concept_relations_prompt: str | None = None,
        enable_knowledge_graph: bool = False,
        enable_student_knowledge_graph: bool = False,
        save_memory_snapshots: bool = False,
    ):
        super().__init__(save_memory_snapshots=save_memory_snapshots)

        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._neo4j_database = neo4j_database
        self._neo4j_kwargs = neo4j_kwargs or {}
        self._document_vector_index = document_vector_index
        self._entity_vector_index = entity_vector_index
        self._vector_similarity_threshold = vector_similarity_threshold
        self._enable_entity_extraction = enable_entity_extraction
        self._enable_entity_embeddings = enable_entity_embeddings
        self._enable_entity_expansion = enable_entity_expansion
        self._spacy_models = spacy_models or {"en": "en_core_web_sm"}
        self._extra_technology_patterns=extra_technology_patterns
        self._graph_retrieval_depth=graph_retrieval_depth
        self._graph_decay_factor=graph_decay_factor
        self._connection_pool_size=connection_pool_size
        self._enable_derived_graph = enable_derived_graph
        self._enable_concept_relations = enable_concept_relations
        self._concept_relations_prompt = concept_relations_prompt
        self._enable_knowledge_graph = enable_knowledge_graph
        self._enable_student_knowledge_graph = enable_student_knowledge_graph

        self._driver: Optional[AsyncDriver] = None
        self._pending_entity_tasks: List[asyncio.Task] = []
        # Semaphore: caps concurrent Neo4j write transactions to reduce lock
        # contention and deadlock probability during bulk PDF ingestion.
        # Shared between entity extraction and similarity writes.
        self._neo4j_write_semaphore = asyncio.Semaphore(4)
        self._user_message = None
        self._embedder: Optional[Embeddings] = None
        # In-process fallback lock for reembed_tenant when the Cat distributed
        # lock is unavailable (see the module-level try/except import).
        self._reembed_lock = asyncio.Lock()

        # Lazy embedder-alignment guard: the embedder is injected by the
        # plugin hooks AFTER the core bootstrap, so initialize() cannot rely
        # on it. The first post-hook read/ingestion path performs the
        # alignment (index-dims fix or shadow re-embed) exactly once.
        self._alignment_lock = asyncio.Lock()
        self._alignment_done = False

        # Versioned-schema state: the generation token read from
        # (:Epoch {tenant_id, generation}), the resolved version-suffixed names,
        # and the per-generation compiled-query cache {query_key: {gen: cypher}}.
        self._generation: Optional[str] = None
        self._names: Dict[str, str] = {}
        self._query_cache: Dict[str, Dict[str, str]] = {}

        # Initialize entity extractor
        self._entity_extractor: Optional[EntityExtractor] = EntityExtractor(
            models=self._spacy_models,
            extra_technology_patterns=self._extra_technology_patterns or None,
        ) if self._enable_entity_extraction else None
    
    def to_dict(self):
        return {
            "neo4j_uri": self._neo4j_uri,
            "neo4j_user": self._neo4j_user,
            "neo4j_password": self._neo4j_password,
            "neo4j_database": self._neo4j_database,
            "neo4j_kwargs": self._neo4j_kwargs,
            "document_vector_index": self._document_vector_index,
            "entity_vector_index": self._entity_vector_index,
            "vector_similarity_threshold": self._vector_similarity_threshold,
            "enable_entity_extraction": self._enable_entity_extraction,
            "enable_entity_embeddings": self._enable_entity_embeddings,
            "enable_entity_expansion": self._enable_entity_expansion,
            "spacy_models": self._spacy_models,
            "extra_technology_patterns": self._extra_technology_patterns,
            "graph_retrieval_depth": self._graph_retrieval_depth,
            "graph_decay_factor": self._graph_decay_factor,
            "connection_pool_size": self._connection_pool_size,
            "enable_derived_graph": self._enable_derived_graph,
        }

    def _is_valid_vector(self, vector: List[float]) -> bool:
        """Check if a vector has non-zero and finite L2-norm."""
        _l2_sq = sum(x * x for x in vector)
        return _l2_sq != 0.0 and math.isfinite(math.sqrt(_l2_sq))

    @staticmethod
    def _next_generation(gen: str) -> str:
        """Increment a generation token: 'v1' -> 'v2', 'v2' -> 'v3'."""
        try:
            return f"v{int(gen.lstrip('v')) + 1}"
        except ValueError:
            return "v2"

    def _eq(self, other: "GraphRAGHandler") -> bool:
        return self.to_dict() == other.to_dict()

    @property
    def user_message(self) -> Optional[str]:
        return self._user_message

    @user_message.setter
    def user_message(self, value: str):
        self._user_message = value

    @property
    def embedder(self) -> Optional[Embeddings]:
        return self._embedder

    @embedder.setter
    def embedder(self, value: Embeddings):
        self._embedder = value
        
    @property
    def client(self):
        return self._driver

    @property
    def entity_extractor(self) -> EntityExtractor | None:
        return self._entity_extractor
        
    def tenant_field_condition(self) -> Dict:
        return {"key": "tenant_id", "match": {"value": self.agent_id}}
        
    def _get_session(self) -> AsyncContextManager[AsyncSession]:
        if not self._driver:
            raise RuntimeError("Neo4j driver not initialized")
        return self._driver.session(database=self._neo4j_database)
            
    async def _ensure_connected(self):
        if not self._driver:
            await self._connect()
            
    async def _connect(self):
        try:
            self._driver: AsyncDriver = AsyncGraphDatabase.driver(
                self._neo4j_uri,
                auth=(self._neo4j_user, self._neo4j_password),
                max_connection_pool_size=self._connection_pool_size,
                connection_acquisition_timeout=60,
                # Suppress GQL warnings 01N51 / 01N52 ("relationship type / property
                # key does not exist") that Neo4j emits on a fresh database before any
                # schema elements have been written.  These are harmless — queries that
                # match nothing simply return zero rows — but pollute the log on startup.
                notifications_disabled_categories=["UNRECOGNIZED"],
                **self._neo4j_kwargs,
            )
            assert isinstance(self._driver, AsyncDriver)
            async with self._driver.session(database=self._neo4j_database) as session:
                await session.run("RETURN 1")
            await self._backfill_missing_entity_ids()
            log.info(f"Connected to Neo4j at {self._neo4j_uri}")
        except Exception as e:
            log.error(f"Failed to connect to Neo4j: {e}")
            raise

    async def _backfill_missing_entity_ids(self):
        """Set a stable `id` on Entity nodes that lack one (e.g. concept entities
        created by earlier versions of the plugin before `_store_concept_relations`
        started setting the property).  Uses the same MD5 hash strategy as
        `EntityExtractor.get_entity_hash` to guarantee consistency with newly
        inserted entities."""
        try:
            async with self._driver.session(database=self._neo4j_database) as session:
                result = await session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.id IS NULL
                    RETURN e.tenant_id AS tenant_id,
                           e.name AS name,
                           coalesce(e.type, 'CONCEPT') AS etype,
                           elementId(e) AS _elid
                    """
                )
                rows = []
                async for record in result:
                    rows.append(record)
                if not rows:
                    return
                for row in rows:
                    try:
                        enum_type = EntityType(row["etype"])
                    except ValueError:
                        enum_type = EntityType.CONCEPT
                    eid = EntityExtractor.get_entity_hash(
                        row["name"], enum_type, row["tenant_id"]
                    )
                    await session.run(
                        "MATCH (e) WHERE elementId(e) = $_elid SET e.id = $eid",
                        _elid=row["_elid"], eid=eid,
                    )
                log.info(f"[GraphRAG] Backfilled id on {len(rows)} Entity node(s)")
        except Exception as e2:
            log.warning(f"[GraphRAG] Backfill skipped: {e2}")

    async def _ensure_vector_indexes_in_session(self, session, vector_dimensions: int):
        """Creates vector indexes for Document and Entity, using an already opened session."""
        # Index for Document
        doc_index_query = f"""
        CREATE VECTOR INDEX {self._document_vector_index} IF NOT EXISTS
        FOR (d:Document) ON d.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {vector_dimensions},
                `vector.similarity_function`: 'cosine',
                `vector.hnsw.ef_construction`: 200,
                `vector.hnsw.m`: 16
            }}
        }}
        """

        # Index for Entity (optional)
        entity_index_query = f"""
        CREATE VECTOR INDEX {self._entity_vector_index} IF NOT EXISTS
        FOR (e:Entity) ON e.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {vector_dimensions},
                `vector.similarity_function`: 'cosine',
                `vector.hnsw.ef_construction`: 200,
                `vector.hnsw.m`: 16
            }}
        }}
        """

        # Create document index
        try:
            await session.run(doc_index_query)
            log.info(f"Document vector index ensured: {self._document_vector_index}")
        except Exception as e:
            if "already exists" not in str(e):
                log.error(f"Document index creation warning: {e}")
                raise e

        # Create an entity index (if enabled)
        if self._enable_entity_embeddings:
            try:
                await session.run(entity_index_query)
                log.info(f"Entity vector index ensured: {self._entity_vector_index}")
            except Exception as e:
                if "already exists" not in str(e):
                    log.error(f"Entity index creation warning: {e}")
                    raise e

    @staticmethod
    async def _ensure_constraints_in_session(session):
        """Creates integrity constraints using an already opened session."""
        # noinspection SqlNoDataSourceInspection
        constraints = [
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT collection_unique IF NOT EXISTS FOR (c:Collection) REQUIRE (c.name, c.tenant_id) IS UNIQUE",
            # Composite index to speed up entity lookups by (tenant, name) without toLower() overhead
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id, e.name)",
        ]

        for constraint in constraints:
            try:
                await session.run(constraint)
            except Neo4jError as e:
                if "already exists" not in str(e):
                    log.error(f"Constraint creation failed: {e}")
                    raise e

    async def _get_index_dimensions(self, session, index_name: str) -> Optional[int]:
        """
        Returns the current `vector.dimensions` of an existing Neo4j vector index,
        or None if the index does not exist yet.

        Uses SHOW INDEXES (Neo4j 5.x) and filters in Python to avoid
        parameter-support limitations in SHOW commands.
        """
        result = await session.run("SHOW INDEXES YIELD name, type, options WHERE type = 'VECTOR'")
        records = await result.data()
        for record in records:
            if record.get("name") == index_name:
                index_config = (record.get("options") or {}).get("indexConfig", {})
                dims = index_config.get("vector.dimensions")
                if dims is not None:
                    return int(dims)
        return None

    async def _get_collection_embedder_config(
        self, session, collection_name: str
    ) -> Optional[Tuple[str, int]]:
        """
        Returns the (embedder_name, embedder_size) stored on a Collection node,
        or None if the collection does not exist or was created before this field
        was introduced.
        """
        # Avoid IS NOT NULL filters on properties that may not yet exist in the
        # database schema — Neo4j 5.x emits GQL warning 01N52 in that case.
        # Instead, fetch the raw values and perform the null check in Python.
        query = """
        MATCH (c:Collection {name: $name, tenant_id: $tenant_id})
        RETURN c.embedder_name AS embedder_name, c.embedder_size AS embedder_size
        """
        result = await session.run(query, name=collection_name, tenant_id=self.agent_id)
        record = await result.single()
        if record and record["embedder_name"] is not None and record["embedder_size"] is not None:
            return record["embedder_name"], int(record["embedder_size"])
        return None

    async def _drop_vector_indexes_in_session(self, session) -> None:
        """Drops the document (and optional entity) vector index so they can be
        recreated with the new embedder dimensions."""
        for index_name in [
            self._document_vector_index,
            self._entity_vector_index,
        ]:
            try:
                # noinspection SqlNoDataSourceInspection
                await session.run(f"DROP INDEX {index_name} IF EXISTS")
                log.info(f"[GraphRAG] Dropped vector index: {index_name}")
            except Exception as e:
                log.error(f"[GraphRAG] Failed to drop index {index_name}: {e}")

    async def _drop_tenant_data_in_session(self, session) -> None:
        """
        Deletes all Document and Collection nodes belonging to this tenant.
        Orphaned Entity nodes (no remaining MENTIONS) are pruned as well.
        Orphaned SourceFile nodes (no remaining PART_OF) are pruned as well.

        Entity nodes may have RELATED_TO edges, so DETACH DELETE is used
        to avoid relationship-constraint errors from Neo4j.

        Called when an embedder change is detected — all existing embeddings
        are stale and must be discarded before the indexes are rebuilt.
        """
        await session.run(
            """
            MATCH (c:Collection {tenant_id: $tenant_id})<-[:BELONGS_TO]-(d:Document)
            DETACH DELETE d
            """,
            tenant_id=self.agent_id,
        )
        await session.run(
            """
            MATCH (c:Collection {tenant_id: $tenant_id})
            DETACH DELETE c
            """,
            tenant_id=self.agent_id,
        )
        await session.run(
            """
            MATCH (e:Entity {tenant_id: $tenant_id})
            WHERE NOT (e)<-[:MENTIONS]-()
            DETACH DELETE e
            """,
            tenant_id=self.agent_id,
        )
        await session.run(
            """
            MATCH (sf:SourceFile {tenant_id: $tenant_id})
            WHERE NOT (sf)<-[:PART_OF]-()
            DETACH DELETE sf
            """,
            tenant_id=self.agent_id,
        )
        log.info(f"[GraphRAG] Tenant data wiped for agent_id={self.agent_id}")

    async def _ensure_vector_index_in_session(self, session, index_name: str, embedding_prop: str, vector_dimensions: int) -> None:
        """Creates a single Document vector index at the given (versioned) name.

        Mirror of ``_ensure_vector_indexes_in_session`` scoped to one index,
        used by the shadow-build phase of ``reembed_tenant`` to create
        ``document_embeddings_{gen}`` on ``embedding_{gen}``.
        """
        query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (d:Document) ON d.{embedding_prop}
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {vector_dimensions},
                `vector.similarity_function`: 'cosine',
                `vector.hnsw.ef_construction`: 200,
                `vector.hnsw.m`: 16
            }}
        }}
        """
        try:
            await session.run(query)
            log.info(f"Document vector index ensured: {index_name}")
        except Exception as e:
            if "already exists" not in str(e):
                log.error(f"Document index creation warning: {e}")
                raise e

    async def _create_similarity_relationships_for_gen(
        self,
        point_id: str,
        vector: List[float],
        collection_name: str,
        gen: str,
        tenant_id: str,
    ) -> None:
        """Mirror of ``_create_similarity_relationships`` against an explicit
        generation (used by the shadow-build phase of ``reembed_tenant``).

        Queries the ``{gen}`` vector index and writes ``SIMILAR_TO_{gen}``
        edges via plain ``session.run`` — no transactions, exactly like the
        decorated write path.
        """
        if not self._is_valid_vector(vector):
            return
        try:
            similar = await self._run_cached(
                "find_similar",
                gen,
                {
                    "collection_name": collection_name,
                    "tenant_id": tenant_id,
                    "vector": vector,
                    "point_id": point_id,
                    "threshold": self._vector_similarity_threshold,
                },
            )
            if not similar:
                return
            similar.sort(key=lambda s: s["id"])
            async with self._neo4j_write_semaphore:
                async with self._get_session() as session:
                    await session.run(
                        cast(LiteralString, self._compile_query("create_similar_rel", gen)),
                        similar=similar,
                        point_id=point_id,
                    )
        except Exception as e:
            log.error(f"Failed to create similarity relationships: {e}")

    async def _gc_generation(self, tenant_id: str, gen: str) -> None:
        """Drop the old generation's index, SIMILAR_TO edges and embedding property.

        Runs AFTER the epoch flip so no worker is stranded on a deleted set.
        Every statement is tenant-filtered — other agents' graphs are never
        touched.
        """
        names = self._versioned_names(gen)
        async with self._get_session() as session:
            # Drop the old vector index (Neo4j vector indexes are immutable —
            # the new generation carries its own index at the new dims).
            await session.run(cast(LiteralString, f"DROP INDEX {names['index']} IF EXISTS"))
            # Delete the old SIMILAR_TO edges (both endpoints tenant-filtered).
            await session.run(
                cast(
                    LiteralString,
                    f"""
                    MATCH (a:Document {{tenant_id: $tenant_id}})-[r:{names['relation']}]-(b:Document {{tenant_id: $tenant_id}})
                    DELETE r
                    """,
                ),
                tenant_id=tenant_id,
            )
            # Remove the old embedding property from every tenant Document.
            await session.run(
                cast(
                    LiteralString,
                    f"""
                    MATCH (d:Document {{tenant_id: $tenant_id}})
                    REMOVE d.{names['embedding_prop']}
                    """,
                ),
                tenant_id=tenant_id,
            )

    async def reembed_tenant(self, tenant_id: str, new_embedder) -> None:
        """Seamless shadow-swap of the tenant's embedding generation.

        Phase A (shadow): re-embed every stored Document of *tenant_id* with
        ``new_embedder`` into the NEW versioned names (``embedding_{new_gen}``,
        ``document_embeddings_{new_gen}``, ``SIMILAR_TO_{new_gen}``) while the
        OLD generation stays fully intact and keeps serving reads.
        Phase B (flip): atomically set the Epoch generation token to the new
        generation (single MERGE SET).
        Phase C (GC): AFTER the flip, drop the old index, delete the old
        SIMILAR_TO edges and remove the old embedding property.

        No transactions in the read paths; the shadow writes use plain
        ``session.run`` (the batch embedding write is a single UNWIND).
        Cross-worker sync is handled by the ``ensure_version`` /
        ``retry_on_generation_change`` decorators probing the Epoch token on
        their next operation — no broadcast, no canary, no ``WHERE`` guards.
        The whole swap is guarded by the Cat ``distributed_lock`` so only one
        worker performs the migration for a tenant at a time.
        """
        if new_embedder is None:
            log.warning(
                f"[GraphRAG] reembed_tenant({tenant_id}) called without an "
                "embedder; skipping the swap"
            )
            return

        if distributed_lock is not None:
            async with distributed_lock(f"graphrag_reembed:{tenant_id}"):
                await self._reembed_tenant_impl(tenant_id, new_embedder)
        else:
            # Fallback: single-process lock (documented in the module docstring).
            async with self._reembed_lock:
                await self._reembed_tenant_impl(tenant_id, new_embedder)

    async def _reembed_tenant_impl(self, tenant_id: str, new_embedder) -> None:
        """The shadow-build -> flip -> GC sequence, run under the migration lock."""
        old_gen = await self._read_generation(tenant_id)
        new_gen = self._next_generation(old_gen)
        new_names = self._versioned_names(new_gen)

        # ── Phase A: shadow-build the new generation ─────────────────────
        # 1. Read every stored Document of the tenant (with its collection).
        async with self._get_session() as session:
            result = await session.run(
                cast(
                    LiteralString,
                    """
                    MATCH (d:Document {tenant_id: $tenant_id})-[:BELONGS_TO]->(c:Collection)
                    RETURN d.id AS id, d.content AS content, c.name AS collection_name
                    """,
                ),
                tenant_id=tenant_id,
            )
            docs = [
                (record["id"], record["content"] or "", record["collection_name"])
                async for record in result
            ]

        if not docs:
            # Nothing to re-embed — still flip + GC so the schema stays
            # consistent for the tenant.
            await self._set_generation(tenant_id, new_gen)
            await self._gc_generation(tenant_id, old_gen)
            log.info(
                f"[GraphRAG] reembed_tenant({tenant_id}): no documents, "
                f"flipped {old_gen} -> {new_gen}"
            )
            return

        # 2. Re-embed all contents with the new embedder (non-blocking).
        contents = [content for _, content, _ in docs]
        vectors = await asyncio.to_thread(new_embedder.embed_documents, contents)

        # 3. Write embedding_{new_gen} on each Document (single batch UNWIND).
        payload = []
        valid_pairs = []
        for (doc_id, _, collection_name), vector in zip(docs, vectors):
            if self._is_valid_vector(vector):
                payload.append({"id": doc_id, "vector": vector})
                valid_pairs.append((doc_id, vector, collection_name))
            else:
                log.warning(
                    f"[GraphRAG] Skipping re-embed for {doc_id}: "
                    "zero or non-finite vector"
                )

        if payload:
            async with self._get_session() as session:
                await session.run(
                    cast(
                        LiteralString,
                        f"""
                        UNWIND $docs AS d
                        MATCH (doc:Document {{id: d.id, tenant_id: $tenant_id}})
                        SET doc.{new_names["embedding_prop"]} = d.vector
                        """,
                    ),
                    docs=payload,
                    tenant_id=tenant_id,
                )

        # 4. Create the new vector index at the new embedder's dimensions.
        async with self._get_session() as session:
            await self._ensure_vector_index_in_session(
                session,
                new_names["index"],
                new_names["embedding_prop"],
                new_embedder.size,
            )

        # 5. Recompute SIMILAR_TO_{new_gen} from the new vectors (mirrors
        #    _create_similarity_relationships against the new index/relation).
        for doc_id, vector, collection_name in valid_pairs:
            await self._create_similarity_relationships_for_gen(
                doc_id, vector, collection_name, new_gen, tenant_id
            )

        # ── Phase B: flip the epoch (single atomic write) ────────────────
        await self._set_generation(tenant_id, new_gen)

        # ── Phase C: GC the old generation (flip-before-GC is mandatory) ─
        await self._gc_generation(tenant_id, old_gen)

        log.info(
            f"[GraphRAG] Seamless re-embed swap for tenant_id={tenant_id}: "
            f"{old_gen} -> {new_gen} ({len(payload)} documents re-embedded)"
        )

    async def initialize(self, embedder_name: str, embedder_size: int):
        await self._connect()

        async with self._get_session() as session:
            # Constraints are always idempotent — create them first.
            await self._ensure_constraints_in_session(session)

            # ── Detect embedder change ────────────────────────────────────────
            # 1. Dimension mismatch → the HNSW index must be dropped and
            #    recreated (Neo4j vector indexes are immutable once created).
            index_dims = await self._get_index_dimensions(
                session, self._document_vector_index
            )
            index_needs_rebuild = index_dims is not None and index_dims != embedder_size

            # 2. Same dimension but different model → embeddings are in a
            #    different vector space; stale data must be discarded too.
            name_mismatch = False
            if not index_needs_rebuild:
                for collection_name in self._collection_names:
                    stored = await self._get_collection_embedder_config(
                        session, collection_name
                    )
                    if stored is not None:
                        stored_name, stored_size = stored
                        if stored_name != embedder_name or stored_size != embedder_size:
                            name_mismatch = True
                            break

            if index_needs_rebuild or name_mismatch:
                log.warning(
                    f"[GraphRAG] Embedder change detected "
                    f"(index_dims={index_dims} → {embedder_size}, "
                    f"name_mismatch={name_mismatch}). "
                    "Running seamless shadow-swap re-embed."
                )
                if self.save_memory_snapshots:
                    for collection_name in self._collection_names:
                        await self.save_dump(collection_name)

                # Seamless swap instead of the old full wipe: shadow-build the
                # new generation (embedding_v2 / document_embeddings_v2 /
                # SIMILAR_TO_v2), flip the Epoch token, then GC the old set.
                # No downtime — the old generation keeps serving reads until
                # the flip, and no worker is stranded because GC runs after.
                if self._embedder is not None:
                    await self.reembed_tenant(self.agent_id, self._embedder)
                else:
                    log.warning(
                        "[GraphRAG] Embedder change detected but no embedder "
                        "injected; skipping the seamless re-embed swap."
                    )

                if index_needs_rebuild:
                    await self._drop_vector_indexes_in_session(session)

            # Always ensure indexes exist with the correct dimensions.
            # If they were just dropped, this recreates them;
            # if they already match, IF NOT EXISTS is a no-op.
            await self._ensure_vector_indexes_in_session(session, embedder_size)

            # ───────────────── Versioned-schema bootstrap ─────────────────────
            # The decorated read paths query the versioned index/property for
            # the current generation (e.g. document_embeddings_v1 / embedding_v1).
            # On a fresh install — or on pre-P4 data that only carries the
            # unversioned `embedding` property — that versioned schema does not
            # exist yet, so the first memory recall would target a missing index
            # and fail. Ensure the versioned index for the current generation and
            # backfill the legacy unversioned embedding into it, so reads never
            # hit a missing index.
            gen = await self._read_generation()
            if gen != self._generation:
                self._rebuild_for_generation(gen)
            names = self._names
            # Neo4j vector indexes are immutable: `CREATE ... IF NOT EXISTS`
            # is a silent NO-OP when the index already exists at an OLD
            # dimensionality. The versioned index is the one all read paths
            # actually use (find_similar / recall), so repair it synchronously
            # HERE at boot — before any ingestion worker can race ahead and
            # hit the dimension mismatch. embedder_size is a plain parameter,
            # it does not require the embedder object (injected later by hooks).
            v_index_dims = await self._get_index_dimensions(session, names["index"])
            if v_index_dims is not None and v_index_dims != embedder_size:
                log.warning(
                    f"[GraphRAG] Versioned index {names['index']} has "
                    f"{v_index_dims} dims but embedder is {embedder_size}; "
                    "dropping and recreating it at boot (vectors already "
                    "correct, no re-embed needed)."
                )
                await session.run(
                    cast(LiteralString, f"DROP INDEX {names['index']} IF EXISTS")
                )
            await self._ensure_vector_index_in_session(
                session, names["index"], names["embedding_prop"], embedder_size
            )
            # Backfill: copy the legacy unversioned `embedding` property into the
            # current generation's property for existing documents that lack it,
            # BUT only when the stored vector already matches the embedder dims
            # (a stale pre-change vector in the index property would poison the
            # vector index). (The driver suppresses the 01N52 "property does not
            # exist" GQL warning on a fresh database, so the IS NOT NULL filter
            # is safe.)
            await session.run(
                cast(
                    LiteralString,
                    f"""
                    MATCH (d:Document {{tenant_id: $tenant_id}})
                    WHERE d.embedding IS NOT NULL
                      AND d.{names["embedding_prop"]} IS NULL
                      AND size(d.embedding) = $dims
                    SET d.{names["embedding_prop"]} = d.embedding
                    """,
                ),
                tenant_id=self.agent_id,
                dims=embedder_size,
            )

        # Create / update collections — always store current embedder metadata.
        async with self._get_session() as session:
            for collection_name in self._collection_names:
                await self._ensure_collection_exists_in_session(
                    session, collection_name, embedder_name, embedder_size
                )

        log.info(
            f"Advanced GraphRAG initialized "
            f"(embedder={embedder_name}, dims={embedder_size})"
        )

    # NOTE (2026-08-31): this whole method is a CANDIDATE FOR REMOVAL.
    # The boot-time repair in initialize() (versioned index dims vs
    # embedder_size) already covers the dimension-mismatch case this lazy
    # path was originally built for, and the embedder setting is GLOBAL /
    # system-level (DEFAULT_SYSTEM_KEY) — so the multi-tenant ping-pong
    # worry that justified the safety net is unfounded. The only remaining
    # value would be handling a RUNTIME global embedder change (shadow
    # re-embed v1->v2), which is arguably better delegated to the core's
    # BillTheLizard.reembed_all. Keep until that is settled; see Hindsight
    # "Remove lazy embedder alignment path? — pending decision".
    async def _align_embedder_lazy(self) -> None:
        """Align the versioned schema with the current embedder, once per handler.

        The core bootstraps the handler and calls ``initialize()`` BEFORE the
        plugin hooks inject ``self._embedder``, so any embedder-change detection
        inside ``initialize()`` cannot run a shadow re-embed (it would skip with
        "no embedder injected"). This method is invoked from the plugin hooks
        (first recall / first ingestion) when the embedder IS available, and it
        is idempotent: it runs at most once per handler instance.

        Two distinct misalignments are repaired:

        - *Index dimension mismatch only* (vectors already at the right size,
          but the versioned HNSW index was created at an older dimensionality —
          Neo4j vector indexes are immutable, so `IF NOT EXISTS` is a silent
          no-op): drop the stale versioned index and recreate it at
          ``embedder.size``. No re-embed is needed because the vectors are
          already correct.

        - *Embedder name/size mismatch* (the collections were embedded with a
          different model, so even right-dimensioned vectors live in the wrong
          vector space): run the seamless shadow-swap ``reembed_tenant``, which
          builds ``embedding_v2``/``document_embeddings_v2``/``SIMILAR_TO_v2``
          first, then flips the Epoch token so readers atomically move from the
          v1 to the v2 set.
        """
        embedder = self._embedder
        if embedder is None or not getattr(self, "agent_id", None):
            # Nothing to align against; the build/read paths will lazily ensure
            # the index exists at initialize()-time defaults.
            return

        async with self._alignment_lock:
            if self._alignment_done:
                return
            try:
                gen = await self._read_generation()
                if gen != self._generation:
                    self._rebuild_for_generation(gen)
                names = self._names

                async with self._get_session() as session:
                    # 1) Versioned index dimension check (the current
                    #    generation's index is what read paths actually use).
                    index_dims = await self._get_index_dimensions(session, names["index"])
                    # 2) Embedder metadata stored on the tenant's collections.
                    name_mismatch = False
                    for collection_name in self._collection_names:
                        stored = await self._get_collection_embedder_config(
                            session, collection_name
                        )
                        if stored is not None:
                            stored_name, stored_size = stored
                            if stored_name != embedder.name or stored_size != embedder.size:
                                name_mismatch = True
                                break

                if index_dims is not None and index_dims != embedder.size:
                    log.warning(
                        f"[GraphRAG] Versioned index {names['index']} has "
                        f"{index_dims} dims but embedder is {embedder.size}; "
                        "dropping and recreating it (vectors already correct)."
                    )
                    async with self._get_session() as session:
                        await session.run(
                            cast(LiteralString, f"DROP INDEX {names['index']} IF EXISTS")
                        )
                        await self._ensure_vector_index_in_session(
                            session, names["index"], names["embedding_prop"], embedder.size
                        )

                if name_mismatch:
                    log.warning(
                        f"[GraphRAG] Embedder change detected for tenant "
                        f"{self.agent_id} (collections stored embedder differs "
                        f"from {embedder.name} @ {embedder.size}); running "
                        "seamless shadow-swap re-embed."
                    )
                    await self.reembed_tenant(self.agent_id, embedder)

            except Exception as e:
                # Do not block the read/ingestion path on alignment failures;
                # a subsequent call will retry (the flag stays False).
                log.error(f"[GraphRAG] Lazy embedder alignment failed: {e}")
                return
            finally:
                if not self._alignment_done:
                    self._alignment_done = True

    async def close(self):
        # Cancel and clean up all pending entity tasks
        tasks = getattr(self, '_pending_entity_tasks', [])
        if tasks:
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for cancellations to propagate
            await asyncio.gather(*tasks, return_exceptions=True)
            tasks.clear()

        driver = getattr(self, '_driver', None)
        if driver:
            await driver.close()
            self._driver = None

    def is_db_remote(self) -> bool:
        return True

    # ========== COLLECTION METHODS ==========

    async def _ensure_collection_exists_in_session(
        self,
        session,
        collection_name: str,
        embedder_name: str | None = None,
        embedder_size: int | None = None,
    ):
        """
        Creates the Collection node if it does not exist yet.
        `embedder_name` and `embedder_size` are stored (or updated) on the node
        so that future calls to `initialize` can detect an embedder change.
        """
        query = """
        MERGE (c:Collection {name: $name, tenant_id: $tenant_id})
        ON CREATE SET
            c.created_at     = datetime(),
            c.embedder_name  = $embedder_name,
            c.embedder_size  = $embedder_size
        ON MATCH SET
            c.embedder_name  = $embedder_name,
            c.embedder_size  = $embedder_size
        RETURN c
        """
        await session.run(
            query,
            name=collection_name,
            tenant_id=self.agent_id,
            embedder_name=embedder_name,
            embedder_size=embedder_size,
        )

    async def create_collection(self, embedder_name: str, embedder_size: int, collection_name: str):
        await self._ensure_connected()

        async with self._get_session() as session:
            await self._ensure_collection_exists_in_session(
                session, collection_name, embedder_name, embedder_size
            )

    async def create_hybrid_collection(self, collection_name: str, dense_config: str, sparse_config: str):
        log.warning("Hybrid collections not supported")
        pass

    async def delete_collection(self, collection_name: str, timeout: int | None = None):
        """
        Deletes a collection and its documents.
        Entities are deleted only if they become orphans (no remaining MENTIONS
        from documents in other collections), preserving cross-collection knowledge.
        Orphaned SourceFile nodes are pruned as well.
        """
        await self._ensure_connected()

        # Step 1: delete the collection node and all its documents.
        # DETACH DELETE removes all relationships including MENTIONS,
        # so orphan detection in Step 2 is correct.
        delete_docs_query = """
        MATCH (c:Collection {name: $name, tenant_id: $tenant_id})
        OPTIONAL MATCH (c)<-[:BELONGS_TO]-(d:Document)
        DETACH DELETE c, d
        """
        # Step 2: delete entities that are now unreferenced (no more MENTIONS
        # from any document, across all collections for this tenant).
        # DETACH is required because the entity may still carry other
        # relationship types (RELATED_TO, CO_OCCURS_WITH, CONCEPT_RELATION, etc.)
        # that are not captured by the MENTIONS check.
        delete_orphan_entities_query = """
        MATCH (e:Entity {tenant_id: $tenant_id})
        WHERE NOT (e)<-[:MENTIONS]-()
        DETACH DELETE e
        """
        # Step 3: delete orphaned SourceFile nodes with no remaining PART_OF.
        delete_orphan_sourcefiles_query = """
        MATCH (sf:SourceFile {tenant_id: $tenant_id})
        WHERE NOT (sf)<-[:PART_OF]-()
        DETACH DELETE sf
        """
        async with self._get_session() as session:
            await session.run(cast(LiteralString, delete_docs_query), name=collection_name, tenant_id=self.agent_id)
            await session.run(cast(LiteralString, delete_orphan_entities_query), tenant_id=self.agent_id)
            await session.run(cast(LiteralString, delete_orphan_sourcefiles_query), tenant_id=self.agent_id)
        log.info(f"Collection {collection_name} deleted (orphaned entities pruned)")

    async def check_collection_existence(self, collection_name: str) -> bool:
        await self._ensure_connected()

        query = """
        MATCH (c:Collection {name: $name, tenant_id: $tenant_id})
        RETURN count(c) > 0 AS exists
        """
        async with self._get_session() as session:
            result = await session.run(cast(LiteralString, query), name=collection_name, tenant_id=self.agent_id)
            record = await result.single()
            return record["exists"] if record else False

    async def get_collection_names(self) -> List[str]:
        await self._ensure_connected()

        query = """
        MATCH (c:Collection {tenant_id: $tenant_id})
        RETURN c.name AS name
        """
        async with self._get_session() as session:
            result = await session.run(cast(LiteralString, query), tenant_id=self.agent_id)
            records = await result.data()
            return [r["name"] for r in records]

    async def save_dump(self, collection_name: str, folder: str = "dormouse/"):
        log.info(f"Save dump not implemented, use neo4j-admin dump")
        pass

    # ========== POINT METHODS ==========

    async def add_point_to_tenant(
        self,
        collection_name: str,
        content: str,
        vector: Iterable,
        metadata: Dict = None,
        id_point: str | None = None,
        **kwargs,
    ) -> PointStruct | None:
        """
        Adds a document:
        - Creates a Document node with embedding
        - Starts entity extraction in the background
        """
        await self._ensure_connected()

        point_id = id_point or str(uuid.uuid4())

        # ── Guard: empty content ──────────────────────────────────────────────
        if not content or not content.strip():
            log.warning(
                f"[GraphRAG] Skipping point {point_id}: content is empty or whitespace-only. "
                "Check the document splitter / loader upstream."
            )
            return None

        vector_list = list(vector)

        # ── Guard: zero / non-finite embedding vector ─────────────────────────
        if not self._is_valid_vector(vector_list):
            log.warning(
                f"[GraphRAG] Skipping point {point_id}: embedding vector has zero or "
                "non-finite L2-norm. The embedder may have returned a fallback zero "
                "tensor (e.g. empty input, cold-start failure, or unreachable model)."
            )
            return None

        metadata = metadata or {}
        metadata["tenant_id"] = self.agent_id
        # Neo4j does not support Map-type node properties (only primitives /
        # arrays of primitives are allowed).  Serialise to a JSON string so the
        # CREATE never raises ClientError.Statement.TypeError.  All retrieve
        # helpers already call json.loads() when they get back a string, so this
        # is fully backward-compatible.
        metadata_json = json.dumps(metadata)

        # Resolve the current generation's embedding property name (versioned
        # schema. The write must target the SAME property the
        # versioned read paths query (embedding_{gen}), not the bare unversioned
        # `embedding` — otherwise new documents are invisible to the versioned
        # vector index and the first recall returns embedding: null.
        gen = await self._read_generation()
        if gen != self._generation:
            self._rebuild_for_generation(gen)
        embedding_prop = self._names["embedding_prop"]

        create_query = f"""
        MATCH (c:Collection {{name: $collection_name, tenant_id: $tenant_id}})
        CREATE (d:Document {{
            id: $id,
            content: $content,
            {embedding_prop}: $embedding,
            metadata: $metadata,
            tenant_id: $tenant_id,
            created_at: datetime()
        }})
        CREATE (d)-[:BELONGS_TO]->(c)
        RETURN d.id AS id
        """

        async with self._get_session() as session:
            result = await session.run(
                cast(LiteralString, create_query),
                collection_name=collection_name,
                tenant_id=self.agent_id,
                id=point_id,
                content=content,
                embedding=vector_list,
                metadata=metadata_json,
            )
            # Consuming the result surfaces any server-side error immediately
            # (otherwise the async driver silently discards it on session close).
            record = await result.single()
            if record is None:
                log.warning(
                    f"[GraphRAG] Document {point_id} was NOT created: "
                    f"collection '{collection_name}' not found for tenant '{self.agent_id}'. "
                    "Make sure initialize() was called before ingesting documents."
                )
                return None

        # Start entity extraction in the background
        if self._enable_entity_extraction and self._entity_extractor:
            task = asyncio.create_task(self._extract_and_link_entities(point_id, content, metadata))
            self._pending_entity_tasks.append(task)

        # Create SIMILAR_TO relationships in the background (tracked for clean shutdown)
        sim_task = asyncio.create_task(self._create_similarity_relationships(point_id, vector_list, collection_name))
        self._pending_entity_tasks.append(sim_task)

        # Clean up completed tasks
        self._pending_entity_tasks = [t for t in self._pending_entity_tasks if not t.done()]

        return PointStruct(
            id=point_id,
            payload={
                "id": point_id,
                "page_content": content,
                "metadata": metadata,
                "tenant_id": self.agent_id,
            },
            vector=vector_list,
        )

    async def _extract_and_link_entities(
        self,
        document_id: str,
        content: str,
        metadata: Dict,
    ) -> None:
        """
        Extracts entities from the document and links them to the graph.
        Runs in the background.

        Uses three batched UNWIND queries instead of N sequential calls:
        one for entity nodes, one for MENTIONS edges, one for RELATED_TO edges.
        Relations with the same (source, target, type) key are deduplicated
        in Python before being sent, averaging their weights.
        """
        batch_entity_query = """
        UNWIND $entities AS ent
        MERGE (e:Entity {id: ent.id, tenant_id: $tenant_id})
        ON CREATE SET
            e.name       = ent.name,
            e.type       = ent.type,
            e.created_at = datetime(),
            e.metadata   = ent.metadata,
            e.embedding  = ent.embedding
        ON MATCH SET
            e.last_seen  = datetime(),
            e.embedding  = CASE WHEN ent.embedding IS NOT NULL THEN ent.embedding ELSE e.embedding END
        """

        batch_mention_query = """
        MATCH (d:Document {id: $doc_id, tenant_id: $tenant_id})
        WITH d
        UNWIND $mentions AS m
        MATCH (e:Entity {id: m.entity_id, tenant_id: $tenant_id})
        MERGE (d)-[r:MENTIONS]->(e)
        ON CREATE SET r.created_at  = datetime(), r.confidence = m.confidence
        ON MATCH SET  r.last_seen   = datetime(), r.confidence = m.confidence
        """

        batch_relation_query = """
        UNWIND $relations AS rel
        MATCH (s:Entity {id: rel.source_id, tenant_id: $tenant_id})
        MATCH (t:Entity {id: rel.target_id, tenant_id: $tenant_id})
        MERGE (s)-[r:RELATED_TO {type: rel.rel_type}]->(t)
        ON CREATE SET r.weight = rel.weight, r.created_at = datetime()
        ON MATCH SET  r.weight = (r.weight + rel.weight) / 2
        """

        try:
            extracted = await self._entity_extractor.extract(content, document_id, metadata)
            if not extracted.entities:
                return

            entity_type_map = {e.name: e.type for e in extracted.entities}

            # Build batch payload for entities and mentions in one pass
            entities_batch = []
            mentions_batch = []
            for entity in extracted.entities:
                entity_id = self._entity_extractor.get_entity_hash(
                    entity.name, entity.type, self.agent_id
                )
                entities_batch.append({
                    "id":        entity_id,
                    "name":      entity.name.lower().strip(),
                    "type":      entity.type.value,
                    # Serialise to JSON string: Neo4j does not support Map-type
                    # node properties (only primitives / arrays are allowed).
                    "metadata":  json.dumps({"source_document": document_id, "confidence": entity.confidence}),
                    "embedding": None,  # populated below when enable_entity_embeddings=True
                })
                mentions_batch.append({
                    "entity_id":  entity_id,
                    "confidence": entity.confidence,
                })

            # Batch-embed all entity names in one call (non-blocking via thread)
            if self._enable_entity_embeddings and self._embedder is not None:
                try:
                    names = [ent["name"] for ent in entities_batch]
                    embeddings = await asyncio.to_thread(
                        self._embedder.embed_documents, names
                    )
                    for ent, emb in zip(entities_batch, embeddings):
                        if self._is_valid_vector(emb):
                            ent["embedding"] = emb
                        else:
                            log.warning(
                                f"[GraphRAG] Skipping entity embedding for {ent['name']}: "
                                "zero or non-finite vector"
                            )
                except Exception as emb_err:
                    log.warning(f"[GraphRAG] Entity embedding skipped: {emb_err}")

            # Build and deduplicate relation payload (average weights on collision)
            rel_map: Dict[Tuple, Dict] = {}
            for relation in extracted.relations:
                source_type = entity_type_map.get(relation.source_entity, EntityType.UNKNOWN)
                target_type = entity_type_map.get(relation.target_entity, EntityType.UNKNOWN)
                source_id = self._entity_extractor.get_entity_hash(
                    relation.source_entity, source_type, self.agent_id
                )
                target_id = self._entity_extractor.get_entity_hash(
                    relation.target_entity, target_type, self.agent_id
                )
                if not source_id or not target_id or source_id == target_id:
                    continue
                key = (source_id, target_id, relation.relation_type)
                if key in rel_map:
                    rel_map[key]["weight"] = (rel_map[key]["weight"] + relation.weight) / 2
                else:
                    rel_map[key] = {
                        "source_id": source_id,
                        "target_id": target_id,
                        "rel_type":  relation.relation_type,
                        "weight":    relation.weight,
                    }
            relations_batch = list(rel_map.values())

            # Sort all batches by ID so every concurrent transaction acquires
            # Neo4j node locks in the same order — breaks circular wait chains.
            entities_batch.sort(key=lambda e: e["id"])
            mentions_batch.sort(key=lambda m: m["entity_id"])
            relations_batch.sort(key=lambda r: (r["source_id"], r["target_id"]))

            # execute_write wraps all three queries in a single managed write
            # transaction that the Neo4j driver automatically retries on
            # TransientError (including DeadlockDetected).
            # The semaphore caps concurrent write transactions to further
            # reduce lock contention during bulk PDF ingestion.
            async def _write_entities(tx):
                await tx.run(
                    cast(LiteralString, batch_entity_query),
                    entities=entities_batch,
                    tenant_id=self.agent_id,
                )
                await tx.run(
                    cast(LiteralString, batch_mention_query),
                    mentions=mentions_batch,
                    doc_id=document_id,
                    tenant_id=self.agent_id,
                )
                if relations_batch:
                    await tx.run(
                        cast(LiteralString, batch_relation_query),
                        relations=relations_batch,
                        tenant_id=self.agent_id,
                    )

            async with self._neo4j_write_semaphore:
                async with self._get_session() as session:
                    await session.execute_write(_write_entities)

            log.debug(
                f"Linked {len(entities_batch)} entities and {len(relations_batch)} relations "
                f"for document {document_id}"
            )
        except Exception as e:
            log.error(f"Failed to extract entities for {document_id}: {e}")

    async def refresh_technology_entities(self, tenant_id: str) -> None:
        """
        Refreshes the Technology-entity subgraph for a single tenant.

        Re-runs ``extract_technologies_regex`` (pure regex — no spaCy NER) over
        the stored ``Document`` contents of *tenant_id* and reconciles the
        graph: MERGE Technology entities + MENTIONS edges for newly matched
        terms, delete MENTIONS edges to Technology entities that are no longer
        matched, and prune Technology Entity nodes that lost their last MENTION.

        Graph-only: no re-ingest, no re-embed, no full wipe. Idempotent —
        re-running with the same patterns is a no-op. All queries are filtered
        by ``tenant_id`` so other agents' graphs are never touched.
        """
        if not self._entity_extractor:
            return

        # 1. Fetch every stored Document of this tenant (content = page_content).
        async with self._neo4j_write_semaphore:
            async with self._get_session() as session:
                result = await session.run(
                    cast(
                        LiteralString,
                        """
                        MATCH (d:Document {tenant_id: $tenant_id})
                        RETURN d.id AS id, d.content AS content
                        """,
                    ),
                    tenant_id=tenant_id,
                )
                docs = [
                    (record["id"], record["content"] or "")
                    async for record in result
                ]

        if not docs:
            return

        # 2. Re-run the regex technology extraction per document (no spaCy).
        #    Terms are deduplicated per doc (a term may match several patterns).
        doc_terms: Dict[str, List[str]] = {}
        for doc_id, content in docs:
            seen: set = set()
            unique: List[str] = []
            for term in self._entity_extractor.extract_technologies_regex(content):
                name = term.name.lower().strip()
                if name not in seen:
                    seen.add(name)
                    unique.append(name)
            doc_terms[doc_id] = unique

        # 3. Build batch payloads (entity nodes + MENTIONS edges + per-doc
        #    current entity-id sets for stale-edge removal).
        entities_batch: List[Dict] = []
        mentions_batch: List[Dict] = []
        doc_terms_payload: List[Dict] = []
        entities_by_id: Dict[str, Dict] = {}
        mentions_seen: set = set()
        for doc_id, terms in doc_terms.items():
            entity_ids: List[str] = []
            for term in terms:
                entity_id = self._entity_extractor.get_entity_hash(
                    term, EntityType.TECHNOLOGY, tenant_id
                )
                entity_ids.append(entity_id)
                entities_by_id.setdefault(entity_id, {
                    "id":        entity_id,
                    "name":      term,
                    "type":      EntityType.TECHNOLOGY.value,
                    # Serialise to JSON string: Neo4j does not support Map-type
                    # node properties (only primitives / arrays are allowed).
                    "metadata":  json.dumps({"source_document": doc_id, "confidence": 0.85}),
                    "embedding": None,  # no re-embed during a terminology refresh
                })
                mention_key = (doc_id, entity_id)
                if mention_key not in mentions_seen:
                    mentions_seen.add(mention_key)
                    mentions_batch.append({
                        "doc_id":     doc_id,
                        "entity_id":  entity_id,
                        "confidence": 0.85,
                    })
            doc_terms_payload.append({
                "doc_id":     doc_id,
                "entity_ids": sorted(set(entity_ids)),
            })

        entities_batch = sorted(entities_by_id.values(), key=lambda e: e["id"])
        mentions_batch.sort(key=lambda m: (m["doc_id"], m["entity_id"]))

        # 4. Write in a single managed write transaction (auto-retried by the
        #    driver on TransientError), capped by the shared write semaphore.
        batch_entity_query = """
        UNWIND $entities AS ent
        MERGE (e:Entity {id: ent.id, tenant_id: $tenant_id})
        ON CREATE SET
            e.name       = ent.name,
            e.type       = ent.type,
            e.created_at = datetime(),
            e.metadata   = ent.metadata,
            e.embedding  = ent.embedding
        ON MATCH SET
            e.last_seen  = datetime(),
            e.embedding  = CASE WHEN ent.embedding IS NOT NULL THEN ent.embedding ELSE e.embedding END
        """

        batch_mention_query = """
        UNWIND $mentions AS m
        MATCH (d:Document {id: m.doc_id, tenant_id: $tenant_id})
        MATCH (e:Entity {id: m.entity_id, tenant_id: $tenant_id})
        MERGE (d)-[r:MENTIONS]->(e)
        ON CREATE SET r.created_at  = datetime(), r.confidence = m.confidence
        ON MATCH SET  r.last_seen   = datetime(), r.confidence = m.confidence
        """

        # Remove MENTIONS edges from each doc to Technology entities that are no
        # longer matched by the current patterns. Only TECHNOLOGY-typed entities
        # are touched — other entity types (and their MENTIONS) are preserved.
        delete_stale_mentions_query = """
        UNWIND $doc_terms AS d
        MATCH (doc:Document {id: d.doc_id, tenant_id: $tenant_id})
        MATCH (doc)-[r:MENTIONS]->(e:Entity {tenant_id: $tenant_id})
        WHERE e.type = 'TECHNOLOGY' AND NOT e.id IN d.entity_ids
        DELETE r
        """

        # Prune Technology Entity nodes that lost their last MENTION (mirrors the
        # orphan logic in _drop_tenant_data_in_session, restricted to TECHNOLOGY).
        prune_orphan_tech_query = """
        MATCH (e:Entity {tenant_id: $tenant_id, type: 'TECHNOLOGY'})
        WHERE NOT (e)<-[:MENTIONS]-()
        DETACH DELETE e
        """

        async def _write_refresh(tx):
            if entities_batch:
                await tx.run(
                    cast(LiteralString, batch_entity_query),
                    entities=entities_batch,
                    tenant_id=tenant_id,
                )
            if mentions_batch:
                await tx.run(
                    cast(LiteralString, batch_mention_query),
                    mentions=mentions_batch,
                    tenant_id=tenant_id,
                )
            await tx.run(
                cast(LiteralString, delete_stale_mentions_query),
                doc_terms=doc_terms_payload,
                tenant_id=tenant_id,
            )
            await tx.run(
                cast(LiteralString, prune_orphan_tech_query),
                tenant_id=tenant_id,
            )

        async with self._neo4j_write_semaphore:
            async with self._get_session() as session:
                await session.execute_write(_write_refresh)

        log.info(
            f"[GraphRAG] Refreshed Technology entities for tenant_id={tenant_id} "
            f"({len(entities_batch)} entities, {len(mentions_batch)} mentions)"
        )

    @retry_on_generation_change
    async def _create_similarity_relationships(self, point_id: str, vector: List[float], collection_name: str):
        """
        Creates bidirectional SIMILAR_TO relationships between similar documents.

        Both directions (a→b and b→a) are stored so graph traversal never misses
        a link regardless of the direction used by future queries.
        A single UNWIND query replaces the previous one-round-trip-per-document loop.

        Versioned: the vector index and the SIMILAR_TO relation
        name are suffixed with the current generation token; the whole method
        is re-run once by the decorator if the generation flips mid-run.
        """
        # ── Guard: reject zero / non-finite vectors before hitting Neo4j ──────
        if not self._is_valid_vector(vector):
            log.warning(
                f"[GraphRAG] Skipping similarity search for {point_id}: "
                "vector has zero or non-finite L2-norm."
            )
            return

        try:
            # Read phase — auto-commit, read-only, no write locks acquired.
            similar = await self._run_cached(
                "find_similar",
                self._generation,
                {
                    "collection_name": collection_name,
                    "tenant_id": self.agent_id,
                    "vector": vector,
                    "point_id": point_id,
                    "threshold": self._vector_similarity_threshold,
                },
            )

            if not similar:
                log.debug(f"No similar documents found for {point_id}")
                return

            # Sort by document id so every concurrent writer acquires
            # node relationship-group locks in the same order, breaking
            # circular wait chains between writers.
            similar.sort(key=lambda s: s["id"])

            # Write phase — plain auto-commit session.run (no managed
            # transaction): correctness is guaranteed by the
            # retry_on_generation_change decorator (detect-and-rerun-once),
            # not by driver-managed tx. The semaphore caps concurrent writers
            # to reduce contention.
            async with self._neo4j_write_semaphore:
                async with self._get_session() as session:
                    await session.run(
                        cast(LiteralString, self._compile_query("create_similar_rel", self._generation)),
                        similar=similar,
                        point_id=point_id,
                    )

            log.debug(f"Created {len(similar) * 2} similarity relationships for {point_id}")

        except Exception as e:
            log.error(f"Failed to create similarity relationships: {e}")

    async def add_points_to_tenant(
        self, collection_name: str, points: List[PointStruct]
    ) -> UpdateResult:
        await self._ensure_connected()

        operation_id = random.randint(1, 100000)
        for point in points:
            await self.add_point_to_tenant(
                collection_name,
                point.payload.get("page_content", ""),  # type: ignore[arg-type]
                point.vector,
                point.payload.get("metadata", {}),  # type: ignore[arg-type]
                str(point.id),
            )

        if self._enable_derived_graph:
            sources: Dict[str, List[PointStruct]] = {}
            for p in points:
                meta = (p.payload or {}).get("metadata", {}) or {}
                src = meta.get("source")
                if src:
                    sources.setdefault(src, []).append(p)
            for source, src_points in sources.items():
                await self.create_derived_graph_for_source(source, src_points)

        return UpdateResult(status="completed", operation_id=operation_id)

    async def delete_tenant_points(self, collection_name: str, metadata: Dict | None = None) -> UpdateResult:
        await self._ensure_connected()

        operation_id = random.randint(1, 100000)

        conditions: List[str] = []
        params: Dict = {"tenant_id": self.agent_id, "collection_name": collection_name}

        if metadata:
            for k, v in metadata.items():
                safe_param = f"meta_{k.replace('-', '_').replace('.', '_')}"
                # Same CONTAINS strategy as get_all_tenant_points: match the
                # exact JSON fragment that json.dumps produces for this pair.
                conditions.append(f"d.metadata CONTAINS ${safe_param}")
                params[safe_param] = f'"{k}": {json.dumps(v)}'

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        query = f"""
        MATCH (c:Collection {{name: $collection_name, tenant_id: $tenant_id}})<-[:BELONGS_TO]-(d:Document)
        {where_clause}
        DETACH DELETE d
        """

        async with self._get_session() as session:
            await (await session.run(cast(LiteralString, query), **params)).consume()

        async with self._get_session() as session:
            await session.run(
                """
                MATCH (sf:SourceFile {tenant_id: $tenant_id})
                WHERE NOT (sf)<-[:PART_OF]-()
                DETACH DELETE sf
                """,
                tenant_id=self.agent_id,
            )

        return UpdateResult(status="completed", operation_id=operation_id)

    async def delete_tenant_points_by_ids(self, collection_name: str, points_ids: List) -> UpdateResult:
        await self._ensure_connected()

        operation_id = random.randint(1, 100000)

        query = """
        MATCH (c:Collection {name: $collection_name})<-[:BELONGS_TO]-(d:Document)
        WHERE d.id IN $ids AND d.tenant_id = $tenant_id
        DETACH DELETE d
        """
        orphan_query = """
        MATCH (e:Entity {tenant_id: $tenant_id})
        WHERE NOT (e)<-[:MENTIONS]-()
        DETACH DELETE e
        """
        orphan_sf_query = """
        MATCH (sf:SourceFile {tenant_id: $tenant_id})
        WHERE NOT (sf)<-[:PART_OF]-()
        DETACH DELETE sf
        """
        async with self._get_session() as session:
            await session.run(
                cast(LiteralString, query),
                collection_name=collection_name,
                ids=points_ids,
                tenant_id=self.agent_id
            )
            await session.run(cast(LiteralString, orphan_query), tenant_id=self.agent_id)
            await session.run(cast(LiteralString, orphan_sf_query), tenant_id=self.agent_id)
        return UpdateResult(status="completed", operation_id=operation_id)

    @ensure_version
    async def retrieve_tenant_points(self, collection_name: str, points: List) -> List[Record]:
        await self._ensure_connected()

        records = await self._run_cached(
            "retrieve_tenant_points",
            self._generation,
            {"ids": points, "tenant_id": self.agent_id},
        )

        return [
            Record(
                id=r["id"],
                payload={
                    "id": r["id"],
                    "page_content": r["content"],
                    "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
                },
                vector=r["embedding"]
            )
            for r in records
        ]

    # ========== MAIN METHOD: HYBRID RECALL ==========

    @retry_on_generation_change
    async def recall_tenant_memory_from_embedding(
        self,
        collection_name: str,
        embedding: List[float],
        metadata: Dict | None = None,
        k: int | None = 5,
        threshold: float | None = None,
    ) -> List[DocumentRecall]:
        """
        GraphRAG hybrid retrieval — four parallel signals, then a smart merge.

        Phase A (entity-first):
          ② Direct lookup — find documents that explicitly MENTION entities
             extracted from the raw user message (spaCy pipeline).
             Score = matched_entities / total_query_entities.
             Only when `enable_entity_expansion=True` and query entities found.
          ③ Related lookup — walk the RELATED_TO graph up to
             `graph_retrieval_depth` hops from the query entities.
             Score decays with hop distance.
             Only when `enable_entity_expansion=True` and query entities found.
          ④ Entity vector search — query the entity embedding index with the
             query embedding; retrieve documents that mention the closest entities.
             Score = max entity-similarity score across matched entities.
             Only when `enable_entity_embeddings=True` and embedder injected.

        Phase B (always active):
          ⑤ Standard HNSW vector search on document embeddings.

        Merge:
          A③ and A④ results are combined into one "indirect evidence" pool
          (max score when the same document appears in both).
          Documents found by both the entity pool and Phase B receive a boost.
          The final list is sorted by composite score and capped at k.
        """
        async def _empty() -> List[Dict]:
            return []

        async def retrieve() -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
            # A④ and B are always candidates — build their coroutines now
            ev_coro = (
                self._recall_entity_by_vector(collection_name, embedding, k_fetch)
                if self._enable_entity_embeddings and self._embedder is not None
                else _empty()
            )
            vr_coro = self._recall_by_vector(collection_name, embedding, k_fetch, threshold)

            # Entity name expansion (A② + A③) disabled → only A④ + B
            if not self._enable_entity_expansion or not self._entity_extractor:
                ev, vr = await asyncio.gather(ev_coro, vr_coro)
                return [], [], ev, vr

            # ── Phase A ──────────────────────────────────────────────────────
            query_entity_names = await self._extract_query_entities()

            # No recognisable entities in the query → A④ + B only
            if not query_entity_names:
                ev, vr = await asyncio.gather(ev_coro, vr_coro)
                return [], [], ev, vr

            # Run A② + A③ + A④ + B all in parallel — fully independent
            ed, er, ev, vr = await asyncio.gather(
                self._recall_entity_direct(collection_name, query_entity_names, k_fetch),
                self._recall_entity_related(collection_name, query_entity_names, k_fetch, depth, decay),
                ev_coro,
                vr_coro,
            )
            return ed, er, ev, vr

        await self._ensure_connected()

        # ── Guard: reject zero / non-finite / null embeddings from CAT ──────────
        if not embedding or not self._is_valid_vector(embedding):
            log.warning(
                "[GraphRAG] recall_tenant_memory_from_embedding called with "
                "null, zero, or non-finite embedding from CAT — returning empty"
            )
            return []

        threshold = threshold or self._vector_similarity_threshold
        k = k or 5
        depth = self._graph_retrieval_depth
        decay = self._graph_decay_factor
        # Fetch more than k to compensate for post-hoc collection filtering;
        # $param arithmetic is not supported inside Cypher, so pre-compute in Python.
        k_fetch = k * 2

        entity_direct, entity_related, entity_vector, vector_raw = await retrieve()

        # Merge A③ and A④ into one "indirect evidence" pool.
        # Both phases surface documents through associated entities rather than
        # direct name matches; treat them symmetrically and keep the best score
        # when the same document is found by both.
        indirect_map: Dict[str, Dict] = {}
        for r in entity_related:
            indirect_map[r["id"]] = r
        for r in entity_vector:
            if r["id"] not in indirect_map or r["score"] > indirect_map[r["id"]]["score"]:
                indirect_map[r["id"]] = r
        entity_indirect = list(indirect_map.values())

        return self._merge_and_rerank(entity_direct, entity_indirect, vector_raw, k, decay)  # type: ignore[arg-type]

    # ── Phase A helpers ───────────────────────────────────────────────────────

    async def _extract_query_entities(self) -> List[str]:
        """
        Extracts entity names from the current user message using the spaCy
        pipeline already loaded for document ingestion.

        Returns an empty list if the extractor is not ready or no entities
        are found, so callers can safely skip Phase A without failing.
        """
        if not self.user_message or not self._entity_extractor:
            return []

        doc = await self._entity_extractor.extract_doc(self.user_message)

        entities = self._entity_extractor.extract_entities(doc)
        entities += self._entity_extractor.extract_technologies_regex(self.user_message)  # type: ignore[arg-type]
        entities = self._entity_extractor.deduplicate_entities(entities)

        names = [e.name.lower().strip() for e in entities]
        log.debug(f"[GraphRAG] Query entities: {names}")
        return names

    @ensure_version
    async def _recall_entity_direct(
        self,
        collection_name: str,
        entity_names: List[str],
        k: int,
    ) -> List[Dict]:
        """
        Phase A②: finds documents that directly MENTION at least one query entity.

        Score = (number of query entities mentioned in the document) / (total query
        entities).  A document mentioning all query entities scores 1.0; one that
        mentions half scores 0.5. This naturally surfaces the most topically
        complete answers.

        Versioned: the embedding property read is suffixed with
        the current generation token.
        """
        return await self._run_cached(
            "recall_entity_direct",
            self._generation,
            {
                "entity_names": entity_names,
                "tenant_id": self.agent_id,
                "collection_name": collection_name,
                "num_entities": len(entity_names),
                "k": k,
            },
        )

    @ensure_version
    async def _recall_entity_related(
        self,
        collection_name: str,
        entity_names: List[str],
        k: int,
        depth: int,
        decay: float,
    ) -> List[Dict]:
        """
        Phase A③: walks the RELATED_TO graph from the query entities and
        retrieves documents that mention the reached entities.

        Score decays with hop distance: decay^1 for 1-hop, decay^2 for 2-hop, etc.
        Documents mentioning entities that are already directly in the query are
        excluded (they are already returned by _recall_entity_direct).

        depth is injected as a literal — Neo4j does not allow parameters in
        variable-length relationship bounds (*min..max).

        Versioned: the embedding property read is suffixed with
        the current generation token (depth is part of the cache key since it
        is inlined into the compiled Cypher).
        """
        return await self._run_cached(
            f"recall_entity_related:{depth}",
            self._generation,
            {
                "entity_names": entity_names,
                "tenant_id": self.agent_id,
                "collection_name": collection_name,
                "decay": decay,
                "k": k,
            },
        )

    # ── Phase A④ helper ──────────────────────────────────────────────────────

    @ensure_version
    async def _recall_entity_by_vector(
        self,
        collection_name: str,
        embedding: List[float],
        k: int,
    ) -> List[Dict]:
        """
        Phase A④: searches the entity vector index with the query embedding.

        Finds entity nodes whose stored embedding is semantically close to the
        query, then returns documents that MENTION those entities.  The score
        for each document is the maximum entity-similarity score across all
        matched entities.

        This phase is complementary to A② (direct name match): it catches
        entities that are semantically related to the query even when spaCy
        did not extract them explicitly (paraphrases, abbreviations, synonyms).

        Only active when `enable_entity_embeddings=True` and entity embeddings
        have been stored during ingestion (requires the embedder to be injected).

        Versioned: the embedding property read is suffixed with
        the current generation token (the entity index itself is not versioned).
        """
        return await self._run_cached(
            "recall_entity_by_vector",
            self._generation,
            {
                "k": k,
                "vector": embedding,
                "tenant_id": self.agent_id,
                "collection_name": collection_name,
            },
        )

    # ── Phase B helper ────────────────────────────────────────────────────────

    @ensure_version
    async def _recall_by_vector(
        self,
        collection_name: str,
        embedding: List[float],
        k_fetch: int,
        threshold: float | None = None,
    ) -> List[Dict]:
        """
        Phase B: standard HNSW vector search on document embeddings.
        Returns raw Cypher records (dicts) so they can be merged with Phase A results
        before the final conversion to DocumentRecall.

        Versioned: the vector index and the embedding property
        read are suffixed with the current generation token.
        """
        return await self._run_cached(
            "recall_by_vector",
            self._generation,
            {
                "k_fetch": k_fetch,
                "vector": embedding,
                "threshold": threshold or 0.0,
                "collection_name": collection_name,
                "tenant_id": self.agent_id,
            },
        )

    # ── Merge ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _merge_and_rerank(
        entity_direct: List[Dict],
        entity_indirect: List[Dict],
        vector_results: List[Dict],
        k: int,
        decay: float,
        boost: float = 1.3,
    ) -> List[DocumentRecall]:
        """
        Merges Phase A and Phase B results into a single ranked list.

        `entity_indirect` is the pre-merged pool of A③ (graph traversal) and
        A④ (entity vector search) results — both surfaces documents through
        associated entities rather than direct name matches.

        Scoring rules (applied in order of priority):
          1. Doc found by entity_direct AND vector   → max(es, vs) × boost        (jackpot)
          2. Doc found by entity_direct only         → entity_score
          3. Doc found by entity_indirect AND vector → max(es, vs) × (boost × decay)
          4. Doc found by entity_indirect only       → entity_score
          5. Doc found by vector only                → vector_score

        The boost (default 1.3, capped at 1.0) rewards documents that are both
        semantically similar to the query AND topically grounded in the graph.
        """
        def get_final_score(info) -> float:
            es = float(info["entity_score"])
            vs = float(info["vector_score"])
            is_direct = info["is_direct"]
            if es > 0 and vs > 0:
                applied_boost = boost if is_direct else boost * decay
                return min(1.0, max(es, vs) * applied_boost)
            return es or vs

        def load_metadata(metadata: Dict | str) -> Dict:
            if isinstance(metadata, dict):
                return metadata
            try:
                return json.loads(metadata)
            except json.JSONDecodeError:
                return {}

        # registry: doc_id → {data, entity_score, vector_score, is_direct}
        registry: Dict[str, Dict] = {
            r["id"]: {
                "data": r,
                "entity_score": r["score"],
                "vector_score": 0.0,
                "is_direct": True,
            } for r in entity_direct
        }

        registry.update({
            r["id"]: {
                "data": r,
                "entity_score": r["score"],
                "vector_score": 0.0,
                "is_direct": False,
            } for r in entity_indirect if r["id"] not in registry  # entity_direct always takes priority
        })

        for r in vector_results:
            if r["id"] in registry:
                registry[r["id"]]["vector_score"] = r["score"]
            else:
                registry[r["id"]] = {
                    "data": r,
                    "entity_score": 0.0,
                    "vector_score": r["score"],
                    "is_direct": False,
                }

        final: List[Tuple[str, float, Dict]] = [
            (doc_id, get_final_score(info), info) for doc_id, info in registry.items()
        ]
        final.sort(key=lambda x: x[1], reverse=True)

        documents = [
            DocumentRecall(
                document=LangChainDocument(
                    page_content=info.get("data", {}).get("content", ""),
                    metadata=load_metadata(info.get("data", {}).get("metadata", {})),
                    id=doc_id,
                ),
                vector=info.get("data", {}).get("embedding", []),
                id=doc_id,
                score=final_score,
            ) for doc_id, final_score, info in final[:k]
        ]

        log.debug(
            f"[GraphRAG] Merge: {len(entity_direct)} direct + "
            f"{len(entity_indirect)} indirect (graph+vector) + {len(vector_results)} vector "
            f"→ {len(documents)} final"
        )
        return documents

    @ensure_version
    async def recall_tenant_memory(self, collection_name: str) -> List[DocumentRecall]:
        await self._ensure_connected()

        """Retrieves all memory points."""
        records = await self._run_cached(
            "recall_tenant_memory",
            self._generation,
            {"collection_name": collection_name, "tenant_id": self.agent_id},
        )

        documents = []
        for r in records:
            metadata_dict = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
            documents.append(DocumentRecall(
                document=LangChainDocument(
                    page_content=r["content"],
                    metadata=metadata_dict,
                    id=r["id"],
                ),
                vector=r["embedding"],
                id=r["id"],
            ))
        return documents

    # ========== GET ALL METHODS ==========

    @ensure_version
    async def get_all_tenant_points(
        self,
        collection_name: str,
        limit: int | None = None,
        offset: str | None = None,
        metadata: Dict | None = None,
        with_vectors: bool = True,
    ) -> Tuple[List[Record], int | str | None]:
        await self._ensure_connected()

        skip = int(offset) if offset and offset.isdigit() else 0
        query_limit = limit or 1000

        where_clauses = ["d.tenant_id = $tenant_id", "c.name = $collection_name"]
        params: Dict = {
            "tenant_id": self.agent_id,
            "collection_name": collection_name,
            "skip": skip,
            "limit": query_limit,
        }

        if metadata:
            for k, v in metadata.items():
                safe_param = f"meta_{k.replace('-', '_').replace('.', '_')}"
                # metadata is stored as a JSON string (not a Map property).
                # Use CONTAINS with the exact JSON fragment that json.dumps
                # always produces for this key-value pair so the filter runs
                # server-side without requiring APOC or map-index syntax.
                # e.g.  {"source": "file", ...}  CONTAINS  '"source": "file"'
                where_clauses.append(f"d.metadata CONTAINS ${safe_param}")
                params[safe_param] = f'"{k}": {json.dumps(v)}'

        where_str = " AND ".join(where_clauses)

        records = await self._run_cached(
            f"get_all_tenant_points:{where_str}",
            self._generation,
            params,
        )

        points = []
        for r in records:
            metadata_dict = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else (r["metadata"] or {})
            points.append(Record(
                id=r["id"],
                payload={
                    "id": r["id"],
                    "page_content": r["content"],
                    "metadata": metadata_dict,
                },
                vector=r["embedding"] if with_vectors else None
            ))

        next_offset = str(skip + len(points)) if len(points) == query_limit else None
        return points, next_offset

    async def get_all_tenant_points_from_web(
        self, collection_name: str, limit: int | None = None, offset: str | None = None
    ) -> Tuple[List[Record], int | str | None]:
        return await self.get_all_tenant_points(
            collection_name, limit, offset, {"source": "http"}, with_vectors=False
        )

    async def get_all_tenant_points_from_files(
        self, collection_name: str, limit: int | None = None, offset: str | None = None
    ) -> Tuple[List[Record], int | str | None]:
        return await self.get_all_tenant_points(
            collection_name, limit, offset, {"source": "file"}, with_vectors=False
        )

    async def get_tenant_vectors_count(self, collection_name: str) -> int:
        await self._ensure_connected()

        query = """
        MATCH (c:Collection {name: $collection_name, tenant_id: $tenant_id})<-[:BELONGS_TO]-(d:Document)
        RETURN count(d) AS count
        """
        async with self._get_session() as session:
            result = await session.run(
                cast(LiteralString, query), collection_name=collection_name, tenant_id=self.agent_id,
            )
            record = await result.single()
            return record["count"] if record else 0

    # ========== SEARCH METHODS ==========

    @ensure_version
    async def search_in_tenant(
        self,
        collection_name: str,
        query_vector: List[float],
        query_filter: Any = None,
        with_payload: bool = True,
        with_vectors: bool = True,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> List[ScoredPoint]:
        """Direct vector search (without expansion)."""
        await self._ensure_connected()

        records = await self._run_cached(
            "search_in_tenant",
            self._generation,
            {
                "collection_name": collection_name,
                "tenant_id": self.agent_id,
                "vector": query_vector,
                "limit": limit,
                "threshold": score_threshold or 0.0,
            },
        )

        scored_points = []
        for r in records:
            metadata_dict = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
            scored_points.append(ScoredPoint(
                id=r["id"],
                score=r["score"],
                payload={
                    "id": r["id"],
                    "page_content": r["content"],
                    "metadata": metadata_dict,
                },
                vector=r["embedding"] if with_vectors else None,
                version=r.get("version", 0),
            ))
        return scored_points
        
    async def search_prefetched_in_tenant(
        self,
        collection_name: str,
        query: str,
        query_vector: List[float],
        query_filter: Any,
        k: int,
        k_prefetched: int,
        threshold: float,
    ) -> List[ScoredPoint]:
        await self._ensure_connected()

        return await self.search_in_tenant(
            collection_name, query_vector, query_filter, True, True, k, threshold
        )
        
    # ========== UTILITY METHODS ==========
    
    def build_condition(self, key: str, value: Any) -> List:
        return [{"key": key, "match": {"value": value}}]
        
    def filter_from_dict(self, filter_dict: Dict) -> Any:
        if not filter_dict:
            return None
        return {"must": [{"key": k, "match": {"value": v}} for k, v in filter_dict.items()]}

    async def create_derived_graph_for_source(
        self,
        source: str,
        stored_points: List[PointStruct],
        stray_cat=None,
    ) -> None:
        """
        Create derived graph nodes and relations after all documents for a
        source have been ingested.  Handles chunkers with different metadata
        profiles gracefully — only creates structure that the available metadata
        supports.

        Always created (require only chunk_index + source):
          - :SourceFile node + [:PART_OF] from each document
          - [:NEXT] edges between consecutive chunks (ordered by chunk_index)

        Created when chunker provides rich metadata:
          - :Section / :Paragraph labels on documents with chunk_level
          - [:CHILD_OF] from parent_id field (HierarchicalChunker)
          - :FormulaChunk label on documents with has_formula=true
          - [:HAS_SUMMARY] from SourceFile to the CATALOG card
        """
        if not stored_points:
            return

        await self._ensure_connected()
        tenant_id = self.agent_id

        # Separate regular points from the CATALOG card, extract metadata
        catalogue_id: str | None = None
        regular_points: List[Dict[str, Any]] = []

        for p in stored_points:
            meta = (p.payload or {}).get("metadata", {}) or {}
            if meta.get("is_catalogue_card"):
                catalogue_id = p.id
                continue
            ci = meta.get("chunk_index")
            if ci is not None:
                regular_points.append({
                    "id": p.id,
                    "chunk_index": ci,
                    "chunk_level": meta.get("chunk_level"),
                    "parent_id": meta.get("parent_id"),
                    "has_formula": meta.get("has_formula", False),
                })

        if not regular_points:
            return

        regular_points.sort(key=lambda x: x["chunk_index"])
        point_ids = [rp["id"] for rp in regular_points]

        async with self._get_session() as session:
            # 1 — SourceFile node (idempotent)
            await session.run(
                "MERGE (sf:SourceFile {name: $source, tenant_id: $tenant_id})",
                source=source,
                tenant_id=tenant_id,
            )

            # 2 — PART_OF links
            await session.run(
                """
                UNWIND $point_ids AS pid
                MATCH (d:Document {id: pid, tenant_id: $tenant_id})
                MERGE (sf:SourceFile {name: $source, tenant_id: $tenant_id})
                MERGE (d)-[:PART_OF]->(sf)
                """,
                point_ids=point_ids,
                source=source,
                tenant_id=tenant_id,
            )

            # 3 — NEXT edges between consecutive chunks
            if len(regular_points) > 1:
                pairs = [
                    {"curr": regular_points[i]["id"], "next": regular_points[i + 1]["id"]}
                    for i in range(len(regular_points) - 1)
                ]
                await session.run(
                    """
                    UNWIND $pairs AS pair
                    MATCH (a:Document {id: pair.curr, tenant_id: $tenant_id})
                    MATCH (b:Document {id: pair.next, tenant_id: $tenant_id})
                    MERGE (a)-[:NEXT]->(b)
                    """,
                    pairs=pairs,
                    tenant_id=tenant_id,
                )

            # 4 — Structure labels (Section / Paragraph) from chunk_level
            if any(rp.get("chunk_level") for rp in regular_points):
                section_ids = [rp["id"] for rp in regular_points if rp["chunk_level"] == "section"]
                paragraph_ids = [rp["id"] for rp in regular_points if rp["chunk_level"] == "paragraph"]
                if section_ids:
                    await session.run(
                        """
                        UNWIND $ids AS pid
                        MATCH (d:Document {id: pid, tenant_id: $tenant_id})
                        SET d:Section
                        """,
                        ids=section_ids,
                        tenant_id=tenant_id,
                    )
                if paragraph_ids:
                    await session.run(
                        """
                        UNWIND $ids AS pid
                        MATCH (d:Document {id: pid, tenant_id: $tenant_id})
                        SET d:Paragraph
                        """,
                        ids=paragraph_ids,
                        tenant_id=tenant_id,
                    )

            # 5 — FormulaChunk label
            formula_ids = [rp["id"] for rp in regular_points if rp.get("has_formula")]
            if formula_ids:
                await session.run(
                    """
                    UNWIND $ids AS pid
                    MATCH (d:Document {id: pid, tenant_id: $tenant_id})
                    SET d:FormulaChunk
                    """,
                    ids=formula_ids,
                    tenant_id=tenant_id,
                )

            # 6 — CHILD_OF from parent_id
            child_pairs = [
                {"child": rp["id"], "parent": rp["parent_id"]}
                for rp in regular_points if rp.get("parent_id")
            ]
            if child_pairs:
                await session.run(
                    """
                    UNWIND $pairs AS pair
                    MATCH (child:Document {id: pair.child, tenant_id: $tenant_id})
                    MATCH (parent:Document {id: pair.parent, tenant_id: $tenant_id})
                    MERGE (child)-[:CHILD_OF]->(parent)
                    """,
                    pairs=child_pairs,
                    tenant_id=tenant_id,
                )

            # 7 — HAS_SUMMARY from SourceFile to CATALOG card
            if catalogue_id:
                await session.run(
                    """
                    MATCH (card:Document {id: $card_id, tenant_id: $tenant_id})
                    MATCH (sf:SourceFile {name: $source, tenant_id: $tenant_id})
                    MERGE (sf)-[:HAS_SUMMARY]->(card)
                    """,
                    card_id=catalogue_id,
                    source=source,
                    tenant_id=tenant_id,
                )

        log.info(
            f"[GraphRAG] Derived graph for '{source}': {len(regular_points)} docs, "
            f"{sum(1 for r in regular_points if r.get('chunk_level') == 'section')} sections, "
            f"{sum(1 for r in regular_points if r.get('chunk_level') == 'paragraph')} paragraphs, "
            f"{len(formula_ids)} formula, {len(child_pairs)} child_of, "
            f"card={'yes' if catalogue_id else 'no'}"
        )

        # 8 — LLM-based concept relation extraction
        if self._enable_concept_relations and stray_cat:
            try:
                await self._extract_concept_relations(source, stored_points, stray_cat)
            except Exception as e:
                log.error(f"[GraphRAG] Concept relation extraction failed: {e}")

    async def _extract_concept_relations(
        self, source: str, stored_points: List["PointStruct"], stray_cat
    ) -> None:
        tenant_id = self.agent_id

        # Join full section texts (no per-chunk truncation), up to a total limit
        texts: List[str] = []
        total_len = 0
        max_chars = 12000
        for p in stored_points:
            content = (p.payload or {}).get("page_content", "")
            if not content or not content.strip():
                continue
            texts.append(content)
            total_len += len(content)
            if total_len > max_chars:
                break

        combined = "\n\n".join(texts)
        if not combined.strip():
            return

        if len(combined) > max_chars:
            combined = combined[:max_chars] + " [truncated]"

        relations = await self._llm_extract_relations(combined, stray_cat)
        if not relations:
            return

        await self._store_concept_relations(tenant_id, relations)
        log.info(
            f"[GraphRAG] Stored {len(relations)} concept relations for '{source}'"
        )

    async def _llm_extract_relations(
        self, text: str, stray_cat
    ) -> List[Dict[str, str]]:
        # Use the per-agent prompt from settings; fall back to the built-in
        # default when it is not configured (None / empty). The document text
        # is concatenated after the prompt (no {text} placeholder anymore).
        # Prompts saved while the placeholder still existed are handled too.
        prompt_template = self._concept_relations_prompt or CONCEPT_RELATIONS_EXTRACTION_PROMPT
        if "{text}" in prompt_template:
            full_prompt = prompt_template.replace("{text}", text)
        else:
            full_prompt = f"{prompt_template}\n{text}"

        agent_input = AgenticWorkflowTask(user_prompt=full_prompt)
        agent_output = await stray_cat.agentic_workflow.run(
            task=agent_input,
            llm=stray_cat.large_language_model,
        )
        raw = agent_output.output
        return self._parse_concept_relations(raw)

    @staticmethod
    def _parse_concept_relations(raw: str) -> List[Dict[str, str]]:
        import re
        import json

        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[[\s\S]*\]", text)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return []
            else:
                return []

        if not isinstance(data, list):
            return []

        valid = {
            "IS_A", "PART_OF", "EXAMPLE_OF", "PREREQUISITE_FOR",
            "BUILDS_UPON", "CONTRASTS_WITH", "APPLIES_TO",
            "LEADS_TO", "EVIDENCE_FOR",
        }

        result: List[Dict[str, str]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            s = str(item.get("subject", "")).strip()
            o = str(item.get("object", "")).strip()
            r = str(item.get("relation_type", "")).strip().upper()
            if s and o and r in valid:
                result.append({"subject": s, "relation_type": r, "object": o})
        return result

    async def _store_concept_relations(
        self, tenant_id: str, relations: List[Dict[str, str]]
    ) -> None:
        if not relations:
            return

        async with self._get_session() as session:
            for rel in relations:
                try:
                    subject = rel["subject"]
                    object_ = rel["object"]
                    rel_type = rel["relation_type"]

                    # Generate stable entity IDs using the same hash strategy
                    # as the entity extractor, so concept entities have a
                    # non-null `id` property and the frontend visualisation
                    # can map them correctly in D3.
                    subject_id = EntityExtractor.get_entity_hash(
                        subject, EntityType.CONCEPT, tenant_id
                    )
                    object_id = EntityExtractor.get_entity_hash(
                        object_, EntityType.CONCEPT, tenant_id
                    )

                    await session.run(
                        """
                        MERGE (s:Entity {tenant_id: $tenant_id, name: $subject})
                        SET s.id = coalesce(s.id, $subject_id)
                        SET s.type = coalesce(s.type, 'CONCEPT')
                        MERGE (t:Entity {tenant_id: $tenant_id, name: $object})
                        SET t.id = coalesce(t.id, $object_id)
                        SET t.type = coalesce(t.type, 'CONCEPT')
                        MERGE (s)-[r:RELATED_TO {type: $rel_type}]->(t)
                        SET r.weight = coalesce(r.weight, 1.0) + 0.5
                        """,
                        tenant_id=tenant_id,
                        subject=subject,
                        subject_id=subject_id,
                        object=object_,
                        object_id=object_id,
                        rel_type=rel_type,
                    )
                except Exception as e:
                    log.warning(
                        f"[GraphRAG] Failed to store relation "
                        f"'{rel['subject']} -[{rel['relation_type']}]-> {rel['object']}': {e}"
                    )


CONCEPT_RELATIONS_EXTRACTION_PROMPT = """You are a concept extraction system for educational content. Analyse the text below and extract meaningful conceptual relationships.

For each pair of related concepts return a JSON object with:
- "subject": the source concept (short noun phrase, max 3 words)
- "relation_type": one of IS_A, PART_OF, EXAMPLE_OF, PREREQUISITE_FOR, BUILDS_UPON, CONTRASTS_WITH, APPLIES_TO, LEADS_TO, EVIDENCE_FOR
- "object": the target concept (short noun phrase, max 3 words)

IS_A = specialisation / hierarchy  (e.g. Python IS_A programming language)
PART_OF = composition / containment  (e.g. CPU PART_OF computer)
EXAMPLE_OF = concrete instance  (e.g. Django EXAMPLE_OF web framework)
PREREQUISITE_FOR = learning dependency  (e.g. Algebra PREREQUISITE_FOR Calculus)
BUILDS_UPON = conceptual foundation  (e.g. OOP BUILDS_UPON procedural programming)
CONTRASTS_WITH = comparative distinction  (e.g. REST CONTRASTS_WITH GraphQL)
APPLIES_TO = practical application  (e.g. Bayes theorem APPLIES_TO spam filtering)
LEADS_TO = causal chain  (e.g. Global warming LEADS_TO sea level rise)
EVIDENCE_FOR = supporting evidence  (e.g. Study results EVIDENCE_FOR hypothesis)

Only extract relations that are explicitly stated or clearly implied in the text.
Return ONLY a valid JSON array of objects, with no additional text. If nothing matches return [].

Text:"""


class Neo4jGraphRAGConfig(VectorDatabaseSettings):
    # Neo4j connection
    neo4j_uri: str = Field(default="neo4j://localhost:7687", description="Neo4j URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str | None = Field(default=None, description="Neo4j password")
    neo4j_database: str = Field(default="neo4j", description="Neo4j database name")
    neo4j_kwargs: Dict = Field(default={}, description="Neo4j extra arguments, as a dictionary")

    # Vector indexes
    document_vector_index: str = Field(default="document_embeddings", description="Name of the document vector index")
    entity_vector_index: str = Field(default="entity_embeddings", description="Name of the entity vector index")
    vector_similarity_threshold: float = Field(default=0.7, description="Minimum similarity score for vector search")

    # Entity extraction
    enable_entity_extraction: bool = Field(default=True, description="Enable entity extraction with spaCy")
    enable_entity_embeddings: bool = Field(
        default=False,
        description="Enable vector embeddings for entities (increases storage)",
    )
    enable_entity_expansion: bool = Field(default=True, description="Enable entity expansion during retrieval")
    spacy_models: Dict[str, str] = Field(
        default={"en": "en_core_web_lg"},
        description="spaCy model names for different languages (e.g. {'en': 'en_core_web_lg', 'de': 'de_core_news_lg'})",
    )
    extra_technology_patterns: List[str] | None = Field(
        default=[],
        description=(
            "Additional regex patterns for technology entity extraction. "
            "Useful for domain-specific keywords or non-English tech terms not "
            "covered by the built-in list (e.g. [r'\\b(MioFramework|AltroTool)\\b'])."
        ),
    )

    # Graph retrieval
    graph_retrieval_depth: int = Field(default=2, description="Max depth for graph traversal", ge=1, le=5)
    graph_decay_factor: float = Field(default=0.8, description="Score decay factor per hop", ge=0.5, le=1.0)

    enable_derived_graph: bool = Field(
        default=False,
        description="Automatically create derived graph nodes and relations (SourceFile, Section labels, NEXT edges, PART_OF links, HAS_SUMMARY link) after document ingestion",
    )
    enable_concept_relations: bool = Field(
        default=False,
        description="Extract conceptual relations (IS_A, PART_OF, EXAMPLE_OF, PREREQUISITE_FOR, etc.) using the configured LLM after document ingestion",
    )
    concept_relations_prompt: str = Field(
        default=CONCEPT_RELATIONS_EXTRACTION_PROMPT,
        description=(
            "Prompt template sent to the LLM to extract concept relations. "
            "The ingested document text is appended after the prompt. "
            "Leave empty to use the built-in default."
        ),
    )
    enable_knowledge_graph: bool = Field(
        default=False,
        description="Enable the knowledge graph feature globally",
    )
    enable_student_knowledge_graph: bool = Field(
        default=False,
        description="Enable knowledge graph features for students",
    )

    # Performance
    connection_pool_size: int = Field(default=50,description="Neo4j connection pool size")

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Neo4j GraphRAG Advanced",
            "description": "Advanced GraphRAG with entity extraction, knowledge graph, and native vector indexes",
            "link": "https://neo4j.com/docs/vector-indexes/",
        }
    )

    @classmethod
    def parse_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        # Merge stored config with model defaults so new fields (e.g.
        # enable_derived_graph) appear in existing saved configs.
        config = super().parse_config(config)
        return cls(**config).model_dump()

    @classmethod
    def pyclass(cls) -> Type[GraphRAGHandler]:
        return GraphRAGHandler
