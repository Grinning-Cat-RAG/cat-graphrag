"""Epoch generation helpers for the versioned schema.

Import-safe: no side effects at import time (the Cat plugin loader imports
every ``.py`` in the plugin folder).

A single ``(:Epoch {tenant_id, generation})`` node per tenant holds the
current generation token (e.g. ``"v1"``, ``"v2"``). All versioned schema
names derive from it:

- embedding property: ``embedding_{gen}``
- vector index: ``document_embeddings_{gen}``
- relation: ``SIMILAR_TO_{gen}``

Queries are compiled once per generation into version-suffixed Cypher
strings (the versioned names are INLINED, never passed as params) and cached
in ``{query_key: {gen: compiled_cypher}}``; they are re-written only when the
generation changes. No transactions are used anywhere in these paths.
"""

from typing import Any, Dict, Optional, cast, LiteralString


class EpochMixin:
    """Provides the generation-token helpers used by the versioned schema."""

    async def _read_generation(self, tenant_id: Optional[str] = None) -> str:
        """Probe the tenant's generation token.

        Single always-1-row read: ``MATCH (e:Epoch {tenant_id:$t})
        RETURN e.generation AS gen``. If no Epoch node exists yet, create it
        with ``generation='v1'`` (MERGE) and return ``"v1"``. Never returns
        an empty string.
        """
        tenant_id = tenant_id or self.agent_id
        await self._ensure_connected()
        async with self._get_session() as session:
            result = await session.run(
                cast(
                    LiteralString,
                    "MATCH (e:Epoch {tenant_id: $tenant_id}) RETURN e.generation AS gen",
                ),
                tenant_id=tenant_id,
            )
            record = await result.single()
            if record is not None and record["gen"]:
                return str(record["gen"])

            # No Epoch node yet -> create it with the initial generation.
            await session.run(
                cast(
                    LiteralString,
                    "MERGE (e:Epoch {tenant_id: $tenant_id}) "
                    "ON CREATE SET e.generation = 'v1'",
                ),
                tenant_id=tenant_id,
            )
            return "v1"

    async def _set_generation(self, tenant_id: str, gen: str) -> None:
        """Atomically set the tenant's generation token (single MERGE SET)."""
        await self._ensure_connected()
        async with self._get_session() as session:
            await session.run(
                cast(
                    LiteralString,
                    "MERGE (e:Epoch {tenant_id: $tenant_id}) SET e.generation = $gen",
                ),
                tenant_id=tenant_id,
                gen=gen,
            )

    def _versioned_names(self, gen: str) -> Dict[str, str]:
        """Resolve the version-suffixed schema names for a generation."""
        return {
            "embedding_prop": f"embedding_{gen}",
            "index": f"{self._document_vector_index}_{gen}",
            "relation": f"SIMILAR_TO_{gen}",
            # Entity embeddings follow the SAME generation scheme as document
            # embeddings, so entity vector search always targets the property
            # and index of the CURRENT generation (no legacy `e.embedding`).
            "entity_embedding_prop": f"entity_embedding_{gen}",
            "entity_index": f"{self._entity_vector_index}_{gen}",
        }

    def _rebuild_for_generation(self, gen: str) -> None:
        """Re-suffix and re-cache the versioned names for a generation."""
        self._generation = gen
        self._names = self._versioned_names(gen)

    def _compile_query(self, key: str, gen: str) -> str:
        """Return the version-suffixed Cypher for (key, gen), compiled once.

        The compiled string has the versioned names INLINED (not passed as
        params), so the cache is genuinely per-generation: the same key is
        re-written only when ``gen`` changes.
        """
        cache = getattr(self, "_query_cache", None)
        if cache is None:
            cache = self._query_cache = {}
        per_gen = cache.setdefault(key, {})
        if gen not in per_gen:
            per_gen[gen] = self._build_query(key, gen)
        return per_gen[gen]

    async def _run_cached(self, key: str, gen: str, params: Dict[str, Any]):
        """Run the compiled version-suffixed query for (key, gen).

        Plain ``session.run`` — no transactions in the decorated paths.
        Returns the full result rows (``result.data()``).
        """
        query = self._compile_query(key, gen)
        await self._ensure_connected()
        async with self._get_session() as session:
            result = await session.run(cast(LiteralString, query), **params)
            return await result.data()

    def _build_query(self, key: str, gen: str) -> str:
        """Build the version-suffixed Cypher for a query key and generation.

        The versioned names are inlined as literals (``d.embedding_v1``,
        ``document_embeddings_v1``, ``[:SIMILAR_TO_v1]``) so the compiled
        string is stable per (key, gen) and safe to cache.
        """
        names = self._versioned_names(gen)
        emb = names["embedding_prop"]
        index = names["index"]
        rel = names["relation"]

        if key == "find_similar":
            return f"""
            MATCH (c:Collection {{name: $collection_name, tenant_id: $tenant_id}})
            CALL db.index.vector.queryNodes('{index}', 20, $vector)
            YIELD node, score
            WHERE EXISTS {{ MATCH (node)-[:BELONGS_TO]->(c) }}
              AND node.id <> $point_id
              AND score >= $threshold
            RETURN node.id AS id, score
            ORDER BY score DESC
            """

        if key == "create_similar_rel":
            return f"""
            UNWIND $similar AS sim
            MATCH (a:Document {{id: $point_id}})
            MATCH (b:Document {{id: sim.id}})
            MERGE (a)-[r1:{rel}]->(b)
            SET r1.score = sim.score, r1.updated_at = datetime()
            MERGE (b)-[r2:{rel}]->(a)
            SET r2.score = sim.score, r2.updated_at = datetime()
            """

        if key == "recall_by_vector":
            return f"""
            CALL db.index.vector.queryNodes('{index}', $k_fetch, $vector)
            YIELD node AS doc, score AS doc_score
            WHERE doc_score >= $threshold
              AND EXISTS {{
                  MATCH (doc)-[:BELONGS_TO]->(:Collection {{name: $collection_name, tenant_id: $tenant_id}})
              }}
            RETURN doc.id        AS id,
                   doc.content   AS content,
                   doc.metadata  AS metadata,
                   doc.{emb} AS embedding,
                   doc_score     AS score
            ORDER BY score DESC
            LIMIT $k_fetch
            """

        if key == "recall_entity_direct":
            return f"""
            UNWIND $entity_names AS q_name
            MATCH (q_e:Entity {{tenant_id: $tenant_id}})
            WHERE q_e.name = q_name
            WITH DISTINCT q_e

            MATCH (d:Document {{tenant_id: $tenant_id}})-[:MENTIONS]->(q_e)
            WHERE EXISTS {{
                MATCH (d)-[:BELONGS_TO]->(:Collection {{name: $collection_name, tenant_id: $tenant_id}})
            }}

            WITH d, count(DISTINCT q_e) AS matched_count
            RETURN d.id        AS id,
                   d.content   AS content,
                   d.metadata  AS metadata,
                   d.{emb} AS embedding,
                   toFloat(matched_count) / $num_entities AS score
            ORDER BY score DESC
            LIMIT $k
            """

        if key.startswith("recall_entity_related:"):
            depth = key.split(":", 1)[1]
            return f"""
            UNWIND $entity_names AS q_name
            MATCH (q_e:Entity {{tenant_id: $tenant_id}})
            WHERE q_e.name = q_name

            MATCH path = (q_e)-[:RELATED_TO*1..{depth}]-(r_e:Entity {{tenant_id: $tenant_id}})
            WHERE NOT r_e.name IN $entity_names

            MATCH (d:Document {{tenant_id: $tenant_id}})-[:MENTIONS]->(r_e)
            WHERE EXISTS {{
                MATCH (d)-[:BELONGS_TO]->(:Collection {{name: $collection_name, tenant_id: $tenant_id}})
            }}

            WITH d, min(length(path)) AS min_hops
            RETURN d.id        AS id,
                   d.content   AS content,
                   d.metadata  AS metadata,
                   d.{emb} AS embedding,
                   $decay ^ min_hops AS score
            ORDER BY score DESC
            LIMIT $k
            """

        if key == "recall_entity_by_vector":
            return f"""
            CALL db.index.vector.queryNodes('{names["entity_index"]}', $k, $vector)
            YIELD node AS ent, score AS ent_score
            WHERE ent.tenant_id = $tenant_id
            MATCH (d:Document {{tenant_id: $tenant_id}})-[:MENTIONS]->(ent)
            WHERE EXISTS {{
                MATCH (d)-[:BELONGS_TO]->(:Collection {{name: $collection_name, tenant_id: $tenant_id}})
            }}
            WITH d, max(ent_score) AS score
            RETURN d.id        AS id,
                   d.content   AS content,
                   d.metadata  AS metadata,
                   d.{emb} AS embedding,
                   score
            ORDER BY score DESC
            LIMIT $k
            """

        if key == "search_in_tenant":
            return f"""
            MATCH (c:Collection {{name: $collection_name, tenant_id: $tenant_id}})
            CALL db.index.vector.queryNodes('{index}', $limit, $vector)
            YIELD node, score
            WHERE EXISTS {{ MATCH (node)-[:BELONGS_TO]->(c) }}
              AND score >= $threshold
            RETURN node.id AS id, node.content AS content, node.metadata AS metadata,
                   node.{emb} AS embedding, score
            ORDER BY score DESC
            LIMIT $limit
            """

        if key == "retrieve_tenant_points":
            return f"""
            MATCH (d:Document)
            WHERE d.id IN $ids AND d.tenant_id = $tenant_id
            RETURN d.id AS id, d.content AS content, d.metadata AS metadata, d.{emb} AS embedding
            """

        if key == "recall_tenant_memory":
            return f"""
            MATCH (c:Collection {{name: $collection_name, tenant_id: $tenant_id}})<-[:BELONGS_TO]-(d:Document)
            RETURN d.id AS id, d.content AS content, d.metadata AS metadata, d.{emb} AS embedding
            """

        if key.startswith("get_all_tenant_points:"):
            where_str = key.split(":", 1)[1]
            return f"""
            MATCH (c:Collection)<-[:BELONGS_TO]-(d:Document)
            WHERE {where_str}
            RETURN d.id AS id, d.content AS content, d.metadata AS metadata, d.{emb} AS embedding
            SKIP $skip
            LIMIT $limit
            """

        raise KeyError(f"Unknown versioned query key: {key}")