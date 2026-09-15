"""Standalone QA harness for the name-only entity identity changes (todo 7, backfill-entity-id-converge): scenarios (a)-(j) against a real Neo4j, in a throwaway tenant. Import-safe: zero top-level side effects — everything runs inside the ``__main__`` guard (the host imports every .py recursively at activation)."""


# ── Everything below runs ONLY when executed as a script (docker exec
#    cheshire_cat_core python /app/test_backfill_duplicates.py). The Cat
#    plugin loader imports this module with zero effect.
if __name__ == "__main__":

    import argparse
    import asyncio
    import importlib
    import json
    import sys
    import types
    import uuid
    from typing import Any
    from neo4j import AsyncGraphDatabase
    from neo4j.exceptions import Neo4jError

    # -------------------------------------------------------------------------
    # Small runtime helpers (defined inside the guard so no top-level code is
    # ever executed at import time).
    # -------------------------------------------------------------------------

    def _find_plugin_module(plugin_id):
        """Import the deployed GraphRAG plugin module, like maintenance_agent does."""
        if plugin_id:
            return importlib.import_module(f"cat.plugins.{plugin_id}.graphrag_handler")
        for pid in ("cat_graphrag", "cat-graphrag", "cat_graphrag_advanced"):
            try:
                return importlib.import_module(f"cat.plugins.{pid}.graphrag_handler")
            except (ImportError, AttributeError, ModuleNotFoundError):
                continue
        raise SystemExit(
            "ERROR: GraphRAG plugin module not importable via cat.plugins.*. "
            "Run inside the Cat container (docker exec cheshire_cat_core ...) and "
            "pass --plugin-id if the deployed folder differs from the defaults."
        )


    class _Db:
        """Tiny wrapper over a shared AsyncGraphDatabase driver + database."""

        def __init__(self, driver, database):
            self._driver = driver
            self._database = database

        async def run(self, cypher, **params):
            async with self._driver.session(database=self._database) as s:
                res = await s.run(cypher, **params)
                return await res.data()

        async def single(self, cypher, **params):
            rows = await self.run(cypher, **params)
            return rows[0] if rows else None

        async def scalar(self, cypher, key, **params):
            row = await self.single(cypher, **params)
            return row[key] if row else None


    def _entity_hash(mod, name, tenant_id):
        """Name-only entity hash (the type is ignored for identity)."""
        return mod.EntityExtractor.get_entity_hash(name, mod.EntityType.CONCEPT, tenant_id)


    def _build_handler(mod, uri, user, password, database):
        """Real GraphRAGHandler (no connect: _driver is injected by the caller)."""
        return mod.GraphRAGHandler(
            neo4j_uri=uri,
            neo4j_user=user,
            neo4j_password=password,
            neo4j_database=database,
        )


    def _scoped_handler(mod, tenant, uri, user, password, database):
        """Build a handler owning its OWN driver (never the harness's _Db driver).

        The Cat core's ``BaseVectorDatabaseHandler.__del__`` schedules
        ``close()`` when a handler is garbage-collected and ``close()`` closes
        ``self._driver`` — so sharing one driver across scenario handlers makes
        a GC'd handler kill the driver for every later scenario ("Driver
        closed" cascade). Every scenario therefore gets a fresh driver, injected
        only into ITS handler, and is responsible for closing it via the
        returned ``close()`` coroutine in a ``finally``.
        """
        hdriver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        handler = _build_handler(mod, uri, user, password, database)
        handler._driver = hdriver
        handler.agent_id = tenant

        async def close():
            try:
                await handler.close()
            except Exception as _exc:  # noqa: BLE001 - never mask the scenario result
                print(f"WARN: close failed: {_exc}", flush=True)

        return handler, close


    async def _ensure_names(mod, handler, tenant):
        """Sync the versioned schema names with the tenant's current generation."""
        gen = await handler._read_generation(tenant)
        if gen != getattr(handler, "_generation", None):
            handler._rebuild_for_generation(gen)
        return gen


    async def _seed_doc(db, tenant, doc_id, content):
        await db.run(
            "MERGE (d:Document {id: $did, tenant_id: $t}) "
            "SET d.content = $content, d.metadata = '{}'",
            did=doc_id, t=tenant, content=content,
        )


    def _stub_ner(mod, handler, name, etype, confidence):
        """Replace the spaCy NER call with a deterministic extraction.

        The REAL ``_extract_and_link_entities`` write path (id-MERGE, name-only
        hash, types/metadata accumulation, untag-on-match) is exercised against
        live Neo4j; only the external spaCy model dependency is substituted.
        """
        async def _stub_extract(text, document_id, metadata=None):
            return types.SimpleNamespace(
                entities=[
                    types.SimpleNamespace(name=name, type=etype, confidence=confidence)
                ],
                relations=[],
            )
        handler._entity_extractor.extract = _stub_extract


    def _constraint_hint(exc):
        if isinstance(exc, Neo4jError) and "ConstraintValidationFailed" in (str(exc) + (exc.code or "")):
            return " [ConstraintValidationFailed]"
        return ""


    # -------------------------------------------------------------------------
    # Scenarios (a)-(j)
    # -------------------------------------------------------------------------

    async def _scenario_a(mod, db, tenant, uri, user, password, database):
        """NER writes 'GraphRAG', then a concept write for 'GraphRAG' -> ONE node."""
        handler, _close = _scoped_handler(mod, tenant, uri, user, password, database)
        try:
            await _seed_doc(db, tenant, "doc-a", "We use GraphRAG in the pipeline.")
            await _ensure_names(mod, handler, tenant)
            _stub_ner(mod, handler, "GraphRAG", mod.EntityType.TECHNOLOGY, 0.85)
            await handler._extract_and_link_entities(
                "doc-a", "We use GraphRAG in the pipeline.", {}
            )
            await handler._store_concept_relations(
                tenant,
                {"concepts": [{"text": "GraphRAG", "type": "DEFINITION"}], "relations": []},
                source="src-a", document_ids=["doc-a"], concept_gen="gen1",
            )
            node = await db.single(
                "MATCH (e:Entity {tenant_id: $t, name: $n}) "
                "RETURN e.id AS id, e.name AS name, e.type AS ptype, e.types AS types, "
                "e.concept_gen AS concept_gen",
                t=tenant, n="graphrag",
            )
            assert node is not None, "no entity 'graphrag' found"
            assert node["id"] == _entity_hash(mod, "graphrag", tenant)
            assert node["name"] == "graphrag"
            assert node["ptype"] == "TECHNOLOGY", "first-wins primary type must be TECHNOLOGY"
            assert "TECHNOLOGY" in node["types"] and "DEFINITION" in node["types"]
            assert node["concept_gen"] is None, "adopted concept ownership must not tag a NER node"
            count = await db.scalar(
                "MATCH (e:Entity {tenant_id: $t, name: $n}) RETURN count(e) AS c",
                "c", t=tenant, n="graphrag",
            )
            assert count == 1, f"expected exactly ONE entity node, found {count}"
        finally:
            await _close()


    async def _scenario_b(mod, db, tenant, uri, user, password, database):
        """Concept writes first, then NER adopts it -> ONE node, extractor untags."""
        handler, _close = _scoped_handler(mod, tenant, uri, user, password, database)
        try:
            await _seed_doc(db, tenant, "doc-b", "We use GraphWay everywhere.")
            await _ensure_names(mod, handler, tenant)
            await handler._store_concept_relations(
                tenant,
                {"concepts": [{"text": "GraphWay", "type": "DEFINITION"}], "relations": []},
                source="src-b", document_ids=["doc-b"], concept_gen="gen1",
            )
            before = await db.single(
                "MATCH (e:Entity {tenant_id: $t, name: $n}) RETURN e.concept_gen AS g",
                t=tenant, n="graphway",
            )
            assert before is not None and before["g"] == "gen1", "concept-created node must be tagged"
            _stub_ner(mod, handler, "GraphWay", mod.EntityType.TECHNOLOGY, 0.85)
            await handler._extract_and_link_entities(
                "doc-b", "We use GraphWay everywhere.", {}
            )
            node = await db.single(
                "MATCH (e:Entity {tenant_id: $t, name: $n}) "
                "RETURN e.name AS name, e.type AS ptype, e.types AS types, "
                "e.concept_gen AS concept_gen",
                t=tenant, n="graphway",
            )
            assert node is not None
            assert node["name"] == "graphway"
            assert node["concept_gen"] is None, "extractor adoption must REMOVE concept_gen"
            assert node["types"] == ["DEFINITION", "TECHNOLOGY"], node["types"]
            count = await db.scalar(
                "MATCH (e:Entity {tenant_id: $t, name: $n}) RETURN count(e) AS c",
                "c", t=tenant, n="graphway",
            )
            assert count == 1, f"expected exactly ONE entity node, found {count}"
        finally:
            await _close()


    async def _scenario_c(mod, db, tenant, uri, user, password, database):
        """NER TECHNOLOGY + concept CONCEPT -> ONE node, types union, metadata merge, confidence max."""
        handler, _close = _scoped_handler(mod, tenant, uri, user, password, database)
        try:
            await _seed_doc(db, tenant, "doc-c", "We use GraphVault daily.")
            await _seed_doc(db, tenant, "doc-c2", "GraphVault is central.")
            await _ensure_names(mod, handler, tenant)
            _stub_ner(mod, handler, "GraphVault", mod.EntityType.TECHNOLOGY, 0.7)
            await handler._extract_and_link_entities("doc-c", "We use GraphVault daily.", {})
            await handler._store_concept_relations(
                tenant,
                {"concepts": [{"text": "GraphVault", "type": "CONCEPT"}], "relations": []},
                source="src-c", document_ids=["doc-c2"], concept_gen="gen1",
            )
            node = await db.single(
                "MATCH (e:Entity {tenant_id: $t, name: $n}) "
                "RETURN e.types AS types, e.metadata AS metadata, e.concept_gen AS concept_gen",
                t=tenant, n="graphvault",
            )
            assert node is not None
            assert "TECHNOLOGY" in node["types"] and "CONCEPT" in node["types"], node["types"]
            meta = json.loads(node["metadata"])
            assert sorted(meta.get("source_documents", [])) == ["doc-c", "doc-c2"], meta
            assert meta.get("confidence") == 0.7, meta
            count = await db.scalar(
                "MATCH (e:Entity {tenant_id: $t, name: $n}) RETURN count(e) AS c",
                "c", t=tenant, n="graphvault",
            )
            assert count == 1, f"expected exactly ONE entity node, found {count}"
        finally:
            await _close()


    async def _scenario_d(mod, db, tenant, uri, user, password, database):
        """Tech-refresh membership + prune guard: PROVENANCE-tracked concept node survives."""
        handler, _close = _scoped_handler(mod, tenant, uri, user, password, database)
        try:
            rag_id = _entity_hash(mod, "rag", tenant)
            old_id = _entity_hash(mod, "oldtool", tenant)
            await db.run(
                "CREATE (d1:Document {id:'doc1', tenant_id: $t, "
                "content: 'We use RAG for the search pipeline.', metadata: '{}'})",
                t=tenant,
            )
            await db.run(
                "CREATE (d2:Document {id:'doc2', tenant_id: $t, "
                "content: 'We use OldTool in production.', metadata: '{}'})",
                t=tenant,
            )
            await db.run(
                "CREATE (a:Entity {tenant_id: $t, id: $aid, name: 'rag', "
                "type: 'CONCEPT', types: ['CONCEPT'], metadata: $ameta}), "
                "(b:Entity {tenant_id: $t, id: $bid, name: 'oldtool', "
                "type: 'TECHNOLOGY', types: ['TECHNOLOGY'], metadata: $bmeta})",
                t=tenant, aid=rag_id, bid=old_id,
                ameta=json.dumps({"source_documents": ["doc1"], "confidence": 0.9}),
                bmeta=json.dumps({"confidence": 0.5}),
            )
            await db.run(
                "MATCH (d1:Document {id:'doc1', tenant_id: $t}), "
                "(a:Entity {id: $aid, tenant_id: $t}) "
                "MERGE (d1)-[:PROVENANCE]->(a) MERGE (d1)-[:MENTIONS]->(a)",
                t=tenant, aid=rag_id,
            )
            await db.run(
                "MATCH (d2:Document {id:'doc2', tenant_id: $t}), "
                "(a:Entity {id: $aid, tenant_id: $t}) MERGE (d2)-[:MENTIONS]->(a)",
                t=tenant, aid=rag_id,
            )
            await db.run(
                "MATCH (d2:Document {id:'doc2', tenant_id: $t}), "
                "(b:Entity {id: $bid, tenant_id: $t}) MERGE (d2)-[:MENTIONS]->(b)",
                t=tenant, bid=old_id,
            )

            await handler.refresh_technology_entities(tenant)

            a = await db.single(
                "MATCH (e:Entity {id: $id, tenant_id: $t}) "
                "RETURN e.type AS ptype, e.types AS types, e.metadata AS metadata",
                id=rag_id, t=tenant,
            )
            assert a is not None, "PROVENANCE-tracked concept-owned node must survive the refresh"
            assert a["ptype"] == "CONCEPT", f"primary type must stay first-wins CONCEPT, got {a['ptype']}"
            assert "TECHNOLOGY" in a["types"], a["types"]
            meta = json.loads(a["metadata"])
            assert meta["source_documents"] == ["doc1"] and meta["confidence"] == 0.9, meta
            assert await db.scalar(
                "MATCH (:Document {id:'doc1', tenant_id: $t})-[:MENTIONS]->"
                "(:Entity {id: $id, tenant_id: $t}) RETURN count(*) AS c",
                "c", t=tenant, id=rag_id,
            ) == 1, "valid MENTIONS edge must survive"
            assert await db.scalar(
                "MATCH (:Document {id:'doc2', tenant_id: $t})-[:MENTIONS]->"
                "(:Entity {id: $id, tenant_id: $t}) RETURN count(*) AS c",
                "c", t=tenant, id=rag_id,
            ) == 0, "stale MENTIONS edge must be deleted"
            assert await db.scalar(
                "MATCH (:Document {id:'doc1', tenant_id: $t})-[:PROVENANCE]->"
                "(:Entity {id: $id, tenant_id: $t}) RETURN count(*) AS c",
                "c", t=tenant, id=rag_id,
            ) == 1, "PROVENANCE edge must survive"
            old_count = await db.scalar(
                "MATCH (e:Entity {id: $id, tenant_id: $t}) RETURN count(e) AS c",
                "c", id=old_id, t=tenant,
            )
            assert old_count == 0, "prov-free Technology node with a stale MENTIONS edge must be pruned"
        finally:
            await _close()


    async def _scenario_e(mod, db, tenant, uri, user, password, database):
        """Guarded fallback: an id-less name-keyed node still gets its id set by the concept path."""
        await db.run(
            "MERGE (e:Entity {tenant_id: $t, name: 'graphcore'}) "
            "SET e.type = 'TECHNOLOGY', e.types = ['TECHNOLOGY']",
            t=tenant,
        )
        handler, _close = _scoped_handler(mod, tenant, uri, user, password, database)
        try:
            await handler._store_concept_relations(
                tenant,
                {"concepts": [{"text": "GraphCore", "type": "DEFINITION"}], "relations": []},
                source="src-e", document_ids=[], concept_gen="gen1",
            )
            node = await db.single(
                "MATCH (e:Entity {tenant_id: $t, name: $n}) "
                "RETURN e.id AS id, e.name AS name, e.types AS types",
                t=tenant, n="graphcore",
            )
            assert node is not None
            assert node["id"] == _entity_hash(mod, "graphcore", tenant), "id-less name-keyed node must get its id"
            assert "TECHNOLOGY" in node["types"] and "DEFINITION" in node["types"], node["types"]
            count = await db.scalar(
                "MATCH (e:Entity {tenant_id: $t, name: $n}) RETURN count(e) AS c",
                "c", t=tenant, n="graphcore",
            )
            assert count == 1, f"expected exactly ONE entity node, found {count}"
        finally:
            await _close()


    async def _scenario_f(mod, db, tenant, uri, user, password, database):
        """Pre-clean effect, scoped to the throwaway tenant (global pre-clean is NOT run)."""
        await db.run("CREATE (e1:Entity {tenant_id: $t, name: 'graphghost'})", t=tenant)
        await db.run("CREATE (e2:Entity {tenant_id: $t, name: 'graphghost2'})", t=tenant)
        kept_id = _entity_hash(mod, "graphkept", tenant)
        await db.run(
            "MERGE (e3:Entity {id: $id, tenant_id: $t}) SET e3.name = 'graphkept'",
            id=kept_id, t=tenant,
        )
        await db.run(
            "MATCH (e:Entity {tenant_id: $t}) WHERE e.id IS NULL DETACH DELETE e",
            t=tenant,
        )
        left = await db.scalar(
            "MATCH (e:Entity {tenant_id: $t}) WHERE e.id IS NULL RETURN count(e) AS c",
            "c", t=tenant,
        )
        assert left == 0, f"expected 0 id-less entities after the scoped pre-clean, found {left}"
        kept = await db.scalar(
            "MATCH (e:Entity {id: $id, tenant_id: $t}) RETURN count(e) AS c",
            "c", id=kept_id, t=tenant,
        )
        assert kept == 1, "id-ful entities must survive the pre-clean"


    async def _scenario_g(mod):
        """Boot path static check: the legacy migration backfill is gone."""
        assert not hasattr(
            mod.GraphRAGHandler, "_backfill_missing_entity_ids"
        ), "_backfill_missing_entity_ids must be absent from GraphRAGHandler"


    async def _scenario_h(mod, db, tenant, uri, user, password, database):
        """Concurrent _ensure_connected -> exactly ONE AsyncGraphDatabase.driver call."""
        original = AsyncGraphDatabase.driver
        calls = {"n": 0}

        def _spy(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        handler = _build_handler(mod, uri, user, password, database)
        # Real __init__ already set _driver=None, _connect_lock and the
        # connection attrs (uri/user/password/pool-size/kwargs/database).
        AsyncGraphDatabase.driver = _spy
        try:
            await asyncio.gather(
                handler._ensure_connected(), handler._ensure_connected()
            )
            assert calls["n"] == 1, (
                f"expected exactly ONE AsyncGraphDatabase.driver call, got {calls['n']}"
            )
            created: Any = handler._driver
            assert created is not None
        finally:
            AsyncGraphDatabase.driver = original
            try:
                await handler.close()
            except Exception as _exc:  # noqa: BLE001
                print(f"WARN: close failed: {_exc}", flush=True)


    def _index_dim(opts):
        """Best-effort extraction of the vector dimensions from SHOW INDEXES options."""
        if not isinstance(opts, dict):
            return None
        cfg = opts.get("indexConfig")
        if isinstance(cfg, dict):
            for key in ("vector.dimensions", "vector_dimensions"):
                if key in cfg:
                    try:
                        return int(cfg[key])
                    except (TypeError, ValueError):
                        pass
        for key in ("vector.dimensions", "vector_dimensions"):
            if key in opts:
                try:
                    return int(opts[key])
                except (TypeError, ValueError):
                    pass
        return None


    async def _index_name_exists(db, name):
        c = await db.scalar(
            "SHOW INDEXES YIELD name WHERE name = $n RETURN count(name) AS c",
            "c", n=name,
        )
        return bool(c)


    async def _restore_v1_indexes(db, v1_meta):
        """Recreate any v1 vector index the reembed GC dropped by global name.

        The generation swap drops v1 indexes by NAME (not tenant-scoped), so a
        throwaway-tenant migration would leave other tenants' v1 indexes missing.
        We restore exactly the ones that pre-existed, at their captured dims.
        """
        plan = [
            ("document_embeddings_v1", "Document", "embedding_v1"),
            ("entity_embeddings_v1", "Entity", "entity_embedding_v1"),
        ]
        for name, label, prop in plan:
            dims = v1_meta.get(name)
            if dims is None:
                continue
            await db.run(
                f"CREATE VECTOR INDEX {name} IF NOT EXISTS "
                f"FOR (n:{label}) ON n.{prop} "
                "OPTIONS { indexConfig: { `vector.dimensions`: " + str(dims) + ", "
                "`vector.similarity_function`: 'cosine', "
                "`vector.hnsw.ef_construction`: 200, `vector.hnsw.m`: 16 } }"
            )


    async def _scenario_i(mod, db, tenant, uri, user, password, database, created_v2_indexes):
        """UPDATE-THEN-SWITCH with a merged multi-type node, via the REAL reembed_tenant."""
        entity_id = _entity_hash(mod, "graphi", tenant)

        # Capture pre-existing v1 index dims so GC's global index drops are restored.
        v1_meta = {}
        for idx in ("document_embeddings_v1", "entity_embeddings_v1"):
            opts = await db.scalar(
                "SHOW INDEXES YIELD name, options WHERE name = $n RETURN options AS opts",
                "opts", n=idx,
            )
            if opts is not None:
                v1_meta[idx] = _index_dim(opts)

        await db.run("MERGE (e:Epoch {tenant_id: $t}) SET e.generation = 'v1'", t=tenant)
        await db.run("MERGE (c:Collection {name: 'declarative', tenant_id: $t})", t=tenant)
        # Capture which v2 vector indexes pre-exist BEFORE the reembed: the
        # swap creates them, so checking in the finally (after the run) can
        # never see them as new and the cleanup would leak them.
        v2_pre = {
            candidate: await _index_name_exists(db, candidate)
            for candidate in ("document_embeddings_v2", "entity_embeddings_v2")
        }
        await db.run(
            "MERGE (d:Document {id: 'doc-i', tenant_id: $t}) "
            "SET d.content = 'GraphI powers the search layer.', d.metadata = '{}'",
            t=tenant,
        )
        await db.run(
            "MATCH (d:Document {id: 'doc-i', tenant_id: $t}), "
            "(c:Collection {name: 'declarative', tenant_id: $t}) "
            "MERGE (d)-[:BELONGS_TO]->(c)",
            t=tenant,
        )
        await db.run(
            "MERGE (e:Entity {id: $id, tenant_id: $t}) "
            "SET e.name = 'graphi', e.type = 'CONCEPT', "
            "e.types = ['CONCEPT', 'TECHNOLOGY'], "
            "e.metadata = $m, e.entity_embedding_v1 = $v1",
            id=entity_id, t=tenant,
            m=json.dumps({"source_documents": ["doc-i"], "confidence": 0.9}),
            v1=[0.01] * 8,
        )

        class _StubEmbedder:
            size = 8

            def embed_documents(self, texts):
                return [[0.1] * 8 for _ in texts]

        handler, _close = _scoped_handler(mod, tenant, uri, user, password, database)
        handler._generation = "v1"
        handler._names = {
            "embedding_prop": "embedding_v1",
            "index": "document_embeddings_v1",
            "relation": "SIMILAR_TO_v1",
            "entity_embedding_prop": "entity_embedding_v1",
            "entity_index": "entity_embeddings_v1",
        }
        handler._enable_entity_embeddings = True
        handler._embedder = _StubEmbedder()

        try:
            await handler.reembed_tenant(tenant, _StubEmbedder())
        finally:
            await _close()
            # Restore the v1 indexes GC just dropped, and hand the v2 indexes we
            # created to the global cleanup (dropped only if not pre-existing).
            await _restore_v1_indexes(db, v1_meta)
            for candidate in ("entity_embeddings_v2", "document_embeddings_v2"):
                if not v2_pre[candidate]:
                    created_v2_indexes.add(candidate)

        node = await db.single(
            "MATCH (e:Entity {id: $id, tenant_id: $t}) "
            "RETURN e.entity_embedding_v2 AS v2, e.entity_embedding_v1 AS v1, "
            "e.type AS ptype, e.types AS types, e.name AS name",
            id=entity_id, t=tenant,
        )
        assert node is not None
        assert node["v2"] is not None and len(node["v2"]) == 8, "entity_embedding_v2 must be present"
        assert node["v1"] is None, "entity_embedding_v1 must be removed by GC"
        assert node["ptype"] == "CONCEPT" and node["types"] == ["CONCEPT", "TECHNOLOGY"], node
        count = await db.scalar(
            "MATCH (e:Entity {id: $id, tenant_id: $t}) RETURN count(e) AS c",
            "c", id=entity_id, t=tenant,
        )
        assert count == 1, f"single merged node must stay intact, found {count}"
        gen = await db.scalar(
            "MATCH (e:Epoch {tenant_id: $t}) RETURN e.generation AS g", "g", t=tenant
        )
        assert gen == "v2", f"generation must be flipped to v2, got {gen}"
        assert await _index_name_exists(db, "entity_embeddings_v2"), (
            "entity_embeddings_v2 vector index must exist"
        )


    async def _scenario_j(mod, db, tenant, uri, user, password, database):
        """After _flip_concept_gen to a new gen, an extractor-adopted merged node stays visible."""
        handler, _close = _scoped_handler(mod, tenant, uri, user, password, database)
        try:
            viz_id = _entity_hash(mod, "graphviz", tenant)
            rel_id = _entity_hash(mod, "retrieval", tenant)
            hidden_id = _entity_hash(mod, "hiddenconcept", tenant)

            await _seed_doc(db, tenant, "doc-j1", "We use GraphViz for retrieval.")
            await _seed_doc(db, tenant, "doc-j2", "The hidden concept is tagged.")
            await db.run("MERGE (c:Collection {name: 'declarative', tenant_id: $t})", t=tenant)
            for doc_id in ("doc-j1", "doc-j2"):
                await db.run(
                    "MATCH (d:Document {id: $did, tenant_id: $t}), "
                    "(c:Collection {name: 'declarative', tenant_id: $t}) "
                    "MERGE (d)-[:BELONGS_TO]->(c)",
                    did=doc_id, t=tenant,
                )

            await _ensure_names(mod, handler, tenant)
            # Build the merged node via the REAL paths: concept creates + NER adopts (untag).
            _stub_ner(mod, handler, "GraphViz", mod.EntityType.TECHNOLOGY, 0.85)
            await handler._store_concept_relations(
                tenant,
                {"concepts": [{"text": "GraphViz", "type": "DEFINITION"}], "relations": []},
                source="src-j", document_ids=["doc-j1"], concept_gen="gen1",
            )
            await handler._extract_and_link_entities(
                "doc-j1", "We use GraphViz for retrieval.", {}
            )

            # Support nodes: an untagged related entity and an old-gen-tagged entity.
            await db.run(
                "MERGE (r:Entity {id: $id, tenant_id: $t}) "
                "SET r.name = 'retrieval', r.type = 'CONCEPT', r.types = ['CONCEPT']",
                id=rel_id, t=tenant,
            )
            await db.run(
                "MERGE (h:Entity {id: $id, tenant_id: $t}) "
                "SET h.name = 'hiddenconcept', h.type = 'CONCEPT', h.types = ['CONCEPT'], "
                "h.concept_gen = 'gen1'",
                id=hidden_id, t=tenant,
            )
            await db.run(
                "MATCH (v:Entity {id: $vid, tenant_id: $t}), (r:Entity {id: $rid, tenant_id: $t}) "
                "MERGE (r)-[:RELATED_TO {type: 'RELATES'}]->(v)",
                vid=viz_id, rid=rel_id, t=tenant,
            )
            await db.run(
                "MATCH (h:Entity {id: $hid, tenant_id: $t}), (v:Entity {id: $vid, tenant_id: $t}) "
                "MERGE (h)-[:RELATED_TO {type: 'RELATES', concept_gen: 'gen1'}]->(v)",
                hid=hidden_id, vid=viz_id, t=tenant,
            )
            await db.run(
                "MATCH (d:Document {id: 'doc-j1', tenant_id: $t}), "
                "(r:Entity {id: $rid, tenant_id: $t}) MERGE (d)-[:MENTIONS]->(r)",
                rid=rel_id, t=tenant,
            )
            await db.run(
                "MATCH (d:Document {id: 'doc-j2', tenant_id: $t}), "
                "(h:Entity {id: $hid, tenant_id: $t}) MERGE (d)-[:MENTIONS]->(h)",
                hid=hidden_id, t=tenant,
            )

            merged = await db.single(
                "MATCH (e:Entity {id: $id, tenant_id: $t}) "
                "RETURN e.concept_gen AS concept_gen, e.types AS types",
                id=viz_id, t=tenant,
            )
            assert merged["concept_gen"] is None, "extractor adoption must have untagged the merged node"
            assert "TECHNOLOGY" in merged["types"], merged["types"]

            async def _recall_ids():
                rows = await handler._recall_entity_related(
                    "declarative", ["graphviz"], k=5, depth=2, decay=0.8
                )
                return {r["id"] for r in rows}

            flipped1 = await handler._flip_concept_gen(tenant, None, "gen1")
            assert flipped1 is True, "first flip (uninitialised marker) must succeed"
            pre_ids = await _recall_ids()
            assert "doc-j1" in pre_ids, f"merged untagged node lost pre-flip: {sorted(pre_ids)}"
            assert "doc-j2" in pre_ids, f"old-gen tagged node hidden pre-flip: {sorted(pre_ids)}"

            flipped2 = await handler._flip_concept_gen(tenant, "gen1", "gen2")
            assert flipped2 is True, "second flip (marker gen1 -> gen2) must succeed"
            post_ids = await _recall_ids()
            assert "doc-j1" in post_ids, (
                f"extractor-adopted merged node hidden by the concept-gen flip: {sorted(post_ids)}"
            )
            assert "doc-j2" not in post_ids, (
                f"old-gen tagged node still visible after the flip: {sorted(post_ids)}"
            )
        finally:
            await _close()


    # -------------------------------------------------------------------------
    # Runner + CLI
    # -------------------------------------------------------------------------

    async def _main(argv):
        parser = argparse.ArgumentParser(
            prog="test_backfill_duplicates",
            description=(
                "Standalone QA harness for the name-only entity identity changes "
                "(scenarios a-j). Runs inside the Cat container against a real "
                "Neo4j, in a throwaway tenant that is fully cleaned up at the end."
            ),
        )
        parser.add_argument("--uri", default="neo4j://localhost:7687", help="Neo4j URI")
        parser.add_argument("--user", default="neo4j", help="Neo4j username")
        parser.add_argument("--password", required=True, help="Neo4j password")
        parser.add_argument("--database", default="neo4j", help="Neo4j database name")
        parser.add_argument("--tenant", default=None, help="Throwaway tenant id (default: fresh uuid)")
        parser.add_argument("--plugin-id", default=None, help="Deployed plugin folder id (default: autodetect)")
        args = parser.parse_args(argv)

        tenant = args.tenant or uuid.uuid4().hex
        mod = _find_plugin_module(args.plugin_id)
        # Standalone mode: force the in-process lock path in reembed_tenant
        # (the module-level global read by reembed_tenant; setattr keeps the
        # static analyzer quiet about a runtime-injected module attribute).
        setattr(mod, "distributed_lock", None)

        driver = AsyncGraphDatabase.driver(args.uri, auth=(args.user, args.password))
        db = _Db(driver, args.database)

        existing = await db.scalar(
            "MATCH (c:Collection {tenant_id: $t}) RETURN count(c) AS c", "c", t=tenant
        )
        if existing:
            print(
                f"REFUSE: tenant {tenant} already exists as a Collection.tenant_id; "
                "refusing to run (a throwaway tenant is mandatory)."
            )
            await driver.close()
            return 2

        results = {}
        created_v2_indexes = set()

        async def _run(name, fn):
            try:
                await fn()
                print(f"PASS ({name})", flush=True)
                results[name] = True
            except Exception as exc:  # noqa: BLE001 - harness reports any failure
                print(
                    f"FAIL ({name}) :: {exc}{_constraint_hint(exc)}",
                    flush=True,
                )
                results[name] = False

        scenarios = [
            ("a", lambda: _scenario_a(mod, db, tenant, args.uri, args.user, args.password, args.database)),
            ("b", lambda: _scenario_b(mod, db, tenant, args.uri, args.user, args.password, args.database)),
            ("c", lambda: _scenario_c(mod, db, tenant, args.uri, args.user, args.password, args.database)),
            ("d", lambda: _scenario_d(mod, db, tenant, args.uri, args.user, args.password, args.database)),
            ("e", lambda: _scenario_e(mod, db, tenant, args.uri, args.user, args.password, args.database)),
            ("f", lambda: _scenario_f(mod, db, tenant, args.uri, args.user, args.password, args.database)),
            ("g", lambda: _scenario_g(mod)),
            ("h", lambda: _scenario_h(mod, db, tenant, args.uri, args.user, args.password, args.database)),
            ("i", lambda: _scenario_i(mod, db, tenant, args.uri, args.user, args.password, args.database, created_v2_indexes)),
            ("j", lambda: _scenario_j(mod, db, tenant, args.uri, args.user, args.password, args.database)),
        ]

        try:
            for name, fn in scenarios:
                await _run(name, fn)
        finally:
            # Scoped cleanup of the throwaway tenant + the v2 vector indexes this
            # run created (never flushdb/flushall, never another tenant's data).
            try:
                await db.run(
                    "MATCH (n) WHERE n.tenant_id = $t DETACH DELETE n", t=tenant
                )
            except Exception as exc:  # noqa: BLE001
                print(f"WARN: tenant cleanup failed: {exc}", flush=True)
            for idx in sorted(created_v2_indexes):
                try:
                    await db.run(f"DROP INDEX {idx} IF EXISTS")
                except Exception as exc:  # noqa: BLE001
                    print(f"WARN: index cleanup {idx} failed: {exc}", flush=True)
            await driver.close()

        failed = [k for k, ok in results.items() if not ok]
        print(f"tenant={tenant}", flush=True)
        if failed:
            print(f"FAILED scenarios: {', '.join(failed)}")
            return 1
        print("All scenarios passed.")
        return 0


    def main():
        return asyncio.run(_main(sys.argv[1:]))

    sys.exit(main())