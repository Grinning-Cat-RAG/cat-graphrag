"""The source-scoped Cypher of the graph phases, executed on a LIVE Neo4j.

- ``purge_source_graph_artifacts(source, collection, chat_id)`` removes the entity / similarity edges of the documents
  of THAT source only: a file with the same name in the knowledge base keeps its graph when the copy in a chat is
  purged (and vice versa); the documents and the SourceFile stay;
- ``get_source_points`` reads every point of a source exactly once across pages (the paged listing is ordered);
- ``wipe_tenant_graph`` (the maintenance ``--graph``) removes every graph edge and entity of the tenant, keeps its
  documents and structure, and never touches another tenant;
- the metadata read back are the ones the caller stored (no ``tenant_id``, also on Documents written before);
- ``has_stale_visible_concepts``: visible concepts of another generation, on nodes or on edges only, per tenant;
- a concept found again by a recompute takes the new generation and survives the flip and the GC;
- the RELATED_TO provenance is scoped by knowledge base / chat in every deletion path (a source, a chat destroyed, a
  whole collection), in the concept store and in the boot backfill.
- the read-only graph checks of the maintenance ``--verify`` (``_graph_checks``) on a healthy and a broken graph.

Standalone script (plugin rules: the body runs only as ``__main__``). Needs the core checkout, a Redis Stack on
localhost:6379 (database 14, for the tenant write lock) and a Neo4j 5 reachable at ``--uri`` (default
``bolt://localhost:7687``) with ``--password`` (or ``NEO4J_PASSWORD``); without a password it reports SKIP. The test
tenant is removed at the end.

    cd ../grinning-cat-core
    .venv/bin/python /path/to/cat-graphrag/tests/test_graph_purge_live.py --uri bolt://localhost:7687 --password ...
"""

if __name__ == "__main__":
    import argparse
    import asyncio
    import json
    import os
    import sys
    import traceback
    import uuid

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import graph_fakes as gf

    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD"))
    args = parser.parse_args()
    if not args.password:
        print("SKIP test_graph_purge_live: no Neo4j password (--password or NEO4J_PASSWORD)")
        sys.exit(0)

    main, handler_mod, phases_mod = gf.bootstrap("_graphrag_purge_live")
    from cat.services.memory.models import PointStruct

    cli = gf.load_cli(handler_mod, phases_mod)
    TENANT = f"purge_live_{uuid.uuid4().hex[:8]}"
    OTHER = f"purge_live_{uuid.uuid4().hex[:8]}"

    def make_handler(tenant=TENANT):
        handler = handler_mod.GraphRAGHandler(
            neo4j_uri=args.uri,
            neo4j_user=args.user,
            neo4j_password=args.password,
            enable_entity_extraction=False,
            enable_concept_relations=False,
        )
        handler.agent_id = tenant
        return handler

    async def add_points(handler, collection, source, n, chat_id=None):
        meta = {"source": source} | ({"chat_id": chat_id} if chat_id else {})
        points = [
            PointStruct(
                id=uuid.uuid4().hex,
                payload={"page_content": f"{source} chunk {i}", "metadata": meta | {"chunk_index": i}},
                vector=[1.0, float(i + 1)],
            )
            for i in range(n)
        ]
        await handler.add_points_to_tenant(collection, points)
        # the similarity edges are written by background tasks: settle them before measuring
        await asyncio.gather(*handler._pending_entity_tasks, return_exceptions=True)
        return [p.id for p in points]

    async def run(handler, query, **params):
        async with handler._get_session() as session:
            result = await session.run(query, tenant_id=handler.agent_id, **params)
            return [record async for record in result]

    async def link(handler, doc_ids, entity, source, gen):
        """The graph the phases write for a source: MENTIONS + PROVENANCE, a similarity edge, a RELATED_TO whose
        provenance is ``source`` (the source's provenance key).
        Entity ids are unique per run: the id constraint spans every tenant of the database."""
        await run(
            handler,
            """
            MERGE (e:Entity {id: $entity, tenant_id: $tenant_id}) SET e.name = $entity, e.tracked_by_provenance = true
            MERGE (o:Entity {id: $entity + '_other', tenant_id: $tenant_id}) SET o.name = $entity + '_other'
            MERGE (e)-[r:RELATED_TO {type: 'NEXT_TO'}]->(o) SET r.source_files = [$source]
            WITH e
            UNWIND $doc_ids AS did
            MATCH (d:Document {id: did, tenant_id: $tenant_id})
            MERGE (d)-[:MENTIONS]->(e)
            MERGE (d)-[:PROVENANCE]->(e)
            """,
            entity=f"{handler.agent_id}:{entity}", doc_ids=doc_ids, source=source,
        )
        await run(
            handler,
            f"""
            MATCH (a:Document {{id: $a, tenant_id: $tenant_id}}), (b:Document {{id: $b, tenant_id: $tenant_id}})
            MERGE (a)-[:SIMILAR_TO_{gen}]->(b)
            """,
            a=doc_ids[0], b=doc_ids[1],
        )

    async def edges_of(handler, doc_ids, exclude=()):
        """The graph edges of the documents, leaving out the ones towards ``exclude`` (a similarity edge between
        two sources belongs to both: purging one removes it)."""
        rows = await run(
            handler,
            """
            MATCH (d:Document {tenant_id: $tenant_id})-[r]-(x)
            WHERE d.id IN $doc_ids AND NOT type(r) IN ['BELONGS_TO', 'PART_OF', 'NEXT']
              AND NOT coalesce(x.id, '') IN $exclude
            RETURN count(r) AS n
            """,
            doc_ids=doc_ids, exclude=list(exclude),
        )
        return rows[0]["n"]

    async def test_purge_is_scoped_to_collection_and_chat(handler):
        gen = await handler._read_generation(TENANT)
        kb = await add_points(handler, gf.DECLARATIVE, "x.pdf", 2)
        chat = await add_points(handler, gf.EPISODIC, "x.pdf", 2, chat_id="c1")
        other_chat = await add_points(handler, gf.EPISODIC, "x.pdf", 2, chat_id="c2")
        await link(handler, kb, "KbEntity", "x.pdf", gen)
        await link(handler, chat, "ChatEntity", handler_mod.provenance_key("x.pdf", "c1"), gen)
        await link(handler, other_chat, "OtherEntity", handler_mod.provenance_key("x.pdf", "c2"), gen)
        before = {
            name: await edges_of(handler, ids, exclude=chat + kb)
            for name, ids in (("kb", kb), ("c1", chat), ("c2", other_chat))
        }
        assert all(before.values()), before

        count = await handler.purge_source_graph_artifacts("x.pdf", gf.EPISODIC, "c1")
        assert count == 2, count
        assert await edges_of(handler, chat) == 0, "the chat copy kept its graph"
        assert await edges_of(handler, kb, exclude=chat + kb) == before["kb"], "purging a chat touched the KB"
        assert await edges_of(handler, other_chat, exclude=chat + kb) == before["c2"], "purging touched another chat"
        orphan = await run(
            handler, "MATCH (e:Entity {id: $eid, tenant_id: $tenant_id}) RETURN e", eid=f"{TENANT}:ChatEntity",
        )
        assert not orphan, "the entity only the purged documents pointed at is still there"
        docs = await run(handler, "MATCH (d:Document {tenant_id: $tenant_id}) WHERE d.id IN $ids RETURN d.id", ids=chat)
        assert len(docs) == 2, "the purge deleted documents"
        files = await run(handler, "MATCH (s:SourceFile {name: 'x.pdf', tenant_id: $tenant_id}) RETURN s")
        assert files, "the purge deleted the SourceFile"

        assert await handler.purge_source_graph_artifacts("x.pdf", gf.DECLARATIVE) == 2
        assert await edges_of(handler, kb) == 0
        assert await edges_of(handler, other_chat, exclude=chat + kb) == before["c2"], "purging the KB touched a chat"
        assert await handler.purge_source_graph_artifacts("x.pdf", gf.DECLARATIVE) == 2  # idempotent

    async def test_points_read_once_across_pages(handler):
        # pages of 100 instead of 1000: the same paging over 3 pages, without inserting thousands of points (an insert
        # is slow on recent Neo4j versions)
        saved, handler_mod._PAGE_SIZE = handler_mod._PAGE_SIZE, 100
        try:
            ids = await add_points(handler, gf.DECLARATIVE, "big.pdf", 235)
            points = await handler.get_source_points(gf.DECLARATIVE, "big.pdf")
            read = [p.id for p in points]
            assert len(read) == len(set(read)) == 235 and set(read) == set(ids), (len(read), len(set(read)))
            # the core's contract, which its embedding phase and the vector-memory transfer rely on: no limit = all
            every, next_offset = await handler.get_all_tenant_points(
                gf.DECLARATIVE, metadata={"source": "big.pdf"}, with_vectors=False
            )
            assert len(every) == 235 and next_offset is None, (len(every), next_offset)
            page, next_offset = await handler.get_all_tenant_points(gf.DECLARATIVE, limit=100, metadata={"source": "big.pdf"})
            assert len(page) == 100 and next_offset == "100", (len(page), next_offset)
        finally:
            handler_mod._PAGE_SIZE = saved

    async def count(handler, query):
        return (await run(handler, query))[0]["n"]

    async def tenant_graph(handler):
        return {
            "edges": await count(handler, "MATCH (n {tenant_id: $tenant_id})-[r]-() "
                                          "WHERE NOT type(r) IN ['BELONGS_TO', 'PART_OF', 'NEXT'] RETURN count(r) AS n"),
            "entities": await count(handler, "MATCH (e:Entity {tenant_id: $tenant_id}) RETURN count(e) AS n"),
            "documents": await count(handler, "MATCH (d:Document {tenant_id: $tenant_id}) RETURN count(d) AS n"),
            "files": await count(handler, "MATCH (s:SourceFile {tenant_id: $tenant_id}) RETURN count(s) AS n"),
        }

    async def test_wipe_tenant_graph(handler, other):
        gen = await handler._read_generation(TENANT)
        ids = await add_points(handler, gf.DECLARATIVE, "w.pdf", 2)
        await link(handler, ids, "WipeEntity", "w.pdf", gen)
        other_gen = await other._read_generation(OTHER)
        other_ids = await add_points(other, gf.DECLARATIVE, "w.pdf", 2)
        await link(other, other_ids, "WipeEntity", "w.pdf", other_gen)
        await run(handler, "CREATE (:Entity {id: $eid, tenant_id: $tenant_id, name: 'lonely'})",
                  eid=f"{TENANT}:lonely")
        before, other_before = await tenant_graph(handler), await tenant_graph(other)
        assert before["edges"] and before["entities"], before

        await handler.wipe_tenant_graph()
        after = await tenant_graph(handler)
        assert after["edges"] == 0 and after["entities"] == 0, after
        assert (after["documents"], after["files"]) == (before["documents"], before["files"]), (before, after)
        assert await tenant_graph(other) == other_before, "the wipe touched another tenant"
        await handler.wipe_tenant_graph()  # idempotent on an empty graph

    async def test_verify_graph_checks(handler, other):
        await add_points(handler, gf.DECLARATIVE, "v.pdf", 2)
        checks = dict(await cli._graph_checks(handler, TENANT))
        assert checks == {"no-null-embeddings": True, "no-orphan-entities": True}, checks
        await run(handler, "CREATE (:Entity {id: $eid, tenant_id: $tenant_id, name: 'orphan'})", eid=f"{TENANT}:orphan")
        await run(handler, "CREATE (:Document {id: $did, tenant_id: $tenant_id})", did=uuid.uuid4().hex)
        checks = dict(await cli._graph_checks(handler, TENANT))
        assert checks == {"no-null-embeddings": False, "no-orphan-entities": False}, checks
        handler._enable_concept_relations = True
        try:
            checks = dict(await cli._graph_checks(handler, TENANT))
            assert checks["concept-gen-active"] is True, checks  # no marker yet: the current fingerprint is visible
            await handler._write_concept_gen(TENANT, "an-old-generation")
            checks = dict(await cli._graph_checks(handler, TENANT))
            assert checks["concept-gen-active"] is False, checks
        finally:
            handler._enable_concept_relations = False

    async def test_stored_metadata_are_the_callers(handler, other):
        """The metadata read back are the ones the caller stored (the core takes the metadata shared by the points of
        a source for its upload metadata): no tenant_id, also for the Documents older versions wrote it in."""
        meta = {"source": "m.pdf", "author": "Erin"}
        point = PointStruct(id=uuid.uuid4().hex, payload={"page_content": "Erin wrote", "metadata": meta}, vector=[1.0, 2.0])
        await handler.add_points_to_tenant(gf.DECLARATIVE, [point])
        await asyncio.gather(*handler._pending_entity_tasks, return_exceptions=True)
        assert meta == {"source": "m.pdf", "author": "Erin"}, f"the caller's metadata were changed: {meta}"
        await run(handler, "MATCH (d:Document {id: $id, tenant_id: $tenant_id}) SET d.metadata = $old",
                  id=point.id, old=json.dumps(meta | {"tenant_id": TENANT}))  # written by an older version
        points, _ = await handler.get_all_tenant_points(gf.DECLARATIVE, metadata={"source": "m.pdf"}, with_vectors=False)
        assert [p.payload["metadata"] for p in points] == [meta], [p.payload["metadata"] for p in points]
        records = await handler.retrieve_tenant_points(gf.DECLARATIVE, [point.id])
        assert [r.payload["metadata"] for r in records] == [meta], [r.payload["metadata"] for r in records]
        recalled = [d.document.metadata for d in await handler.recall_tenant_memory(gf.DECLARATIVE)
                    if d.document.id == point.id]
        assert recalled == [meta], recalled
        found = [p.payload["metadata"] for p in await handler.search_in_tenant(gf.DECLARATIVE, [1.0, 2.0], limit=1000)
                 if p.id == point.id]
        assert found == [meta], found

    async def test_stale_visible_concepts(handler, other):
        """Visible concepts of a generation other than the current fingerprint: on nodes, or on RELATED_TO edges only
        (concepts linking entities spaCy also found); deactivated ones and other tenants' do not count."""
        assert await handler.has_stale_visible_concepts("gen-new") is False  # no concept at all
        await run(handler, "CREATE (:Entity {id: $id, tenant_id: $tenant_id, name: 'c', concept_gen: 'gen-old'})",
                  id=f"{TENANT}:concept-old")
        assert await handler.has_stale_visible_concepts("gen-new") is True
        assert await handler.has_stale_visible_concepts("gen-old") is False
        await run(handler, "MATCH (e:Entity {id: $id, tenant_id: $tenant_id}) SET e.concept_gen_active = false",
                  id=f"{TENANT}:concept-old")
        assert await handler.has_stale_visible_concepts("gen-new") is False, "a deactivated concept counted"
        await run(
            handler,
            """
            CREATE (a:Entity {id: $a, tenant_id: $tenant_id, name: 'a'})-[:RELATED_TO {type: 'USES', concept_gen: 'gen-old'}]->
                   (b:Entity {id: $b, tenant_id: $tenant_id, name: 'b'})
            """,
            a=f"{TENANT}:spacy-a", b=f"{TENANT}:spacy-b",
        )
        assert await handler.has_stale_visible_concepts("gen-new") is True, "a concept edge between spaCy nodes missed"
        await run(handler, "MATCH (:Entity {tenant_id: $tenant_id})-[r:RELATED_TO]->() SET r.concept_gen_active = false")
        assert await handler.has_stale_visible_concepts("gen-new") is False
        await run(other, "CREATE (:Entity {id: $id, tenant_id: $tenant_id, name: 'c', concept_gen: 'gen-old'})",
                  id=f"{OTHER}:concept-old")
        assert await handler.has_stale_visible_concepts("gen-new") is False, "another tenant's concepts counted"

    async def test_a_concept_found_again_survives_the_flip(handler, other):
        """A recompute finds again a concept (and its relation) of the previous generation: the node takes the new
        generation like its edge, so the flip to it keeps both visible and the GC keeps the node."""
        payload = {
            "concepts": [{"text": "Fattura elettronica", "type": "DEFINITION"}, {"text": "SDI", "type": "DEFINITION"}],
            "relations": [{"origin": "Fattura elettronica", "destination": "SDI", "type": "SENT_TO", "text": "sent to"}],
        }
        await handler._store_concept_relations(TENANT, payload, source="f.pdf", document_ids=["d-f"], concept_gen="gen-a")
        await handler._flip_concept_gen(TENANT, await handler._read_concept_gen(TENANT), "gen-a")
        # the recompute with another LLM: the same concepts again, tagged with the new generation
        await handler._store_concept_relations(TENANT, payload, source="f.pdf", document_ids=["d-f"], concept_gen="gen-b")
        assert await handler._flip_concept_gen(TENANT, "gen-a", "gen-b")
        await handler._gc_stale_concept_nodes(TENANT)
        nodes = await run(handler, "MATCH (e:Entity {tenant_id: $tenant_id}) WHERE e.name IN $names "
                                   "RETURN e.name AS name, e.concept_gen AS gen, e.concept_gen_active AS active",
                          names=["fattura elettronica", "sdi"])
        assert sorted((r["name"], r["gen"], r["active"]) for r in nodes) == [
            ("fattura elettronica", "gen-b", None), ("sdi", "gen-b", None),
        ], [dict(r) for r in nodes]
        edges = await run(handler, "MATCH (:Entity {tenant_id: $tenant_id})-[r:RELATED_TO {type: 'SENT_TO'}]->() "
                                   "RETURN r.concept_gen AS gen, r.concept_gen_active AS active")
        assert [(r["gen"], r["active"]) for r in edges] == [("gen-b", None)], [dict(r) for r in edges]

    async def relate(handler, doc_ids, key, a="A", b="B"):
        """A RELATED_TO between two entities mentioned by the documents, written by the real source-graph writer with
        the provenance ``key``."""
        ids = [f"{TENANT}:{a}", f"{TENANT}:{b}"]
        await handler._write_source_graph(
            [{"id": i, "name": i.lower(), "type": "PERSON", "metadata": "{}", "embedding": None} for i in ids],
            [{"doc_id": d, "entity_id": i, "confidence": 0.9} for d in doc_ids for i in ids],
            [{"source_id": ids[0], "target_id": ids[1], "rel_type": "MET", "weight": 1.0, "source_files": [key]}],
        )

    async def keys_of(handler, a="A", b="B"):
        rows = await run(handler, "MATCH (:Entity {id: $a, tenant_id: $tenant_id})-[r:RELATED_TO {type: 'MET'}]->"
                                  "(:Entity {id: $b, tenant_id: $tenant_id}) RETURN r.source_files AS keys",
                         a=f"{TENANT}:{a}", b=f"{TENANT}:{b}")
        return None if not rows else set(rows[0]["keys"] or [])

    async def test_provenance_is_scoped_by_kb_and_chat(handler, other):
        """The same relation from a knowledge base file and a chat file with the same name: each deletion path
        strips only its own scope."""
        key = handler_mod.provenance_key
        kb = await add_points(handler, gf.DECLARATIVE, "p.pdf", 1)
        chat = await add_points(handler, gf.EPISODIC, "p.pdf", 1, chat_id="c1")
        await relate(handler, kb, key("p.pdf"))
        await relate(handler, chat, key("p.pdf", "c1"))
        assert await keys_of(handler) == {"p.pdf", "chat:c1:p.pdf"}

        await handler.delete_tenant_points(gf.EPISODIC, {"source": "p.pdf", "chat_id": "c1"})  # the chat file deleted
        assert await keys_of(handler) == {"p.pdf"}, "deleting the chat file touched the knowledge base"

        chat = await add_points(handler, gf.EPISODIC, "p.pdf", 1, chat_id="c1")
        await relate(handler, chat, key("p.pdf", "c1"))
        await handler.purge_source_graph_artifacts("p.pdf", gf.DECLARATIVE)  # the KB file's graph rebuilt
        assert await keys_of(handler) == {"chat:c1:p.pdf"}, "purging the knowledge base touched the chat"

        await relate(handler, kb, key("p.pdf"))
        await handler.delete_tenant_points(gf.EPISODIC, {"chat_id": "c1"})  # the chat destroyed
        assert await keys_of(handler) == {"p.pdf"}, "destroying the chat left (or took too much)"

        await handler.delete_tenant_points(gf.PROCEDURAL)  # at every plugin toggle: it holds no source
        assert await keys_of(handler) == {"p.pdf"}, "deleting the procedural memory stripped the knowledge base"

        chat2 = await add_points(handler, gf.EPISODIC, "p.pdf", 1, chat_id="c2")
        await relate(handler, chat2, key("p.pdf", "c2"))
        await handler.delete_tenant_points(gf.EPISODIC)  # the whole episodic memory
        assert await keys_of(handler) == {"p.pdf"}, "deleting the chats' collection touched the knowledge base"
        await handler.delete_tenant_points(gf.DECLARATIVE)
        assert await keys_of(handler) is None, "a relation without provenance survived"

    async def test_concepts_and_backfill_key_by_the_chat(handler, other):
        key = handler_mod.provenance_key
        payload = {
            "concepts": [{"text": "Zeta", "type": "DEFINITION"}, {"text": "Eta", "type": "DEFINITION"}],
            "relations": [{"origin": "Zeta", "destination": "Eta", "type": "LINKS", "text": "links"}],
        }
        await handler._store_concept_relations(TENANT, payload, source=key("z.pdf", "c9"), document_ids=["dz"],
                                               concept_gen="g")
        rows = await run(handler, "MATCH (:Entity {tenant_id: $tenant_id})-[r:RELATED_TO {type: 'LINKS'}]->() "
                                  "RETURN r.source_files AS keys")
        assert [r["keys"] for r in rows] == [["chat:c9:z.pdf"]], [dict(r) for r in rows]
        await handler.delete_tenant_points(gf.EPISODIC, {"chat_id": "c9"})
        rows = await run(handler, "MATCH (:Entity {tenant_id: $tenant_id})-[r:RELATED_TO {type: 'LINKS'}]->() RETURN r")
        assert not rows, "the chat's concept relation survived its chat"

        # an edge written before the provenance existed, between entities a chat document mentions: the boot
        # backfill keys it by the chat
        docs = await add_points(handler, gf.EPISODIC, "q.pdf", 1, chat_id="c7")
        await relate(handler, docs, key("q.pdf", "c7"), a="Q1", b="Q2")
        await run(handler, "MATCH (:Entity {id: $a, tenant_id: $tenant_id})-[r:RELATED_TO]->() REMOVE r.source_files",
                  a=f"{TENANT}:Q1")
        await handler.recompute_provenance()
        assert await keys_of(handler, "Q1", "Q2") == {"chat:c7:q.pdf"}, await keys_of(handler, "Q1", "Q2")

    async def main_async():
        handler = make_handler()
        other = make_handler(OTHER)
        tests = (
            test_purge_is_scoped_to_collection_and_chat, test_points_read_once_across_pages,
            test_wipe_tenant_graph, test_verify_graph_checks, test_stored_metadata_are_the_callers,
            test_stale_visible_concepts, test_a_concept_found_again_survives_the_flip,
            test_provenance_is_scoped_by_kb_and_chat, test_concepts_and_backfill_key_by_the_chat,
        )
        failed = 0
        try:
            await handler.initialize("fake-embedder", 2)
            await other.initialize("fake-embedder", 2)
            for test in tests:
                try:
                    await test(handler) if test.__code__.co_argcount == 1 else await test(handler, other)
                    print(f"PASS {test.__name__}")
                except Exception:  # noqa: BLE001 - report every failure
                    failed += 1
                    print(f"FAIL {test.__name__}")
                    traceback.print_exc()
        finally:
            for h in (handler, other):
                await run(h, "MATCH (n {tenant_id: $tenant_id}) DETACH DELETE n")
                await h.close()
                await gf.destroy_agent(h.agent_id)
        print(f"\ntest_graph_purge_live: {len(tests) - failed}/{len(tests)} passed")
        return failed

    sys.exit(1 if asyncio.run(main_async()) else 0)
