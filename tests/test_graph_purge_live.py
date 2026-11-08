"""The source-scoped Cypher of the graph phases, executed on a LIVE Neo4j.

- ``purge_source_graph_artifacts(source, collection, chat_id)`` removes the entity / similarity edges of the documents
  of THAT source only: a file with the same name in the knowledge base keeps its graph when the copy in a chat is
  purged (and vice versa); the documents and the SourceFile stay;
- ``get_source_points`` reads every point of a source exactly once across pages (the paged listing is ordered);
- ``wipe_tenant_graph`` (the maintenance ``--graph``) removes every graph edge and entity of the tenant, keeps its
  documents and structure, and never touches another tenant;
- the metadata read back are the ones the caller stored (no ``tenant_id``, also on Documents written before);
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
        """The graph the phases write for a source: MENTIONS + PROVENANCE, a similarity edge, a RELATED_TO.
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
        kb = await add_points(handler, "declarative", "x.pdf", 2)
        chat = await add_points(handler, "episodic", "x.pdf", 2, chat_id="c1")
        other_chat = await add_points(handler, "episodic", "x.pdf", 2, chat_id="c2")
        await link(handler, kb, "KbEntity", "x.pdf", gen)
        await link(handler, chat, "ChatEntity", "x.pdf", gen)
        await link(handler, other_chat, "OtherEntity", "x.pdf", gen)
        before = {
            name: await edges_of(handler, ids, exclude=chat + kb)
            for name, ids in (("kb", kb), ("c1", chat), ("c2", other_chat))
        }
        assert all(before.values()), before

        count = await handler.purge_source_graph_artifacts("x.pdf", "episodic", "c1")
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

        assert await handler.purge_source_graph_artifacts("x.pdf", "declarative") == 2
        assert await edges_of(handler, kb) == 0
        assert await edges_of(handler, other_chat, exclude=chat + kb) == before["c2"], "purging the KB touched a chat"
        assert await handler.purge_source_graph_artifacts("x.pdf", "declarative") == 2  # idempotent

    async def test_points_read_once_across_pages(handler):
        # pages of 100 instead of 1000: the same paging over 3 pages, without inserting thousands of points (an insert
        # is slow on recent Neo4j versions)
        saved, handler_mod._PAGE_SIZE = handler_mod._PAGE_SIZE, 100
        try:
            ids = await add_points(handler, "declarative", "big.pdf", 235)
            points = await handler.get_source_points("declarative", "big.pdf")
            read = [p.id for p in points]
            assert len(read) == len(set(read)) == 235 and set(read) == set(ids), (len(read), len(set(read)))
            # the core's contract, which its embedding phase and the vector-memory transfer rely on: no limit = all
            every, next_offset = await handler.get_all_tenant_points(
                "declarative", metadata={"source": "big.pdf"}, with_vectors=False
            )
            assert len(every) == 235 and next_offset is None, (len(every), next_offset)
            page, next_offset = await handler.get_all_tenant_points("declarative", limit=100, metadata={"source": "big.pdf"})
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
        ids = await add_points(handler, "declarative", "w.pdf", 2)
        await link(handler, ids, "WipeEntity", "w.pdf", gen)
        other_gen = await other._read_generation(OTHER)
        other_ids = await add_points(other, "declarative", "w.pdf", 2)
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
        await add_points(handler, "declarative", "v.pdf", 2)
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
        await handler.add_points_to_tenant("declarative", [point])
        await asyncio.gather(*handler._pending_entity_tasks, return_exceptions=True)
        assert meta == {"source": "m.pdf", "author": "Erin"}, f"the caller's metadata were changed: {meta}"
        await run(handler, "MATCH (d:Document {id: $id, tenant_id: $tenant_id}) SET d.metadata = $old",
                  id=point.id, old=json.dumps(meta | {"tenant_id": TENANT}))  # written by an older version
        points, _ = await handler.get_all_tenant_points("declarative", metadata={"source": "m.pdf"}, with_vectors=False)
        assert [p.payload["metadata"] for p in points] == [meta], [p.payload["metadata"] for p in points]
        records = await handler.retrieve_tenant_points("declarative", [point.id])
        assert [r.payload["metadata"] for r in records] == [meta], [r.payload["metadata"] for r in records]
        recalled = [d.document.metadata for d in await handler.recall_tenant_memory("declarative")
                    if d.document.id == point.id]
        assert recalled == [meta], recalled
        found = [p.payload["metadata"] for p in await handler.search_in_tenant("declarative", [1.0, 2.0], limit=1000)
                 if p.id == point.id]
        assert found == [meta], found

    async def main_async():
        handler = make_handler()
        other = make_handler(OTHER)
        tests = (
            test_purge_is_scoped_to_collection_and_chat, test_points_read_once_across_pages,
            test_wipe_tenant_graph, test_verify_graph_checks, test_stored_metadata_are_the_callers,
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
