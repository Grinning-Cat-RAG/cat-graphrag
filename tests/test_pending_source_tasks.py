"""Standalone regression test for the lost extraction result.

Run as a plain script:

    python tests/test_pending_source_tasks.py

The Cat plugin loader imports EVERY ``.py`` under a plugin folder, so the whole
body lives inside ``if __name__ == "__main__":`` — importing this file executes
nothing and needs no prerequisites.

Regression: ``add_point_to_tenant`` used to prune FINISHED tasks out of
``_pending_source_tasks`` at the end of every call. A chunk whose extraction
finished before that prune was removed from the per-source list and its result
was never gathered — silently missing entities and relations for a random
subset of chunks. These checks drive points through the REAL
``add_point_to_tenant`` (Neo4j session and spaCy extraction replaced by test
doubles) and the REAL spaCy phase body (``run_spacy_extraction_phase``):

- every finished task contributes, in ONE batch write, after the purge;
- a failed or cancelled chunk makes the phase re-extract every chunk;
- a failed similarity task does not;
- the LLM phase never consumes the spaCy batch;
- the tasks of a source in a chat and in the knowledge base with the same name
  are separate, and the delete/purge cleanup releases them;
- the tasks of points the source no longer has (replaced by a new upload) are
  left out; a current point without its extraction makes the phase re-extract.
"""

if __name__ == "__main__":
    import asyncio
    import importlib
    import importlib.util
    import os
    import sys
    import types

    TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(TESTS_DIR)

    def _load_kg_harness():
        """Reuse the import-safe stub set from the sibling standalone test."""
        path = os.path.join(TESTS_DIR, "test_kg_master_switch.py")
        spec = importlib.util.spec_from_file_location("_kg_stub_harness", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module._install_stubs()
        return module

    _load_kg_harness()

    _pkg = types.ModuleType("catgraphrag_pendingtest")
    _pkg.__path__ = [REPO_ROOT]
    sys.modules["catgraphrag_pendingtest"] = _pkg
    graphrag_handler = importlib.import_module("catgraphrag_pendingtest.graphrag_handler")
    from cat.services.memory.models import VectorMemoryType  # the stub of the core's enum (test_kg_master_switch)

    DECLARATIVE, EPISODIC = str(VectorMemoryType.DECLARATIVE), str(VectorMemoryType.EPISODIC)

    class _FakeResult:
        async def single(self):
            return {"id": "fake"}

        async def data(self):
            return []

        async def consume(self):
            return None

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def run(self, query, **params):
            return _FakeResult()

    def _make_handler(writes):
        handler = graphrag_handler.GraphRAGHandler(
            neo4j_uri="bolt://fake",
            neo4j_user="u",
            neo4j_password="p",
            enable_knowledge_graph=True,
            enable_concept_relations=False,
        )
        handler.agent_id = "agent_pending"
        handler._generation = 1
        handler._names = {"embedding_prop": "embedding_1"}
        handler._get_session = lambda: _FakeSession()

        async def ensure_connected():
            return None

        async def read_generation():
            return 1

        async def noop_similarity(document_id, vector, collection_name):
            return None

        async def extract(document_id, content, metadata):
            return {
                "entities": [{"id": f"entity-{document_id}", "name": document_id}],
                "mentions": [{"doc_id": document_id, "entity_id": f"entity-{document_id}"}],
                "relations": [],
            }

        async def write_source_graph(entities, mentions, relations, operation=""):
            writes.append((list(entities), list(mentions), list(relations), operation))

        handler._ensure_connected = ensure_connected
        handler._read_generation = read_generation
        handler._rebuild_for_generation = lambda gen: None
        handler._create_similarity_relationships = noop_similarity
        handler._extract_document_graph = extract
        handler._write_source_graph = write_source_graph
        return handler

    class _Point:
        def __init__(self, pid, content, source):
            self.id = pid
            self.vector = None
            self.payload = {"page_content": content, "metadata": {"source": source}}

    async def _track_points(handler, source, point_ids, chat_id=None):
        for point_id in point_ids:
            await handler.add_point_to_tenant(
                collection_name=DECLARATIVE if chat_id is None else EPISODIC,
                content=f"content {point_id}",
                vector=[0.1, 0.2, 0.3],
                metadata={"source": source} | ({"chat_id": chat_id} if chat_id else {}),
                id_point=point_id,
            )
            # Let this point's extraction finish BEFORE the next
            # add_point_to_tenant reaches its (former) prune point.
            for _ in range(5):
                await asyncio.sleep(0)

    def _stored(handler, source, point_ids, events):
        async def get_source_points(collection_name, src, chat_id=None):
            return [_Point(pid, f"content {pid}", source) for pid in point_ids]

        async def purge(src, collection_name, chat_id=None):
            events.append(("purge", src, collection_name, chat_id))
            handler._drop_source_tasks(src, chat_id)
            return len(point_ids)

        handler.get_source_points = get_source_points
        handler.purge_source_graph_artifacts = purge

    async def test_finished_before_prune_still_contributes():
        source = "file.pdf"
        writes = []
        handler = _make_handler(writes)
        point_ids = [f"chunk-{i}" for i in range(3)]
        await _track_points(handler, source, point_ids)

        tracked = handler._pending_source_tasks.get((source, None), [])
        # One extraction + one similarity task per point; NONE may be dropped.
        assert len(tracked) == 6, f"finished tasks were dropped: {len(tracked)}"
        assert all(task.done() for task in tracked), "tasks should be finished before the phase"

        events = []
        _stored(handler, source, point_ids, events)
        extracted = []
        original = handler._extract_document_graph

        async def counting(document_id, content, metadata):
            extracted.append(document_id)
            return await original(document_id, content, metadata)

        handler._extract_document_graph = counting
        result = await handler.run_spacy_extraction_phase(source, DECLARATIVE)
        assert result == {"status": "done", "processed": 3}, result
        assert events == [("purge", source, DECLARATIVE, None)], events
        assert len(writes) == 1, f"expected ONE batch write, got {len(writes)}"
        assert {e["id"] for e in writes[0][0]} == {f"entity-{p}" for p in point_ids}
        assert extracted == [], "a complete batch must not be extracted again"
        assert (source, None) not in handler._pending_source_tasks, "the phase must consume the tasks"

        # A re-run (no task in memory: e.g. after a restart) replaces the graph by re-extracting.
        result = await handler.run_spacy_extraction_phase(source, DECLARATIVE)
        assert result == {"status": "done", "processed": 3}, result
        assert sorted(extracted) == sorted(point_ids) and len(writes) == 2
        print("PASS test_finished_before_prune_still_contributes")

    async def test_failed_or_cancelled_chunk_reextracts_every_chunk():
        """A failed or cancelled chunk task must not let the phase skip the
        re-extraction: the incomplete batch is dropped and every chunk is
        extracted again, written in ONE batch."""
        for kind in ("failed", "cancelled"):
            source = f"{kind}.pdf"
            writes = []
            handler = _make_handler(writes)
            point_ids = ["ok-0", "boom-1", "ok-2"]
            good_extract = handler._extract_document_graph

            async def flaky_extract(document_id, content, metadata, good=good_extract):
                if document_id.startswith("boom"):
                    raise RuntimeError("spaCy crashed")
                return await good(document_id, content, metadata)

            if kind == "failed":
                handler._extract_document_graph = flaky_extract
                await _track_points(handler, source, point_ids)
                handler._extract_document_graph = good_extract
            else:
                await _track_points(handler, source, point_ids)
                pending = asyncio.get_running_loop().create_future()
                pending.cancel()
                handler._pending_source_tasks[(source, None)].append(pending)

            _stored(handler, source, point_ids, [])
            extracted = []

            async def counting(document_id, content, metadata, good=good_extract):
                extracted.append(document_id)
                return await good(document_id, content, metadata)

            handler._extract_document_graph = counting
            result = await handler.run_spacy_extraction_phase(source, DECLARATIVE)
            assert result["status"] == "done", result
            assert sorted(extracted) == sorted(point_ids), f"{kind}: phase did not re-extract: {extracted}"
            assert len(writes) == 1 and len(writes[0][0]) == 3, f"{kind}: {writes}"
        print("PASS test_failed_or_cancelled_chunk_reextracts_every_chunk")

    async def test_llm_phase_leaves_spacy_tasks_alone():
        """The LLM phase must not join/write the spaCy batch: a spaCy write
        failure would be recorded as an LLM-phase error."""
        source = "llm.pdf"
        writes = []
        handler = _make_handler(writes)
        await _track_points(handler, source, ["l-0", "l-1"])
        _stored(handler, source, ["l-0"], [])
        result = await handler.run_llm_extraction_phase(source, DECLARATIVE)
        assert result["status"] == "skipped", result
        assert writes == [], f"LLM phase wrote the spaCy batch: {writes}"
        assert handler._pending_source_tasks.get((source, None)), "LLM phase consumed the spaCy tasks"
        print("PASS test_llm_phase_leaves_spacy_tasks_alone")

    async def test_similarity_failure_keeps_the_batch_complete():
        """A failed similarity task loses no extraction product: the batch is
        still complete and the phase must not re-run NER for every chunk."""
        source = "sim.pdf"
        writes = []
        handler = _make_handler(writes)

        async def failing_similarity(document_id, vector, collection_name):
            raise RuntimeError("similarity write failed")

        handler._create_similarity_relationships = failing_similarity
        await _track_points(handler, source, ["s-0", "s-1"])
        handler._create_similarity_relationships = lambda *a, **k: asyncio.sleep(0)
        _stored(handler, source, ["s-0", "s-1"], [])
        extracted = []

        async def counting(document_id, content, metadata):
            extracted.append(document_id)
            return None

        handler._extract_document_graph = counting
        result = await handler.run_spacy_extraction_phase(source, DECLARATIVE)
        assert result == {"status": "done", "processed": 2}, result
        assert extracted == [], "a similarity failure must not mark the batch incomplete"
        assert len(writes) == 1 and len(writes[0][0]) == 2, writes
        print("PASS test_similarity_failure_keeps_the_batch_complete")

    async def test_chat_and_kb_sources_with_the_same_name_are_separate():
        writes = []
        handler = _make_handler(writes)
        await _track_points(handler, "same.pdf", ["kb-0"])
        await _track_points(handler, "same.pdf", ["chat-0"], chat_id="c1")
        assert len(handler._pending_source_tasks[("same.pdf", None)]) == 2
        assert len(handler._pending_source_tasks[("same.pdf", "c1")]) == 2

        _stored(handler, "same.pdf", ["kb-0"], [])
        await handler.run_spacy_extraction_phase("same.pdf", DECLARATIVE)
        assert ("same.pdf", None) not in handler._pending_source_tasks
        assert handler._pending_source_tasks.get(("same.pdf", "c1")), "KB phase consumed the chat tasks"
        assert {e["id"] for e in writes[0][0]} == {"entity-kb-0"}, writes

        # Explicit delete/purge cleanup releases an abandoned source's list.
        handler._drop_source_tasks("same.pdf", "c1")
        assert ("same.pdf", "c1") not in handler._pending_source_tasks
        print("PASS test_chat_and_kb_sources_with_the_same_name_are_separate")

    async def test_tasks_without_points_are_joined_not_leaked():
        """No stored point: the phase fails, but the tasks it took are joined
        (nothing keeps a reference to them) and nothing is purged or written."""
        writes = []
        handler = _make_handler(writes)
        await _track_points(handler, "gone.pdf", ["g-0"])
        events = []
        _stored(handler, "gone.pdf", [], events)
        result = await handler.run_spacy_extraction_phase("gone.pdf", DECLARATIVE)
        assert result["status"] == "error", result
        assert events == [] and writes == [], (events, writes)
        assert ("gone.pdf", None) not in handler._pending_source_tasks
        assert not handler._pending_entity_tasks, "tasks left in the flat list"
        print("PASS test_tasks_without_points_are_joined_not_leaked")

    async def test_tasks_of_replaced_points_are_left_out():
        """Points written outside an ingestion and then uploaded again: the tasks of the replaced points are in the
        source's list too. Their product is stale (it would bring back the old content's graph): only the current
        points' extraction is written; a current point without its extraction makes the phase re-extract."""
        source = "replaced.pdf"
        writes = []
        handler = _make_handler(writes)
        await _track_points(handler, source, ["old-0", "old-1"])
        await _track_points(handler, source, ["new-0"])
        _stored(handler, source, ["new-0"], [])
        extracted = []
        good = handler._extract_document_graph

        async def counting(document_id, content, metadata):
            extracted.append(document_id)
            return await good(document_id, content, metadata)

        handler._extract_document_graph = counting
        result = await handler.run_spacy_extraction_phase(source, DECLARATIVE)
        assert result == {"status": "done", "processed": 1}, result
        assert [e["id"] for e in writes[0][0]] == ["entity-new-0"], writes
        assert extracted == [], "the current point had its extraction: nothing to extract again"

        writes.clear()
        handler._extract_document_graph = good
        await _track_points(handler, source, ["new-0"])
        _stored(handler, source, ["new-0", "new-1"], [])  # new-1 has no extraction in memory
        handler._extract_document_graph = counting
        result = await handler.run_spacy_extraction_phase(source, DECLARATIVE)
        assert result == {"status": "done", "processed": 2}, result
        assert sorted(extracted) == ["new-0", "new-1"], extracted
        assert sorted(e["id"] for e in writes[0][0]) == ["entity-new-0", "entity-new-1"], writes

        # a chunk whose extraction found nothing (no result) is extracted all the same: no re-extraction
        writes.clear()
        extracted.clear()

        async def nothing(document_id, content, metadata):
            return None

        handler._extract_document_graph = nothing
        await _track_points(handler, source, ["empty-0"])
        _stored(handler, source, ["empty-0"], [])
        handler._extract_document_graph = counting
        result = await handler.run_spacy_extraction_phase(source, DECLARATIVE)
        assert result["status"] == "done" and extracted == [] and writes == [], (result, extracted, writes)
        print("PASS test_tasks_of_replaced_points_are_left_out")

    async def main():
        await test_finished_before_prune_still_contributes()
        await test_failed_or_cancelled_chunk_reextracts_every_chunk()
        await test_llm_phase_leaves_spacy_tasks_alone()
        await test_similarity_failure_keeps_the_batch_complete()
        await test_chat_and_kb_sources_with_the_same_name_are_separate()
        await test_tasks_without_points_are_joined_not_leaked()
        await test_tasks_of_replaced_points_are_left_out()

    asyncio.run(main())
