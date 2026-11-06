"""Standalone regression test for the lost extraction result.

Run as a plain script:

    python tests/test_pending_source_tasks.py

The Cat plugin loader imports EVERY ``.py`` under a plugin folder, so the whole
body lives inside ``if __name__ == "__main__":`` — importing this file executes
nothing and needs no prerequisites.

Regression: ``add_point_to_tenant`` used to prune FINISHED tasks out of
``_pending_source_tasks`` at the end of every call. A chunk whose extraction
finished before that prune was removed from the per-source list and its result
was never gathered by ``drain_source_tasks`` — silently missing entities and
relations for a random subset of chunks. This test drives three points through
the REAL ``add_point_to_tenant`` (Neo4j session and spaCy extraction replaced by
test doubles), lets every extraction task finish BEFORE the next prune point,
then drains: all three results must be written in ONE batch.
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

    async def test_finished_before_prune_still_contributes():
        source = "file.pdf"
        writes = []
        handler = _make_handler(writes)

        point_ids = [f"chunk-{i}" for i in range(3)]
        for point_id in point_ids:
            await handler.add_point_to_tenant(
                collection_name="declarative",
                content=f"content {point_id}",
                vector=[0.1, 0.2, 0.3],
                metadata={"source": source},
                id_point=point_id,
            )
            # Let this point's extraction finish BEFORE the next
            # add_point_to_tenant reaches its (former) prune point.
            for _ in range(5):
                await asyncio.sleep(0)

        tracked = handler._pending_source_tasks.get(source, [])
        # One extraction + one similarity task per point; NONE may be dropped.
        assert len(tracked) == 6, f"finished tasks were dropped: {len(tracked)}"
        assert all(task.done() for task in tracked), "tasks should be finished before drain"
        assert sum(isinstance(task.result(), dict) for task in tracked) == 3

        flushed = await handler.drain_source_tasks(source)
        assert flushed is True, "drain must flush the completed batch"
        assert len(writes) == 1, f"expected ONE batch write, got {len(writes)}"
        entities = writes[0][0]
        assert len(entities) == 3, f"expected 3 entities, got {len(entities)}"
        assert {e["id"] for e in entities} == {f"entity-{p}" for p in point_ids}

        # Idempotent: a second drain must not write again.
        assert await handler.drain_source_tasks(source) is False
        assert len(writes) == 1, "second drain must not duplicate the batch"

        # Explicit delete/purge cleanup releases an abandoned source's list.
        handler._track_source_task("deleted.pdf", asyncio.get_running_loop().create_future())
        handler._drop_source_tasks("deleted.pdf")
        assert "deleted.pdf" not in handler._pending_source_tasks

        print("PASS test_finished_before_prune_still_contributes")

    class _Point:
        def __init__(self, pid, content, source):
            self.id = pid
            self.vector = None
            self.payload = {"page_content": content, "metadata": {"source": source}}

    async def _track_points(handler, source, point_ids):
        for point_id in point_ids:
            await handler.add_point_to_tenant(
                collection_name="declarative",
                content=f"content {point_id}",
                vector=[0.1, 0.2, 0.3],
                metadata={"source": source},
                id_point=point_id,
            )
        for _ in range(5):
            await asyncio.sleep(0)

    async def test_partial_failure_is_not_a_complete_flush():
        """A failed or cancelled chunk task must not let the phase skip the
        re-extraction: the partial batch is written, but drain reports the
        flush as incomplete and the spaCy phase re-extracts every chunk."""
        source = "partial.pdf"
        writes = []
        handler = _make_handler(writes)
        point_ids = ["ok-0", "boom-1", "ok-2"]
        good_extract = handler._extract_document_graph

        async def flaky_extract(document_id, content, metadata):
            if document_id.startswith("boom"):
                raise RuntimeError("spaCy crashed")
            return await good_extract(document_id, content, metadata)

        handler._extract_document_graph = flaky_extract
        await _track_points(handler, source, point_ids)

        flushed = await handler.drain_source_tasks(source)
        assert flushed is False, "a partial batch must not count as a complete flush"
        assert len(writes) == 1 and len(writes[0][0]) == 2, f"partial batch not written: {writes}"

        # Cancelled task: same rule (CancelledError is not an Exception).
        writes.clear()
        await _track_points(handler, "cancel.pdf", ["c-0"])
        pending = asyncio.get_running_loop().create_future()
        pending.cancel()
        handler._track_source_task("cancel.pdf", pending)
        assert await handler.drain_source_tasks("cancel.pdf") is False

        # The phase re-extracts every chunk after an incomplete flush.
        writes.clear()
        handler._extract_document_graph = flaky_extract
        await _track_points(handler, source, point_ids)
        extracted = []

        async def counting_extract(document_id, content, metadata):
            extracted.append(document_id)
            return await good_extract(document_id, content, metadata)

        handler._extract_document_graph = counting_extract

        async def by_collection(src):
            return {"declarative": [_Point(pid, f"content {pid}", src) for pid in point_ids]}

        handler.get_source_points_by_collection = by_collection
        result = await handler.run_spacy_extraction_phase(source)
        assert result["status"] == "done", result
        assert sorted(extracted) == sorted(point_ids), f"phase did not re-extract: {extracted}"
        assert result["processed"] == 3, result
        print("PASS test_partial_failure_is_not_a_complete_flush")

    async def test_llm_phase_leaves_spacy_tasks_alone():
        """The LLM phase must not join/write the spaCy batch: a spaCy write
        failure would be recorded as an LLM-phase error."""
        source = "llm.pdf"
        writes = []
        handler = _make_handler(writes)
        await _track_points(handler, source, ["l-0", "l-1"])

        async def by_collection(src):
            return {"declarative": [_Point("l-0", "content l-0", src)]}

        handler.get_source_points_by_collection = by_collection
        result = await handler.run_llm_extraction_phase(source)
        assert result["status"] == "skipped", result
        assert writes == [], f"LLM phase wrote the spaCy batch: {writes}"
        assert handler._pending_source_tasks.get(source), "LLM phase consumed the spaCy tasks"
        print("PASS test_llm_phase_leaves_spacy_tasks_alone")

    async def test_drain_counts_released_after_phase():
        """Per-source drain counters are consumed by the phase, not kept for
        the life of the process."""
        source = "counts.pdf"
        writes = []
        handler = _make_handler(writes)
        await _track_points(handler, source, ["k-0"])

        async def by_collection(src):
            return {"declarative": [_Point("k-0", "content k-0", src)]}

        handler.get_source_points_by_collection = by_collection
        result = await handler.run_spacy_extraction_phase(source)
        assert result == {"status": "done", "processed": 1}, result
        assert source not in getattr(handler, "_last_drain_counts", {}), "per-source drain state kept"
        print("PASS test_drain_counts_released_after_phase")

    async def test_similarity_failure_keeps_flush_complete():
        """A failed similarity task loses no extraction product: the batch is
        still complete and the phase must not re-run NER for every chunk."""
        source = "sim.pdf"
        writes = []
        handler = _make_handler(writes)

        async def failing_similarity(document_id, vector, collection_name):
            raise RuntimeError("similarity write failed")

        handler._create_similarity_relationships = failing_similarity
        await _track_points(handler, source, ["s-0", "s-1"])
        flushed = await handler.drain_source_tasks(source)
        assert flushed is True, "a similarity failure must not mark the batch incomplete"
        assert len(writes) == 1 and len(writes[0][0]) == 2, writes
        print("PASS test_similarity_failure_keeps_flush_complete")

    async def main():
        await test_finished_before_prune_still_contributes()
        await test_partial_failure_is_not_a_complete_flush()
        await test_llm_phase_leaves_spacy_tasks_alone()
        await test_drain_counts_released_after_phase()
        await test_similarity_failure_keeps_flush_complete()

    asyncio.run(main())
