"""The heavy work of the plugin's ingestion runs on the core's ingestion lane (``run_in_ingestion_executor``).

The core keeps a pool of de-prioritized threads for the heavy steps of an ingestion (``CAT_INGESTION_WORKERS``,
``CAT_INGESTION_NICENESS``): an upload must not compete on equal terms with the chats. The plugin's heavy steps of an
ingestion belong to it:

- the spaCy analysis of a chunk (``EntityExtractor.extract``);
- the embedding of the entities of a chunk (``_extract_document_graph``);
- the re-embedding of every Document of the tenant on a change of the embedder (``_reembed_tenant_impl``).

What serves a chat stays out of it: the analysis of the user's message (``extract_doc``) never waits behind an upload.
With a lane of ONE worker, many chunks extracted at once while the lane is busy all complete (nothing waits on itself).

Standalone script (plugin rules: the body runs only as ``__main__``). Needs the core checkout and a Redis Stack on
localhost:6379 (database 14):

    cd ../grinning-cat-core
    .venv/bin/python /path/to/cat-graphrag/tests/test_ingestion_lane.py
"""

if __name__ == "__main__":
    import asyncio
    import importlib
    import os
    import sys
    import threading
    import time
    import traceback
    import types

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import graph_fakes as gf

    os.environ["CAT_INGESTION_WORKERS"] = "1"  # the smallest lane: a job waiting on another job would hang
    main_mod, handler_mod, _ = gf.bootstrap("_graphrag_lane")
    extractor_mod = importlib.import_module("_graphrag_lane.entity_extractor")

    import spacy

    import cat.rabbit_hole as rabbit_hole_pkg
    from cat.rabbit_hole import core as rabbit_hole_core

    LANE = []  # the names of the callables the lane ran
    real_lane = rabbit_hole_pkg.run_in_ingestion_executor

    async def recording_lane(func, *args, **kwargs):
        LANE.append(getattr(func, "__name__", repr(func)))
        return await real_lane(func, *args, **kwargs)

    rabbit_hole_pkg.run_in_ingestion_executor = recording_lane
    rabbit_hole_core.run_in_ingestion_executor = recording_lane

    def ready_extractor():
        extractor = extractor_mod.EntityExtractor({"en": "en_core_web_sm"})
        blank = spacy.blank("en")  # no model to download: a tokenizer-only pipeline
        extractor._nlps = {"default": blank, "en": blank}
        extractor._initialized = True
        return extractor

    async def test_the_spacy_analysis_of_a_chunk_runs_on_the_lane():
        LANE.clear()
        extracted = await ready_extractor().extract("Alice met Bob in Paris. " * 20, "doc-1", {"source": "a.pdf"})
        assert extracted is not None
        assert LANE == ["_run_segment_pipe"], LANE

    async def test_the_analysis_of_a_chat_message_stays_off_the_lane():
        LANE.clear()
        await ready_extractor().extract_doc("Who is Alice?")
        assert LANE == [], LANE

    async def test_the_embedding_of_the_entities_runs_on_the_lane():
        handler = gf.make_handler(handler_mod, "lane_agent", enable_entity_embeddings=True)
        threads = []

        class Embedder:
            def embed_documents(self, texts):
                threads.append(threading.current_thread().name)
                return [[1.0, 0.5] for _ in texts]

        handler._embedder = Embedder()

        async def extract(text, document_id, metadata=None):
            entity = types.SimpleNamespace(name="Alice", type=handler_mod.EntityType.PERSON, confidence=0.9)
            return types.SimpleNamespace(entities=[entity], relations=[])

        handler._entity_extractor = types.SimpleNamespace(
            extract=extract, get_entity_hash=lambda name, etype, tenant: f"{tenant}:{name}",
        )
        LANE.clear()
        graph = await handler._extract_document_graph("doc-1", "Alice wrote", {"source": "a.pdf"})
        assert graph["entities"][0]["embedding"] == [1.0, 0.5], graph
        assert LANE == ["embed_documents"], LANE
        assert threads and threads[0].startswith("cat-ingestion"), threads

    async def test_the_reembedding_of_the_tenant_runs_on_the_lane():
        handler = gf.make_handler(handler_mod, "lane_agent")

        class Documents:
            """The read of every Document of the tenant; any later statement (the swap) is out of this test."""

            def __init__(self):
                self.rows = [{"id": "doc-1", "content": "Alice wrote", "collection_name": gf.DECLARATIVE}]

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.rows:
                    raise StopAsyncIteration
                return self.rows.pop()

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def run(self, query, **params):
                if "RETURN d.id AS id, d.content AS content" in query:
                    return Documents()
                raise RuntimeError("the swap is out of this test")

        async def generation(*a, **k):
            return "v1"

        handler._get_session = lambda: Session()
        handler._read_generation = generation

        class Embedder:
            def embed_documents(self, texts):
                return [[0.5, 0.5] for _ in texts]

        LANE.clear()
        try:
            await handler._reembed_tenant_impl("lane_agent", Embedder())
        except RuntimeError as exc:
            assert "out of this test" in str(exc), exc
        assert LANE == ["embed_documents"], LANE

    async def test_a_one_worker_lane_completes_many_chunks_while_busy():
        """The lane holds one job (e.g. the embedding of a source) while every chunk's analysis queues behind it:
        everything completes, in order, nothing waits on itself."""
        extractor = ready_extractor()
        release = threading.Event()

        def busy():
            release.wait(5)
            return "busy done"

        busy_job = asyncio.ensure_future(real_lane(busy))
        chunks = [extractor.extract(f"Alice met Bob {i}. " * 10, f"doc-{i}", {}) for i in range(20)]
        started = time.perf_counter()
        release.set()
        results = await asyncio.wait_for(asyncio.gather(busy_job, *chunks), timeout=60)
        assert results[0] == "busy done" and len(results) == 21
        assert time.perf_counter() - started < 60

    TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_")]

    async def run_all():
        failed = 0
        for test in TESTS:
            try:
                await test()
                print(f"PASS {test.__name__}")
            except Exception:  # noqa: BLE001 - report every failure
                failed += 1
                print(f"FAIL {test.__name__}")
                traceback.print_exc()
        print(f"\ntest_ingestion_lane: {len(TESTS) - failed}/{len(TESTS)} passed")
        return failed

    sys.exit(1 if asyncio.run(run_all()) else 0)
