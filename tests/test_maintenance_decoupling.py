#!/usr/bin/env python3
"""
Regression test: the maintenance CLI decouples ENGINE-level ops from GraphRAG.

``--reingest`` / ``--reembed`` must run for ANY agent (their phases are
engine-level; the graph phases no-op when the active handler is not
GraphRAG). Only ``--graph`` requires the ``GraphRAGHandler`` and must report
``skip reason=handler-not-graphrag`` otherwise. The re-embed preflight must
still block ``--reingest`` before any destructive wipe (needs the real EffING:
deployed in the container, or ``EFFING_DIR`` on the host).

Run from ``grinning-cat-core`` with ``EFFING_DIR`` set (no container needed):

    EFFING_DIR=/path/to/cat-efficient-ingestion MULTIMODAL_DIR=/path/to/cat-multimodal-ingestion PYTHONPATH=. \
        .venv/bin/python /path/to/cat-graphrag/tests/test_maintenance_decoupling.py

IMPORT-SAFETY: this file lives in the plugin folder, which the Cat plugin
loader imports recursively at activation. The ENTIRE body lives under
``if __name__ == "__main__":`` so importing the file executes zero code.
"""

if __name__ == "__main__":
    # Real EffING: deployed in the container, or EFFING_DIR on the host.
    from effing_checkout import ensure_effing_importable

    ensure_effing_importable()

    import asyncio
    import io
    import sys
    from contextlib import redirect_stdout
    from types import SimpleNamespace

    import importlib.util
    from pathlib import Path

    # Load the CLI by file path: no deployed plugin or container needed (the
    # CLI keeps every ``cat`` import inside functions).
    _cli = Path(__file__).resolve().parent.parent / "scripts" / "maintenance_agent.py"
    _spec = importlib.util.spec_from_file_location("maintenance_agent_under_test", _cli)
    m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(m)


    def _fail(msg: str) -> None:
        print(f"FAIL: {msg}")
        sys.exit(1)


    class FakeCcat:
        def __init__(self) -> None:
            self.agent_key = "test-agent"
            self.vector_memory_handler = None
            self.chunker = object()

        async def get_stored_sources_with_metadata(self):
            from cat.services.memory.models import VectorMemoryType

            return {VectorMemoryType("declarative"): []}


    class FakeVectorHandler:
        def __init__(self) -> None:
            self.deleted = []

        async def delete_tenant_points(self, collection, metadata=None):
            self.deleted.append((collection, metadata))


    async def _case_reingest_non_graphrag() -> None:
        calls = []

        async def fake_bootstrap(agent_id):
            return (FakeCcat(), None, "handler-not-graphrag")

        async def fake_engine(ccat, op="reembed"):
            return "EfficientIngestionConfiguration"

        async def fake_reingest(ccat, handler, collection):
            calls.append((handler, collection))
            return True

        orig = (
            m._bootstrap_agent,
            m._resolve_ingestion_engine,
            m._op_reingest,
        )
        m._bootstrap_agent = fake_bootstrap
        m._resolve_ingestion_engine = fake_engine
        m._op_reingest = fake_reingest
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                ok = await m._run_agent("test-agent", ["reingest"])
            out = buf.getvalue()
        finally:
            (m._bootstrap_agent, m._resolve_ingestion_engine, m._op_reingest) = orig

        if not ok:
            _fail(f"--reingest on a non-GraphRAG agent failed: {out!r}")
        if len(calls) != 1:
            _fail(f"--reingest helper calls={len(calls)}, expected 1: {out!r}")
        # The ``result=ok`` line is printed by ``_op_reingest`` itself (from the
        # real per-source phase outcome), which this case replaces.
        if "result=fail" in out:
            _fail(f"--reingest reported a failure: {out!r}")
        print("PASS (a): --reingest planned+executed for non-GraphRAG agent")


    async def _case_graph_non_graphrag_skipped() -> None:
        graph_calls = []

        async def fake_bootstrap(agent_id):
            return (FakeCcat(), None, "handler-not-graphrag")

        async def fake_graph_a(ccat, handler, collection):
            graph_calls.append(handler)
            return True

        orig = (m._bootstrap_agent, m._op_graph_part_a)
        m._bootstrap_agent = fake_bootstrap
        m._op_graph_part_a = fake_graph_a
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                ok = await m._run_agent("test-agent", ["graph"])
            out = buf.getvalue()
        finally:
            m._bootstrap_agent, m._op_graph_part_a = orig

        if not ok:
            _fail(f"--graph skip was not a success: {out!r}")
        if graph_calls:
            _fail("--graph ran graph work on a non-GraphRAG agent")
        if "op=graph result=skip reason=handler-not-graphrag" not in out:
            _fail(f"missing handler-not-graphrag skip: {out!r}")
        print("PASS (b): --graph skipped with reason=handler-not-graphrag")


    async def _case_preflight_blocks() -> None:
        from cat.services.memory.models import VectorMemoryType

        # 1. The real preflight refuses an unusable re-embed path.
        import cat.plugins.cat_efficient_ingestion.reembed as reembed

        orig_rh = reembed._resolve_rabbit_hole

        async def fake_rh(ccat):
            return SimpleNamespace(cat=ccat)

        reembed._resolve_rabbit_hole = fake_rh
        try:
            ccat = FakeCcat()
            ccat.chunker = None
            reason = await m._preflight_reembed_path(ccat)
        finally:
            reembed._resolve_rabbit_hole = orig_rh
        if reason != "chunker-unavailable":
            _fail(f"preflight reason={reason!r}, expected chunker-unavailable")

        # 2. _op_reingest aborts before any destructive delete when the
        #    preflight reports the path unusable.
        handler = FakeVectorHandler()
        ccat = FakeCcat()
        ccat.vector_memory_handler = handler
        ccat.get_stored_sources_with_metadata = (
            lambda: _sources_with_one_doc(VectorMemoryType)
        )

        orig_preflight = m._preflight_reembed_path

        async def fake_preflight(_ccat):
            return "rabbit-hole-unbound"

        m._preflight_reembed_path = fake_preflight
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                ok = await m._op_reingest(ccat, None, "declarative")
            out = buf.getvalue()
        finally:
            m._preflight_reembed_path = orig_preflight

        if ok:
            _fail(f"_op_reingest succeeded despite preflight failure: {out!r}")
        if handler.deleted:
            _fail(f"preflight failure still deleted points: {handler.deleted!r}")
        if "detail=preflight-reembed-path-unusable" not in out:
            _fail(f"missing preflight failure message: {out!r}")
        print("PASS (c): preflight still blocks --reingest before any wipe")


    async def _sources_with_one_doc(vector_memory_type):
        return {vector_memory_type("declarative"): [SimpleNamespace(name="doc.txt")]}


    async def _case_reingest_graphrag_unchanged() -> None:
        calls = []
        graph_handler = SimpleNamespace(
            _graphrag_detected=True,
            _concept_relations_enabled=True,
        )

        async def fake_bootstrap(agent_id):
            return (FakeCcat(), graph_handler, None)

        async def fake_engine(ccat, op="reembed"):
            return "EfficientIngestionConfiguration"

        async def fake_reingest(ccat, handler, collection):
            calls.append((handler, collection))
            return True

        orig = (m._bootstrap_agent, m._resolve_ingestion_engine, m._op_reingest)
        m._bootstrap_agent = fake_bootstrap
        m._resolve_ingestion_engine = fake_engine
        m._op_reingest = fake_reingest
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                ok = await m._run_agent("test-agent", ["reingest"])
            out = buf.getvalue()
        finally:
            (m._bootstrap_agent, m._resolve_ingestion_engine, m._op_reingest) = orig

        if not ok or len(calls) != 1:
            _fail(f"GraphRAG --reingest changed: {out!r}")
        if calls[0][0] is not graph_handler:
            _fail("GraphRAG --reingest did not receive the GraphRAG handler")
        print("PASS (d): --reingest for a GraphRAG agent is unchanged")


    async def _case_graph_part_a_leaves_llm_to_part_b() -> None:
        """--graph part A3 rebuilds the document structure only: the LLM step
        belongs to part B, so A3 must not run it a second time per source."""
        from unittest.mock import AsyncMock

        handler = SimpleNamespace(
            _walk_gen="v1",
            _read_generation=AsyncMock(return_value="v1"),
            create_derived_graph_for_source=AsyncMock(),
        )
        docs = [
            {"id": "d1", "content": "a", "metadata": {"source": "f.pdf", "chunk_index": 0}},
        ]
        names = ("_graph_wipe", "_reset_graph_phase_entries", "_graph_fetch_docs", "_graph_walk_batch")
        orig = tuple(getattr(m, n) for n in names)
        m._graph_wipe = AsyncMock()
        m._reset_graph_phase_entries = AsyncMock()
        m._graph_fetch_docs = AsyncMock(return_value=docs)
        m._graph_walk_batch = AsyncMock()
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                ok = await m._op_graph_part_a(FakeCcat(), handler, "declarative")
        finally:
            for n, v in zip(names, orig):
                setattr(m, n, v)
        if not ok:
            _fail(f"part A failed: {buf.getvalue()!r}")
        call = handler.create_derived_graph_for_source.await_args
        if call is None or call.kwargs.get("run_concepts") is not False:
            _fail(f"part A3 did not disable the LLM step: {call!r}")
        print("PASS (e): --graph part A3 leaves the LLM step to part B")


    async def _main() -> int:
        try:
            await _case_reingest_non_graphrag()
            await _case_graph_non_graphrag_skipped()
            await _case_preflight_blocks()
            await _case_reingest_graphrag_unchanged()
            await _case_graph_part_a_leaves_llm_to_part_b()
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 - report and fail
            _fail(f"{type(exc).__name__}: {exc}")
        print("PASS: maintenance engine ops decoupled from the GraphRAG handler")
        return 0


    raise SystemExit(asyncio.run(_main()))
