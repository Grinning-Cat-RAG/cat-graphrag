#!/usr/bin/env python3
"""Standalone, import-safe test: the CLI reports the REAL phase outcome.

Live bug: ``maintenance_agent.py --reingest`` printed
``op=reingest result=ok detail=reingested`` while the source's
``graph_llm_extraction`` phase was ``error`` and the source status was
``error``. The summary described "sources handed to the engine", not "phases
succeeded". ``_report_ingestion_outcome`` now reads every processed source's
status doc back and owns the final line.

Covered:
  (a) all sources completed        -> result=ok, returns True;
  (b) one source error             -> result=fail detail=phase-failed, naming
                                      the source, the failing phase(s) and the
                                      verbatim reason, returns False;
  (c) one source still processing  -> result=fail detail=still-processing;
  (d) two sources, one ok one error -> only the failing source is reported,
                                      the aggregate is fail, returns False.

IMPORT-SAFETY: this file lives in the plugin folder, which the Cat plugin
loader imports recursively at activation. The ENTIRE body lives under
``if __name__ == "__main__":`` so importing the file executes zero code.
Run it as a plain script from the host repository:

    cd ../grinning-cat-core
    EFFING_DIR=/path/to/cat-efficient-ingestion MULTIMODAL_DIR=/path/to/cat-multimodal-ingestion PYTHONPATH=. .venv/bin/python /path/to/cat-graphrag/tests/test_maintenance_exit_summary.py
"""

if __name__ == "__main__":
    # Real EffING: deployed in the container, or EFFING_DIR on the host.
    from effing_checkout import ensure_effing_importable

    ensure_effing_importable()

    import asyncio
    import contextlib
    import importlib.util
    import io
    import sys
    import types
    from pathlib import Path
    from unittest.mock import patch

    def _load_maintenance_agent():
        path = Path(__file__).resolve().parent.parent / "scripts" / "maintenance_agent.py"
        spec = importlib.util.spec_from_file_location(
            "maintenance_agent_exit_summary", path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _source(name):
        return types.SimpleNamespace(name=name, metadata={}, path=name)

    def _completed():
        return {
            "source": "ok.pdf",
            "status": "completed",
            "completed_phases": {
                "parsing_chunking": {"state": "complete"},
                "embedding": {"state": "complete"},
            },
        }

    def _failed(reason):
        return {
            "source": "bad.pdf",
            "status": "error",
            "error": reason,
            "completed_phases": {
                "parsing_chunking": {"state": "complete"},
                "graph_llm_extraction": {"state": "error", "reason": reason},
            },
        }

    def _processing():
        return {
            "source": "slow.pdf",
            "status": "processing",
            "phase": "embedding",
            "completed_phases": {
                "parsing_chunking": {"state": "complete"},
                "embedding": {"state": "processing"},
            },
        }

    async def _run(maintenance_agent, docs, names, op="reingest"):
        async def fake_read(_agent_id, source_name):
            return docs.get(source_name)

        stdout = io.StringIO()
        with patch.object(
            maintenance_agent, "_read_ingestion_status", fake_read
        ), contextlib.redirect_stdout(stdout):
            ok = await maintenance_agent._report_ingestion_outcome(
                "agent1", op, [_source(name) for name in names]
            )
        return ok, stdout.getvalue()

    async def test_all_completed_is_ok(maintenance_agent):
        docs = {"a.pdf": _completed(), "b.pdf": _completed()}
        ok, out = await _run(maintenance_agent, docs, ["a.pdf", "b.pdf"])
        assert ok is True, out
        assert "op=reingest result=ok detail=reingested" in out, out
        print("PASS (a) all completed -> result=ok, returns True")

    async def test_error_reports_source_phase_reason(maintenance_agent):
        reason = "graph extraction timed out after 120s: connection reset"
        docs = {"bad.pdf": _failed(reason)}
        ok, out = await _run(maintenance_agent, docs, ["bad.pdf"])
        assert ok is False, out
        assert "result=fail" in out and "detail=phase-failed" in out, out
        assert "bad.pdf" in out, out
        assert "graph_llm_extraction" in out, out
        assert reason in out, out
        print("PASS (b) error -> result=fail detail=phase-failed with the reason")

    async def test_still_processing_is_fail(maintenance_agent):
        docs = {"slow.pdf": _processing()}
        ok, out = await _run(maintenance_agent, docs, ["slow.pdf"])
        assert ok is False, out
        assert "result=fail" in out and "detail=still-processing" in out, out
        assert "slow.pdf" in out, out
        print("PASS (c) still processing -> result=fail detail=still-processing")

    async def test_one_ok_one_error_aggregates_fail(maintenance_agent):
        docs = {"good.pdf": _completed(), "bad.pdf": _failed("boom")}
        ok, out = await _run(maintenance_agent, docs, ["good.pdf", "bad.pdf"])
        assert ok is False, out
        assert "result=fail" in out and "detail=phase-failed" in out, out
        assert "bad.pdf" in out and "boom" in out, out
        assert "good.pdf" not in out, out
        print("PASS (d) two sources, one ok one error -> aggregate fail, ok not failed")

    async def _main():
        maintenance_agent = _load_maintenance_agent()
        await test_all_completed_is_ok(maintenance_agent)
        await test_error_reports_source_phase_reason(maintenance_agent)
        await test_still_processing_is_fail(maintenance_agent)
        await test_one_ok_one_error_aggregates_fail(maintenance_agent)

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    print("OK maintenance exit summary reflects the real phase outcome")
