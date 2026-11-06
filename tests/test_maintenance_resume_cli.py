#!/usr/bin/env python3
"""Standalone, import-safe harness for ``--resume-failed``.

The operation resumes ONLY the diary phases whose explicit state is ``error``:
it keeps every ``complete`` entry (with its ``tag`` / ``started_at``), writes
``status=processing`` / ``error=None`` and re-dispatches ONLY the cleared
phases through ``reembed_sources(..., stale_after=0.0)``. No purge, no point
wipe, no re-parse and no re-embed of a healthy phase.

Covered:
  (a) dry-run resolves the plan and exits 2 with ZERO writes;
  (b) --yes deletes only the ``error`` entries and keeps the complete ones
      with their ``tag`` / ``started_at`` byte-for-byte unchanged;
  (c) a source with no failed phase is ``nothing-to-resume`` and untouched;
  (d) per-source failure isolation: one failing source never aborts the
      others and the process exits 1 naming the failing source;
  (e) a source whose LLM phase is disabled ends ``skipped``, never ``error``;
  (f) ``--all`` vs ``--agent`` scoping (protect system/``__*__`` unless
      explicitly named) and repeatable ``--agent``.

Run it inside the Cat container (EffING deployed):

    docker exec <cat-container> python \\
        /app/cat/plugins/cat_graphrag/tests/test_maintenance_resume_cli.py

or from the host, pointing EFFING_DIR at a checkout of cat-efficient-ingestion:

    cd ../grinning-cat-core
    EFFING_DIR=/path/to/cat-efficient-ingestion MULTIMODAL_DIR=/path/to/cat-multimodal-ingestion PYTHONPATH=. \\
        .venv/bin/python /path/to/cat-graphrag/tests/test_maintenance_resume_cli.py

IMPORT-SAFETY: this file lives in the plugin folder, which the Cat plugin
loader imports recursively at activation. The ENTIRE body lives under
``if __name__ == "__main__":`` so importing the file executes zero code.
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
            "maintenance_agent_resume_under_test", path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _complete(tag, started_at):
        return {
            "state": "complete",
            "tag": tag,
            "started_at": started_at,
            "settings_version": None,
            "deps": {},
        }

    def _error(reason, tag="err-tag", started_at=900):
        return {"state": "error", "reason": reason, "tag": tag, "started_at": started_at}

    def _source(name):
        return types.SimpleNamespace(
            name=name, metadata={}, path=name, content=None
        )

    def _doc(name, status, diary, error=None):
        return {
            "source": name,
            "scope": "agent",
            "type": "file",
            "status": status,
            "error": error,
            "completed_phases": dict(diary),
        }

    class Store:
        """In-memory status docs + the writes the op performs."""

        def __init__(self, docs):
            self.docs = {name: dict(doc) for name, doc in docs.items()}
            self.reset_calls = []
            self.set_calls = []
            self.reembed_calls = []

        async def list_statuses(self, agent_id, chat_id=None):
            return [dict(doc) for doc in self.docs.values()]

        async def get_status(self, agent_id, scope, source):
            doc = self.docs.get(source)
            return dict(doc) if doc is not None else None

        async def reset_phase_entries(self, agent_id, scope, source, phases):
            self.reset_calls.append((source, sorted(phases)))
            diary = self.docs[source]["completed_phases"]
            for phase in phases:
                diary.pop(phase, None)
            return self.docs[source]

        async def set_status(
            self, agent_id, scope, source, *, type_, status, chat_id=None,
            error=None, **kwargs
        ):
            self.set_calls.append(
                (source, getattr(status, "value", status), error)
            )
            self.docs[source]["status"] = getattr(status, "value", status)
            self.docs[source]["error"] = error
            return self.docs[source]

    def _patch_op(store, behaviors):
        import cat.plugins.cat_efficient_ingestion.reembed as effing_reembed
        import cat.plugins.cat_efficient_ingestion.registry as effing_registry
        import cat.plugins.cat_efficient_ingestion.resume as effing_resume

        async def fake_reembed(ccat, collection, sources, stale_after=None):
            store.reembed_calls.append(
                (
                    sources[0].name,
                    str(collection),
                    stale_after,
                )
            )
            behavior = behaviors.get(sources[0].name)
            if behavior is not None:
                await behavior(store.docs[sources[0].name])

        def fake_source_from_entry(ccat, agent_id, scope, source, type_="file"):
            return _source(source)

        return (
            patch.object(effing_registry, "list_statuses", store.list_statuses),
            patch.object(effing_registry, "get_status", store.get_status),
            patch.object(
                effing_registry, "reset_phase_entries", store.reset_phase_entries
            ),
            patch.object(effing_registry, "set_status", store.set_status),
            patch.object(effing_reembed, "reembed_sources", fake_reembed),
            patch.object(effing_resume, "_source_from_entry", fake_source_from_entry),
        )

    def _enter(patches):
        for p in patches:
            p.start()
        return patches

    def _exit(patches):
        for p in reversed(patches):
            p.stop()

    async def test_dry_run_exits_two_with_zero_writes(maintenance_agent):
        store = Store(
            {
                "broken.pdf": _doc(
                    "broken.pdf",
                    "error",
                    {
                        "parsing_chunking": _complete("p", 1),
                        "embedding": _complete("e", 2),
                        "graph_llm_extraction": _error("boom"),
                    },
                    error="boom",
                )
            }
        )

        async def fake_list_agents(include_protected=False):
            return ["a1"]

        async def fake_resolve(agent_id):
            return [
                {
                    "source": "broken.pdf",
                    "scope": "agent",
                    "type": "file",
                    "failed_phases": ["graph_llm_extraction"],
                    "kept_phases": ["parsing_chunking", "embedding"],
                }
            ]

        async def forbidden_bootstrap(agent_id):
            raise AssertionError("dry-run must not bootstrap the agent")

        patches = list(_patch_op(store, {}))
        patches += [
            patch.object(maintenance_agent, "_runtime_guard", lambda: True),
            patch.object(maintenance_agent, "_list_agents", fake_list_agents),
            patch.object(maintenance_agent, "_resolve_resume_failed_plan", fake_resolve),
            patch.object(maintenance_agent, "_bootstrap_agent", forbidden_bootstrap),
        ]
        _enter(patches)
        stdout = io.StringIO()
        argv = ["maintenance_agent", "--agent", "a1", "--resume-failed", "--dry-run"]
        try:
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
                try:
                    await maintenance_agent.main()
                except SystemExit as exc:
                    code = exc.code
                else:
                    code = None
        finally:
            _exit(patches)

        out = stdout.getvalue()
        assert code == 2, f"dry-run must exit 2, got {code}: {out}"
        assert store.reset_calls == [], store.reset_calls
        assert store.set_calls == [], store.set_calls
        assert store.reembed_calls == [], store.reembed_calls
        assert "failed_phases=graph_llm_extraction" in out, out
        assert "kept_phases=2" in out, out
        print("PASS (a) dry-run resolves the plan, exits 2, zero writes")

    async def test_yes_removes_only_error_and_keeps_complete(maintenance_agent):
        diary = {
            "parsing_chunking": _complete("p-tag", 111),
            "embedding": _complete("e-tag", 222),
            "graph_llm_extraction": _error("llm exploded", "l-tag", 333),
            "catalog_cards": _error(
                "dependency graph_llm_extraction is error", "c-tag", 444
            ),
        }
        store = Store(
            {
                "doc.pdf": _doc("doc.pdf", "error", diary, error="llm exploded"),
            }
        )
        kept_before = {
            phase: dict(store.docs["doc.pdf"]["completed_phases"][phase])
            for phase in ("parsing_chunking", "embedding")
        }

        async def finish(doc):
            doc["completed_phases"]["graph_llm_extraction"] = _complete("l2", 555)
            doc["completed_phases"]["catalog_cards"] = _complete("c2", 666)
            doc["status"] = "completed"
            doc["error"] = None

        patches = list(_patch_op(store, {"doc.pdf": finish}))
        _enter(patches)
        try:
            ok = await maintenance_agent._op_resume_failed(
                types.SimpleNamespace(agent_key="a1", _id="a1"), None, "declarative"
            )
        finally:
            _exit(patches)

        assert ok is True, ok
        assert store.reset_calls == [
            ("doc.pdf", ["catalog_cards", "graph_llm_extraction"])
        ], store.reset_calls
        assert ("doc.pdf", "processing", None) in store.set_calls, store.set_calls
        after = store.docs["doc.pdf"]["completed_phases"]
        for phase, expected in kept_before.items():
            assert after[phase] == expected, (phase, after[phase], expected)
        assert store.reembed_calls == [
            ("doc.pdf", "declarative", 0.0)
        ], store.reembed_calls
        print("PASS (b) --yes deletes only error entries, complete entries unchanged")

    async def test_nothing_to_resume_is_untouched(maintenance_agent):
        store = Store(
            {
                "ok.pdf": _doc(
                    "ok.pdf",
                    "error",
                    {
                        "parsing_chunking": _complete("p", 1),
                        "embedding": _complete("e", 2),
                    },
                    error="listed as error but no phase failed",
                )
            }
        )
        patches = list(_patch_op(store, {}))
        _enter(patches)
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                ok = await maintenance_agent._op_resume_failed(
                    types.SimpleNamespace(agent_key="a1", _id="a1"),
                    None,
                    "declarative",
                )
        finally:
            _exit(patches)

        out = stdout.getvalue()
        assert ok is True, out
        assert store.reset_calls == [] and store.set_calls == [], (
            store.reset_calls,
            store.set_calls,
        )
        assert store.reembed_calls == [], store.reembed_calls
        assert "detail=nothing-to-resume" in out, out
        assert store.docs["ok.pdf"]["status"] == "error", store.docs["ok.pdf"]
        print("PASS (c) no failed phase -> nothing-to-resume, untouched")

    async def test_per_source_failure_isolation(maintenance_agent):
        store = Store(
            {
                "good.pdf": _doc(
                    "good.pdf",
                    "error",
                    {
                        "parsing_chunking": _complete("p", 1),
                        "graph_llm_extraction": _error("boom"),
                    },
                ),
                "bad.pdf": _doc(
                    "bad.pdf",
                    "error",
                    {
                        "parsing_chunking": _complete("p", 1),
                        "embedding": _error("embed down"),
                    },
                ),
            }
        )

        async def finish_good(doc):
            doc["completed_phases"]["graph_llm_extraction"] = _complete("l2", 2)
            doc["status"] = "completed"

        async def explode_bad(doc):
            raise RuntimeError("reembed backend unavailable")

        patches = list(
            _patch_op(store, {"good.pdf": finish_good, "bad.pdf": explode_bad})
        )
        _enter(patches)
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                ok = await maintenance_agent._op_resume_failed(
                    types.SimpleNamespace(agent_key="a1", _id="a1"),
                    None,
                    "declarative",
                )
        finally:
            _exit(patches)

        out = stdout.getvalue()
        assert ok is False, out
        assert "source=good.pdf result=ok" in out, out
        assert "source=bad.pdf result=fail" in out, out
        assert "bad.pdf" in out and "result=fail" in out, out
        assert store.reembed_calls[0][0] == "good.pdf"
        assert store.reembed_calls[1][0] == "bad.pdf"
        print("PASS (d) one failing source is isolated and named")

    async def test_main_exits_one_on_partial_failure(maintenance_agent):
        store = Store(
            {
                "good.pdf": _doc(
                    "good.pdf",
                    "error",
                    {"graph_llm_extraction": _error("boom")},
                ),
                "bad.pdf": _doc(
                    "bad.pdf",
                    "error",
                    {"graph_llm_extraction": _error("boom")},
                ),
            }
        )

        async def finish_good(doc):
            doc["completed_phases"]["graph_llm_extraction"] = _complete("l2", 2)
            doc["status"] = "completed"

        async def explode_bad(doc):
            raise RuntimeError("boom")

        async def fake_list_agents(include_protected=False):
            return ["a1"]

        async def fake_bootstrap(agent_id):
            return (
                types.SimpleNamespace(agent_key=agent_id, _id=agent_id, agent_id=agent_id),
                None,
                None,
            )

        async def fake_engine(ccat, op="reembed"):
            return "EfficientIngestionConfiguration"

        async def fake_marker(agent_id):
            return None

        async def fake_clear(agent_id):
            return None

        patches = list(
            _patch_op(store, {"good.pdf": finish_good, "bad.pdf": explode_bad})
        )
        patches += [
            patch.object(maintenance_agent, "_runtime_guard", lambda: True),
            patch.object(maintenance_agent, "_list_agents", fake_list_agents),
            patch.object(maintenance_agent, "_bootstrap_agent", fake_bootstrap),
            patch.object(
                maintenance_agent, "_resolve_ingestion_engine", fake_engine
            ),
            patch.object(maintenance_agent, "_set_maintenance_marker", fake_marker),
            patch.object(maintenance_agent, "_clear_maintenance_marker", fake_clear),
        ]
        _enter(patches)
        argv = ["maintenance_agent", "--agent", "a1", "--resume-failed", "--yes"]
        try:
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                try:
                    await maintenance_agent.main()
                except SystemExit as exc:
                    code = exc.code
                else:
                    code = None
        finally:
            _exit(patches)

        assert code == 1, f"partial failure must exit 1, got {code}"
        print("PASS (d) main exits 1 on a partial per-source failure")

    async def test_disabled_llm_phase_ends_skipped(maintenance_agent):
        store = Store(
            {
                "doc.pdf": _doc(
                    "doc.pdf",
                    "error",
                    {
                        "parsing_chunking": _complete("p", 1),
                        "embedding": _complete("e", 2),
                        "graph_llm_extraction": _error(
                            "concept-relations extraction disabled"
                        ),
                    },
                )
            }
        )

        async def finish_skipped(doc):
            doc["completed_phases"]["graph_llm_extraction"] = {
                "state": "skipped",
                "reason": "concept-relations extraction disabled",
            }
            doc["status"] = "completed"
            doc["error"] = None

        patches = list(_patch_op(store, {"doc.pdf": finish_skipped}))
        _enter(patches)
        try:
            ok = await maintenance_agent._op_resume_failed(
                types.SimpleNamespace(agent_key="a1", _id="a1"), None, "declarative"
            )
        finally:
            _exit(patches)

        assert ok is True, store.docs["doc.pdf"]
        entry = store.docs["doc.pdf"]["completed_phases"]["graph_llm_extraction"]
        assert entry["state"] == "skipped", entry
        assert store.docs["doc.pdf"]["status"] == "completed", store.docs["doc.pdf"]
        print("PASS (e) disabled LLM phase ends skipped, never error")

    async def test_all_vs_agent_scoping(maintenance_agent):
        async def fake_list_agents(include_protected=False):
            if include_protected:
                return ["a1", "a2", "system", "__hidden__"]
            return ["a1", "a2"]

        async def fake_resolve(agent_id):
            return []

        patches = [
            patch.object(maintenance_agent, "_list_agents", fake_list_agents),
            patch.object(maintenance_agent, "_resolve_resume_failed_plan", fake_resolve),
        ]
        saved_argv = sys.argv
        _enter(patches)
        try:
            all_args = types.SimpleNamespace(
                agent=None,
                all=True,
                ops=["resume-failed"],
                collection="declarative",
            )
            all_plan = await maintenance_agent._build_ops_plan(all_args)
            one = types.SimpleNamespace(
                agent=["a1"],
                all=False,
                ops=["resume-failed"],
                collection="declarative",
            )
            one_plan = await maintenance_agent._build_ops_plan(one)

            sys.argv = [
                "maintenance_agent", "--agent", "system",
                "--resume-failed", "--dry-run",
            ]
            explicit = maintenance_agent._parse_args()
            assert explicit.agent == ["system"], explicit.agent

            sys.argv = [
                "maintenance_agent",
                "--agent", "a1",
                "--agent", "a2",
                "--resume-failed", "--dry-run",
            ]
            multi = maintenance_agent._parse_args()
            assert multi.agent == ["a1", "a2"], multi.agent
        finally:
            sys.argv = saved_argv
            _exit(patches)

        assert [entry["agent_id"] for entry in all_plan] == ["a1", "a2"], all_plan
        assert [entry["agent_id"] for entry in one_plan] == ["a1"], one_plan
        print("PASS (f) --all skips protected agents, --agent is repeatable/explicit")

    async def _main():
        maintenance_agent = _load_maintenance_agent()
        await test_dry_run_exits_two_with_zero_writes(maintenance_agent)
        await test_yes_removes_only_error_and_keeps_complete(maintenance_agent)
        await test_nothing_to_resume_is_untouched(maintenance_agent)
        await test_per_source_failure_isolation(maintenance_agent)
        await test_main_exits_one_on_partial_failure(maintenance_agent)
        await test_disabled_llm_phase_ends_skipped(maintenance_agent)
        await test_all_vs_agent_scoping(maintenance_agent)

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    print("OK maintenance resume-failed CLI")
