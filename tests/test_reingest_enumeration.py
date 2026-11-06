#!/usr/bin/env python3
"""Standalone, import-safe test: ``--reingest`` enumerates stored FILES too.

``_op_reingest`` used to enumerate sources from the VECTOR STORE only
(``ccat.get_stored_sources_with_metadata()``). A source whose points were
already wiped by a failed re-ingest is therefore invisible, and the recovery
path could never recover it — exactly the live bug where 1 of 9 stored files
was re-ingested. The enum must be the UNION of the vector-store sources and
``ccat.file_manager.list_files(agent_id)``, deduped by name, with derived
image artifacts (``<stem>_img_<index>_<digest>.<ext>``) skipped.

Run from the host repository (the plugin is importable by file path):

    cd ../grinning-cat-core
    EFFING_DIR=/path/to/cat-efficient-ingestion MULTIMODAL_DIR=/path/to/cat-multimodal-ingestion PYTHONPATH=. .venv/bin/python /path/to/cat-graphrag/tests/test_reingest_enumeration.py

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
        spec = importlib.util.spec_from_file_location("maintenance_agent_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _source(name):
        return types.SimpleNamespace(name=name, metadata={}, path=name)

    def _file(name):
        return types.SimpleNamespace(name=name, path=name, hash="h", size=1, last_modified="")

    class FakeFileManager:
        def __init__(self, names, unreadable=()):
            self.names = names
            self.unreadable = set(unreadable)
            self.listed_dirs = []
            self.read = []

        def list_files(self, remote_root_dir):
            self.listed_dirs.append(remote_root_dir)
            return [_file(name) for name in self.names]

        def read_file(self, remote_filename, remote_root_dir=None):
            self.read.append((remote_filename, remote_root_dir))
            if remote_filename in self.unreadable:
                return None
            return b"content of " + remote_filename.encode()

    class FakeHandler:
        def __init__(self):
            self.deleted = []

        async def delete_tenant_points(self, collection, metadata=None):
            self.deleted.append((metadata or {}).get("source"))

    class FakeCcat:
        agent_key = "agent1"
        _id = "agent1"

        def __init__(self, vector_sources, file_manager):
            self.vector_memory_handler = FakeHandler()
            self.file_manager = file_manager
            self._vector_sources = vector_sources

        async def get_stored_sources_with_metadata(self):
            from cat.services.memory.models import VectorMemoryType

            return {
                VectorMemoryType("declarative"): [
                    _source(name) for name in self._vector_sources
                ]
            }

    async def _run_reingest(ccat):
        from cat.plugins.cat_efficient_ingestion import registry as effing_registry
        from cat.plugins.cat_efficient_ingestion import reembed as effing_reembed

        maintenance_agent = _load_maintenance_agent()
        reembedded = []

        async def fake_preflight(_ccat):
            return None

        async def fake_marker(_agent_id):
            return None

        async def fake_clear(_agent_id, _scope, _source):
            return True

        async def fake_delete_status(_agent_id, _scope, _source):
            return None

        async def fake_purge(_agent_id, _scope, _source, delete_points):
            await delete_points()

        async def fake_reembed(_ccat, _collection, sources):
            reembedded.extend(source.name for source in sources)

        async def fake_outcome(_agent_id, _op, _sources):
            return True

        with patch.object(maintenance_agent, "_preflight_reembed_path", fake_preflight), \
             patch.object(maintenance_agent, "_set_maintenance_marker", fake_marker), \
             patch.object(maintenance_agent, "_report_ingestion_outcome", fake_outcome), \
             patch.object(effing_registry, "purge_source_for_reingest", fake_purge), \
             patch.object(effing_registry, "clear_completed_phases", fake_clear), \
             patch.object(effing_registry, "delete_status", fake_delete_status), \
             patch.object(effing_reembed, "reembed_sources", fake_reembed):
            ok = await maintenance_agent._op_reingest(ccat, None, "declarative")
        return ok, reembedded

    async def test_union_dedupes_and_skips_images():
        # Vector store holds one source (a ghost with no recoverable file) and
        # the file manager holds the three real files of the live bug, one of
        # which is a derived image artifact of a parent PDF.
        file_manager = FakeFileManager([
            "consigli.pdf",
            "criteri_correzioni.pdf",
            "consigli_img_0_deadbeef.jpg",
        ])
        ccat = FakeCcat(["ghost_source.pdf"], file_manager)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            ok, reembedded = await _run_reingest(ccat)

        assert ok is True, "reingest should succeed"
        assert file_manager.listed_dirs == ["agent1"], file_manager.listed_dirs
        assert sorted(reembedded) == [
            "consigli.pdf",
            "criteri_correzioni.pdf",
            "ghost_source.pdf",
        ], reembedded
        assert "detail=derived-image-skip source=consigli_img_0_deadbeef.jpg" in stdout.getvalue(), (
            stdout.getvalue()
        )
        assert file_manager.read == [
            ("consigli.pdf", "agent1"),
            ("criteri_correzioni.pdf", "agent1"),
        ], file_manager.read
        print("PASS reingest unions stored files, skips derived images")

    async def test_union_dedupes_by_name():
        # The vector store already knows a.pdf: it must not be listed twice.
        file_manager = FakeFileManager(["a.pdf", "b.pdf", "c.pdf"])
        ccat = FakeCcat(["a.pdf"], file_manager)

        with contextlib.redirect_stdout(io.StringIO()):
            ok, reembedded = await _run_reingest(ccat)

        assert ok is True
        assert reembedded == ["a.pdf", "b.pdf", "c.pdf"], reembedded
        assert file_manager.read == [("b.pdf", "agent1"), ("c.pdf", "agent1")], (
            file_manager.read
        )
        print("PASS reingest dedupes the union by name")

    async def _main():
        await test_union_dedupes_and_skips_images()
        await test_union_dedupes_by_name()

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    print("OK reingest enumeration")
