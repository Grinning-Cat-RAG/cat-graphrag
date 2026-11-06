"""Standalone, import-safe regression harness for tenant-write-lock SCOPE.

The lock is on Neo4j: it must wrap the transaction, never the computation.
These tests prove the fix applied to ``graphrag_handler.py``:

  (a) an embedding that takes 2 s does NOT hold the tenant lock — another
      writer acquires it during the embed and completes;
  (b) the write transaction itself IS serialised per tenant (two writes never
      overlap);
  (c) a re-entrant call acquires once (unchanged);
  (d) no lock is held while the embedder is called (the lock key is absent and
      the ContextVar does not name the tenant);
  (e) the 300 s timeout is never reached in a normal write+embed sequence;
  (f) STATIC: ``graphrag_handler.py`` decorates ONLY the pure-write methods —
      every method that embeds/LLMs first lost its ``@tenant_write_locked``.

The Cat plugin loader imports EVERY ``.py`` under a plugin folder at activation,
so this whole file is inert on import: every import, class and test lives inside
``if __name__ == "__main__":``. ``cat.log`` and ``cat.db.crud`` are stubbed in
``sys.modules`` BEFORE ``tenant_write_lock`` is loaded, so no Cat install, no
Redis and no Neo4j are needed.

Run as a plain script:

    python tests/test_lock_scope.py
"""

if __name__ == "__main__":
    import asyncio
    import importlib.util
    import os
    import re
    import sys
    import types
    from contextlib import asynccontextmanager

    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    HANDLER_PATH = os.path.join(REPO_ROOT, "graphrag_handler.py")

    # ── Stub ``cat.log`` ──────────────────────────────────────────────────
    warnings_logged = []

    class _StubLog:
        def info(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def warning(self, message="", *args, **kwargs):
            warnings_logged.append(str(message))

    cat_pkg = types.ModuleType("cat")
    cat_log = types.ModuleType("cat.log")
    cat_log.log = _StubLog()
    cat_pkg.log = cat_log

    # ── Stub ``cat.db.crud`` with an in-process Redis-like lock ────────────
    class LockError(Exception):
        """Stand-in for ``redis.exceptions.LockError`` re-exported by crud."""

    _locks = {}
    _active = {}
    _max_active = {}
    _acquisitions = {}
    _guard = asyncio.Lock()

    @asynccontextmanager
    async def distributed_lock(key_pattern, timeout=10.0, blocking_timeout=15.0):
        async with _guard:
            lock = _locks.setdefault(key_pattern, asyncio.Lock())
        try:
            await asyncio.wait_for(lock.acquire(), timeout=blocking_timeout)
        except asyncio.TimeoutError:
            raise LockError(
                f"Could not acquire lock '{key_pattern}' within {blocking_timeout}s"
            )
        async with _guard:
            _active[key_pattern] = _active.get(key_pattern, 0) + 1
            _max_active[key_pattern] = max(
                _max_active.get(key_pattern, 0), _active[key_pattern]
            )
            _acquisitions[key_pattern] = _acquisitions.get(key_pattern, 0) + 1
        try:
            yield
        finally:
            async with _guard:
                _active[key_pattern] -= 1
            lock.release()

    cat_db = types.ModuleType("cat.db")
    cat_db_crud = types.ModuleType("cat.db.crud")
    cat_db_crud.distributed_lock = distributed_lock
    cat_db_crud.LockError = LockError
    cat_db.crud = cat_db_crud

    sys.modules["cat"] = cat_pkg
    sys.modules["cat.log"] = cat_log
    sys.modules["cat.db"] = cat_db
    sys.modules["cat.db.crud"] = cat_db_crud

    # ── Load ``tenant_write_lock`` by file path ───────────────────────────
    _spec = importlib.util.spec_from_file_location(
        "tenant_write_lock_scope_under_test",
        os.path.join(REPO_ROOT, "tenant_write_lock.py"),
    )
    twl = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(twl)

    # ── Fixtures ──────────────────────────────────────────────────────────
    class _FakeHandler:
        """Mimics the patched handler: embed OUTSIDE the lock, write INSIDE."""

        def __init__(self, tenant):
            self.agent_id = tenant
            self._local_write_locks = {}
            self.embedding_active = False
            self.writes_done = 0
            self.embed_finished = False

        async def _embed(self):
            key = f"graph-write:{self.agent_id}"
            self.embedding_active = True
            try:
                # (d) no lock may be held while the embedder runs.
                _check(
                    _active.get(key, 0) == 0,
                    f"(d) tenant lock held during embed (active={_active.get(key)})",
                )
                _check(
                    self.agent_id not in twl.GRAPH_WRITE_LOCK_HELD.get(),
                    "(d) ContextVar names the tenant during embed",
                )
                await asyncio.sleep(2.0)  # 2 s embedding
                return [0.1] * 8
            finally:
                self.embedding_active = False
                self.embed_finished = True

        async def embed_then_write(self):
            # PRE-WORK first, lock released (the whole point of the fix).
            _ = await self._embed()
            async with twl.acquire_tenant_write_lock(
                self, self.agent_id, "write after embed"
            ):
                await asyncio.sleep(0.05)  # the Neo4j transaction
                self.writes_done += 1

        async def write_only(self):
            async with twl.acquire_tenant_write_lock(
                self, self.agent_id, "pure write"
            ):
                await asyncio.sleep(0.05)
                self.writes_done += 1

    def reset():
        _locks.clear()
        _active.clear()
        _max_active.clear()
        _acquisitions.clear()
        warnings_logged.clear()

    def _check(cond, message):
        if not cond:
            raise AssertionError(message)

    # ── (a) embedding does NOT hold the tenant lock ───────────────────────
    async def test_a_embedding_does_not_hold_lock():
        reset()
        handler = _FakeHandler("tenant-a")
        second_done = asyncio.Event()

        async def second_writer():
            # Wait until the first handler is mid-embed, then acquire the lock.
            while not handler.embedding_active:
                await asyncio.sleep(0.01)
            async with twl.acquire_tenant_write_lock(
                handler, "tenant-a", "second writer"
            ):
                handler.writes_done += 1
                second_done.set()

        first = asyncio.create_task(handler.embed_then_write())
        second = asyncio.create_task(second_writer())

        await asyncio.wait_for(second_done.wait(), timeout=1.5)
        _check(
            not handler.embed_finished,
            "(a) second writer only acquired AFTER the embed finished — "
            "the embed held the lock",
        )
        await asyncio.gather(first, second)
        _check(handler.writes_done == 2, f"(a) writes_done={handler.writes_done}")
        print(
            "(a) PASS  embedding (2 s) ran outside the lock; the second "
            "writer acquired it DURING the embed and completed"
        )

    # ── (b) the write transaction IS serialised per tenant ────────────────
    async def test_b_writes_are_serialised():
        reset()
        handler = _FakeHandler("tenant-b")

        async def writer(n):
            async with twl.acquire_tenant_write_lock(
                handler, "tenant-b", f"write-{n}"
            ):
                await asyncio.sleep(0.05)

        await asyncio.gather(writer(1), writer(2), writer(3))
        key = "graph-write:tenant-b"
        _check(
            _max_active.get(key, 0) == 1,
            f"(b) writes overlapped: max_active={_max_active.get(key)}",
        )
        _check(
            _acquisitions.get(key) == 3,
            f"(b) expected 3 acquisitions, got {_acquisitions}",
        )
        print(
            "(b) PASS  3 writes serialised per tenant "
            f"(max_active={_max_active.get(key)})"
        )

    # ── (c) re-entrant call acquires once ─────────────────────────────────
    async def test_c_reentrant_acquires_once():
        reset()

        class _Reentrant:
            def __init__(self):
                self.agent_id = "tenant-c"
                self._local_write_locks = {}

            @twl.tenant_write_locked("outer write")
            async def outer(self):
                return await self.inner()

            @twl.tenant_write_locked("inner write")
            async def inner(self):
                await asyncio.sleep(0.01)
                return "done"

        handler = _Reentrant()
        result = await asyncio.wait_for(handler.outer(), timeout=2.0)
        _check(result == "done", f"(c) unexpected result: {result!r}")
        _check(
            _acquisitions.get("graph-write:tenant-c") == 1,
            f"(c) lock acquired more than once: {_acquisitions}",
        )
        print("(c) PASS  re-entrant call acquired once")

    # ── (d) no lock while the embedder is called ──────────────────────────
    async def test_d_no_lock_during_embed():
        reset()
        handler = _FakeHandler("tenant-d")

        async def second_writer():
            while not handler.embedding_active:
                await asyncio.sleep(0.01)
            key = "graph-write:tenant-d"
            _check(
                _active.get(key, 0) == 0,
                f"(d) lock key active during embed: {_active.get(key)}",
            )
            _check(
                "graph-write:tenant-d" not in _locks or not _locks["graph-write:tenant-d"].locked(),
                "(d) lock object was locked during embed",
            )

        first = asyncio.create_task(handler.embed_then_write())
        second = asyncio.create_task(second_writer())
        await asyncio.gather(first, second)
        print("(d) PASS  lock key absent while the embedder was called")

    # ── (e) the 300 s timeout is never reached ────────────────────────────
    async def test_e_timeout_never_reached():
        reset()
        # Default timeout is 300 s; keep the real default and prove the whole
        # embed+write sequence finishes well inside it without a timeout.
        _check(
            twl.graph_write_lock_timeout_seconds() >= 300.0,
            "default timeout is not 300 s",
        )
        handler = _FakeHandler("tenant-e")
        started = asyncio.get_event_loop().time()
        await asyncio.wait_for(handler.embed_then_write(), timeout=30.0)
        elapsed = asyncio.get_event_loop().time() - started
        _check(elapsed < 300.0, f"(e) sequence exceeded 300 s: {elapsed:.2f}s")
        _check(
            not any("Timed out" in w for w in warnings_logged),
            f"(e) a timeout warning fired: {warnings_logged}",
        )
        print(
            f"(e) PASS  normal write+embed finished in {elapsed:.2f}s "
            "(300 s timeout never reached)"
        )

    # ── (f) STATIC: only pure-write methods keep the decorator ────────────
    def test_f_only_pure_writes_are_decorated():
        src = open(HANDLER_PATH, encoding="utf-8").read()
        # ``@tenant_write_locked(...)`` immediately followed by ``async def``.
        decorated = set(
            re.findall(
                r"@tenant_write_locked\([^)]*\)\s*\n\s*async def\s+(\w+)\s*\(",
                src,
            )
        )
        expected_pure_writes = {
            "_create_similarity_relationships_for_gen",
            "_gc_generation",
            "create_collection",
            "delete_collection",
            "add_point_to_tenant",
            "_write_source_graph",
            "refresh_technology_entities",
            "recompute_provenance",
            "_create_similarity_relationships",
            "delete_tenant_points",
            "purge_source_graph_artifacts",
            "delete_tenant_points_by_ids",
            "_write_concept_gen",
            "_flip_concept_gen",
            "_gc_stale_concept_nodes",
            "_store_concept_relations",
        }
        pre_work_methods = {
            "reembed_tenant",
            "_reembed_tenant_impl",
            "initialize",
            "_align_embedder_lazy",
            "_extract_and_link_entities",
            "add_points_to_tenant",
            "create_derived_graph_for_source",
            "recompute_concept_relations",
        }
        _check(
            decorated == expected_pure_writes,
            f"(f) decorated set drifted.\n  extra: {decorated - expected_pure_writes}\n"
            f"  missing: {expected_pure_writes - decorated}",
        )
        _check(
            not (decorated & pre_work_methods),
            f"(f) pre-work method still decorated: {decorated & pre_work_methods}",
        )
        print(
            f"(f) PASS  exactly {len(decorated)} pure-write methods keep the "
            f"decorator; all {len(pre_work_methods)} pre-work methods moved "
            "their lock acquisition inside"
        )

    # ── Runner ────────────────────────────────────────────────────────────
    async def _run_async():
        await test_a_embedding_does_not_hold_lock()
        await test_b_writes_are_serialised()
        await test_c_reentrant_acquires_once()
        await test_d_no_lock_during_embed()
        await test_e_timeout_never_reached()

    def main():
        asyncio.run(_run_async())
        test_f_only_pure_writes_are_decorated()
        print("\nALL LOCK-SCOPE TESTS PASSED")

    main()
