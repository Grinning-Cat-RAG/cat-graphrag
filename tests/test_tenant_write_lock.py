"""Standalone, import-safe harness for the tenant-scoped Neo4j write lock.

Verifies ``tenant_write_lock`` (used by ``graphrag_handler`` to serialise ALL
Neo4j writes for one tenant behind ``cat.db.crud.distributed_lock``):

  (a) two concurrent writers on the SAME tenant NEVER overlap;
  (b) two writers on DIFFERENT tenants DO run concurrently (no global lock);
  (c) a re-entrant call inside a locked write completes WITHOUT deadlocking and
      acquires the lock only ONCE;
  (d) a lock timeout logs a WARNING and RAISES; the write does NOT proceed;
  (e) the delete path serialises against a concurrent write for the SAME tenant.

The Cat plugin loader imports EVERY ``.py`` under a plugin folder at activation,
so this whole file is inert on import: every import, class and test lives inside
``if __name__ == "__main__":``. ``cat.log`` and ``cat.db.crud`` are stubbed in
``sys.modules`` BEFORE ``tenant_write_lock`` is loaded, so no Cat install, no
Redis and no Neo4j are needed.

Run as a plain script:

    python tests/test_tenant_write_lock.py
"""

if __name__ == "__main__":
    import asyncio
    import importlib.util
    import os
    import sys
    import types
    from contextlib import asynccontextmanager

    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    # Mirrors ``cat.db.crud.distributed_lock``: an async context manager keyed
    # by a pattern, with a blocking-acquisition timeout that raises LockError.
    class LockError(Exception):
        """Stand-in for ``redis.exceptions.LockError`` re-exported by crud."""

    _locks = {}
    _active = {}
    _max_active = {}
    _acquisitions = {}
    _total_active = 0
    _max_total_active = 0
    _events = []
    _guard = asyncio.Lock()

    @asynccontextmanager
    async def distributed_lock(key_pattern, timeout=10.0, blocking_timeout=15.0):
        global _total_active, _max_total_active
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
            _total_active += 1
            _max_total_active = max(_max_total_active, _total_active)
            _events.append(("acquire", key_pattern))
        try:
            yield
        finally:
            async with _guard:
                _active[key_pattern] -= 1
                _total_active -= 1
                _events.append(("release", key_pattern))
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

    # ── Load ``tenant_write_lock`` by file path (absolute imports work) ────
    _spec = importlib.util.spec_from_file_location(
        "tenant_write_lock_under_test",
        os.path.join(REPO_ROOT, "tenant_write_lock.py"),
    )
    twl = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(twl)

    # ── Test fixtures ─────────────────────────────────────────────────────
    class _Handler:
        """Minimal handler stand-in: the lock only needs ``agent_id``."""

        def __init__(self, tenant):
            self.agent_id = tenant
            self._local_write_locks = {}

    def reset():
        global _total_active, _max_total_active
        _locks.clear()
        _active.clear()
        _max_active.clear()
        _acquisitions.clear()
        _events.clear()
        _total_active = 0
        _max_total_active = 0
        warnings_logged.clear()

    def _check(cond, message):
        if not cond:
            raise AssertionError(message)

    # ── (a) same tenant: writers never overlap ────────────────────────────
    async def test_a_same_tenant_never_overlaps():
        reset()
        handler = _Handler("tenant-a")
        order = []

        async def writer(n):
            async with twl.acquire_tenant_write_lock(
                handler, "tenant-a", f"write-{n}"
            ):
                order.append(f"enter-{n}")
                await asyncio.sleep(0.05)
                order.append(f"exit-{n}")

        await asyncio.gather(writer(1), writer(2))

        _check(
            _max_active.get("graph-write:tenant-a", 0) == 1,
            f"(a) writers overlapped: max_active={_max_active.get('graph-write:tenant-a')}",
        )
        # Strict enter/exit pairing in a single serial order (no interleaving).
        _check(
            order[0].startswith("enter") and order[1].startswith("exit"),
            f"(a) interleaved enter/exit: {order}",
        )
        _check(
            order[2].startswith("enter") and order[3].startswith("exit"),
            f"(a) interleaved enter/exit: {order}",
        )
        _check(
            _acquisitions.get("graph-write:tenant-a") == 2,
            f"(a) expected 2 acquisitions, got {_acquisitions}",
        )
        print(f"(a) PASS  same tenant serialised, order={order}")

    # ── (b) different tenants: writers run concurrently ───────────────────
    async def test_b_different_tenants_run_concurrently():
        reset()
        handler = _Handler("tenant-a")

        async def writer(tenant):
            async with twl.acquire_tenant_write_lock(
                handler, tenant, f"write-{tenant}"
            ):
                await asyncio.sleep(0.1)

        await asyncio.gather(writer("tenant-a"), writer("tenant-b"))

        _check(
            _max_total_active >= 2,
            f"(b) writers did NOT run concurrently: max_total_active={_max_total_active}",
        )
        _check(
            _max_active.get("graph-write:tenant-a", 0) == 1,
            f"(b) same-key overlap for tenant-a: {_max_active}",
        )
        _check(
            _max_active.get("graph-write:tenant-b", 0) == 1,
            f"(b) same-key overlap for tenant-b: {_max_active}",
        )
        print(
            f"(b) PASS  different tenants ran concurrently "
            f"(max_total_active={_max_total_active})"
        )

    # ── (c) re-entrant call: completes, acquires once ─────────────────────
    async def test_c_reentrant_call_acquires_once():
        reset()

        class _Reentrant:
            def __init__(self):
                self.agent_id = "tenant-r"
                self._local_write_locks = {}
                self.calls = 0
                self.inner_ran = False

            @twl.tenant_write_locked("outer write")
            async def outer(self):
                self.calls += 1
                await self.inner()
                return "outer-done"

            @twl.tenant_write_locked("inner write")
            async def inner(self):
                self.calls += 1
                self.inner_ran = True
                await asyncio.sleep(0.01)
                return "inner-done"

        handler = _Reentrant()
        result = await asyncio.wait_for(handler.outer(), timeout=2.0)

        _check(result == "outer-done", f"(c) unexpected result: {result!r}")
        _check(handler.calls == 2, f"(c) expected 2 calls, got {handler.calls}")
        _check(handler.inner_ran, "(c) inner write did not run")
        _check(
            _acquisitions.get("graph-write:tenant-r") == 1,
            f"(c) lock acquired more than once: {_acquisitions}",
        )
        print(
            "(c) PASS  re-entrant call completed without deadlock "
            f"(acquisitions={_acquisitions.get('graph-write:tenant-r')})"
        )

    # ── (d) timeout: logs WARNING, raises, write does not proceed ─────────
    async def test_d_timeout_raises_and_blocks_write():
        reset()
        os.environ["CAT_GRAPH_WRITE_LOCK_TIMEOUT_SECONDS"] = "0.2"
        try:
            handler = _Handler("tenant-t")
            body_ran = []
            holder_ready = asyncio.Event()

            async def holder():
                async with twl.acquire_tenant_write_lock(
                    handler, "tenant-t", "holder write"
                ):
                    holder_ready.set()
                    await asyncio.sleep(0.6)

            async def contender():
                await holder_ready.wait()
                try:
                    async with twl.acquire_tenant_write_lock(
                        handler, "tenant-t", "contender write"
                    ):
                        body_ran.append(True)
                except twl.TenantWriteLockTimeout:
                    body_ran.append("timeout")

            holder_task = asyncio.create_task(holder())
            contender_task = asyncio.create_task(contender())
            await asyncio.gather(holder_task, contender_task)

            _check(
                "timeout" in body_ran and True not in body_ran,
                f"(d) write proceeded after timeout: {body_ran}",
            )
            _check(
                any("Timed out" in w and "tenant-t" in w for w in warnings_logged),
                f"(d) missing timeout WARNING naming tenant: {warnings_logged}",
            )
            _check(
                any("contender write" in w for w in warnings_logged),
                f"(d) WARNING does not name the operation: {warnings_logged}",
            )
            print(
                "(d) PASS  timeout logged a WARNING and raised; "
                "the write did not proceed"
            )
        finally:
            os.environ.pop("CAT_GRAPH_WRITE_LOCK_TIMEOUT_SECONDS", None)

    # ── (e) delete path serialises against a concurrent write ─────────────
    async def test_e_delete_serialises_against_write():
        reset()

        class _HandlerMethods:
            def __init__(self):
                self.agent_id = "tenant-d"
                self._local_write_locks = {}
                self.order = []

            @twl.tenant_write_locked("add points to tenant")
            async def add_points_to_tenant(self):
                self.order.append("add-enter")
                await asyncio.sleep(0.05)
                self.order.append("add-exit")

            @twl.tenant_write_locked("delete tenant points")
            async def delete_tenant_points(self):
                self.order.append("delete-enter")
                await asyncio.sleep(0.05)
                self.order.append("delete-exit")

        handler = _HandlerMethods()
        await asyncio.gather(
            handler.add_points_to_tenant(),
            handler.delete_tenant_points(),
        )

        _check(
            _max_active.get("graph-write:tenant-d", 0) == 1,
            f"(e) delete/write overlapped: {_max_active}",
        )
        # Strict pairing in one serial order.
        _check(
            handler.order[0].endswith("enter") and handler.order[1].endswith("exit"),
            f"(e) interleaved enter/exit: {handler.order}",
        )
        _check(
            handler.order[2].endswith("enter") and handler.order[3].endswith("exit"),
            f"(e) interleaved enter/exit: {handler.order}",
        )
        _check(
            _acquisitions.get("graph-write:tenant-d") == 2,
            f"(e) expected 2 acquisitions, got {_acquisitions}",
        )
        print(
            f"(e) PASS  delete path serialised against write, order={handler.order}"
        )

    # ── (f) in-process fallback: a pre-initialised EMPTY dict still serialises ──
    async def test_f_fallback_with_empty_lock_dict_serialises():
        """The real handler initialises ``_local_write_locks = {}``; an empty
        dict must be reused, not replaced by a fresh one on every call."""
        reset()
        handler = _Handler("tenant-f")
        saved = twl.distributed_lock
        twl.distributed_lock = None
        order = []
        try:
            async def writer(n):
                async with twl.acquire_tenant_write_lock(handler, "tenant-f", f"w{n}"):
                    order.append(f"enter-{n}")
                    await asyncio.sleep(0.05)
                    order.append(f"exit-{n}")

            await asyncio.gather(writer(1), writer(2))
        finally:
            twl.distributed_lock = saved
        _check(
            order[0].startswith("enter") and order[1].startswith("exit"),
            f"(f) fallback writers overlapped: {order}",
        )
        _check(
            "tenant-f" in handler._local_write_locks,
            f"(f) fallback lock not stored on the handler: {handler._local_write_locks}",
        )
        print(f"(f) PASS  fallback lock serialises with a pre-initialised dict, order={order}")

    # ── Runner ────────────────────────────────────────────────────────────
    async def _run_all():
        await test_a_same_tenant_never_overlaps()
        await test_b_different_tenants_run_concurrently()
        await test_c_reentrant_call_acquires_once()
        await test_d_timeout_raises_and_blocks_write()
        await test_e_delete_serialises_against_write()
        await test_f_fallback_with_empty_lock_dict_serialises()

    asyncio.run(_run_all())
    print("\nALL 6 TENANT WRITE LOCK TESTS PASSED")
