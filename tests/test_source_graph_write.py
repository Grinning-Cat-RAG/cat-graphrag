"""Standalone, import-safe harness for per-source graph writes.

Verifies ``source_graph_writer`` (used by ``graphrag_handler`` to replace the
per-chunk concurrent graph writes with ONE transaction per source):

  (a) two chunks of one source produce exactly ONE write transaction;
  (b) entities are deduped by id (across 3 chunks) and relations by
      (source_id, target_id, rel_type);
  (c) above the size threshold the transactions run SEQUENTIALLY (recorded
      start/end never overlap);
  (d) a re-entrant decorated call completes without acquiring the tenant lock
      twice;
  (e) two different tenants do not block each other.

The Cat plugin loader imports EVERY ``.py`` under a plugin folder at
activation, so this whole file is inert on import: every import, class and
test lives inside ``if __name__ == "__main__":``. ``cat.log`` and
``cat.db.crud`` are stubbed in ``sys.modules`` BEFORE the modules are loaded,
so no Cat install, no Redis and no Neo4j are needed.

Run as a plain script:

    python tests/test_source_graph_write.py
"""

if __name__ == "__main__":
    import asyncio
    import importlib.util
    import json
    import os
    import sys
    import types
    from contextlib import asynccontextmanager

    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ── Stub ``cat.log`` ──────────────────────────────────────────────────
    class _StubLog:
        def info(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

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
    _total_active = 0
    _max_total_active = 0
    _guard = asyncio.Lock()

    @asynccontextmanager
    async def distributed_lock(key_pattern, timeout=10.0, blocking_timeout=15.0):
        global _total_active, _max_total_active
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
        try:
            yield
        finally:
            async with _guard:
                _active[key_pattern] -= 1
                _total_active -= 1
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

    def _load(name, filename):
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(REPO_ROOT, filename)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    writer = _load("source_graph_writer_under_test", "source_graph_writer.py")
    twl = _load("tenant_write_lock_under_test", "tenant_write_lock.py")

    # ── Fixtures ──────────────────────────────────────────────────────────
    class _Handler:
        def __init__(self, tenant):
            self.agent_id = tenant
            self._local_write_locks = {}

    def reset_locks():
        global _total_active, _max_total_active
        _locks.clear()
        _active.clear()
        _max_active.clear()
        _acquisitions.clear()
        _total_active = 0
        _max_total_active = 0

    def _check(cond, message):
        if not cond:
            raise AssertionError(message)

    def _entity(entity_id, doc_id, confidence=1.0):
        return {
            "id": entity_id,
            "name": entity_id,
            "type": "CONCEPT",
            "metadata": json.dumps(
                {"source_documents": [doc_id], "confidence": confidence}
            ),
            "embedding": None,
        }

    def _mention(doc_id, entity_id, confidence=1.0):
        return {"doc_id": doc_id, "entity_id": entity_id, "confidence": confidence}

    # ── (a) two chunks -> one transaction ─────────────────────────────────
    async def test_a_two_chunks_one_transaction():
        entities = [_entity("e1", "doc-1"), _entity("e2", "doc-2")]
        mentions = [_mention("doc-1", "e1"), _mention("doc-2", "e2")]
        relations = [
            {
                "source_id": "e1",
                "target_id": "e2",
                "rel_type": "RELATED_TO",
                "weight": 1.0,
                "source_files": ["book.md"],
            }
        ]
        plans = []

        async def execute(plan):
            plans.append(plan)

        count = await writer.write_source_graph(
            entities, mentions, relations, execute
        )

        _check(count == 1, f"(a) expected 1 transaction, got {count}")
        _check(len(plans) == 1, f"(a) executor called {len(plans)} times")
        _check(len(plans[0]["entities"]) == 2, "(a) both chunks not in one plan")
        print(f"(a) PASS  two chunks -> {count} write transaction")

    # ── (b) dedupe entities and relations ─────────────────────────────────
    async def test_b_dedupe_across_chunks():
        entities = [
            _entity("same", "doc-1", 0.4),
            _entity("same", "doc-2", 0.9),
            _entity("same", "doc-3", 0.6),
        ]
        mentions = [
            _mention("doc-1", "same"),
            _mention("doc-2", "same"),
            _mention("doc-3", "same"),
        ]
        relations = [
            {
                "source_id": "same",
                "target_id": "other",
                "rel_type": "RELATED_TO",
                "weight": 1.0,
                "source_files": ["book.md"],
            },
            {
                "source_id": "same",
                "target_id": "other",
                "rel_type": "RELATED_TO",
                "weight": 2.0,
                "source_files": ["book.md"],
            },
            {
                "source_id": "same",
                "target_id": "other",
                "rel_type": "RELATED_TO",
                "weight": 3.0,
                "source_files": ["book.md"],
            },
        ]
        plans = []

        async def execute(plan):
            plans.append(plan)

        await writer.write_source_graph(entities, mentions, relations, execute)

        plan = plans[0]
        _check(len(plan["entities"]) == 1, f"(b) entities not deduped: {len(plan['entities'])}")
        _check(len(plan["relations"]) == 1, f"(b) relations not deduped: {len(plan['relations'])}")
        meta = json.loads(plan["entities"][0]["metadata"])
        _check(
            meta["source_documents"] == ["doc-1", "doc-2", "doc-3"],
            f"(b) source_documents not unioned: {meta}",
        )
        _check(
            abs(meta["confidence"] - 0.9) < 1e-9,
            f"(b) confidence not max: {meta}",
        )
        _check(
            abs(plan["relations"][0]["weight"] - 2.25) < 1e-9,
            f"(b) relation weight not averaged: {plan['relations'][0]['weight']}",
        )
        _check(
            len(plan["mentions"]) == 3,
            f"(b) per-document mentions must stay distinct: {len(plan['mentions'])}",
        )
        print("(b) PASS  entities by id and relations by key deduped")

    # ── (c) above threshold -> sequential transactions ────────────────────
    async def test_c_above_threshold_is_sequential():
        entities = [_entity(f"e{i}", f"doc-{i}") for i in range(5)]
        mentions = [_mention(f"doc-{i}", f"e{i}") for i in range(5)]
        spans = []
        active = 0
        max_active = 0

        async def execute(plan):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            start = asyncio.get_running_loop().time()
            await asyncio.sleep(0.02)
            end = asyncio.get_running_loop().time()
            active -= 1
            spans.append((start, end))

        count = await writer.write_source_graph(
            entities, mentions, [], execute, max_items=2
        )

        _check(count > 1, f"(c) expected multiple batches, got {count}")
        _check(len(spans) == count, f"(c) spans={len(spans)} count={count}")
        _check(max_active == 1, f"(c) writes overlapped: max_active={max_active}")
        for previous, current in zip(spans, spans[1:]):
            _check(
                current[0] >= previous[1],
                f"(c) overlapping transactions: {previous} then {current}",
            )
        print(f"(c) PASS  {count} transactions ran strictly sequentially")

    # ── (d) re-entrant decorated call acquires once ───────────────────────
    async def test_d_reentrant_decorated_call():
        reset_locks()

        class _Reentrant:
            def __init__(self):
                self.agent_id = "tenant-r"
                self._local_write_locks = {}
                self.inner_ran = False

            @twl.tenant_write_locked("outer")
            async def outer(self):
                await self.inner()
                return "done"

            @twl.tenant_write_locked("inner")
            async def inner(self):
                self.inner_ran = True
                await asyncio.sleep(0.01)

        handler = _Reentrant()
        result = await asyncio.wait_for(handler.outer(), timeout=2.0)
        _check(result == "done", f"(d) unexpected result: {result!r}")
        _check(handler.inner_ran, "(d) inner did not run")
        _check(
            _acquisitions.get("graph-write:tenant-r") == 1,
            f"(d) lock acquired more than once: {_acquisitions}",
        )
        print("(d) PASS  re-entrant decorated call acquired the lock once")

    # ── (e) different tenants do not block each other ─────────────────────
    async def test_e_different_tenants_concurrent():
        reset_locks()
        handler = _Handler("tenant-a")

        async def take(tenant):
            async with twl.acquire_tenant_write_lock(handler, tenant, "write"):
                await asyncio.sleep(0.1)

        await asyncio.gather(take("tenant-a"), take("tenant-b"))
        _check(
            _max_total_active >= 2,
            f"(e) tenants blocked each other: max_total_active={_max_total_active}",
        )
        print(
            f"(e) PASS  different tenants ran concurrently "
            f"(max_total_active={_max_total_active})"
        )

    async def _run_all():
        await test_a_two_chunks_one_transaction()
        await test_b_dedupe_across_chunks()
        await test_c_above_threshold_is_sequential()
        await test_d_reentrant_decorated_call()
        await test_e_different_tenants_concurrent()

    asyncio.run(_run_all())
    print("\nALL 5 SINGLE-SOURCE WRITE TESTS PASSED")
