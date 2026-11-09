"""The tenant write lock hands over at once between the writers of one process.

The core's ``distributed_lock`` (redis-py) polls a busy lock every 100 ms: with many writers of the same tenant in one
process (the extraction of every chunk of an upload, the similarity links, the structure) each handover left the lock
idle until the next poll. The writers of a process now queue on a local gate (FIFO, immediate handover) and only the
head of the queue waits on Redis, so the lock polls only against another pod.

- many writers of one process: never overlapping, handed over at once;
- the lock held by another pod: every writer gives up within its timeout (the gate's wait counts), and the next write
  runs once the other pod releases;
- a writer cancelled while it waits does not block the ones behind it;
- another pod gets its turn while this pod keeps writing (a streak of local handovers pauses for longer than a poll).

Standalone script (plugin rules: the body runs only as ``__main__``). Needs the core checkout and a Redis Stack on
localhost:6379 (database 14):

    cd ../grinning-cat-core
    .venv/bin/python /path/to/cat-graphrag/tests/test_write_lock_handoff.py
"""

if __name__ == "__main__":
    import asyncio
    import importlib
    import os
    import sys
    import time
    import traceback
    import types

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import graph_fakes as gf

    gf.bootstrap("_graphrag_handoff")
    twl = importlib.import_module("_graphrag_handoff.tenant_write_lock")

    from cat.db.crud import distributed_lock

    def handler(agent):
        return types.SimpleNamespace(agent_id=agent)

    async def test_the_writers_of_a_process_hand_over_at_once():
        writers, inside, overlaps = 20, [], []

        async def write(i):
            async with twl.acquire_tenant_write_lock(handler("handoff_a"), None, f"write {i}"):
                if inside:
                    overlaps.append(i)
                inside.append(i)
                await asyncio.sleep(0.001)
                inside.remove(i)

        start = time.monotonic()
        await asyncio.wait_for(asyncio.gather(*(write(i) for i in range(writers))), timeout=30)
        elapsed = time.monotonic() - start
        assert not overlaps, f"writers overlapped: {overlaps}"
        assert elapsed < writers * 0.02, f"{writers} writers took {elapsed:.2f}s: the lock sat idle between them"

    async def test_a_lock_held_by_another_pod_times_out_every_writer_then_frees():
        os.environ["CAT_GRAPH_WRITE_LOCK_TIMEOUT_SECONDS"] = "1"
        try:
            async with distributed_lock("graph-write:handoff_b", timeout=10, blocking_timeout=1):  # the other pod
                start = time.monotonic()
                outcomes = await asyncio.wait_for(asyncio.gather(*(
                    _write_once("handoff_b") for _ in range(3)
                ), return_exceptions=True), timeout=10)
                elapsed = time.monotonic() - start
            assert all(isinstance(o, twl.TenantWriteLockTimeout) for o in outcomes), outcomes
            assert elapsed < 2.0, f"the writers waited {elapsed:.2f}s, beyond their timeout"
            assert await asyncio.wait_for(_write_once("handoff_b"), timeout=5) == "written"
        finally:
            del os.environ["CAT_GRAPH_WRITE_LOCK_TIMEOUT_SECONDS"]

    async def _write_once(agent):
        async with twl.acquire_tenant_write_lock(handler(agent), None, "write"):
            return "written"

    async def test_a_cancelled_writer_does_not_block_the_others():
        release = asyncio.Event()
        holding = asyncio.Event()

        async def holder():
            async with twl.acquire_tenant_write_lock(handler("handoff_c"), None, "hold"):
                holding.set()
                await release.wait()

        first = asyncio.create_task(holder())
        waiter = None
        try:
            await asyncio.wait_for(holding.wait(), timeout=5)
            waiter = asyncio.create_task(_write_once("handoff_c"))
            behind = asyncio.create_task(_write_once("handoff_c"))
            await asyncio.sleep(0.05)
            waiter.cancel()
            release.set()
            assert await asyncio.wait_for(behind, timeout=5) == "written"
        finally:
            release.set()
            for task in (first, waiter):
                if task is not None:
                    task.cancel()
            await asyncio.gather(first, *( [waiter] if waiter else []), return_exceptions=True)

    async def test_another_pod_gets_its_turn_while_this_pod_keeps_writing():
        """The writers of this pod hand over to each other without a pause; another pod polls the lock every 100 ms.
        A streak of local handovers pauses for longer than a poll, so the other pod gets its turn."""
        stop = asyncio.Event()

        async def local_writer():
            while not stop.is_set():
                async with twl.acquire_tenant_write_lock(handler("handoff_d"), None, "local"):
                    await asyncio.sleep(0.3)

        writers = [asyncio.create_task(local_writer()) for _ in range(3)]
        try:
            await asyncio.sleep(0.05)
            start = time.monotonic()
            async with distributed_lock("graph-write:handoff_d", timeout=5, blocking_timeout=2.5):  # the other pod
                waited = time.monotonic() - start
            assert waited < 2.5, waited
        finally:
            stop.set()
            await asyncio.wait_for(asyncio.gather(*writers, return_exceptions=True), timeout=10)

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
        print(f"\ntest_write_lock_handoff: {len(TESTS) - failed}/{len(TESTS)} passed")
        return failed

    sys.exit(1 if asyncio.run(run_all()) else 0)
