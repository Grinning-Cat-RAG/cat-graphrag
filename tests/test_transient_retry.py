"""Standalone, import-safe harness for Neo4j transient-error retries.

Verifies ``versioning.with_transient_retry``:

  (a) a ``DeadlockDetected`` transient error twice then success -> succeeds
      with exactly two WARNING logs (one per retry) and three attempts;
  (b) a non-transient error is NOT retried and propagates immediately;
  (c) exhausting the attempts re-raises the ORIGINAL transient type;
  (d) the exponential backoff is bounded by ``max_delay`` even at high
      attempt numbers.

The Cat plugin loader imports EVERY ``.py`` under a plugin folder at
activation, so this whole file is inert on import: every import, class and
test lives inside ``if __name__ == "__main__":``. ``cat.log`` and
``neo4j.exceptions`` are stubbed in ``sys.modules`` before ``versioning`` is
loaded, so it needs no Cat install, no Neo4j and no network.

Run as a plain script:

    python tests/test_transient_retry.py
"""

if __name__ == "__main__":
    import importlib.util
    import os
    import sys
    import types

    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ── Stub ``cat.log`` (versioning imports ``from cat.log import log``) ──
    warnings = []

    class _StubLog:
        def info(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

        def warning(self, message, *args, **kwargs):
            warnings.append(str(message))

        def error(self, *args, **kwargs):
            pass

    cat_mod = types.ModuleType("cat")
    cat_mod.__path__ = []
    cat_mod.log = _StubLog()
    sys.modules["cat"] = cat_mod
    log_mod = types.ModuleType("cat.log")
    log_mod.log = _StubLog()
    sys.modules["cat.log"] = log_mod

    # ── Stub ``neo4j.exceptions`` with the transient hierarchy ──
    neo4j_mod = types.ModuleType("neo4j")
    neo4j_mod.__path__ = []
    exc_mod = types.ModuleType("neo4j.exceptions")

    class TransientError(Exception):
        pass

    class DeadlockDetected(TransientError):
        pass

    class LockClientStopped(TransientError):
        pass

    class LeaderSwitch(TransientError):
        pass

    class Termination(TransientError):
        pass

    class ClientError(Exception):
        pass

    exc_mod.TransientError = TransientError
    exc_mod.DeadlockDetected = DeadlockDetected
    exc_mod.LockClientStopped = LockClientStopped
    exc_mod.LeaderSwitch = LeaderSwitch
    exc_mod.Termination = Termination
    exc_mod.ClientError = ClientError
    neo4j_mod.exceptions = exc_mod
    sys.modules["neo4j"] = neo4j_mod
    sys.modules["neo4j.exceptions"] = exc_mod

    # ── Load the real versioning.py from this repository root ──
    spec = importlib.util.spec_from_file_location(
        "catgraphrag_versioning_retrytest",
        os.path.join(REPO_ROOT, "versioning.py"),
    )
    versioning = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(versioning)

    # Capture sleeps instead of actually waiting.
    real_sleep = versioning.asyncio.sleep
    slept = []

    async def _fake_sleep(delay):
        slept.append(delay)

    def _reset():
        warnings.clear()
        slept.clear()
        versioning.asyncio.sleep = _fake_sleep

    def _restore():
        versioning.asyncio.sleep = real_sleep

    def _run(coro):
        import asyncio

        return asyncio.new_event_loop().run_until_complete(coro)

    # ── (a) transient twice then success ──
    async def _case_a():
        class FakeSession:
            def __init__(self):
                self.calls = 0

            async def execute_write(self, fn):
                self.calls += 1
                if self.calls <= 2:
                    raise DeadlockDetected(
                        "deadlock detected while trying to acquire locks"
                    )
                return "ok"

        session = FakeSession()

        result = await versioning.with_transient_retry(
            lambda: session.execute_write(lambda tx: None),
            operation="create similarity relationships",
        )
        assert result == "ok", result
        assert session.calls == 3, session.calls
        assert len(warnings) == 2, warnings
        assert len(slept) == 2, slept
        assert all("attempt" in w and "create similarity" in w for w in warnings), warnings

    # ── (b) non-transient is not retried ──
    async def _case_b():
        calls = {"n": 0}

        async def op():
            calls["n"] += 1
            raise ClientError("bad query")

        try:
            await versioning.with_transient_retry(op, operation="write")
        except ClientError:
            pass
        else:
            raise AssertionError("non-transient error was swallowed")
        assert calls["n"] == 1, calls["n"]
        assert warnings == [], warnings
        assert slept == [], slept

    # ── (c) exhaustion re-raises the original type ──
    async def _case_c():
        calls = {"n": 0}

        async def op():
            calls["n"] += 1
            raise LockClientStopped("lock client stopped")

        try:
            await versioning.with_transient_retry(op, attempts=3, operation="write")
        except LockClientStopped:
            pass
        else:
            raise AssertionError("exhausted retries did not re-raise")
        assert calls["n"] == 3, calls["n"]
        assert len(warnings) == 2, warnings

    # ── (d) backoff bounded by max_delay ──
    async def _case_d():
        calls = {"n": 0}

        async def op():
            calls["n"] += 1
            raise DeadlockDetected("deadlock")

        try:
            await versioning.with_transient_retry(
                op, attempts=12, base_delay=0.05, max_delay=0.2, operation="write"
            )
        except DeadlockDetected:
            pass
        else:
            raise AssertionError("did not raise")
        assert calls["n"] == 12, calls["n"]
        assert len(slept) == 11, slept
        assert all(0 < d <= 0.2 for d in slept), slept
        # Direct check: even at an extreme attempt the delay stays capped.
        assert versioning._retry_delay(999, 0.05, 0.2) <= 0.2
        assert versioning._retry_delay(1, 0.05, 0.2) > 0

    try:
        _reset()
        _run(_case_a())
        _reset()
        _run(_case_b())
        _reset()
        _run(_case_c())
        _reset()
        _run(_case_d())
    finally:
        _restore()

    print("PASS (a) DeadlockDetected twice then success: 2 warnings, 3 attempts")
    print("PASS (b) non-transient ClientError propagated immediately, no retry")
    print("PASS (c) exhausted attempts re-raised original LockClientStopped")
    print("PASS (d) backoff bounded by max_delay=0.2 over 11 retries")
