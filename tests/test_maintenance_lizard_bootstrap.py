#!/usr/bin/env python3
"""
Regression test: the maintenance CLI bootstraps the Cat lizard lazily.

``CheshireCat.create`` resolves non-system agents' plugins through
``BillTheLizard().plugin_manager.plugins``. In a fresh ``docker exec`` process
the lizard is never bootstrapped (the server does it in the uvicorn
lifespan), so the registry is empty and ``mad_hatter`` ``KeyError: 'base_plugin'``
fired. ``maintenance_agent._ensure_lizard_booted()`` must populate the lizard's
plugin registry once, lazily, and be idempotent.

Run INSIDE the container (like test_backfill_duplicates.py):

    docker exec cheshire_cat_core python /app/cat/plugins/cat_graphrag/test_maintenance_lizard_bootstrap.py

IMPORT-SAFETY: this file lives in the plugin folder, which the Cat plugin
loader imports recursively at activation. The ENTIRE body lives under
``if __name__ == "__main__":`` so importing the file executes zero code.
"""

if __name__ == "__main__":
    import asyncio
    import os
    import sys


    def _fail(msg: str) -> None:
        print(f"FAIL: {msg}")
        sys.exit(1)


    async def _check() -> int:
        from cat.looking_glass.bill_the_lizard import BillTheLizard
        from cat.plugins.cat_graphrag.maintenance_agent import _ensure_lizard_booted

        # Fresh process: the lizard must NOT be bootstrapped yet, so the empty
        # registry is the state the bug reproduced in.
        if BillTheLizard().plugin_manager.plugins:
            _fail("lizard plugin registry already populated at process start")

        await _ensure_lizard_booted()

        plugins = BillTheLizard().plugin_manager.plugins
        if not plugins:
            _fail("lizard plugin registry still empty after _ensure_lizard_booted")
        if "base_plugin" not in plugins:
            _fail(f"'base_plugin' missing from lizard plugins (have: {sorted(plugins)[:10]}...)")
        before = len(plugins)

        # Idempotence: a second call must not error and must not wipe/re-add.
        await _ensure_lizard_booted()
        after = len(BillTheLizard().plugin_manager.plugins)
        if after != before:
            _fail(f"lizard plugin count changed on second call: {before} -> {after}")

        return before


    async def _main() -> int:
        count = 0
        try:
            count = await _check()
        except Exception as exc:  # noqa: BLE001 - report and fail
            _fail(f"{type(exc).__name__}: {exc}")
        print(f"PASS: lizard bootstrapped with {count} plugins including 'base_plugin', idempotent")
        return 0


    raise SystemExit(asyncio.run(_main()))