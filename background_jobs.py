"""Background work of the plugin that outlives the request that started it.

The core loads an agent (a ``CheshireCat`` with its own vector memory handler) for every request and shuts it down
when the request ends: shutting it down closes the handler, which cancels the tasks the handler tracks. Work that
must outlive the request (the recompute of the concepts, the graph of points written outside an ingestion) is held
here instead, and works on an agent it loads itself.

Import-safe: only the standard library at import time.
"""
import asyncio
from typing import Coroutine, Set

#: The running jobs: held here so that nothing but their end releases them.
JOBS: Set[asyncio.Task] = set()


def start(job: Coroutine) -> asyncio.Task:
    """Run ``job`` in background, held until it ends (whatever the request that started it does meanwhile)."""
    task = asyncio.create_task(job)
    JOBS.add(task)
    task.add_done_callback(JOBS.discard)
    return task


async def load_agent(tenant: str):
    """The agent a job works on: loaded as a request loads it (its handler, its LLM), owned by the job. None when the
    agent does not exist anymore."""
    from cat.looking_glass.bill_the_lizard import BillTheLizard

    return await BillTheLizard().get_cheshire_cat(tenant)
