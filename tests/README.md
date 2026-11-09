# Standalone harnesses

The files in this directory are import-safe standalone **scripts**: each `.py`
under the plugin is imported by the Cat loader at activation, so every script
keeps its test body inside `if __name__ == "__main__":` and importing it
executes nothing. Run them as scripts, never by pytest, from the host venv (it
has the Cat dependencies) or inside the Cat container:

```bash
cd ../grinning-cat-core
uv run python /path/to/cat-graphrag/tests/test_hook_reload_order.py
```

Each script registers the plugin folder as a synthetic package, stubs the
external services it does not need and prints `PASS <test>` lines.

The graph phases run on the core's own ingestion engine (`cat.rabbit_hole`):
their harnesses load the REAL core and need a Redis Stack on `localhost:6379`
(database 14 is used and cleaned). They share the test doubles of
`graph_fakes.py` (an in-memory graph at the Neo4j boundary, a Cat and a
RabbitHole for the engine).

Requirements per script:

- self-contained, run from the host venv (stubbed Neo4j/spaCy/Cat/tenant lock):
  `test_hook_reload_order.py`, `test_kg_master_switch.py`,
  `test_ner_segmentation.py`, `test_terminology_refresh.py`,
  `test_versioned_schema.py`, `test_reindex_swap.py`,
  `test_graph_storage_contracts.py`, `test_llm_call_deadline.py`, `test_llm_error_mapping.py`,
  `test_batched_delete.py`, `test_lock_scope.py`, `test_tenant_write_lock.py`,
  `test_source_graph_write.py`, `test_pending_source_tasks.py`,
  `test_transient_retry.py`;
- `test_cypher_comments.py` — static guard against `#` comments inside Cypher
  strings, plus a live execution of the concept-relation template when a Neo4j
  password is provided (`NEO4J_PASSWORD`/`--password`); without one it reports
  `SKIP live execution` and the static audit is the fallback;
- `test_backfill_duplicates.py` — a CLI that needs a live Neo4j: pass
  `--password` (plus `--uri`/`--user`/`--tenant` as needed);
- `test_derived_graph_unconditional.py`, `test_phase_fingerprint.py` — load the
  REAL `cat` package: run them from `grinning-cat-core` with the plugin's
  dependencies (`neo4j`, `spacy`, `langdetect`) installed;
- `test_graph_phases.py`, `test_graph_phases_stateful.py` — the REAL core and
  a Redis Stack on `localhost:6379`; the stateful one also needs `hypothesis`
  and accepts `--examples N` and `--mutate llm-dependency|no-purge|name-only-provenance|outside-as-ingestion|no-row-rebuild` (a wrong
  rule injected on purpose: the run must then report the mutation as caught);
- `test_outside_writes.py` — the REAL core and Redis: points written outside
  an ingestion get their graph (a source with a row through the core's resume,
  one without through a background job; nothing for the procedures);
- `test_provenance_scope.py` — the REAL core: the RELATED_TO provenance
  key is scoped by knowledge base / chat in every writer (spaCy, LLM,
  recompute); the stripping Cypher runs in `test_graph_purge_live.py`;
- `test_concept_generation.py` — the REAL core and Redis (the recompute
  lock): a change of the agent's LLM recomputes the concepts once, across
  pods, and a failure is retried;
- `test_ingestion_lane.py` — the REAL core and Redis: the heavy steps of an
  ingestion (language detection included) run on the core's ingestion lane
  (a one-worker lane), a chat message's analysis stays off it and off the
  event loop (a thread of its own);
- `test_write_lock_handoff.py` — the REAL core and Redis: the tenant write
  lock hands over at once between the writers of one process (a local FIFO
  gate in front of the polling distributed lock), bounded by its timeout
  against another pod, unaffected by a cancelled waiter;
- `test_maintenance_cli.py`, `test_maintenance_cli_stateful.py` — the
  maintenance CLI on the REAL core RabbitHole (`reingest_source` /
  `rerun_source`, rows, locks, phase engine, automatic resume) and a Redis
  Stack on `localhost:6379`; the stateful one also needs `hypothesis` and
  accepts `--examples N` and `--mutate no-reset|no-legacy`;
- `test_graph_purge_live.py` — the REAL core, Redis, and a live Neo4j 5
  (`--uri`, `--password` or `NEO4J_PASSWORD`; without a password it reports
  `SKIP`): the source-scoped purge, the paged point listing, the tenant wipe of
  the maintenance `--graph` and the graph checks of `--verify`.

The core's virtualenv does not ship the plugin's dependencies: either install
them there, or append the plugin's own `.venv` site-packages AFTER the core's
(so the core's packages win), e.g. with a small runner that does
`sys.path.append(<plugin>/.venv/lib/python3.13/site-packages)` and then
`runpy.run_path(<harness>, run_name="__main__")`.
