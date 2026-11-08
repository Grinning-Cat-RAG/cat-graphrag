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
- `test_maintenance_lizard_bootstrap.py` — needs the plugin DEPLOYED as
  `cat.plugins.cat_graphrag`, i.e. run it inside the Cat container with the
  plugin installed, not from a bare host checkout;
- `test_derived_graph_unconditional.py`, `test_phase_fingerprint.py` — load the
  REAL `cat` package: run them from `grinning-cat-core` with the plugin's
  dependencies (`neo4j`, `spacy`, `langdetect`) installed;
- `test_graph_phases.py`, `test_graph_phases_stateful.py` — the REAL core and
  a Redis Stack on `localhost:6379`; the stateful one also needs `hypothesis`
  and accepts `--examples N` and `--mutate llm-dependency|no-purge` (a wrong
  rule injected on purpose: the run must then report the mutation as caught);
- `test_graph_purge_live.py` — the REAL core, Redis, and a live Neo4j 5
  (`--uri`, `--password` or `NEO4J_PASSWORD`; without a password it reports
  `SKIP`): the source-scoped purge and the paged point listing.

The core's virtualenv does not ship the plugin's dependencies: either install
them there, or append the plugin's own `.venv` site-packages AFTER the core's
(so the core's packages win), e.g. with a small runner that does
`sys.path.append(<plugin>/.venv/lib/python3.13/site-packages)` and then
`runpy.run_path(<harness>, run_name="__main__")`.
- `test_maintenance_resume_cli.py`, `test_reingest_clears_diary.py`,
  `test_reingest_enumeration.py`, `test_maintenance_exit_summary.py`,
  `test_ingestion_safety_fixes.py`, `test_maintenance_decoupling.py` — need the
  REAL EffING: inside the Cat container, or from an EffING-enabled
  `grinning-cat-core` checkout with `EFFING_DIR` and `MULTIMODAL_DIR` pointing at
  checkouts of `cat-efficient-ingestion` and `cat-multimodal-ingestion`
  (registered by `effing_checkout.py`; a missing checkout is a FAIL, never a
  skip):

  ```bash
  cd ../grinning-cat-core
  EFFING_DIR=/path/to/cat-efficient-ingestion \
  MULTIMODAL_DIR=/path/to/cat-multimodal-ingestion PYTHONPATH=. \
      .venv/bin/python /path/to/cat-graphrag/tests/test_maintenance_resume_cli.py
  ```
