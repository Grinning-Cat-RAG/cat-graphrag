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

The harnesses that touch the phase contract (`test_ingestion_phase_machine.py`,
`test_phase_abc_migration.py`) install a minimal stub of the efficient
ingestion plugin (EffING, `cat.plugins.cat_efficient_ingestion`). The
maintenance CLI harnesses drive EffING's registry/re-embed/resume code and need
the REAL plugin (see the last group below).

Requirements per script:

- self-contained, run from the host venv (stubbed Neo4j/spaCy/Cat/tenant lock):
  `test_hook_reload_order.py`, `test_kg_master_switch.py`,
  `test_ner_segmentation.py`, `test_terminology_refresh.py`,
  `test_versioned_schema.py`, `test_reindex_swap.py`,
  `test_ingestion_phase_machine.py`, `test_phase_abc_migration.py`,
  `test_llm_call_deadline.py`, `test_llm_error_mapping.py`,
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
  dependencies (`neo4j`, `spacy`, `langdetect`) installed.
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
