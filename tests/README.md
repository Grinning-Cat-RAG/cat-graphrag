# Standalone harnesses

The authoritative behavioral suite is the HOST repository at
`grinning-cat-core/tests/mygraph/` (pytest).

The files in this directory are import-safe standalone **scripts**: each `.py`
under the plugin is imported by the Cat loader at activation, so every script
keeps its entire body (imports included) inside `if __name__ == "__main__":`
and importing it executes nothing. Run them as scripts, never by pytest, from
the host venv (it has the Cat dependencies) or inside the Cat container:

```bash
cd ../grinning-cat-core
uv run python /path/to/cat-graphrag.my/tests/test_hook_reload_order.py
```

Each script registers the plugin folder as a synthetic package (the folder name
`cat-graphrag.my` cannot be imported directly), stubs the external services it
does not need and prints `PASS <test>` lines. Requirements per script:

- `test_hook_reload_order.py`, `test_kg_master_switch.py`,
  `test_ner_segmentation.py`, `test_terminology_refresh.py`,
  `test_versioned_schema.py`, `test_reindex_swap.py` — self-contained (stubbed
  Neo4j/spaCy/Cat), run from the host venv;
- `test_backfill_duplicates.py` — a CLI that needs a live Neo4j: pass
  `--password` (plus `--uri`/`--user`/`--tenant` as needed);
- `test_maintenance_lizard_bootstrap.py` — needs the plugin DEPLOYED as
  `cat.plugins.cat_graphrag`, i.e. run it inside the Cat container with the
  plugin installed, not from a bare host checkout.

## Retired: `tests/test_ingestion_phase_machine.py`

Removed. It documented the obsolete single-phase `graph_building` model
(`ingestion_phase_pending` + a diary/generation check in
`before_ingestion_status_completed`) and imported the pre-extraction module
path (the legacy `cat.plugins.*` namespace).

The current two-phase model (`graph_spacy_extraction` /
`graph_llm_extraction`, registered through EffING's `ingestion_phase_specs`)
is covered by:

- `../test_ingestion_phase_machine.py` — the maintained standalone harness at
  the plugin root (registration, markers, tri-state `ingestion_phase_run`,
  completion gate, mode truth table, transfer cleanup);
- `grinning-cat-core/tests/mygraph/` — the authoritative host suite.
