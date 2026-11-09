# Neo4j GraphRAG Advanced for Grinning Cat

Bring graph-aware memory retrieval to your Grinning Cat instance.

**Neo4j GraphRAG Advanced** is a plugin that replaces the standard vector-only memory layer with a hybrid GraphRAG pipeline powered by **Neo4j** and **spaCy**. It stores memories as documents, extracts entities and relations into a knowledge graph, and combines **vector similarity**, **direct entity matches**, and **graph expansion** to retrieve more relevant context.

If your Cat needs to remember not only *similar text* but also *who*, *what*, *where*, and *how things are connected*, this plugin is built for that.

---

## Why this plugin exists

Traditional vector search is great at semantic similarity, but it can miss documents that are topically relevant when the same concept is expressed with different wording.

This plugin improves recall by adding a graph layer on top of embeddings:

- documents are stored in Neo4j
- entities are extracted from each memory
- relationships are created between entities
- retrieval combines semantic search with graph-aware expansion

The result is a memory system that can better answer queries involving:

- people, organizations, products, dates, and locations
- technical stacks and domain keywords
- connections between concepts across multiple memories
- multi-hop knowledge that would be hard to recover with vectors alone

---

## What it does

### Core features

- **Native Neo4j vector search** for document embeddings
- **Entity extraction with spaCy** during ingestion
- **Technology keyword detection** with regex patterns for tools and frameworks
- **Knowledge graph construction** with `Entity`, `Document`, and `Collection` nodes
- **Semantic relationship extraction** between entities
- **Hybrid recall pipeline**:
  - direct entity match from the user query
  - related-entity graph traversal
  - vector similarity search
  - score merging and reranking
- **Collection-aware storage** for Grinning Cat memories
- **Tenant isolation** using the Cat agent identifier
- **Cross-document entity reuse** so knowledge persists across multiple memories

### Retrieval strategy at a glance

When Grinning Cat searches memory, the plugin can:

1. capture the raw user message before recall starts
2. extract entities from the query
3. find documents that directly mention those entities
4. traverse related entities in the graph up to a configurable depth
5. run standard vector search in parallel
6. merge and rerank all candidates into a final result list

This makes retrieval both **semantic** and **topically grounded**.

---

## How it integrates with Grinning Cat

This repository is a plugin for **Grinning Cat** and depends on the `base_plugin` package.

At runtime it provides:

- a custom vector database handler: `GraphRAGHandler`
- a settings model: `Neo4jGraphRAGConfig`, registered through `factory_allowed_vector_databases`
- ingestion hooks: embedder injection and chunk ordering (`before_rabbithole_stores_documents`), and the two graph phases added to the core's ingestion (`rabbithole_ingestion_phases`, see below)
- a query hook that captures the raw user message before memory recall
- maintenance hooks: provenance reconciliation of a newly created agent, and old-store cleanup on vector-memory switch (`after_vector_memory_transfer_on_agent`)

The query hook is important: it allows the retrieval stage to extract entities from the actual query text, instead of relying only on whatever vector search returns.

---

## Architecture overview

### Ingestion

When a memory is added:

1. the document is saved in Neo4j with its embedding
2. the document is linked to a collection
3. entity extraction runs in the background
4. the graph phases join it: extracted entities are deduplicated and merged across documents
5. `MENTIONS` edges are created from documents to entities
6. `RELATED_TO`-style edges are created between entities
7. `SIMILAR_TO` document relationships are generated from vector similarity

### Retrieval

The recall pipeline combines three signals:

- **Direct entity recall**: documents mentioning the same entities found in the query
- **Related entity recall**: documents linked through entity relationships in the graph
- **Vector recall**: standard semantic nearest-neighbor search on document embeddings

Results found by both graph and vector phases receive a boost during reranking.

---

## Graph phases of the ingestion

The core RabbitHole ingests every source through recorded **phases** (`parsing_chunking`, then `embedding`, plus the phases of the plugins), keeps a diary of each one per source, resumes an ingestion interrupted by a crash and runs a phase again when its settings change (see the `Ingestion` section of the Grinning Cat README). Where the vector memory of the agent is GraphRAG, this plugin adds **two** phases through the `rabbithole_ingestion_phases` hook (`phases.py`):

| Phase | Runs after | Does | Tag (what makes it run again) |
|---|---|---|---|
| **`graph_spacy_extraction`** | `embedding` | purges the graph the source contributed, then writes its spaCy NER + entity linking graph in one transaction and recreates its similarity edges | spaCy models, `enable_entity_extraction`, `enable_concept_relations` |
| **`graph_llm_extraction`** | `graph_spacy_extraction` | extracts the concepts and relations with the agent's LLM | `enable_concept_relations` |

How a source goes through them:

- **upload / re-upload**: the embedding phase stores the points; `add_point_to_tenant` starts the per-chunk extraction in the background; the spaCy phase joins it, so the source's graph is written **once**. A re-upload replaces the points, and the spaCy phase replaces the graph: nothing of the previous content survives (the extraction of points the source no longer has is left out; a current point without its extraction makes the phase re-extract every chunk).
- **embedder change**: the core runs the embedding again on the kept chunks, then both graph phases. The handler adopts the current system embedder before the extraction starts, so entity embeddings never use a stale one.
- **settings change**: a change of the spaCy models or of a toggle moves the tag, and the core's automatic resume runs the phases again on every completed source (a source in error waits for a new upload or the maintenance, by the core's choice). The **technology patterns** and the **concept definitions / relation definitions / prompt** are deliberately out of the tags: saving them already triggers a dedicated job (`refresh_technology_entities`, and the concept recompute with its atomic generation flip), so a per-source re-run would repeat the same work, LLM calls included. `enable_concept_relations` is in the spaCy tag too: the spaCy phase owns the purge of the source's graph, concepts included, so turning the concepts off removes them.
- **crash / restart**: the core resumes the source from the phase that was running; with no extraction task in memory, the spaCy phase re-extracts every stored chunk (purge first, so the result is the same).
- **deletion**: both graph phases remove the graph the source contributed (`purge_source_graph_artifacts`); the documents go with the points of the source.
- **switch of the vector memory to GraphRAG**: the transfer writes the points outside any ingestion, which makes the graph phases of every transferred source due (failed ones included) and reruns them; once the transfer is done, `after_vector_memory_transfer_on_agent` brings every source up to date as a safety net, without rebuilding the ones already done; switching away from GraphRAG wipes the agent's graph.

Outcomes: `done` (the spaCy phase reports `processed` = entities + relations, or an `empty-extraction` warning when the content has none), `skipped` (the LLM phase with `enable_concept_relations` off), `not_ready` for a transient LLM outcome (retried by the core within the phase's 300 s budget), `error` otherwise — including a source without stored points, which the embedding phase always writes before. A failed spaCy phase fails the LLM phase after it (the core marks the dependents of a failed phase).

**Sources are scoped by name AND chat**: a file with the same name in the knowledge base and in a chat are two sources. The points read by the phases, their background tasks and the purge are limited to the collection (declarative / episodic) and the chat of the source. The `source_files` provenance of the `RELATED_TO` relations is scoped too: the name for a knowledge base file, `chat:<chat_id>:<name>` for a chat file. Every deletion strips only its own scope: a source deleted, a chat destroyed (all its keys), a whole collection deleted (the knowledge base keys, or all the chat keys). Relations written by an earlier version for a chat file carry the name alone, as if they came from the knowledge base: on upgrade, run `--graph` (maintenance CLI) on the agents with files uploaded in chats, which rebuilds them scoped. Deleting another collection (e.g. `procedural`, at every plugin toggle) strips no provenance.

### Concept and relation extraction (agent-level LLM)

The LLM phase extracts `concept` nodes and typed `RELATED_TO` relations using the **agent's own LLM** (`cat.llm`, resolved per agent from that agent's saved `llm` configuration) — it is **not** a system-level service, so two agents can use different models and prompts.

Three per-agent settings drive it:

- **`concept_definitions`** — one `TYPE: definition` per line; each key becomes an admissible concept type (uppercased) and is interpolated into `{concept_definitions}`.
- **`relation_definitions`** — one `TYPE: definition` per line; each key becomes an admissible relation type and is interpolated into `{relation_definitions}`.
- **`concept_relations_prompt`** — the prompt template; the document text is appended after it. Empty falls back to the built-in default.

Each structured LLM call is bounded by a wall-clock deadline, `CAT_INGESTION_LLM_CALL_TIMEOUT_SECONDS` (default `180` seconds; invalid/non-positive values fall back to the default), and runs with streaming disabled (`stream=False`); when the LLM's base URL is OpenRouter, reasoning is also turned off through a `reasoning: {"enabled": false}` override in `extra_body` (other providers reject that field, so they never receive it). Both apply per call only — the agent's LLM config is never changed. A stalled streaming socket therefore becomes `not_ready` on the first attempt (the phase's 300 s not-ready budget governs retries) and then `error` naming the timeout, never an unbounded `processing` phase.

The fingerprint also includes the **agent-level LLM identity** (`<config class name>:<model>`, resolved from the agent's `llm` category) and the `enable_concept_relations` toggle, because both change what the phase produces. `enable_student_knowledge_graph` and the display-only `enable_knowledge_graph` are deliberately excluded. `_concept_fingerprint` is SHA-256 over the parsed definitions + the whitespace-stripped prompt + the LLM identity + the toggle. When any material input changes, the fingerprint changes: the next ingestion tags its concepts with it, and saving the definitions or the prompt re-runs the extraction for the existing documents and flips the active generation. A change of the agent's LLM fires no hook: before a recall or an ingestion the plugin checks, once per agent and fingerprint in each process, whether concepts of another generation are still visible (one tenant-scoped Neo4j read) and then starts the same recompute. An agent never recomputed, whose concepts are those of the current fingerprint, triggers nothing. The recompute runs in background on an agent of its own (the end of the request that noticed the change does not stop it) and holds a Redis lock per agent and generation (2 h, expiring on its own if the pod dies): two pods noticing the same change pay the LLM pass once, and the pod finding the lock taken checks again after 10 minutes. A recompute that fails, is cancelled or finds the agent gone is checked again at the next recall. A concept the recompute finds again takes the new generation as soon as it is found (like its relations), so the flip keeps it visible and the cleanup keeps it; between that moment and the flip it is hidden, as its relations already were.

Writes are **generation-versioned**: every concept node/edge is tagged with the fingerprint as its `concept_gen`, the tenant's active generation is persisted on its `Collection` node (`concept_gen_active`), and superseded generations are hidden by the retrieval gate and garbage-collected. The stale-concept GC only deletes nodes marked inactive **and** with no remaining `MENTIONS` edge, so shared spaCy-owned entities survive a generation flip.

Toggles and phase markers:

- The document-structure graph (`:SourceFile`, `[:PART_OF]`, `[:NEXT]`, `[:HAS_SUMMARY]`) is **always built**.
- **`enable_concept_relations`** is the only switch for the LLM concept/relation phase; with it off, that phase is `skipped` and writes nothing. **`enable_knowledge_graph`** is display-only: the frontend UI uses it, but it does not gate any backend write or retrieval. **`enable_student_knowledge_graph`** is display-only as well. The spaCy phase independently honours `enable_entity_extraction`.
- The tags of the two phases are listed in the table of the graph phases above; the concept generation (`_concept_fingerprint`) is a separate token, tagging the concept writes.

### Shared Entity-node schema (internal contract)

Both phases write to the SAME `Entity` nodes, which is why the LLM phase is NOT extracted into a separate plugin. The contract is internal to this plugin:

- the spaCy phase (`_write_source_graph`) writes shared Entity nodes and `REMOVE`s the `concept_gen` / `concept_gen_active` tags from them;
- the LLM concept-relations step tags its own Entity nodes with `concept_gen`;
- the stale-concept GC (`_gc_stale_concept_nodes`) only deletes nodes marked `concept_gen_active = false` AND with no remaining `MENTIONS` edge — the `NOT (e)<-[:MENTIONS]-()` guard protects shared spaCy nodes, so an entity that spaCy picked up keeps its `MENTIONS` edge and is never deleted when the LLM generation flips.

Because the two phases coordinate on the same node schema, they must live in the same plugin.

### Configuration scope

Only two services are **system-level** in Cat; everything else is **per-agent**:

| Setting / service | Scope | Notes |
|---|---|---|
| `embedder` | **system** | embedding model / dimension |
| `llm` (agent LLM) | per-agent | used by concept/relation extraction |
| `chunker` | per-agent | |
| vector store (`Neo4jGraphRAGConfig` / `GraphRAGHandler`) | per-agent | |
| concept definitions / relation definitions / prompt / `enable_concept_relations` / agent `llm` | per-agent | drive `_concept_fingerprint` |

## Requirements

Before using this plugin, make sure you have:

- a working **Grinning Cat** installation
- the `base_plugin` dependency available in your Cat environment
- a running **Neo4j** instance
- a Neo4j version with **vector index support** available
- the Python dependencies declared in `pyproject.toml`
- at least one installed **spaCy model**

### Python dependencies

From this repository:

```bash
uv sync
```

### Install a spaCy model

The default configuration uses `en_core_web_lg`.

```bash
python -m spacy download en_core_web_lg
```

If the configured model is unavailable, the extractor attempts to fall back to `en_core_web_sm`.

Only standard statistical spaCy pipelines (e.g. `en_core_web_sm/lg`, `it_core_news_lg`) are supported; `*_trf` transformer pipelines are not supported and `spacy-transformers` is not a dependency.

---

## Installation

Add this plugin to your Grinning Cat plugins environment, then ensure the Python dependencies and spaCy model are installed.

A typical setup is carried out by using the [Admin UI](https://github.com/matteocacciola/grinning-cat-admin), or by
using the proper endpoint exposed by the Grinning Cat API.

---

## Configuration

The plugin exposes a `Neo4jGraphRAGConfig` settings model.

### Connection settings

| Setting          | Default                  | Description          |
|------------------|--------------------------|----------------------|
| `neo4j_uri`      | `neo4j://localhost:7687` | Neo4j connection URI |
| `neo4j_user`     | `neo4j`                  | Neo4j username       |
| `neo4j_password` | `None`                   | Neo4j password       |
| `neo4j_database` | `neo4j`                  | Neo4j database name  |

### Vector index settings

| Setting                       | Default               | Description                              |
|-------------------------------|-----------------------|------------------------------------------|
| `document_vector_index`       | `document_embeddings` | Name of the document vector index        |
| `entity_vector_index`         | `entity_embeddings`   | Name of the entity vector index          |
| `vector_similarity_threshold` | `0.7`                 | Minimum score for vector matches         |

### Entity extraction settings

| Setting                     | Default          | Description                                         |
|-----------------------------|------------------|-----------------------------------------------------|
| `enable_entity_extraction`  | `True`           | Enables spaCy-based entity extraction               |
| `enable_entity_embeddings`  | `False`          | Enables entity embeddings and entity vector index   |
| `enable_entity_expansion`   | `True`           | Enables graph-aware expansion during recall         |
| `spacy_models`              | `{"en": "en_core_web_lg"}` | spaCy model per language, used for extraction |
| `extra_technology_patterns` | `[]`             | Extra regex patterns for domain-specific tech terms |

### Graph retrieval settings

| Setting                 | Default | Description                   |
|-------------------------|---------|-------------------------------|
| `graph_retrieval_depth` | `2`     | Maximum graph traversal depth |
| `graph_decay_factor`    | `0.8`   | Score decay applied per hop   |

### Performance and maintenance

| Setting                 | Default | Description                     |
|-------------------------|---------|---------------------------------|
| `connection_pool_size`  | `50`    | Neo4j connection pool size      |
| `save_memory_snapshots` | `False` | Enables snapshot backup support |

### Graph building and LLM extraction

| Setting | Default | Description |
|---|---|---|
| `enable_knowledge_graph` | `False` | Display-only frontend flag; no backend extraction/retrieval gating |
| `enable_concept_relations` | `False` | Enables LLM concept/relation extraction (the only concept-write gate) |
| `concept_definitions` | built-in RITA defaults | Admissible concept types, one `TYPE: definition` per line |
| `relation_definitions` | built-in RITA defaults | Admissible relation types, one `TYPE: definition` per line |
| `concept_relations_prompt` | built-in template | Prompt sent to the agent LLM; document text is appended |
| `enable_student_knowledge_graph` | `False` | Frontend-visualisation flag only; no backend extraction/retrieval gating |

---

## Custom technology patterns

Besides spaCy NER, the plugin also uses regex patterns to catch technology names that NER models often miss.

Built-in patterns already cover terms such as:

- Neo4j
- MongoDB
- PostgreSQL
- Docker
- Kubernetes
- Python
- TypeScript
- LangChain
- spaCy
- RAG / GraphRAG / LLM / GPT
- AWS / Azure / GCP

You can extend this list with `extra_technology_patterns`, for example:

```python
[
    r"\\b(MyFramework|MyTool|InternalPlatform)\\b",
    r"\\b(CatOps|KnowledgeMesh)\\b",
]
```

This is especially useful for:

- internal project names
- domain-specific technologies
- non-English technical vocabulary
- branded tools that spaCy may not recognize reliably

---

## Data model

The graph is centered around a few core concepts:

- **`Collection`**: a logical memory collection for a Cat tenant
- **`Document`**: a stored memory chunk with content, metadata, and embedding
- **`Entity`**: a normalized concept extracted from documents

Main relationships:

- **`BELONGS_TO`**: document → collection
- **`MENTIONS`**: document → entity
- **`RELATED_TO`**: entity ↔ entity semantic relation
- **`SIMILAR_TO`**: document ↔ document similarity relation

Entities are hashed using tenant ID, entity type, and normalized name so they can be reused across multiple documents.

---

## Entity and relation extraction

The extraction pipeline supports:

- standard spaCy named entities
- technology detection via regex
- entity deduplication by normalized name
- relation extraction from dependency parsing
- fallback relation inference from local text proximity
- optional co-occurrence links for small entity sets

Supported entity categories include:

- `PERSON`
- `ORGANIZATION`
- `TECHNOLOGY`
- `CONCEPT`
- `LOCATION`
- `DATE`
- `PRODUCT`
- `EVENT`
- `FINANCIAL`
- `UNKNOWN`

---

## Notes on collections and deletion

Collections are isolated by tenant and collection name.

When a collection is deleted:

- its collection node is removed
- its documents are deleted
- orphaned entities are pruned only if they are not mentioned anywhere else

This preserves shared knowledge across collections whenever entities are still referenced.

---

## Operational notes

### Embedding dimension must match

The plugin validates that the running embedder size matches the configured Neo4j vector index dimension.

If you change embedding model size, you may need to recreate the collection or rebuild indexes.

### Neo4j is treated as a remote database

This handler is designed for a remote database backend and uses the Neo4j async driver.

### Background tasks are used during ingestion

Entity extraction and similarity-link creation are scheduled asynchronously when new memories are added; the `graph_spacy_extraction` phase joins the extraction of its source and writes it once. The CPU-heavy steps of an ingestion (the spaCy analysis of a chunk, the embedding of its entities, the re-embedding of the tenant on a change of the embedder) run on the core's ingestion lane (`run_in_ingestion_executor`: `CAT_INGESTION_WORKERS` de-prioritized threads, `CAT_INGESTION_NICENESS`), never on the threads serving the chats (with the lane disabled, `CAT_INGESTION_WORKERS<=0`, they share the default executor, as the core's own parsing and embedding do); the analysis of a chat message and the one-off loading of a spaCy model stay off it. Points written outside an ingestion get their graph too, told apart by the source's state row: a source with a row (e.g. a cloned agent, whose rows are copied completed or in error, or a vector-database transfer) has its graph phases made due again at once and rerun through the core by a background worker of the plugin (if the pod dies first, the core's automatic resume rebuilds a completed one); a source without a row (e.g. a memory dump, `store_documents`, a point of the memory API, whose source is the user by default) is rebuilt whole by that worker, on an agent of its own, one source at a time and under the core's lock of the source (a burst of writes to one source is rebuilt once). That rebuild is best effort: with no state, a pod dying meanwhile loses it, and a later change of the spaCy or concept settings does not rebuild it. The procedures get no graph; a point without a source gets its similarity edges only.

---

## Limitations and practical considerations

A few things are worth knowing before production use:

- the plugin is optimized for **graph-aware recall**, not as a general-purpose graph management UI
- vector indexes must be compatible with the embedding dimension produced by your Cat embedder
- spaCy model quality strongly affects entity extraction quality
- relation extraction is heuristic by nature and may need tuning for specialized domains
- enabling entity embeddings increases storage usage
- large memory imports may create many background extraction tasks
- the `save_dump` method is not implemented; Neo4j-native backup tools should be used instead
- hybrid collections are currently not supported

---

## Repository structure

| File | Purpose |
|---|---|
| `main.py` | Hook registration: query capture, embedder injection, graph phase registration, transfer cleanup, provenance reconciliation of a new agent, concept recompute on save and on an LLM change |
| `phases.py` | The two graph phases added to the core's ingestion (`graph_spacy_extraction`, `graph_llm_extraction`) |
| `graphrag_handler.py` | `GraphRAGHandler` core logic + `Neo4jGraphRAGConfig` settings model |
| `entity_extractor.py` | spaCy-based entity and relation extraction pipeline |
| `constants.py` | Entity mappings, regex technology patterns, relation verb map, default concept/relation definitions |
| `models.py` | Pydantic models for entities and extracted relations |
| `epoch.py` / `versioning.py` | Generation-token helpers and versioned read/write decorators |
| `structured_llm.py` | Structured JSON output helpers for the agent LLM |
| `plugin.json` | Plugin metadata for Grinning Cat |
| `pyproject.toml` | Python dependencies |
| `tests/` | Standalone, import-safe behavioral harnesses (run as scripts) |
| `scripts/` | `maintenance_agent.py` and `cleanup_null_vectors.py` utilities |

### Maintenance CLI

`maintenance_agent.py` runs inside the Cat container, as a process of its own,
and drives the core's public ingestion API (`RabbitHole.reingest_source` /
`rerun_source`): every source goes through the same phase engine as an upload,
under the lock of the source, with its state updated. Scope is `--agent <id>`
(repeatable; an explicit id is the only way to target a protected/`__*__`
agent) or `--all` (protected agents always skipped). Ops are combinable and run
in this order:

| Op | What runs again |
|---|---|
| `--reingest` | every phase of every source, from the bytes kept for it |
| `--reembed` | the embedding of every source, on its stored chunks, and the phases depending on it (all invalidated together: if the CLI dies midway, the automatic resume still finds them to do) |
| `--resume-failed` | the failed phases of every source in error, with the phases depending on them (the completed ones are kept), as the core's `POST /ingestion/retry`; a source in error with phases still to do (e.g. left by a `--graph` that died) runs them; a source that failed before any phase (e.g. a download) is reported: it needs a new upload |
| `--graph` | GraphRAG agents: wipe of the tenant graph, then the graph phases of every source and the alignment of the concept generation marker; it runs whatever the concept toggle |

`--collection declarative` (default: the knowledge base) or `episodic` (the
sources uploaded in the chats) picks the sources; `--graph` always covers both,
since its wipe is tenant-wide. A source ingested before the ingestion states
existed (points in the vector memory, no state) has no stored chunks: `--reingest`,
`--reembed` and `--graph` reingest it, which gives it a state. A source whose
bytes were not kept (a file ingested without storing it) cannot be reingested:
it is reported `source-not-kept` (exit 1) and left untouched; under `--graph`
its graph is wiped like every other and cannot be rebuilt (its text and
vectors stay).

```bash
# Re-embed only (the stored chunks are reused) for one agent
docker exec cheshire_cat_core python /app/cat/plugins/cat_graphrag/scripts/maintenance_agent.py --agent <id> --reembed --yes

# Re-ingest from scratch for all agents
docker exec cheshire_cat_core python /app/cat/plugins/cat_graphrag/scripts/maintenance_agent.py --all --reingest --yes

# Run again ONLY the failed phases
docker exec cheshire_cat_core python /app/cat/plugins/cat_graphrag/scripts/maintenance_agent.py --agent <id> --resume-failed --yes

# Rebuild the GraphRAG graph
docker exec cheshire_cat_core python /app/cat/plugins/cat_graphrag/scripts/maintenance_agent.py --agent <id> --graph --yes

# Show the plan without touching anything (exits 2)
docker exec cheshire_cat_core python /app/cat/plugins/cat_graphrag/scripts/maintenance_agent.py --all --resume-failed --dry-run
```

Safety contract: `--dry-run` prints the plan (per agent and op the steps, and
the sources in error with their failed phases for `--resume-failed`) and exits
**2** with zero writes and without loading any agent; without `--yes` the same
plan is printed and the run aborts with **2**. Agents are processed one at a
time; a failing source never stops the others and a failing op never stops the
next one. The outcome of every op counts the sources completed, the ones deleted
meanwhile (`gone`) and the failures (with the failed phases and their reasons);
`--json` prints the plan, every op outcome and the final summary as one JSON
document per line.

`--verify` (read-only) checks that every source of the collection is completed
with the current tag of every phase (nothing left for the automatic resume) and,
for a GraphRAG agent, that every Document has the embedding of the current
generation, that no entity is left without edges and that the visible concept
generation is the one of the current settings.

Exit codes: **0** = every op succeeded (and `--verify` passed), **1** = partial
failure (a source or an op failed, or a `--verify` check failed), **2** =
aborted (dry-run, missing `--yes`, unknown agent, or `--graph` without the
plugin deployed).

### Maintenance script coordination

While an agent is processed it is in maintenance
(`crud_ingestions.set_maintenance`, TTL 2h, refreshed before every source;
cleared in a `finally`, on success or failure): the automatic resume of the
server leaves the agent alone meanwhile. If the CLI process dies the marker
expires on its own, then the resume takes over what it left in flight.
`--graph` resets the graph entries of the diaries BEFORE its wipe: if it dies
or fails at any later point the sources are stale, and the resume runs their
graph phases again (sources without a state are not in the resume: run
`--graph` again). An ingestion of the server running on a source meanwhile is
waited for (the lock of the source). A save of the concept settings during
`--graph` flips the concept marker on its own: the CLI then does not bring the
older generation back and reports `concept-marker-conflict` (exit 1).

---

## Best fit use cases

This plugin is especially useful when your Grinning Cat needs to work with:

- technical documentation
- project knowledge bases
- product or team memory
- entity-rich conversations
- domain knowledge with many cross-references
- retrieval tasks where graph structure matters as much as semantics

---

## Author

Created by **Matteo Cacciola**.

Plugin metadata name: **Neo4j GraphRAG Advanced**

---

## Known Limitations

### Technology pattern list is English-centric

The built-in `TECHNOLOGY_PATTERNS` list targets English proper nouns (framework names, cloud providers, programming
languages). It will miss technology terms written in other scripts or domain-specific jargon outside the list.

Two mitigations are available:

- **Use a multilingual spaCy model** — models such as `xx_ent_wiki_sm` cover named entities across many languages.
  The NER component is language-agnostic; only the regex patterns are English-centric.
  Set `spacy_models` in the plugin configuration to switch model.
- **Extend the regex list** — use `extra_technology_patterns` to add domain-specific or non-English terms at
  configuration time.

For knowledge bases in languages other than English, combining a multilingual model with a curated
`extra_technology_patterns` list gives the best coverage.


---

## Final note

If your Grinning Cat should reason over memories like a connected knowledge space instead of a flat vector pile, this
plugin gives it sharper recall, richer context, and a more structured memory backbone.
