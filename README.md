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
- ingestion hooks: embedder injection, the classic `after_rabbithole_stored_documents` graph writer, the optional EffING phase registration (`ingestion_phase_specs`, `ingestion_phase_settings_marker`, `ingestion_phase_run`, `before_ingestion_status_completed`)
- a query hook that captures the raw user message before memory recall
- maintenance hooks: boot-time provenance reconciliation, and old-store cleanup on vector-memory switch (`after_vector_memory_transfer_on_agent`)

The query hook is important: it allows the retrieval stage to extract entities from the actual query text, instead of relying only on whatever vector search returns.

---

## Architecture overview

### Ingestion

When a memory is added:

1. the document is saved in Neo4j with its embedding
2. the document is linked to a collection
3. entity extraction runs in the background
4. extracted entities are deduplicated and merged across documents
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

## Ingestion modes: classic vs efficient phase machine

The plugin can build the graph in two mutually exclusive ways, decided per ingestion by `_phase_path_active(cat)`:

| [`cat_efficient_ingestion`](https://github.com/DEH-RAG/cat-efficient-ingestion) installed | SYSTEM-level `ingestion` engine | Path used |
|---|---|---|
| no | any | **classic** |
| yes | core engine | **classic** |
| yes | `EfficientIngestionConfiguration` | **phase machine** |

The `ingestion` engine is one of only two system-level services in Cat (the other is `embedder`), so the decision is read against the system settings, never the agent's. When the phase machine is active, the classic `after_rabbithole_stored_documents` callback becomes a no-op, so graph work never runs twice; when it is inactive, the classic callback is the only writer.

### Phase registration (EffING)

The plugin registers **exactly two** phases through the `ingestion_phase_specs` accumulator, in this order:

- **`graph_spacy_extraction`** — spaCy NER + entity linking + similarity edges.
- **`graph_llm_extraction`** — LLM concept/relation extraction.

Both declare `settings_category=None` (the material marker *is* the settings fingerprint) and `depends_on=("parsing_chunking", "embedding")` (the graph work needs the stored chunks and valid vectors). `ingestion_phase_settings_marker` returns `_spacy_fingerprint` for the spaCy phase and `_concept_fingerprint` for the LLM phase; `ingestion_phase_run` dispatches each phase to its awaited executor (`run_spacy_extraction_phase` / `run_llm_extraction_phase`), returning `{"status": "done"}` or `{"status": "not_ready", "retry_after": 5}` while the source's points are still absent. `before_ingestion_status_completed` is the completion gate: it fails the ingestion hard if live background graph tasks remain for the source.

A graph phase is considered stale when the dependencies' marker identity or the plugin's own fingerprint changes; EffING computes this itself (no timestamps, no full settings dumps, no raw Redis). Re-running is always safe: every write is an idempotent MERGE.

### Concept and relation extraction (agent-level LLM)

The LLM phase extracts `concept` nodes and typed `RELATED_TO` relations using the **agent's own LLM** (`cat.llm`, resolved per agent from that agent's saved `llm` configuration) — it is **not** a system-level service, so two agents can use different models and prompts.

Three per-agent settings drive it:

- **`concept_definitions`** — one `TYPE: definition` per line; each key becomes an admissible concept type (uppercased) and is interpolated into `{concept_definitions}`.
- **`relation_definitions`** — one `TYPE: definition` per line; each key becomes an admissible relation type and is interpolated into `{relation_definitions}`.
- **`concept_relations_prompt`** — the prompt template; the document text is appended after it. Empty falls back to the built-in default.

The fingerprint also includes the **agent-level LLM identity** (`<config class name>:<model>`, resolved from the agent's `llm` category) and the `enable_concept_relations` toggle, because both change what the phase produces. `enable_student_knowledge_graph` and the display-only `enable_knowledge_graph` are deliberately excluded. `_concept_fingerprint` is SHA-256 over the parsed definitions + the whitespace-stripped prompt + the LLM identity + the toggle. When any material input changes, the fingerprint changes; the next ingestion recomputes concept extraction, and a save-triggered recompute re-runs it for existing documents.

Writes are **generation-versioned**: every concept node/edge is tagged with the fingerprint as its `concept_gen`, the tenant's active generation is persisted on its `Collection` node (`concept_gen_active`), and superseded generations are hidden by the retrieval gate and garbage-collected. The stale-concept GC only deletes nodes marked inactive **and** with no remaining `MENTIONS` edge, so shared spaCy-owned entities survive a generation flip.

Toggles and phase markers:

- The document-structure graph (`:SourceFile`, `[:PART_OF]`, `[:NEXT]`, `[:HAS_SUMMARY]`) is **always built**. The `enable_derived_graph` key is retired: saved configs that still contain it are accepted and ignored, and nothing reads derived edges.
- **`enable_concept_relations`** is the only switch for the LLM concept/relation phase; with it off, that phase still completes but does no extraction. **`enable_knowledge_graph`** is display-only: the frontend UI uses it, but it does not gate any backend write or retrieval. **`enable_student_knowledge_graph`** is display-only as well. The spaCy phase independently honours `enable_entity_extraction`.
- **`graph_llm_extraction` marker** — SHA-256 over the parsed `concept_definitions` + `relation_definitions` + whitespace-stripped `concept_relations_prompt` + the agent-level LLM identity (`<config class name>:<model>`) + `enable_concept_relations`. Display-only toggles are excluded because they cannot change the produced concepts.
- **`graph_spacy_extraction` marker** — SHA-256 over the normalized `spacy_models` map + whitespace-stripped `extra_technology_patterns` + `enable_entity_extraction`.


### One extraction per document (empty-pass discriminator)

The efficient engine stores each chunk twice: first with the empty mapping `vector={}` (parsing/chunking), then again with real vectors after embedding. Graph extraction is vector-independent, so the plugin gates it on the **pass**, not on vector validity. `_is_empty_vector_staging_pass` requires BOTH the intrinsic empty mapping `vector={}` (only `_store_empty_vectors` / the efficient image builder writes it) AND the source's recorded `parsing_chunking` phase (`embedding` -> build). The empty mapping is what distinguishes the staging pass from a classic write: the core writes the `List[float]` returned by the embedder (a zero-norm list on fallback) and the multimodal hook either embeds an image or drops it, so `{}` never occurs on the classic path. The phase machine's `resume_owner` lease is deliberately not used — the host never clears it on a terminal status, so a classic re-ingest after an efficient ingest inherited `resume_owner` + `parsing_chunking` and the graph was silently skipped. This keeps the graph work at **one extraction per document** instead of two, while a legitimate payload-only / zero-vector document (e.g. an image under a non-multimodal embedder) still builds the graph exactly once. Vector validity is still required only for the similarity edges, where a zero vector is meaningless.

### Shared Entity-node schema (internal contract)

Both phases write to the SAME `Entity` nodes, which is why the LLM phase is NOT extracted into a separate plugin. The contract is internal to this plugin:

- the spaCy NER step (`_extract_and_link_entities`) writes shared Entity nodes and `REMOVE`s the `concept_gen` / `concept_gen_active` tags from them;
- the LLM concept-relations step tags its own Entity nodes with `concept_gen`;
- the stale-concept GC (`_gc_stale_concept_nodes`) only deletes nodes marked `concept_gen_active = false` AND with no remaining `MENTIONS` edge — the `NOT (e)<-[:MENTIONS]-()` guard protects shared spaCy nodes, so an entity that spaCy picked up keeps its `MENTIONS` edge and is never deleted when the LLM generation flips.

Because the two phases coordinate on the same node schema, they must live in the same plugin.

### Configuration scope

Only two services are **system-level** in Cat; everything else is **per-agent**:

| Setting / service | Scope | Notes |
|---|---|---|
| `ingestion` engine | **system** | `DEFAULT_SYSTEM_KEY`; decides classic vs phase path |
| `embedder` | **system** | embedding model / dimension |
| `llm` (agent LLM) | per-agent | used by concept/relation extraction |
| `chunker` | per-agent | |
| vector store (`Neo4jGraphRAGConfig` / `GraphRAGHandler`) | per-agent | |
| concept definitions / relation definitions / prompt / `enable_concept_relations` / agent `llm` | per-agent | drive `_concept_fingerprint` |

### Optional dependency

`cat_efficient_ingestion` is **optional**. Without it the plugin runs the classic path unchanged; the phase hooks are import-safe and no-op when EffING is absent. Deploying EffING and selecting its engine enables the phase path; no plugin reconfiguration beyond that is needed.

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

Entity extraction and similarity-link creation are scheduled asynchronously when new memories are added.

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
| `main.py` | Hook registration: query capture, embedder injection, classic graph writer, EffING phase hooks, transfer cleanup, boot reconciliation |
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
