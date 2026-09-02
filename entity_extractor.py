import asyncio
import hashlib
import re
from langdetect import DetectorFactory, detect_langs
from cat import log
from typing import List, Dict, Tuple
from spacy import load as spacy_load
from spacy.util import is_package as spacy_is_package
from spacy.cli.download import download as spacy_download
from spacy.language import Language
from spacy.tokens import Doc

from .constants import (
    SPACY_TO_ENTITY_TYPE,
    TECHNOLOGY_PATTERNS,
    MAX_CO_OCCURRENCE_ENTITIES,
    VERB_TO_RELATION_TYPE,
    SPACY_MAX_SEGMENT_CHARS,
    SPACY_PIPE_BATCH_SIZE,
)
from .models import ExtractedEntity, ExtractedRelation, EntityType, DocumentWithEntities


# ---------------------------------------------------------------------------
# Module-level spaCy model cache
#
# Rationale: spaCy models are read-only after loading — they carry no
# per-tenant state.  Sharing a single Language object across all
# EntityExtractor instances (i.e. all CheshireCat tenants) is therefore
# safe and avoids loading the same large binary multiple times.
#
# The cache is bounded by the number of *distinct* spaCy model names that
# are actually requested, which in practice is a small, fixed set regardless
# of how many CheshireCat instances are running.
#
# _SPACY_REGISTRY_LOCK  – serialises creation of per-model locks so that
#                          two coroutines cannot both decide "this lock does
#                          not exist yet" and create it simultaneously.
# _SPACY_MODEL_LOCKS    – one asyncio.Lock per model name; ensures that when
#                          the model is not yet cached, exactly one coroutine
#                          loads it while all others wait and then reuse it.
# _SPACY_MODEL_CACHE    – the actual model store, populated lazily.
# ---------------------------------------------------------------------------

_SPACY_MODEL_CACHE: Dict[str, Language] = {}
_SPACY_MODEL_LOCKS: Dict[str, asyncio.Lock] = {}
_SPACY_REGISTRY_LOCK = asyncio.Lock()


async def _get_or_load_model(model_name: str) -> Language:
    """Return the cached spaCy Language for *model_name*, loading it if needed.

    Uses double-checked locking so that:
    * after the first successful load, all subsequent callers take the fast
      path (no lock acquisition);
    * during the first load, concurrent callers block on a per-model lock
      rather than all attempting to load simultaneously.
    """
    # Fast-path: model already in cache
    if model_name in _SPACY_MODEL_CACHE:
        return _SPACY_MODEL_CACHE[model_name]

    # Ensure a dedicated lock exists for this model name
    async with _SPACY_REGISTRY_LOCK:
        if model_name not in _SPACY_MODEL_LOCKS:
            _SPACY_MODEL_LOCKS[model_name] = asyncio.Lock()
        model_lock = _SPACY_MODEL_LOCKS[model_name]

    # Double-checked locking: re-test inside the per-model lock
    async with model_lock:
        if model_name in _SPACY_MODEL_CACHE:
            return _SPACY_MODEL_CACHE[model_name]

        # Download if needed, then load — both are blocking operations
        if not spacy_is_package(model_name):
            await asyncio.to_thread(spacy_download, model_name)

        nlp: Language = await asyncio.to_thread(spacy_load, model_name)
        _SPACY_MODEL_CACHE[model_name] = nlp
        log.info(f"Loaded spaCy model '{model_name}' (globally cached)")
        return nlp


# ---------------------------------------------------------------------------
# Long-text segmentation
#
# spaCy's max_length guard (default 10**6) raises E088 for longer texts and
# parser/NER memory grows superlinearly with text length. Per the spaCy
# team's guidance for large documents, long inputs are split at paragraph
# boundaries into contiguous segments (each ≤ SPACY_MAX_SEGMENT_CHARS chars)
# processed with a single batched ``nlp.pipe`` call. Joining the segments
# reproduces the original text exactly (partition invariant), so character
# offsets stay global: each segment's local offsets are shifted by its start
# offset in the original text.
# ---------------------------------------------------------------------------


def _hard_split(chunk: str, cap: int) -> Tuple[str, str]:
    """Split *chunk* (``len(chunk) > cap``) into ``(head, rest)``.

    *head* is always ``<= cap`` chars. Prefers a whitespace-aware cut: the
    last space within ``[0.75 * cap, cap]``, keeping the space at the end of
    *head* (so *rest* does not start with whitespace). Falls back to a raw cut
    at ``cap`` when the window contains no space.
    """
    window = chunk[:cap]
    cut = window.rfind(" ", int(0.75 * cap))
    if cut == -1:
        cut = cap
    else:
        cut += 1  # keep the space with the head
    return chunk[:cut], chunk[cut:]


def _segment_text(text: str) -> List[Tuple[int, str]]:
    """Split *text* into contiguous segments ≤ SPACY_MAX_SEGMENT_CHARS chars.

    Returns a list of ``(start_offset, segment_text)`` pairs. The segment
    texts concatenated in order reproduce *text* exactly (partition
    invariant) and ``start_offset`` is the running cumulative length, i.e. the
    segment's position inside the original text.

    Split points prefer paragraph boundaries (double newlines; the separator
    is kept attached to the preceding paragraph). A single paragraph larger
    than the cap is hard-split (whitespace-aware, else raw).
    """
    cap = SPACY_MAX_SEGMENT_CHARS
    # re.split with a capturing group keeps the separators in the result, so
    # concatenating the chunks (and thus the final segments) is lossless.
    # Empty chunks (leading/trailing/only separators) are dropped safely:
    # they contribute nothing to the partition invariant.
    chunks = [c for c in re.split(r"(\r?\n\s*\r?\n)", text) if c]

    # Hard-split any chunk larger than the cap (recursively for the rest).
    parts: List[str] = []
    for chunk in chunks:
        while len(chunk) > cap:
            head, chunk = _hard_split(chunk, cap)
            parts.append(head)
        parts.append(chunk)

    # Greedily pack adjacent parts into segments of at most `cap` chars.
    segment_texts: List[str] = []
    for part in parts:
        if segment_texts and len(segment_texts[-1]) + len(part) <= cap:
            segment_texts[-1] += part
        else:
            segment_texts.append(part)

    # Offsets are running cumulated lengths → exact partition of the original.
    segments: List[Tuple[int, str]] = []
    offset = 0
    for segment_text in segment_texts:
        segments.append((offset, segment_text))
        offset += len(segment_text)
    return segments


def _run_segment_pipe(
    nlp: Language, segments: List[Tuple[int, str]], batch_size: int
) -> List[Tuple[Doc, int]]:
    """Run ``nlp.pipe`` over the segments, pairing each doc with its offset.

    Must be called from a worker thread: ``nlp.pipe`` is blocking CPU work and
    returns a generator that has to be consumed here. ``nlp.pipe`` yields one
    doc per input text, in input order, so ``zip(segments, docs)`` pairs each
    doc with the offset of the segment it was produced from.
    """
    docs = nlp.pipe([text for _, text in segments], batch_size=batch_size)
    return [(doc, offset) for (offset, _), doc in zip(segments, docs)]


class EntityExtractor:
    """
    Extracts entities and relations from text using spaCy.
    Supports multilingual models and extension with custom rules.

    spaCy models are loaded lazily on the first call to `ensure_initialized`
    and stored in a module-level cache so that multiple EntityExtractor
    instances (one per CheshireCat tenant) share the same Language objects
    in memory.  Each instance keeps a ``_nlps`` dict that maps language codes
    to the shared Language references — no per-instance copies are made.
    """

    def __init__(self, models: Dict[str, str], extra_technology_patterns: List[str] = None):
        """
        Initializes the EntityExtractor.

        Args:
            models: A dictionary mapping language codes (e.g. "en") to spaCy
                model names (e.g. "en_core_web_sm").  The special key
                ``"default"`` is added automatically and maps to
                ``"en_core_web_sm"`` unless already present.
            extra_technology_patterns: additional regex patterns appended to
                TECHNOLOGY_PATTERNS. Useful for non-English tech terms or
                domain-specific keywords not covered by the built-in list.
        """
        self._models = models
        self._models.setdefault("default", "en_core_web_sm")

        # Maps language code → shared Language object (populated by ensure_initialized)
        self._nlps: Dict[str, Language] = {}
        self._initialized = False
        # Per-instance lock: prevents duplicate initialisation when several
        # coroutines call ensure_initialized concurrently on the same instance.
        self._init_lock = asyncio.Lock()

        # Build the instance-level pattern list so it can be extended per-instance
        self._technology_patterns = list(TECHNOLOGY_PATTERNS)
        if extra_technology_patterns:
            self._technology_patterns.extend(extra_technology_patterns)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _download_spacy_model(model_name: str):
        """Synchronously downloads *model_name* if it is not yet installed."""
        if not spacy_is_package(model_name):
            spacy_download(model_name)

    @staticmethod
    def _detect_language(text: str) -> str | None:
        DetectorFactory.seed = 0
        text = text.strip()
        if len(text) < 5:
            return None
        try:
            langs = detect_langs(text)
            langs = [l.lang for l in langs if l.prob > 0.8]
            if len(langs) == 0:
                return None
            return langs[0]
        except Exception:
            return None

    async def ensure_downloaded(self):
        """Downloads all configured spaCy models (asynchronously)."""
        await asyncio.gather(*[
            asyncio.to_thread(self._download_spacy_model, model_name)
            for model_name in self._models.values()
        ])

    async def ensure_initialized(self):
        """Populates ``self._nlps`` from the global cache, loading models as needed.

        Safe to call concurrently: uses double-checked locking so that the
        actual model loading (which may be slow) happens at most once per
        unique model name across *all* EntityExtractor instances.
        """
        # Fast-path: already initialised for this instance
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            # Load all models in parallel; each individual model is protected
            # by its own lock inside _get_or_load_model, so concurrent calls
            # for the same model name across different instances are safe.
            nlps = await asyncio.gather(*[
                _get_or_load_model(model_name)
                for model_name in self._models.values()
            ])
            self._nlps = dict(zip(self._models.keys(), nlps))
            self._initialized = True

    # ------------------------------------------------------------------
    # Extraction API
    # ------------------------------------------------------------------

    async def extract_doc(self, text: str) -> Doc:
        if not self._initialized:
            await self.ensure_initialized()

        lang = self._detect_language(text)
        nlp = self._nlps.get(lang, self._nlps["default"]) if lang else self._nlps["default"]
        return nlp(text)

    async def extract(self, text: str, document_id: str, metadata: Dict = None) -> DocumentWithEntities:
        """
        Extracts entities and relations from text.

        Long texts are split into contiguous segments (paragraph-aligned,
        each ≤ SPACY_MAX_SEGMENT_CHARS chars) processed together with a single
        batched ``nlp.pipe`` call inside a worker thread — this keeps every
        processed text far below spaCy's max_length guard (E088) while the
        entity/relation offsets stay global (per-segment local offsets are
        shifted by the segment's start offset).
        """
        if not self._initialized:
            await self.ensure_initialized()

        # Process text with spaCy
        lang = self._detect_language(text)
        nlp = self._nlps.get(lang, self._nlps["default"]) if lang else self._nlps["default"]

        segments = _segment_text(text)
        # nlp.pipe is blocking CPU work (like the previous nlp(text)); the
        # generator is consumed inside the worker thread, never in the loop.
        pairs = await asyncio.to_thread(_run_segment_pipe, nlp, segments, SPACY_PIPE_BATCH_SIZE)

        # Extract entities (global offsets: per-segment local + segment start)
        extracted_entities = []
        for doc, offset in pairs:
            extracted_entities.extend(self.extract_entities(doc, offset=offset))

        # Add technology entities with regex (captures things spaCy might miss)
        extracted_entities.extend(self.extract_technologies_regex(text))

        # Deduplicate entities
        extracted_entities = self.deduplicate_entities(extracted_entities)

        # Extract relations
        relations = self._extract_relations(pairs, extracted_entities, text)

        # Add co-occurrence relations only for small entity sets.
        # N*(N-1)/2 pairs are too expensive for large documents.
        if len(extracted_entities) <= MAX_CO_OCCURRENCE_ENTITIES:
            relations.extend([
                ExtractedRelation(
                    source_entity=extracted_entities[i].name,
                    target_entity=extracted_entities[j].name,
                    relation_type="CO_OCCURS_WITH",
                    weight=(extracted_entities[i].confidence + extracted_entities[j].confidence) / 4
                )
                for i in range(len(extracted_entities))
                for j in range(i + 1, len(extracted_entities))
            ])

        return DocumentWithEntities(
            document_id=document_id,
            content=text,
            entities=extracted_entities,
            relations=relations,
            metadata=metadata or {}
        )

    # ------------------------------------------------------------------
    # Static / instance extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_entities(doc: Doc, offset: int = 0) -> List[ExtractedEntity]:
        """Extracts named entities from the spaCy document.

        ``offset`` shifts the entity char spans — used when *doc* was produced
        from a segment of a larger text so that the returned offsets stay
        global (see ``extract``); defaults to 0, keeping single-document calls
        (e.g. the query path in graphrag_handler) unchanged.
        """
        entities = []
        for ent in doc.ents:
            entity_type = SPACY_TO_ENTITY_TYPE.get(ent.label_, EntityType.UNKNOWN)
            entities.append(ExtractedEntity(
                name=ent.text.strip(),
                type=entity_type,
                start_char=ent.start_char + offset,
                end_char=ent.end_char + offset,
                confidence=0.9 if entity_type != EntityType.UNKNOWN else 0.5
            ))
        return entities

    def extract_technologies_regex(self, text: str) -> List[ExtractedEntity]:
        """Extracts technology entities using the instance-level regex pattern list.

        Complements spaCy NER for tech terms that statistical models often miss
        (framework names, acronyms, brand names).  The instance-level list can be
        extended at construction time via `extra_technology_patterns`.
        """
        return [
            ExtractedEntity(
                name=match.group(0),
                type=EntityType.TECHNOLOGY,
                start_char=match.start(),
                end_char=match.end(),
                confidence=0.85,
            )
            for pattern in self._technology_patterns
            for match in re.finditer(pattern, text, re.IGNORECASE)
        ]

    @staticmethod
    def deduplicate_entities(entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Deduplicates nearby or identical entities."""
        normalized = {}
        for ent in entities:
            name_lower = ent.name.lower().strip()

            if name_lower in normalized:
                # Keep the one with higher confidence
                existing = normalized[name_lower]
                if ent.confidence > existing.confidence:
                    normalized[name_lower] = ent
            else:
                normalized[name_lower] = ent

        return list(normalized.values())

    @staticmethod
    def _extract_relations(
        segments: List[Tuple[Doc, int]],
        entities: List[ExtractedEntity],
        full_text: str
    ) -> List[ExtractedRelation]:
        """
        Extracts relations between entities in two phases:

        1. **Dependency-parse phase** (primary, weight=0.85):
           Uses spaCy's dependency tree to find Subject-Verb-Object triples.
           Passive voice is supported via the "agent" arc
           (e.g. "Python was developed by Guido" → DEVELOPS(Guido, Python)).
           ``segments`` holds one ``(doc, offset)`` pair per text segment: each
           doc's local token spans (``tok.idx``, subtrees) are shifted by its
           segment offset so entity lookups match the global entity offsets.

        2. **Proximity-fallback phase** (secondary, weight=0.6):
           For pairs not already found by the parser, a 100-char sliding window
           catches implicit relations (enumerations, appositions, etc.) using
           lightweight keyword patterns. Already runs on global offsets + the
           full text, so it also catches pairs spanning segment boundaries.
        """
        relations: List[ExtractedRelation] = []

        if len(entities) < 2:
            return relations

        sorted_entities = sorted(entities, key=lambda e: e.start_char)

        # ── helpers ──────────────────────────────────────────────────────────

        def find_entity_for_span(start: int, end: int) -> ExtractedEntity | None:
            """Returns the first entity whose char-span overlaps [start, end)."""
            for ent in sorted_entities:
                if ent.start_char < end and start < ent.end_char:
                    return ent
            return None

        def find_entity_for_token(tok, offset: int) -> ExtractedEntity | None:
            """
            Tries to resolve a dependency-tree token to one of our entities.

            1. Checks the token's own character span (local ``tok.idx`` shifted
               by the segment ``offset`` into global coordinates).
            2. If not found, expands to the token's subtree (handles compound
               proper nouns like "New York" or "Apache Kafka").
               The expansion is capped at 5 tokens to avoid matching full clauses.
            """
            ent = find_entity_for_span(offset + tok.idx, offset + tok.idx + len(tok.text))
            if ent:
                return ent
            subtree = list(tok.subtree)
            if 2 <= len(subtree) <= 5:
                span_start = offset + subtree[0].idx
                span_end = offset + subtree[-1].idx + len(subtree[-1].text)
                return find_entity_for_span(span_start, span_end)
            return None

        # ── phase 1: SVO via dependency parser ───────────────────────────────
        # seen_pairs is shared across all segments: the same SVO triple found
        # in two segments yields a single relation.

        seen_pairs: set = set()

        for doc, offset in segments:
            for token in doc:
                if token.pos_ not in ("VERB", "AUX"):
                    continue

                # Grammatical subjects (active and passive)
                subjects = [
                    child for child in token.children
                    if child.dep_ in ("nsubj", "nsubjpass")
                ]

                # Grammatical objects: direct, attributive, prepositional
                objects = [
                    child for child in token.children
                    if child.dep_ in ("dobj", "attr", "acomp", "oprd")
                ]
                for prep in (c for c in token.children if c.dep_ == "prep"):
                    objects.extend(
                        gc for gc in prep.children if gc.dep_ in ("pobj", "pcomp")
                    )

                # Passive agent: "X was built by Y" → "by Y" agent arc
                # The pobj of "by" becomes the logical subject.
                for agent in (c for c in token.children if c.dep_ == "agent"):
                    subjects.extend(gc for gc in agent.children if gc.dep_ == "pobj")

                rel_type = VERB_TO_RELATION_TYPE.get(token.lemma_, "RELATED_TO")

                for subj_tok in subjects:
                    for obj_tok in objects:
                        subj_ent = find_entity_for_token(subj_tok, offset)
                        obj_ent = find_entity_for_token(obj_tok, offset)

                        if not subj_ent or not obj_ent or subj_ent.name == obj_ent.name:
                            continue

                        pair_key = (subj_ent.name, obj_ent.name, rel_type)
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)

                        relations.append(ExtractedRelation(
                            source_entity=subj_ent.name,
                            target_entity=obj_ent.name,
                            relation_type=rel_type,
                            weight=0.85,  # syntactically grounded → higher confidence
                        ))

        # ── phase 2: proximity fallback ───────────────────────────────────────
        # Catches pairs not expressed through explicit verbs
        # (appositions, enumerations, nominal predicates, etc.)

        found_pairs = {(r.source_entity, r.target_entity) for r in relations}

        for i in range(len(sorted_entities)):
            e1 = sorted_entities[i]
            for j in range(i + 1, len(sorted_entities)):
                e2 = sorted_entities[j]

                if e2.start_char - e1.end_char >= 100:
                    break  # beyond proximity window, skip remaining

                if (e1.name, e2.name) in found_pairs:
                    continue  # already captured by the dependency phase

                between = full_text[e1.end_char:e2.start_char].strip().lower()

                rel_type = "RELATED_TO"
                if "is a" in between or "are a" in between:
                    rel_type = "IS_A"
                elif "based on" in between:
                    rel_type = "BASED_ON"

                relations.append(ExtractedRelation(
                    source_entity=e1.name,
                    target_entity=e2.name,
                    relation_type=rel_type,
                    weight=0.6,  # heuristic → lower confidence
                ))

        return relations

    @staticmethod
    def get_entity_hash(name: str, entity_type: EntityType, tenant_id: str) -> str:
        """Generates a unique hash for an entity (for cross-document deduplication)."""
        # Remove extra spaces, convert to lowercase
        normalized = name.lower().strip()
        # Remove leading articles
        normalized = re.sub(r'^(the|a|an)\s+', '', normalized)

        key = f"{tenant_id}:{entity_type.value}:{normalized}"
        return hashlib.md5(key.encode()).hexdigest()
