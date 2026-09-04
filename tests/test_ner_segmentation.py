"""Standalone verification for the segmented spaCy NER path.

Runnable with:  python test_ner_segmentation.py

Pure-stdlib, plain asserts, no pytest and no model loading. The plugin's
``entity_extractor`` is imported through an in-memory package whose
``__path__`` points at this repository root (bypassing the plugin
``__init__.py`` side effects), while external dependencies (cat, langdetect,
spacy, pydantic) are stubbed in ``sys.modules`` *before* the import.

What is verified:
- the segmentation partition invariant (joining the segments reproduces the
  original text exactly) incl. edge cases: empty text, separator-only text,
  leading/trailing separators, CRLF, 3+ newlines, >cap single token,
  whitespace-aware and raw hard splits;
- ``extract()`` drives ``nlp.pipe`` over the segments with the configured
  batch size and pairs each doc with its segment offset (input order);
- entity char offsets stay GLOBAL (per-segment local offsets shifted by the
  segment start offset);
- SVO relation extraction works per segment (offset-aware token lookups) and
  ``seen_pairs`` is shared across segments;
- the phase-2 proximity fallback still catches pairs spanning segment
  boundaries;
- ``extract_entities(doc)`` remains backward compatible (no offset → local
  offsets, as used by the graphrag_handler query path);
- langdetect outcome does not matter when only the "default" model is
  configured: any detection falls back to the default nlp.
"""

import asyncio
import os
import sys
import types
from contextlib import contextmanager

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Minimal stubs for the external packages imported by the plugin modules.
# Installed into sys.modules BEFORE importing entity_extractor/models.
# ---------------------------------------------------------------------------


class _Lang:
    """Stand-in for a langdetect language result (has .lang and .prob)."""

    def __init__(self, lang, prob):
        self.lang = lang
        self.prob = prob


def _default_detect_langs(text):
    return [_Lang("en", 0.99)]


_DETECT_LANGS = _default_detect_langs


def _install_cat_stub():
    cat_mod = types.ModuleType("cat")

    class _StubLog:
        def info(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    setattr(cat_mod, "log", _StubLog())
    sys.modules["cat"] = cat_mod


def _install_langdetect_stub():
    langdetect_mod = types.ModuleType("langdetect")

    class DetectorFactory:
        seed = 0

    setattr(langdetect_mod, "DetectorFactory", DetectorFactory)
    setattr(langdetect_mod, "detect_langs", lambda text: _DETECT_LANGS(text))
    sys.modules["langdetect"] = langdetect_mod


def _install_spacy_stubs():
    def _forbidden(*args, **kwargs):
        raise AssertionError("real spaCy must not be loaded inside the standalone test")

    spacy_mod = types.ModuleType("spacy")
    spacy_mod.__path__ = []  # look like a package root for submodule imports
    setattr(spacy_mod, "load", _forbidden)
    sys.modules["spacy"] = spacy_mod

    util_mod = types.ModuleType("spacy.util")
    setattr(util_mod, "is_package", lambda name: True)
    sys.modules["spacy.util"] = util_mod

    cli_mod = types.ModuleType("spacy.cli")
    cli_mod.__path__ = []
    sys.modules["spacy.cli"] = cli_mod

    download_mod = types.ModuleType("spacy.cli.download")
    setattr(download_mod, "download", _forbidden)
    sys.modules["spacy.cli.download"] = download_mod

    language_mod = types.ModuleType("spacy.language")
    setattr(language_mod, "Language", type("Language", (), {}))
    sys.modules["spacy.language"] = language_mod

    tokens_mod = types.ModuleType("spacy.tokens")
    setattr(tokens_mod, "Doc", type("Doc", (), {}))
    sys.modules["spacy.tokens"] = tokens_mod


def _install_pydantic_stub():
    pydantic_mod = types.ModuleType("pydantic")

    class Field:
        """Placeholder for pydantic.Field — stores default/default_factory."""

        def __init__(self, default=None, *, default_factory=None, **kwargs):
            self.default = default
            self.default_factory = default_factory

    class BaseModel:
        """Minimal pydantic.BaseModel — stores keyword init values as attrs."""

        def __init__(self, **kwargs):
            for name, value in kwargs.items():
                setattr(self, name, value)

    setattr(pydantic_mod, "Field", Field)
    setattr(pydantic_mod, "BaseModel", BaseModel)
    sys.modules["pydantic"] = pydantic_mod


# ---------------------------------------------------------------------------
# Fake spaCy objects
# ---------------------------------------------------------------------------


class _FakeToken:
    """Minimal token: text/idx + pos/dep/lemma + children/subtree."""

    def __init__(self, text, idx, pos="NOUN", dep="dep", lemma=None, children=None):
        self.text = text
        self.idx = idx
        self.pos_ = pos
        self.dep_ = dep
        self.lemma_ = lemma if lemma is not None else text.lower()
        self._children = list(children or [])
        self._subtree = None

    @property
    def children(self):
        return iter(self._children)

    @property
    def subtree(self):
        if self._subtree is None:
            self._subtree = [self] + self._children
        return iter(self._subtree)


class _FakeEnt:
    """Minimal named entity with LOCAL char offsets."""

    def __init__(self, label, text, start_char, end_char):
        self.label_ = label
        self.text = text
        self.start_char = start_char
        self.end_char = end_char


class _FakeDoc:
    """Minimal Doc: iterable of tokens + .ents."""

    def __init__(self, tokens, ents):
        self._tokens = list(tokens)
        self.ents = list(ents)

    def __iter__(self):
        return iter(self._tokens)


class _FakeNlp:
    """Records nlp.pipe calls; yields pre-built docs in input order."""

    def __init__(self, docs):
        self._docs = docs
        self.calls = []  # (texts, batch_size)

    def pipe(self, texts, batch_size=8):
        self.calls.append((list(texts), batch_size))
        assert len(texts) == len(self._docs), (
            f"fake plan mismatch: {len(texts)} texts vs {len(self._docs)} docs"
        )
        return iter(self._docs)


def _make_doc_seg0():
    """Fake Doc for 'Alice uses Neo4j.' — SVO tokens + ents at LOCAL offsets."""
    alice = _FakeToken("Alice", 0, "PROPN", "nsubj", "alice")
    neo4j = _FakeToken("Neo4j", 11, "PROPN", "dobj", "neo4j")
    uses = _FakeToken("uses", 6, "VERB", "ROOT", "use", children=[alice, neo4j])
    dot = _FakeToken(".", 17, "PUNCT", "punct")
    return _FakeDoc(
        [alice, uses, neo4j, dot],
        [_FakeEnt("PERSON", "Alice", 0, 5), _FakeEnt("ORG", "Neo4j", 11, 16)],
    )


def _make_doc_seg1():
    """Fake Doc for 'Guido created Python.' — SVO tokens + ents at LOCAL offsets."""
    guido = _FakeToken("Guido", 0, "PROPN", "nsubj", "guido")
    python = _FakeToken("Python", 14, "PROPN", "dobj", "python")
    created = _FakeToken("created", 6, "VERB", "ROOT", "create", children=[guido, python])
    dot = _FakeToken(".", 20, "PUNCT", "punct")
    return _FakeDoc(
        [guido, created, python, dot],
        [_FakeEnt("PERSON", "Guido", 0, 5), _FakeEnt("PRODUCT", "Python", 14, 20)],
    )


def _make_doc_seg2():
    """Fake Doc for 'The end.' — no verbs, no entities."""
    the = _FakeToken("The", 0, "DET", "det")
    end = _FakeToken("end", 4, "NOUN", "ROOT")
    dot = _FakeToken(".", 7, "PUNCT", "punct")
    return _FakeDoc([the, end, dot], [])


def _extractor_with(fake_nlp):
    """White-box EntityExtractor: initialized, only the "default" model."""
    ex = EntityExtractor(models={"default": "fake"})
    ex._initialized = True
    ex._nlps = {"default": fake_nlp}
    return ex


@contextmanager
def _segment_cap(cap):
    """Temporarily lower the segment cap (used to force multi-segment splits)."""
    original = entity_extractor.SPACY_MAX_SEGMENT_CHARS
    entity_extractor.SPACY_MAX_SEGMENT_CHARS = cap
    try:
        yield
    finally:
        entity_extractor.SPACY_MAX_SEGMENT_CHARS = original


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_segment_partition_invariant():
    cases = [
        "",                          # empty text → no segments
        "short single paragraph",
        "Para one.\n\nPara two.",    # basic paragraph boundary
        "\n\nLeading separator.",    # separator at start (empty first chunk)
        "Trailing separator.\n\n",   # separator at end
        "\n\n\n\n",                  # separators only
        "a\r\n\r\nb",                # CRLF boundaries
        "one\n\n\n\ntwo",            # 3+ newlines → one separator
        "one\n \n \ntwo",            # whitespace inside the separator
        "x" * 100_001,               # single >cap token → raw hard splits
        ("word " * 30_000).rstrip(), # >cap whitespace → whitespace-aware splits
    ]
    cap = entity_extractor.SPACY_MAX_SEGMENT_CHARS
    for text in cases:
        segments = _segment_text(text)
        if text:
            assert "".join(s for _, s in segments) == text
        offset = 0
        for seg_offset, seg in segments:
            assert seg_offset == offset  # running cumulative lengths
            assert len(seg) <= cap
            offset += len(seg)
        assert offset == len(text)


def test_segment_small_cap():
    original = entity_extractor.SPACY_MAX_SEGMENT_CHARS
    entity_extractor.SPACY_MAX_SEGMENT_CHARS = 50
    try:
        # whitespace-aware hard split (no paragraph boundary in sight)
        text = "aaa " * 20  # 80 chars
        segments = _segment_text(text)
        assert "".join(s for _, s in segments) == text
        assert all(len(s) <= 50 for _, s in segments)
        assert len(segments) > 1

        # raw hard split (no whitespace at all)
        text2 = "y" * 120
        segments2 = _segment_text(text2)
        assert "".join(s for _, s in segments2) == text2
        assert all(len(s) <= 50 for _, s in segments2)

        # oversized paragraph + following short paragraph interplay
        text3 = "bbbb " * 14 + "\n\n" + "short tail."  # 70 + 2 + 10 chars
        segments3 = _segment_text(text3)
        assert "".join(s for _, s in segments3) == text3
        assert all(len(s) <= 50 for _, s in segments3)
    finally:
        entity_extractor.SPACY_MAX_SEGMENT_CHARS = original


def test_extract_segments_global_offsets():
    text = "Alice uses Neo4j.\n\nGuido created Python.\n\nThe end."
    # cap 25 → 3 segments: (p1+sep), (p2+sep), (p3); the default cap would pack
    # the whole 50-char text into a single segment.
    with _segment_cap(25):
        segments = _segment_text(text)
        assert len(segments) == 3

        fake = _FakeNlp([_make_doc_seg0(), _make_doc_seg1(), _make_doc_seg2()])
        ex = _extractor_with(fake)
        result = asyncio.run(ex.extract(text, "doc1", {"origin": "test"}))

        # pipe got exactly the segment texts, in order, with the configured batch size
        assert len(fake.calls) == 1
        texts, batch_size = fake.calls[0]
        assert texts == [seg for _, seg in segments]
        assert batch_size == SPACY_PIPE_BATCH_SIZE

        # content/metadata preserved
        assert result.content == text
        assert result.metadata == {"origin": "test"}

        # entities carry GLOBAL char offsets (local + segment start)
        by_name = {e.name: e for e in result.entities}
        assert by_name["Alice"].start_char == segments[0][0] + 0
        assert by_name["Alice"].end_char == segments[0][0] + 5
        assert by_name["Guido"].start_char == segments[1][0] + 0
        assert by_name["Guido"].end_char == segments[1][0] + 5

        # spaCy → EntityType mapping + regex dedup unchanged (PRODUCT beats TECH 0.9 > 0.85)
        assert by_name["Alice"].type == EntityType.PERSON
        assert by_name["Neo4j"].type == EntityType.ORGANIZATION
        assert by_name["Python"].type == EntityType.PRODUCT

        # SVO relations found in each segment, via offset-aware token lookups
        rel_pairs = {(r.source_entity, r.target_entity, r.relation_type) for r in result.relations}
        assert ("Alice", "Neo4j", "USES") in rel_pairs
        assert ("Guido", "Python", "CREATES") in rel_pairs


def test_seen_pairs_shared_across_segments():
    text = "Alice uses Neo4j.\n\nAlice uses Neo4j."
    with _segment_cap(25):  # 2 segments; default cap would pack into one
        assert len(_segment_text(text)) == 2
        fake = _FakeNlp([_make_doc_seg0(), _make_doc_seg0()])
        ex = _extractor_with(fake)
        result = asyncio.run(ex.extract(text, "doc2", {}))

        uses = [r for r in result.relations if r.relation_type == "USES"]
        assert len(uses) == 1  # same SVO pair in two segments → emitted once


def test_relations_phase2_cross_boundary():
    full_text = "Alice talks.\n\nACME grows."
    segments = [(0, "Alice talks."), (14, "ACME grows.")]
    pairs = list(zip([_FakeDoc([], []), _FakeDoc([], [])], [off for off, _ in segments]))
    entities = [
        ExtractedEntity(name="Alice", type=EntityType.PERSON, start_char=0, end_char=5, confidence=0.9),
        ExtractedEntity(name="ACME", type=EntityType.ORGANIZATION, start_char=14, end_char=18, confidence=0.9),
    ]
    relations = EntityExtractor._extract_relations(pairs, entities, full_text)
    assert any(
        r.source_entity == "Alice" and r.target_entity == "ACME"
        and r.relation_type == "RELATED_TO"
        for r in relations
    )


def test_extract_entities_backward_compatible():
    # graphrag_handler.py:1164 calls extract_entities(doc) without offset:
    # local offsets must be kept unchanged.
    ents = EntityExtractor.extract_entities(_make_doc_seg0())
    assert len(ents) == 2
    assert ents[0].name == "Alice" and ents[0].start_char == 0 and ents[0].end_char == 5
    assert ents[1].name == "Neo4j" and ents[1].start_char == 11 and ents[1].end_char == 16


def test_extract_empty_text():
    fake = _FakeNlp([])
    ex = _extractor_with(fake)
    result = asyncio.run(ex.extract("", "doc-empty", {}))
    assert result.entities == []
    assert result.relations == []
    texts, _ = fake.calls[0]
    assert texts == []


def test_extract_short_text_single_segment():
    fake = _FakeNlp([_make_doc_seg0()])
    ex = _extractor_with(fake)
    result = asyncio.run(ex.extract("Alice uses Neo4j.", "doc-short", {}))
    texts, batch_size = fake.calls[0]
    assert texts == ["Alice uses Neo4j."]
    assert batch_size == SPACY_PIPE_BATCH_SIZE
    assert result.entities[0].name == "Alice" and result.entities[0].start_char == 0


def test_langdetect_any_result_falls_back_to_default():
    global _DETECT_LANGS
    fake = _FakeNlp([_make_doc_seg0()])
    ex = _extractor_with(fake)
    old = _DETECT_LANGS
    _DETECT_LANGS = lambda text: [_Lang("xx", 0.99)]  # language not configured
    try:
        result = asyncio.run(ex.extract("Alice uses Neo4j.", "doc-lang", {}))
        assert fake.calls  # default model used; no KeyError
        assert any(e.name == "Alice" for e in result.entities)
    finally:
        _DETECT_LANGS = old


def main():
    tests = [
        test_segment_partition_invariant,
        test_segment_small_cap,
        test_extract_segments_global_offsets,
        test_seen_pairs_shared_across_segments,
        test_relations_phase2_cross_boundary,
        test_extract_entities_backward_compatible,
        test_extract_empty_text,
        test_extract_short_text_single_segment,
        test_langdetect_any_result_falls_back_to_default,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    _install_cat_stub()
    _install_langdetect_stub()
    _install_spacy_stubs()
    _install_pydantic_stub()

    # In-memory package pointing at the repo root: imports entity_extractor.py
    # directly (relative imports .constants/.models resolved via __path__) without
    # executing the plugin __init__.py (which pulls in graphrag_handler/neo4j).
    _pkg = types.ModuleType("cat_graphrag_segtest")
    _pkg.__path__ = [REPO_ROOT]
    sys.modules["cat_graphrag_segtest"] = _pkg

    from cat_graphrag_segtest import entity_extractor  # noqa: E402
    from cat_graphrag_segtest.constants import SPACY_PIPE_BATCH_SIZE  # noqa: E402
    from cat_graphrag_segtest.entity_extractor import EntityExtractor, _segment_text  # noqa: E402
    from cat_graphrag_segtest.models import EntityType, ExtractedEntity  # noqa: E402

    main()