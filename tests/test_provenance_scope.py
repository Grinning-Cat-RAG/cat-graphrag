"""The RELATED_TO provenance of a source is scoped: a file with the same name in the knowledge base and in a chat are
two sources, with two provenance keys (``provenance_key``: the name in the knowledge base, ``chat:<chat_id>:<name>``
in a chat). Every writer keys by the scope of the source; the Cypher that strips the keys runs on a live Neo4j in
``test_graph_purge_live.py``.

- the key: unchanged for the knowledge base (the existing edges stay right), scoped by the chat in a chat;
- the spaCy extraction of a chunk keys its relations by the chunk's chat;
- the LLM extraction of a source keys its concepts (and its retry counter) by the source's chat;
- the recompute of the concepts treats a knowledge base file and a chat file with the same name as two sources.

Standalone script (plugin rules: the body runs only as ``__main__``). Needs the core checkout and a Redis Stack on
localhost:6379 (database 14):

    cd ../grinning-cat-core
    .venv/bin/python /path/to/cat-graphrag/tests/test_provenance_scope.py
"""

if __name__ == "__main__":
    import asyncio
    import json
    import os
    import sys
    import traceback
    import types

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import graph_fakes as gf

    main_mod, handler_mod, _ = gf.bootstrap("_graphrag_provenance")

    def handler():
        return gf.make_handler(handler_mod, "provenance_agent")

    async def test_the_key():
        key = handler_mod.provenance_key
        assert key("a.pdf") == "a.pdf" and key("a.pdf", None) == "a.pdf", "a knowledge base key changed"
        assert key("a.pdf", "c1") == "chat:c1:a.pdf"
        assert key("a.pdf", "c1") != key("a.pdf", "c2") != key("a.pdf")

    async def test_the_spacy_extraction_keys_by_the_chat():
        h = handler()

        async def extract(text, document_id, metadata=None):
            alice = types.SimpleNamespace(name="Alice", type=handler_mod.EntityType.PERSON, confidence=0.9)
            bob = types.SimpleNamespace(name="Bob", type=handler_mod.EntityType.PERSON, confidence=0.9)
            met = types.SimpleNamespace(source_entity="Alice", target_entity="Bob", relation_type="MET", weight=1.0)
            return types.SimpleNamespace(entities=[alice, bob], relations=[met])

        h._entity_extractor = types.SimpleNamespace(
            extract=extract, get_entity_hash=lambda name, etype, tenant: f"{tenant}:{name}",
        )
        kb = await h._extract_document_graph("d1", "Alice met Bob", {"source": "a.pdf"})
        chat = await h._extract_document_graph("d2", "Alice met Bob", {"source": "a.pdf", "chat_id": "c1"})
        assert [r["source_files"] for r in kb["relations"]] == [["a.pdf"]], kb["relations"]
        assert [r["source_files"] for r in chat["relations"]] == [["chat:c1:a.pdf"]], chat["relations"]

    async def test_the_llm_extraction_keys_by_the_chat():
        h = handler()
        stored = []
        outcomes = [{"status": "not_ready", "reason": "busy"}, {"status": "done", "concepts": [{"text": "X"}], "relations": []}]

        async def llm_extract(text, llm=None, attempt=1):
            return outcomes.pop(0)

        async def store(tenant_id, relations, source=None, document_ids=None, concept_gen=None):
            stored.append(source)

        async def fingerprint():
            return "fp"

        h._llm_extract_relations = llm_extract
        h._store_concept_relations = store
        h._concept_fingerprint = fingerprint
        points = [types.SimpleNamespace(
            id="d2", payload={"page_content": "Alice met Bob", "metadata": {"source": "a.pdf", "chat_id": "c1"}},
        )]
        outcome = await h._extract_concept_relations("a.pdf", points, llm="llm")
        assert outcome["status"] == "not_ready"
        assert set(h._concept_attempts) == {"chat:c1:a.pdf"}, h._concept_attempts  # not shared with the KB file
        outcome = await h._extract_concept_relations("a.pdf", points, llm="llm")
        assert outcome["status"] == "done" and stored == ["chat:c1:a.pdf"], (outcome, stored)

    async def test_the_recompute_keeps_the_kb_and_the_chat_file_apart():
        h = handler()
        h._enable_concept_relations = True
        rows = [
            {"id": "d1", "content": "Alice", "metadata": json.dumps({"source": "a.pdf"})},
            {"id": "d2", "content": "Bob", "metadata": json.dumps({"source": "a.pdf", "chat_id": "c1"})},
            {"id": "d3", "content": "Carol", "metadata": json.dumps({"source": "a.pdf"})},
        ]

        class Result:
            def __init__(self):
                self.rows = list(rows)

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.rows:
                    raise StopAsyncIteration
                return self.rows.pop(0)

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def run(self, query, **params):
                return Result()

        async def connected():
            return None

        calls = []

        async def extract(source, points, stray_cat=None, gen=None, llm=None):
            calls.append((source, sorted(p.id for p in points), {p.payload["metadata"].get("chat_id") for p in points}))
            return {"status": "done"}

        h._ensure_connected = connected
        h._get_session = lambda: Session()
        h._extract_concept_relations = extract
        await h.recompute_concept_relations(types.SimpleNamespace(), gen="g")
        assert sorted(calls, key=str) == sorted([("a.pdf", ["d1", "d3"], {None}), ("a.pdf", ["d2"], {"c1"})], key=str), calls

    async def test_the_document_structure_is_built_per_kb_and_chat_file():
        """A write holding a knowledge base file and a chat file with the same name builds two document structures."""
        from cat.services.memory.models import PointStruct

        h = gf.FakeGraphStore().install(handler())
        built = []

        async def derived(source, points, chat_id=None):
            built.append((source, chat_id, len(points)))

        h.create_derived_graph_for_source = derived
        await h.add_points_to_tenant(gf.EPISODIC, [
            PointStruct(id=f"p{i}", payload={"page_content": "x", "metadata": {"source": "a.pdf"} | chat}, vector=[1.0, 1.0])
            for i, chat in enumerate([{}, {"chat_id": "c1"}, {"chat_id": "c1"}])
        ])
        assert sorted(built, key=str) == sorted([("a.pdf", None, 1), ("a.pdf", "c1", 2)], key=str), built

    TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_")]

    async def run_all():
        failed = 0
        for test in TESTS:
            try:
                await test()
                print(f"PASS {test.__name__}")
            except Exception:  # noqa: BLE001 - report every failure
                failed += 1
                print(f"FAIL {test.__name__}")
                traceback.print_exc()
        print(f"\ntest_provenance_scope: {len(TESTS) - failed}/{len(TESTS)} passed")
        return failed

    sys.exit(1 if asyncio.run(run_all()) else 0)
