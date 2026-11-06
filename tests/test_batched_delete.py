"""Standalone, import-safe regression harness for BATCHED Neo4j deletes.

A live re-embed of two 37 MB sources staged 529 points each; deleting one
source's points plus their PROVENANCE/MENTIONS/relationship edges in ONE
transaction failed deterministically with
``Neo.TransientError.General.MemoryPoolOutOfMemoryError`` (716.8 MiB cap,
~1.35 MB per point-with-edges). The fix resolves the node ids with a read and
deletes them in bounded sequential transactions (``batched_delete.py``).

Tests (all drive the REAL ``GraphRAGHandler`` delete methods; only the Neo4j
session is an in-memory double):

  (a) a 500-point source with edges is deleted in MULTIPLE transactions and
      every node (documents, RELATED_TO edges, entities, SourceFile) is gone;
  (b) deleting the same source twice is a no-op the second time (idempotent);
  (c) the tenant write lock is acquired ONCE for the whole delete, not per
      batch;
  (d) a small source still uses a single Document-delete transaction;
  (e) the batch budget is a bounded module constant with no env override that
      can raise it (and the helper refuses a larger value).

This whole file is inert on import: every import, class and test lives inside
``if __name__ == "__main__":`` (the Cat plugin loader imports EVERY ``.py``
under the plugin folder). Run from the host venv or inside the Cat container:

    python tests/test_batched_delete.py
"""

if __name__ == "__main__":
    import asyncio
    import importlib
    import importlib.util
    import os
    import sys
    import types
    from collections import Counter
    from contextlib import asynccontextmanager

    TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(TESTS_DIR)

    def _load_kg_harness():
        """Reuse the import-safe stub set from the sibling standalone test."""
        path = os.path.join(TESTS_DIR, "test_kg_master_switch.py")
        spec = importlib.util.spec_from_file_location("_kg_stub_harness", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module._install_stubs()
        return module

    _load_kg_harness()

    _pkg = types.ModuleType("catgraphrag_batcheddelete_test")
    _pkg.__path__ = [REPO_ROOT]
    sys.modules["catgraphrag_batcheddelete_test"] = _pkg
    graphrag_handler = importlib.import_module(
        "catgraphrag_batcheddelete_test.graphrag_handler"
    )
    batched_delete = importlib.import_module(
        "catgraphrag_batcheddelete_test.batched_delete"
    )
    tenant_write_lock = importlib.import_module(
        "catgraphrag_batcheddelete_test.tenant_write_lock"
    )

    class _UpdateResult:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    # The shared stub defines a bare type; the real one takes kwargs.
    setattr(graphrag_handler, "UpdateResult", _UpdateResult)

    class _FakeResult:
        def __init__(self, records=()):
            self._records = list(records)

        async def _iter(self):
            for record in self._records:
                yield record

        def __aiter__(self):
            return self._iter()

        async def single(self):
            return self._records[0] if self._records else None

        async def data(self):
            return list(self._records)

        async def consume(self):
            return None

    class _Store:
        """Minimal in-memory graph for exactly the queries under test."""

        def __init__(self):
            self.tenant = "tenant-batch"
            self.source_name = None
            self.docs = {}
            self.related = {}
            self.entities = {}
            self.similar = {}
            self.sourcefiles = set()
            self.txns = Counter()

        def seed(self, docs=0, related=0, entities=0, similar=0, sourcefiles=0,
                 source=None):
            self.source_name = source
            for i in range(docs):
                self.docs[f"doc-{i}"] = {"tenant": self.tenant, "source": source}
            for i in range(related):
                self.related[f"rel-{i}"] = {
                    "tenant": self.tenant,
                    "files": [source] if source else [],
                }
            for i in range(entities):
                self.entities[f"ent-{i}"] = {
                    "tenant": self.tenant,
                    "tracked": True,
                    "provenance": 0,
                    "mentions": 1,
                }
            for i in range(similar):
                self.similar[f"sim-{i}"] = {"tenant": self.tenant}
            for i in range(sourcefiles):
                self.sourcefiles.add(f"source-{i}")
            if source:
                self.sourcefiles.add(source)

    class _FakeSession:
        def __init__(self, store):
            self.store = store

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def run(self, query, **params):
            q = " ".join(str(query).split())
            store = self.store
            tenant = store.tenant

            # ── reads ─────────────────────────────────────────────────────
            if "RETURN d.id AS id" in q:
                ids = [
                    did
                    for did, meta in store.docs.items()
                    if meta.get("tenant") == tenant
                    and (store.source_name is None
                         or meta.get("source") == store.source_name)
                ]
                return _FakeResult([{"id": did} for did in ids])

            if "RETURN DISTINCT e.id AS id" in q:
                return _FakeResult([
                    {"id": eid}
                    for eid, ent in store.entities.items()
                    if ent.get("tenant") == tenant
                ])

            if "RETURN e.id AS id" in q:
                records = []
                for eid, ent in store.entities.items():
                    if ent.get("tenant") != tenant:
                        continue
                    if "tracked_by_provenance" in q:
                        ok = ent.get("tracked") and ent.get("provenance", 0) == 0
                    else:
                        ok = (ent.get("mentions", 0) == 0
                              and ent.get("provenance", 0) == 0)
                    if ok:
                        records.append({"id": eid})
                return _FakeResult(records)

            if "elementId(r) AS rid" in q:
                if "SIMILAR_TO" in q:
                    return _FakeResult([
                        {"rid": rid}
                        for rid, rel in store.similar.items()
                        if rid in params.get("rids", list(store.similar))
                        and rel.get("tenant") == tenant
                    ])
                source = params.get("source")
                return _FakeResult([
                    {"rid": rid}
                    for rid, rel in store.related.items()
                    if rid in store.related
                    and rid in params.get("rids", list(store.related))
                    and ent_tenant(store, rid) == tenant
                    and source in (store.related[rid].get("files") or [])
                ])

            if "RETURN sf.name AS name" in q:
                return _FakeResult([{"name": name} for name in sorted(store.sourcefiles)])

            # ── writes ────────────────────────────────────────────────────
            if "DETACH DELETE d" in q:
                for did in params.get("ids", []):
                    store.docs.pop(did, None)
                store.txns["doc"] += 1
                return _FakeResult()

            if "SET r.source_files" in q:
                for rid in params.get("rids", []):
                    rel = store.related.get(rid)
                    if rel is None:
                        continue
                    files = [f for f in rel.get("files", []) if f != params["source"]]
                    if files:
                        rel["files"] = files
                    else:
                        store.related.pop(rid, None)
                store.txns["related"] += 1
                return _FakeResult()

            if "DETACH DELETE e" in q:
                for eid in params.get("ids", []):
                    store.entities.pop(eid, None)
                store.txns["entity"] += 1
                return _FakeResult()

            if "DETACH DELETE sf" in q:
                if "sf.name IN $names" in q:
                    for name in params.get("names", []):
                        store.sourcefiles.discard(name)
                else:
                    store.sourcefiles.discard(params.get("source"))
                store.txns["sourcefile"] += 1
                return _FakeResult()

            if "type(r) IN ['PROVENANCE', 'MENTIONS']" in q and "DELETE r" in q:
                store.txns["source_edge"] += 1
                return _FakeResult()

            if "SIMILAR_TO" in q and "DELETE r" in q:
                for rid in params.get("rids", []):
                    store.similar.pop(rid, None)
                store.txns["similar"] += 1
                return _FakeResult()

            return _FakeResult()

    def ent_tenant(store, rid):
        # RELATED_TO endpoints belong to the same tenant by construction.
        return store.related[rid].get("tenant")

    def _make_handler(store):
        handler = graphrag_handler.GraphRAGHandler(
            neo4j_uri="bolt://fake",
            neo4j_user="u",
            neo4j_password="p",
            enable_knowledge_graph=True,
            enable_concept_relations=False,
        )
        handler.agent_id = store.tenant
        handler._generation = 1
        handler._pending_source_tasks = {}
        handler._get_session = lambda: _FakeSession(store)

        async def ensure_connected():
            return None

        handler._ensure_connected = ensure_connected
        return handler

    async def test_a_large_source_is_deleted_in_multiple_transactions():
        store = _Store()
        store.seed(docs=500, related=500, entities=500, sourcefiles=0, source="big.pdf")
        store.sourcefiles.add("big.pdf")
        handler = _make_handler(store)

        await handler.delete_tenant_points("declarative", {"source": "big.pdf"})

        assert store.txns["doc"] > 1, f"expected multiple doc txns, got {store.txns}"
        assert store.txns["doc"] == 5, f"expected ceil(500/100)=5 doc txns, got {store.txns}"
        assert store.txns["related"] == 5, store.txns
        assert store.txns["entity"] == 5, store.txns
        assert not store.docs, f"{len(store.docs)} documents survived"
        assert not store.related, f"{len(store.related)} RELATED_TO edges survived"
        assert not store.entities, f"{len(store.entities)} entities survived"
        assert store.sourcefiles == set(), f"SourceFile survived: {store.sourcefiles}"
        print(
            f"(a) PASS  500 docs + 500 edges deleted in {store.txns['doc']} doc "
            f"batches and {store.txns['related']} related batches"
        )

    async def test_b_second_delete_is_a_noop():
        store = _Store()
        store.seed(docs=500, related=500, entities=500, source="big.pdf")
        store.sourcefiles.add("big.pdf")
        handler = _make_handler(store)

        await handler.delete_tenant_points("declarative", {"source": "big.pdf"})
        assert not store.docs and not store.related and not store.entities

        store.txns.clear()
        await handler.delete_tenant_points("declarative", {"source": "big.pdf"})

        assert store.txns["doc"] == 0, "second delete issued a doc transaction"
        assert not store.docs and not store.related and not store.entities
        assert store.sourcefiles == set()
        print("(b) PASS  second delete of the same ids is a no-op")

    async def test_c_lock_is_acquired_once_for_the_whole_delete():
        store = _Store()
        store.seed(docs=250, related=0, entities=0, source="medium.pdf")
        store.sourcefiles.add("medium.pdf")
        handler = _make_handler(store)

        acquisitions = 0
        original = tenant_write_lock.acquire_tenant_write_lock

        @asynccontextmanager
        async def _counting(handler_obj, tenant_id, operation):
            nonlocal acquisitions
            acquisitions += 1
            yield

        tenant_write_lock.__dict__["acquire_tenant_write_lock"] = _counting
        try:
            await handler.delete_tenant_points("declarative", {"source": "medium.pdf"})
        finally:
            tenant_write_lock.__dict__["acquire_tenant_write_lock"] = original

        assert store.txns["doc"] == 3, f"expected 3 doc txns, got {store.txns}"
        assert acquisitions == 1, (
            f"tenant lock acquired {acquisitions} times; must be once per delete"
        )
        print(
            f"(c) PASS  {store.txns['doc']} batches, tenant lock acquired "
            f"{acquisitions} time(s)"
        )

    async def test_d_small_source_uses_one_document_transaction():
        store = _Store()
        store.seed(docs=3, related=0, entities=0, source="small.pdf")
        store.sourcefiles.add("small.pdf")
        handler = _make_handler(store)

        await handler.delete_tenant_points("declarative", {"source": "small.pdf"})

        assert store.txns["doc"] == 1, f"small source used {store.txns['doc']} doc txns"
        assert store.txns["related"] == 0, f"unnecessary related txns: {store.txns}"
        assert store.txns["entity"] == 0, f"unnecessary entity txns: {store.txns}"
        assert not store.docs
        print(f"(d) PASS  3-point source used {store.txns['doc']} transaction")

    def test_e_budget_is_a_bounded_constant():
        assert batched_delete.DELETE_BATCH_MAX_ITEMS == 100, (
            "memory-guard budget drifted from the measured 100"
        )
        module_path = batched_delete.__file__
        assert isinstance(module_path, str)
        module_source = open(module_path, encoding="utf-8").read()
        assert "os.environ" not in module_source and "getenv" not in module_source, (
            "DELETE_BATCH_MAX_ITEMS must not be environment-overridable"
        )
        try:
            batched_delete.chunk_ids([1, 2, 3], max_items=101)
        except ValueError:
            pass
        else:
            raise AssertionError("chunk_ids accepted a budget above the memory guard")

        async def _oversized_delete_is_rejected():
            try:
                await batched_delete.delete_in_batches(
                    [], lambda batch: None, max_items=101
                )
            except ValueError:
                return True
            return False

        assert asyncio.run(_oversized_delete_is_rejected()), (
            "delete_in_batches accepted an unbounded budget"
        )
        print(
            f"(e) PASS  budget is the constant DELETE_BATCH_MAX_ITEMS="
            f"{batched_delete.DELETE_BATCH_MAX_ITEMS}, no env override, "
            f"larger values rejected"
        )

    async def test_f_phase_purge_keeps_the_source_structure():
        """A graph-phase purge removes only what the graph phases derived. The
        ``SourceFile`` node (document structure: PART_OF/NEXT/HAS_SUMMARY) is
        built by the EMBEDDING phase, which EffING does not re-run when it
        recovers a graph phase, so the purge must leave it in place."""
        store = _Store()
        store.seed(docs=3, related=2, entities=2, similar=1, source="file.pdf")
        handler = _make_handler(store)

        await handler.purge_source_graph_artifacts("file.pdf")

        assert "file.pdf" in store.sourcefiles, "phase purge deleted the SourceFile"
        assert store.txns["sourcefile"] == 0, store.txns
        assert len(store.docs) == 3, "phase purge must never delete chunk Documents"
        print("(f) PASS  phase purge keeps the SourceFile and the chunk Documents")

    async def _run_async():
        await test_a_large_source_is_deleted_in_multiple_transactions()
        await test_b_second_delete_is_a_noop()
        await test_c_lock_is_acquired_once_for_the_whole_delete()
        await test_d_small_source_uses_one_document_transaction()
        await test_f_phase_purge_keeps_the_source_structure()

    def main():
        asyncio.run(_run_async())
        test_e_budget_is_a_bounded_constant()
        print("\nALL BATCHED DELETE TESTS PASSED")

    main()
