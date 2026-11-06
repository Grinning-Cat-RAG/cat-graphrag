"""Standalone verification for the EffING phase-machine registration (T5).

Covers the EffING-contract hooks added to ``main.py`` plus the handler
helpers they drive (``run_spacy_extraction_phase``,
``run_llm_extraction_phase``, ``drain_source_tasks``,
``has_pending_source_tasks``, ``get_source_points_by_collection``,
``wipe_tenant_graph_data``):

- ``ingestion_phase_specs`` appends the two PhaseSpecs
  (``graph_spacy_extraction`` / ``graph_llm_extraction``, both depending on
  ``parsing_chunking`` + ``embedding``, ``settings_category=None``) and
  returns the accumulator unchanged when EffING is not importable OR when
  the ``phases`` import itself raises (T7);
- ``ingestion_phase_settings_marker`` returns ``_spacy_fingerprint`` for
  ``graph_spacy_extraction``, ``_concept_fingerprint`` for
  ``graph_llm_extraction``, None otherwise;
- ``ingestion_phase_run`` returns None only for unknown phases (fail-hard),
  ``done`` when there is nothing to do (non-GraphRAG handler), not_ready when
  the source's points are absent, and done (with
  the NER + similarity work / the LLM concept-relations step awaited) when
  the points are present; dispatch to ``run_spacy_extraction_phase`` /
  ``run_llm_extraction_phase`` is asserted directly with AsyncMocks (T7);
- ``before_ingestion_status_completed`` raises when live background tasks
  remain for the source, and passes when the work is clean (the diary check
  is gone: EffING owns the completed-phases diary); it is a no-op only when
  the handler is not GraphRAG, and it never reaches into the core
  ingestion-status registry (T7);
- ``after_rabbithole_stored_documents`` is a NO-OP when the phase path is
  active and keeps the classic ``create_derived_graph_for_source`` behavior
  otherwise;
- ``_phase_path_active`` is True exactly when EffING is importable AND the
  SYSTEM-level ``ingestion`` selection resolves to the efficient engine
  (truth table: uninstalled / installed-core-engine / installed-efficient);
- the spaCy entity batch write keeps ``REMOVE e.concept_gen,
  e.concept_gen_active`` — the shared Entity-node contract stays internal
  to the plugin (T7 shared-schema guard);
- ``after_vector_memory_transfer_on_agent`` wipes the tenant's Neo4j data
  ONLY on success when the NEW handler is not GraphRAG (host no longer passes
  the old handler name), and never on failure;
- ``wipe_tenant_graph_data`` skips without a recorded config and wipes the
  tenant (tenant-scoped) when one exists.

Runnable:  python tests/test_ingestion_phase_machine.py
Pure-stdlib + unittest.mock, plain asserts, no pytest, no model loading. All
external packages (cat, langchain_core, spacy, langdetect, pydantic, neo4j)
are stubbed in sys.modules *before* the plugin modules are imported (same
pattern as test_kg_master_switch.py). The EffING phase-machine contract
(``cat.plugins.cat_efficient_ingestion.phases.PhaseSpec``) is stubbed with a
recording fake.
"""

import asyncio
import os
import sys
import types
from unittest.mock import AsyncMock, Mock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SYSTEM-level ingestion selection returned by the stubbed resolver. Tests
# flip this between "EfficientIngestionConfiguration" (phase path) and
# "BaseIngestionConfiguration" (classic path).
_RESOLVED_INGESTION = {"name": "EfficientIngestionConfiguration"}


class _StubLog:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    warnings = []

    def warning(self, *args, **kwargs):
        type(self).warnings.append(" ".join(str(a) for a in args))

    def error(self, *args, **kwargs):
        pass


def _install_cat_stub():
    cat_mod = types.ModuleType("cat")
    cat_mod.__path__ = []  # look like a package root for submodule imports

    # @hook decorator: supports both bare ``@hook`` and ``@hook(priority=...)``;
    # returns the function unchanged either way.
    def _hook(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

    cat_mod.hook = _hook
    cat_mod.RecallSettings = type("RecallSettings", (), {})
    cat_mod.VectorDatabaseSettings = type("VectorDatabaseSettings", (), {})
    cat_mod.Embeddings = type("Embeddings", (), {})
    cat_mod.AgenticWorkflowTask = type("AgenticWorkflowTask", (), {})

    class _BaseHandler:
        def __init__(self, **kwargs):
            pass

    cat_mod.BaseVectorDatabaseHandler = _BaseHandler
    cat_mod.log = _StubLog()
    sys.modules["cat"] = cat_mod

    log_mod = types.ModuleType("cat.log")
    log_mod.log = _StubLog()
    sys.modules["cat.log"] = log_mod

    looking = types.ModuleType("cat.looking_glass")
    looking.__path__ = []
    sys.modules["cat.looking_glass"] = looking

    stray = types.ModuleType("cat.looking_glass.stray_cat")
    stray.StrayCat = type("StrayCat", (), {})
    sys.modules["cat.looking_glass.stray_cat"] = stray

    services = types.ModuleType("cat.services")
    services.__path__ = []
    sys.modules["cat.services"] = services

    memory = types.ModuleType("cat.services.memory")
    memory.__path__ = []
    sys.modules["cat.services.memory"] = memory

    models_mod = types.ModuleType("cat.services.memory.models")
    models_mod.PointStruct = type("PointStruct", (), {})
    models_mod.DocumentRecall = type("DocumentRecall", (), {})
    models_mod.Record = type("Record", (), {})
    models_mod.ScoredPoint = type("ScoredPoint", (), {})
    models_mod.UpdateResult = type("UpdateResult", (), {})
    sys.modules["cat.services.memory.models"] = models_mod

    factory = types.ModuleType("cat.services.factory")
    factory.__path__ = []
    sys.modules["cat.services.factory"] = factory

    ingestion_factory = types.ModuleType("cat.services.factory.ingestion")

    async def resolved_config_name(lizard):
        return _RESOLVED_INGESTION["name"]

    ingestion_factory.resolved_config_name = resolved_config_name
    sys.modules["cat.services.factory.ingestion"] = ingestion_factory


def _install_neo4j_stub():
    neo4j_mod = types.ModuleType("neo4j")
    neo4j_mod.AsyncGraphDatabase = type("AsyncGraphDatabase", (), {})
    neo4j_mod.AsyncDriver = type("AsyncDriver", (), {})
    neo4j_mod.AsyncSession = type("AsyncSession", (), {})
    sys.modules["neo4j"] = neo4j_mod

    exceptions_mod = types.ModuleType("neo4j.exceptions")
    exceptions_mod.Neo4jError = type("Neo4jError", (), {})
    sys.modules["neo4j.exceptions"] = exceptions_mod


def _install_langchain_stub():
    lc = types.ModuleType("langchain_core")
    lc.__path__ = []
    sys.modules["langchain_core"] = lc

    docs = types.ModuleType("langchain_core.documents")
    docs.Document = type("Document", (), {})
    sys.modules["langchain_core.documents"] = docs


def _install_langdetect_stub():
    langdetect_mod = types.ModuleType("langdetect")

    class DetectorFactory:
        seed = 0

    setattr(langdetect_mod, "DetectorFactory", DetectorFactory)
    setattr(langdetect_mod, "detect_langs", lambda text: [])
    sys.modules["langdetect"] = langdetect_mod


def _install_spacy_stubs():
    def _forbidden(*args, **kwargs):
        raise AssertionError("real spaCy must not be loaded inside the standalone test")

    spacy_mod = types.ModuleType("spacy")
    spacy_mod.__path__ = []
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
        def __init__(self, default=None, *, default_factory=None, **kwargs):
            self.default = default
            self.default_factory = default_factory

    class ConfigDict(dict):
        pass

    class BaseModel:
        def __init__(self, **kwargs):
            for name, value in kwargs.items():
                setattr(self, name, value)

    def BeforeValidator(fn):
        # Identity is enough for the stub: the real one wraps a validator fn.
        return fn

    def create_model(name, **kwargs):
        return type(name, (BaseModel,), {})

    setattr(pydantic_mod, "Field", Field)
    setattr(pydantic_mod, "ConfigDict", ConfigDict)
    setattr(pydantic_mod, "BaseModel", BaseModel)
    setattr(pydantic_mod, "BeforeValidator", BeforeValidator)
    setattr(pydantic_mod, "create_model", create_model)
    sys.modules["pydantic"] = pydantic_mod


def _install_effing_stub():
    """Stub the EffING phase-machine contract the hooks import lazily.

    ``PhaseSpec`` is a recording fake (id / settings_category / depends_on);
    ``cat.plugins.cat_efficient_ingestion`` importable is one half of the
    ``_phase_path_active`` gate (the other is the system ingestion selection).
    """
    plugins = types.ModuleType("cat.plugins")
    plugins.__path__ = []
    sys.modules["cat.plugins"] = plugins

    effing = types.ModuleType("cat.plugins.cat_efficient_ingestion")
    effing.__path__ = []
    sys.modules["cat.plugins.cat_efficient_ingestion"] = effing

    phases = types.ModuleType("cat.plugins.cat_efficient_ingestion.phases")

    class PhaseSpec:
        def __init__(self, id, settings_category=None, depends_on=()):
            self.id = id
            self.settings_category = settings_category
            self.depends_on = tuple(depends_on)

    phases.PhaseSpec = PhaseSpec
    sys.modules["cat.plugins.cat_efficient_ingestion.phases"] = phases


def _install_stubs():
    _install_cat_stub()
    _install_neo4j_stub()
    _install_langchain_stub()
    _install_langdetect_stub()
    _install_spacy_stubs()
    _install_pydantic_stub()
    _install_effing_stub()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeCat:
    """Minimal stand-in for CheshireCat: no ``id`` (agent-scoped), handler set."""

    def __init__(self, handler):
        self.vector_memory_handler = handler
        self.agent_key = "agent_test"


class _FakePoint:
    def __init__(self, pid, vector=None, payload=None):
        self.id = pid
        self.vector = vector
        self.payload = payload or {}


def _make_handler(kg=True, cr=True):
    """Build a GraphRAGHandler with the given flags (no driver needed)."""
    from catgraphrag_phase import graphrag_handler

    handler = graphrag_handler.GraphRAGHandler(
        neo4j_uri="bolt://fake",
        neo4j_user="u",
        neo4j_password="p",
        enable_knowledge_graph=kg,
        enable_concept_relations=cr,
    )
    handler.agent_id = "agent_test"
    return handler


def _completed(deps):
    return [{"phase": pid, "gen": gen} for pid, gen in deps]


# ---------------------------------------------------------------------------
# T5 — phase registration (ingestion_phase_specs / settings marker)
# ---------------------------------------------------------------------------


def test_specs_registers_two_graph_phases():
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    specs = main.ingestion_phase_specs([], cat)
    assert len(specs) == 2, f"expected two PhaseSpecs, got {specs}"
    by_id = {s.id: s for s in specs}
    assert set(by_id) == {"graph_spacy_extraction", "graph_llm_extraction"}, (
        f"got {set(by_id)}"
    )
    for spec in specs:
        assert spec.settings_category is None, (
            f"{spec.id}: settings_category must be None (marker IS the fingerprint), got {spec.settings_category}"
        )
        assert spec.depends_on == ("parsing_chunking", "embedding"), (
            f"{spec.id}: depends_on must be parsing_chunking+embedding, got {spec.depends_on}"
        )


def test_specs_preserves_prior_accumulator_entries():
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    accumulator = [object()]  # opaque prior entry from another plugin
    specs = main.ingestion_phase_specs(accumulator, cat)
    assert specs[0] is accumulator[0], "prior accumulator entries must be preserved"
    assert len(specs) == 3, f"got {len(specs)}"


def test_specs_returns_accumulator_when_effing_absent():
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    saved = sys.modules.pop("cat.plugins.cat_efficient_ingestion", None)
    saved_phases = sys.modules.pop("cat.plugins.cat_efficient_ingestion.phases", None)
    try:
        accumulator = [object()]
        specs = main.ingestion_phase_specs(accumulator, cat)
        assert specs is accumulator, "must return the accumulator unchanged when EffING is absent"
    finally:
        if saved is not None:
            sys.modules["cat.plugins.cat_efficient_ingestion"] = saved
        if saved_phases is not None:
            sys.modules["cat.plugins.cat_efficient_ingestion.phases"] = saved_phases


def test_specs_returns_accumulator_when_phases_import_raises():
    """T7: even when the ``phases`` module is present but its import raises
    (broken EffING install), the accumulator must come back unchanged."""
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)

    class _RaisingPhases(types.ModuleType):
        def __getattr__(self, name):
            raise ImportError(f"cannot import {name} from phases")

    saved = sys.modules.get("cat.plugins.cat_efficient_ingestion.phases")
    sys.modules["cat.plugins.cat_efficient_ingestion.phases"] = _RaisingPhases(
        "cat.plugins.cat_efficient_ingestion.phases"
    )
    try:
        accumulator = [object()]
        specs = main.ingestion_phase_specs(accumulator, cat)
        assert specs is accumulator, (
            "must return the accumulator unchanged when the PhaseSpec import raises"
        )
    finally:
        if saved is not None:
            sys.modules["cat.plugins.cat_efficient_ingestion.phases"] = saved
        else:
            sys.modules.pop("cat.plugins.cat_efficient_ingestion.phases", None)


def test_marker_returns_spacy_fingerprint():
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    marker = asyncio.run(
        main.ingestion_phase_settings_marker("graph_spacy_extraction", cat)
    )
    assert marker == handler._spacy_fingerprint(), f"got {marker}"


def test_marker_returns_concept_fingerprint():
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    marker = asyncio.run(
        main.ingestion_phase_settings_marker("graph_llm_extraction", cat)
    )
    assert marker == asyncio.run(handler._concept_fingerprint()), f"got {marker}"


def test_marker_returns_none_for_unknown_phase():
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    assert asyncio.run(
        main.ingestion_phase_settings_marker("embedding", cat)
    ) is None


def test_marker_returns_none_when_not_graphrag_handler():
    from catgraphrag_phase import main

    class _OtherHandler:
        pass

    cat = _FakeCat(_OtherHandler())
    assert asyncio.run(
        main.ingestion_phase_settings_marker("graph_spacy_extraction", cat)
    ) is None
    assert asyncio.run(
        main.ingestion_phase_settings_marker("graph_llm_extraction", cat)
    ) is None


# ---------------------------------------------------------------------------
# T5 — phase execution (ingestion_phase_run dispatch)
# ---------------------------------------------------------------------------


def test_phase_run_returns_none_for_unknown_phase():
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    result = asyncio.run(main.ingestion_phase_run(
        "embedding", "file.pdf", _completed([]), cat
    ))
    assert result is None, f"expected None for unknown phase (fail-hard), got {result}"


def test_phase_run_returns_done_when_not_graphrag_handler():
    """B1: a registered-but-inactive phase is a no-op, never an ERROR."""
    from catgraphrag_phase import main

    class _OtherHandler:
        pass

    cat = _FakeCat(_OtherHandler())
    result = asyncio.run(main.ingestion_phase_run(
        "graph_spacy_extraction", "file.pdf", _completed([]), cat
    ))
    assert result == {"status": "done"}, f"got {result}"


def test_phase_run_dispatches_spacy_to_run_spacy_extraction_phase():
    """T7: graph_spacy_extraction must dispatch to the handler's
    run_spacy_extraction_phase (awaited, with the cat for LLM resolution)."""
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    with patch.object(type(handler), "run_spacy_extraction_phase",
                      new=AsyncMock(return_value={"status": "done"})) as run:
        result = asyncio.run(main.ingestion_phase_run(
            "graph_spacy_extraction", "file.pdf", _completed([]), cat
        ))
    assert result == {"status": "done"}, f"got {result}"
    run.assert_awaited_once_with("file.pdf", cat=cat)


def test_phase_run_dispatches_llm_to_run_llm_extraction_phase():
    """T7: graph_llm_extraction must dispatch to the handler's
    run_llm_extraction_phase (awaited, with the cat for LLM resolution)."""
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    with patch.object(type(handler), "run_llm_extraction_phase",
                      new=AsyncMock(return_value={"status": "done"})) as run:
        result = asyncio.run(main.ingestion_phase_run(
            "graph_llm_extraction", "file.pdf", _completed([]), cat
        ))
    assert result == {"status": "done"}, f"got {result}"
    run.assert_awaited_once_with("file.pdf", cat=cat)


def test_spacy_phase_not_ready_when_points_absent():
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    with patch.object(type(handler), "get_source_points_by_collection", new=AsyncMock(return_value={})), \
         patch.object(type(handler), "_drain_source_tasks_counted", new=AsyncMock(return_value=None)) as drain:
        result = asyncio.run(main.ingestion_phase_run(
            "graph_spacy_extraction", "file.pdf", _completed([]), cat
        ))
    assert result == {"status": "not_ready", "retry_after": 5}, f"got {result}"
    drain.assert_awaited_once_with("file.pdf")


def test_spacy_phase_done_when_points_present():
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    points = [
        _FakePoint(
            "doc1",
            vector=[0.1, 0.2, 0.3],
            payload={"page_content": "We use GraphVault daily.", "metadata": {"source": "file.pdf"}},
        )
    ]
    with patch.object(type(handler), "get_source_points_by_collection",
                      new=AsyncMock(return_value={"declarative": points})), \
         patch.object(type(handler), "_drain_source_tasks_counted", new=AsyncMock(return_value=None)) as drain, \
         patch.object(type(handler), "_create_similarity_relationships", new=AsyncMock()) as sim, \
         patch.object(type(handler), "_extract_document_graph", new=AsyncMock(
             return_value={"entities": [{"name": "GraphVault"}], "mentions": [], "relations": []}
         )) as ner, \
         patch.object(type(handler), "_write_source_graph", new=AsyncMock()) as write:
        result = asyncio.run(main.ingestion_phase_run(
            "graph_spacy_extraction", "file.pdf", _completed([]), cat
        ))
    assert result["status"] == "done", f"got {result}"
    assert result["processed"] == 1, f"got {result}"
    drain.assert_awaited_once_with("file.pdf")
    sim.assert_awaited_once_with("doc1", [0.1, 0.2, 0.3], "declarative")
    # the NER step runs AWAITED for the source's points (id, content, metadata)
    ner.assert_awaited_once_with(
        "doc1", "We use GraphVault daily.", {"source": "file.pdf"}
    )


def test_spacy_phase_skips_ner_when_entity_extraction_off():
    from catgraphrag_phase import main

    handler = _make_handler()
    handler._enable_entity_extraction = False
    handler._entity_extractor = None
    cat = _FakeCat(handler)
    points = [
        _FakePoint(
            "doc1",
            vector=[0.1, 0.2, 0.3],
            payload={"page_content": "text", "metadata": {"source": "file.pdf"}},
        )
    ]
    with patch.object(type(handler), "get_source_points_by_collection",
                      new=AsyncMock(return_value={"declarative": points})), \
         patch.object(type(handler), "_drain_source_tasks_counted", new=AsyncMock(return_value=None)), \
         patch.object(type(handler), "_create_similarity_relationships", new=AsyncMock()), \
         patch.object(type(handler), "_extract_document_graph", new=AsyncMock()) as ner:
        result = asyncio.run(main.ingestion_phase_run(
            "graph_spacy_extraction", "file.pdf", _completed([]), cat
        ))
    assert result["status"] == "done", f"got {result}"
    # extraction disabled: no product count is produced (current behaviour)
    assert "processed" not in result, f"got {result}"
    ner.assert_not_awaited()


def test_llm_phase_not_ready_when_points_absent():
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    with patch.object(type(handler), "get_source_points_by_collection", new=AsyncMock(return_value={})), \
         patch.object(type(handler), "_drain_source_tasks_counted", new=AsyncMock()) as drain:
        result = asyncio.run(main.ingestion_phase_run(
            "graph_llm_extraction", "file.pdf", _completed([]), cat
        ))
    assert result == {"status": "not_ready", "retry_after": 5}, f"got {result}"
    # the spaCy batch belongs to the spaCy phase
    drain.assert_not_awaited()


def test_llm_phase_done_when_points_present():
    from catgraphrag_phase import main

    handler = _make_handler(kg=True, cr=True)
    cat = _FakeCat(handler)
    points = [_FakePoint("doc1", vector=[0.1, 0.2, 0.3])]
    with patch.object(type(handler), "get_source_points_by_collection",
                      new=AsyncMock(return_value={"declarative": points})), \
         patch.object(type(handler), "_drain_source_tasks_counted", new=AsyncMock()) as drain, \
         patch.object(type(handler), "_extract_concept_relations", new=AsyncMock()) as extract:
        result = asyncio.run(main.ingestion_phase_run(
            "graph_llm_extraction", "file.pdf", _completed([]), cat
        ))
    assert result == {"status": "done"}, f"got {result}"
    drain.assert_not_awaited()
    extract.assert_awaited_once()
    # the LLM step receives the source's points and the cat (LLM resolution)
    _args = extract.await_args.args
    assert _args[0] == "file.pdf"
    assert _args[1] == points


def test_llm_phase_runs_when_kg_off_display_only():
    """``enable_knowledge_graph`` is display-only, so the LLM phase
    still runs when it is off; only ``enable_concept_relations`` gates it."""
    from catgraphrag_phase import main

    handler = _make_handler(kg=False, cr=True)
    cat = _FakeCat(handler)
    points = [_FakePoint("doc1", vector=[0.1, 0.2, 0.3])]
    with patch.object(type(handler), "get_source_points_by_collection",
                      new=AsyncMock(return_value={"declarative": points})), \
         patch.object(type(handler), "_drain_source_tasks_counted", new=AsyncMock()), \
         patch.object(type(handler), "_extract_concept_relations", new=AsyncMock()) as extract:
        result = asyncio.run(main.ingestion_phase_run(
            "graph_llm_extraction", "file.pdf", _completed([]), cat
        ))
    assert result == {"status": "done"}, f"got {result}"
    extract.assert_awaited_once()


# ---------------------------------------------------------------------------
# T5 — completion gate + legacy fire-and-forget gating
# ---------------------------------------------------------------------------


def test_gate_raises_when_pending_tasks():
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)

    async def _run():
        task = asyncio.create_task(asyncio.sleep(30))
        handler._pending_source_tasks["file.pdf"] = [task]
        try:
            await main.before_ingestion_status_completed("file.pdf", cat)
            raise AssertionError("expected RuntimeError for pending tasks")
        except RuntimeError as e:
            assert "Pending background graph tasks" in str(e)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(_run())


def test_gate_passes_when_clean():
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    # no pending tasks, no diary check (EffING owns the diary): must pass
    asyncio.run(main.before_ingestion_status_completed("file.pdf", cat))


def test_gate_noop_when_not_graphrag_handler():
    from catgraphrag_phase import main

    class _OtherHandler:
        pass

    cat = _FakeCat(_OtherHandler())
    # must not raise even with a "pending" source (no graphrag handler)
    asyncio.run(main.before_ingestion_status_completed("file.pdf", cat))


def test_gate_raises_when_has_pending_source_tasks_true():
    """T7: the gate must fail-hard whenever has_pending_source_tasks is True,
    regardless of how the pending state is represented."""
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    with patch.object(type(handler), "has_pending_source_tasks",
                      new=Mock(return_value=True)):
        try:
            asyncio.run(main.before_ingestion_status_completed("file.pdf", cat))
            raise AssertionError("expected RuntimeError when has_pending_source_tasks is True")
        except RuntimeError as e:
            assert "Pending background graph tasks" in str(e)


def test_gate_does_not_import_core_registry():
    """T7: the completion gate must not reach into the core ingestion-status
    registry (EffING owns the diary + staleness). Assert the forbidden
    references are absent from main.py's source."""
    import inspect
    from catgraphrag_phase import main

    src = inspect.getsource(main)
    forbidden = (
        "core_plugins." + "ingestion_status",
        "ingestion_status." + "fingerprints",
        "ingestion_" + "phase_pending",
        "cat_efficient_ingestion." + "registry",
        "cat_efficient_ingestion." + "fingerprints",
    )
    for ref in forbidden:
        assert ref not in src, f"forbidden core-registry reference found: {ref}"


def test_rabbithole_hook_noop_when_phase_path_active():
    """EffING installed + efficient engine: the classic fire-and-forget must NOT
    run (the registered phases own the graph work)."""
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    _RESOLVED_INGESTION["name"] = "EfficientIngestionConfiguration"
    with patch.object(type(handler), "create_derived_graph_for_source", new=AsyncMock()) as create:
        asyncio.run(main.after_rabbithole_stored_documents("file.pdf", [], cat))
        create.assert_not_awaited()


def test_rabbithole_hook_keeps_classic_path_when_effing_absent():
    """Without EffING importable, the classic fire-and-forget must run once."""
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    saved = sys.modules.pop("cat.plugins.cat_efficient_ingestion", None)
    try:
        with patch.object(type(handler), "create_derived_graph_for_source", new=AsyncMock()) as create:
            asyncio.run(main.after_rabbithole_stored_documents("file.pdf", [], cat))
            create.assert_awaited_once_with("file.pdf", [], cat)
    finally:
        if saved is not None:
            sys.modules["cat.plugins.cat_efficient_ingestion"] = saved


def test_rabbithole_hook_keeps_classic_path_with_core_engine():
    """EffING installed but the SYSTEM ingestion selection is the core engine:
    the classic fire-and-forget must run once (no dead path)."""
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    _RESOLVED_INGESTION["name"] = "BaseIngestionConfiguration"
    try:
        with patch.object(type(handler), "create_derived_graph_for_source", new=AsyncMock()) as create:
            asyncio.run(main.after_rabbithole_stored_documents("file.pdf", [], cat))
            create.assert_awaited_once_with("file.pdf", [], cat)
    finally:
        _RESOLVED_INGESTION["name"] = "EfficientIngestionConfiguration"


def test_phase_path_active_truth_table():
    """uninstalled -> False; installed+core engine -> False; installed+efficient -> True."""
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    saved = sys.modules.pop("cat.plugins.cat_efficient_ingestion", None)
    _StubLog.warnings.clear()
    try:
        _RESOLVED_INGESTION["name"] = "EfficientIngestionConfiguration"
        assert asyncio.run(main._phase_path_active(cat)) is False
        # The efficient engine is selected but not importable: never silent.
        assert any("cat_efficient_ingestion" in w for w in _StubLog.warnings), _StubLog.warnings
        _StubLog.warnings.clear()
        _RESOLVED_INGESTION["name"] = "BaseIngestionConfiguration"
        assert asyncio.run(main._phase_path_active(cat)) is False
        assert _StubLog.warnings == [], _StubLog.warnings
    finally:
        if saved is not None:
            sys.modules["cat.plugins.cat_efficient_ingestion"] = saved

    _RESOLVED_INGESTION["name"] = "BaseIngestionConfiguration"
    assert asyncio.run(main._phase_path_active(cat)) is False

    _RESOLVED_INGESTION["name"] = "EfficientIngestionConfiguration"
    assert asyncio.run(main._phase_path_active(cat)) is True


# ---------------------------------------------------------------------------
# T24 — old-store cleanup on switch
# ---------------------------------------------------------------------------


def test_transfer_cleanup_noop_on_failure():
    from catgraphrag_phase import main

    cat = _FakeCat(_make_handler())
    with patch("catgraphrag_phase.graphrag_handler.wipe_tenant_graph_data", new=AsyncMock()) as wipe:
        asyncio.run(main.after_vector_memory_transfer_on_agent(False, cat))
        wipe.assert_not_awaited()


def test_transfer_cleanup_noop_when_new_handler_is_graphrag():
    from catgraphrag_phase import main

    cat = _FakeCat(_make_handler())
    with patch("catgraphrag_phase.graphrag_handler.wipe_tenant_graph_data", new=AsyncMock()) as wipe:
        asyncio.run(main.after_vector_memory_transfer_on_agent(True, cat))
        wipe.assert_not_awaited()


def test_transfer_cleanup_wipes_on_success_when_new_handler_not_graphrag():
    from catgraphrag_phase import main

    cat = _FakeCat(None)
    with patch("catgraphrag_phase.graphrag_handler.wipe_tenant_graph_data", new=AsyncMock()) as wipe:
        asyncio.run(main.after_vector_memory_transfer_on_agent(True, cat))
        wipe.assert_awaited_once_with("agent_test")


def test_wipe_tenant_graph_data_skips_without_config():
    from catgraphrag_phase import graphrag_handler

    graphrag_handler._last_graphrag_config.pop("agent_test", None)
    with patch.object(graphrag_handler.GraphRAGHandler, "_ensure_connected", new=AsyncMock()) as conn:
        asyncio.run(graphrag_handler.wipe_tenant_graph_data("agent_test"))
        conn.assert_not_awaited()


def test_wipe_tenant_graph_data_wipes_tenant():
    from catgraphrag_phase import graphrag_handler

    graphrag_handler._last_graphrag_config["agent_test"] = {
        "neo4j_uri": "bolt://fake",
        "neo4j_user": "u",
        "neo4j_password": "p",
        "neo4j_database": "neo4j",
        "neo4j_kwargs": {},
    }

    class _FakeSessionCM:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *a):
            return False

    captured = {}

    async def _fake_drop(self, session):
        captured["handler"] = self

    try:
        with patch.object(graphrag_handler.GraphRAGHandler, "_ensure_connected", new=AsyncMock()) as conn, \
             patch.object(graphrag_handler.GraphRAGHandler, "_get_session",
                          new=Mock(return_value=_FakeSessionCM())), \
             patch.object(graphrag_handler.GraphRAGHandler, "_drop_tenant_data_in_session", new=_fake_drop):
                asyncio.run(graphrag_handler.wipe_tenant_graph_data("agent_test"))
        conn.assert_awaited_once()
        # the wipe handler must be scoped to the target agent
        assert captured["handler"].agent_id == "agent_test"
    finally:
        graphrag_handler._last_graphrag_config.pop("agent_test", None)


# ---------------------------------------------------------------------------
# T25 — graph-phase write idempotency (restart-safe re-run)
# ---------------------------------------------------------------------------


class _EmptyResult:
    """Minimal async result: no rows, no single record."""

    async def single(self):
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _RecordingTx:
    def __init__(self, session):
        self._session = session

    async def run(self, query, **params):
        self._session.queries.append(str(query))
        return _EmptyResult()


class _RecordingSession:
    """Captures every Cypher statement sent through run/execute_write."""

    def __init__(self):
        self.queries = []

    async def run(self, query, **params):
        self.queries.append(str(query))
        return _EmptyResult()

    async def execute_write(self, fn):
        await fn(_RecordingTx(self))


class _RecordingSessionCM:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *a):
        return False


def _capture_session(handler, session):
    """Patch the handler's _get_session to yield a recording session."""
    from unittest.mock import Mock

    return patch.object(
        type(handler),
        "_get_session",
        new=Mock(return_value=_RecordingSessionCM(session)),
    )


def test_concept_store_rerun_issues_identical_merge_queries():
    """Re-running the LLM concept step for the same source+gen must issue
    identical MERGE-based statements (no bare CREATE, no weight accumulation)."""
    from catgraphrag_phase import graphrag_handler

    handler = _make_handler()
    handler.agent_id = "agent_test"

    relations = {
        "concepts": [
            {"type": "CONCEPT", "text": "Overfitting", "_norm": "overfitting"},
            {"type": "CONCEPT", "text": "Generalization", "_norm": "generalization"},
        ],
        "relations": [
            {
                "type": "CAUSES",
                "origin": "Overfitting",
                "destination": "Generalization",
                "text": "Overfitting causes poor generalization",
            },
        ],
    }

    runs = []
    for _ in range(2):
        rec = _RecordingSession()
        with _capture_session(handler, rec):
            asyncio.run(handler._store_concept_relations(
                "agent_test",
                relations,
                source="file.pdf",
                document_ids=["doc1", "doc2"],
                concept_gen="gen_abc",
            ))
        runs.append(rec.queries)

    assert runs[0], "expected captured Cypher statements"
    # Re-running the same source+gen must issue the SAME statements.
    assert runs[0] == runs[1], "re-run must issue identical queries"

    all_queries = "\n".join(runs[0])
    # No bare CREATE anywhere in the concept-store writes.
    assert "CREATE (" not in all_queries, f"bare CREATE found: {all_queries}"
    # Entity / edge / provenance writes are MERGE-based.
    assert "MERGE (z:Entity" in all_queries
    assert "MERGE (s)-[r:RELATED_TO" in all_queries
    assert "MERGE (d)-[:PROVENANCE]->(s)" in all_queries
    # The weight write is idempotent: no `+ 0.5` accumulation on re-run.
    assert "SET r.weight = coalesce(r.weight, 1.0)" in all_queries
    assert "coalesce(r.weight, 1.0) + 0.5" not in all_queries


def test_similar_rel_query_is_merge_based():
    """The SIMILAR_TO_{gen} write (epoch.py create_similar_rel) must be a
    MERGE in both directions — re-running the graph phase for the same source
    must not duplicate similarity edges."""
    from catgraphrag_phase import graphrag_handler

    handler = _make_handler()
    query = handler._compile_query("create_similar_rel", "v1")
    assert "MERGE (a)-[r1:SIMILAR_TO_v1]->(b)" in query
    assert "MERGE (b)-[r2:SIMILAR_TO_v1]->(a)" in query
    assert "CREATE (" not in query


def test_derived_graph_writes_are_merge_based():
    """The structural derived graph (SourceFile / PART_OF / NEXT / CHILD_OF /
    HAS_SUMMARY) must issue only MERGE statements for a source re-run."""
    from catgraphrag_phase import graphrag_handler

    handler = _make_handler()
    handler.agent_id = "agent_test"

    class _FakePoint:
        def __init__(self, pid, chunk_index, parent_id=None):
            self.id = pid
            self.payload = {
                "metadata": {
                    "source": "file.pdf",
                    "chunk_index": chunk_index,
                    "parent_id": parent_id,
                }
            }

    points = [
        _FakePoint("doc1", 0),
        _FakePoint("doc2", 1, parent_id="doc1"),
        _FakePoint("doc3", 2),
    ]

    rec = _RecordingSession()
    with patch.object(type(handler), "_ensure_connected", new=AsyncMock()), \
         _capture_session(handler, rec):
        asyncio.run(handler.create_derived_graph_for_source("file.pdf", points))

    assert rec.queries, "expected captured Cypher statements"
    all_queries = "\n".join(rec.queries)
    # No bare CREATE in the derived-graph writes (Document CREATE lives in
    # add_point_to_tenant and is guarded by the document_id_unique constraint).
    assert "CREATE (" not in all_queries, f"bare CREATE found: {all_queries}"
    assert "MERGE (sf:SourceFile" in all_queries
    assert "MERGE (d)-[:PART_OF]->(sf)" in all_queries
    assert "MERGE (a)-[:NEXT]->(b)" in all_queries
    assert "MERGE (child)-[:CHILD_OF]->(parent)" in all_queries


def test_spacy_batch_query_keeps_concept_gen_removal():
    """T7 shared-schema guard: the spaCy entity batch write must keep
    REMOVE-ing the LLM-owned concept_gen markers. The Entity-node contract
    (concept_gen / concept_gen_active) stays internal to the plugin: the
    spaCy phase must never resurrect stale LLM generations."""
    import inspect
    from catgraphrag_phase import graphrag_handler

    # Every spaCy write (single document and per-source batch) goes through
    # the shared transaction body, so the guard lives there.
    src = inspect.getsource(graphrag_handler.GraphRAGHandler._write_graph_plan_in_tx)
    assert "REMOVE e.concept_gen, e.concept_gen_active" in src, (
        "spaCy batch entity query must REMOVE the LLM-owned concept_gen markers"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    _install_stubs()

    # In-memory package pointing at the repo root: imports the plugin modules
    # directly (relative imports resolved via __path__) without executing the
    # plugin __init__.py side effects.
    _pkg = types.ModuleType("catgraphrag_phase")
    _pkg.__path__ = [REPO_ROOT]
    sys.modules["catgraphrag_phase"] = _pkg

    from catgraphrag_phase import graphrag_handler  # noqa: E402
    from catgraphrag_phase import main as main_mod  # noqa: E402

    tests = [
        test_specs_registers_two_graph_phases,
        test_specs_preserves_prior_accumulator_entries,
        test_specs_returns_accumulator_when_effing_absent,
        test_specs_returns_accumulator_when_phases_import_raises,
        test_marker_returns_spacy_fingerprint,
        test_marker_returns_concept_fingerprint,
        test_marker_returns_none_for_unknown_phase,
        test_marker_returns_none_when_not_graphrag_handler,
        test_phase_run_returns_none_for_unknown_phase,
        test_phase_run_returns_done_when_not_graphrag_handler,
        test_phase_run_dispatches_spacy_to_run_spacy_extraction_phase,
        test_phase_run_dispatches_llm_to_run_llm_extraction_phase,
        test_spacy_phase_not_ready_when_points_absent,
        test_spacy_phase_done_when_points_present,
        test_spacy_phase_skips_ner_when_entity_extraction_off,
        test_llm_phase_not_ready_when_points_absent,
        test_llm_phase_done_when_points_present,
        test_llm_phase_runs_when_kg_off_display_only,
        test_gate_raises_when_pending_tasks,
        test_gate_passes_when_clean,
        test_gate_noop_when_not_graphrag_handler,
        test_gate_raises_when_has_pending_source_tasks_true,
        test_gate_does_not_import_core_registry,
        test_rabbithole_hook_noop_when_phase_path_active,
        test_rabbithole_hook_keeps_classic_path_when_effing_absent,
        test_rabbithole_hook_keeps_classic_path_with_core_engine,
        test_phase_path_active_truth_table,
        test_transfer_cleanup_noop_on_failure,
        test_transfer_cleanup_noop_when_new_handler_is_graphrag,
        test_transfer_cleanup_wipes_on_success_when_new_handler_not_graphrag,
        test_wipe_tenant_graph_data_skips_without_config,
        test_wipe_tenant_graph_data_wipes_tenant,
        test_concept_store_rerun_issues_identical_merge_queries,
        test_similar_rel_query_is_merge_based,
        test_derived_graph_writes_are_merge_based,
        test_spacy_batch_query_keeps_concept_gen_removal,
    ]

    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()

    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()