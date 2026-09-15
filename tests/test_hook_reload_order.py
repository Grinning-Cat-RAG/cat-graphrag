"""Standalone verification for the FX-7/FX-9 fixes: the hooks
``after_rabbithole_stored_documents`` (FX-7) and
``after_vector_database_settings_update`` (FX-9) survive the plugin loader's
reload order via a lazy in-function ``GraphRAGHandler`` import.

The MyCAT plugin loader (mad_hatter/plugin.py) walks the plugin's ``*.py``
files in glob order and ``importlib.reload``s each one. In this folder glob
order puts ``main.py`` BEFORE ``graphrag_handler.py``:

  1. main.py is (re)loaded -> it binds ``main.GraphRAGHandler`` to the
     PRE-reload class (A);
  2. graphrag_handler.py is reloaded afterwards -> the module now holds a NEW
     class (B);
  3. the factory instantiates the vector handler from the CURRENT class (B);
  4. a hook checking ``isinstance(handler, main.GraphRAGHandler)`` with a
     module-level import compares against the STALE class A -> ALWAYS False
     -> silent early return (FX-7: the hook has never run).

This harness reproduces that exact sequence (import main, then reload
graphrag_handler), asserts the stale-binding bug condition holds, then calls
the hook with a class-B handler and asserts ``create_derived_graph_for_source``
IS awaited — proving the lazy import resolves the current class.

Runnable:  python test_hook_reload_order.py
Pure-stdlib + unittest.mock, plain asserts, no pytest, no model loading. All
external packages (cat, langchain_core, spacy, langdetect, pydantic, neo4j)
are stubbed in sys.modules *before* the plugin modules are imported (same
pattern as test_kg_master_switch.py).
"""

import asyncio
import importlib
import os
import sys
import types
from unittest.mock import AsyncMock

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = "catgraphrag_reload"


class _StubLog:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _install_cat_stub():
    cat_mod = types.ModuleType("cat")
    cat_mod.__path__ = []  # look like a package root for submodule imports

    # @hook decorator: identity for both bare ``@hook`` and ``@hook(priority=N)``.
    def _hook(*args, **kwargs):
        if args and callable(args[0]):
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


def _install_stubs():
    _install_cat_stub()
    _install_neo4j_stub()
    _install_langchain_stub()
    _install_langdetect_stub()
    _install_spacy_stubs()
    _install_pydantic_stub()


# ---------------------------------------------------------------------------
# Reload-order simulation + hook tests
# ---------------------------------------------------------------------------


def _make_package():
    """In-memory package whose ``__path__`` points at the repo root: imports
    the plugin modules directly (relative imports resolved via ``__path__``)
    without executing the plugin ``__init__.py`` side effects."""
    pkg = types.ModuleType(PKG)
    pkg.__path__ = [REPO_ROOT]
    sys.modules[PKG] = pkg
    return pkg


def _make_handler(handler_mod, *, derived_graph: bool):
    """Instantiate a handler of the CURRENT (post-reload) class and stub the
    parts the hook touches."""
    handler = handler_mod.GraphRAGHandler(
        neo4j_uri="bolt://fake",
        neo4j_user="u",
        neo4j_password="p",
        enable_derived_graph=derived_graph,
    )
    handler._enable_derived_graph = derived_graph
    handler.agent_id = "agent_reload_test"
    handler.create_derived_graph_for_source = AsyncMock()
    return handler


def _make_settings_handler(handler_mod, *, active_gen="gen_old", new_gen="gen_new"):
    """Handler for the settings-update hook: fingerprint + persisted-marker
    reads stubbed so the hook reaches the scheduling branch."""
    handler = handler_mod.GraphRAGHandler(
        neo4j_uri="bolt://fake",
        neo4j_user="u",
        neo4j_password="p",
    )
    handler.agent_id = "agent_reload_test"
    handler._concept_fingerprint = lambda: new_gen
    handler._read_concept_gen = AsyncMock(return_value=active_gen)
    return handler


def test_hook_calls_derived_graph_after_reload_order():
    """Reproduces the loader's glob order (main BEFORE graphrag_handler) and
    proves the lazy in-function import resolves the CURRENT class, so the hook
    fires with a handler of that class (the FX-7 failure mode)."""
    importlib.import_module(f"{PKG}.main")  # binds main.GraphRAGHandler = class A
    handler_mod = importlib.import_module(f"{PKG}.graphrag_handler")
    importlib.reload(handler_mod)  # class B now lives in the module
    main = importlib.import_module(f"{PKG}.main")

    # The bug condition: main's module-level binding is the PRE-reload class.
    assert main.GraphRAGHandler is not handler_mod.GraphRAGHandler

    handler = _make_handler(handler_mod, derived_graph=True)
    cat = types.SimpleNamespace(vector_memory_handler=handler)

    asyncio.run(main.after_rabbithole_stored_documents("file.txt", [], cat))

    handler.create_derived_graph_for_source.assert_awaited_once_with(
        "file.txt", [], cat
    )


def test_hook_skips_when_derived_graph_disabled():
    """The FX-1 gate (_enable_derived_graph) still holds after the lazy-import
    fix."""
    main = importlib.import_module(f"{PKG}.main")
    handler_mod = importlib.import_module(f"{PKG}.graphrag_handler")

    handler = _make_handler(handler_mod, derived_graph=False)
    cat = types.SimpleNamespace(vector_memory_handler=handler)

    asyncio.run(main.after_rabbithole_stored_documents("file.txt", [], cat))

    handler.create_derived_graph_for_source.assert_not_awaited()


def test_hook_skips_non_graphrag_handler():
    """The isinstance gate still rejects non-GraphRAG handlers (no attribute
    access, no call)."""
    main = importlib.import_module(f"{PKG}.main")

    other = type("OtherHandler", (), {})()
    cat = types.SimpleNamespace(vector_memory_handler=other)

    asyncio.run(main.after_rabbithole_stored_documents("file.txt", [], cat))


def test_settings_hook_schedules_recompute_after_reload_order():
    """FX-9: reproduces the loader's glob order (main BEFORE graphrag_handler)
    and proves the lazy in-function import resolves the CURRENT class, so the
    settings-update hook passes the isinstance guard with a class-B handler and
    schedules the concept-relations recompute (the FX-9 failure mode: 0
    'scheduled recompute' lines in the F3 smoke)."""
    importlib.import_module(f"{PKG}.main")  # binds main.GraphRAGHandler = class A
    handler_mod = importlib.import_module(f"{PKG}.graphrag_handler")
    importlib.reload(handler_mod)  # class B now lives in the module
    main = importlib.import_module(f"{PKG}.main")

    # The bug condition: main's module-level binding is the PRE-reload class.
    assert main.GraphRAGHandler is not handler_mod.GraphRAGHandler

    handler = _make_settings_handler(handler_mod)
    cat = types.SimpleNamespace(
        vector_memory_handler=handler, agent_key="agent_reload_test"
    )

    # Capture the scheduled job without running it (no loop, no real graph).
    captured = {}

    def _fake_create_task(coro):
        captured["coro"] = coro
        coro.close()
        return types.SimpleNamespace()

    main._concept_recompute_job = AsyncMock()
    original_create_task = main.asyncio.create_task
    main.asyncio.create_task = _fake_create_task
    try:
        asyncio.run(
            main.after_vector_database_settings_update(
                "Neo4jGraphRAGConfig",
                {"concept_definitions": "old"},
                {"concept_definitions": "new"},
                cat,
            )
        )
    finally:
        main.asyncio.create_task = original_create_task

    main._concept_recompute_job.assert_called_once_with(
        handler, cat, "agent_reload_test", "gen_new"
    )
    assert len(handler._pending_entity_tasks) == 1, (
        "the scheduled task must be tracked for close()"
    )


def test_settings_hook_skips_when_no_concept_change():
    """No concept-related key changed -> no recompute scheduled."""
    main = importlib.import_module(f"{PKG}.main")
    handler_mod = importlib.import_module(f"{PKG}.graphrag_handler")

    handler = _make_settings_handler(handler_mod)
    cat = types.SimpleNamespace(
        vector_memory_handler=handler, agent_key="agent_reload_test"
    )

    main._concept_recompute_job = AsyncMock()
    asyncio.run(
        main.after_vector_database_settings_update(
            "Neo4jGraphRAGConfig",
            {"concept_definitions": "same"},
            {"concept_definitions": "same"},
            cat,
        )
    )
    main._concept_recompute_job.assert_not_called()
    assert handler._pending_entity_tasks == []


def test_settings_hook_skips_when_marker_already_active():
    """The persisted-marker no-op (active == new_gen) still holds after the
    lazy-import fix."""
    main = importlib.import_module(f"{PKG}.main")
    handler_mod = importlib.import_module(f"{PKG}.graphrag_handler")

    handler = _make_settings_handler(
        handler_mod, active_gen="gen_new", new_gen="gen_new"
    )
    cat = types.SimpleNamespace(
        vector_memory_handler=handler, agent_key="agent_reload_test"
    )

    main._concept_recompute_job = AsyncMock()
    asyncio.run(
        main.after_vector_database_settings_update(
            "Neo4jGraphRAGConfig",
            {"concept_definitions": "old"},
            {"concept_definitions": "new"},
            cat,
        )
    )
    main._concept_recompute_job.assert_not_called()
    assert handler._pending_entity_tasks == []


def test_settings_hook_skips_non_graphrag_handler():
    """The isinstance gate still rejects non-GraphRAG handlers (no scheduling,
    no attribute access)."""
    main = importlib.import_module(f"{PKG}.main")

    other = type("OtherHandler", (), {})()
    cat = types.SimpleNamespace(
        vector_memory_handler=other, agent_key="agent_reload_test"
    )

    main._concept_recompute_job = AsyncMock()
    asyncio.run(
        main.after_vector_database_settings_update(
            "Neo4jGraphRAGConfig",
            {"concept_definitions": "old"},
            {"concept_definitions": "new"},
            cat,
        )
    )
    main._concept_recompute_job.assert_not_called()


def main():
    _install_stubs()
    _make_package()

    tests = [
        test_hook_calls_derived_graph_after_reload_order,
        test_hook_skips_when_derived_graph_disabled,
        test_hook_skips_non_graphrag_handler,
        test_settings_hook_schedules_recompute_after_reload_order,
        test_settings_hook_skips_when_no_concept_change,
        test_settings_hook_skips_when_marker_already_active,
        test_settings_hook_skips_non_graphrag_handler,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
