"""End-to-end proof that the two plugins are independent on the REAL app.

F3b — ``cat_graphrag`` without ``cat_alog``
    The app boots with an explicit ``system:agent`` ``active_plugins`` list that
    contains every core plugin plus ``cat_graphrag`` and NO ``cat_alog``. A test
    agent on ``Neo4jGraphRAGConfig`` ingests a 3-line text file carrying a
    unique token, then a recall whose query contains that token runs on the
    real GraphRAG handler: at least one returned chunk must contain the token.
    The app log captured during the run must contain no ``Hook ... not present``
    and no ``run_in_ingestion_executor``.

F3c — ``cat_alog`` without ``cat_graphrag``
    A second app boots with the core defaults only (no ``cat_graphrag``), a
    temporary agent uses the core's DEFAULT vector backend (Qdrant), ``cat_alog``
    is activated on it, the same text file is ingested and the real
    ``list_loaded_files`` tool must list the file (and the token, i.e. it really
    read the ingested chunk, not just the file manager entry).

Standalone script (plugin rules): the host imports every ``.py`` under the
plugin folder, so the ENTIRE body lives inside ``if __name__ == "__main__":``
and importing this file executes zero code. Run it from the host checkout:

    cd /root/git/grinning-cat-core
    CAT_REDIS_DB=15 .venv/bin/python \\
        /root/git/plugins/cat-graphrag-port/tests/test_independence_no_catalog.py

Isolation
---------
* Redis: forced to the dedicated ``CAT_REDIS_DB=15`` BEFORE ``cat`` is imported;
  every operation aborts if the bound db is not 15. This script NEVER calls
  ``flushdb``/``flushall``: it resets only its own namespaces
  (``system:*`` and ``agents:independence_*``) through ``cat.db.crud``.
* Qdrant: a fresh in-memory client per scenario (same double as the host
  conftest); its collections die with the client.
* Neo4j: the live instance at ``bolt://127.0.0.1:7687``. Isolation is a unique
  tenant id (``independence_probe_<uuid>``) AND unique vector index names
  (``doc``/``entity`` index bases are suffixed with the run id), so the shared
  ``document_embeddings_v1`` / ``entity_embeddings_v1`` of the real tenants are
  never touched. Cleanup deletes only this run's tenant nodes and index names.
* Every failure is explicit: no silent skip, no "returned something" assertion
  in place of the token check.
"""

if __name__ == "__main__":
    import asyncio
    import json
    import os
    import re
    import shutil
    import subprocess
    import sys
    import time
    import traceback
    import uuid
    from contextlib import asynccontextmanager
    from typing import Any, LiteralString, cast
    from inspect import iscoroutinefunction
    from pathlib import Path

    PLUGINS_ROOT = Path("/root/git/plugins")

    #: The plugin root: this file lives in ``tests/`` of the GraphRAG port.
    REPO_ROOT = Path(__file__).resolve().parent.parent

    GRAPH_PLUGIN_ID = "cat_graphrag"
    GRAPH_PLUGIN_SRC = PLUGINS_ROOT / "cat-graphrag-port"
    ALOG_PLUGIN_ID = "cat_alog"
    ALOG_PLUGIN_SRC = PLUGINS_ROOT / "cat_alog"

    # ---------------------------------------------------------------------------
    # Environment BEFORE importing cat (cat.startup builds an app at import time).
    # Redis: dedicated db 15, NEVER production. No flushdb/flushall is used.
    # ---------------------------------------------------------------------------
    TEST_REDIS_DB = "15"


    def _resolve_redis_host():
        explicit = os.environ.get("CAT_REDIS_HOST")
        if explicit:
            return explicit
        for name in ("grinning_cat_redis", "dev-grinning-cat-redis-1", "redis"):
            try:
                out = subprocess.run(
                    ["docker", "inspect", "-f",
                     "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", name],
                    capture_output=True, text=True, timeout=5,
                )
            except Exception:  # noqa: BLE001 - fall through to localhost
                continue
            ip = out.stdout.strip()
            if out.returncode == 0 and ip:
                return ip
        return "127.0.0.1"


    os.environ.setdefault("CAT_JWT_SECRET", "test_only_jwt_secret_0123456789_abcdefghij")
    # Settings crypto: the core encrypts secret settings at rest; test-only values
    # so the script runs with the documented command alone (db 15 only).
    os.environ.setdefault("CAT_CRYPTO_SALT", "independence_test_salt_0123456789abcdef")
    os.environ.setdefault("CAT_CRYPTO_KEY", "independence_test_key_0123456789abcdef")
    os.environ.setdefault("CAT_API_KEY", "meow_http")
    os.environ.setdefault("CAT_ADMIN_DEFAULT_PASSWORD", "wandering_in_wonderland")
    os.environ["CAT_REDIS_DB"] = TEST_REDIS_DB
    os.environ.setdefault("CAT_REDIS_HOST", _resolve_redis_host())
    os.environ.setdefault("CAT_REDIS_PORT", "6379")
    os.environ.setdefault("CAT_QDRANT_HOST", "grinning_cat_vector_memory")

    from asgi_lifespan import LifespanManager
    from httpx import ASGITransport, AsyncClient
    from neo4j import AsyncGraphDatabase
    from qdrant_client import AsyncQdrantClient

    from cat import utils
    from cat.auth import auth_utils
    from cat.db import crud
    from cat.db.database import DEFAULT_SYSTEM_KEY, Database, get_sync_db
    import cat.db.database as db_module
    from cat.db.models import Setting
    from cat.db.cruds import settings as crud_settings
    from cat.looking_glass.mad_hatter.plugin import Plugin
    from cat.routes import routes_utils
    from cat.services.factory.vector_db import QdrantHandler
    from cat.services.memory.models import VectorMemoryType
    from cat.startup import create_app

    # ---------------------------------------------------------------------------
    # Constants
    # ---------------------------------------------------------------------------
    PLUGINS_DIR = Path(utils.get_base_path()) / "plugins"

    #: links created by this run; must be gone at the end (cleanup receipt)
    CREATED_LINKS: list[str] = []
    #: agents created by this run; their namespaces are the only ones cleaned
    CREATED_AGENTS: set[str] = set()
    SCOPE_PREFIX = "independence_"
    AGENT_PREFIX = "independence_probe_"
    CATALOG_AGENT_PREFIX = "independence-check-"
    #: every test-agent prefix (minus the "agents:" key prefix); the catalog
    #: agent uses a hyphen, so a single underscore glob would miss it
    AGENT_KEY_PREFIXES = (SCOPE_PREFIX, CATALOG_AGENT_PREFIX)

    NEO4J_URI = "bolt://127.0.0.1:7687"
    NEO4J_USER = "neo4j"
    NEO4J_DATABASE = "neo4j"
    NEO4J_AUTH_CONTAINERS = ("neo4j-db", "dev-neo4j-1", "minimal-neo4j-1")

    DECLARATIVE = str(VectorMemoryType.DECLARATIVE)

    #: unique per run, so this run's Neo4j vector indexes never collide with the
    #: shared ``document_embeddings_v1`` / ``entity_embeddings_v1`` of real tenants
    RUN_ID = uuid.uuid4().hex[:10]
    DOC_INDEX_BASE = f"independence_doc_{RUN_ID}"
    ENT_INDEX_BASE = f"independence_ent_{RUN_ID}"
    #: The versioned schema names the index <base>_<generation> on the property
    #: embedding_<generation>. The live Neo4j already has a vector index on
    #: Document.embedding_v1 (the real tenants), and Neo4j silently refuses a
    #: second vector index on the same label+property: a unique GENERATION gives
    #: this run its own property and therefore its own index, tenant-scoped by
    #: the Epoch node the plugin itself reads.
    GRAPH_GENERATION = f"probe{RUN_ID}"

    TOKEN = f"independence-probe-{int(time.time())}-{RUN_ID}"
    PROBE_FILE = f"independence_probe_{RUN_ID}.txt"
    PROBE_TEXT = (
        f"{TOKEN} opens the probe file.\n"
        "The second line exists only to force a real multi-chunk document.\n"
        "The third line closes the independence probe file.\n"
    ).encode()

    # The suite expects to run from the host checkout (file storage root).
    _core_root = Path("/root/git/grinning-cat-core")
    if Path.cwd() != _core_root and (_core_root / "cat").exists():
        os.chdir(_core_root)


    # ---------------------------------------------------------------------------
    # Minimal pytest-monkeypatch equivalent (setattr + undo)
    # ---------------------------------------------------------------------------
    _UNSET = object()


    class _MonkeyPatch:
        def __init__(self):
            self._undo = []

        def setattr(self, target, name, value=None, raising=True):
            old = getattr(target, name, _UNSET)
            if old is _UNSET and raising:
                raise AttributeError(f"{target!r} has no attribute {name!r}")
            self._undo.append((target, name, old))
            setattr(target, name, value)

        def undo(self):
            while self._undo:
                target, name, old = self._undo.pop()
                if old is _UNSET:
                    try:
                        delattr(target, name)
                    except AttributeError:
                        pass
                else:
                    setattr(target, name, old)


    # ---------------------------------------------------------------------------
    # Host test doubles + inlined fixtures
    # ---------------------------------------------------------------------------
    def _install_host_doubles(mp, memory_client):
        def _mock_qdrant_init(self, *args, **kwargs):
            super(QdrantHandler, self).__init__()
            self._client = memory_client
            self.save_memory_snapshots = kwargs.get("save_memory_snapshots", False)

        mp.setattr(QdrantHandler, "__init__", _mock_qdrant_init)
        mp.setattr(QdrantHandler, "close", lambda self: None)

        def _mock_get_redis_kwargs():
            return {
                "host": os.environ["CAT_REDIS_HOST"],
                "port": int(os.environ["CAT_REDIS_PORT"]),
                "db": TEST_REDIS_DB,
                "encoding": "utf-8",
                "decode_responses": True,
            }

        mp.setattr(db_module, "get_redis_kwargs", _mock_get_redis_kwargs)
        mp.setattr(utils, "get_plugins_path", lambda: "tests/mocks/mock_plugin_folder/")
        mp.setattr(utils, "get_file_manager_root_storage_path", lambda: "tests/data/storage")

        async def _noop_install_requirements(self):
            return None

        mp.setattr(Plugin, "_install_requirements", _noop_install_requirements)
        mp.setattr(auth_utils, "extract_agent_id_from_request", lambda *a, **k: "agent_test")

        async def _immediate_task(background_tasks, func, *args, **kwargs):
            if iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)

        mp.setattr(routes_utils, "run_background_task", _immediate_task)


    def _create_plugin_links():
        """Link the two plugins into the world plugins folder (never core_plugins).

        World plugins are discovered but only loaded when they are in the active
        list, so ``cat_alog`` is genuinely absent from the F3b boot while its
        folder is still on disk (the same discovery surface the deployment has).
        """
        for plugin_id, target in ((GRAPH_PLUGIN_ID, GRAPH_PLUGIN_SRC), (ALOG_PLUGIN_ID, ALOG_PLUGIN_SRC)):
            link = PLUGINS_DIR / plugin_id
            if link.exists() or link.is_symlink():
                current = os.readlink(link) if link.is_symlink() else str(link)
                if Path(current).resolve() != target.resolve():
                    raise AssertionError(f"{link} already exists and points to {current}")
                continue
            link.symlink_to(target, target_is_directory=True)
            CREATED_LINKS.append(str(link))


    def _remove_plugin_links():
        for link in list(CREATED_LINKS):
            if os.path.lexists(link):
                os.remove(link)
            if link in CREATED_LINKS:
                CREATED_LINKS.remove(link)


    def _core_plugin_ids() -> list[str]:
        return sorted(
            p.name for p in Path(utils.get_core_plugins_path()).iterdir() if p.is_dir()
        )


    async def _seed_system_active_plugins(extra: list[str]) -> list[str]:
        """Boot the system agent with an EXPLICIT active list.

        An empty/missing list makes the core activate every folder in the
        plugins dir (``load_active_plugins_ids_from_folders``), which would
        silently re-include ``cat_alog``. Seeding the list is what makes the
        exclusion real. Non-toggleable plugins are force-added by the core.
        """
        value = _core_plugin_ids() + [p for p in extra if p not in _core_plugin_ids()]
        await crud_settings.upsert_setting_by_name(
            DEFAULT_SYSTEM_KEY, Setting(name="active_plugins", value=value),
        )
        return value


    def _bound_db() -> str:
        db = get_sync_db()
        return str(db.connection_pool.connection_kwargs.get("db", os.environ.get("CAT_REDIS_DB")))


    def _assert_test_db() -> None:
        bound = _bound_db()
        if bound != TEST_REDIS_DB:
            raise RuntimeError(
                f"Refusing to run: bound Redis db={bound!r}, expected the dedicated test db={TEST_REDIS_DB!r}."
            )


    async def _reset_scoped_state() -> None:
        # Scoped cleanup only: this suite's namespaces, never FLUSHDB/FLUSHALL.
        _assert_test_db()
        await crud.destroy("system:*")
        for prefix in AGENT_KEY_PREFIXES:
            await crud.destroy(f"agents:{prefix}*")


    def _remove_storage_dirs():
        storage = Path("tests/data/storage")
        if not storage.exists():
            return
        leftovers = [p for prefix in AGENT_KEY_PREFIXES for p in storage.glob(f"{prefix}*")]
        for child in leftovers:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        # prune only the directories this run may have created (never a populated one)
        for parent in (storage, Path("tests/data")):
            try:
                parent.rmdir()
            except OSError:
                pass


    @asynccontextmanager
    async def _scenario_env(extra_plugins: list[str]):
        """Inlined conftest + fixtures: one fresh app, same setup/teardown order."""
        mp = _MonkeyPatch()
        memory_client = AsyncQdrantClient(":memory:")
        app = None
        previous_api_key = os.environ.get("CAT_API_KEY")
        previous_jwt = os.environ.get("CAT_JWT_SECRET")
        try:
            utils.singleton.instances.clear()
            _install_host_doubles(mp, memory_client)

            _create_plugin_links()
            # restore the real plugins folder so the plugins are discovered by the app
            mp.setattr(utils, "get_plugins_path", lambda: str(PLUGINS_DIR))

            os.environ["CAT_API_KEY"] = api_key
            os.environ["CAT_JWT_SECRET"] = jwt_secret

            await _reset_scoped_state()
            await _seed_system_active_plugins(extra_plugins)

            app = create_app()
            async with LifespanManager(app) as manager:
                async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac:
                    setattr(ac, "_fastapi_test_app", app)
                    _assert_test_db()
                    yield ac, mp
        finally:
            if previous_api_key is None:
                os.environ.pop("CAT_API_KEY", None)
            else:
                os.environ["CAT_API_KEY"] = previous_api_key
            if previous_jwt is None:
                os.environ.pop("CAT_JWT_SECRET", None)
            else:
                os.environ["CAT_JWT_SECRET"] = previous_jwt
            _remove_plugin_links()
            _remove_storage_dirs()
            try:
                _assert_test_db()
                for prefix in AGENT_KEY_PREFIXES:
                    await crud.destroy(f"agents:{prefix}*")
                await crud.destroy("system:*")
            finally:
                collections = await memory_client.get_collections()
                for collection in collections.collections:
                    await memory_client.delete_collection(collection.name)
                await memory_client.close()
                mp.undo()
                utils.singleton.instances.clear()


    # Host test utilities inlined from tests.utils (standalone script).
    api_key = "meow_http"
    jwt_secret = "meow_jwt_test_secret_at_least_32_characters"

    # ---------------------------------------------------------------------------
    # Neo4j helpers (real instance; tenant- and index-scoped cleanup in finally)
    # ---------------------------------------------------------------------------
    def _neo4j_password() -> str:
        password = os.environ.get("CAT_NEO4J_PASSWORD") or os.environ.get("NEO4J_PASSWORD")
        if password:
            return password

        for container in NEO4J_AUTH_CONTAINERS:
            try:
                proc = subprocess.run(
                    ["docker", "inspect", container, "--format", "{{json .Config.Env}}"],
                    capture_output=True, text=True, timeout=10,
                )
            except Exception:  # noqa: BLE001 - try the next container
                continue
            if proc.returncode != 0:
                continue
            try:
                env = json.loads(proc.stdout)
            except json.JSONDecodeError:
                continue
            for entry in env:
                if entry.startswith("NEO4J_AUTH="):
                    _, _, password = entry.partition("=")[2].partition("/")
                    if password:
                        return password

        raise RuntimeError(
            "Neo4j password not found: set CAT_NEO4J_PASSWORD/NEO4J_PASSWORD or run "
            f"one of the local containers {NEO4J_AUTH_CONTAINERS} with NEO4J_AUTH."
        )


    def _neo4j_driver():
        return AsyncGraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, _neo4j_password()), connection_timeout=10,
        )


    async def _assert_neo4j_reachable() -> None:
        driver = _neo4j_driver()
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:
                result = await session.run("RETURN 1 AS x")
                await result.single()
                await result.consume()
        except Exception as exc:  # noqa: BLE001 - surfaced explicitly, never silently skipped
            raise AssertionError(f"Neo4j is unreachable at {NEO4J_URI}: {exc!r}")
        finally:
            await driver.close()


    async def _wipe_tenant(tenant: str) -> dict[str, Any]:
        driver = _neo4j_driver()
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:
                result = await session.run("MATCH (n {tenant_id: $tenant}) DETACH DELETE n", tenant=tenant)
                await result.consume()
        finally:
            await driver.close()
        return await _count_tenant(tenant)


    async def _count_tenant(tenant: str) -> dict[str, Any]:
        driver = _neo4j_driver()
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:
                async def one(query):
                    result = await session.run(query, tenant=tenant)
                    record = await result.single()
                    await result.consume()
                    return record["c"] if record else 0

                return {
                    "nodes": await one("MATCH (n {tenant_id: $tenant}) RETURN count(n) AS c"),
                    "relationships": await one(
                        "MATCH (a {tenant_id: $tenant})-[r]->(b {tenant_id: $tenant}) RETURN count(r) AS c"
                    ),
                }
        finally:
            await driver.close()


    async def _count_tenant_prefix(prefix: str) -> int:
        driver = _neo4j_driver()
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:
                result = await session.run(
                    "MATCH (n) WHERE n.tenant_id STARTS WITH $prefix RETURN count(n) AS c",
                    prefix=prefix,
                )
                record = await result.single()
                await result.consume()
                return record["c"] if record else 0
        finally:
            await driver.close()


    async def _drop_test_indexes() -> list[str]:
        """Drop ONLY this run's vector indexes (never the shared production ones)."""
        names = [
            f"{DOC_INDEX_BASE}_{GRAPH_GENERATION}", f"{ENT_INDEX_BASE}_{GRAPH_GENERATION}",
            f"{DOC_INDEX_BASE}_v1", f"{ENT_INDEX_BASE}_v1",
            DOC_INDEX_BASE, ENT_INDEX_BASE,
        ]
        driver = _neo4j_driver()
        dropped = []
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:
                for name in names:
                    result = await session.run(cast(LiteralString, f"DROP INDEX `{name}` IF EXISTS"))
                    await result.consume()
                    dropped.append(name)
                result = await session.run(
                    "SHOW INDEXES YIELD name WHERE name IN $names RETURN collect(name) AS left",
                    names=names,
                )
                record = await result.single()
                await result.consume()
                leftover = record["left"] if record else []
                assert not leftover, f"test indexes survived the cleanup: {leftover}"
        finally:
            await driver.close()
        return dropped


    # ---------------------------------------------------------------------------
    # Log capture
    # ---------------------------------------------------------------------------
    class _LogCapture:
        """Capture every loguru record emitted while the block runs."""

        def __init__(self):
            self.lines: list[str] = []
            self._sink_id = None

        def _sink(self, message):
            self.lines.append(str(message))

        def __enter__(self):
            from loguru import logger

            self._sink_id = logger.add(self._sink, level="DEBUG", format="{message}")
            return self

        def __exit__(self, *exc_info):
            from loguru import logger

            if self._sink_id is not None:
                logger.remove(self._sink_id)
            return False

        def forbidden(self) -> list[str]:
            patterns = (r"Hook .* not present", r"run_in_ingestion_executor")
            return [line for line in self.lines if any(re.search(p, line) for p in patterns)]


    # ---------------------------------------------------------------------------
    # App-level configuration helpers
    # ---------------------------------------------------------------------------
    def _headers(agent_key: str) -> dict[str, str]:
        return {"X-Agent-ID": agent_key, "Authorization": f"Bearer {api_key}"}


    def _neo4j_payload() -> dict[str, Any]:
        # Entity/concept extraction is disabled: the probe isolates the vector
        # plumbing (the point of F3b), and the live indexes are this run's own.
        return {
            "neo4j_uri": NEO4J_URI,
            "neo4j_user": NEO4J_USER,
            "neo4j_password": _neo4j_password(),
            "neo4j_database": NEO4J_DATABASE,
            "document_vector_index": DOC_INDEX_BASE,
            "entity_vector_index": ENT_INDEX_BASE,
            "enable_entity_extraction": False,
            "spacy_models": {"en": "en_core_web_sm"},
            "enable_knowledge_graph": False,
            "enable_concept_relations": False,
        }


    async def _activate_plugin(lizard, plugin_id: str) -> None:
        if plugin_id not in lizard.plugin_manager.active_plugins:
            await lizard.plugin_manager.activate_plugin(plugin_id)


    async def _set_neo4j(client, agent_key: str) -> None:
        response = await client.put(
            "/vector_database/settings/Neo4jGraphRAGConfig",
            json=_neo4j_payload(),
            headers=_headers(agent_key),
        )
        assert response.status_code == 200, response.text
        assert response.json()["name"] == "Neo4jGraphRAGConfig"


    async def _set_file_manager(client, agent_key: str) -> None:
        response = await client.put(
            "/file_manager/settings/LocalFileManagerConfig", json={}, headers=_headers(agent_key),
        )
        assert response.status_code == 200, response.text


    async def _seed_epoch_generation(tenant: str, generation: str) -> None:
        """Pin the test tenant's schema generation before the handler boots.

        The plugin reads it (``_read_generation``) and derives every versioned
        name from it. Unique per run and tenant-scoped, so nothing shared moves.
        """
        driver = _neo4j_driver()
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:
                result = await session.run(
                    "MERGE (e:Epoch {tenant_id: $tenant}) SET e.generation = $generation",
                    tenant=tenant, generation=generation,
                )
                await result.consume()
        finally:
            await driver.close()


    async def _assert_run_indexes(cat) -> list[dict[str, Any]]:
        """The handler's REAL initialize created this run's vector indexes.

        Verification only: a consumed ``CREATE`` is not needed (the plugin's DDL
        commits), and nothing is created here that the plugin did not create.
        """
        names = cat.vector_memory_handler._names
        wanted = [names["index"], names["entity_index"]]
        driver = _neo4j_driver()
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:
                result = await session.run(
                    "SHOW INDEXES YIELD name, state WHERE name IN $names RETURN name, state",
                    names=wanted,
                )
                rows = await result.data()
        finally:
            await driver.close()
        assert {row["name"] for row in rows} == set(wanted), rows
        print("GRAPH_INDEXES_READY", rows)
        return rows


    async def _configure_agent(client, agent_key: str, *, graphrag: bool = False, plugin_ids: tuple[str, ...] = ()):
        CREATED_AGENTS.add(agent_key)

        lizard = getattr(client, "_fastapi_test_app").state.lizard
        for plugin_id in plugin_ids:
            await _activate_plugin(lizard, plugin_id)

        cat = await lizard.create_cheshire_cat(agent_key)
        for plugin_id in plugin_ids:
            if plugin_id not in cat.plugin_manager.active_plugins:
                await cat.plugin_manager.activate_plugin(plugin_id)
            assert plugin_id in cat.plugin_manager.active_plugins, (plugin_id, cat.plugin_manager.active_plugins)

        if graphrag:
            await _set_neo4j(client, agent_key)
        # the upstream default file manager is a no-op; the prompt list / tool
        # need the real local storage
        await _set_file_manager(client, agent_key)
        cat = await lizard.get_cheshire_cat(agent_key)
        if graphrag:
            # The Neo4j config is applied AFTER the agent creation, so the
            # creation-time ``initialize`` ran on the default backend. Pin the
            # tenant generation, then bring the (now GraphRAG) handler up: the
            # real initialize creates THIS run's versioned vector indexes before
            # any ingestion worker can need them.
            from cat.services.factory.embedder import embedding_size

            await _seed_epoch_generation(agent_key, GRAPH_GENERATION)
            embedder = await cat.embedder()
            await cat.vector_memory_handler.initialize(embedder.name, await embedding_size(embedder))
            await _assert_run_indexes(cat)
        return cat


    async def _ingest(client, agent_key: str, name: str, data: bytes, content_type: str = "text/plain"):
        return await client.post(
            "/rabbithole/", files={"file": (name, data, content_type)}, headers=_headers(agent_key),
        )


    async def _status(agent_key: str, source: str) -> dict[str, Any]:
        from cat.db.cruds import ingestions as crud_ingestions

        return await crud_ingestions.get_status(agent_key, None, source) or {}


    async def _await_completed(agent_key: str, source: str, timeout: float = 180.0) -> dict[str, Any]:
        deadline = time.time() + timeout
        status: dict[str, Any] = {}
        while time.time() < deadline:
            status = await _status(agent_key, source)
            if status.get("status") in ("completed", "error"):
                return status
            await asyncio.sleep(0.2)
        raise AssertionError(f"Ingestion of {source!r} did not finish in {timeout}s: {status}")


    async def _recall(cat, query: str, k: int = 5):
        embedder = await cat.embedder()
        vector = await asyncio.to_thread(embedder.embed_query, query)
        return await cat.vector_memory_handler.recall_tenant_memory_from_embedding(
            DECLARATIVE, vector, metadata={"source": PROBE_FILE}, k=k,
        )


    def _delete_agent_namespace(agent_key: str) -> None:
        """Remove the agent's local file-storage folder (prefix-scoped)."""
        storage = Path("tests/data/storage")
        for child in list(storage.glob(f"{agent_key}*")):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)


    # ---------------------------------------------------------------------------
    # F3b — GRAPHRAG without CATALOG
    # ---------------------------------------------------------------------------
    async def _f3b_graphrag_without_catalog():
        await _assert_neo4j_reachable()
        agent_key = f"{AGENT_PREFIX}b_{RUN_ID}"
        cat = None
        try:
            async with _scenario_env([GRAPH_PLUGIN_ID]) as (client, mp):
                lizard = getattr(client, "_fastapi_test_app").state.lizard
                active = list(lizard.plugin_manager.active_plugins)
                assert GRAPH_PLUGIN_ID in active, active
                assert "base_plugin" in active, active
                assert "white_rabbit" in active and "march_hare" in active, active
                assert "mgmt_message" in active, active
                assert "memory" in active and "interactions" in active, active
                assert "conversation_history" in active and "analytics" in active and "why" in active, active
                # the exclusion is REAL: not loaded, not in the active list
                assert ALOG_PLUGIN_ID not in active, active
                assert ALOG_PLUGIN_ID not in lizard.plugin_manager.plugins, list(lizard.plugin_manager.plugins)

                cat = await _configure_agent(
                    client, agent_key, graphrag=True, plugin_ids=(GRAPH_PLUGIN_ID,),
                )
                assert ALOG_PLUGIN_ID not in cat.plugin_manager.active_plugins, cat.plugin_manager.active_plugins
                assert ALOG_PLUGIN_ID not in cat.plugin_manager.plugins, list(cat.plugin_manager.plugins)
                assert type(cat.vector_memory_handler).__name__ == "GraphRAGHandler", type(cat.vector_memory_handler)

                with _LogCapture() as logs:
                    response = await _ingest(client, agent_key, PROBE_FILE, PROBE_TEXT)
                    assert response.status_code == 200, response.text

                    status = await _await_completed(agent_key, PROBE_FILE)
                    assert status.get("status") == "completed", status

                    recalls = await _recall(cat, f"what does {TOKEN} say about the probe?")
                    contents = [(r.document.page_content or "") for r in recalls]
                    assert contents, "recall returned no chunk at all"
                    matched = [c for c in contents if TOKEN in c]
                    assert matched, f"no recalled chunk contains the token {TOKEN!r}: {contents}"

                forbidden = logs.forbidden()
                assert not forbidden, f"forbidden log lines during F3b: {forbidden}"
                assert logs.lines, "the log capture saw nothing: the sink is not wired"

                print("F3B_EVIDENCE", json.dumps({
                    "agent": agent_key,
                    "boot_active_plugins": active,
                    "cat_alog_loaded": False,
                    "vector_backend": type(cat.vector_memory_handler).__name__,
                    "status": status.get("status"),
                    "completed_phases": sorted(status.get("completed_phases") or []),
                    "recalled_chunks": len(contents),
                    "token_matches": len(matched),
                    "chunk_with_token": matched[0][:200],
                    "captured_log_lines": len(logs.lines),
                    "forbidden_log_lines": 0,
                    "graph_generation": GRAPH_GENERATION,
                    "neo4j_indexes": [
                        f"{DOC_INDEX_BASE}_{GRAPH_GENERATION}",
                        f"{ENT_INDEX_BASE}_{GRAPH_GENERATION}",
                    ],
                }))
                # proper teardown while the app and the handler are alive
                await cat.destroy()
                cat = None
                assert await _status(agent_key, PROBE_FILE) == {}, "ingestion state survived the agent delete"
        finally:
            if cat is not None:
                try:
                    await cat.destroy()
                except Exception:  # noqa: BLE001 - the direct wipe below is authoritative
                    traceback.print_exc()
            leftover = await _wipe_tenant(agent_key)
            assert leftover == {"nodes": 0, "relationships": 0}, leftover
            _delete_agent_namespace(agent_key)
            CREATED_AGENTS.discard(agent_key)


    # ---------------------------------------------------------------------------
    # F3c — CATALOG without GRAPHRAG
    # ---------------------------------------------------------------------------
    async def _f3c_catalog_without_graphrag():
        agent_key = f"{CATALOG_AGENT_PREFIX}{int(time.time())}-{RUN_ID}"
        cat = None
        try:
            async with _scenario_env([ALOG_PLUGIN_ID]) as (client, mp):
                lizard = getattr(client, "_fastapi_test_app").state.lizard
                active = lizard.plugin_manager.active_plugins
                assert ALOG_PLUGIN_ID in active, active
                assert GRAPH_PLUGIN_ID not in active, active
                assert GRAPH_PLUGIN_ID not in lizard.plugin_manager.plugins, list(lizard.plugin_manager.plugins)

                cat = await _configure_agent(
                    client, agent_key, graphrag=False, plugin_ids=(ALOG_PLUGIN_ID,),
                )
                assert GRAPH_PLUGIN_ID not in cat.plugin_manager.active_plugins, cat.plugin_manager.active_plugins
                backend = type(cat.vector_memory_handler).__name__
                assert backend == "QdrantHandler", f"expected the DEFAULT backend, got {backend}"

                response = await _ingest(client, agent_key, PROBE_FILE, PROBE_TEXT)
                assert response.status_code == 200, response.text
                status = await _await_completed(agent_key, PROBE_FILE)
                assert status.get("status") == "completed", status

                procedures = cat.plugin_manager.procedures_registry
                assert "list_loaded_files" in procedures, sorted(procedures)
                tool = procedures["list_loaded_files"]

                # The tool is SYNC and bridges via ``asyncio.run`` (the production
                # LangChain path runs it in a worker thread). The tool's own body
                # is untouched: only the Redis client singleton is reset around
                # the thread so it is not reused on a foreign event loop.
                def _call_tool():
                    Database().reset_async()
                    try:
                        return tool.func(cat)
                    finally:
                        Database().reset_async()

                output = await asyncio.to_thread(_call_tool)
                assert isinstance(output, str) and output.strip(), output
                assert PROBE_FILE in output, f"the ingested file is not listed:\n{output}"
                assert "**Content preview:**" in output, f"the tool listed no preview:\n{output}"

                # The preview of a catalogued source is the catalogue-card summary
                # (by design), so the token is asserted on the ingested chunk: the
                # file is not just on disk, its content really reached the vectors.
                records, _ = await cat.vector_memory_handler.get_all_tenant_points(
                    DECLARATIVE, metadata={"source": PROBE_FILE}, with_vectors=False,
                )
                chunk_texts = [
                    (r.payload or {}).get("page_content", "") or "" for r in (records or [])
                ]
                assert any(TOKEN in text for text in chunk_texts), (
                    f"the ingested chunk lost the token: {chunk_texts}"
                )

                print("F3C_EVIDENCE", json.dumps({
                    "agent": agent_key,
                    "boot_active_plugins": active,
                    "cat_graphrag_loaded": False,
                    "vector_backend": backend,
                    "status": status.get("status"),
                    "tool": "list_loaded_files",
                    "tool_lists_file": True,
                    "tool_preview_present": True,
                    "stored_chunks_with_token": sum(1 for t in chunk_texts if TOKEN in t),
                    "tool_output": output,
                }))
                # proper teardown while the app and the default handler are alive
                await cat.destroy()
                cat = None
        finally:
            if cat is not None:
                try:
                    await cat.destroy()
                except Exception:  # noqa: BLE001 - the namespace wipe below is authoritative
                    traceback.print_exc()
            _delete_agent_namespace(agent_key)
            CREATED_AGENTS.discard(agent_key)


    # ---------------------------------------------------------------------------
    # Cleanup receipt (runs last)
    # ---------------------------------------------------------------------------
    async def _cleanup_receipt():
        """No leftover symlinks, seeded keys/files/dirs, Neo4j tenants or indexes."""
        for link in list(CREATED_LINKS):
            assert not os.path.lexists(link), link

        if Path("tests/data").exists():
            for prefix in AGENT_KEY_PREFIXES:
                assert not list(Path("tests/data/storage").glob(f"{prefix}*")), prefix

        utils.singleton.instances.clear()
        _assert_test_db()
        agents_left = [
            key for prefix in AGENT_KEY_PREFIXES for key in await crud.scan_keys(f"agents:{prefix}*")
        ]
        system_left = await crud.scan_keys("system:*")
        assert agents_left == [], agents_left
        assert system_left == [], system_left

        await _assert_neo4j_reachable()
        tenants_left = await _count_tenant_prefix(SCOPE_PREFIX)
        assert tenants_left == 0, tenants_left
        dropped = await _drop_test_indexes()

        print("CLEANUP_RECEIPT", json.dumps({
            "redis_independence_keys": len(agents_left),
            "redis_system_keys": len(system_left),
            "neo4j_independence_tenants": tenants_left,
            "plugin_links": len(CREATED_LINKS),
            "neo4j_test_indexes_dropped": dropped,
        }))


    async def _main():
        results = []
        for name, func in (
            ("f3b_graphrag_without_catalog", _f3b_graphrag_without_catalog),
            ("f3c_catalog_without_graphrag", _f3c_catalog_without_graphrag),
        ):
            try:
                await func()
                results.append((name, True))
                print(f"PASS {name}")
            except Exception:  # noqa: BLE001 - report every scenario, never hide
                traceback.print_exc()
                results.append((name, False))
                print(f"FAIL {name}")

        try:
            await _cleanup_receipt()
            results.append(("cleanup_receipt", True))
            print("PASS cleanup_receipt")
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            results.append(("cleanup_receipt", False))
            print("FAIL cleanup_receipt")

        passed = sum(1 for _, ok in results if ok)
        print(f"\ntest_independence_no_catalog: {passed}/{len(results)} passed")
        if passed != len(results):
            raise SystemExit(1)


    try:
        asyncio.run(_main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"FAIL test_independence_no_catalog: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
