"""Make the real EffING plugin importable for a host run of the CLI harnesses.

The maintenance CLI drives EffING (``cat.plugins.cat_efficient_ingestion``)
directly, so its harnesses need the real plugin. Inside the Cat container it is
deployed; on the host, point ``EFFING_DIR`` and ``MULTIMODAL_DIR`` at checkouts
of ``cat-efficient-ingestion`` and of its dependency
``cat-multimodal-ingestion`` and they are registered under the module ids the
code imports (EffING's own code still imports the multimodal plugin from
``cat.core_plugins``). EffING needs the EffING-enabled core
(``escape_glob`` / ``CustomHookVetoException``), so run from such a checkout.

Import-safe: the Cat plugin loader imports every ``.py`` under the plugin
folder, so this module only defines functions.
"""

#: (module id, env var with the checkout, a file proving the checkout)
_CORE_PLUGINS = (
    ("cat.core_plugins.cat_multimodal_ingestion", "MULTIMODAL_DIR", "ingestion.py"),
    ("cat.plugins.cat_efficient_ingestion", "EFFING_DIR", "registry.py"),
)


def ensure_effing_importable() -> None:
    """Register the plugin checkouts under the module ids the code imports.

    No-op for a plugin that is already importable. Exits with a FAIL line (never
    a silent skip) when one is neither importable nor provided.
    """
    import importlib
    import os
    import sys
    import types

    for name, env, marker in _CORE_PLUGINS:
        try:
            importlib.import_module(name)
            continue
        except ImportError:
            pass

        src = os.environ.get(env)
        if not src or not os.path.isfile(os.path.join(src, marker)):
            print(
                f"FAIL: '{name}' is not importable: set {env} to a checkout of "
                f"the plugin (got {src!r})"
            )
            sys.exit(1)

        package = types.ModuleType(name)
        package.__path__ = [os.path.abspath(src)]
        package.__package__ = name
        sys.modules[name] = package
