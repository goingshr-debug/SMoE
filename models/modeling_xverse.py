"""
modeling_xverse.py — lazy loader for XverseForCausalLM.

Xverse MoE is not in the official transformers library.
This module loads XverseForCausalLM from the model directory at runtime.
model_loader.py calls set_model_path(model_path) before first access.
"""
from __future__ import annotations
import os, sys, importlib, importlib.util, types, logging

logger = logging.getLogger(__name__)
_XverseForCausalLM = None


def _load_xverse_class(model_path: str | None = None):
    global _XverseForCausalLM
    if _XverseForCausalLM is not None:
        return _XverseForCausalLM

    search = [model_path, os.environ.get("XVERSE_MODEL_PATH", "")]
    for path in search:
        if path and os.path.isfile(os.path.join(path, "modeling_xverse.py")):
            # The model files use relative imports (e.g. `from .configuration_xverse import …`).
            # importlib.util.spec_from_file_location with a bare name has no parent package,
            # so relative imports fail.  Fix: register the directory as a synthetic package
            # in sys.modules first, then load modeling_xverse.py as a sub-module of it.
            _PKG = "xversemoe_model_dir"
            if _PKG not in sys.modules:
                pkg = types.ModuleType(_PKG)
                pkg.__path__    = [path]
                pkg.__package__ = _PKG
                pkg.__file__    = os.path.join(path, "__init__.py")
                sys.modules[_PKG] = pkg

            def _load_sub(name):
                full_name = _PKG + "." + name
                if full_name in sys.modules:
                    return sys.modules[full_name]
                fpath = os.path.join(path, name + ".py")
                spec = importlib.util.spec_from_file_location(
                    full_name, fpath,
                    submodule_search_locations=[]
                )
                m = importlib.util.module_from_spec(spec)
                m.__package__ = _PKG
                sys.modules[full_name] = m
                spec.loader.exec_module(m)
                return m

            # Pre-load configuration_xverse so the relative import resolves
            _load_sub("configuration_xverse")

            mod = _load_sub("modeling_xverse")
            _XverseForCausalLM = mod.XverseForCausalLM
            logger.info("XverseForCausalLM loaded from: %s", path)
            return _XverseForCausalLM

    raise ImportError(
        "Cannot find XverseForCausalLM. "
        "Call modeling_xverse.set_model_path('/path/to/xversemoe') first, "
        "or set XVERSE_MODEL_PATH environment variable."
    )


class _LazyModule(types.ModuleType):
    _model_path: str | None = None

    @staticmethod
    def set_model_path(path: str):
        _LazyModule._model_path = path

    def __getattr__(self, name):
        if name == "XverseForCausalLM":
            return _load_xverse_class(self._model_path)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_this = _LazyModule(__name__)
_this.__file__ = __file__
_this.__package__ = __package__
_this.__spec__ = __spec__
sys.modules[__name__] = _this
