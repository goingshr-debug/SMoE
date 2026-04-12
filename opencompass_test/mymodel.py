import os
import sys
from typing import Optional

import torch

# ── Locate SMoE project root ──────────────────────────────────────────────────
_SMOE_ROOT = os.environ.get('SMOE_ROOT')
if _SMOE_ROOT is None:
    raise EnvironmentError(
        "SMOE_ROOT environment variable is not set.\n"
        "Please set it to the SMoE project root directory, e.g.:\n"
        "  export SMOE_ROOT=/path/to/SMoE"
    )
if _SMOE_ROOT not in sys.path:
    sys.path.insert(0, _SMOE_ROOT)

# ── OpenCompass base class ────────────────────────────────────────────────────
# Relative import: this file is installed inside opencompass/models/
from .huggingface import HuggingFaceCausalLM

# ── SMoE model loader ─────────────────────────────────────────────────────────
from utils.model_loader import build_model

# SMoE config paths (contain window_size, replaceScoreRatio and other SMoE params)
_CFG_DEEPSEEK = os.path.join(_SMOE_ROOT, 'configs', 'deepseekmoe_config.json')
_CFG_XVERSE   = os.path.join(_SMOE_ROOT, 'configs', 'xversemoe_config.json')
_CFG_QWEN     = os.path.join(_SMOE_ROOT, 'configs', 'qwen2moe_config.json')


class MydeepseekmoeModel(HuggingFaceCausalLM):
    """OpenCompass wrapper for DeepSeek MoE with SMoE expert cache."""

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        self._set_model_kwargs_torch_dtype(model_kwargs)
        self.model = build_model(
            model_path=path,
            model_type="deepseekmoe",
            device=torch.device("cuda:0"),
            main_size=520,
            config_path=_CFG_DEEPSEEK,
        )
        self.model.eval()
        self.model.generation_config.do_sample = False


class MyxversemoeModel(HuggingFaceCausalLM):
    """OpenCompass wrapper for Xverse MoE with SMoE expert cache."""

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        self._set_model_kwargs_torch_dtype(model_kwargs)
        self.model = build_model(
            model_path=path,
            model_type="xversemoe",
            device=torch.device("cuda:0"),
            main_size=520,
            config_path=_CFG_XVERSE,
        )
        self.model.eval()
        self.model.generation_config.do_sample = False


class MyqwenmoeModel(HuggingFaceCausalLM):
    """OpenCompass wrapper for Qwen2 MoE with SMoE expert cache."""

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        self._set_model_kwargs_torch_dtype(model_kwargs)
        self.model = build_model(
            model_path=path,
            model_type="qwenmoe",
            device=torch.device("cuda:0"),
            main_size=520,
            config_path=_CFG_QWEN,
        )
        self.model.eval()
        self.model.generation_config.do_sample = False
