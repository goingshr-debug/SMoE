"""
qwen2_config.py — re-export Qwen2MoeConfig from the official transformers library.
"""
from transformers.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig  # noqa: F401

__all__ = ["Qwen2MoeConfig"]
