"""
modeling_qwen.py — re-export Qwen2MoeForCausalLM from official transformers library.
"""
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM  # noqa: F401

__all__ = ["Qwen2MoeForCausalLM"]
