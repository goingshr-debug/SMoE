from transformers import DynamicCache
from typing import Any, Dict, Optional, Tuple
import torch


class SMoECache(DynamicCache):
    """
    Drop-in replacement for DynamicCache that adds two capabilities:

    1. _readonly flag: when True, update() returns existing KV without writing.
       This allows prefetch attention runs (if_update=False path) to work through
       the vanilla attention code unchanged — the patcher sets/clears the flag
       around those calls instead of modifying the attention forward signature.

    2. attn_context: per-layer dict of attention outputs. The patcher stores
       residual/attn_weights/etc. here after each attention block so that
       DeepseekMoEwithCache can read them without receiving them as explicit args.
    """

    def __init__(self):
        super().__init__()
        self._readonly: bool = False
        self._attn_context: Dict[int, Dict[str, Any]] = {}

    @classmethod
    def from_legacy_cache(cls, past_key_values=None):
        cache = cls()
        if past_key_values is None:
            return cache
        if isinstance(past_key_values, DynamicCache):
            # Copy key/value tensors directly from DynamicCache internal storage
            for layer_idx in range(len(past_key_values.key_cache)):
                cache.update(past_key_values.key_cache[layer_idx],
                             past_key_values.value_cache[layer_idx],
                             layer_idx)
        else:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        if_update_or_cache_kwargs=None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # modeling_deepseek.py calls update(k, v, layer_idx, if_update, cache_kwargs)
        # Standard transformers calls update(k, v, layer_idx, cache_kwargs)
        # Detect which calling convention is used by checking argument type.
        if isinstance(if_update_or_cache_kwargs, bool):
            # Non-standard: if_update_or_cache_kwargs is actually if_update (bool)
            if_update = if_update_or_cache_kwargs
            real_cache_kwargs = cache_kwargs
        else:
            # Standard: if_update_or_cache_kwargs is cache_kwargs (dict or None)
            if_update = True
            real_cache_kwargs = if_update_or_cache_kwargs

        if self._readonly or not if_update:
            # Prefetch path: read existing KV without writing new tokens.
            if layer_idx < len(self.key_cache):
                return self.key_cache[layer_idx], self.value_cache[layer_idx]
            # Layer not yet cached — fall back to a normal write.
        return super().update(key_states, value_states, layer_idx, real_cache_kwargs)

    def set_attn_context(self, layer_idx: int, **kwargs) -> None:
        """Called by the patched DeepseekDecoderLayer before invoking the MoE MLP."""
        self._attn_context[layer_idx] = kwargs

    def get_attn_context(self, layer_idx: int) -> Dict[str, Any]:
        """Called by DeepseekMoEwithCache to retrieve attention outputs."""
        return self._attn_context.get(layer_idx, {})

    def get_max_length(self) -> Optional[int]:
        """
        Called by DeepseekForCausalLM.prepare_inputs_for_generation.
        DynamicCache has no fixed max length — return None to indicate unlimited.
        """
        return None

