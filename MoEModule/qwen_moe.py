"""
qwen_moe.py — SMoE expert module for Qwen2 MoE.

Provides:
  - Qwen2MoeMLP                        plain MLP expert
  - Qwen2MoeSparseMoeBlockwithCache    cache-aware MoE block (inherits AbstractMoELayer)
"""

import logging
import os

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import List, Optional, Tuple

from utils.expertcache import (
    ExpertCache,
    replaceset_between_tokens,
)
from MoEModule.SMoE_base import AbstractMoELayer

logger = logging.getLogger(__name__)

ExpertUID = Tuple[int, int]


class Qwen2MoeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size       = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size,
                                   bias=False, dtype=torch.bfloat16, device=config.device)
        self.up_proj   = nn.Linear(self.hidden_size, self.intermediate_size,
                                   bias=False, dtype=torch.bfloat16, device=config.device)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size,
                                   bias=False, dtype=torch.bfloat16, device=config.device)
        self.act_fn = ACT2FN[config.hidden_act]
        self._cpu_accel_packs = None
        self._cpu_accel_mode = "bf16"

    @staticmethod
    def _pack_dynamic_int8(weight: torch.Tensor):
        weight_float = weight.detach().float()
        weight_min, weight_max = torch.aminmax(weight_float)
        max_abs = max(abs(weight_min.item()), abs(weight_max.item()))
        scale = max(max_abs / 127.0, 1.0e-8)
        quantized = torch.quantize_per_tensor(
            weight_float, scale, 0, torch.qint8
        )
        return torch.ops.quantized.linear_prepack(quantized, None)

    @staticmethod
    def _pack_groupwise_int4(weight: torch.Tensor, group_size: int = 128):
        rows, cols = weight.shape
        if cols % group_size:
            raise ValueError(
                f"INT4 group_size={group_size} must divide in_features={cols}"
            )
        grouped = weight.detach().float().reshape(
            rows, cols // group_size, group_size
        )
        group_min = grouped.amin(dim=-1)
        group_max = grouped.amax(dim=-1)
        scales = ((group_max - group_min) / 15.0).clamp_min(1.0e-8)
        zeros = group_min + 8.0 * scales
        codes = torch.round(
            (grouped - zeros.unsqueeze(-1)) / scales.unsqueeze(-1) + 8.0
        )
        codes = codes.clamp_(0, 15).to(torch.int32).reshape(rows, cols)
        packed = torch.ops.aten._convert_weight_to_int4pack_for_cpu(codes, 8)
        scales_and_zeros = (
            torch.stack((scales, zeros), dim=-1)
            .transpose(0, 1)
            .to(torch.bfloat16)
            .contiguous()
        )
        return packed, scales_and_zeros

    @staticmethod
    def _linear_int4(x: torch.Tensor, packed, group_size: int = 128):
        weight, scales_and_zeros = packed
        return torch.ops.aten._weight_int4pack_mm_for_cpu(
            x, weight, group_size, scales_and_zeros
        )

    @torch.no_grad()
    def prepare_cpu_acceleration(self) -> bool:
        """Prepack CPU-only weights while retaining BF16 H2D backing."""
        if self.gate_proj.weight.device.type != "cpu":
            return False
        mode = os.environ.get("SMOE_CPU_QUANT", "int4").strip().lower()
        if mode == "int4":
            self._cpu_accel_packs = (
                self._pack_groupwise_int4(self.gate_proj.weight),
                self._pack_groupwise_int4(self.up_proj.weight),
                self._pack_groupwise_int4(self.down_proj.weight),
            )
        elif mode == "int8":
            self._cpu_accel_packs = (
                self._pack_dynamic_int8(self.gate_proj.weight),
                self._pack_dynamic_int8(self.up_proj.weight),
                self._pack_dynamic_int8(self.down_proj.weight),
            )
        else:
            return False
        self._cpu_accel_mode = mode
        return True

    def clear_cpu_acceleration(self):
        self._cpu_accel_packs = None
        self._cpu_accel_mode = "bf16"

    def forward(self, x):
        if x.device.type == "cpu" and self._cpu_accel_packs is not None:
            gate_pack, up_pack, down_pack = self._cpu_accel_packs
            if self._cpu_accel_mode == "int4":
                gate = self._linear_int4(x, gate_pack)
                up = self._linear_int4(x, up_pack)
                return self._linear_int4(self.act_fn(gate) * up, down_pack)
            x_float = x.float()
            gate = torch.ops.quantized.linear_dynamic(x_float, gate_pack, True)
            up = torch.ops.quantized.linear_dynamic(x_float, up_pack, True)
            output = torch.ops.quantized.linear_dynamic(
                self.act_fn(gate) * up, down_pack, True
            )
            return output.to(x.dtype)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2MoeSparseMoeBlockwithCache(AbstractMoELayer):
    """
    Qwen2 MoE block with expert cache.

    Qwen2-specific attributes vs AbstractMoELayer:
      - gate: nn.Linear
      - shared_expert: Qwen2MoeMLP + shared_expert_gate (gated output)
      - norm_topk_prob: from config
      - prefetch: lightweight predict (no attention, layernorm-only approx)
    """

    def __init__(self, config, expertcache_: ExpertCache, layerid,
                 gate: nn.Linear, shared_experts: Qwen2MoeMLP,
                 shared_expert_gate: nn.Linear,
                 next_attention, next_gate_weight,
                 next_input_layernorm, next_post_attention_layernorm):
        super().__init__(config, expertcache_, layerid)

        self.num_experts    = config.num_experts
        self.top_k          = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate               = gate
        self.shared_expert      = shared_experts
        self.shared_expert_gate = shared_expert_gate

        # Next-layer modules for prefetch prediction
        self.next_attention                = next_attention
        self.next_gate_weight              = next_gate_weight
        self.next_input_layernorm          = next_input_layernorm
        self.next_post_attention_layernorm = next_post_attention_layernorm

    # ------------------------------------------------------------------
    # AbstractMoELayer interface
    # ------------------------------------------------------------------

    def get_gate(self) -> nn.Module:
        return self.gate

    def get_num_experts(self) -> int:
        return self.num_experts

    def get_top_k(self) -> int:
        return self.top_k

    def get_norm_topk_prob(self) -> bool:
        return self.norm_topk_prob

    def compute_shared_expert(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shared_out = self.shared_expert(hidden_states)
        return F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_out

    def predict_next_layer_experts(self, residual_cur, identity, bsh,
                                   shared_expert_output) -> Optional[List[int]]:
        if self.next_gate_weight is None:
            return None
        return self._predict_next_layer_experts(
            residual_cur, identity, bsh, shared_expert_output)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, hidden_states, residual=None, attn_weights=None,
                present_key_value=None, attention_mask=None, position_ids=None,
                output_attentions=False, cache_position=None,
                position_embeddings=None):

        final_hidden_states, router_logits = self.run_with_cache(
            hidden_states,
            residual=residual,
            attn_weights=attn_weights,
            present_key_value=present_key_value,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        return final_hidden_states, router_logits

    # ------------------------------------------------------------------
    # Predict next-layer top experts (Qwen2: lightweight layernorm-only approx)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_next_layer_experts(self, residual_cur, identity, bsh,
                                    shared_expert_output) -> List[int]:
        """
        Approximate next-layer router scores using current residual stream.
        Skips attention (too expensive) — uses layernorm + gate only.
        Returns list of predicted top expert IDs for layer+1.
        """
        h = identity + shared_expert_output   # MoE output (approx)
        h = h.reshape(*bsh)
        if residual_cur is not None:
            h = residual_cur + h              # residual add
        next_residual = h
        h = self.next_input_layernorm(h)
        h = h + next_residual                 # skip attention
        h = self.next_post_attention_layernorm(h)

        logits = self.next_gate_weight(h.view(-1, bsh[2]))
        scores = F.softmax(logits, dim=1, dtype=torch.float)

        top_experts, _ = replaceset_between_tokens(
            scores.tolist(), self.replaceScoreRatio, self.top_k)
        return top_experts
