#!/usr/bin/env python3
"""Validate the zero-copy, full-precision BF16 Qwen gate/up projection."""

import os
from types import SimpleNamespace

import torch

from MoEModule.qwen_moe import Qwen2MoeMLP
from utils.model_loader import ExpertWrapper


def main():
    os.environ["SMOE_CPU_BF16_FUSED_GATE_UP"] = "1"
    torch.manual_seed(20260903)
    config = SimpleNamespace(
        hidden_size=128,
        moe_intermediate_size=64,
        hidden_act="silu",
        device="cpu",
    )
    expert = Qwen2MoeMLP(config).eval()
    wrapper = ExpertWrapper(
        expert, "qwenmoe", device=torch.device("cpu"), tocpu=True)

    gate = expert.gate_proj.weight
    up = expert.up_proj.weight
    fused = expert._cpu_gate_up_weight
    assert wrapper.cpu_bf16_gate_up_fused
    assert gate.dtype == up.dtype == fused.dtype == torch.bfloat16
    assert gate.data_ptr() + gate.nbytes == up.data_ptr()
    assert fused.data_ptr() == gate.data_ptr()
    assert fused.nbytes == gate.nbytes + up.nbytes
    assert tuple(fused.shape) == (128, 128)

    for tokens in (1, 2, 17):
        x = torch.randn((tokens, config.hidden_size), dtype=torch.bfloat16)
        expert._cpu_gate_up_weight = None
        reference = expert(x)
        expert._cpu_gate_up_weight = fused
        candidate = expert(x)
        torch.testing.assert_close(candidate, reference, rtol=0, atol=0)

    os.environ["SMOE_CPU_BF16_FUSED_GATE_UP"] = "0"
    fallback = Qwen2MoeMLP(config)
    assert fallback._cpu_gate_up_weight is None
    assert not fallback.configure_cpu_bf16_gate_up(fused)
    print("PASS: zero-copy BF16 gate/up fusion matches the BF16 reference")


if __name__ == "__main__":
    main()
