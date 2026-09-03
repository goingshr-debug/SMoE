#!/usr/bin/env python3
"""Capture separate and fused BF16 CPU expert torch-profiler traces."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from MoEModule.qwen_moe import Qwen2MoeMLP
from utils.cpu_affinity import select_cpu_placement


def capture(expert, x, fused_weight, mode: str, output_dir: Path):
    expert._cpu_gate_up_weight = fused_weight if mode == "fused" else None

    for _ in range(10):
        expert(x)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=5, repeat=1),
        record_shapes=True,
        with_stack=True,
    ) as profile:
        for _ in range(8):
            expert(x)
            profile.step()

    trace_path = output_dir / f"cpu_expert_bf16_{mode}.trace.json"
    profile.export_chrome_trace(str(trace_path))
    print(f"trace={trace_path}")
    print(profile.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=15
    ))


def main():
    placement = select_cpu_placement(9, 0.05)
    os.sched_setaffinity(0, placement.compute_cores)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260902)

    config = SimpleNamespace(
        hidden_size=3584,
        moe_intermediate_size=2560,
        hidden_act="silu",
        device="cpu",
    )
    expert = Qwen2MoeMLP(config).eval()
    gate_up = torch.randn(
        (2 * config.moe_intermediate_size, config.hidden_size),
        dtype=torch.bfloat16,
    )
    expert.gate_proj.weight.data = gate_up[:config.moe_intermediate_size]
    expert.up_proj.weight.data = gate_up[config.moe_intermediate_size:]
    x = torch.randn((1, config.hidden_size), dtype=torch.bfloat16)
    output_dir = ROOT / "gpt_output" / "profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    capture(expert, x, gate_up, "reference", output_dir)
    capture(expert, x, gate_up, "fused", output_dir)


if __name__ == "__main__":
    main()
