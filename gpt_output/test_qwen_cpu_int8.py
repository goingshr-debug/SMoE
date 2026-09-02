#!/usr/bin/env python3
"""Exercise the production Qwen quantized CPU expert path at decode shape."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from MoEModule.qwen_moe import Qwen2MoeMLP
from utils.cpu_affinity import select_cpu_placement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--mode", choices=("int4", "int8"), default="int4")
    args = parser.parse_args()
    os.environ["SMOE_CPU_QUANT"] = args.mode

    placement = select_cpu_placement(args.threads + 1, 0.05)
    os.sched_setaffinity(0, placement.compute_cores)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260902)

    config = SimpleNamespace(
        hidden_size=3584,
        moe_intermediate_size=2560,
        hidden_act="silu",
        device="cpu",
    )
    expert = Qwen2MoeMLP(config).eval()
    x = torch.randn((1, config.hidden_size), dtype=torch.bfloat16)
    reference = expert(x)

    rss_before_pack = psutil.Process().memory_info().rss
    pack_start = time.perf_counter()
    assert expert.prepare_cpu_acceleration()
    pack_seconds = time.perf_counter() - pack_start
    rss_after_pack = psutil.Process().memory_info().rss
    candidate = expert(x)
    assert candidate.shape == reference.shape
    assert candidate.dtype == torch.bfloat16
    assert torch.isfinite(candidate).all()
    multi_token_shapes = []
    for tokens in (2, 17):
        multi_x = torch.randn((tokens, config.hidden_size), dtype=torch.bfloat16)
        multi_output = expert(multi_x)
        assert multi_output.shape == (tokens, config.hidden_size)
        assert multi_output.dtype == torch.bfloat16
        assert torch.isfinite(multi_output).all()
        multi_token_shapes.append(list(multi_output.shape))

    accel_packs = expert._cpu_accel_packs
    accel_samples_ms = []
    bf16_samples_ms = []
    for _ in range(10):
        expert._cpu_accel_packs = None
        expert(x)
        expert._cpu_accel_packs = accel_packs
        expert(x)
    for _ in range(args.repeats):
        expert._cpu_accel_packs = None
        start = time.perf_counter_ns()
        expert(x)
        bf16_samples_ms.append((time.perf_counter_ns() - start) / 1_000_000)
        expert._cpu_accel_packs = accel_packs
        start = time.perf_counter_ns()
        expert(x)
        accel_samples_ms.append((time.perf_counter_ns() - start) / 1_000_000)

    expert.clear_cpu_acceleration()
    fallback = expert(x)
    assert torch.equal(fallback, reference)

    print(json.dumps({
        "threads": args.threads,
        "mode": args.mode,
        "multi_token_shapes": multi_token_shapes,
        "cores": list(placement.compute_cores),
        "pack_seconds": pack_seconds,
        "packed_rss_delta_bytes": rss_after_pack - rss_before_pack,
        "accel_median_ms": statistics.median(accel_samples_ms),
        "accel_mean_ms": statistics.fmean(accel_samples_ms),
        "accel_min_ms": min(accel_samples_ms),
        "accel_max_ms": max(accel_samples_ms),
        "bf16_median_ms": statistics.median(bf16_samples_ms),
        "bf16_mean_ms": statistics.fmean(bf16_samples_ms),
        "bf16_min_ms": min(bf16_samples_ms),
        "bf16_max_ms": max(bf16_samples_ms),
        "accel_speedup_vs_bf16": (
            statistics.median(bf16_samples_ms) / statistics.median(accel_samples_ms)
        ),
        "max_abs_diff": (candidate.float() - reference.float()).abs().max().item(),
        "mean_abs_diff": (candidate.float() - reference.float()).abs().mean().item(),
        "reference_mean_abs": reference.float().abs().mean().item(),
        "fallback_exact": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
