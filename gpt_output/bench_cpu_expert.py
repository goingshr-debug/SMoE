#!/usr/bin/env python3
"""CPU-only Qwen2-MoE expert microbenchmark.

This is a diagnostic benchmark, not the end-to-end acceptance benchmark.  It
uses the production decode shape/dtype and compares the current three-linear
reference with a gate+up projection fusion candidate.  Run one configuration
per process so the OpenMP pool is created after affinity is fixed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cpu_affinity import select_cpu_placement


def _read_int(path: Path, default: int) -> int:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return default


def physical_cores_for_node(node: int) -> list[int]:
    allowed = os.sched_getaffinity(0)
    unique: dict[tuple[int, int], int] = {}
    for cpu in sorted(allowed):
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        package = _read_int(root / "physical_package_id", 0)
        core = _read_int(root / "core_id", cpu)
        node_dirs = list(Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node[0-9]*"))
        cpu_node = int(node_dirs[0].name[4:]) if node_dirs else package
        if cpu_node == node:
            unique.setdefault((package, core), cpu)
    return list(unique.values())


def select_cores(placement: str, threads: int, node: int) -> list[int]:
    if placement == "topology":
        return list(
            select_cpu_placement(threads + 1, sample_interval=0.2).compute_cores
        )
    if placement == "physical":
        cores = physical_cores_for_node(node)
        if len(cores) < threads:
            raise RuntimeError(
                f"NUMA node {node} has only {len(cores)} physical cores; requested {threads}"
            )
        return cores[:threads]

    usage = psutil.cpu_percent(percpu=True, interval=0.2)
    allowed = os.sched_getaffinity(0)
    ranked = sorted(allowed, key=lambda cpu: (usage[cpu], cpu))
    return ranked[:threads]


def run_forward(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, fused: bool):
    intermediate = gate_up.shape[0] // 2
    if fused:
        gate_up_out = F.linear(x, gate_up)
        gate, up = gate_up_out.split(intermediate, dim=-1)
    else:
        gate = F.linear(x, gate_up[:intermediate])
        up = F.linear(x, gate_up[intermediate:])
    return F.linear(F.silu(gate) * up, down)


def pack_int8(weight: torch.Tensor):
    scale = max(weight.float().abs().max().item() / 127.0, 1.0e-8)
    quantized = torch.quantize_per_tensor(weight.float(), scale, 0, torch.qint8)
    return torch.ops.quantized.linear_prepack(quantized, None)


def run_forward_int8(x: torch.Tensor, gate_pack, up_pack, down_pack):
    x_float = x.float()
    gate = torch.ops.quantized.linear_dynamic(x_float, gate_pack, True)
    up = torch.ops.quantized.linear_dynamic(x_float, up_pack, True)
    return torch.ops.quantized.linear_dynamic(F.silu(gate) * up, down_pack, True).to(x.dtype)


def run_forward_int8_fused(x: torch.Tensor, gate_up_pack, down_pack, intermediate: int):
    gate_up = torch.ops.quantized.linear_dynamic(x.float(), gate_up_pack, True)
    gate, up = gate_up.split(intermediate, dim=-1)
    return torch.ops.quantized.linear_dynamic(
        F.silu(gate) * up, down_pack, True
    ).to(x.dtype)


def pack_int4(weight: torch.Tensor, group_size: int = 128):
    rows, cols = weight.shape
    grouped = weight.float().reshape(rows, cols // group_size, group_size)
    group_min = grouped.amin(dim=-1)
    group_max = grouped.amax(dim=-1)
    scales = ((group_max - group_min) / 15.0).clamp_min(1.0e-8)
    zeros = group_min + 8.0 * scales
    codes = torch.round((grouped - zeros.unsqueeze(-1)) / scales.unsqueeze(-1) + 8.0)
    codes = codes.clamp_(0, 15).to(torch.int32).reshape(rows, cols)
    packed = torch.ops.aten._convert_weight_to_int4pack_for_cpu(codes, 8)
    scales_and_zeros = torch.stack((scales, zeros), dim=-1).transpose(0, 1)
    return packed, scales_and_zeros.to(torch.bfloat16)


def linear_int4(x: torch.Tensor, pack, group_size: int = 128):
    weight, scales_and_zeros = pack
    return torch.ops.aten._weight_int4pack_mm_for_cpu(
        x, weight, group_size, scales_and_zeros
    )


def run_forward_int4(x: torch.Tensor, gate_pack, up_pack, down_pack):
    gate = linear_int4(x, gate_pack)
    up = linear_int4(x, up_pack)
    return linear_int4(F.silu(gate) * up, down_pack)


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument(
        "--placement", choices=("legacy", "physical", "topology"), required=True
    )
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("reference", "fused", "int8", "int8_fused", "int4"),
        required=True,
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()

    cores = select_cores(args.placement, args.threads, args.node)
    os.sched_setaffinity(0, cores)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)

    hidden = 3584
    intermediate = 2560
    dtype = torch.bfloat16
    x = torch.randn((1, hidden), dtype=dtype)
    gate_up = torch.randn((2 * intermediate, hidden), dtype=dtype)
    down = torch.randn((hidden, intermediate), dtype=dtype)

    fused = args.mode == "fused"
    int8_packs = None
    if args.mode == "int8":
        int8_packs = (
            pack_int8(gate_up[:intermediate]),
            pack_int8(gate_up[intermediate:]),
            pack_int8(down),
        )
    elif args.mode == "int8_fused":
        int8_packs = (pack_int8(gate_up), pack_int8(down))
    elif args.mode == "int4":
        int8_packs = (
            pack_int4(gate_up[:intermediate]),
            pack_int4(gate_up[intermediate:]),
            pack_int4(down),
        )

    def execute():
        if int8_packs is not None:
            if args.mode == "int8_fused":
                return run_forward_int8_fused(x, *int8_packs, intermediate)
            if args.mode == "int4":
                return run_forward_int4(x, *int8_packs)
            return run_forward_int8(x, *int8_packs)
        return run_forward(x, gate_up, down, fused)

    for _ in range(args.warmup):
        execute()

    samples_ms = []
    for _ in range(args.repeats):
        start = time.perf_counter_ns()
        execute()
        samples_ms.append((time.perf_counter_ns() - start) / 1_000_000)

    reference = run_forward(x, gate_up, down, False)
    candidate = execute()
    result = {
        "threads": args.threads,
        "placement": args.placement,
        "node": args.node,
        "cores": cores,
        "mode": args.mode,
        "shape": {"tokens": 1, "hidden": hidden, "intermediate": intermediate},
        "dtype": str(dtype),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": min(samples_ms),
        "p95_ms": percentile(samples_ms, 0.95),
        "max_abs_diff_vs_reference": (candidate.float() - reference.float()).abs().max().item(),
        "bitwise_equal_vs_reference": torch.equal(candidate, reference),
        "torch_version": torch.__version__,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
