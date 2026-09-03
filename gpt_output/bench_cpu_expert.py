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
        choices=("reference", "fused"),
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

    def execute():
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
