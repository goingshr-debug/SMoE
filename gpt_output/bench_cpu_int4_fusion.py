#!/usr/bin/env python3
"""Paired gate/up INT4 fusion A/B at the production decode shape."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpt_output.bench_cpu_expert import (
    pack_int4,
    run_forward_int4,
    run_forward_int4_fused,
)
from utils.cpu_affinity import select_cpu_placement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()

    placement = select_cpu_placement(args.threads + 1, 0.05)
    os.sched_setaffinity(0, placement.compute_cores)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260902)

    hidden = 3584
    intermediate = 2560
    x = torch.randn((1, hidden), dtype=torch.bfloat16)
    gate_up = torch.randn((2 * intermediate, hidden), dtype=torch.bfloat16)
    down = torch.randn((hidden, intermediate), dtype=torch.bfloat16)
    down_pack = pack_int4(down)
    separate_packs = (
        pack_int4(gate_up[:intermediate]),
        pack_int4(gate_up[intermediate:]),
        down_pack,
    )
    fused_packs = (pack_int4(gate_up), down_pack)

    separate_output = run_forward_int4(x, *separate_packs)
    fused_output = run_forward_int4_fused(x, *fused_packs, intermediate)
    assert torch.equal(separate_output, fused_output)

    for _ in range(10):
        run_forward_int4(x, *separate_packs)
        run_forward_int4_fused(x, *fused_packs, intermediate)

    separate_ms = []
    fused_ms = []
    for _ in range(args.repeats):
        start = time.perf_counter_ns()
        run_forward_int4(x, *separate_packs)
        separate_ms.append((time.perf_counter_ns() - start) / 1_000_000)
        start = time.perf_counter_ns()
        run_forward_int4_fused(x, *fused_packs, intermediate)
        fused_ms.append((time.perf_counter_ns() - start) / 1_000_000)

    separate_median = statistics.median(separate_ms)
    fused_median = statistics.median(fused_ms)
    print(json.dumps({
        "threads": args.threads,
        "cores": list(placement.compute_cores),
        "repeats": args.repeats,
        "separate_median_ms": separate_median,
        "fused_median_ms": fused_median,
        "fused_speedup": separate_median / fused_median,
        "outputs_equal": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
