#!/usr/bin/env python3
"""Run the paired full-BF16 CPU scaling matrix after enforcing the GPU gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from gpu_process_gate import require_process_free_gpus

require_process_free_gpus()


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "gpt_output" / "bench_cpu_expert.py"


def run_one(threads, mode, experts, node, warmup, repeats):
    command = [
        sys.executable,
        str(BENCHMARK),
        "--threads", str(threads),
        "--placement", "physical",
        "--node", str(node),
        "--mode", mode,
        "--experts", str(experts),
        "--pinned-weights",
        "--warmup", str(warmup),
        "--repeats", str(repeats),
    ]
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode:
        if completed.stdout:
            print(completed.stdout, file=sys.stderr, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        completed.check_returncode()
    return json.loads(completed.stdout)


def write_checkpoint(path, node, results, status):
    payload = {
        "dtype_contract": "BF16 weights, BF16 activation, BF16 output",
        "gpu_process_gate": "passed before parent and every completed child benchmark",
        "weight_memory": "CUDA pinned host memory (production offload layout)",
        "node": node,
        "status": status,
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument("--threads", default="1,2,4,8,16")
    parser.add_argument("--expert-counts", default="1,16")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "gpt_output" / "bf16_scaling_sweep.json",
    )
    args = parser.parse_args()
    threads = [int(value) for value in args.threads.split(",")]
    expert_counts = [int(value) for value in args.expert_counts.split(",")]
    results = []
    for trial in range(1, args.trials + 1):
        for experts in expert_counts:
            for thread_count in threads:
                for mode in ("reference", "fused"):
                    result = run_one(
                        thread_count, mode, experts, args.node,
                        args.warmup, args.repeats)
                    result["trial"] = trial
                    results.append(result)
                    print(json.dumps(result, sort_keys=True), flush=True)
                    write_checkpoint(args.output, args.node, results, "running")

    write_checkpoint(args.output, args.node, results, "complete")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
