#!/usr/bin/env python3
"""Convert one deployment.md log into a comparable structured run record."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


CPU_RE = re.compile(
    r"\[CPU expert\] prompt=(\d+) decode_forward_mean=([0-9.]+) ms "
    r"median=([0-9.]+) ms p95=([0-9.]+) ms sampled_tokens=(\d+)"
)
E2E_RE = re.compile(
    r"\[SMoE\] prompt=(\d+)\s+prefill=([0-9.]+) s\s+"
    r"avg_decode=([0-9.]+) s\s+total=([0-9.]+) s\s+decode_tokens=(\d+)"
)
AFFINITY_RE = re.compile(r"\[AFFINITY\] n=(\d+)\s+compute=(\[[^]]*\]).*shared=(\d+)")


def mean_and_cv(values):
    mean = statistics.fmean(values)
    return {
        "mean": mean,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "cv": statistics.pstdev(values) / mean if mean else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cpu-cores", type=int, required=True)
    parser.add_argument("--validity", choices=("valid", "invalid"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    text = args.log.read_text(errors="replace")
    cpu = [
        {
            "prompt": int(match[0]),
            "mean_ms": float(match[1]),
            "median_ms": float(match[2]),
            "p95_ms": float(match[3]),
            "sampled_tokens": int(match[4]),
        }
        for match in CPU_RE.findall(text)
    ]
    e2e = [
        {
            "prompt": int(match[0]),
            "prefill_s": float(match[1]),
            "avg_decode_s": float(match[2]),
            "total_s": float(match[3]),
            "decode_tokens": int(match[4]),
        }
        for match in E2E_RE.findall(text)
    ]
    if len(cpu) != 5 or len(e2e) != 5:
        raise ValueError(
            f"expected five completed prompts, found CPU={len(cpu)} E2E={len(e2e)}"
        )
    affinity = AFFINITY_RE.search(text)
    if not affinity or int(affinity.group(1)) != args.cpu_cores:
        raise ValueError("missing or mismatched affinity record")
    if "'if_prefetch': False" not in text:
        raise ValueError("formal log does not prove prefetch is disabled")

    record = {
        "schema": "smoe-formal-run-v1",
        "variant": args.variant,
        "revision": args.revision,
        "validity": args.validity,
        "log": str(args.log.resolve()),
        "workload": {
            "model": "qwenmoe",
            "model_path": "/mnt/data/zgy/qwen2_moe",
            "dataset": "wic",
            "input_num": 5,
            "batch_size": 1,
            "output_len": 100,
            "gpu_mem_gb": 43,
            "prefetch": False,
            "dtype": "BF16 CPU expert",
        },
        "cpu": {
            "total_cores": args.cpu_cores,
            "compute_cores": json.loads(affinity.group(2)),
            "shared_core": int(affinity.group(3)),
        },
        "prompts": {"cpu_expert": cpu, "end_to_end": e2e},
        "metrics": {
            "cpu_expert_mean_ms": mean_and_cv([row["mean_ms"] for row in cpu]),
            "cpu_expert_prompt_median_ms": mean_and_cv(
                [row["median_ms"] for row in cpu]
            ),
            "decode_s": mean_and_cv([row["avg_decode_s"] for row in e2e]),
            "prefill_s": mean_and_cv([row["prefill_s"] for row in e2e]),
            "total_s": mean_and_cv([row["total_s"] for row in e2e]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record["metrics"], indent=2, sort_keys=True))
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
