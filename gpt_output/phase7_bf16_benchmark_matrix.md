# Phase 7: BF16 scaling benchmark matrix readiness

## Purpose

Prepare one reproducible full-BF16 operator matrix for the next interval in
which GPU 0 and GPU 1 have no processes. No benchmark was executed in this
phase.

## Hard process gate

Every test, benchmark, profiler, and sweep entrypoint in `gpt_output` imports
`gpu_process_gate.require_process_free_gpus()` before importing PyTorch or SMoE
modules. The gate queries all NVIDIA compute processes and raises before the
test body if any process exists.

This covers:

- CPU affinity validation;
- routing equivalence validation;
- CPU transfer batching validation;
- Qwen BF16 fused-storage validation;
- BF16 operator benchmark;
- BF16 torch-profiler capture;
- the full BF16 scaling orchestrator.

The deployment command still requires a separate explicit `nvidia-smi` check
immediately before launch; it is not modified.

## Matrix

`run_bf16_scaling_sweep.py` launches one process per configuration so affinity,
GNU OpenMP worker count, and weight allocation are established independently.

```yaml
dtype: BF16 weights + BF16 activation + BF16 output
compute_threads: [1, 2, 4, 8, 16]
operator_paths: [separate_gate_up, fused_gate_up]
weight_sets:
  hot: 1
  rotating: 16
trials: 2
default_repeats: 80
placement: fixed physical cores from one requested NUMA node
metrics:
  - mean/median/p95 latency
  - effective weight GB/s
  - selected logical CPUs
  - resident and per-forward weight bytes
  - BF16 reference difference/equality
```

One expert's roughly 55 MB BF16 weights exceed the per-core caches and are near
the 45 MB LLC capacity of one socket; rotating 16 experts also prevents a hot
single-weight result from representing decode's changing experts.

## Invocation after the GPU gate passes

```bash
conda run -n Nmoe python gpt_output/run_bf16_scaling_sweep.py \
  --node 0 --threads 1,2,4,8,16 --expert-counts 1,16 \
  --trials 2 --warmup 16 --repeats 80 \
  --output gpt_output/bf16_scaling_sweep.json
```

The produced JSON is written under `gpt_output`. This operator matrix is
supplemental; end-to-end acceptance still uses `deploy/deployment.md` unchanged
except for `CPU_CORES=9/17`.

## Current gate state

At preparation time both GPUs were occupied by two approximately 37 GB Qwen3
evaluation processes, and GPU 1 additionally contained the Isaac Sim process.
Therefore the status remains `blocked-for-measurement`, not `pass` or `ship`.
