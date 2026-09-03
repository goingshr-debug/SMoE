# Phase 5: BF16-only CPU operator reset and fusion candidate

## Constraint correction

The CPU expert acceptance path is now strictly BF16. Previous INT4/INT8
experiments and their scaling numbers are withdrawn from P0 evidence. Quantized
CPU operators, packed quantized weights, quantization environment switches, and
quantization preparation hooks have been removed from production code.

## BF16 scaling hypothesis

The target Qwen expert has three BF16 matrices:

```text
gate: 2560 x 3584 x 2 bytes
up:   2560 x 3584 x 2 bytes
down: 3584 x 2560 x 2 bytes
total weight traffic lower bound: about 55 MB per cold expert forward
```

At decode `M=1`, this behaves primarily like three memory-intensive GEMVs. More
cores can reduce latency until socket memory bandwidth, NUMA placement, or the
BF16 kernel's parallel granularity saturates. Unlike common llama.cpp quantized
paths, BF16 does not reduce bytes read per weight, so linear scaling is possible
but not guaranteed.

The host is a dual-socket Intel Xeon Gold 6444Y with 16 physical cores per
socket and AVX512-BF16/AMX-BF16 support. CPU=9 and CPU=17 correspond to 8 and 16
compute cores and remain within one socket under the topology-aware placement.
This is the correct range for measuring the requested scaling before crossing
the NUMA boundary.

## Candidate

Qwen gate and up projections consume the same activation. The CPU backing-store
layout is changed from `gate/down/up` to `gate/up/down`, for both CPU experts and
their matching GPU cache slots. A zero-copy BF16 view spanning the adjacent
gate/up rows is attached to CPU experts:

```text
[tokens, 3584] BF16
  -> one F.linear with [5120, 3584] BF16
  -> split gate/up at 2560
  -> SiLU(gate) * up
  -> BF16 down projection
```

This remains a BF16 weight/BF16 operator path. It adds no quantized copy and no
second BF16 weight allocation. The original two-projection BF16 path is retained
with `SMOE_CPU_BF16_FUSED_GATE_UP=0` for paired reference measurements.

Raw expert storage is copied as one unit during cache swaps. CPU and GPU wrapper
layouts are changed together, so gate/up/down parameter views on a loaded GPU
slot continue to address the corresponding raw weight ranges.

## Required validation

No test or benchmark was run in this phase because GPU 1 was not process-free,
and the project rule requires both GPU 0 and GPU 1 to have zero processes before
all tests.

After the gate clears:

1. run `test_qwen_cpu_bf16_fusion.py` for storage aliasing, token shapes
   1/2/17, BF16 reference equality, and feature-off fallback;
2. profile separate and fused BF16 paths;
3. sweep 1/2/4/8/16 compute cores with cold and warm weight conditions;
4. report median/p95 latency, throughput scaling, effective weight bandwidth,
   CPU frequency, NUMA locality, and dominant operators;
5. run exact `deployment.md` commands at CPU=9 and CPU=17 for baseline and
   candidate, with prefetch still disabled;
6. retain fusion only if CPU expert and end-to-end decode results exceed noise.

## Decision

`continue`: implementation is prepared, but correctness and performance remain
unverified until both GPUs are process-free. If BF16 8-to-16-core scaling remains
below 50%, the final report will attribute the observed ceiling using measured
bandwidth, NUMA, frequency, and profiler evidence rather than substituting
quantized results.
