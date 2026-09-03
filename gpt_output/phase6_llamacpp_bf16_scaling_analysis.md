# Phase 6: llama.cpp versus SMoE BF16 CPU scaling

## Question

Can doubling SMoE compute cores from 8 to 16 approximately double a full-BF16
Qwen expert forward, as observed in some llama.cpp workloads?

Answer before measurement: it is possible on this host, but llama.cpp does not
establish it as an unconditional BF16 result. Scaling depends on weight bytes,
kernel partitioning, persistent-worker overhead, affinity, NUMA placement, and
the socket bandwidth ceiling.

## Source-backed llama.cpp findings

The referenced `ggml-backend.cpp` schedules graphs/backends and copies. The CPU
matrix kernel and thread scheduling are implemented under `ggml/src/ggml-cpu/`.

1. `ggml-cpu.c` defines a persistent thread pool with per-worker state,
   configurable active-thread count, hybrid polling/sleeping, and reusable
   barriers (`struct ggml_threadpool`, around lines 441-570 and 2938-3170).
   Workers are not created for every matrix operation.
2. With strict CPU placement, each worker receives its own CPU mask and applies
   it using `pthread_setaffinity_np` (around lines 2495-2519 and 3076-3138).
3. Decode-shaped matrix multiplication is split across output rows. The kernel
   creates row chunks, assigns one initial chunk per worker, and then uses an
   atomic work queue for remaining chunks (`ggml_compute_forward_mul_mat`,
   around lines 1301-1393). When chunk count is small or NUMA is enabled, it
   rechunks directly by thread.
4. The BF16 dot kernel uses `_mm512_dpbf16_ps` when AVX512-BF16 is available
   (`ggml-cpu/vec.cpp`, around lines 139-160), accumulating into FP32 vectors.
5. NUMA modes can distribute workers, isolate them to the current node, or
   honor a `numactl` CPU map (`ggml-cpu.c`, around lines 590-680 and 2088-2125).
   llama.cpp also warns when automatic NUMA balancing is enabled.

These mechanisms explain why a sufficiently large row-parallel GEMV can scale
well. They do not imply that every reported llama.cpp result used BF16 weights;
common quantized formats read substantially fewer weight bytes and therefore
reach a different bandwidth roof.

## SMoE comparison

| Mechanism | llama.cpp | Current SMoE candidate |
|---|---|---|
| Weight/operator dtype | workload-dependent, commonly quantized | BF16 only |
| Worker lifetime | explicit persistent pool | PyTorch/libgomp intra-op pool |
| Worker affinity | optional strict per-worker mask | process mask restricted to distinct physical cores in one socket |
| GEMV work split | explicit output-row chunks plus dynamic stealing | selected internally by PyTorch `aten::mm`/backend |
| BF16 ISA | explicit AVX512-BF16 dot path | backend-dependent; host supports AVX512-BF16 and AMX-BF16 |
| NUMA | explicit modes and warning | same-socket core selection and first-touch intent; no memory-policy API |
| Gate/up dispatch | separate graph ops unless model/kernel fuses | one zero-copy combined BF16 `F.linear` candidate |

The Nmoe PyTorch binary links GNU OpenMP (`libgomp`). Main-process affinity is
set before the model and first expert operation, so workers inherit a mask that
contains only selected physical cores. Unlike strict llama.cpp placement, a
worker may still migrate among those cores.

## Hardware facts relevant to the ceiling

- CPU: 2-socket Intel Xeon Gold 6444Y, 16 physical cores/socket, SMT2.
- ISA: AVX512-BF16 and AMX-BF16 present.
- NUMA distance: local 10, remote 21.
- Kernel automatic NUMA balancing: enabled (`1`).
- Reported scaling governor: `powersave` on both sockets. With Intel P-state this
  does not alone prove low frequency; per-core frequency must be measured.
- CPU=9 and CPU=17 use 8 and 16 compute cores plus one shared load/background
  core. Both compute sets fit within one socket.

## BF16 traffic and scaling bounds

For one Qwen expert at token count 1, the three BF16 weights total roughly
55 MB. Activation and output traffic is small relative to weights. A first-order
effective bandwidth estimate is:

```text
effective_GBps = 55 MB / expert_forward_seconds
```

If 8 cores already consume most sustainable local-socket bandwidth, 16 cores
cannot deliver 2x. If 8 cores are limited by insufficient row parallelism,
worker wake-up/barriers, frequency, or poor placement, the fused 5120-row BF16
projection and topology changes can still produce the requested 1.5-2.0x
throughput gain.

The acceptance threshold is:

```text
throughput_gain_8_to_16 = latency_8 / latency_16 - 1
target: 50% to 100%
equivalently: latency_16 <= latency_8 / 1.5
```

## Measurement and attribution plan

After both GPUs have zero processes:

1. Run separate and fused BF16 operator paths at 1/2/4/8/16 compute cores.
2. Measure warm repeated weights and a rotating/cold expert set; decode uses
   many experts, so a single hot matrix is insufficient evidence.
3. Record mean/median/p95, effective BF16 weight bandwidth, selected CPUs,
   socket/NUMA placement, CPU frequency, migrations, cache misses, and memory
   bandwidth where counters are available.
4. Profile operator count and cumulative CPU time for `aten::mm`, SiLU, split,
   and multiply.
5. Run baseline/candidate CPU=9/17 with the exact deployment command, changing
   only `CPU_CORES`, and report CPU expert plus end-to-end decode separately.

## Failure attribution if 8-to-16 misses 50%

- Bandwidth saturation: effective bandwidth plateaus while cores increase.
- Remote/pinned-memory placement: remote accesses or NUMA migrations rise.
- Parallel granularity: CPU utilization or active workers remain below 16 for
  the M=1 BF16 matrix.
- Dispatch/barrier limit: three small projections spend a material fraction in
  worker wake-up/barriers; compare fused gate/up.
- Frequency collapse/contention: all-core frequency falls or external CPU load
  overlaps selected cores.
- Backend limitation: PyTorch's selected BF16 M=1 kernel does not row-partition
  efficiently; compare `linear`, `mv`, and a source-backed row-parallel kernel
  before proposing a native extension.

## Status

`continue`. This is a source and hardware analysis, not performance evidence.
No test or benchmark was run while GPU 1 remained occupied.
