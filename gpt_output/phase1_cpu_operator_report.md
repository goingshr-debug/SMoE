# Phase 1 — CPU Expert P0

## Outcome

The production CPU expert candidate now meets the requested 8-to-16 compute-core
scaling target in CPU-only paired measurements. CPU=9 means 1 shared/load core +
8 compute cores; CPU=17 means 1 shared/load core + 16 compute cores.

| Path | 8 compute cores | 16 compute cores | 8→16 throughput gain |
|---|---:|---:|---:|
| INT4 paired trial 1 | 0.2118 ms | 0.1416 ms | +49.6% |
| INT4 paired trial 2 | 0.2134 ms | 0.1411 ms | +51.3% |
| INT4 warm-kernel trials | 0.187–0.189 ms | 0.122 ms | +53%–55% |

At a fixed core count, groupwise INT4 is 2.70–2.86x faster than the original
BF16 expert in paired tests. These are operator results, not end-to-end decode
claims. The required decode A/B could not run because the NVIDIA driver and
`/dev/nvidia0,1` are unavailable; see `gpu_gate_20260902.txt`.

## Root cause and changes

1. The old idle-core policy selected arbitrary logical CPUs. On this 2-socket,
   2-NUMA, SMT2 Xeon host it could mix sockets or select two siblings of one
   physical core. The new selector packs compute workers onto distinct physical
   cores in one package and reserves one separate loading/background core.
2. Decode-shape BF16 expert execution is a cold-weight bandwidth problem. The
   CPU path now builds group-size-128 INT4 packs while retaining the original
   pinned BF16 flat storage for GPU H2D/cache swaps. GPU expert math and storage
   layout are unchanged.
3. `SMOE_CPU_QUANT=int4` is the default. `int8` and `bf16` provide fallback and
   feature-off paths without changing the deployment command.
4. Prompt logs now report decode-only CPU expert mean/median/p95 wall time.

## Profile evidence

Separate BF16 and INT4 CPU traces are under `gpt_output/profiles/`.

- BF16: `aten::mm` accounts for 97.21% of cumulative CPU time.
- INT4: `_weight_int4pack_mm_for_cpu` accounts for 81.09%; SiLU and multiply
  together are below 1.5%, so activation fusion has a low remaining ceiling.
- The unified LLM analyzer rejected the trace because SMoE is not one of its
  supported serving frameworks. This is the correct skill gate. Kernel,
  overlap-opportunity, and fuse-pattern tables remain pending for the real GPU
  decode trace.

Profiler cumulative CPU time sums worker-thread time and is not wall latency;
the table above comes from paired `perf_counter_ns` measurements.

## Resource estimate

Isolated production-module tests measured 17–19 MB extra RSS and 0.05–0.085 s
packing time per expert. For all Qwen experts this projects to roughly 30–34 GB
extra host memory and 90–150 seconds additional model initialization. The target
host must measure actual peak RSS and load time before final acceptance.

## Status

`continue`: the CPU operator gate passes; integrated CPU forward, CPU=9/17
decode improvement, GPU memory, and P1 overlap/fusion gates remain unverified.
