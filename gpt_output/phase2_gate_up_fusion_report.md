# Phase 2 — Gate/Up INT4 Fusion Experiment

## Hypothesis

Qwen gate and up projections consume the same hidden state. Packing both row
ranges into one INT4 matrix and splitting its output removes one operator
dispatch and one activation read. The hypothesis predicted lower CPU expert
wall time, especially at CPU=17.

## Contract

After making `scales_and_zeros` contiguous, separate and fused INT4 paths return
exactly equal BF16 output for the production decode shape. The fused path changes
only packing/dispatch; group size, quantization formula, activation, and down
projection remain identical.

## Evidence

Warm, paired operator A/B:

| Compute cores | Separate | Fused | Result |
|---:|---:|---:|---:|
| 8, trial 1 | 0.1863 ms | 0.1891 ms | -1.5% |
| 8, trial 2 | 0.1857 ms | 0.1886 ms | -1.5% |
| 16, trial 1 | 0.1198 ms | 0.1111 ms | +7.9% |
| 16, trial 2 | 0.1208 ms | 0.1116 ms | +8.3% |

The temporary production auto-fusion path was then tested while alternating
BF16 and INT4 calls to evict packed weights from cache. At 16 compute cores its
median was 0.1408 ms and 0.1451 ms, versus 0.1412 ms and 0.1405 ms for the
separate production path. The benefit disappeared and one trial regressed.

Raw evidence:

- `cpu_int4_fusion_paired_{8,16}_t{1,2}.json`
- `qwen_cpu_int4_autofuse_{8,16}_t{1,2}.json`

## Decision

`stop`. Warm-cache kernel benefit does not improve the representative cold
expert path and CPU=9 consistently regresses. Production keeps three separate
INT4 packed matmuls. The benchmark is retained so this rejected optimization is
not reintroduced without stronger end-to-end evidence.
