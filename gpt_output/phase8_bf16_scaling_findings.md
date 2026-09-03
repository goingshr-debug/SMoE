# Phase 8: BF16 CPU scaling findings

## Contract

- CPU expert weights, activations, linear outputs, and final output are BF16.
- Accumulation is the native PyTorch/oneDNN BF16 linear behavior; no CPU
  quantization, dequantization, or approximate arithmetic is present.
- Decode operator shape is one token, hidden size 3584, and expert intermediate
  size 2560.
- Each timed process is fixed to physical cores on NUMA node 0 and configures
  PyTorch intra-op threads before the first timed forward.
- GPU 0/1 gate passed with only the explicitly authorized Isaac Sim monitoring
  process present on GPU 1.

## Completed pageable-memory diagnostic matrix

Artifact: `gpt_output/bf16_scaling_sweep.json`

SHA256: `ce742bacd397f39127d8d47699ba7e0dd28b9bc599f6f69d52dcd998664c9ee2`

The table reports the mean of two trial medians. Throughput gain is
`latency_8 / latency_16 - 1`.

| Weight working set | BF16 path | 8-core median (ms) | 16-core median (ms) | 8→16 throughput gain | Latency reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 hot expert | three linear reference | 0.5254 | 0.3604 | 45.8% | 31.4% |
| 1 hot expert | fused gate/up | 0.4978 | 0.4150 | 19.9% | 16.6% |
| 16 rotating experts | three linear reference | 0.6586 | 0.4835 | 36.2% | 26.6% |
| 16 rotating experts | fused gate/up | 0.6371 | 0.4803 | 32.6% | 24.6% |

All 40 results reported `torch.bfloat16`, zero maximum difference from their
BF16 reference, and bitwise equality. This matrix is diagnostic rather than
the final production-memory result because the benchmark initially used
ordinary pageable tensors, while SMoE offloaded expert storage is CUDA-pinned.
The sweep now explicitly requests pinned weights; the replacement formal matrix
will use a separate artifact and will not overwrite this evidence.

## Native kernel engagement

With `DNNL_VERBOSE=1`, PyTorch reported:

```text
cpu,runtime:OpenMP,nthr:8
cpu,isa:Intel AVX-512 ... Intel AMX with bfloat16
matmul,brg_matmul:avx10_1_512_amx,...,src:bf16,wei:bf16,dst:bf16
```

Therefore the existing path already hits oneDNN's AMX BF16 implementation; the
scaling limit is not caused by silently falling back to scalar FP32 or a CPU
quantized kernel. Torch-profiler CPU traces are under `gpt_output/profiles/`.
They attribute 55%–76% of self CPU time to `aten::mm`; SiLU, multiply, split,
and dispatcher overhead are secondary.

Hardware-counter profiling is unavailable because the host sets
`perf_event_paranoid=4`. The experiment did not change that system policy.

## Rejected mechanisms

### Cross-socket split placement

The diagnostic split eight compute threads per socket. With 16 rotating
experts, a 16-thread reference median of 0.6132 ms was slower than the
single-socket matrix result of 0.4835 ms and exhibited more variance. Remote
NUMA traffic outweighed access to the second socket's memory controllers.
Production placement remains on one socket for CPU=9 and CPU=17.

### Decode-only `torch.mv`

Using BF16 `torch.mv` for the M=1 projections improved the hot-weight scaling,
but it did not improve the production-like rotating case. One completed
rotating run measured fused MV at 0.6961 ms (8 cores) and 0.4894 ms (16 cores),
versus the matrix fused linear values of 0.6371 ms and 0.4803 ms. It is not a
production candidate.

## Completed pinned-memory matrix

Artifact: `gpt_output/bf16_scaling_sweep_pinned.json`

SHA256: `1914fc5f034eb4a76144e1c620160be7f12eda67673ec9feff137840cf8723ea`

All 40 configurations completed. The table again reports the mean of two trial
medians.

| Weight working set | BF16 path | 8 cores (ms) | 16 cores (ms) | 8→16 throughput gain | Latency reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 hot expert | three linear reference | 0.5192 | 0.4260 | 21.9% | 17.9% |
| 1 hot expert | fused gate/up | 0.5027 | 0.3253 | 54.5% | 35.3% |
| 16 rotating experts | three linear reference | 0.6762 | 0.4918 | 37.5% | 27.3% |
| 16 rotating experts | fused gate/up | 0.6595 | 0.4485 | 47.1% | 32.0% |

The fused operator crosses the 50% throughput target for a hot expert. The
rotating working set, which better models decode across layers and expert IDs,
is reproducibly just below it at 47.1%; it is therefore reported as close but
not accepted against a strict 50% operator threshold.

Three longer rotating trials on NUMA node 1 produced mean trial medians of
0.8182 ms (8 cores) and 0.4904 ms (16 cores), or 66.8% scaling. That is not a
win: both absolute latencies are worse than node 0, and one 16-core trial had a
14.1 ms p95 interruption. The result reinforces selecting the faster/less-busy
single socket and not optimizing the ratio by slowing CPU=9.

## Current decision

Status: `continue`.

The evidence supports topology-aware physical-core placement and rejects
cross-socket placement and `torch.mv`. The pinned rotating result is close
enough to proceed to formal end-to-end CPU=9/17 A/B. The final decision must
use the end-to-end and rotating results, not the favorable hot-cache row or the
slower node-1 ratio.
