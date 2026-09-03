# Phase 9: formal BF16 end-to-end A/B

## Decision

Use the optimized revision with `CPU_CORES=9` on this host.  The candidate
substantially improves the nine-core deployment, but increasing the same BF16
path from 9 total cores (8 compute) to 17 total cores (16 compute) does not
provide additional speed.  In the formal run it is slower, not twice as fast.

No INT4/INT8 weights, quantized CPU operators, dequantization, approximate
arithmetic, or prefetch results are part of this conclusion.

## Frozen command and contract

All valid runs used the command from `deploy/deployment.md`.  The only changed
field was `CPU_CORES=9` or `CPU_CORES=17`:

```bash
CONDA_ENV=Nmoe MODEL_NAME=qwenmoe \
MODEL_PATH=/mnt/data/zgy/qwen2_moe \
CONFIG_PATH=configs/qwen2moe_config.json \
DATASET_PATH=wic \
INPUT_NUM=5 BATCH_SIZE=1 OUTPUT_LEN=100 \
GPU_MEM=43 CPU_CORES={9|17} LOG_LEVEL=INFO \
bash run.sh
```

- CPU expert weights, inputs, linear outputs, and expert outputs are BF16.
- `if_prefetch=false` is present in every accepted log.
- Baseline revision: `2de217b` with timing-only instrumentation in its
  isolated worktree.
- Candidate revision: `972ec84`.
- Hardware: 2-socket Intel Xeon Gold 6444Y, 16 physical cores/socket,
  AVX512-BF16 and AMX-BF16; 2x RTX A6000.
- Every accepted run has five prompts and 99 measured decode tokens/prompt.
- GPU occupancy was sampled before, during, and after each accepted run.
  GPU 0 contained only that run's SMoE process and GPU 1 contained only the
  authorized Isaac Sim monitor, PID 1393879.

Structured records are under `gpt_output/formal_runs/` and raw logs are listed
below.  The structured directory is intentionally gitignored because it
contains absolute machine paths.

## Formal results

The primary values below are means over the five prompt summaries.  CV is the
population coefficient of variation across those five summaries.

| Variant | CPU cores | CPU expert mean (ms) | CPU CV | Mean decode (s/token) | Decode CV | Mean total (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 9 | 1.3774 | 22.3% | 0.18926 | 14.3% | 23.0377 |
| candidate | 9 | 0.8416 | 2.1% | 0.13210 | 1.5% | 16.7102 |
| baseline | 17 | 1.3670 | 24.8% | 0.15320 | 9.0% | 20.4145 |
| candidate | 17 | 1.4276 | 63.0% | 0.15054 | 15.5% | 18.5060 |

Same-core candidate improvements from the perf skill's `compare_runs.py`:

| CPU cores | CPU expert mean | Prompt-median mean | Mean decode | Mean total |
| ---: | ---: | ---: | ---: | ---: |
| 9 | **38.9%** | **34.9%** | **30.2%** | **27.5%** |
| 17 | -4.4% | 1.8% | 1.7% | 9.3% |

The 17-core candidate contains one clearly exposed scheduling tail in prompt 4
(3.226 ms CPU-expert mean, while its per-token median remained 0.972 ms).
Reporting only the overall mean would overstate the regression, so the robust
across-prompt medians are also retained:

| 17-core baseline -> candidate | Baseline | Candidate | Improvement |
| --- | ---: | ---: | ---: |
| Median prompt CPU-expert mean | 1.220 ms | 0.979 ms | 19.8% |
| Median prompt CPU-expert median | 0.989 ms | 0.963 ms | 2.6% |
| Median decode | 0.1472 s/token | 0.1383 s/token | 6.0% |
| Median total | 19.7912 s | 17.7182 s | 10.5% |

Thus the optimization is a strong CPU=9 result.  At CPU=17 it remains modestly
better than the baseline under robust statistics, but it no longer produces a
large CPU-operator win.

## Core-count scaling result

The comparison below changes only `CPU_CORES` within each revision.

| Variant, 9 -> 17 | CPU expert mean | Median prompt CPU-expert mean | Mean decode | Median decode |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.8% faster | 12.1% faster | 19.1% faster | 26.3% faster |
| candidate | 69.6% slower | **17.1% slower** | 14.0% slower | **5.3% slower** |

The candidate's five per-prompt CPU-expert medians were tightly grouped at
0.956--0.972 ms (CV 0.65%), compared with 0.818--0.825 ms at CPU=9 (CV 0.35%).
The 17-core slowdown is therefore not explained by the single long-tail prompt.

The requested llama.cpp-style doubling does not transfer to this workload:

1. A Qwen2-MoE BF16 decode expert streams about 55 MB of weight for an M=1
   forward.  Unlike quantized llama.cpp paths, BF16 does not reduce those bytes.
2. `DNNL_VERBOSE=1` proves that PyTorch already selects oneDNN
   `brg_matmul:avx10_1_512_amx` BF16 AMX kernels.  The issue is not a scalar or
   FP32 fallback.
3. The pinned rotating-expert operator matrix, which is closer to real decode
   than one hot expert, improved throughput only 47.1% from 8 to 16 compute
   cores.  It missed a strict 50% target even in isolation.
4. The formal path also pays OpenMP team coordination, routing, cache activity,
   CPU/GPU transfer, page and scheduler effects.  Sixteen compute threads fill
   the physical cores of one socket, where memory bandwidth and coordination
   overhead dominate.  Cross-socket placement was measured and was slower.
5. Hardware PMU attribution cannot be collected because
   `perf_event_paranoid=4`; no system policy was changed.

## Engagement and resources

Candidate logs prove the intended path is live:

```text
[CPU BF16] fused_gate_up=1792/1792
[CPU transfer] batch_repeated_activations_and_outputs=True
compute_packages=[0]
```

The fallback switches remain available:

- `SMOE_CPU_BF16_FUSED_GATE_UP=0`
- `SMOE_CPU_BATCH_TRANSFERS=0`

Baseline and candidate both reported 42515.34 MB allocated on GPU 0.  Their
post-load process RSS values were approximately 119.8 GB, with less than 5 MB
difference.  The fused gate/up view aliases existing BF16 storage and creates
no second weight copy.

## Valid and rejected artifacts

Accepted logs:

- baseline CPU=9: `gpt_output/worktrees/baseline/logs/qwenmoe_20260903_025850.log`
- candidate CPU=9: `logs/qwenmoe_20260903_031254.log`
- baseline CPU=17: `gpt_output/worktrees/baseline/logs/qwenmoe_20260903_033128.log`
- candidate CPU=17: `logs/qwenmoe_20260903_033727.log`

Rejected runs are not used in any table:

- baseline CPU=9 `025117`: external GPU process appeared during the run.
- baseline CPU=17 `031848`: external Qwen3 PID 177120 appeared.
- baseline CPU=17 `032440`: external Qwen3 PID 184577 appeared by the end gate.
- candidate CPU=17 repeat `034210`: continuous sampling caught external Qwen3
  PID 198890 during model loading and the local SMoE was stopped.

External processes were never terminated.
