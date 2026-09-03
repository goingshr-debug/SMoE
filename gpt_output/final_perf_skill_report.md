# Final perf-skill acceptance report

```yaml
scope:
  target: Qwen2-MoE CPU expert and exposed decode latency
  dtype: BF16 weights, BF16 inputs, BF16 outputs
  excluded: [INT4, INT8, quantized operators, prefetch]
  baseline_revision: 2de217b
  candidate_revision: 972ec84
primary_route:
  orchestrator: gpu-operator-optimization-engineer
  mechanism: BF16 operator fusion plus CPU topology and transfer reductions
  specialist: none; the supplied specialist routing table has no CPU BF16 route
secondary_routes:
  - llm-torch-profiler-analysis (mandatory profiler route)
acceptance:
  result: continue
  deployment_recommendation: candidate revision with CPU_CORES=9
  reason: CPU=9 target-level gains pass, but CPU=17 scaling is refuted and the
    unified profiler hard gate is unsupported for the custom SMoE trace
```

## Seven ledgers

### 1. reproduction

`REP-001` — **passed**

- Formal command and the only allowed `CPU_CORES` variation are recorded in
  `gpt_output/phase9_formal_ab.md`.
- Workload: Qwen2-MoE, WiC, 5 prompts, batch 1, 100 output tokens, 43 GB GPU
  budget, prefetch disabled.
- Hardware: dual-socket Xeon Gold 6444Y, 16 physical cores/socket, BF16 AMX;
  2x RTX A6000, driver 570.211.01.
- Four accepted structured records are in `gpt_output/formal_runs/`; every one
  contains five completed prompt records, the exact affinity, revision, dtype,
  command fields, statistics, and `validity=valid`.
- Continuous five-second GPU sampling was used for the accepted 17-core runs.
  Runs with any non-authorized process were rejected, not averaged in.

### 2. contract

`CON-001` — **passed**

- Qwen gate, up, and down weights remain `torch.bfloat16`.
- The fused gate/up path uses one BF16 `F.linear`, splits at the original
  intermediate dimension, applies the original SiLU and multiply, and executes
  the original BF16 down projection.
- The view aliases adjacent gate/up backing storage; it is not a copied or
  packed quantized representation.
- CPU activation reuse is keyed by the same token slice.  Batched expert
  outputs are split back into the original expert rows before weighted reduce.
- Named references are the original three-linear BF16 Qwen expert, original
  per-expert transfers, and original routing implementation.
- User acceptance does not require bitwise end-to-end generation.  The focused
  BF16 helper test nevertheless achieved exact output equality on 1, 2, and 17
  token shapes.

### 3. bottleneck

`BOT-001` — **supported with profiler limitation**

- Target-level evidence: `gpt_output/phase9_formal_ab.md`.
- BF16 operator evidence: `gpt_output/phase8_bf16_scaling_findings.md` and
  `gpt_output/bf16_scaling_sweep_pinned.json`, SHA256
  `1914fc5f034eb4a76144e1c620160be7f12eda67673ec9feff137840cf8723ea`.
- oneDNN verbose evidence identifies an AMX BF16 brgemm kernel.
- CPU torch-profiler traces are under `gpt_output/profiles/`: the reference
  attributes about 76% and the fused path about 55% of self CPU time to
  `aten::mm`.
- The mandatory unified profiler entrypoint was run against
  `gpt_output/profiles/cpu_expert_bf16_fused.trace.json`.  It correctly refused
  the trace because it only supports `sglang`, `vllm`, `trtllm`, and
  `tokenspeed`.  The trace was not mislabeled as another framework.

Fixed profiler tables required by the skill:

| Table | Status | Reason |
| --- | --- | --- |
| kernel table | blocked | custom SMoE is not a supported framework |
| overlap-opportunity table | blocked | custom SMoE is not a supported framework |
| fuse-pattern table | blocked | custom SMoE is not a supported framework |

The fixed three-table profiler gate is therefore unsupported, not silently
replaced by a fourth/custom table.  Hardware PMU evidence is separately blocked
by `perf_event_paranoid=4`.

### 4. hypothesis

`HYP-001` — **supported for CPU=9; refuted for 9 -> 17 scaling**

- Cause chain: repeated gate/up activation plus redundant transfers and poor
  logical-core placement -> extra BF16 operator, transfer, and topology cost ->
  fuse gate/up, reuse D2H, batch H2D, and pin physical cores -> lower CPU expert
  and decode latency.
- Prediction for same-core CPU=9 is supported: CPU expert mean improves 38.9%
  and mean decode improves 30.2%.
- Prediction that twice the compute cores can approach twice the throughput is
  refuted.  The pinned rotating operator reaches 47.1% throughput improvement;
  formal candidate CPU=17 is slower than candidate CPU=9.
- Rejected alternatives: cross-socket split placement and decode `torch.mv`.

### 5. implementation

`IMP-001` — **passed**

- Production changes are confined to six source files:
  `MoEModule/SMoE_base.py`, `MoEModule/qwen_moe.py`, `main.py`,
  `utils/cpu_affinity.py`, `utils/expertcache.py`, and
  `utils/model_loader.py`.
- Engagement logs and feature-off fallbacks exist for BF16 gate/up fusion and
  CPU transfer batching.
- The perf skill's read-only `summarize_diff.py` reports 26 total files,
  2014 additions, 79 deletions, and no binary files from `2de217b..972ec84`.
  The remaining files are focused tests, benchmark tooling, and reports under
  `gpt_output`.
- No quantized production code is present in the candidate diff.

### 6. validation

`VAL-001` — **passed for the recommended CPU=9 deployment**

- `test_qwen_cpu_bf16_fusion.py`: exact BF16 reference equality, zero-copy
  aliasing, 1/2/17-token shapes, and feature-off fallback — pass.
- `test_cpu_transfer_batching.py`: output equality, repeated activation reuse,
  one combined output copy, empty and single-expert cases — pass.
- `test_routing_sort_reuse.py`: original reference equality for 20 seeds,
  1/2/17 tokens, tied scores, cache states, and fallback sort — pass.
- `test_cpu_affinity.py`: physical-core, package, and shared-core placement for
  2/3/5/9/17 core requests — pass.
- Formal CPU=9: CPU expert mean 1.3774 -> 0.8416 ms; decode 0.18926 ->
  0.13210 s/token; total 23.0377 -> 16.7102 s.
- Formal CPU=17: results are retained without hiding the long tail.  Robust
  same-core metrics improve modestly, but 9 -> 17 candidate scaling regresses.

### 7. resource-risk

`RSK-001` — **continue**

- Device allocation is unchanged at 42515.34 MB in all four formal logs.
- Post-load RSS is approximately 119.8 GB for baseline and candidate, with less
  than 5 MB difference.
- The fused BF16 view allocates no duplicate expert weights.
- Topology risk is handled by keeping compute on one socket.  A 17-core request
  consumes all 16 physical cores there and places the shared worker on the
  other socket; this is functional but does not improve target latency.
- Rollback switches: `SMOE_CPU_BF16_FUSED_GATE_UP=0` and
  `SMOE_CPU_BATCH_TRANSFERS=0`.
- Remaining risks are host scheduling tails, lack of hardware counter access,
  and the unsupported unified-profiler framework gate.

## Hard-gate summary

```yaml
comparability: pass
diff_scope: pass
engagement: pass
correctness:
  criterion: exact helper references; BF16 semantic end-to-end
  result: pass
profiler:
  source_skill: llm-torch-profiler-analysis
  result: blocked
  reason: custom SMoE trace is outside the supported framework registry
performance:
  target_level: CPU expert and end-to-end decode
  repeats: five prompt summaries per formal configuration; two trials per
    pinned operator configuration
  CPU_9: pass
  CPU_17_same_core_candidate: modest improvement under robust statistics
  CPU_9_to_17_scaling: fail
resources:
  memory: pass
  stability: pass for accepted runs; contaminated runs rejected
fallback: pass
result: continue
evidence_ids: [REP-001, CON-001, BOT-001, HYP-001, IMP-001, VAL-001, RSK-001]
next_action: deploy candidate with CPU_CORES=9; do not increase to 17 expecting
  llama.cpp-like doubling on this BF16 M=1 expert workload
```
