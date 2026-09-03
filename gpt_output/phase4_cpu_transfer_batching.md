# Phase 4: Batched CPU-expert activation/output transfers

## Decision

`continue` — the mechanism and feature-off fallback pass CPU-only semantic
tests. GPU trace, target-topology latency, driver-visible memory, and full
deployment A/B remain blocked by the GPU process gate, so this is not yet a
`ship` result.

## Baseline contract

```yaml
repository: /home/guoying/SMoE
baseline_revision: 094e464
entrypoint: AbstractMoELayer._cpu_compute
workload:
  decode_typical: one token routed to multiple experts
  dtype: bfloat16 activation/output/CPU weights
hardware:
  gpu: unknown until compliant run
  interconnect: PCIe
correctness: exact tensor equality against feature-off CPU path in CPU-only test
feature_off: SMOE_CPU_BATCH_TRANSFERS=0
prefetch: disabled by user config
```

## Root cause and implementation

For one-token decode, every selected expert consumes the same hidden-state row.
The reference loop nevertheless executed one activation `.to("cpu")` and one
output `.to(cuda)` per CPU expert. These small transfers are latency dominated.

The candidate:

1. preserves the host token-index list built during expert dispatch;
2. caches one CPU activation per identical token-index tuple;
3. runs CPU experts in the original order;
4. concatenates ragged expert outputs along the token dimension;
5. issues one output H2D and exposes non-overlapping views to the existing
   weighting/scatter path;
6. retains the original per-expert transfer loop behind
   `SMOE_CPU_BATCH_TRANSFERS=0`;
7. logs requested D2H/H2D call and byte counts per prompt.

No stream, Event, weight residency, weight H2D, quantization pack, or prefetch
behavior changes in this phase.

## Live-range ledger

| Object | Owner | Candidate live range | Reuse/release condition |
|---|---|---|---|
| GPU selected activation | existing default stream | unchanged | existing expert dispatch lifetime |
| cached CPU activation | `_cpu_compute` local dict | first matching D2H through last CPU expert using that token tuple | function return |
| individual CPU output | `cpu_results` | expert completion through concatenation | after batched H2D/view creation |
| concatenated CPU output | `_cpu_compute` local | concatenation through synchronous `.to(device)` return | function return |
| GPU output batch | PyTorch allocator; views retained in `expert_out_dict` | H2D through B12 scatter | views released with layer dictionaries |

The extra host live range is bounded by CPU-expert outputs already produced in
the layer. There is no new device allocation beyond the concatenated payload,
which has the same logical bytes as the prior individual GPU outputs plus
allocator/concatenation effects that still require measurement.

## H2D/D2H ledger

Let `C` be CPU experts and `U` unique token-index tuples in a layer.

| Transfer | Reference calls | Candidate calls | Payload bytes |
|---|---:|---:|---:|
| activation D2H | `C` | `U` | candidate removes repeated identical rows |
| CPU output H2D | `C` | `1` when `C>0` | unchanged logical output bytes |

For the common one-token decode case, `U=1`, so both directions issue one
framework copy request per layer regardless of the number of CPU experts.
Runtime counters describe requested `.to` calls, not profiler-proven PCIe
transactions.

## CPU-only validation

```text
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. conda run -n Nmoe \
  python gpt_output/test_cpu_transfer_batching.py
PASS: CPU transfer batching preserves outputs and reduces copy calls
```

Coverage includes BF16, repeated token slices, ragged multi-token slices,
multiple experts, one expert, no CPU experts, exact feature-on/off outputs,
activation storage reuse, and transfer-call/byte counters. The main modules
also pass `py_compile` and `git diff --check`.

## Case calibration

- Success candidate: repeated one-token activation transfers are coalesced and
  output calls become one; mechanism is proven by counters and exact output.
- Failure condition: if concatenation/allocation costs exceed saved PCIe call
  latency or increase device peak beyond the guard, set
  `SMOE_CPU_BATCH_TRANSFERS=0` and mark the diff `stop`.
- Not applicable: layers with zero or one CPU expert do not reduce call count;
  the candidate has an explicit fast path and preserves behavior.

## Remaining validation gates

- Both GPUs must report zero processes.
- Run the `deploy/deployment.md` command unchanged except CPU=9/17.
- Compare baseline and candidate in paired runs with identical config/workload.
- Collect actual D2H/H2D timeline, exposed tail, decode latency, CPU expert
  forward statistics, and allocated/reserved/driver-visible memory.
- Promote to `ship` only if layer/end-to-end gains exceed noise and capacity is
  safe; otherwise use the feature-off fallback and remove the default change.
