# Phase 3 — Decode synchronization and stream audit

## GPU gate

The 2026-09-02 recheck found GPU 1 occupied by Isaac Sim (PID 1393879,
approximately 2.6 GB and 51% utilization). GPU 0 was idle. No performance or
decode test was started because both devices must be process-free.

## Serial chain and streams

Current per-layer chain with prefetch disabled:

```text
default stream: gate -> softmax
host:           routing_weights.tolist()
host/GPU:       route/top-k -> selected activation gathers
default stream: shared expert
host thread:    submit cache-hit GPU work
load stream:    queued BF16 expert H2D
main host:      D2H activation -> CPU BF16 expert -> H2D output
host:           load queue drain + load_stream.synchronize()
host:           background submission drain
default stream: PCIe-loaded expert compute
device-wide:    torch.cuda.synchronize()
default stream: index_add -> shared-expert add -> next layer
```

Dependency observations:

- Load-stream weight writes have a RAW edge to loaded-expert compute. The
  current `load_stream.synchronize()` is conservative but correct.
- Cache-hit, loaded-expert, CPU-output weighting, scatter, and shared-expert
  operations are all submitted to the device default stream. After the host
  background worker is joined, stream order provides their execution ordering.
- The B12 device-wide synchronize therefore appears redundant, but removing it
  remains a candidate until a GPU-idle correctness and timeline A/B is possible.
- `compute_stream` and `predict_stream` are allocated but unused. The comments
  claiming `compute_stream.wait_event(copy_done_event)` do not match the active
  code; `on_expert_loaded` is always `None`.

## Implemented exact hot-path reduction

With `if_replace=true`, the old decode path did all of the following per layer:

1. run GPU `torch.topk`;
2. run host `cache_router`, which replaces every top-k index;
3. create the replacement index tensor on GPU;
4. call `.tolist()` on that same tensor to return the indices to the host.

The candidate skips the unused GPU top-k and reuses `cache_router`'s CPU index
list. Routing weights are gathered on GPU from those indices. Prefill and all
non-replacement paths retain the original `torch.topk` behavior.

This removes one unused kernel and one host/device synchronization per MoE layer
for the configured decode path without changing selected expert IDs or weights.

## Hidden synchronization ledger

| Site | Status | Reason/next action |
|---|---|---|
| `routing_weights.tolist()` | required today | host cache routing consumes all scores |
| replacement `topk_idx.tolist()` | removed | list already exists on host |
| replacement GPU `torch.topk` | removed | result was always overwritten |
| activation `.to("cpu")` | required today | CPU expert input dependency |
| CPU result `.to(cuda)` | required today | output consumed by GPU scatter |
| `load_stream.synchronize()` | conservative | replace with last-copy Event only after timeline/correctness proof |
| B12 `torch.cuda.synchronize()` | likely redundant | candidate for feature-gated GPU A/B |

## Skill decision

`continue`. The dependency DAG supports narrower Event waits and removal of the
B12 device-wide barrier, but the cuda-stream skill requires real timeline,
correctness, resource, and end-to-end evidence. Those gates cannot run while
GPU 1 is occupied, so no stream-order change is claimed as shippable yet.
