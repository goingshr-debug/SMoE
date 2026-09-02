# Phase 3: Decode routing sort reuse

## Decision

`continue` — the change passes CPU-only routing equivalence and static lifetime
review. It is not yet labeled `ship` because the required deployment A/B is
blocked by a process on GPU 1.

## Scope

- Reuse the descending expert sort produced by
  `replaceset_between_tokens()` in `cache_router()`.
- Scan only the top-k prefix when building `replaceset`/`allset`.
- Stop the replacement-candidate scan once descending scores reach `rscore`.
- Remove write-only routing dictionaries and counters.
- Keep selected experts pinned immediately inside `cache_router()`, while
  skipping the redundant B2 pin pass on the replacement path.
- Preserve the default two-value `replaceset_between_tokens()` API used by
  Qwen, DeepSeek, and Xverse predictor paths.

## Equivalence evidence

CPU-only test:

```text
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. conda run -n Nmoe \
  python gpt_output/test_routing_sort_reuse.py
PASS: fused routing sort matches the pre-change reference
```

Coverage:

- 20 deterministic random seeds;
- token counts 1, 2, and 17;
- 64 experts, top-k 4, replacement ratio 0.25;
- multiple resident-cache layouts;
- tied scores around routing thresholds;
- both the sort-reuse path and compatibility fallback path;
- exact routed indices, `top_uid`, `ready_compute()` order, and full slot fill;
- legacy two-return and new four-return helper APIs.

The early exit is equivalent because `sort_scores` is descending: after a
score is `<= rscore`, no later score can enter the open interval
`(rscore, midscore)`. Cache pinning remains before the first GPU consumer can
use or expose the routed selection.

## Performance gate

No routing microbenchmark or decode benchmark was run. At review time GPU 1
was occupied by:

```text
PID 1393879  /isaac-sim/kit/python/bin/python3  2575 MiB
```

Per the run gate, formal performance work waits until both GPU 0 and GPU 1
have zero processes. The next measurement must use the command from
`deploy/deployment.md` unchanged except for `CPU_CORES=9` and
`CPU_CORES=17`, with prefetch remaining disabled in the user-owned config.
