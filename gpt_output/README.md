# SMoE CPU/Decode Optimization Artifacts

This directory contains the authoritative artifacts for the CPU expert and
decode-latency optimization work. Formal end-to-end runs use the command in
`deploy/deployment.md`; only `CPU_CORES` may vary.

CPU expert acceptance is BF16-only. Quantized CPU weights or quantized CPU
operators are excluded from the production path and from P0 performance claims.

Planned contents:

- baseline and optimized CPU=9/17 run logs
- CPU/operator and pipeline profile summaries
- before/after comparison tables
- correctness and performance validation notes
- final optimization report

The pre-existing untracked `优化/` directory is reference input and is not part
of these generated artifacts.
