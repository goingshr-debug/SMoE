# Phase 1 follow-up — INT4 scale layout

The PyTorch CPU INT4 kernel requires `qScaleAndZeros` to be contiguous after
transposing it to `[groups, output_rows, 2]`. The initial implementation passed
a non-contiguous view. This follow-up materializes the expected layout before
packing/forward and reruns all selected CPU=9/17 evidence.

- Separate and fused gate/up packs now produce exactly equal INT4 outputs.
- CPU=9 paired medians: 0.2155 ms and 0.2132 ms.
- CPU=17 paired medians: 0.1412 ms and 0.1405 ms.
- CPU=9→17 throughput improvement: 52.5% and 51.7%.
- Fixed-core speedup over BF16: 2.67x–2.82x.
- Token counts 1, 2, and 17 return finite BF16 tensors of the expected shape.

The previous performance conclusion remains valid, while the scale metadata
contract is now explicit and correctly implemented.
