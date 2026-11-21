# Roadmap

All implementation milestones are pending.

## M0: Bootstrap (complete)

Independent repo, license, upstream references, architecture, model shortlist,
and hardware-free checks. No upstream implementation imported.

## M1: CPU expert microbenchmark

Audit pinned Strata dependency closure/licenses; add minimal native bindings,
not the GPU application. Cover actual mixed formats. Compare deterministic small
tensors with a trusted reference; measure single/multi-token separately. Sweep
threads/NUMA and measure bandwidth. Exit: correct experts and reproducible local
throughput, not model speed. Avoid full-expert FP32 expansion in measured paths.

## M2: Minimal heterogeneous chain

Discover capabilities through the FPGA adapter. Transfer activations, execute
an eligible partition, merge CPU results; check lifetime, ordering, errors and
numerics. Compare synchronized latency with identical CPU-only work. Reject
unhelpful FPGA partitions.

## M3: Complete target model

Implement GGUF and required operators. Text first, no MTP, full top-10, stated
short context. Compare logits/tokens against a trusted runtime with controlled
arithmetic/sampling. IQ2_XS primary, Q2_0 speed and IQ3_S quality comparisons.
Whole-model CPU baseline needs implementation or a separately verified reference
runtime; CPU expert libraries alone are insufficient.

## M4: Profiling-driven optimization

Place only measured-beneficial FPGA work. MTP separately. Add low-bit hardware
or RTL only with evidence. Report wall decode, TTFT, correctness/quality, memory
and link traffic. >10 tokens/s stays unproven until the full-model protocol passes
without dropped experts.
