# ADR 0001: Independent integration repository

- Date: 2026-11-21
- Status: superseded by [ADR 0002](0002-three-repository-integration.md)

This is the historical bootstrap decision, not the current implementation plan.
The independent integration repository remains, but the decision to avoid
capability forks and own scheduling/model execution here has been replaced.

## Context

Strata offers AVX2 quantized experts, but its full engine depends on supported
GPU backends. openTPU supplies FPGA execution. Missing work is heterogeneous
model execution, not simply replacing a few GPU calls.

## Decision

Create `lvyufeng/strata-opentpu` independently, not as a GitHub fork. Original
code is Apache-2.0; upstream licenses remain intact. Record immutable upstream
reference commits at bootstrap, without vendoring/submodules. Adapt audited CPU
kernels and FPGA execution separately; own scheduling/model execution here.

## Alternatives and consequences

Forking Strata is appropriate if a small backend patch suffices, but currently
brings unrelated GPU/product code and broad merge burden. Putting everything
inside openTPU suits an ordinary model port, not this CPU/mixed-quantization
experiment's separate lifecycle.

An independent repo is not a shortcut to working inference. We own correctness
and new operator work. If upstream interfaces need modification, maintain a
small reviewed patch or dependency fork and propose generic changes upstream.
Revisit this decision if a small Strata backend addition proves sufficient.
