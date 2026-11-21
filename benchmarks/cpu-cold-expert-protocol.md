# CPU cold-expert evidence protocol (source-only)

> **No measurements have been made by this protocol or its tests.** Strata already
> computes cold experts on CPU while keeping hot experts resident on an accelerator.
> This integration reuses that design; this document is not a new CPU engine,
> hardware authorization, a compatibility certificate, or a >10 tok/s result.

Protocol: `strata-cpu-cold-expert-v1`, schema version `1`. The source input is
[Strata at 3142b66b96e9c0e775e30f620b11b191a1427c2a](<https://github.com/lvyufeng/Strata/tree/3142b66b96e9c0e775e30f620b11b191a1427c2a>),
as pinned by [the integration source lock](<../dependencies.lock.json>).
The executable specification is [the offline checker](<../scripts/check_cpu_evidence.py>);
[the sanitized partial template](<templates/cpu-cold-expert.json>) contains no
host observations, model paths, credentials, real hashes, or invented timings.
[The adversarial unit tests](<../tests/test_cpu_evidence.py>) are explicitly
synthetic and do not run any Strata kernels.

## 1. Scope, permissions and ownership

Strata owns numerical kernels, quantization, CPU worker pools, expert layout,
cache/MTP and accelerator backends. openTPU owns RTL/runtime/DMA. This repository
owns source combinations, experiments, evidence validation and handoff.
No upstream implementation is copied here; fixed metadata below is an audited
source contract, not a second quantizer.

Creating or validating a report authorizes **no** model download/extraction,
package installation, dependency build, CPU experiment, host probe, NUMA workload,
FPGA programming or publication. Each actual operation needs its separately
recorded authorization. A future experiment may validly conclude **Reject**.
Private raw logs stay in ignored `local/`, `results/` or `logs/`; publish only
reviewed sanitized summaries. Do not include weights, bitstreams, vendor IP,
addresses, secrets or unsanitized logs.

Static findings are **unconfirmed static analysis** until independently measured.
CPU branch timing is not model decode timing. Full routing remains top-10,
expert-skip=0, MTP off for this baseline. Any later MTP-on study is separate and
counts only accepted output tokens. Qwen3.8-Flash-Next quality and single-user
wall decode **>10 tokens/s** remain independent, unfulfilled acceptance gates.

## 2. Pinned packing and activation contracts

The
[type IDs and block metadata](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/types.hpp#L12-L36>),
[native vec-dot mappings and whole-block rejection](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/quant/types.cpp#L22-L61>)
are:

| Target | Type ID | Weight block | Packed bytes | Native activation |
| --- | ---: | ---: | ---: | --- |
| IQ2_XS | 17 | 256 values | 74 | Q8_K, 256 values / 292 bytes |
| Q2_0 | 42 | 64 values | 18 | Q8_0, 32 values / 34 bytes |
| IQ3_S | 21 | 256 values | 110 | Q8_K, 256 values / 292 bytes |

IQ2_XS contains an FP16 scale, grid/sign indices and scale bytes; it is **not
FP4**. See its
[packed struct](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/strata-quant-common.h#L379-L385>).
IQ3_S contains FP16 `d`, `qs`, `qh`, signs and scales; it is not interchangeable
with same-sized Q3_K. See
[IQ3_S](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/strata-quant-common.h#L403-L412>)
and [Q3_K](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/strata-quant-common.h#L317-L324>).
Q2_0 has FP16 `d` and 16 packed code bytes, reconstructing `d * (code - 1)`;
see [the Q2_0 block](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/strata-quant-common.h#L195-L201>)
and [Q8_0](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/strata-quant-common.h#L266-L272>).

**Native metadata is not proof of actual pool dispatch.** The Q2 pool uses
[ActQ](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/kernels/cpu/expert.hpp#L68-L101>)
with int8 values, FP32 scale, chunk sums/auxiliary data, and
[act_quant_any/q2_rows_any](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/kernels/cpu/expert_layout.hpp#L93-L102>).
Legacy ActQ uses half-away rounding; the oracle Q8_0-like path uses FP16 scale
and x86 nearest-even. Record `ActQ-legacy` or `ActQ-oracle-Q8_0` as actually
selected, not merely `Q8_0`. See
[the Q2 pool GU/down call sites](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cpu/pool.cpp#L1225-L1304>).

`STRATA_Q2_BITPLANE=1` is an
[opt-in path, not the default](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cpu/q2_avx2.cpp#L67-L85>).
It can change floating summation; it requires its own correctness report before
performance comparison. This version requires one build/run environment across
its whole matrix. Use a fresh process and separate report for each arithmetic,
activation or environment choice; do not hot-toggle cached environment decisions.
Supported arithmetic declarations are `default` and `window-invariant`. Every
known arithmetic declaration must agree report-wide, including Reject/error
attempts; `null` remains missing, never an assumed default. Activation agreement
is per **actual weight format**, not equality between GU and down: full cases use
profile gate/down formats, while both block/dot dispatch slots use their target
format. Keep the fixed `Q8_K`/`Q8_0` contracts; all actual `Q2_0` slots across
cases, units and roles must choose one ActQ mode, because the
[oracle flag is process-global](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cpu/expert_layout.cpp#L400-L411>).
Unknown profile/format/activation stays incomplete; it is not inferred from a
full case's target label.

## 3. Real per-role geometry, not a fabricated homogeneous expert

The
[native layout constructor](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cpu/native_expert.cpp#L365-L417>)
and [per-role layout contract](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/kernels/cpu/expert_layout.hpp#L23-L57>)
require:

- Exactly three selected profiles, one for each target gate/up format.
- Gate and up shapes `[n_ff, n_embd]`; down `[n_embd, n_ff]`.
- Gate and up share the selected target format. Down records its **actual**
  tensor type, name, shape, block size, row stride, hash and activation mapping.
- All columns must be whole weight and native activation blocks; no rounding
  down, invented padding, transposition or silently substituted tensor.
- Blob packing is `[gate rows | up rows | down rows]`:
  `up_offset = gu_row_bytes * n_ff`, `down_offset = 2 * up_offset`,
  `blob_bytes = down_offset + down_row_bytes * n_embd`, gate offset `0`.
- Record model identity/hash, actual selected layer/expert, source tensor-metadata
  hash, each role's tensor hash and the packed blob hash.

A width-640 down projection **cannot** be packed as IQ2_XS or IQ3_S:
`640 % 256 != 0`. Do not call it compatible or reshape it to make a test pass.
Retain an explicit rejection in raw attempted-layout observations; the checker
rejects contradictory profile JSON rather than accepting it as a valid measured
profile. Select real compatible mixed-role metadata, or report the concrete
unsupported geometry and stop. No incompatible profile can satisfy strict
completeness by adding an error reason.

The only auxiliary down format audited in this schema besides the three targets
is Q8_0, type 8, 32 values / 34 bytes. This is **not a claim that the real model
uses Q8_0 down tensors**; the test fixture uses it only as unmistakably synthetic
mixed-role geometry. If actual tensors require another type, extend the protocol,
source anchors and adversarial tests through review; never relabel real tensors.

## 4. Build, ISA and observed NUMA requirements

Record compiler/version, all build flags, AVX2 translation-unit flags, quant
backend (`strata-own` or `ggml-pinned`), pinned ggml revision when used, compiled
ISA floor, binary SHA256, build-log SHA256 and the complete selected relevant
environment map. Build flags must account for AVX2/FMA/F16C (`-mavx2`, `-mfma`,
`-mf16c`, or `/arch:AVX2` for the appropriate toolchain).

The
[compile-floor caveat](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/cmake/backends/cpu.cmake#L276-L330>)
means runtime ISA caps **cannot** downgrade instructions compiled into a ggml
object. This schema supports floors `none`, `avx`, `avx2`, never an unaudited
AVX512 floor. Unsupported hosts must produce a guarded refusal before executing
unsupported kernels, not a successful benchmark label.

Record raw CPUID leaf-1 ECX and leaf-7 EBX, feature booleans and whether XGETBV
was executed. The
[pinned probe](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/quant/isa.cpp#L37-L105>)
requires CPUID1 OSXSAVE bit27, AVX bit28, FMA bit12, F16C bit29; then safely
check `(xcr0 & 0x6) == 0x6`, and CPUID7 AVX2 bit5. Unexecuted XGETBV has
`xcr0=null`, not invented zero. Missing OSXSAVE must not execute XGETBV.
CPUID registers are uint32 and the observed XCR0 register is uint64; larger
integers are contradictory register observations, not additional feature bits.
Feature-probe and topology artifacts get hashes; architecture is x86_64 here.

Record socket count, NUMA nodes, CPU/core/socket/node topology and at least two
selected worker NUMA nodes. For **every** attempted case, including Reject/error:

- Requested CPU IDs, memory nodes/policy and first-touch plan must describe the
  selected matrix strategy and worker node. A contradictory request is malformed,
  not an attempted placement failure.
- Each worker's observed effective CPU mask and sampled execution CPUs, mapped
  through observed topology. Sampled CPUs must be within the effective mask.
- Observed memory policy, positive page counts per node and matching total pages.
  The observed policy must agree with its own residency: local/contention uses
  the selected worker node, remote excludes it, interleave uses at least two nodes.
- Observed first-touch CPUs/nodes, unique pages touched and observation hash; do
  not substitute the requested plan. Touch CPU/node identities must agree with
  topology; unique pages touched cannot exceed the observed allocation.
- For **acceptance**, local/contention pages and touch are on the worker node;
  remote pages/touch exclude that node; interleave pages/touch span at least two
  nodes. All effective worker CPUs remain on the selected worker node, every
  allocated page is touched, and achieved affinity/residency meet requested masks
  and nodes. Requested flags alone never count as observed success.
- Coherent requested-versus-observed differences, incomplete first-touch coverage
  or an ineffective competing workload may yield honest **Reject** (or a factual
  execution error). Internally contradictory observed policy/pages, topology,
  sample/effective masks or page totals are malformed even with a Reject reason.
- For **contention acceptance**, the competing workload is active, with CPU/node
  IDs, sanitized description and artifact hash on the contested node. Inactive
  competition or coherent placement outside that node may yield honest Reject;
  active observations still need their IDs and provenance. Other strategies need
  no competition for acceptance. Any `active=false` observation has empty IDs
  and null details; unknown is not false.
- Affinity/memory hashes plus actual attempted status, exit code, positive elapsed
  wall time and raw-attempt observation hash.

The
[pool NUMA planner](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cpu/pool.cpp#L841-L898>)
can refuse unknown topology or clear partial affinity plans. A requested pinning
plan therefore cannot prove effective placement. Hashes are envelope identities,
not independent authentication or proof that placement was physically achieved.

## 5. Mandatory derived matrix

The checker derives keys; it does **not** trust a claimed coverage count or
accept deletion of formats/roles/policies/cache states to shrink requirements.
Fixed targets: IQ2_XS, Q2_0, IQ3_S. Roles: gate, up, down.
Batch sizes include `1` and at least one `2..8`; thread counts include `1` and at
least one multi-thread count. Repeat count is at least `5`; warmup count at
least `1`. Selected worker nodes include at least two observed topology nodes.

| Unit | Required axes | Fixed controls |
| --- | --- | --- |
| `block` | 3 formats × 3 roles | batch1/thread1/warm/local, smallest selected node |
| `dot` | 3 formats × 3 roles × every batch × every thread count | warm/local, smallest selected node |
| `full` | 3 actual mixed profiles × every batch × every thread × warm/cold × local/remote/interleave/contention × every selected worker node | all projection roles, same-iteration full chain |

For two batch sizes, two thread counts and two worker nodes: **237 cases**
(`9 + 36 + 192`). Every selected node sees every full strategy. Unknown worker
nodes make the expected count unknown and the report incomplete; no fabricated
host is inserted to satisfy it. Cases have unique IDs and unique derived keys;
out-of-matrix entries and duplicate IDs/profiles/keys are errors.

Generated block/dot controls use finite seeded values covering exactly `zero`,
`signed`, `scale-edges`, `grid-sign-edges`. Block controls have one whole weight
block per scenario row. Dot controls record rows, whole-block columns, row bytes,
packed bytes and input/packed hashes; exercise every selected batch/thread.
Controls are generated data, not real model tensors. This does not make the
**report** synthetic if they are actually executed as part of an authorized
measured experiment; fabricated unit-test outcomes do make it synthetic.

Full cases use the real selected profile, not generated block/dot geometry.
One profile per target does not prove exhaustive model-layer/expert coverage;
expand the protocol if such a claim is needed.

## 6. Numerical and full-chain gates

For every required numerical stage, keep finite actual/reference vectors, shape,
canonical vector hashes, reference kind/revision/artifact hash, tolerances,
recomputed failure count and declared pass. The only accepted criterion is:

```text
for every element i:
abs(actual[i] - reference[i]) <= atol + rtol * abs(reference[i])
```

Never OR global maximum-absolute and maximum-relative errors. There is no
summary-counter-only shortcut in v1: the checker recomputes **every** element.
Nonfinite or overflowing input, derived acceptance arithmetic, vector conversion
or wall-time statistics fail closed as malformed evidence (CLI exit1), not pass.
It checks declared tolerance consistency; domain reviewers must independently
approve meaningful tolerances/reference implementations. Identical fabricated
arrays in unit tests prove only the checker, not kernel parity.

Block checks have shape `[scenario_rows, block_values]`; dot `[batch, rows]`.
Full stages are `gate`, `up`, `hidden`, `down`, with shapes `[batch, n_ff]`
for the first three and `[batch, n_embd]` for down.
References are `strata-scalar-reference` at the exact source SHA or
`ggml-pinned-reference` at the exact reviewed SHA recorded in
`build.ggml_revision`; a missing build revision remains incomplete and a
contradictory reference revision is malformed. Artifact hashes do not
authenticate or execute those implementations.

Vector hash format is SHA256 of compact ASCII JSON (`separators=(",", ":")`)
after converting each finite number to float64, without NaN/Infinity. It is an
envelope checksum, **not** the checksum of packed weights or output memory.

Full execution must be:

```text
quantize x -> gate/up -> SiLU(gate) * up -> quantize resulting h -> down -> sync
```

Trace and timing stage order is exactly `x_quant`, `gate`, `up`, `silu_multiply`,
`h_quant`, `down`, `synchronize`. Link hashes from actual same-iteration inputs
and outputs, not invented names:

- input x → x quantization; quantized x → both gate and up inputs;
- checked gate/up outputs → the corresponding SiLU/multiply inputs;
- checked hidden output → **down quantization input**;
- resulting quantized h → **down kernel input**;
- checked down output → synchronized output.

The checker structurally enforces each equality. It cannot certify that the
trace was honestly produced; keep raw trace artifacts for independent review.

The existing
[kq_cpu_bench setup](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cpu/kq_cpu_bench.cpp#L132-L164>)
creates unrelated random `h` and quantizes it before timing.
[Its down path](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cpu/kq_cpu_bench.cpp#L175-L219>)
does not consume the same iteration's gate/up hidden output.
**`--part both` is not a full branch** and GB/s/ms outputs are not inference.
The
[IQ AVX2 parity helper](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cpu/iq_avx2_parity.cpp#L275-L394>)
is an existing synthetic correctness control, not real model/NUMA/full-chain
measurements and not a replacement for the matrix.

## 7. Timing and outcome semantics

Every executed block/dot/full case records monotonic **wall** clock, positive
warmup durations, ordered nonoverlapping raw start/end timestamps, positive raw
repeat durations, explicit cache preparation and its observation hash. Full
scope is `full-branch-with-quantization-and-synchronization`; CPU-time-only,
precomputed-independent-down and isolated dot timing cannot claim that scope.
Recompute median, nearest-rank p95 (`ceil(0.95*n)-1` in sorted zero-based values),
and population variance from the raw repeat durations. Include both warm/cold
conditions and contention. Timing a batch is not counting accepted decode tokens.

- `pass`: attempted, exit0, all elementwise checks, usable ISA and effective
  placement pass; no reason. A contradictory declared pass is malformed JSON.
- `reject`: attempted, exit0, all required checks/timing/trace/observations present,
  at least one numerical/placement/ISA gate failed; a nonempty factual reason.
  A reason without an observed failure is not a rejection certificate.
- `error`: actual guarded refusal/failed execution, nonzero exit, nonempty reason,
  full actual source/build/feature/model/affinity/first-touch/page/attempt
  observations for **every derived case**. `checks=[]`, `trace=null`,
  `timing=null` because completed execution observations are inapplicable, not
  fabricated. An error reason never waives provenance, build, ISA, tensor,
  placement, model geometry or matrix attempt coverage.

Incomplete observations remain null and block strict completeness even when
outcome is Reject/error. Planned/unattempted entries (`started=false`) are not
experimental records. A one-line error cannot stand in for the attempted matrix.

## 8. JSON and offline CLI contract

```bash
python3 scripts/check_cpu_evidence.py
python3 scripts/check_cpu_evidence.py results/sanitized-cpu-report.json
python3 scripts/check_cpu_evidence.py --strict-complete results/sanitized-cpu-report.json
python3 scripts/check_cpu_evidence.py --strict-accept results/sanitized-cpu-report.json
python3 -m unittest discover -s tests -p test_cpu_evidence.py -v
```

The checker reads **only** the explicit JSON input (default partial template)
and the fixed integration source lock. JSON strings, paths, URLs, environment
entries and command-looking text are **inert data**: no fetching, importing,
building, shell execution, binary loading or traversal to referenced artifacts.
The test suite has no upstream/host/model/device dependency.

Top-level fields are exactly `schema_version`, `protocol`, `evidence_kind`,
`source`, `provenance`, `build`, `host`, `routing`, `matrix`, `profiles`, `cases`,
`notes`; nested object field sets are fixed by the checker. Unknown fields,
duplicate JSON keys, nonfinite/overflowing numeric data, booleans as numbers,
contradictory metadata, false passes and non-wall timing are errors.
Unknown observations are null, never guessed zero or `unsupported`.

| Mode | Coherent incomplete | Complete synthetic | Complete measured Reject/error | Complete measured all gates pass |
| --- | --- | --- | --- | --- |
| Default consistency check | exit0, incomplete | exit0, schema-complete | exit0 | exit0 |
| `--strict-complete` | exit1 | exit1 | exit0 | exit0 |
| `--strict-accept` | exit1 | exit1 | exit1 | exit0 |

Malformed/contradictory input exits1 with an error; mutually exclusive strict
flags exit2. `evidence_kind` is `partial`, `synthetic` or `measured` and must
match each case. A measured declaration cannot use `synthetic-fixture` model
profiles. Unit fixtures always declare synthetic; the isolated mocked tests of
measured **declarations** are not real measurements and are not published reports.
A dishonest fully rewritten declaration cannot be authenticated by a JSON checker.

Output distinguishes `schema_complete`, `measured_evidence_complete`,
`declared_cpu_correctness_pass`, missing fields/cases and mode satisfaction.
A complete synthetic report can have `schema_complete=true`, but never
`measured_evidence_complete=true` or a measured correctness pass. Every output
has `structural_consistency_only=true`,
`measurements_independently_verified=false`, `inference_ready=false`.
Neither strict mode changes that truth boundary.

Before any performance/acceptance claim, reviewers must inspect raw experimental
provenance, reference/tolerance policy, compiler/dispatch, real per-role tensor
identities, effective NUMA/affinity, same-iteration dataflow and raw wall samples.
Then independently validate model correctness and actual accepted-token
wall decode throughput. None of that work has been performed by these artifacts.
