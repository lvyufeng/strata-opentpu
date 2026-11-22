# CPU cold-expert evidence protocol v2

Status: **draft for review — not in force**. v1 is still the only reviewed
protocol; this file and `scripts/check_cpu_evidence_v2.py` become authoritative
only after review. v2 does not replace, amend or reinterpret any v1 artifact, and
it records no measurement.

- Schema version: `1` (unchanged; the `protocol` string distinguishes v2).
- Protocol id: `strata-cpu-cold-expert-v2`.
- Superseded-on-review: `benchmarks/cpu-cold-expert-protocol.md` (v1).
- Checker: [scripts/check_cpu_evidence_v2.py](../scripts/check_cpu_evidence_v2.py)
  (offline; imports the frozen v1 core privately, asserts its reviewed `LAYOUT`
  still matches, then rebinds only `PROTOCOL`, `FORMATS` and `TYPE_INFO` in that
  private copy).
- Template: [templates/cpu-cold-expert-v2.json](templates/cpu-cold-expert-v2.json).

v2 exists because **v1 cannot express a strict-complete report from the real
downloaded model**. v1's target formats are IQ2_XS / Q2_0 / IQ3_S; the audited
routed-expert gate/up tensors are IQ2_S / IQ2_XXS / IQ1_M and every routed-expert
down tensor is Q2_0 at `n_ff = 640` columns. A v2 report must be rejected by v1
and a v1 report must be rejected by v2; the two protocols never share a report.

This document is structurally self-contained: sections 4–8 restate the v1
build/ISA, matrix, numerical, timing and CLI contracts unchanged in substance.
Where wording differs, the v1 document remains the authority for v1 evidence.

## 1. Scope and honesty boundary

This protocol covers CPU cold-expert evidence for a model whose routed experts
are partly resident on GPU and partly computed on CPU. It defines the evidence
schema, derived case matrix, geometry and numerical gates and the offline
validation rules. It does **not** authorize an experiment, provide a build, read
model payloads, identify a host, or measure anything.

A report produced under this protocol is still subject to these fixed flags,
which no mode can turn off:

- `structural_consistency_only: true` — the checker verifies declared JSON
  structure and derived consistency only.
- `measurements_independently_verified: false` — no number in a report is
  verified against raw artifacts by the checker.
- `inference_ready: false` — complete CPU branch evidence is not end-to-end
  inference, model quality or decode throughput evidence.

Missing or `null` means unknown, never a default, never a failure and never a
passed measurement. The protocol name `IQ2_S`/`IQ2_XXS`/`IQ1_M` identifies the
actual routed gate/up tensor types; the `ffn_down_exps` tensor is Q2_0 and must
be recorded as such. Shared-expert rows, attention, embeddings, output heads and
non-routed tensors are out of scope and must not be folded into this schema,
`FORMATS` or `TYPE_INFO`.

## 2. v2 format matrix and why it changed

v1 audited exactly three target formats. The GGUF header inventory of the
downloaded model shows the real routed gate/up tensors are IQ2_S (type 22),
IQ2_XXS (type 16) or IQ1_M (type 29) depending on the layer, and each layer's
routed down tensor is Q2_0 (type 42). The audited inventory is the evidence
input; this protocol records the classification, it does not re-derive it.

v2 differs from v1 in exactly four points:

1. `PROTOCOL` becomes `strata-cpu-cold-expert-v2`.
2. `FORMATS`, the `matrix.formats` enum and the target set become
   `IQ2_S`, `IQ2_XXS`, `IQ1_M` in that order; gate/up per profile use these.
3. `TYPE_INFO` gains exactly three entries — IQ2_S, IQ2_XXS, IQ1_M. The four
   entries inherited from v1 (IQ2_XS, Q2_0, IQ3_S, Q8_0) stay unchanged so Q2_0
   remains recordable as the real down role and so shared tables/v1 fixtures
   still parse; IQ2_XS and IQ3_S are **not** v2 targets.
4. The template's representative profiles, their source anchors, byte
   derivations and the adversarial tests describe the real routed-expert
   geometry (section 3). The layout contract itself is inherited unchanged, and
   no gate, tolerance, ordering or outcome rule changes.

Everything else — schema version, roles, policies, cache states, stages,
scenarios, features, kinds, hash formats, matrix derivation, numerical checks,
trace/timing requirements, modes and CLI behaviour — is inherited unchanged from
v1.

Mechanically the checker pins the frozen v1 checker by SHA256, loads it as a
module instance only it owns, and reads that instance's `PROTOCOL`, `FORMATS`,
`TYPE_INFO` and `LAYOUT` to assert they still equal the reviewed v1 values
([scripts/check_cpu_evidence_v2.py:74–104](../scripts/check_cpu_evidence_v2.py#L74-L104)).
It then makes exactly three code rebindings in that private instance —
`PROTOCOL`, `FORMATS`, `TYPE_INFO` — and no fourth rebinding
([scripts/check_cpu_evidence_v2.py:107–111](../scripts/check_cpu_evidence_v2.py#L107-L111)).
`LAYOUT` is only compared: v2 keeps v1's layout contract, so that comparison is a
read-only drift assertion, not a rebind. The template path, the CLI default
report, tests and documentation are retargeted separately without rebinding
anything else. An unrebased v1 semantic drifting underneath v2 is therefore
impossible without the pin failing.

The audit's naming warning still applies: pack-family names describe a
quantization family, not a promise that every tensor uses that type
([docs/audits/strata-execution.md:282](../docs/audits/strata-execution.md#L282);
[docs/architecture.md:43–44](../docs/architecture.md#L43-L44)). The acquisition
directory, `pin.json` and `urls.txt` say the artifact is an "IQ2_XS" build; the
header inventory says the routed expert tensors are not IQ2_XS. **Record the
actual type IDs; never relabel a tensor to match a family name.** The roadmap
already requires actual tensor formats to be covered
([docs/roadmap.md:83–84](../docs/roadmap.md#L83-L84)).

| Target | Type ID | Weight block | Packed bytes | Native activation |
| --- | ---: | ---: | ---: | --- |
| IQ2_S | 22 | 256 values | 82 | Q8_K, 256 values / 292 bytes |
| IQ2_XXS | 16 | 256 values | 66 | Q8_K, 256 values / 292 bytes |
| IQ1_M | 29 | 256 values | 56 | Q8_K, 256 values / 292 bytes |
| Q2_0 (recorded down role) | 42 | 64 values | 18 | Q8_0, 32 values / 34 bytes |
| Q8_0 (reviewed auxiliary only) | 8 | 32 values | 34 | Q8_0, 32 values / 34 bytes |
| IQ2_XS (v1 entry, not a v2 target) | 17 | 256 values | 74 | Q8_K, 256 values / 292 bytes |
| IQ3_S (v1 entry, not a v2 target) | 21 | 256 values | 110 | Q8_K, 256 values / 292 bytes |

Block padding is fixed by asserts, so the byte counts are not negotiable:
IQ2_S is `2 + 256/4 + 256/16 = 82` bytes
([ggml-common.h:401–402](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/third_party/strata-ggml/ggml/src/ggml-common.h#L401-L402>)),
IQ2_XXS is `2 + 256/8*2 = 66` bytes
([ggml-common.h:384–385](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/third_party/strata-ggml/ggml/src/ggml-common.h#L384-L385>)),
IQ1_M is `256/8 + 256/16 + 256/32 = 56` bytes
([ggml-common.h:437–438](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/third_party/strata-ggml/ggml/src/ggml-common.h#L437-L438>)).
All three take Q8_K activations with `nrows = 1` on the audited x86 path:
IQ2_XXS at
[ggml-cpu.c:337–342](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/third_party/strata-ggml/ggml/src/ggml-cpu/ggml-cpu.c#L337-L342>),
IQ2_S at
[ggml-cpu.c:362–367](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/third_party/strata-ggml/ggml/src/ggml-cpu/ggml-cpu.c#L362-L367>),
IQ1_M at
[ggml-cpu.c:374–379](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/third_party/strata-ggml/ggml/src/ggml-cpu/ggml-cpu.c#L374-L379>).
Type IDs and block metadata come from the Strata quant table
([types.hpp:12–39](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/types.hpp#L12-L39>),
[types.cpp:27,33,35](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/quant/types.cpp#L27,33,35>));
the ggml trait table agrees on block size and Q8_K mapping
([ggml.c:808–815,840–847,856–863](<https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/third_party/strata-ggml/ggml/src/ggml.c#L808-L863>)).

IQ2_S and IQ2_XXS are not FP4 and are not interchangeable; neither is IQ1_M.
A byte-count or name match never proves a format
([docs/audits/strata-execution.md:282](../docs/audits/strata-execution.md#L282)).

**Native metadata is not proof of actual pool dispatch.** The Q2 pool uses ActQ;
legacy ActQ rounds half-away, the oracle Q8_0-like path uses FP16 scale and x86
nearest-even. Record `ActQ-legacy` or `ActQ-oracle-Q8_0` as actually selected,
not merely `Q8_0`. `STRATA_Q2_BITPLANE=1` is an opt-in path, not the default; it
can change floating summation and needs its own correctness report before any
performance comparison. This version requires one build/run environment across
its whole matrix; use a fresh process and a separate report for each arithmetic,
activation or environment choice. Supported arithmetic declarations are
`default` and `window-invariant`, agreed report-wide, including Reject/error
attempts; `null` stays missing. Activation agreement is per actual weight format,
not equality between GU and down: full cases use profile gate/down formats, while
both block/dot dispatch slots use their target format. Keep the fixed Q8_K/Q8_0
contracts; all actual Q2_0 slots across cases, units and roles must choose one
ActQ mode because the oracle flag is process-global. Unknown
profile/format/activation stays incomplete; it is never inferred from a full
case's target label.

## 3. Real per-role geometry, not a fabricated homogeneous expert

The native layout constructor and per-role layout contract require
([v1 protocol §3](cpu-cold-expert-protocol.md#L89-L120)):

- Exactly three selected profiles, one for each target gate/up format.
- Gate and up shapes `[n_ff, n_embd]`; down `[n_embd, n_ff]`.
- Gate and up share the selected target format. Down records its **actual**
  tensor type, name, shape, block size, row stride, hash and activation mapping.
- All columns must be whole weight and native activation blocks; no rounding
  down, invented padding, transposition or silently substituted tensor.
- Blob packing is `[gate rows | up rows | down rows]`:
  `up_offset = gu_row_bytes * n_ff`, `down_offset = 2 * up_offset`,
  `blob_bytes = down_offset + down_row_bytes * n_embd`, gate offset `0`.
- Record model identity/hash, actual selected layer/expert, source
  tensor-metadata hash, each role's tensor hash and the packed blob hash.

The audited model has `n_embd = 2560`, `n_ff = 640`, 512 routed experts per
layer, top-10 routing plus a shared expert, MTP off, `expert_skip = 0`. Rows of
a gate/up tensor are `n_embd = 2560` wide (10 whole 256-value blocks), and rows
of a down tensor are `n_ff = 640` wide (10 whole 64-value blocks for Q2_0).
Three anchored representative profiles, one per target format:

| Profile id | Layer | Gate/up | Type | GU row bytes | Up offset | Down offset | Down | Blob bytes |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `iq2-s-layer0-mixed-role` | 0 | IQ2_S | 22 | 820 | 524800 | 1049600 | Q2_0 `{18,64,180}` | 1510400 |
| `iq2-xxs-layer1-mixed-role` | 1 | IQ2_XXS | 16 | 660 | 422400 | 844800 | Q2_0 `{18,64,180}` | 1305600 |
| `iq1-m-layer8-mixed-role` | 8 | IQ1_M | 29 | 560 | 358400 | 716800 | Q2_0 `{18,64,180}` | 1177600 |

Gate and up have shape `[640, 2560]`, the down role `[2560, 640]`; every profile
uses `layout = native-row-packed-gate-up-down-v1`, `expert = null` (one packed
blob per layer, not one expert index). Tensor names are the real GGUF names
`blk.<layer>.ffn_gate_exps.weight`, `blk.<layer>.ffn_up_exps.weight`,
`blk.<layer>.ffn_down_exps.weight`. `{18,64,180}` is Q2_0's block bytes / block
values / row bytes at 640 columns. The arithmetic above is derived, not measured:
`820 * 640 = 524800`, `2 * 524800 = 1049600`,
`1049600 + 180 * 2560 = 1510400`; likewise `660*640 = 422400`,
`2 * 422400 = 844800`, `844800 + 460800 = 1305600`; `560*640 = 358400`,
`716800 + 460800 = 1177600`. Source GGUF file offsets belong to the inventory
artifact and acceptance record, not to this schema; profiles record per-layer
packed geometry only.

One profile per target format is a representative selection: one real layer per
format records its actual tensors. It is not exhaustive model-layer or expert
coverage (the audited model has 48 layers and 512 routed experts per layer), and
it is not a claim that all 48 layers were observed. Expand the protocol, anchors
and adversarial tests through review if such coverage is needed.

A width-640 down projection **cannot** be packed as IQ2_S, IQ2_XXS or IQ1_M:
`640 % 256 = 128 != 0`. It is also not IQ2_XS or IQ3_S for the same reason. Do
not call it compatible, do not pad, reshape or transpose it, and do not relabel
it. A profile that declares a 256-value-block format on a 640-column down role
is rejected by the checker as contradictory JSON; the only recorded down format
for this model is Q2_0 at 640 columns, and a rejection reason can never make an
incompatible profile strict-complete. A deliberate rejection observation is
recorded as raw attempted-layout data outside the profile JSON, not as a valid
profile.

The only auxiliary down format this schema still carries besides Q2_0 is Q8_0
(type 8, 32 values / 34 bytes). It is inherited from v1 where it was explicitly
a synthetic fixture aid, **not a claim that the real model uses Q8_0 down
tensors**; here it must not be substituted for a real Q2_0 down role either.
That is an authoring rule, not a validator rule: a Q8_0 down role at 640 columns
is structurally legal (`640 % 32 == 0`) and must still be rejected in review (see
the validation boundary below). The v1 entries IQ2_XS and IQ3_S remain in
`TYPE_INFO` only so shared tables and v1 fixtures parse; a v2 evidence report
that uses either as a target format or as a role format on this geometry is
invalid and fails validation.

Shared-expert tensors observed in the inventory (`ffn_down_shexp`, the
`ffn_gate_shexp` variants, `ffn_gate_inp`, `gate_inp_shexp`) are **not routed
expert tensors** and are out of scope. Their types must not be added to
`FORMATS`, `TYPE_INFO` or a profile, and they are never evidence that a routed
format is compatible.

**Validation boundary.** The checker is structural and offline: it verifies that
declared roles, blocks, offsets, hashes and sizes are mutually consistent, not
that the declaration describes tensors that actually exist. The per-role
"actual tensor" rules inherited from
[v1 protocol:98–120](cpu-cold-expert-protocol.md#L98-L120) — record the actual
type, and never substitute, pad, reshape, transpose or relabel — are therefore
the author's obligation, not a machine-enforced gate. A report that substitutes
Q8_0 for the recorded Q2_0 down role (structurally legal at 640 columns), drifts
the representative geometry away from the anchored `n_embd = 2560` /
`n_ff = 640` of a real layer, records synthetic/control-derived data as a real
profile, records a tensor name or hash it did not take from the payload, or
declares a layer/format mix that is not in the audited inventory is **invalid
protocol evidence**, even if the validator exits `0` under `--strict-accept`.
Section 8 restates the exit-code boundary; reviewers must apply these authoring
rules.

## 4. Build, ISA and observed NUMA requirements

Record compiler/version, all build flags, AVX2 translation-unit flags, quant
backend (`strata-own` or `ggml-pinned`), pinned ggml revision when used, compiled
ISA floor, binary SHA256, build-log SHA256 and the complete selected relevant
environment map. Build flags must account for AVX2/FMA/F16C (`-mavx2`, `-mfma`,
`-mf16c`, or `/arch:AVX2` for the appropriate toolchain). The compile-floor
caveat means runtime ISA caps cannot downgrade instructions compiled into a ggml
object; supported floors are `none`, `avx`, `avx2`, never an unaudited AVX512
floor. Unsupported hosts must produce a guarded refusal before executing
unsupported kernels, not a successful benchmark label.

Record raw CPUID leaf-1 ECX and leaf-7 EBX, feature booleans and whether XGETBV
was executed. The pinned probe requires CPUID1 OSXSAVE bit 27, AVX bit 28, FMA
bit 12, F16C bit 29; then safely check `(xcr0 & 0x6) == 0x6`, and CPUID7 AVX2
bit 5. Unexecuted XGETBV has `xcr0 = null`, not invented zero; missing OSXSAVE
must not execute XGETBV. CPUID registers are uint32 and the observed XCR0
register is uint64; larger integers are contradictory register observations, not
additional feature bits. Feature-probe and topology artifacts get hashes;
architecture is `x86_64` here.

Record socket count, NUMA nodes, CPU/core/socket/node topology and at least two
selected worker NUMA nodes. For **every** attempted case, including Reject/error:
requested CPU IDs, memory nodes/policy and first-touch plan must describe the
selected matrix strategy and worker node; each worker's observed effective CPU
mask and sampled execution CPUs must be mapped through observed topology, with
samples inside the effective mask; observed memory policy, positive page counts
per node and matching totals are required, and the observed policy must agree
with its own residency (local/contention uses the selected worker node, remote
excludes it, interleave uses at least two nodes); observed first-touch CPUs/nodes,
unique pages touched and an observation hash are required and must agree with
topology, without substituting the requested plan. For **acceptance**,
local/contention pages and touch are on the worker node, remote pages/touch
exclude it, interleave pages/touch span at least two nodes, all effective worker
CPUs remain on the selected worker node, every allocated page is touched, and
achieved affinity/residency meet requested masks and nodes. Requested flags alone
never count as observed success. Coherent requested-versus-observed differences,
incomplete first-touch coverage or an ineffective competing workload may yield an
honest **Reject** (or a factual execution error); internally contradictory
observations are malformed even with a Reject reason. For **contention
acceptance**, the competing workload is active, with CPU/node IDs, a sanitized
description and artifact hash on the contested node; inactive competition or
coherent placement outside that node may yield honest Reject, and other
strategies need no competition. Any `active = false` observation has empty IDs
and null details; unknown is not false.

## 5. Mandatory derived matrix

The checker derives keys; it does **not** trust a claimed coverage count or
accept deletion of formats/roles/policies/cache states to shrink requirements.
Fixed targets: IQ2_S, IQ2_XXS, IQ1_M. Roles: gate, up, down. Batch sizes include
`1` and at least one `2..8`; thread counts include `1` and at least one
multi-thread count. Repeat count is at least `5`; warmup count at least `1`.
Selected worker nodes include at least two observed topology nodes.

| Unit | Required axes | Fixed controls |
| --- | --- | --- |
| `block` | 3 formats × 3 roles | batch1/thread1/warm/local, smallest selected node |
| `dot` | 3 formats × 3 roles × every batch × every thread count | warm/local, smallest selected node |
| `full` | 3 actual mixed profiles × every batch × every thread × warm/cold × local/remote/interleave/contention × every selected worker node | all projection roles, same-iteration full chain |

For two batch sizes, two thread counts and two worker nodes: **237 cases**
(`9 + 36 + 192`), exactly as v1 — the axes are unchanged, only the three format
labels are. Every selected node sees every full strategy. Unknown worker nodes
make the expected count unknown and the report incomplete; no fabricated host is
inserted to satisfy it. Cases have unique IDs and unique derived keys;
out-of-matrix entries and duplicate IDs/profiles/keys are errors.

Generated block/dot controls use finite seeded values covering exactly `zero`,
`signed`, `scale-edges`, `grid-sign-edges`. Block controls have one whole weight
block per scenario row. Dot controls record rows, whole-block columns, row bytes,
packed bytes and input/packed hashes; exercise every selected batch/thread.
Controls are generated data, not real model tensors. This does not make the
**report** synthetic if they are actually executed as part of an authorized
measured experiment; fabricated unit-test outcomes do make it synthetic. Full
cases use the real selected profile, not generated block/dot geometry.

## 6. Numerical and full-chain gates

For every required numerical stage, keep finite actual/reference vectors, shape,
canonical vector hashes, reference kind/revision/artifact hash, tolerances,
recomputed failure count and declared pass. The only accepted criterion is:

```text
for every element i:
abs(actual[i] - reference[i]) <= atol + rtol * abs(reference[i])
```

Never OR global maximum-absolute and maximum-relative errors. There is no
summary-counter-only shortcut: the checker recomputes **every** element.
Nonfinite or overflowing input, derived acceptance arithmetic, vector conversion
or wall-time statistics fail closed as malformed evidence (CLI exit 1), not pass.
It checks declared tolerance consistency; domain reviewers must independently
approve meaningful tolerances/reference implementations. Identical fabricated
arrays in unit tests prove only the checker, not kernel parity.

Block checks have shape `[scenario_rows, block_values]`; dot `[batch, rows]`.
Full stages are `gate`, `up`, `hidden`, `down`, with shapes `[batch, n_ff]` for
the first three and `[batch, n_embd]` for down. References are
`strata-scalar-reference` at the exact source SHA or `ggml-pinned-reference` at
the exact reviewed SHA recorded in `build.ggml_revision`; a missing build
revision remains incomplete and a contradictory reference revision is malformed.
Artifact hashes do not authenticate or execute those implementations.

Vector hash format is SHA256 of compact ASCII JSON (`separators=(",", ":")`)
after converting each finite number to float64, without NaN/Infinity. It is an
envelope checksum, **not** the checksum of packed weights or output memory.

Full execution must be:

```text
quantize x -> gate/up -> SiLU(gate) * up -> quantize resulting h -> down -> sync
```

Trace and timing stage order is exactly `x_quant`, `gate`, `up`, `silu_multiply`,
`h_quant`, `down`, `synchronize`. Link hashes from actual same-iteration inputs
and outputs, not invented names: input x → x quantization; quantized x → both
gate and up inputs; checked gate/up outputs → the corresponding SiLU/multiply
inputs; checked hidden output → **down quantization input**; resulting quantized
h → **down kernel input**; checked down output → synchronized output. The checker
structurally enforces each equality; it cannot certify that the trace was
honestly produced, so keep raw trace artifacts for independent review.

The existing `kq_cpu_bench` setup creates unrelated random `h` and quantizes it
before timing; its down path does not consume the same iteration's gate/up hidden
output. **`--part both` is not a full branch**, and GB/s or ms outputs are not
inference. The IQ AVX2 parity helper is an existing synthetic correctness
control, not real model/NUMA/full-chain measurement and not a replacement for
the matrix. For v2 the Q2_0 down role is the only real ActQ-relevant slot; its
declared ActQ mode must be consistent report-wide as described in section 2.

### 6.1 Declared native dispatch, ActQ and build record

**Names come from the runtime query.** Fill `dispatch.implementation` from a probe
printing `dot_kernel(Type(id), default_arith())->name`, `default_arith()`,
`best_isa()` and `native_gu_mt_min(gu_type)` (dot.hpp:18,28; dispatch.cpp:56,64;
isa.cpp:110–130; native_expert.hpp:73; all `file:line` cites are the pinned tree
`3142b66b96e9c0e775e30f620b11b191a1427c2a`) — on the section-4 host
`default_arith()`=`Avx2`, `best_isa()`=`Avx2`, `native_gu_mt_min()`=`2`. The field is
free text to the checker (v1 L113, L746); label-edited names are defects.

**batch=1: every grouped kernel is off.** `native_gu_mt_min()` defaults to 2
(`STRATA_IQ_MT_MIN` unset; native_expert.cpp:539–557) and grouping needs `nt >=
mt_min` (:653, :684), so one token takes the per-row tail `traits(f.gu_type)` +
`dot.dot` (:733, :742–762), binding `avx2_iq2_s_q8_K`, `avx2_iq2_xxs_q8_K` and
`avx2_iq1_m_q8_K` (dot_avx2.cpp:18,23–24). Q2_0 down matches no grouped branch
(:776–831), so it takes the per-row loop (:832, :838–845) with the
`Isa::Scalar`-registered `avx2_q2_0_generic` (dot_avx2.cpp:1–2, :26); no grouped name
is valid at batch=1.

**nt>=2: grouped only for IQ2_S/IQ2_XXS under `default`.** `iq512_supported` is
{16,17,18,21,22} (:279–281) and `iq256_supported` is {16,17,18,21,22,23} (:283–285).
This host has no AVX-512/VNNI, so `cpu512` is false (:683) and :684 falls to `avx2 &&
iq256_supported` → `iq256_gu_rows` (:699–701). IQ2_S (22) and IQ2_XXS (16) group;
IQ1_M (29) is in neither set and keeps `avx2_iq1_m_q8_K` at any `nt` —
`iq256x_supported` lists 29 (iq_avx2.cpp:538) but the gate uses `iq256_supported`.
Under `window-invariant`, on this host `singleton_iq` is true (:654, :685–693) and
there is no AVX-512 reference branch, so these are not grouped either; down stays
per-row.

**No environment switch may manufacture a name.** `build.environment` and each
`dispatch.environment` are inert string-to-string maps that must be equal (v1 checker
L199–203, L747–753): write absence as a string like `"absent"`, never `false`, `0` or
`null`. Leave absent and declare: `STRATA_CPU_IQ_ALL_EXACT1` (read at
native_expert.cpp:588), `STRATA_IQ_MT_MIN` (:544–545), `STRATA_IQ3S_MT1` (:553),
`STRATA_Q2_BITPLANE` (opt-in Q2_0 path, q2_avx2.cpp:65,85,147) and
`STRATA_NO_Q8K_AVX2` (:417–421). Declare absent, too, the switches that move
dispatch or layout: `STRATA_FORCE_ISA` (isa.cpp:50), `STRATA_FORCE_AVX2`
(expert_layout.cpp:76,116), the `STRATA_NO_IQ256/NO_IQ512/NO_IQ128/NO_Q8K_AVX1`
family (:575, :567, :583, :426), the `STRATA_KQ256/NO_KQ512/KQ512_MIN` group
(:658–662), `STRATA_NO_Q50_512`/`STRATA_Q50_512_MIN` (:795–796),
`STRATA_NO_IQ4NL` (:770), `STRATA_NO_IQ512X1` (:604) and `STRATA_ACT_DUMP*`
(:431–435).

**ActQ, restated for the native path.** Gate/up uses `native_quant_act` (:513–524):
while the Q8_K binding holds (:412–422: `cpu_avx2_ok()`, no `STRATA_NO_Q8K_AVX2`,
`gu_act == kTQ8_K`) it calls `q8k_quant_avx2` (:521; iq_avx2.cpp:394), else the Q8_K
ActKernel `avx2_q8_K` (act_avx2.cpp:22). Down uses `native_quant_h` (:526–537) and
takes the Q8_0 ActKernel `avx2_q8_0` (:536; act_avx2.cpp:20). Report fields are not
kernel names: declare `gate_up_activation="Q8_K"`; any case whose down role format is
Q2_0 must record `down_activation` as the one ActQ mode actually selected report-wide
(`ActQ-legacy` or `ActQ-oracle-Q8_0`, section 2; checker L756–765) — `"Q8_0"` names
the native kernel only and is never a legal Q2_0 label. `g_oracle_q8_0` is read only
through `expert_oracle_q8_0_enabled()` (expert.cpp:386,458,607; pool.cpp:1392,1412),
never in native_expert.cpp, so that process-global mode cannot change this branch's
kernel; the Q2_0 down is still the only real ActQ-relevant slot.

**Build and probe record.** From the checkout root:

```text
cmake -S engine -B build-native -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_OWN_CPU_QUANT=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF
```

`STRATA_GEOMETRY` defaults to `flash_next` (strata_geometry.cmake:11); with tests off
the CPU backend links ggml only inside `if(TARGET ggml-base)` (cpu.cmake:318–325), so
record `quant_backend="strata-own"`. The build makes `strata_kernels_cpu`
(cpu.cmake:196–202, :326–330) and `strata_quant` (quant.cmake:13); AVX2 units compile
`-mavx2;-mfma;-mf16c` (cpu.cmake:292–301) plus `-ffp-contract=fast` (quant.cmake:42),
so record the full flag string, compiler and version, `compiled_isa_floor` from the
`STRATA_ISA_FLOOR` used (checker L196–197; cpu.cmake:314–317), SHA256 of
`libstrata_kernels_cpu.a` and `libstrata_quant.a`, `build.build_log_sha256`,
`host.feature_probe_sha256`, `host.topology_sha256` and the environment map. The
schema allows one `build.binary_sha256` and one `provenance.raw_artifact_sha256` and
rejects unknown keys (checker L184, L57–63), so put both archive digests in `notes`
(`archive libstrata_kernels_cpu.a sha256=<hex>`, `archive libstrata_quant.a
sha256=<hex>`), use `binary_sha256` for the measurement executable and pin the bundle
(both archives, build log, probe/topology output, manifest) with
`raw_artifact_sha256`; reviewers must recompute those digests — the checker
authenticates none (section 4).

## 7. Timing and outcome semantics

Every executed block/dot/full case records monotonic **wall** clock, positive
warmup durations, ordered nonoverlapping raw start/end timestamps, positive raw
repeat durations, explicit cache preparation and its observation hash. Full scope
is `full-branch-with-quantization-and-synchronization`; CPU-time-only,
precomputed-independent-down and isolated dot timing cannot claim that scope.
Recompute median, nearest-rank p95 (`ceil(0.95*n)-1` in sorted zero-based
values), and population variance from the raw repeat durations. Include both
warm/cold conditions and contention. Timing a batch is not counting accepted
decode tokens.

- `pass`: attempted, exit 0, all elementwise checks, usable ISA and effective
  placement pass; no reason. A contradictory declared pass is malformed JSON.
- `reject`: attempted, exit 0, all required checks/timing/trace/observations
  present, at least one numerical/placement/ISA gate failed; a nonempty factual
  reason. A reason without an observed failure is not a rejection certificate.
- `error`: actual guarded refusal/failed execution, nonzero exit, nonempty
  reason, full actual source/build/feature/model/affinity/first-touch/page/attempt
  observations for **every** derived case. `checks=[]`, `trace=null`,
  `timing=null` because completed execution observations are inapplicable, not
  fabricated. An error reason never waives provenance, build, ISA, tensor,
  placement, model geometry or matrix attempt coverage.

Strict mode meanings are inherited unchanged from v1: `--strict-complete`
requires a coherent, schema-complete report with the full derived matrix
(declared pass, partial, synthetic or incomplete evidence cannot satisfy it);
`--strict-accept` additionally requires every gate to pass. Both strict flags
are mutually exclusive at the argument parser level.

## 8. CLI and JSON contract

The v2 checker is offline, reads only an explicit report path and
`dependencies.lock.json`, never reads the model, never writes anything and never
executes the engine:

```text
python3 scripts/check_cpu_evidence_v2.py                 # template, partial: exit 0
python3 scripts/check_cpu_evidence_v2.py --strict-complete
python3 scripts/check_cpu_evidence_v2.py --strict-accept
python3 scripts/check_cpu_evidence_v2.py --strict-complete --strict-accept  # exit 2
python3 scripts/check_cpu_evidence_v2.py path/to/report.json [mode]
```

Mode/exit matrix (as specified; the empty template exercises the first three
rows and the mutual-exclusion row):

| Invocation | Report state | Exit |
| --- | --- | --- |
| default (`partial`) | any structurally consistent report, including incomplete | `0` if `mode_satisfied`, else `1` |
| `--strict-complete` | coherent schema-complete report with full derived matrix | `0` only then; `1` for incomplete/synthetic/partial/rejected |
| `--strict-accept` | complete **and** every gate accepted | `0` only then; `1` otherwise |
| both strict flags | — | `2` (argument error, no validation) |
| malformed JSON/schema/derived contradiction | — | `1`, `CPU evidence validation failed: <message>` on stderr, no traceback |

stdout is a JSON validation summary with at least: `schema_version`, `protocol`,
`validation_mode`, `mode_satisfied`, `status`, `evidence_kind`,
`schema_complete`, `measured_evidence_complete`,
`declared_cpu_correctness_pass`, `structural_consistency_only`,
`measurements_independently_verified`, `inference_ready`, `cases_present`,
`cases_required`, `outcomes`, `missing_fields`, `missing_cases`, `note`. The
`protocol` field must equal `strata-cpu-cold-expert-v2`; a v1 report is rejected
by v2 and a v2 report is rejected by v1. The `note` field restates the honesty
boundary and is not an acceptance message even when `mode_satisfied` is true.

Exit `0` and a `mode_satisfied` summary mean only that the declared JSON contains
no derived contradiction. They are not evidence authentication: the checker never
reads model payload, so a schema-legal declaration that violates the authoring
rules of section 3 (Q8_0 substituted for the real Q2_0 down role, drifted
representative geometry, synthetic data recorded as a real profile, tensor names
or hashes not taken from the payload, an unaudited layer/format mix) can still
exit `0` and must fail evidence review. `--strict-accept` exit `0` is a
consistency signal, never an authenticity, correctness or completeness
certificate.

## 9. Freeze and review procedure

v1 is **byte-frozen**. The following four files must not change when v2 lands;
the checker's own SHA256 is pinned by the v2 checker, and all four are asserted
by the v2 tests:

| Frozen v1 file | Lines | SHA256 |
| --- | ---: | --- |
| `scripts/check_cpu_evidence.py` | 936 | `d727806ba677563cd4bfa267bebec9d04c67c4b1b048da04f1405678c318826a` |
| `benchmarks/cpu-cold-expert-protocol.md` | 355 | `5a7ca9ac1fda336bcbc2286e0526379753878bcda096fa1b708cd4083514dce9` |
| `tests/test_cpu_evidence.py` | 875 | `704dc4b9fb801f8c2070d4ad07cd07a4d6ce327dc0e32c271d95f17ffd8830dd` |
| `benchmarks/templates/cpu-cold-expert.json` | 136 | `522150850c473692d001ec9e9e2360d8c0cb6c704b320e3e731206af9c7ffeb2` |

v2 changes only the four points in section 2; it changes no field set, tolerance,
matrix axis, ordering rule, outcome definition or CLI mode. The v2 template has
the identical field set and ordering as the v1 template, `evidence_kind =
"partial"`, three real anchored profiles and `cases = []`.

v2 takes effect only after review. Until then, v1 remains the authoritative
protocol, the v2 files are review material, and no report may claim v2 evidence.
If actual per-role formats or geometry differ from the anchors here — another
layer mix, another artifact, a non-Q2_0 down role, a different `n_ff` — extend
the protocol, source anchors and adversarial tests through review; never relabel
real tensors, never accept a 640-column down row under a 256-value-block format,
and never claim completeness from an error reason. The freeze exists so that
evidence recorded before review stays reproducible; it is not a claim that v1 is
sufficient for this model.

"Source anchors and template only" describes this v2 material: no CPU build,
model payload read, host probe, NUMA experiment, timing or benchmark is
implied or authorized by it.
