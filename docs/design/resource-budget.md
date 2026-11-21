# Offline residency and full-wall budget declarations

**Status: analytical planning only.** The [checker](../../scripts/resource_budget.py)
reads one local JSON document, validates supplied arithmetic and declarations, and
prints a JSON report. It does not inspect model artifacts, authenticate evidence,
probe NUMA or PCIe, allocate/pin pages, execute Strata/openTPU, program a board,
fetch URLs or run commands found in the input. A `fit` result is not device
readiness, inference correctness, or a throughput measurement.

This is the source-only preparation for [central Issue #15](https://github.com/lvyufeng/strata-opentpu/issues/15),
under [the memory/PCIe/NUMA budget parent #6](https://github.com/lvyufeng/strata-opentpu/issues/6).
The long-term [>10 accepted-output tok/s objective #1](https://github.com/lvyufeng/strata-opentpu/issues/1)
remains separate and unachieved.
Actual residency, numerical correctness and complete decode wall throughput remain
unverified. Strata owns routing, hot/cold expert selection, adaptive caching and
MTP accepted-state semantics; this repository must not duplicate or replace them.
Preserve full top10 plus the shared expert and `expert_skip=0`.

## 1. Run without hardware or external dependencies

The checker uses only the Python standard library. The
[conservative reference](../../configs/budgets/kintex7-qwen38-reference.json) is the
cwd-independent default; it intentionally has unknown inputs and is not a usable
model/host profile.

```sh
python3 scripts/resource_budget.py
python3 scripts/resource_budget.py --strict
python3 scripts/resource_budget.py /absolute/path/to/local-budget.json --strict
python3 -m unittest discover -s tests -p 'test_resource_budget.py' -v
```

Keep private overrides/raw reports in ignored local or results directories. An
input pathname is the only file read. Provenance artifact names, URLs, notes and
evidence details inside the document are inert data, never follow-up operations.

| Result | Normal exit | `--strict` exit |
| --- | --- | --- |
| Complete, declared fitting inventory and wall inputs | 0 | 0 |
| Valid but unknown/incomplete inputs | 0 | 1 |
| Known bank/node/global capacity or pinned-quota excess | 0 | 1 |
| Invalid schema/JSON, arithmetic overflow, unreadable input | 2 | 2 |

Valid input always produces a JSON report, including on strict exit 1. Invalid
input produces an error on stderr and no report on stdout. Duplicate JSON keys,
nonfinite JSON constants, booleans used as numbers, negative sizes/times and
unsupported enum values are rejected, not coerced or silently treated as zero.

## 2. Schema v1 and unknown policy

The [reference](../../configs/budgets/kintex7-qwen38-reference.json) and
[synthetic test fixtures](../../tests/test_resource_budget.py) show the complete
structure. Required top-level sections are:

- `schema_version: 1`, nonempty `id`, and `input_kind` of `reference`, `synthetic`
  or `declared`. These labels are declarations, not authenticated provenance.
- `model`: positive integer `layers`, `context_tokens`, `sessions`; exactly
  `routed_top_k: 10`, `expert_skip: 0`, `shared_expert: true`.
- `memory`: `device`, `host`, `coverage`, `tensors`, `allocations`.
- `wall`: `mode`, `coverage`, `numa`, `link`, `stages`, `accepted_tokens`, and
  `count_basis: "accepted_output_tokens"`; MTP-on also has `accepted_accounting`.

Unknown scalar values are `null` or missing where the reader permits it. They
appear in `unknown_inputs`; they are never assumed to be zero. Malformed
structural objects/lists or invalid declared values are errors rather than unknown
inputs. IDs and coverage reasons must be nonempty strings. Numeric bytes/counts
are exact unsigned 64-bit integers; sums, products and alignment are checked for
overflow. Times and payload rates must be finite numeric values; payload rates are
strictly positive when supplied.

Every memory, global wall and per-stage coverage category has an object:

```json
{"status": "unknown", "reason": "Actual allocation or full-stage timing not supplied."}
```

`status` is `complete`, `absent` or `unknown`. Every explicit status needs a reason.
Missing categories become unknown; unknown category names are rejected.
`complete` memory coverage requires a corresponding allocation, and complete
stage coverage requires a covering operation. `absent` forbids allocations or
operations in that category. Global complete wall categories require mapped stage
operations; global absent categories cannot hide non-absent stage roles. These
checks enforce declaration consistency, not the truth of an absence or duration.

Unknown model layers/context/session count makes the memory inventory unknown,
even if all supplied byte counts look fitting. Unknown or over-capacity residency
also suppresses the total wall estimate and accepted-token rate. An independently
known capacity excess is retained even when other inputs are unknown.

## 3. Count all simultaneously resident memory

All allocations are conservatively simultaneous. There is no lifetime scheduler,
overlap reuse, automatic interleaving, expert-size inference, eviction simulation
or allocator probe. Supply actual peak inventory for the chosen context/session
and MTP/cache policy.

### Resources and placement

`memory.device` contains positive/unknown `capacity_bytes` and a nonempty `banks`
list. Each bank has a unique `id`, `capacity_bytes`, `reserve_bytes` and
`fragmentation_bytes`. Device totals sum bank usage including each bank's reserve
and fragmentation; each bank and the aggregate capacity are separate gates.

`memory.host` contains global `capacity_bytes`, `reserve_bytes`,
`fragmentation_bytes`, `pinned_quota_bytes`, and nonempty `numa_nodes`. Each node
has its own unique `id`, capacity, reserve, fragmentation and pinned quota.
Global host reserve/fragmentation are additional to the per-node values; avoid
counting the same reserve twice in supplied numbers. Node/global total capacities
and node/global pinned quotas are independently enforced. Nominal 1 TiB RAM is not
permission for unlimited pinning.

An allocation has a unique `id`, one category, `bytes`, boolean/unknown `pinned`,
`tensor_ids` and explicit `placements`. Each placement gives `resource`, `bytes`
and a positive power-of-two `alignment_bytes`. For example, 65 bytes at alignment
64 consume 128 bytes. Placement raw bytes must sum to allocation raw bytes when
known. Every device bank split is explicit and independently aligned. A host
allocation occupies exactly one NUMA node. Host/device or cross-NUMA physical
copies require separate allocation IDs; they are not aliases counted once. Device
allocations cannot claim host pinning. Pinned host usage includes alignment.

All memory coverage categories are:

- Weights: `fixed_weights`, `hot_experts`, `cold_experts`, `shared_experts`,
  `ple_ngram_weights`, `draft_weights`.
- State/buffers: `fixed_buffers`, `kv`, `qsa_indexer`, `deltanet_recurrent`,
  `deltanet_conv`, `ple_ngram_state`, `residual_draft`, `sampler_history`,
  `cache_metadata`, `workspace`, `dma_staging`, `conversion_workspace`.

Separate accepted and speculative state, retained host sources, cache residency,
DMA staging, row padding, external scales, runtime overhead and conversion
workspace as applicable. Do not omit QSA/DeltaNet/PLE or MTP state because the
initial placement experiment targets only experts.

### Mixed packed tensor metadata, not aggregate GGUF division

A weight allocation uses `tensor_storage: "resident"` or `"packed_source"` and
references actual tensor IDs. Empty tensor references leave weight inventory
unknown. Declared tensor storage sums must equal the allocation's raw bytes;
alignment is charged separately by placements. Every tensor must be used in a
resident allocation. Duplicate tensor IDs, duplicate source tensor/selection
metadata, dangling references and unused metadata are rejected. Separate physical
copies may reference the same declared tensor and are charged by their allocations.

A tensor record supplies:

- `source`: `kind` is `tensor_metadata` or `synthetic`; `artifact`, `revision`,
  64-hex `sha256`, `tensor`, optionally nonempty `selection` for a selected slice.
  The checker validates these declarations but never opens or hashes the artifact.
- `packed`: `dtype`, `layout`, positive `shape`, `block_values`, `block_bytes`,
  `row_stride_bytes`, `data_bytes`, `external_scale_bytes`. `shape[0]` is the
  contiguous row dimension; remaining dimensions multiply the row count. Rows must
  contain integral blocks, stride must include all blocks, and stride times row
  count must equal packed data bytes when supplied.
- `resident`: `status` of `native`, `converted` or `unknown`, `dtype`, `layout`,
  `data_bytes`, `scale_bytes`, `overhead_bytes`, `source_retained`, and a nonempty
  conversion `evidence` string when converted.

Native storage must preserve declared packed dtype/layout/data/external scales.
Its runtime overhead is still counted. A conversion must explicitly declare its
new resident storage and conversion workspace allocation (zero only if explicitly
justified). If `source_retained: true`, a separate placed and charged
`packed_source` allocation is required; it cannot be merged into the converted
resident copy. A declared conversion is not proof of numerical equivalence or an
implemented FPGA decoder.

Never divide aggregate GGUF/shard bytes by 512 to manufacture expert sizes. Mixed
tensor types, non-expert matrices, expert selection and strides must come from
real artifact-specific metadata. IQ2_XS is nonlinear integer-codebook packing, not
openTPU FP4. Pinned Strata definitions document IQ2_XS at 74 bytes/256 values,
Q2_0 at 18 bytes/64 values and IQ3_S at 110 bytes/256 values, but these block sizes
alone do not supply actual tensor geometry or prove FPGA support:
[IQ2_XS definition](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/strata-quant-common.h#L379-L385),
[Q2_0 layout](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/kernels/cpu/expert_layout.hpp#L84-L102),
[IQ3_S definition](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/strata-quant-common.h#L403-L412).
See the [source audit](../audits/strata-execution.md) for format and state limitations.

## 4. Full-wall stages and restricted overlap

Global wall coverage categories are `non_expert`, `cache_misses`, `transfers`,
`numa_traffic`, `sync_merge`, `sampling`, `fallback`, `mtp_verify_commit`. Supply all
work in the chosen decode window, not device kernels alone. Layer stage `repeat`
values must sum to `model.layers`; otherwise no full-wall rate is reported.
Include at least one non-expert stage and an MTP stage when MTP is enabled.
Separate genuinely different layer/state paths rather than assigning one measured
kernel time to all 48 reference layers.

Each stage has a unique `id`, `scope`, positive/unknown `repeat`, `coverage`,
`overlap`, and four operation lists in this order:

```text
before -> [cpu expert chain | fpga expert chain] -> after
```

`before` and `after` are serial. Each expert list is internally serial. `serial`
overlap mode sums both expert chains. `independent_experts` takes their maximum
only for a `layer` stage with `supported: true` and structured independence
evidence. Unsupported/unknown evidence suppresses the estimate; it does not
assume perfect overlap. The checker cannot authenticate actual independence.

For an independent layer, the declaration arithmetic is:

```text
T_layer = sum(before) + max(sum(cpu), sum(fpga)) + sum(after)
T_window = sum(T_stage * repeat)
estimated accepted output tokens/s = accepted_tokens * 1000 / T_window_ms
```

Do not add CPU tokens/s to FPGA tokens/s. DMA in a chain is charged there, not
removed by applying `max` to unrelated rates. Cache misses, fixed/shared work,
QSA/DeltaNet/PLE updates, residual/state work, fences, merge, sampling, fallback,
NUMA traffic, MTP verification and commit/rollback belong in the full path.

### Roles, dependencies and transfer accounting

Each operation has a globally unique `id`, `kind`, nonempty distinct `covers`
roles, and distinct `depends_on` IDs. Dependencies must name earlier operations
in the same stage. A CPU-to-FPGA cross-expert-chain dependency is rejected even in
serial mode: move work that consumes another branch's result into `after`. For
example, H2D of a CPU miss result depends on the CPU producer and is charged
serially after the expert chains, not hidden as an independent FPGA input upload.

Supported operation kinds are `cpu`, `fpga`, `host`, `sync`, `merge`, `h2d`, `d2h`.
CPU cold-expert and FPGA hot-expert computation must have the corresponding
compute kind and role. Neither compute can be laundered into a transfer/sync tag.
Expert chains may contain only the applicable expert role and related transfers,
cache misses or NUMA traffic. Fixed/shared/routing/activation/state and
synchronization/merge stay in serial lists. At least one of cold/hot expert work
must be non-absent; full top10 cannot omit both.

Stage coverage roles are:

| Scope | Roles (mandatory roles in bold) |
| --- | --- |
| `layer` | **fixed**, **shared_expert**, **routing**, **activation**, **state**, cold_expert, hot_expert, transfers, cache_misses, numa_traffic, **sync_merge** |
| `non_expert` | **embedding**, **head**, **sampling**, **state**, numa_traffic, fallback |
| `mtp` | **draft**, **verification**, **commit_rollback**, **state**, transfers, numa_traffic, **sync_merge** |

Mandatory roles cannot be `absent`; `unknown` is allowed but blocks a complete
estimate. One operation may cover several roles when its supplied duration is the
aggregate for that work; the duration is charged once. Labels do not independently
prove that the aggregate actually includes every role.

A transfer must cover `transfers` and supplies unsigned `bytes` and `submissions`.
Nonzero bytes require nonzero submissions. No free-form duration can substitute
for transfer accounting:

```text
T_transfer_ms = bytes / directional_payload_bytes_per_second * 1000
                + submissions * directional_submission_milliseconds
```

`wall.link` separately declares `endpoint_capability` and `negotiated` generation
and lanes, and `h2d`/`d2h` payload rates and submission costs. Capability or
historical bandwidth never fills unknown negotiated/rate fields. H2D and D2H can
have different payload rates and overhead. Even zero transfer bytes do not make
unknown link/topology declarations complete.

`wall.numa` requires declared PCIe-local node, transfer-worker node, distinct
CPU-worker nodes and `page_placement_declared: true`. All node IDs must appear in
the memory inventory. These declarations do not set or verify OS affinity/page
placement. Cross-socket traffic and migrations must be included in timing rather
than inferred from a two-socket core count.

### Evidence and accepted MTP outputs

Independence and MTP accepted accounting evidence use:

```json
{"kind": "synthetic", "detail": "A fictional fixture, not a hardware trace."}
```

Allowed kinds are `synthetic`, `analytical`, `measured`, `source`; `detail` must be
nonempty. A bare string, unsupported kind or empty detail is invalid. Even the
`measured` label is input data, not authenticated evidence; the tool always reports
`input_provenance_verified: false` and `estimate_is_measurement: false`.

`mode: "mtp_off"` requires global `mtp_verify_commit` absent and forbids MTP
stages. `mtp_on` cannot omit verification/commit and requires an MTP stage plus
`accepted_accounting` status of `declared` or `unknown` with structured evidence.
Only accepted output tokens count. Zero or unknown accepted count suppresses the
rate. Draft/rejected tokens are not output throughput. All draft, verification,
state, commit/rollback, transfer and join work must be included; this checker does
not validate Strata's accepted frontier or execute rollback.

Pinned Strata ordering shows GPU expert enqueue before CPU expert work, followed
by CPU-miss upload/combine and final synchronization. It is source evidence for a
restricted candidate overlap model, not measured FPGA independence:
[session execution](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/session.cpp#L723-L807).
The [device ABI design](device-abi.md) and [Strata audit](../audits/strata-execution.md)
describe broader state, drain, rollback and ownership obligations.

## 5. Read the report conservatively

The report always sets `analytical_only: true`, `input_provenance_verified: false`,
`device_ready: false`, `inference_ready: false`; the wall report always sets
`estimate_is_measurement: false`. No input boolean can turn this into readiness.

- Top-level `status` is `over_capacity`, `unknown` or `fit`. Known capacity excess
  takes precedence over unknown inputs. `strict_pass` requires complete inputs
  and no excess.
- `memory.status` aggregates all bank/node/global capacity and pinning gates.
  Each constraint reports `known_used_lower_bound_bytes`, and `used_bytes` only
  when usage is fully declared. A lower bound below capacity is not proof of fit.
  `headroom_upper_bound_bytes` may be informative with incomplete usage, but it is
  not allocatable or certified headroom.
- `wall.status` is `estimated` only for a complete fitting inventory, complete
  model/context, stages, topology/link, evidence declarations and accepted-count
  accounting. Otherwise total `milliseconds` and
  `estimated_accepted_tokens_per_second` are `null`.
- `known_lower_bound_milliseconds` and stage details retain known arithmetic even
  when the total is unavailable. They are neither measured traces nor upper bounds
  on achievable performance. Do not invert a partial lower bound into a rate.

The [tests](../../tests/test_resource_budget.py) include fictional parallel/serial
windows, directional DMA submission cost, CPU-result-dependent transfer placement,
per-bank and NUMA/pinned overflow, unknown context/residency, mixed packed/native/
converted tensors and retained copies, malformed declarations and CLI behavior.
They validate offline arithmetic and fail-closed contracts only, not model
correctness, actual fit, executable kernels or the >10 tok/s objective.

## 6. Reference facts and unresolved inputs

The [board reference](../../configs/boards/kintex7-reference.json) records
`xc7k480tffg1156-2`. Pinned openTPU source describes two 2 GiB DDR banks, channel
bases and DMA chunking:
[board constants](https://github.com/lvyufeng/openTPU/blob/d84db55712aba0d40811d7762cb052f05d55b7e5/opentpu/host/board.py#L4-L19),
[channel placement](https://github.com/lvyufeng/openTPU/blob/d84db55712aba0d40811d7762cb052f05d55b7e5/opentpu/host/board.py#L64-L75).
Source constants are not board identification or available allocator capacity.
Usable reserve, fragmentation and actual tensor-to-bank mapping remain unknown.

The host reference is dual E5-2696v4, 44 physical/88 logical threads, AVX2/FMA and
1 TiB total RAM. Per-NUMA capacities, pin quotas, PCIe root/socket, affinity and
page placement are unknown. Gen2x8 is endpoint capability only. Historical
Gen2x4/H2D approximately 1.39 GB/s is not a current negotiated link or sustained
payload measurement and is intentionally not used as a rate.

The [model descriptor](../../configs/models/qwen38-flash-next-iq2-xs-reference.json)
is a source-only IQ2_XS pack reference, not an executable download profile. The
[Strata source audit](../audits/strata-execution.md#L24-L33) records default geometry
of 48 layers, 512 experts, top10, hidden size 2560 and expert FFN size 640. These do
not determine mixed tensor residency. The budget
reference leaves artifact-specific tensors, allocations, context, rates and all
actual stage costs unknown. Its source pins match the source-only
[dependency lock](../../dependencies.lock.json); compatibility remains
`not-validated-for-inference`. No model, bitstream, host build or device operation
was performed by this checker.

To advance [budget parent #6](https://github.com/lvyufeng/strata-opentpu/issues/6),
supply reviewed artifact-specific metadata, full context/session/cache/MTP peak
allocations, actual NUMA/pinned quotas and current negotiated-link/H2D/D2H costs.
The separate [throughput objective #1](https://github.com/lvyufeng/strata-opentpu/issues/1)
also requires format/numerical/state correctness on authorized implementations and
complete accepted-output decode wall traces, including CPU cold work, FPGA hot
work, misses, joins, state, sampling, fallback and MTP commit/rollback. Keep
prefill/TTFT/decode and MTP-off/on separate. Those operations require their own
authorization and evidence. This offline tool neither performs them nor closes
either acceptance.
