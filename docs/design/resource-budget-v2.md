# Partial-tensor expert residency budgets (schema v2)

**Status: analytical planning only.** The
[v2 checker](../../scripts/resource_budget_v2.py) reads one local JSON document,
validates supplied per-expert selection and resource arithmetic, and prints a JSON
report. It does not parse GGUF, open model artifacts, authenticate evidence, probe
NUMA/PCIe, allocate or pin pages, execute Strata/openTPU, program a board, fetch
URLs or run commands found in the input. A `fit` result is not device readiness,
inference correctness, a placement result or a throughput measurement.

This is the source-only preparation for [central Issue #20](https://github.com/lvyufeng/strata-opentpu/issues/20),
a child of [the memory/PCIe/NUMA budget parent #6](https://github.com/lvyufeng/strata-opentpu/issues/6).
The [schema v1 checker](../../scripts/resource_budget.py) and its
[design](resource-budget.md) remain frozen and keep their accepted scope.
Strata owns routing, hot/cold expert selection, adaptive caching and MTP
accepted-state semantics; this repository must not duplicate or replace them.
Preserve full top10 plus the shared expert and `expert_skip=0`.

## 1. Why schema v1 is insufficient

In the target Qwen3.8-Flash-Next GGUF pack an expert tensor is one whole packed
tensor, for example `blk.N.ffn_gate_exps.weight` with shape `[2560, 640, 512]` and
`blk.N.ffn_down_exps.weight` with shape `[640, 2560, 512]`. The 512 experts are a
sub-slice of the **last** dimension, not separate tensors. Schema v1 charges an
allocation against whole packed tensor metadata, so it cannot express "keep 128 of
512 experts resident" and would either over-charge or silently under-charge the
resident bytes. That input is not accepted by v1 and is not retrofitted into it.

## 2. Run without hardware or external dependencies

```sh
python3 scripts/resource_budget_v2.py /absolute/path/to/local-budget-v2.json
python3 scripts/resource_budget_v2.py /absolute/path/to/local-budget-v2.json --strict
python3 -m unittest discover -s tests -p 'test_resource_budget_v2*.py' -v
```

The input is a single positional path; there is no default input, no `--input` or
`--output` option, and no network access. Keep private overrides and raw reports in
ignored local or results directories. Provenance names, URLs and evidence details
inside the document are inert data, never follow-up operations.

| Result | Normal exit | `--strict` exit |
| --- | --- | --- |
| Complete, declared fitting selection and wall inputs | 0 | 0 |
| Valid but unknown or partial-coverage inputs | 0 | 1 |
| Known bank/node/global capacity or pinned-quota excess | 0 | 1 |
| Invalid schema/JSON, overlap, out-of-range, overflow, unreadable input | 2 | 2 |

Valid input always produces a JSON report, including on strict exit 1. Invalid input
produces an error on stderr and no report on stdout. Duplicate JSON keys, nonfinite
JSON constants, booleans used as numbers, negative sizes/counts and unsupported enum
values are rejected, not coerced. `strict_pass` appears only at report top level.

## 3. Selection contract and per-tensor records

The top-level `selection_contract` makes the expert axis explicit:

- `convention: "last_dimension"`, `expert_dimension` (must equal
  `len(shape) - 1`), `expert_count`, and `roles` mapping each of
  `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` to the non-expert shape prefix.
- Every tensor record carries `layer`, `role` and `selection`; block indices are
  data, never parsed out of a `blk.N` tensor name.
- `selection.mode` is one of:
  - `range`: contiguous `start` + `count` over the expert axis;
  - `indices`: explicit sorted unique `indices` + `count`;
  - `unknown`: no `start`/`count`/`indices`, plus a nonempty `reason`.
- Selection is bounded (`MAX_SELECTIONS`, `MAX_SELECTION_ITEMS`,
  `MAX_SELECTIONS_PER_TENSOR`, `MAX_TENSOR_RECORDS`,
  `MAX_TOTAL_SELECTED_ITEMS`, `MAX_EXPERT_COUNT`). Overlapping selections for one
  source tensor, out-of-range ranges/indices, mode/field contradictions and a
  `packed.shape` prefix that disagrees with the role or `expert_count` are errors.
- Resident bytes are arithmetic from the declared packed metadata:
  `prod(shape[1:expert_dimension]) * selected_items * row_stride_bytes`; packed
  `data_bytes` must agree with row-stride/block metadata, block layout must not
  omit blocks, and external (non-embedded) scale storage is rejected.

Each tensor record also declares `resident`:

- `native`: resident layout/dtype must match the packed metadata;
- `converted`: needs concrete `data_bytes`/`scale_bytes`/`overhead_bytes`, a
  `declared_evidence` block and a declared `conversion_workspace` allocation,
  because conversion needs scratch workspace;
- `unknown`: no byte claim; affected resources stay unknown.
- `source_retained: true` additionally requires a second, separately placed and
  charged `packed_source` allocation holding the retained packed rows; the
  converted resident copy cannot count as its own retained source.

## 4. Coverage, unknown policy and wall accounting

`selection_coverage` reports declared/accounted/unaccounted items and
`full_coverage`. Coverage is `complete` only when every record on the declared
expert axis is fully selected. A partial or unknown selection keeps every affected
per-bank/node total `null`, adds explicit `unknown_inputs`, and can never be
reported as a fit; `strict_pass` is false. The same holds for any unknown category
in `memory.coverage`, `wall.coverage` or a stage operation list.

Memory categories, allocation categories, stage scopes and operation kinds are the
same enforced domains as v1 (18 memory/allocation categories, 8 wall categories,
11/6/7 layer/non-expert/MTP stage categories, operations `cpu`, `fpga`, `host`,
`sync`, `merge`, `h2d`, `d2h`). `cold_expert` work must be `cpu`, `hot_expert` work
must be `fpga`, sync/merge are forbidden on the independent compute branches, and
overlap is modelled only for layer expert branches with explicit
`supported`/`evidence`; unverified independence is an unknown input.

Wall time is a declared lower-bound arithmetic model:
`T_layer ~= T_serial + max(T_CPU_experts, T_FPGA_experts) + T_sync_and_merge`.
CPU and FPGA token rates are never added, MTP is counted separately, and only
accepted output tokens are counted. Any missing timing, transfer submission count,
link rate, stage or category leaves the result unknown with a known lower bound
where the declared arithmetic permits it. The report never claims a full-wall
measurement or the >10 accepted-output tok/s objective.

## 5. Fixtures, schema and acceptance record

The v2 contract has its own
[descriptive JSON schema](../../configs/contracts/resource-budget-v2.schema.json)
and [acceptance record](../../configs/contracts/strata-resource-budget-v2.json);
the v1 schema, validator, reference budget, tests and design document are unchanged.
The validator enforces the exact fields/types/categories itself with the standard
library; the schema is documentation, not an executed validator.

Six synthetic fixtures cover the required matrix and are pinned by bytes and SHA256
in the record and in [the contract test](../../tests/test_resource_budget_v2_contract.py):

| Fixture | Expectation |
| --- | --- |
| `full-coverage` | fit, strict pass, converted non-native resident |
| `partial-hot-subset` | 128 of 512 experts, unknown, never fit |
| `unknown-selection` | explicit unknown selection, unknown |
| `real-shape-split` | real shard-1 shapes split 128 hot / 384 cold, fit |
| `overlap-reject` | overlapping selections, exit 2 |
| `range-out-of-bounds-reject` | range past the expert axis, exit 2 |

The real-shape fixture reuses only the reviewed header **shapes**; its sources are
declared synthetic, and its 128/384 split is hypothetical, not an actual Strata
placement or cache state. No packed tensor, GGUF file, artifact hash, router logit
or measurement is read or authenticated. A complete v2 report still needs an
actual hot/cold split plus independently reviewed Strata placement and measured
target evidence before any parent gate can move.