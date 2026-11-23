# Integration CLI

Python 3.11+ and Git are the only bootstrap requirements. No package installation
is performed by these entry points. Commands emit JSON on success and a diagnostic
on stderr with exit code 2 when refused. The default workspace is the repository
containing this script; global `--workspace DIR` selects another existing root.

## Implemented commands

```bash
python3 scripts/sop.py doctor
python3 scripts/sop.py doctor --profile configs/profiles/bootstrap.json
python3 scripts/sop.py bootstrap
# Only after explicit authorization for dependency source fetching:
python3 scripts/sop.py bootstrap --apply
```

`doctor` is read-only. All commands validate
`configs/profiles/bootstrap.json` by default; `--profile RELATIVE_PATH` selects
another workspace-relative bootstrap profile. It reports Python/Git and basic host inventory. Linux
additionally exposes CPU flags, total memory, NUMA CPU lists and PCI identifiers
where accessible. It **does not** verify FPGA identity, bitstream, DMA, board
compatibility or inference readiness. Inventory may identify the host; do not
publish raw output without review.

`bootstrap` uses the profile's `dependency_lock`, falling back to the root
`dependencies.lock.json` when omitted. `--lock RELATIVE_PATH` explicitly overrides
that selection with another workspace-relative lock. A plan never clones or checks out code,
although existing dependencies are checked for conflicts. `--apply` is explicit
network opt-in for these two locked public GitHub repositories only:

- Clone absent dependencies using `--no-checkout --no-recurse-submodules`.
- Check out the exact locked commit detached and verify origin, HEAD and cleanliness.
- Verify all existing checkouts before creating any missing dependency.
- Reuse an exact, clean checkout without network or checkout mutation.
- Refuse wrong paths, URLs, revisions, symlinks, dirty or non-detached checkouts,
  and unexpected origins.
- Never reset, clean, overwrite, move or delete an existing checkout. A failed new
  clone/checkout remains for inspection; no automatic cleanup is performed.

Global/system Git configuration is not inherited; hooks, filesystem monitor,
credential helpers and non-HTTPS transports are disabled. Fetching private
repositories, proxies relying on Git config and automatic credentials are not
supported by this public-source bootstrap. No submodule, build, upstream script,
model download, install or hardware operation runs automatically. Dependency
checkout paths are fixed to `third_party/Strata` and `third_party/openTPU`.

Concurrent bootstrap operations are not supported. Symlinks are checked before
operations, but this is not an isolation sandbox against another process changing
filesystem paths while setup runs. Use a trusted, non-concurrently-mutated workspace.

## Planned commands

```bash
python3 scripts/sop.py build
python3 scripts/sop.py program
python3 scripts/sop.py smoke-test
python3 scripts/sop.py launch
python3 scripts/sop.py benchmark
```

These commands **only report an unimplemented plan**. Adding `--apply` fails
nonzero without side effects, including if a profile tries to enable them.
`program` does not configure the FPGA or write persistent flash. Arbitrary
commands from profiles are never executed. An optional `--profile` must use the
supported bootstrap schema; changing capability booleans cannot enable features.

## Static audit citation checks

The [audit manifest](../data/strata-execution-audit.json) is a machine-readable
source index for the [execution audit](../docs/audits/strata-execution.md).
The separate [citation validator](check_source_anchors.py) supports:

```sh
python3 scripts/check_source_anchors.py
python3 scripts/check_source_anchors.py --check-sources
```

Default mode checks source URLs/revisions against the lock and validates entry
IDs, citation paths and line ranges structurally. It needs no dependency sources
and never fetches anything. Opt-in source checking requires existing clean,
exact-HEAD local source checkouts and additionally checks file bounds and any
provided excerpt. It does not run source code, prove a citation's interpretation,
inspect model tensors or certify inference. Success prints JSON; validation
failure exits 1. This is independent of the bootstrap CLI's exit-code contract.

## Offline resource budgets

The [budget checker](resource_budget.py) reads the
[reference budget](../configs/budgets/kintex7-qwen38-reference.json) by default,
or one explicit JSON input:

```sh
python3 scripts/resource_budget.py
python3 scripts/resource_budget.py local/budget.json --strict
```

It checks per-bank placement, host/global/NUMA and pinned quotas, complete
allocation inventories, packed tensor metadata and ordered wall-time dependencies.
It does not use total GGUF size as expert bytes or add CPU/FPGA token rates.
Unknown reserves, state, metadata or timings stay unknown, not zero or a fit.
Default mode reports incomplete or over-capacity budgets with exit 0;
`--strict` exits 1 unless the required budget is complete and within all quotas.
Malformed JSON/schema or I/O errors exit 2. Reports are analytical only, not device
or inference readiness. See the [budget design](../docs/design/resource-budget.md).

## Partial-tensor expert residency budgets (v2)

The [v2 checker](resource_budget_v2.py) reads exactly one positional JSON input and
validates declared per-expert selection over the target pack's 512-expert
sub-slice, plus the same bank/node/pinned and ordered wall-time arithmetic:

```sh
python3 scripts/resource_budget_v2.py local/budget-v2.json
python3 scripts/resource_budget_v2.py local/budget-v2.json --strict
```

Each expert tensor record declares `layer`, `role` and a `selection` of a contiguous
range, explicit indices or explicit unknown over the declared expert axis; resident
bytes are charged only for the selected sub-slice. Overlap, out-of-range,
contract-mismatch, duplicate/unknown-field and inconsistent residency declarations
exit 2 with no report. A partial or unknown selection keeps affected totals `null`,
appears in `unknown_inputs` and can never be a fit; `--strict` exits 1 unless
`strict_pass` is true. The checker uses only the standard library, parses no GGUF,
opens no model artifact and runs no external command. Reports are declaration
arithmetic, not placement, device or inference evidence. See the
[v2 design](../docs/design/resource-budget-v2.md) and the pinned
[synthetic fixture matrix](../tests/test_resource_budget_v2.py) plus
[acceptance record](../configs/contracts/strata-resource-budget-v2.json). Schema v1
files stay frozen.

## Offline CPU evidence

The [CPU evidence validator](check_cpu_evidence.py) reads the
[partial template](../benchmarks/templates/cpu-cold-expert.json) by default:

```sh
python3 scripts/check_cpu_evidence.py
python3 scripts/check_cpu_evidence.py local/cpu-evidence.json --strict-complete
python3 scripts/check_cpu_evidence.py local/cpu-evidence.json --strict-accept
```

Default mode reports missing evidence without claiming completeness. Malformed or
contradictory input fails nonzero. `--strict-complete` requires structurally complete,
explicitly measured evidence; a documented Reject/error experiment can qualify
without claiming numerical acceptance. `--strict-accept` additionally requires every
required numerical, ISA and observed placement gate to pass. Neither strict mode
accepts synthetic evidence.
Read the [protocol](../benchmarks/cpu-cold-expert-protocol.md) for source/role/ISA,
thread/NUMA, elementwise-error, coverage and repeated full-branch wall-time rules.
The checker cannot independently verify supplied measurements or turn a branch
experiment into a full-model CPU decode result.

Both tools use only the standard library and read input data; they do not execute
commands, paths or URLs embedded in reports, fetch sources/models, build kernels
or access target hosts/devices. Default inputs resolve relative to the repository,
not the caller's working directory. Passing them is not operation authorization
and does not implement the planned bootstrap commands.

## Packed-oracle receipt and opt-in source verification

The [packed-oracle checker](check_packed_oracle.py) reads the integration
[receipt](../configs/contracts/opentpu-packed-oracle.json), not a copy of the math:

```sh
python3 scripts/check_packed_oracle.py
python3 scripts/check_packed_oracle.py --strict-receipt
python3 scripts/check_packed_oracle.py --fixture /path/to/supplied-fixture.json
python3 scripts/check_packed_oracle.py --strict-receipt --oracle-root /path/to/reviewed-opentpu
```

Default and strict-receipt modes check recorded schema/source/acceptance
consistency only. The fixture mode also checks provenance, counts, lengths and
pinned digests without doing decode/dot. None certifies review/CI assertions as
true. The integration CI uses strict-receipt only and never obtains fork sources.

The **explicit** oracle-root opt-in requires a complete receipt and an existing
nonsymlink independent checkout at the exact reviewed commit, clean including
ignored/untracked files, correct fork origin/base ancestry, full MIT notices and
fixture digests. It compares the index and every ordinary tracked file's actual
Git blob hash/mode to the recorded commit tree, rejects hidden/sparse/unmerged
index flags and symlinks/gitlinks, then runs only the fixed stdlib verification
module with the current interpreter (`-B -E -S`). It repeats tree/index/byte/extra
file checks and checks HEAD after execution; Git status/filters are not relied on
for content integrity. Effective partial-clone/promisor configuration is rejected
before tree-object lookup, and Git lazy fetching is disabled. This is
trusted-source execution, **not** an OS/hardware/resource sandbox;
environment cleanup and timeouts cannot confine arbitrary malicious code and
captured-output limits apply after allocation. Use trusted reviewed sources and
an authorized operation. No fetching, installation, build or device procedure is
provided.

Exit 0 means the selected checks passed (default may report incomplete); strict
incomplete receipts exit 1; malformed/unsafe/failed-source checks exit 2 with
stable JSON errors. All modes retain device/inference readiness false. Read the
[design and frozen evidence scope](../docs/design/packed-oracle.md): synthetic
Q2_0/Q8_0 packed tests are not IQ, quantizer, native/SIMD, model, RTL, DMA or
performance evidence and do not accept the parent primitives/backend tasks.

## JSON-only GGUF header inventory

The [inventory validator](check_gguf_inventory.py) checks the
[receipt schema](../configs/contracts/gguf-header-inventory.schema.json) and
cross-field semantics without parsing GGUF or executing Strata:

```sh
python3 scripts/check_gguf_inventory.py
python3 scripts/check_gguf_inventory.py --receipt local/inventory.json
python3 scripts/check_gguf_inventory.py --receipt local/inventory.json --strict-complete
```

Default input is the repository-anchored
[synthetic receipt](../tests/fixtures/gguf-header-inventory.synthetic.json).
Only the supplied JSON is read; names, paths, URLs and SHA declarations never open
referenced artifacts or authorize operations. The bounded strict stdlib loader
rejects duplicates, nonfinite values, unsafe files and oversized/deep input.
Validation requires exact fields/types/nulls, generated shard order, raw-shape
row-block arithmetic, safe offsets/extents and coherent split/count totals.
Unknown layouts stay null, not zero or backend-supported. Header SHA covers only
consumed logical header, not padding/payload or authenticated artifact integrity.

A valid incomplete split exits 3 with a summary. `--strict-complete` also exits 3
for unknown split status or any unknown storage. Malformed/I/O/schema/semantic
failures exit 2 with empty stdout and a diagnostic on stderr. Valid unknown or
complete inventories otherwise exit 0; even strict success retains all
provenance/readiness flags false and `source_identity_verified: false`.

The opt-in source exporter is owned by Strata, not this CLI. See the
[design/limits/legacy guarantees](../docs/design/gguf-inventory.md) and
[separate source/CI record](../configs/contracts/strata-gguf-inventory.json).
There is no model acquisition, full-file/tensor hash, expert slicing, native
helper, device operation or inferred model completeness. This preparation does
not accept or unblock the parent budget/backend/full-decode gates.

## Reference-only 4 GiB residency arithmetic

The [residency calculator](residency_reference.py) turns a validated mixed-GGUF
**header** inventory receipt into declared byte arithmetic against the reference
4 GiB device envelope:

```sh
python3 scripts/residency_reference.py --receipt local/inventory.v2.json
python3 scripts/residency_reference.py --receipt local/inventory.v2.json --format json
python3 scripts/residency_reference.py --receipt local/inventory.v2.json --contexts 4096,32768
```

The input is a positional-free `--receipt` path; the default model geometry is the
pinned 48-layer / 512-expert reference config and `--contexts` defaults to
`4096,32768`. Only the supplied JSON is read: the CLI does not parse GGUF, open
model payload, probe PCIe/NUMA, allocate memory, run Strata/openTPU or execute
anything named in the receipt. Missing/unknown tensor storage, duplicate names,
inconsistent totals, wrong layer/expert geometry and a non-last expert axis are
rejected with exit 2, empty stdout and a diagnostic on stderr. Success exits 0 and
reports declared bytes only.

Five scenarios are computed: all persistent weights resident with or without the
fittable global tensors, all persistent weights streamed, and a hypothetical
cheapest-first layer pack. Hot space is a byte ceiling divided by the per-expert
footprint, not a cache hit rate or a placement; the oversize per-layer embedding
table is excluded from fit rows and reported separately. Every report repeats the
`unknown` list (usable capacity, bank layout, negotiated link, DMA rates,
workspace, fragmentation, routing trace, device compute, MTP) so no downstream
record inherits a fit claim. KV arithmetic is declared from the pinned model
config, not measured.

See the [design and measurement plan](../docs/design/reference-residency-budget.md).
The CPU projection in that document reuses accepted v2 single-branch medians and is
explicitly not a decode or tokens/s result. Tests:
`python3 -m unittest tests.test_residency_reference -v`.

## Hardware/network-free tests

```bash
python3 -m unittest discover -s tests -v
```

Tests generate temporary lock/profile data and local Git fixtures; mocked clones
never contact upstream. These checks validate orchestration safety, not FPGA or
model correctness, nor compatibility of the two source revisions.
