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

## Hardware/network-free tests

```bash
python3 -m unittest discover -s tests -v
```

Tests generate temporary lock/profile data and local Git fixtures; mocked clones
never contact upstream. These checks validate orchestration safety, not FPGA or
model correctness, nor compatibility of the two source revisions.
