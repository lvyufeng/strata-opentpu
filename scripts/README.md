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

## Hardware/network-free tests

```bash
python3 -m unittest discover -s tests -v
```

Tests generate temporary lock/profile data and local Git fixtures; mocked clones
never contact upstream. These checks validate orchestration safety, not FPGA or
model correctness, nor compatibility of the two source revisions.
