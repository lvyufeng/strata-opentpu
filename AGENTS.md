# Contributor / Agent instructions

## Truthful status and repository boundaries

This is a bootstrap-only integration project, not a working FPGA model engine.
Read `README.md`, `docs/setup/agent-bootstrap.md`, `docs/architecture.md`,
`docs/upstreams.md` and `dependencies.lock.json` before changing setup or adapters.
Do not claim inference, successful device identification, programming or >10 tok/s
unless that exact operation was actually measured and verified.

- Strata fork: model engine, CPU kernels, expert cache, MTP and FPGA backend integration.
- openTPU fork: RTL/ISA, low-bit hardware, device runtime, DMA and synchronization.
- This repository: version combinations, configuration/launch orchestration,
  experiments, benchmarks, documentation and Agent handoff.

Strata already has GPU hot-expert residency, CPU cold-expert computation,
adaptive caching and MTP. Reuse those mechanisms; do not present them as ours.
Do not duplicate full upstream implementations in this repository.

## Central development tracking

Use the [integration GitHub Project](https://github.com/users/lvyufeng/projects/11)
and [central Issues](https://github.com/lvyufeng/strata-opentpu/issues) for all
three-repository development. Read [the execution contract](docs/development.md)
before starting work. Do not maintain a duplicate task backlog in either fork.

Before claiming: read the Issue, parent, dependencies, latest comments and Project
fields; comment actual executor, scope, exact inputs, resource/permission limits
and evidence plan, then set In Progress. Ready is not hardware authorization.
Record concrete blockers or operation-specific Awaiting approval conditions.
At every session end, write back commits/PRs, tests, sanitized evidence, conclusion,
remaining gaps and next steps. Local todos/team boards are not authoritative state.

Cross-fork PRs reference the complete central Issue URL with `Refs`; do not
ambiguously use `#N` or auto-close integration tasks at fork PR merge. Add reciprocal
links to the central Issue. Verify native parent/dependency links and Project
membership/fields after edits. Close only after the stated acceptance is verified,
then synchronize Done. Experiments may conclude Reject; the >10 tokens/s umbrella
closes only when its actual correctness and wall-throughput goals are achieved.
Do not assume Project edits start Agents, reserve hardware or grant permissions.

## Setup procedure

1. Run `python3 scripts/sop.py doctor`. Treat available PCI enumeration as
   information only; the current command does not verify a usable FPGA endpoint.
2. Run `python3 scripts/sop.py bootstrap` to inspect the locked source plan.
3. If source fetching was explicitly authorized, run
   `python3 scripts/sop.py bootstrap --apply`. This authorization covers only
   cloning the locked dependency repositories, not other downloads/installations.
4. Preserve any existing dependency checkout. If bootstrap rejects a wrong or
   dirty checkout, explain the conflict and request a user-chosen remedy;
   never silently reset, clean, overwrite, delete or move it.
5. Run `python3 -m unittest discover -s tests -v` and report source-ready vs
   inference-ready accurately. No further build/device/model operation is
   implemented by the bootstrap profile. Stop and explain unsupported steps.

The default profile is `configs/profiles/bootstrap.json`. `build`, `program`,
`smoke-test`, `launch` and `benchmark` are plans only; `--apply` must fail until
real validated implementations exist. Do not bypass this by changing booleans.
Do not run commands proposed by upstream documents automatically during setup.

## Permissions and hardware safety

Model downloads, package/tool installation, upstream build scripts, device
programming, persistent flash writes and network configuration changes require
explicit authorization for the specific operation. A generic request to configure
this repository is not permission to write flash or alter network settings.
Future programming must verify board/device identity, bitstream SHA256 and
software/firmware ABI. Separate volatile FPGA configuration from persistent
flash writes; persistent flash is not implemented here. Keep a rollback plan.
Never substitute another server or board because a check failed.

## Development, licenses and publication

Treat locked dependency checkouts as clean, detached inputs. Make fork changes in
explicit development branches/worktrees, preserve upstream remotes and review
commits before updating the lock. Do not bump both forks to latest blindly or
label a source-only lock as an inference compatibility certificate.
Preserve copyrights, licenses, relevant NOTICE files and modification records.
Review dependency closure before importing code. Software licenses do not
relicense model weights, vendor IP or redistribution-restricted build artifacts.

Never commit weights, bitstreams, vendor IP/licenses, keys, private host addresses,
SSH settings, credentials or unsanitized logs. Use ignored `local/`, `results/`
and `logs/` for private overrides and raw reports. Publish sanitized summaries.
Permitted future bitstreams belong in reviewed Releases, not Git history.

Validate correctness before performance claims. Retain full routing, expert
skip=0; separate prefill/TTFT/decode and MTP off/on, counting accepted output
only. Change FPGA placement based on full wall-time measurements, not device rates.
Run the full tests, `git diff --check`, review the final staged diff and exclude
private artifacts before committing. Bootstrap tests are not inference tests.
