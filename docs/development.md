# Development management: one Project, central Issues

## Canonical tracking

- [Development Project](https://github.com/users/lvyufeng/projects/11)
- [Central Issues](https://github.com/lvyufeng/strata-opentpu/issues)
- [Design roadmap](roadmap.md) and [three-repository ownership](architecture.md)

GitHub Project + Issues in **strata-opentpu** are the source of truth for live
execution status, dependencies, decisions and evidence across all three repositories.
The Project is owned by the GitHub user and linked to the integration repository;
GitHub does not make it a repository-owned Project.

The Strata and openTPU forks own their implementation branches/commits/PRs, not a
second backlog. Create one central Issue for each task or experiment and link all
relevant fork work to it. Upstream bug reports may have their own upstream Issue;
link that external report rather than mirroring the execution state there.
Local todos and Agent-team boards coordinate the current session only. Architecture,
ADRs and experiment reports remain versioned design/evidence, not another live
status system. If a design document and live task disagree, record the discrepancy
in the Issue and update the document; do not silently change the scope.

## Initial task map

The live [roadmap umbrella](https://github.com/lvyufeng/strata-opentpu/issues/1)
contains native sub-issues; the links below are an entry-point index, not a status
snapshot. Read GitHub for current state and follow-up tasks.

| Stage | Central tasks |
| --- | --- |
| M1 | [Source scaffold](https://github.com/lvyufeng/strata-opentpu/issues/2), [development management](https://github.com/lvyufeng/strata-opentpu/issues/3) |
| M2 | [Operator/format audit](https://github.com/lvyufeng/strata-opentpu/issues/4), [CPU/NUMA baseline](https://github.com/lvyufeng/strata-opentpu/issues/5), [memory/PCIe budget](https://github.com/lvyufeng/strata-opentpu/issues/6) |
| M3 | [Device ABI](https://github.com/lvyufeng/strata-opentpu/issues/7), [IQ primitives](https://github.com/lvyufeng/strata-opentpu/issues/8), [Strata backend](https://github.com/lvyufeng/strata-opentpu/issues/9) |
| M4 | [Build/volatile-program profile](https://github.com/lvyufeng/strata-opentpu/issues/10), [real hardware smoke](https://github.com/lvyufeng/strata-opentpu/issues/11) |
| M5 | [Ordinary-decode correctness](https://github.com/lvyufeng/strata-opentpu/issues/12), [MTP validation](https://github.com/lvyufeng/strata-opentpu/issues/13), [wall-decode experiment](https://github.com/lvyufeng/strata-opentpu/issues/14) |

Independent M2 preparation has narrower native child Issues:
[offline resource-budget checker](https://github.com/lvyufeng/strata-opentpu/issues/15)
under the memory/PCIe task and
[CPU evidence protocol/validator](https://github.com/lvyufeng/strata-opentpu/issues/16)
under the CPU/NUMA baseline. Accepting either preparatory artifact does not accept
its parent's real measurements, supply unknown model/device facts or authorize
future builds, downloads or hardware access. Follow native relations and Project
fields for their live state, not this entry-point index.

Independent M3 source preparation is tracked in
[supplied-packed Q2_0/Q8_0 oracle subtask #17](https://github.com/lvyufeng/strata-opentpu/issues/17)
under the low-bit primitives task. The openTPU fork owns scalar math, independently
generated synthetic vectors and their source-test CI; integration owns the
[receipt/integrity contract](design/packed-oracle.md) and central acceptance.
Only exact reviewed commits and successful public CI qualify at this narrow
synthetic scope. Accepting the preparation does not accept or unblock #8, #9 or
the full-model roadmap; native/RTL/device, actual mixed tensors and full-wall
measurements remain separate gates. The bootstrap source lock is unchanged.

## Project fields and states

| Field | Meaning |
| --- | --- |
| Status | Backlog / Ready / In Progress / Blocked / Awaiting approval / Done |
| Priority | P0 foundations/feasibility; P1 capability/integration; P2 optional/follow-up |
| Component | Integration / Strata / openTPU / Cross-repo; where the change belongs |
| Phase | M1 scaffold; M2 audit/budget; M3 capability; M4 hardware profile; M5 model/performance |
| Assignees | Actual responsible GitHub account, if chosen; never a fabricated Agent identity |

- **Backlog:** recorded scope, not yet selected for execution. The long-term
  umbrella remains open until its actual correctness/performance goals are met.
- **Ready:** selected scope is actionable and hard prerequisites are satisfied.
  This is **not** authorization to install, download, program or use hardware.
- **In Progress:** an actual executor has claimed and is working on the scope.
  Record the Agent/session identity in a comment even when no GitHub assignee is set.
- **Blocked:** a concrete dependency, resource or technical condition prevents
  progress; write the condition, evidence and resumption steps.
- **Awaiting approval:** state the exact installation, model/artifact download,
  device test or programming operation awaiting user permission. Do not interpret
  an unrelated permission as authorization.
- **Done:** the Issue's acceptance criteria and evidence have been verified and
  the Issue is closed. A merged fork PR alone is insufficient.

Use labels `type:epic`, `type:task`, `type:experiment`, `scope:*` and `priority:*`
for repository filtering. Keep priority/scope labels aligned when their Project
fields change. Project fields organize work; Issue text defines acceptance.

## Issue and dependency contract

Use the [development Issue template](../.github/ISSUE_TEMPLATE/development-task.md).
Every task needs scope/repository ownership, hypothesis or problem, deliverables,
acceptance criteria, dependencies, authorization boundaries and an evidence plan.
Check existing Issues first. Do not assign unrealistic completion dates or claim
hardware resources that have not actually been granted.

The root roadmap has native GitHub sub-issues. Hard prerequisites use native
Issue `blocked_by` dependencies **and** readable Issue links in the body. Verify
both after changing them. Completion of a dependency does not itself choose an
executor, authorize an operation or start an Agent. Re-read conditions and move
an eligible task from Blocked to Ready deliberately.

Distinguish a true execution blocker from related design input or a final
integration gate. CPU baselines, operator audit, memory budget and ABI design can
start independently. Native IQ block/dot correctness needs confirmed formats,
not a complete model backend. Device profile/smoke testing precedes full-model
integration. If a blocked large task has independent preparatory work, create a
narrow central sub-task for that work rather than ignoring its hard dependencies.
MTP-off wall measurements need ordinary-decode correctness; MTP-on additionally
needs the MTP state/accepted-token validation. Record this conditional gate in
text, not as an unconditional blocker for MTP-off.

## Agent execution protocol

1. Read the parent roadmap, task body, dependencies, latest comments, Project
   fields and repository Agent instructions. Reconcile stale state first.
2. Claim only an actionable scope. Comment actual executor, branch/worktree,
   hypothesis, exact input/source revisions, resource budget, intended operations,
   authorization already received and required evidence. Set In Progress.
3. Keep fork development out of clean detached bootstrap inputs. For parallel
   agents, split write scopes and explicitly coordinate integration/formatting.
4. Post meaningful progress, decisions and blockers. Check remote state after
   uncertain API results before retrying a create/update; avoid duplicate tasks.
5. At session end, write back commits/PRs, tests, sanitized results, conclusions,
   remaining gaps and next actions. Do not leave the only result in chat/local todo.
6. Close only after the explicit acceptance criteria are met, then synchronize
   Project Done. Partial work remains open or is split into documented follow-ups.

A reproducible experiment can conclude **Reject / no benefit / below target** and
close after its evidence contract is satisfied. That does not satisfy or close
the umbrella >10 tokens/s goal. Administrative completion, source-ready,
device-ready and inference-ready are distinct claims.

## Cross-repository commits and PRs

Use the complete central Issue URL in Strata/openTPU PR descriptions and commit
messages when relevant:

```text
Refs https://github.com/lvyufeng/strata-opentpu/issues/N
```

Do not use an ambiguous `#N` in a fork, or a cross-repo `Closes`/`Fixes` that closes
an integration Issue before its validation. Add the reciprocal PR/commit link to
the central Issue. A single Issue may require PRs in both forks plus an integration
lock/profile change. Review exact commits and the combined tests before updating
[dependencies.lock.json](../dependencies.lock.json). A source lock is not an
inference compatibility certificate.

## Practical GitHub CLI handoff

```sh
gh issue list --repo lvyufeng/strata-opentpu --state open
gh issue view N --repo lvyufeng/strata-opentpu --comments
gh project item-list 11 --owner lvyufeng --format json
# After creating a central Issue, add it to the Project:
gh project item-add 11 --owner lvyufeng \
  --url https://github.com/lvyufeng/strata-opentpu/issues/N
# After prerequisites/permission and a real claim are recorded:
gh project item-edit 11 --owner lvyufeng \
  --url https://github.com/lvyufeng/strata-opentpu/issues/N \
  --field Status --value 'In Progress'
```

Creating an Issue does not guarantee it was added to the Project. Check membership
and fields. No auto-add workflow, PAT/secret, automatic agent execution or resource
reservation was installed by this setup. Do not treat default GitHub workflow
behavior as a tested acceptance gate; verify Issue and Project state explicitly.

## Evidence and safety

Reuse Strata's hot-resident experts, CPU cold experts, adaptive cache and MTP;
IQ2_XS is not existing openTPU FP4, and CPU expert kernels alone are not a complete
CPU engine. Preserve full top-10 and expert skip=0. Report prefill/TTFT and wall
decode separately, including host work, transfers and synchronization. MTP off/on
are separate; only accepted output counts for on. Never add CPU and FPGA rates.
See the [benchmark protocol](../benchmarks/README.md).

Issue/Project creation and ordinary source development do not authorize package
installation, model downloads, vendor scripts, device use or programming. Obtain
operation-specific authorization; volatile FPGA loading is distinct from
persistent flash, which remains unsupported. Do not alter host networking or
reset shared drivers as a workaround. Keep public Issues, comments and Project
fields free of credentials, private host addresses/paths, unsanitized logs, weights,
vendor IP/licenses and unreviewed bitstreams. Publish sanitized evidence and keep
private raw artifacts only in ignored local storage.
