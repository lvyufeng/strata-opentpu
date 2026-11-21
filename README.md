# strata-opentpu

Integration and experiments for CPU + FPGA heterogeneous MoE inference, combining
our forks of [Strata](https://github.com/lvyufeng/Strata) and
[openTPU](https://github.com/lvyufeng/openTPU).

**Status: project bootstrap only.** Host inspection and locked dependency source
setup are implemented. No FPGA inference backend, integrated build, programming,
or model launch is implemented here yet. This is an independent, unofficial
integration project; the two implementation repositories are upstream forks.

## Three repositories

| Repository | Responsibility |
| --- | --- |
| [Strata fork](https://github.com/lvyufeng/Strata) | Preserve the model engine, CPU kernels, hot-expert cache and MTP; add FPGA backend interfaces and heterogeneous/NUMA execution changes |
| [openTPU fork](https://github.com/lvyufeng/openTPU) | FPGA RTL/ISA, low-bit computation, device runtime and transport/synchronization capabilities |
| This integration repository | Lock compatible versions, orchestrate setup/build/program/launch, organize experiments, benchmarks, ideas and Agent instructions |

These are intended change boundaries, **not implemented FPGA capabilities**.
Strata already runs resident hot experts on GPU and cold experts on CPU, adapts
its expert cache and supports MTP. We aim to preserve that execution system,
not invent the same architecture again or rewrite its full model engine here.
See the [accepted three-repository decision](docs/adr/0002-three-repository-integration.md).

## Clone and hand off to an Agent

```sh
git clone https://github.com/lvyufeng/strata-opentpu.git
cd strata-opentpu
python3 scripts/sop.py doctor
python3 scripts/sop.py bootstrap          # plan only; no cloning
python3 scripts/sop.py bootstrap --apply  # explicitly fetch the locked forks
python3 -m unittest discover -s tests -v
```

Requirements: Python 3.11+ and Git; no Python packages needed. Source setup fetches
exact revisions into ignored `third_party/Strata` and `third_party/openTPU` without
recursively initializing submodules. It does not install tools, run upstream
build scripts, download models or touch hardware. Existing wrong/dirty checkouts
are rejected rather than reset. Use dedicated branches/worktrees for fork edits;
bootstrap checkouts are detached and should remain clean.

Give an Agent this instruction:

> Follow AGENTS.md and the Agent setup guide. Inspect this host and configure
> the locked source dependencies. Report what is validated and what remains
> unsupported. Ask before installing tools, downloading models or programming
> hardware; do not write flash or alter network settings implicitly.

The current handoff ends at **source-ready**, not inference-ready. The
[bootstrap profile](configs/profiles/bootstrap.json) explicitly disables build,
program, smoke-test, launch and benchmark execution. Those CLI commands display
plans; `--apply` fails closed. Future validated profiles will extend the same
entry point. See [Agent setup](docs/setup/agent-bootstrap.md) and
[script reference](scripts/README.md).

## Goal and reference model

Target: **single-user wall-clock decode above 10 tokens/s**, with full top-10
routing and stated context length. This is a target, not a result. Decode
includes host execution, device transfers and synchronization; MTP off/on are
reported separately. Never sum standalone CPU and FPGA throughput.

Reference hardware: dual AVX2 Xeon E5 v4, NUMA host RAM and a Kintex-7 FPGA with
4 GiB DDR. Packed cold experts stay in host RAM. FPGA work and resident weights
must earn their placement through measured end-to-end benefit.

[Qwen3.8-Flash-Next GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF):

| Variant | Role | Main GGUF shards combined, decimal GB |
| --- | --- | ---: |
| IQ2_XS | Primary | 68.0 |
| Q2_0 | Speed comparison | 66.4 |
| IQ3_S | Quality comparison | 83.6 |

Sizes include the common 28.8 GB n-gram shard, not the optional vision projector
or runtime allocations. Pack names are mixed-precision labels. IQ2_XS is not
directly consumable by openTPU FP4 kernels. Models are **not included or downloaded**.
The model and board configuration files are reference-only, not runnable profiles.

## Development management

All three repositories use the [central GitHub Project](https://github.com/users/lvyufeng/projects/11)
and [integration Issues](https://github.com/lvyufeng/strata-opentpu/issues) as the
live development-state source of truth. The forks own code/PRs and link back to
central tasks; merge alone does not close an integration acceptance task.
See the [development contract](docs/development.md) for claims, dependencies,
permissions, evidence and Agent handoff. No automatic Agent/hardware execution is
configured by the tracker.

## Source audit and proposed ABI

The [pinned execution/format audit](docs/audits/strata-execution.md) records
source-backed boundaries and explicitly missing actual-model metadata. The
[device ABI proposal](docs/design/device-abi.md) has a hardware-free
[conformance model](runtime/abi_contract.py); it is not an implemented device
transport or inference backend. These are development artifacts, not new enabled
bootstrap capabilities.

```sh
python3 scripts/check_source_anchors.py                 # metadata only, CI-safe
python3 scripts/check_source_anchors.py --check-sources # existing locked sources only
```

The second command never fetches sources. It checks citation ranges/excerpts
against clean pinned local checkouts, not the semantic truth of the audit or
model correctness. See the [script reference](scripts/README.md).

## Offline feasibility and CPU evidence preparation

The [resource-budget design](docs/design/resource-budget.md) and
[offline checker](scripts/resource_budget.py) account for each DDR bank and host
NUMA quota separately, packed tensor metadata, state/workspace/DMA allocations and
dependency-aware wall time. The
[reference budget](configs/budgets/kintex7-qwen38-reference.json) deliberately keeps
unmeasured quantities unknown; an incomplete budget is not a fit or a speed result.

The [CPU cold-expert protocol](benchmarks/cpu-cold-expert-protocol.md),
[partial evidence template](benchmarks/templates/cpu-cold-expert.json) and
[offline validator](scripts/check_cpu_evidence.py) define source/format/ISA/NUMA,
numerical and repeated full-branch wall-time evidence. They neither run Strata nor
provide a second CPU engine, obtain model tensors or certify submitted data as true.
Synthetic tests and accepted evidence structure are not measured CPU performance.

These independently scoped preparations are tracked in
[resource-budget subtask #15](https://github.com/lvyufeng/strata-opentpu/issues/15)
and [CPU-evidence subtask #16](https://github.com/lvyufeng/strata-opentpu/issues/16).
Their parent tasks still require actual mixed-tensor metadata, authorized resources
and reproducible measurements. No bootstrap execution capability is enabled.
See the [script reference](scripts/README.md) for offline entry points and limits.

## Supplied-packed scalar preparation

[Packed-oracle subtask #17](https://github.com/lvyufeng/strata-opentpu/issues/17)
adds an opt-in Q2_0/Q8_0 scalar decode/dot reference and independently generated
synthetic goldens in the openTPU fork. The integration
[contract/receipt](configs/contracts/opentpu-packed-oracle.json) and
[checker](scripts/check_packed_oracle.py) pin source identity, fixture integrity,
licenses and reviewed public CI; they do not duplicate the math.

```sh
python3 scripts/check_packed_oracle.py --strict-receipt # receipt only; no source execution
```

See the [packed-oracle design](docs/design/packed-oracle.md) for the selected
binary32 arithmetic, explicit trusted-source opt-in and limitations. Its 146
synthetic cases are not native/SIMD, IQ2_XS/IQ3_S, real-model, RTL/device or
throughput evidence. Accepting this narrow preparation does not accept or unblock
its parent #8, the Strata backend or the full-decode goal, and enables no bootstrap
execution capability.

## Bounded mixed-GGUF header preparation

[Inventory subtask #18](https://github.com/lvyufeng/strata-opentpu/issues/18)
adds a default-off, local-only header export to Strata's existing quant-format
tool. The integration [JSON schema](configs/contracts/gguf-header-inventory.schema.json)
and [validator](scripts/check_gguf_inventory.py) check a supplied receipt without
parsing GGUF, executing fork source or opening referenced artifacts:

```sh
python3 scripts/check_gguf_inventory.py # repository-anchored synthetic receipt
python3 scripts/check_gguf_inventory.py --receipt local/inventory.json --strict-complete
```

Confirmed row-block layouts use checked raw-shape byte arithmetic; unknown layouts
retain null storage, not zero, a bank fit or backend support. Header SHA excludes
padding/payload and is not an artifact SHA or authenticated provenance. Declared
split completeness never means model completeness; the synthetic fixture is not
an actual mixed pack. All readiness/provenance flags remain false, even in strict
mode. See the [inventory design](docs/design/gguf-inventory.md) and
[separate source acceptance record](configs/contracts/strata-gguf-inventory.json).
No model acquisition, installations, native builds, device operations or new
bootstrap execution capability are included, and the source lock is unchanged.

## Documentation and experiments

- [Architecture](docs/architecture.md) and [roadmap](docs/roadmap.md)
- [Configuration contracts](configs/README.md) and [dependency lock](dependencies.lock.json)
- [Upstream policy](docs/upstreams.md) and [historical reference commits](upstreams.json)
- [Experiments](experiments/README.md), [ideas](docs/ideas/README.md), [benchmark protocol](benchmarks/README.md)
- [Runtime integration boundary](runtime/README.md), [CPU boundary](backends/cpu/README.md), [FPGA boundary](backends/opentpu/README.md), [models](models/README.md)

The source lock pins reproducible inputs, **not a proven software/bitstream/model
combination**. No bitstream or inference API is provided. Tests and CI validate
metadata, documentation and source-setup safeguards, not inference performance.

## License

Copyright (c) 2026 lvyufeng and contributors. Original integration code is
[Apache-2.0](LICENSE). Strata remains MIT and openTPU remains Apache-2.0;
third-party components retain their licenses/notices. This does not relicense
weights or vendor IP. Dependencies are fetched separately, not vendored here.
