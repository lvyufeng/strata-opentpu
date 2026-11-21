# ADR 0002: Three-repository integration

- Date: 2026-11-21
- Status: accepted
- Supersedes: [ADR 0001](0001-independent-repository.md)

## Context

The initial decision treated this repository as the owner of a new model
execution layer, mainly borrowing Strata CPU expert kernels. That would risk
reimplementing substantial existing functionality. Strata already has resident
hot GPU experts, CPU cold-expert execution, adaptive expert caching, and MTP;
these are the baseline to reuse, not new ideas of this project. Its GPU also
executes fixed model operators and maintains state, so an FPGA backend requires
a full boundary audit rather than a few call substitutions.

Both upstreams need capability additions: Strata needs model/backend/scheduling
adaptation, while openTPU needs hardware/device-runtime additions. The user wants
one cloneable integration repository for experiments, setup/launch entry points,
bitstream loading, and Agent instructions, not a third implementation engine.

## Decision

1. Develop model execution, CPU/NUMA, cache/MTP adaptation, and FPGA backend
   integration in the [Strata fork](https://github.com/lvyufeng/Strata).
2. Develop RTL/ISA, low-bit compute, device runtime, DMA, and synchronization in
   the [openTPU fork](https://github.com/lvyufeng/openTPU).
3. Keep [strata-opentpu](https://github.com/lvyufeng/strata-opentpu) independent
   for source locks, compatible profiles, scripts, experiments, and documentation.
   It does not own a new full model engine.
4. Lock exact fork commits in [dependencies.lock.json](../../dependencies.lock.json).
   Preserve [upstreams.json](../../upstreams.json) as historical reference metadata,
   not a verified configuration.
5. Keep Agent operation plan-first and script-driven. The bootstrap profile only
   supports environment inspection and explicitly applied dependency acquisition.
   Other operational commands remain plans and reject application until a
   supported profile exists.

## Consequences and boundaries

The combined deliverable is a verified configuration, not three floating branch
tips. Future profiles must identify software commits, board, toolchain, bitstream
SHA256 and software/firmware ABI, model revision/checksums, and runtime settings.
No such inference-ready combination is provided at bootstrap.

Volatile FPGA loading is distinct from persistent flash writing. Neither is
implemented in the current profile; persistent flash support is not implemented.
Installs, model downloads, and hardware operations require explicit authorization
for the operation. Dependency bootstrap is not blanket permission for them.

Capability forks have upstream merge costs, so changes should stay scoped and
maintain modification history; generic improvements should be proposed upstream
where suitable. Each fork keeps its upstream/third-party licenses. Original
Apache-2.0 integration code does not relicense Strata MIT, openTPU Apache-2.0,
models, or vendor IP. This is unofficial, not upstream-endorsed work.

CPU fallback coverage, low-bit FPGA compute, memory residency, and end-to-end
performance remain to be verified. No working FPGA model inference or >10 tok/s
result is implied by the repository structure. See the
[architecture](../architecture.md), [roadmap](../roadmap.md), and
[Agent setup guide](../setup/agent-bootstrap.md).
