# Architecture: three-repository integration

## Current status

This repository is an integration scaffold, not a working FPGA inference engine.
The initial CLI can inspect the environment and obtain locked dependency sources.
Build, programming, hardware smoke testing, inference launch, and benchmarking
are planning interfaces only; they do not execute those operations in the
bootstrap profile. See the [Agent setup guide](setup/agent-bootstrap.md).

## Repository ownership

| Repository | Owns | Does not own |
| --- | --- | --- |
| [Strata fork](https://github.com/lvyufeng/Strata) | Existing model engine and CPU kernels; planned FPGA backend, heterogeneous scheduling, and NUMA changes | FPGA RTL or board programming procedures |
| [openTPU fork](https://github.com/lvyufeng/openTPU) | FPGA RTL/ISA, low-bit compute, device runtime, transfers, and synchronization | Complete model application or experiment management |
| [strata-opentpu](https://github.com/lvyufeng/strata-opentpu) | Locked dependency combinations, profiles, setup/build/launch/program entry points, experiments, and ideas | A second independent implementation of Strata's model engine |

This division is accepted in [ADR 0002](adr/0002-three-repository-integration.md).
All three projects are unofficial integration work, not upstream endorsement.

## What Strata already provides

The [upstream architecture description](https://github.com/Niko1221/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/docs/HOW_IT_WORKS.md)
describes GPU-resident hot experts, CPU computation of cold experts directly
from host RAM, adaptive expert caching, and MTP draft/verification. The GPU also
owns fixed model work such as attention/DeltaNet, gated residual, routing, shared
experts, output head, and model state. These are Strata's existing ideas, not
innovations invented by this integration project.

The intended approach is to preserve the existing model execution, cache, CPU
kernels, and MTP machinery where practical, then audit and extend the backend
boundaries. It is not a mechanical replacement of CUDA calls. An AVX2 expert
library alone is not a complete CPU-only Strata engine.

## Planned FPGA capability work

1. Inventory every GPU-side operation, tensor layout, device buffer, state
   dependency, and synchronization point at the locked Strata revision.
2. Classify each operation: reusable openTPU path, new low-bit hardware needed,
   or possible CPU fallback. Fallback correctness and latency must be verified;
   neither CPU coverage nor useful FPGA speedup is currently established.
3. Audit actual GGUF tensor formats. IQ2_XS pack names describe mixed precision,
   not one format universally compatible with an existing FP4 kernel.
4. Evaluate compressed-block decode/dot hardware and complete expert branches
   in the openTPU fork, without first expanding entire experts to FP32.
5. Budget fixed weights, KV/recurrent state, transfer buffers, and hot experts
   within board memory before claiming a useful resident cache.
6. Retain independent ordinary-decode and MTP measurements. Reuse existing MTP
   mechanisms where possible, but validate backend state commit/rollback.

The integration layer selects a tested combination and invokes its supported
entry points; it must not hide missing operators behind a profile marked ready.

The [supplied-packed scalar preparation](design/packed-oracle.md) is a narrow
source-level step in the openTPU fork: Q2_0/Q8_0 decode/dot and independently
generated synthetic goldens. This repository checks a pinned acceptance receipt
and fixture integrity rather than duplicating low-bit math. It is not an IQ
codebook implementation, quantizer, native CPU replacement or RTL/device adapter.
Its oracle commit is pinned separately from the unchanged bootstrap dependency
lock; passing synthetic tests does not accept or unblock the parent primitives
or model backend tasks and does not enable inference or device operations.

The [bounded local header inventory](design/gguf-inventory.md) extends Strata's
existing quant-format tool through a thin bounded stream adapter, not another
parser or intake. Integration checks only JSON structure, row-block arithmetic,
offsets and declared split facts. Unknown layouts remain null; header digests
are not artifact/tensor integrity or provenance. Even a complete synthetic header
set does not supply actual mixed-model, bank/state/residency, CPU/reference or
device evidence. No model is acquired and the bootstrap lock is unchanged.

## Reference development target

Dual Xeon E5 v4 (AVX2/FMA, no AVX-512/VNNI/AMX), NUMA host RAM (1 TiB reference),
and Inspur YPCB-00338-1P1 / Kintex-7 XC7K480T with two 2 GiB DDR3 banks.
Discover actual negotiated PCIe link, DMA throughput, and device NUMA locality;
endpoint capability is not a measured transfer rate. Host RAM capacity does not
enlarge the FPGA's 4 GiB memory. No private connection data belongs in Git.

The primary Qwen3.8-Flash-Next model has 48 layers, 512 routed experts per layer,
full top-10 plus a shared expert, QSA, DeltaNet, gated residual, n-gram/PLE, and
MTP. Existing upstream model support is a reuse opportunity, not proof of support
on a new FPGA backend. See the [official configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json).

## Reproducible configuration boundary

[dependencies.lock.json](../dependencies.lock.json) selects exact fork commits.
The [bootstrap profile](../configs/profiles/bootstrap.json) is bootstrap-only,
not inference-ready. Future verified profiles must additionally identify board,
toolchain, bitstream SHA256, software/firmware ABI, model revision/checksums,
and runtime settings. Loading an arbitrary bitstream against arbitrary source
revisions is not a supported configuration.

Volatile FPGA programming and persistent flash writes are separate operations.
Neither is implemented by the bootstrap profile; persistent flash support is not
implemented at all. Installs, model downloads, and programming need explicit
operation-specific authorization. See the [upstream policy](upstreams.md).

## Performance acceptance

For genuinely independent expert branches only:

```text
T_layer ~= T_serial + max(T_CPU_experts, T_FPGA_experts) + T_sync_and_merge
```

Transfers, contention, and dependencies can reduce overlap. Whole-model CPU and
FPGA tokens/s cannot be added. The >10 single-user wall decode tokens/s target
requires <100 ms/token including all layers, host work, transfers, and sync.
No result is claimed. Use full routing, expert skip=0, and separate MTP off/on
runs under the [benchmark protocol](../benchmarks/README.md).

The [offline budget contract](design/resource-budget.md) makes the bank/host,
packed-weight, complete-state and dependency accounting explicit. Its reference
inputs remain incomplete, not measured feasibility. The
[CPU evidence protocol](../benchmarks/cpu-cold-expert-protocol.md) covers existing
Strata cold-expert branches, not a replacement engine or proof that fixed work
already runs on CPU. Both are preparatory artifacts; operation-specific permission
and actual model/host/device evidence remain separate gates.
