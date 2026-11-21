# Roadmap

This roadmap replaces the initial plan to own a new model engine here. The
[three-repository decision](adr/0002-three-repository-integration.md) keeps model
execution in the Strata fork and hardware capability work in the openTPU fork.
The integration CLI remains bootstrap-only; no FPGA inference, usable bitstream
release, or throughput result is available from this repository yet.

## M0: Public scaffold (complete)

Original license, upstream reference metadata, documentation, and hardware-free
checks. [upstreams.json](../upstreams.json) preserves the original reference
commits; it is not a verified build combination.

## M1: Reproducible integration scaffold (current stage)

- Maintain the two forks and lock exact commits in
  [dependencies.lock.json](../dependencies.lock.json).
- Provide `doctor`, plan-first dependency bootstrap, and a bootstrap-only profile.
- `bootstrap --apply` obtains locked Git sources; it does not build, install,
  download models, program hardware, or make inference ready.
- Keep build/program/smoke-test/launch/benchmark as explicit plans. Applying
  these operations must fail until a supported, verified profile exists.
- Document the Agent procedure and isolate local state/results from Git.

Exit: repeatable source acquisition and honest capability checks, not working
inference. See the [setup guide](setup/agent-bootstrap.md).

## M2: Backend boundary audit and performance budget (pending)

- Audit the complete Strata GPU execution boundary, not only CPU expert kernels.
- Preserve existing model operators, adaptive hot-expert caching, cold-expert CPU
  execution, and MTP wherever practical.
- Inventory required operators, layouts, buffers, memory, and state dependencies.
- Benchmark actual mixed-quantized experts and whole expert branches on AVX2;
  compare single/multi-token work, threads, and NUMA allocation.
- Record routing traces and model a board-memory budget before selecting resident
  fixed work or hot experts. CPU fallback needs both correctness and latency proof.

Exit: an explicit supported/missing-operation matrix and measured feasibility
budget. No assumption that all GPU work fits or performs well on the FPGA.

## M3: Fork capability development (pending; parallel tracks)

**Strata fork:** add narrowly scoped backend and scheduling changes, reuse model
execution, define error/state handling, and improve CPU/NUMA work distribution.

**openTPU fork:** implement required device interfaces and evidence-driven low-bit
compute/RTL/ISA extensions. Validate decoded blocks, dots, and expert branches
against trusted small-tensor references before measuring speed.

**Integration repository:** maintain plans, experimental profiles, dependency
combinations, test invocation, sanitized results, and failure diagnostics.

Exit: a numerically correct minimal CPU/FPGA chain with real transfer/sync costs,
compared against identical CPU work. Synthetic or layer tests do not prove
complete-model inference support.

## M4: Verified build, program, and smoke-test profile (pending)

Select exact software commits, board identity, toolchains, bitstream SHA256, and
software/firmware ABI. Add real command adapters only for supported profiles.
Validate target selection before volatile programming; keep persistent flash
writes separate and unsupported until explicitly implemented. Installs, model
acquisition, and programming remain separately authorized operations.

Exit: an independently checked hardware smoke-test path and reproducible build
or permitted release-artifact acquisition. A successful device test alone does
not mark a profile inference-ready.

## M5: Complete model and throughput validation (pending)

- IQ2_XS primary, Q2_0 speed comparison, IQ3_S quality comparison; actual tensor
  formats must be covered. Pin model revisions and shard checksums.
- Validate the complete ordinary-decode path against a trusted reference, retain
  full top-10 and expert skip=0, and report stated context/output lengths.
- Reuse/adapt Strata MTP only with verified state commit/rollback; report MTP
  off/on separately with accepted-token statistics.
- Profile fixed work, expert misses, NUMA, transfers, and sync before tuning
  residency or adding hardware. Publish wall decode, TTFT, quality, and variation.

Exit: a genuinely inference-ready profile with reproducible correctness and
performance evidence. >10 single-user wall decode tokens/s is a target, not a
promise. See the [benchmark protocol](../benchmarks/README.md) and
[experiment policy](../experiments/README.md).
