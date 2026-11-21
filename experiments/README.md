# Experiments

No hardware, inference, or throughput experiments are implemented here yet.
The integration CLI's `benchmark` and `smoke-test` commands are plans only in the
bootstrap profile; applying them fails until supported adapters exist. See the
[Agent setup guide](../docs/setup/agent-bootstrap.md).

This directory will hold reproducible experiment definitions and reviewed,
sanitized summaries, not private raw logs, model weights, or bitstreams. It is
separate from the software/RTL implementation in the two capability forks.

## Required experiment record

- Question/hypothesis and connection to an [idea](../docs/ideas/README.md).
- Exact Strata/openTPU commits and profile; distinguish a source-only lock from
  a validated hardware/inference combination.
- Board/toolchains, bitstream SHA256 and software/firmware ABI when applicable.
- Model revision, shard checksums, actual tensor formats, context/output lengths,
  and sampling/seed when applicable.
- CPU features, thread placement/NUMA, memory policy, device locality, negotiated
  link, and measured DMA when relevant; omit private host identifiers.
- Baseline, correctness criteria, timing boundaries, repetitions/variation, and
  acceptance/rejection criteria established before interpreting speed.
- Whether evidence is analytical, simulated, CPU-only, device-only, single-layer,
  or complete-model. State missing capabilities and failed checks explicitly.

For decode results, use the [benchmark protocol](../benchmarks/README.md): full
routing/top-10, expert skip=0, warm wall time including transfer/synchronization,
accepted output only, and separate prefill/TTFT/decode and MTP off/on. Do not add
standalone CPU and FPGA rates or label a kernel benchmark as model throughput.

## Storage and safety

Keep raw local artifacts in ignored `local/`, `results/`, or `logs/`; review
summaries before publication. No private addresses/paths, credentials, weights,
vendor IP/licenses, bitstreams, or unreviewed build outputs in Git. Any future
public bitstream must pass licensing review and belong in a reviewed Release,
not this directory.

An experiment definition is not permission to install tools, download models,
run upstream build scripts, program a device, write flash, or change networking.
Obtain operation-specific authorization and verify the exact target. Persistent
flash programming is not implemented by this project.
