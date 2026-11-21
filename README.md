# strata-opentpu

Experimental CPU + FPGA heterogeneous inference runtime, inspired by
[Strata](https://github.com/Niko1221/Strata) and built around
[openTPU](https://github.com/FeSens/openTPU).

**Status: project bootstrap only.** No inference runtime, CPU kernel bindings,
or FPGA backend is implemented here yet. This is an independent, unofficial
project, not an upstream-endorsed release or a fork of Strata's GPU application.

## Goal

Explore larger low-bit MoE models on AVX2 CPUs, host RAM, and a Kintex-7 FPGA.
Keep cold experts compressed in RAM and compute them on the CPU. Transfer
activations/results rather than repeatedly moving expert weights over PCIe.
Assign FPGA work only when end-to-end measurements justify it.

Target: **single-user wall-clock decode above 10 tokens/s**, with complete routing
and a stated context length. This is a target, not a result. CPU-only may win for
some partitions; adding FPGA work is not assumed to improve throughput.

## Initial model selection

[Qwen3.8-Flash-Next GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF):

| Variant | Role | Main GGUF shards combined, decimal GB |
| --- | --- | ---: |
| IQ2_XS | Primary starting point | 68.0 |
| Q2_0 | Speed comparison | 66.4 |
| IQ3_S | Quality comparison | 83.6 |

Sizes include the common 28.8 GB n-gram shard, not the optional vision projector
or runtime allocations. Pack names are mixed-precision labels, not universal
tensor formats. IQ2_XS is not directly consumable by openTPU FP4 kernels.
Models are **not included or downloaded**. The expert-pruned Coder release is a
separate capability-targeted option, not our first general-purpose baseline.

## Planned boundaries

- CPU adapter: audited Strata AVX2 quantized experts, packed blocks decoded during
  dot products rather than full-expert FP32 expansion.
- openTPU adapter: device capabilities, transfers, synchronization, FPGA work.
- Runtime: model execution, routing, NUMA-aware scheduling and merging.
- Benchmarks: correctness, expert throughput, layer latency and wall decode.

Strata's CPU expert library does not provide its complete engine CPU-only;
the existing model engine depends on supported GPU backends. Our execution path
requires substantive operator work, including QSA, DeltaNet, gated residual and
n-gram support. MTP follows a correct non-speculative baseline.

## Documentation

- [Architecture](docs/architecture.md)
- [Roadmap](docs/roadmap.md)
- [Independent repository decision](docs/adr/0001-independent-repository.md)
- [Upstream licensing policy](docs/upstreams.md) and [reference commits](upstreams.json)
- [Benchmark protocol](benchmarks/README.md)
- [Runtime](runtime/README.md), [CPU](backends/cpu/README.md), [FPGA](backends/opentpu/README.md), [models](models/README.md)

## Bootstrap checks

Python 3.11+, no external packages, downloads or hardware needed:

```sh
python3 -m unittest discover -s tests -v
```

These validate metadata and documentation, **not inference or FPGA performance**.
CI performs only these bootstrap checks.

## License

Copyright (c) 2026 lvyufeng and contributors. Original contributions are
[Apache-2.0](LICENSE). Imported code retains its own licenses/notices; this does
not relicense upstream software, weights or vendor IP. No upstream implementation
is vendored in this initial repository.
