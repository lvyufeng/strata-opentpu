# Architecture: planned, not implemented

## Reference target

Dual Xeon E5 v4 (AVX2/FMA, no AVX-512/VNNI/AMX), NUMA host RAM (1 TiB reference),
Inspur YPCB-00338-1P1 / Kintex-7 XC7K480T with two 2 GiB DDR3 banks.
Discover actual negotiated PCIe link, DMA throughput and NUMA locality; endpoint
capability is not a measured transfer rate. No private connection data is stored.

## Execution

1. Read GGUF metadata and actual tensor types rather than filename labels.
2. Keep experts packed in host RAM, invoke audited per-format CPU kernels.
   AVX2 suffices for several IQ paths; AVX-VNNI is optional. Activation
   quantization, block decoding, integer SIMD and scaling still cost time.
3. Keep n-gram lookup in host RAM or a measured mmap policy. Host RAM capacity
   does not enlarge the FPGA's 4 GiB memory.
4. Implement attention/DeltaNet, gated residual, shared experts, router, embeddings
   and output head. CPU fallback is a design, not an existing full Strata backend.
5. Run eligible independent CPU/FPGA work, transfer activations/results, then
   synchronize and merge before dependent work starts.
6. Add MTP only after a correct target-model baseline.

## Model scope

Qwen3.8-Flash-Next has 48 layers, 512 routed experts per layer, top-10 plus
shared expert, hidden size 2560 and expert intermediate size 640. QSA, 4-branch
gated residual, n-gram/PLE and MTP mean older Qwen ports are not sufficient.
Official overview: 125B main model / about 6B active, plus 51B n-gram and 4B MTP.
[Official configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json).

## Backend contracts to define

CPU: packed views, activation quantization, expert calls, format capabilities,
thread placement, deterministic outputs. FPGA: capabilities, buffer lifetime,
transfers, submission/completion, weight layouts and errors. Runtime: routing,
partitions, dependencies, cancellation, merging and traces.
CPU byte-serving is not CPU expert computation.

Initially prefer existing openTPU mechanisms. Mailbox/WAITW sufficiency and the
need for new RTL/ISA must be tested in a correct minimal chain, not assumed.

## Performance

For independent branches only:

```text
T_layer ~= T_serial + max(T_CPU_experts, T_FPGA_experts) + T_sync_and_merge
```

Transfers/contention/dependencies can reduce overlap. Whole-model CPU and FPGA
rates cannot be added. >10 wall decode tokens/s means <100 ms/token including
all layers and host overhead. No programming, downloads or inference API at
bootstrap; no throughput promised.
