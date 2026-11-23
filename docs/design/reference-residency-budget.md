# Reference-only 4 GiB residency arithmetic

**Status: analytical planning only.** The
[residency calculator](../../scripts/residency_reference.py) reads one validated
mixed-GGUF **header** inventory receipt and performs declared byte arithmetic
against the reference board envelope. It does not open a device, read model
payload, enumerate PCIe, allocate or pin memory, run Strata/openTPU, program a
board, fetch URLs or execute anything named in the input. No number below is a
residency fit, a cache placement, a link speed, a wall-time measurement or a
tokens/s result.

This is the hardware-free part of [central Issue #6](https://github.com/lvyufeng/strata-opentpu/issues/6),
the memory/PCIe/NUMA budget parent, and the companion of
[partial-tensor budget schema v2](resource-budget-v2.md) and the
[full budget contract v1](resource-budget.md). The device side of #6 remains
blocked on authorized measurement; this document only declares byte ceilings,
unknowns and a measurement plan so that measurement can be executed without
redesign.

## 1. Inputs and identity

| Input | Identity | Use |
| --- | --- | --- |
| `local/acquisition/iq2xs-inventory/inventory.v2.json` | sha256 `fac6339ae1cb33e9742e24ff519aba721b27053c6d97330cd62a79606243c588`; 1,224 tensors; `known_storage_bytes` 68,015,068,160; `input_kind` `local-unverified`; `claims.header_only` true, all byte/verification claims false | per-tensor native GGUF storage bytes `[W]` |
| `local/acquisition/qwen38flasnext-config.json` | sha256 `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b` | layer/KV/state/head geometry, dtypes `[C]` |
| [architecture.md](../architecture.md) | reference target Inspur YPCB-00338-1P1 / Kintex-7 XC7K480T with two 2 GiB DDR3 banks; the budget line names fixed weights, KV/recurrent state, transfer buffers and hot experts | device envelope and decomposition `[D]` |
| [cpu-cold-expert-protocol-v2.md](../../benchmarks/cpu-cold-expert-protocol-v2.md) and its accepted report | sha256 `6e9f0644…` report, 237 measured cases, one expert branch per case | CPU branch timings used only for the projection `[M]` |

The receipt file name and the acquisition note say `IQ2_XS`, but the real
per-tensor type ids do not: expert gate/up matrices are **IQ2_S on 34 layers,
IQ2_XXS on 11 layers and IQ1_M on 3 layers**, and every expert down matrix is
**Q2_0**. Those are the types charged here; do not reuse the `IQ2_XS` shorthand in
new records. Unknown-layout tensors make the calculator fail closed (exit 2)
instead of guessing bytes.

Reproduce (no network, no device):

```sh
python3 scripts/residency_reference.py --receipt local/acquisition/iq2xs-inventory/inventory.v2.json
python3 scripts/residency_reference.py --receipt local/acquisition/iq2xs-inventory/inventory.v2.json \
    --format json --contexts 4096,32768
```

The calculator rejects a receipt whose summed tensor bytes disagree with its own
`totals`, whose layer/expert geometry is not the pinned 48-layer/512-expert model,
whose expert axis is not the last dimension, or that repeats a tensor name. Exit 0
means the declared arithmetic ran; it is not a verification of the file's contents.

## 2. Declared per-layer weight bytes `[W]`

48 decoder layers, 512 experts per layer, `hidden_size` 2560, `moe_intermediate_size`
640. Bytes are native GGUF storage bytes from the receipt, including its quant
overhead and row padding.

| Component | Bytes | Notes |
| --- | --- | --- |
| Persistent (non-`_exps`) per layer, min | 58,294,272 | layer 35 |
| Persistent (non-`_exps`) per layer, max | 127,660,416 | layer 1 (includes PLE tensors) |
| Persistent sum over 48 layers | **3,071,375,360** | attention/SSM/router/norm/hc/ple per-layer weights |
| Expert set per layer, IQ2_S (34 layers) | 773,324,800 | 1,510,400 per expert |
| Expert set per layer, IQ2_XXS (11 layers) | 668,467,200 | 1,305,600 per expert |
| Expert set per layer, IQ1_M (3 layers) | 602,931,200 | 1,177,600 per expert |
| Expert sum over 48 layers | **35,454,976,000** | 34×773,324,800 + 11×668,467,200 + 3×602,931,200 |

Per-expert native footprint `[W]`, from the frozen slice fixture
(`local/issue5-slices.json`,
sha256 `4c570bcb…`), blob = `[gate|up|down]`:

| Format | gate bytes | up bytes | down bytes | footprint |
| --- | --- | --- | --- | --- |
| IQ2_S | 524,800 | 524,800 | 460,800 | 1,510,400 |
| IQ2_XXS | 422,400 | 422,400 | 460,800 | 1,305,600 |
| IQ1_M | 358,400 | 358,400 | 460,800 | 1,177,600 |

Non-layer globals `[W]`: `token_embd.weight` 337,715,200 and `output.weight`
337,715,200, `output_hc_down.weight` / `output_hc_up.weight` 6,553,600 each,
`output_hc_norm.weight` 40,960 — **688,578,560** fit candidate bytes in total. The
per-layer token embedding table `per_layer_token_embd.weight` is
**28,800,138,240** for a declared size of `[160, 320001536]`; it exceeds the whole
4 GiB envelope by itself, so its placement is `unknown` and it is excluded from
every fit-candidate row instead of being silently dropped. A declared
`source_retained` or converted-on-load copy is not charged here at all.

The inventory's total known storage is 68,015,068,160 bytes. Consequently the
4 GiB envelope cannot hold the model: 35,454,976,000 expert bytes alone exceed it
by roughly 8×. Any residency statement is therefore about a **partial** selection
with a CPU cold-expert path, which matches the architecture intent recorded in
[architecture.md](../architecture.md) and the Strata ownership boundary; this
calculator does not decide which experts are hot.

## 3. Declared recurrent state and KV arithmetic `[C]`

State dtype is declared `float32` (`mamba_ssm_dtype`); KV dtype is assumed
`bfloat16` from the top-level `dtype` and is **not** verified. Declared geometry:
36 linear-attention layers, 12 full-attention layers at indices 3, 7, …, 47, 48
linear value heads × 128, 16 linear key heads × 128, conv kernel 4, 2 KV heads ×
256 head dim, indexer 1 KV head × 128 head dim, `max_position_embeddings` 262,144.

| Item | Arithmetic | Bytes |
| --- | --- | --- |
| DeltaNet value state | 36 × 48 × 128 × 128 × 4 | 113,246,208 |
| Conv state | 36 × (4−1) × 10,240 × 4 | 4,423,680 |
| Recurrent state total | sum | **117,669,888** |
| Full-attention KV per token | 12 × 2 × 256 × 2 × 2 | 24,576 |
| Indexer KV per token | 12 × 1 × 128 × 2 | 3,072 |
| KV + indexer per token | sum | **27,648** |

| Context tokens | 2,048 | 4,096 | 8,192 | 32,768 | 131,072 | 262,144 |
| --- | --- | --- | --- | --- | --- | --- |
| KV + indexer bytes | 56,623,104 | 113,246,208 | 226,492,416 | 905,969,664 | 3,623,878,656 | 7,247,757,312 |

Not charged, and not known: KV block/page metadata and allocator rounding, sampler
state, MTP state (`mtp_num_hidden_layers` 1 is declared but its placement and
resident cost are unmeasured), activation workspace, DMA ring, routing metadata and
any fragmentation. The table is thus a **lower bound on required bytes**, and the
hot space derived from it is an upper bound.

## 4. Device budget scenarios on 4 GiB `[P]`

Envelope: 4,294,967,296 bytes in two declared 2,147,483,648-byte banks. Charged
items: persistent fixed weights or their double-buffered stream, the recurrent
state, KV for the stated context, a double-buffered one-expert DMA staging area
(2 × 1,510,400 = 3,020,800), and for streaming scenarios a double-buffered
largest-layer staging area (2 × 127,660,416 = 255,320,832). “Hot” is the remainder
available for selected expert slots before fragmentation/workspace, divided by the
per-expert footprint; the slot range spans the cheapest and most expensive format.
Coverage percent is slots ÷ (48 × 512) resident expert tensors. One token routes
10 experts per layer = 480 expert applications, so “token-equivalents” below is
slots ÷ 480 with **no reuse assumed** — it is not a hit rate.

| Scenario | Resident composition | ctx 4,096 hot bytes | slots | coverage | token-equiv. | ctx 32,768 hot bytes | slots |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| S0 all fixed resident | fixed 3,071,375,360 + state 117,669,888 + DMA 3,020,800 = 3,192,066,048 | 989,655,040 | 655–840 | 2.67 % | 1.36 | 196,931,584 | 130–167 |
| S0G S0 + fittable globals | + globals 688,578,560 = 3,880,644,608 | 301,076,480 | 199–255 | 0.81 % | 0.41 | −491,646,976 (does not fit) | 0 |
| S1 fixed streamed | state 117,669,888 + layer stream 255,320,832 + DMA 3,020,800 = 376,011,520 | 3,805,709,568 | 2,519–3,231 | 10.25 % | 5.25 | 3,012,986,112 | 1,994–2,558 |
| S1G S1 + fittable globals | 1,064,590,080 | 3,117,131,008 | 2,063–2,647 | 8.39 % | 4.30 | 2,324,407,552 | 1,538–1,973 |
| S2 layer-partitioned hot | same envelope as S1, greedy cheapest-first layer pack | 3,805,709,568 | 2,519 | — | — | — | — |

Context headroom with no hot experts (`max_context_tokens_at_zero_hot`): S0
39,890; S0G 14,985; S1/S2 141,744; S1G 116,839. S0G cannot hold 32,768 tokens of
KV at all — the shortfall is 491,646,976 bytes.

S2 greedy cheapest-first packs **5 of 48 layers** (3× IQ1_M 602,931,200 +
2× IQ2_XXS 668,467,200 = 3,145,728,000 bytes) into the ctx-4,096 remainder, i.e.
10.42 % of expert bytes. That is a hypothetical arithmetic order, not a placement:
a real layer-contiguous or tensor-row-pinned selection would fit fewer, and no
adaptive cache policy is modeled.

Bank layout. The frozen native layout keeps an expert blob as two allocations —
gate+up (1,049,600 max) and down (460,800) — so a slot must be split across the two
declared banks; per-expert blob maximum 1,510,400 bytes. Every persistent weight
tensor and the state tensors individually fit one bank (largest single resident
item is layer 1 fixed weights at 127,660,416). Whether the board actually
interleaves the two banks is unknown; the calculator charges the envelope as one
pool and therefore **cannot** prove a bank-local placement. Fragmentation,
per-allocation padding, DMA ring, activation workspace and metadata are not
charged, so each “hot bytes” cell is a ceiling that a real allocator has not yet
met.

## 5. What is unknown, and why #6 stays blocked

`[U]` for every item: device usable capacity after reserve/ECC and allocator
overhead; largest contiguous allocation and bank interleaving; device-side DDR3
bandwidth and latency; negotiated PCIe width/speed (capability is not the link);
DMA directions, sizes, repeat counts and measured rates; activation/workspace,
DMA ring, metadata and sampling buffer bytes; fragmentation and padding; the
routing trace and expert hit distribution; device compute availability and
per-layer rate; MTP placement cost. The calculator prints this list in every
report, so no downstream record can inherit a “fit” claim from it by accident.

Gen2 x8 module capability must never be reported as the negotiated link or as a
throughput. Likewise 655 expert slots is not a cache hit rate: without a routing
trace, an unknown fraction of the 480 per-token expert applications is cold and
must traverse the CPU path.

On 2026-11-23 a read-only run on the reference host replaced part of that `[U]`
list with measured inputs: the device-reported capacity and configuration, the
negotiated link, both DMA directions at the declared sizes, and the host NUMA
topology and placement (§6, “Measured results”). Capacity *allocation* and bank
interleaving, device-side DDR bandwidth/latency, bitstream provenance,
workspace/fragmentation, routing trace and device compute availability remain
`[U]`; the measured items are evidence only and none of them is a throughput or a
tokens/s result.

## 6. Hardware measurement plan (executable checklist)

Only read-only operations are listed. Nothing here flashes, resets, reprograms or
reconfigures the device, driver or network, and a Ready Project field is not
authorization. Before running any of it the operator records machine name, board
identity source, allowed read-only command set, evidence directory and timebox in
the #6 comment; any step that needs a write becomes a separate explicit approval.

### Phase 0 — preconditions

- [ ] Operator confirms the target machine and that the FPGA board is present;
      authorization text quoted in the #6 comment.
- [ ] Read-only evidence directory chosen outside Git (`local/` or host scratch);
      raw files stay private, only sanitized summaries are published.
- [ ] Record host fingerprint: `uname -a`, `lscpu`, `nproc`, `/proc/meminfo`
      (sanitize hostnames/IPs before publication).
- [ ] Confirm tools actually present on that machine. Do not assume `numactl`,
      `ninja` or `timeout`; the earlier host lacked all three. Use `/sys` and
      `lscpu` fallbacks when a tool is missing.

### Phase 1 — device identity (read-only)

- [ ] Enumerate candidates read-only: `lspci -nn` and `lspci -Dnnttt`; capture the
      vendor/device/subsystem triple and class. Unknown vendor id → record
      `unknown`, do not substitute another server or board.
- [ ] For the selected BDF record `/sys/bus/pci/devices/<bdf>/{vendor,device,class,
      revision,subsystem_vendor,subsystem_device,numa_node}` and the bound driver
      name (`readlink /sys/bus/pci/devices/<bdf>/driver`).
- [ ] Record IOMMU group (`ls /sys/kernel/iommu_groups/*/devices`) presence only;
      do not change kernel command line or grouping.
- [ ] Write the facts into a template of the
      [v2 budget schema](../../configs/contracts/resource-budget-v2.schema.json)
      `device` section, leaving unknown fields explicitly unknown.

### Phase 2 — negotiated link (read-only)

- [ ] `lspci -vv -s <bdf>` and record `LnkCap`, `LnkSta`, `LnkCap2`, `LnkSta2`
      verbatim into the private evidence file: capability Gen/width **and**
      current/negotiated Gen/width separately.
- [ ] Cross-check `/sys/bus/pci/devices/<bdf>/max_link_speed`,
      `max_link_width`, `current_link_speed`, `current_link_width`.
- [ ] If a field is absent or “Unknown”, record `unknown`; never infer the
      negotiated speed from the board or module datasheet.
- [ ] Decision rule: only the current/negotiated pair may be entered into a budget
      evidence record; the capability pair is labeled capability-only.

### Phase 3 — DMA smoke (read-only traffic; no programming)

- [ ] Confirm an already-authorized user-space runtime and record its binary
      SHA256; if none exists, stop and mark the phase **Awaiting approval** with
      the exact missing artifact. Building or installing one is out of scope.
- [ ] For each direction (`h2d`, `d2h`, and `h2d → device-side read → d2h`) use the
      declared transfer sizes 1,510,400 (max expert blob), 127,660,416 (largest
      layer staging) and 3,020,800 (double buffer); record pinned vs pageable,
      repeat count ≥ 5 with one warmup, and every raw sample.
- [ ] Record per-sample timestamps, size, direction, host CPU id, NUMA node, and
      the tool/runtime identity; compute median and p95 offline, not by hand.
- [ ] Abort conditions: any step that would reset the link, reload a driver,
      reprogram the FPGA, change IOMMU settings or alter the network. Stop and
      report instead.
- [ ] Decision rule: rates are evidence only after the raw samples and identity
      are archived; a single fast sample is not a rate.

### Phase 4 — NUMA locality (read-only first)

- [ ] Capture `/sys/devices/system/node/node*/cpulist`, `lscpu -e` and the BDF's
      `numa_node`; if `numactl --hardware` exists, capture it too.
- [ ] Verify the CPU set used for any CPU-side branch is inside one NUMA node and
      matches the receipt's shape-only `affinity_sha256` caveat; do not treat a
      shape digest as locality evidence.
- [ ] Any `mbind`/`set_mempolicy` experiment (mode 3 interleave) is a separate
      memory-policy operation and needs its own authorization; it is not part of
      read-only enumeration.

### Phase 5 — evidence packaging

- [ ] Sanitize hostnames/IPs/serials; write a SHA256 manifest of the private raw
      files; publish only the sanitized summary into the issue comment.
- [ ] Update the #6 checklist only with facts produced by Phases 1–4; leave
      remaining items unchecked and the Project status blocked until then.
- [ ] Store the budget arithmetic JSON next to the evidence and mark which cells
      changed from ceiling to measured, and which remain `unknown`.

### Measured results — 2026-11-23 (read-only, reference host)

Authorization and scope are quoted in the #6 comment trail; the operations were
read-only (BAR0 register reads through `/dev/xdma0_user`, DMA writes/reads of
synthetic patterns only, in-process memory placement). Nothing was flashed, reset,
reprogrammed or reconfigured, no register was written, and the driver, kernel
command line and network were not touched. The host is a 2 × Xeon E5-2696 v4
machine (88 logical CPUs, 2 socket-aligned NUMA nodes) running Ubuntu 22.04,
kernel 5.15.0-203-generic; no hostname, address, login or serial is recorded in
this repository.

Device facts read from the register map (v5, LE words; a card left initialized by
a prior session, no reset was issued):

| Register | Value | Reading |
| --- | --- | --- |
| `0x000 ID` | `0x4f545055` | `ID_OTPU` — live openTPU bitstream, AXI-Lite responds |
| `0x03C REGMAP` | 5 | register map v5 |
| `0x040 CAPS` | `0xef0a0eff` | matches pinned RTL top-bit constant for this build |
| `0x044 CORE_KHZ` | 133333 | 133.333 MHz accelerator clock |
| `0x048 BUILD_ID` | `0xd84db557` | build *parameter*, not a bitstream hash |
| `0x054 DDR_MTS` | 1066 | DDR3-1066 |
| `0x058 ACT_ROWS` | 16 | activation rows |
| `0x05C CAPS2` | 0 | r2 default `COLL_GATHER=0` (not an anomaly) |
| `0x0C0/0x0C4/0x0C8/0x0CC CFG` | 1 / 65536 / 32768 / 128 | slice / TMEM words / IMEM words / activation blocks |
| `0x0D0 CFG DRAM` | `0x001f0220` | **4,294,967,296 B total, 2,147,483,648 B per channel, 64-byte beats** |
| `0x0D8 CFG PLATFORM` | 1 | YPCB-00338 platform id |
| `0x0DC CFG ARCH_HASH` | `0x4ffd1e17` | equals the `variants.m4` config hash |

The device therefore **reports 4 GiB of DDR3 in two 2 GiB channels**, which
corroborates the operator-confirmed reference board (Inspur YPCB-00338-1P1,
XC7K480T) and the configuration file's declared `ddr_total_bytes`. `BUILD_ID` is
the pinned openTPU commit's first eight hex digits as passed at build time; it
does **not** identify the flashed image, so bitstream SHA256 remains unknown.

Negotiated link, read from sysfs because `lspci -vv` capability output is not
available to a non-root user: endpoint capability **Gen2 x8**, negotiated
**Gen2 x4**; the parent root port caps `max_link_width` at 4, so x4 is the slot
ceiling, not a card or cable downgrade. At 5.0 GT/s × 4 lanes the raw 8b/10b
signalling ceiling is about 2.0 GB/s per direction — a theoretical bound that is
never reported as achieved bandwidth.

DMA smoke, synthetic patterns, 5 repeats + 1 warmup per group, median of 5; host
buffer pageable (no `mlock`) with the documented +2048 placement offset inside the
mapping; 8 MiB chunk for the largest group. Every rep verified by full-region
compare and SHA256, and the card status register and all identity registers were
identical before and after the run. Values are MiB/s.

| Bytes | Chunk | H2C | C2H |
| ---: | ---: | ---: | ---: |
| 1,510,400 (max expert blob) | 4 KiB | 178.0 | 179.5 |
| 1,510,400 | 1 MiB | 1124.2 | 944.8 |
| 3,020,800 (double buffer) | 1 MiB | 1197.0 | 1025.8 |
| 127,660,416 (largest layer) | 8 MiB | **1352.0** | **996.6** |

4 KiB calls collapse to about 175–180 MiB/s on a fixed ≈21.7 µs per-call cost;
the 8 MiB numbers are the best observed rates, about 1.32 GiB/s H2C and 0.97 GiB/s
C2H. Both directions are far inside the Gen2 x4 envelope. These are DMA rates on
this link and host, not token rates.

NUMA facts, from sysfs, libnuma (read-only use), raw `mbind`/`get_mempolicy`
syscalls on this process's own buffers, and `/proc/self/numa_maps`:

| Node | CPUs | Distance | MemTotal | MemFree at probe |
| --- | --- | --- | ---: | ---: |
| node0 | `0-21,44-65` (44) | 10 / 21 | 540,902,154,240 B | 449,771,462,656 B |
| node1 | `22-43,66-87` (44) | 21 / 10 | 541,147,926,528 B | 49,041,395,712 B |

The endpoint is on node 0. Default first-touch anonymous pages landed on node 1
(remote from the card); `numa_alloc_onnode(..., 0)` placed every page on node 0,
and `mbind(MPOL_BIND, node 0, MPOL_MF_MOVE)` migrated an already-touched 8193-page
buffer node1 → node0, confirmed by `get_mempolicy` and `numa_maps` — so binding is
demonstrably effective on this host. It did **not** move DMA throughput: H2C
1344.9 (node-1 default) vs 1342.0 MiB/s (node-0 bound), C2H 1118.3 vs 1116.7
MiB/s, deltas of -0.22 % and -0.14 %, inside run-to-run spread. That is an
observation, not a tuning gain, and it says nothing about how host memory should
be placed for real residency.

Executed versus not executed: identity/capacity reads, negotiated link, H2C and
C2H at the declared sizes, NUMA topology/placement and `mbind` were executed.
Not executed: the allocator/bank-local capacity check (needs device memory-write
authorization), a bitstream readback/SHA256, a device-side DDR bandwidth probe, a
`h2d → device-side read → d2h` round trip, and any placement of real expert data
or model execution. Raw JSON and runner sources, with a SHA256 manifest, are kept
in the git-ignored `local/` evidence directory and are not committed; only this
sanitized summary is published.

Minimum facts required before #6 can move: negotiated link (Phase 2), DMA rate for
at least the max expert blob in both directions (Phase 3), NUMA node of the device
(Phase 4), and usable/bank-local capacity (Phase 1 plus an allocator check that is
still unapproved). The first three are now measured (2026-11-23) and the device
reports 4 GiB of two-bank DDR3; usable capacity after reserve/ECC and bank
interleaving still needs the allocator check, which is a device memory-write
operation and remains unauthorized. #6 is therefore not Done: it stays Blocked on
that authorization and on a bitstream provenance record. The acceptance item
“reference-only and measured are separated, capability not used as measured
link/throughput” is satisfied by this document together with the measured table
above.

## 7. Feasibility projection (no wall measurement)

The measured unit in the accepted v2 report is **one expert branch** (gate+up+down
for one expert at batch 1–2, threads 1–2), median-of-medians 1,330,298 ns for IQ2_S
at b1/t1, 2,140,428.5 ns for IQ2_XXS and 2,095,642.5 ns for IQ1_M `[M]`. Those are
not tokens, not decode, and were measured with the CPU probe only.

Declared arithmetic, format-weighted over the 48 layers for one routed expert per
layer:

```
one worker, 1 expert/layer: 34×1,330,298 + 11×2,140,428.5 + 3×2,095,642.5
                         = 75,061,773 ns per token over all layers
full top-10 routing:     × 10 = 750,617,730 ns ≈ 750.6 ms/token ⇒ ≈1.33 tok/s
per-layer serial mean:   750,617,730 / 48 = 15,637,869 ns ≈ 15.64 ms
10 parallel workers:     750,617,730 / 10 = 75,061,773 ns ≈ 75.1 ms/token ⇒ ≈13.3 tok/s
```

The parallel row is a projection that assumes 10 independent workers, no memory
bandwidth limit, no dense/attention time and no sync/merge time. It is not a
decode measurement. Under the architecture decomposition
`T_layer ≈ T_serial + max(T_CPU_experts, T_FPGA_experts) + T_sync_and_merge`
`[D]`, only the CPU-expert term has any measured input; `T_serial` (dense weights
and attention over 48 layers), `T_FPGA_experts` and `T_sync_and_merge` are
unmeasured, so no layer or token wall time can be assembled from this document.

Bandwidth consequences `[P]`:

| Path | Bytes per token | Required rate at 10 tok/s |
| --- | --- | --- |
| Cold experts, full routing, no cache hits | 10 × 35,454,976,000 / 512 = 692,480,000 | ≈6.9 GB/s |
| All persistent fixed weights streamed | 3,071,375,360 | ≈30.7 GB/s (not plausible) |

Two consequences are declared arithmetic only. First, the persistent weights
cannot be streamed at the target rate; they must be resident, which is why S0 is
the reference scenario even though it leaves only 2.67 % of expert tensors
resident. Second, ≈6.9 GB/s of cold-expert traffic per token at 10 tok/s is a
requirement that must be compared against the **measured** negotiated link and DMA
rate from Phases 2–3; the module's capability number cannot satisfy it, and a hot
cache can only lower it by the measured hit rate. 10 tok/s also needs at least
7.51 effective parallel workers (750.6 ms / 100 ms) even if bandwidth were free;
the host has 88 logical CPUs but the device-side memory and link are unknown.

Conclusion for #6 (hardware-free part): on the declared 4 GiB envelope the full
mixed model cannot reside (68.0 GB known storage; 35.45 GB experts; 28.8 GB PLE
table). With all persistent weights resident, ctx-4,096 leaves a ≈0.99 GB ceiling
for 655 max-size expert slots — 2.67 % of expert tensors and 1.36 token-equivalents
of routed applications without reuse — so the CPU cold path can never be assumed
away. The projection is that a parallel CPU cold-expert path is arithmetically
capable of a ≈13 tok/s order of magnitude at 10 workers, but that figure is
conditional on unmeasured bandwidth, dense/attention time and sync, and must not be
published as a result. The Phase 1–4 device facts are measured (§6, “Measured
results”), but the projection still assumes unmeasured device compute and wall
time: no end-to-end model execution exists, so the issue stays Blocked on the
remaining unapproved allocator/bank-local check and bitstream provenance record,
and the reference-only conclusions above are final for this document.

## 8. Limits

- Device identity/capacity, negotiated PCIe, DMA rates and host NUMA placement were
  measured read-only on the reference host on 2026-11-23 (§6, “Measured results”).
  No allocator/bank-interleaving, device-side DDR, bitstream, model or wall-time
  measurement was performed, and no measured DMA rate is converted to tokens/s.
- The receipt is header-only and self-consistent, not independently verified;
  `structural_consistency_only` semantics of the v2 CPU evidence do not transfer
  to it.
- KV dtype (`bfloat16`), MTP cost, KV paging metadata, workspace and fragmentation
  are assumptions or unknowns; every byte table is a declared ceiling.
- “Hot slots” is a byte division, not a cache hit rate, a routing result or a
  Strata placement; the calculator never selects experts.
- 13.3 tok/s is a projection from single-branch kernel medians, not a decode
  measurement; no >10 tok/s claim is made or implied.
- The 28.8 GB PLE table cannot be placed in 4 GiB; how it is handled (host-side
  lookup, per-layer embedding, offload) is a design decision not taken here.
- Gen2 x8 capability is not reported as a negotiable link speed or throughput.