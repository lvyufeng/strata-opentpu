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
[full budget contract v1](resource-budget.md). Section 6 records authorized
device inspections on 2026-11-23 (read-only register/sysfs reads, DMA-size
checks and one synthetic host-DMA capacity write); those measured inputs are
kept separate from the declared byte ceilings below and are not fed back into
the arithmetic. The table bodies remain declared ceilings, and bitstream
provenance, device-side DRAM performance and real placement remain unknown.

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

The hot columns below are ceilings: only the named items are subtracted there,
and the declared reserve in §4 subtracts fragmentation, activation workspace,
DMA ring and metadata before any real placement. MTP state
(`mtp_num_hidden_layers` 1 is declared but its placement and resident cost are
unmeasured), routing metadata and sampler state stay unmodeled and are covered
only by that reserve. The table is thus a **lower bound on required bytes** for
the named items, and the hot space derived from it is an upper bound until the
reserve is applied.

## 4. Device budget scenarios on 4 GiB `[P]`

Envelope: 4,294,967,296 bytes in two declared 2,147,483,648-byte banks. Charged
items: persistent fixed weights or their double-buffered stream, the recurrent
state, KV for the stated context, a double-buffered one-expert DMA staging area
(2 × 1,510,400 = 3,020,800), and for streaming scenarios a double-buffered
largest-layer staging area (2 × 127,660,416 = 255,320,832). “Hot” is the remainder
available for selected expert slots before the declared reserve below, divided by
the per-expert footprint; the slot range spans the cheapest and most expensive
format.
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

**Declared allocation reserve.** A budget that must not over-allocate charges
costs the hot columns omit. Transfer staging is already charged (3,020,800 bytes,
plus 255,320,832 in the streaming scenarios); the remaining items are declared
assumptions, not measurements:

| Reserve item | Declared bytes | Basis |
| --- | ---: | --- |
| Fragmentation and per-allocation padding | 134,217,728 | 128 MiB ceiling, 3.1 % of the envelope |
| Activation workspace and dense scratch | 33,554,432 | 32 MiB declared |
| DMA descriptor ring and sync buffers | 8,388,608 | 8 MiB declared |
| Routing, sampler and metadata | 4,194,304 | 4 MiB declared |
| **Total reserve** | **180,355,072** | 4.2 % of the envelope, subtracted from every hot cell |

Subtracting 180,355,072 bytes from the ctx-4,096 cells gives usable hot space of
809,299,968 (S0), 120,721,408 (S0G), 3,625,354,496 (S1) and 2,936,775,936 (S1G)
bytes. Even after the reserve S0 leaves 536–687 max-size expert slots at ctx
4,096 (≈2.3 % of the 35,454,976,000 expert bytes) and 11–14 at ctx 32,768; S0G
falls to 80–103
slots and its ctx-32,768 cell becomes −672,002,048 bytes; S1 keeps 2,400–3,079
slots at ctx 4,096 and S1G 1,944–2,494. The reserve is a declared ceiling, not an
allocator measurement: hot experts occupy the positive remainder while the CPU
cold-expert path stays mandatory, so the budget keeps Strata's mechanism instead
of assuming full residency. No fragmentation or workspace byte here is measured.

Bank layout. The frozen native layout keeps an expert blob as two allocations —
gate+up (1,049,600 max) and down (460,800) — so a slot may be split across the two
declared banks; per-expert blob maximum 1,510,400 bytes. Every persistent weight
tensor and the state tensors individually fit one bank (largest single resident
item is layer 1 fixed weights at 127,660,416). Measured 2026-11-23: the device
reports two 2,147,483,648-byte DDR3 channels, the host-DMA window spans all
4,294,967,296 bytes contiguously, and every 8 MiB chunk of both raw channel windows
was written and read back byte-exact, including the last 64-byte beat of each
channel (§6, capacity probe). The accelerator's logical interleaved address map
(CAPS bit 7 `CHASH` set; two channels, 64-byte-beat hash interleave as pinned
upstream documents describe) was not exercised, because no device compute ran:
the raw windows are proven, the logical mapping is documented upstream rather
than measured here. The calculator still charges the envelope as one pool, so it
**cannot** prove a bank-local placement; the declared reserve above is what keeps
each “hot bytes” cell from being an over-allocation.

## 5. What is unknown, and what measurement has and has not settled

`[U]` for every item no run has settled: allocator-level usable capacity after
reserve (the raw 4 GiB window is measured and the reserve is declared, but no
allocator has run); largest contiguous allocation and the accelerator's logical
bank interleave; device-side DDR3 bandwidth and latency; activation/workspace, DMA
ring, metadata and sampling buffer bytes; fragmentation and padding; the routing
trace and expert hit distribution; device compute availability and per-layer rate;
MTP placement cost. The calculator prints this list in every report, so no
downstream record can inherit a “fit” claim from it by accident.

Gen2 x8 module capability must never be reported as the negotiated link or as a
throughput. Likewise 655 expert slots is not a cache hit rate: without a routing
trace, an unknown fraction of the 480 per-token expert applications is cold and
must traverse the CPU path.

On 2026-11-23 a read-only run on the reference host replaced part of that `[U]`
list with measured inputs: device-reported capacity and configuration, negotiated
link, both DMA directions at the declared sizes, host NUMA topology and placement
(§6). A second authorized run then wrote and read back synthetic patterns over the
full 4 GiB host-DMA window and both raw channel windows, which settles usable and
bank-local capacity for raw host DMA (§6, capacity probe). Capacity *allocation*
and the logical interleave, device-side DDR bandwidth/latency, bitstream
provenance, workspace/fragmentation, routing trace and device compute availability
remain `[U]`. The measured items are evidence only and none of them is a
throughput or a tokens/s result.

## 6. Hardware measurement plan (executable checklist)

Only read-only operations are listed. Nothing here flashes, resets, reprograms or
reconfigures the device, driver or network, and a Ready Project field is not
authorization. Before running any of it the operator records machine name, board
identity source, allowed read-only command set, evidence directory and timebox in
the #6 comment; any step that needs a write becomes a separate explicit approval.

Execution status 2026-11-23: Phases 0, 1, 2, 4 and the host-side `h2d`/`d2h` part
of Phase 3 were executed on the reference host, and the usable-capacity check
recorded in the measured results below was executed under a separate
device-memory-write authorization. That capacity check is host-DMA evidence only;
it is not a device-side read and not device DDR performance. The `h2d → device-side
read → d2h` part of Phase 3, a device-side DDR probe, a bitstream readback and a
logical-interleave exercise were not executed, so their boxes stay unchecked. The
checkboxes below are the executable plan; the measured-results section states what
actually ran.

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
- [ ] Update the #6 checklist only with facts actually produced by the executed
      phases; do not re-check a planned item. The 2026-11-23 facts move the issue
      out of its former blocked state, but its remaining evidence gaps stay open.
- [ ] Store the budget arithmetic JSON next to the evidence and mark which cells
      changed from ceiling to measured, and which remain `unknown`.

### Measured results — 2026-11-23 (reference host)

Authorization and scope are quoted in the #6 comment trail. Two authorized phases
ran: a read-only inspection (BAR0 register reads through `/dev/xdma0_user`, sysfs
link/NUMA reads, DMA write/read checks of synthetic patterns at declared sizes,
in-process memory placement) and one bounded synthetic host-DMA
write-then-read-back over the full 4 GiB window (the capacity probe below).
Nothing was flashed, reset, reprogrammed or reconfigured, no register was written,
and the driver, kernel command line and network were not touched; the card was
left idle with DRAM zeroed. The host is a 2 × Xeon E5-2696 v4 machine (88 logical
CPUs, 2 socket-aligned NUMA nodes) running Ubuntu 22.04, kernel
5.15.0-203-generic; no hostname, address, login or serial is recorded in this
repository.

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
| `0x080 CFG S` | 1 | one slice |
| `0x084 CFG TMEM_WORDS` | 65536 | tensor memory words |
| `0x088 CFG IMEM_WORDS` | 32768 | instruction memory words |
| `0x08C CFG ACT_BLOCKS` | 128 | activation blocks |
| `0x090 CFG DRAM` | `0x001f0220` | **4,294,967,296 B total, 2,147,483,648 B per channel, 64-byte beats** |
| `0x094 CFG UARCH` | `0x01080232` | systolic MXU, TMEM impl 3, 2 VPU clusters, 8 micro-lanes |
| `0x098 CFG PLATFORM` | 1 | YPCB-00338 platform id |
| `0x09C CFG ARCH_HASH` | `0x4ffd1e17` | equals the `variants.m4` config hash |

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

4 KiB calls collapse to about 164–180 MiB/s (175–180 H2C, 164–180 C2H) on a fixed
≈21.7 µs per-call cost;
the 8 MiB numbers are the best observed rates, about 1.32 GiB/s H2C and 0.97 GiB/s
C2H. Both directions are far inside the Gen2 x4 envelope. These are DMA rates on
this link and host, not token rates.

NUMA facts, from sysfs, libnuma (read-only use), raw `mbind`/`get_mempolicy`
syscalls on this process's own buffers, and `/proc/self/numa_maps`:

| Node | CPUs | Distance | MemTotal | MemFree at probe |
| --- | --- | --- | ---: | ---: |
| node0 | `0-21,44-65` (44) | 10 / 21 | 540,902,154,240 B | 449,771,462,656 B |
| node1 | `22-43,66-87` (44) | 21 / 10 | 541,147,926,528 B | 49,041,395,712 B |

The endpoint is on node 0. Under the default policy, the first-touch anonymous pages
of this unpinned probe process landed on node 1 (remote from the card) — an observed
placement outcome for that run, not a property of the default policy;
`numa_alloc_onnode(..., 0)` placed every page on node 0,
and `mbind(MPOL_BIND, node 0, MPOL_MF_MOVE)` migrated an already-touched 8193-page
buffer node1 → node0, confirmed by `get_mempolicy` and `numa_maps` — so binding is
demonstrably effective on this host. It did **not** move DMA throughput: H2C
1344.9 (node-1 default) vs 1342.0 MiB/s (node-0 bound), C2H 1118.3 vs 1116.7
MiB/s, deltas of -0.22 % and -0.14 %, inside run-to-run spread. That is an
observation, not a tuning gain, and it says nothing about how host memory should
be placed for real residency.

#### Usable-capacity probe (authorized device memory write)

Scope and authorization are quoted in #6 comment `7105781807`, and the runner's
own `authorization` record repeats them: a bounded synthetic host-DMA
write-then-read-back over both raw DDR3 channels across the full 4 GiB window —
not a register write, reset, bitstream or flash operation, and no root. Runner
source sha256 `e36af23522c90d929df59d5df9c5a0a5560ec8a9b3cbcf8bf706447caa4fc7dc`,
result JSON sha256
`c2e53b57183b1a2df08c8a79cd1febdb30c6698eee6721d08da91d78a585a3a3` and log sha256
`4d95dbdac3a79b7b28000dc76f3965ee57942f0efd43259ab890eafed68a8cb7`; the raw files
stay in the git-ignored `local/hw-evidence/` and are not committed. 8,388,608-byte
chunks, 512 calls per 4 GiB phase, host-buffer placement offset 2048 below the
device address on every call, existing `/tmp/otpu` lock files taken and released,
wall 27.89 s (`15:48:23Z–15:48:51Z`), schema
`strata-fpga-capacity-probe-v1`. All six gates are true (`status_idle_pre`,
`regs_unchanged`, `status_idle_post`, `no_foreign_fds_end`, `verification_clean`,
`ok`) with no error or failure records, and identity registers are identical
before and after (`0x000 ID` `0x4f545055`, `0x00C STATUS` `0x68`, `0x040 CAPS`
`0xef0a0eff`, `0x048 BUILD_ID` `0xd84db557`).

| Phase | Bytes | Calls | MiB/s | median µs | p95 µs | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A pattern write | 4,294,967,296 | 512 | 1408.3 | 5,678.4 | 5,757.1 | all 512 chunks written |
| B pattern read-back | 4,294,967,296 | 512 | 1430.4 | 5,568.0 | 5,706.9 | byte-exact, 512/512 chunks |
| C edge beats | 64 each | 4 | — | 84.8–87.2 | — | all four beats verified, neighbours intact |
| D per-channel blobs | 1,510,400 each | 2 | — | — | — | max expert blob verified on each channel |
| E zero restore | 4,294,967,296 | 512 | 1328.5 | 6,028.9 | 6,067.7 | card left fully zeroed |
| F zero verify | 4,294,967,296 | 512 | 1337.1 | 5,965.0 | 6,018.6 | `nonzero_chunks` 0 |

The four 64-byte edges are at device addresses 0, 2,147,483,584, 2,147,483,648 and
4,294,967,232 — the first and last beat of each declared 2,147,483,648-byte
channel window. The two blobs are the largest expert blob (1,510,400 bytes) at
1,048,576 on channel 0 and at 2,148,532,224 on channel 1, each read back to the
expected SHA256 with its neighbouring beat intact before and after the write.
Conclusion: all 4,294,967,296 bytes of the host-DMA window are backed and
writable, both declared channel windows are individually addressable to their last
beat, and the maximum expert blob fits and verifies at a low address on each
channel. That settles usable and bank-local capacity for raw host DMA; it does
**not** settle the accelerator's logical interleaved address map, device-side DDR
bandwidth/latency (no `h2d → device read → d2h` round trip), allocator
fragmentation, or any model placement, so those stay withheld and unclaimed. The
MiB/s cells are host-DMA rates on this link and host, not token rates.

Executed versus not executed: identity/capacity reads, negotiated link, H2C and
C2H at the declared sizes, NUMA topology/placement and `mbind`, and the authorized
allocator/bank-local capacity write/read-back over the full 4 GiB window were
executed. Not executed: a bitstream readback/SHA256, a device-side DDR bandwidth
probe, a `h2d → device-side read → d2h` round trip, an accelerator-visible
logical-interleave exercise, a real allocator or expert placement, and any model
execution. Raw JSON and runner sources, with a SHA256 manifest, are kept in the
git-ignored `local/` evidence directory and are not committed; only this
sanitized summary is published.

Minimum facts required before #6 can move: negotiated link (Phase 2), DMA rate for
at least the max expert blob in both directions (Phase 3), NUMA node of the device
(Phase 4), and usable/bank-local capacity (Phase 1 plus an allocator check). All
four are now measured (2026-11-23): Gen2 x4 negotiated, DMA at the declared sizes
in both directions, node 0, and the full 4 GiB window written and read back
verified. The one operation that kept #6 Blocked — the allocator/bank-local
capacity check that needed device memory-write authorization — was executed under
the recorded #6 scope and is discharged. Bitstream provenance remains an evidence
gap, and device-side DRAM performance, the logical interleave, real
workspace/fragmentation and placement remain unmeasured and are not claimed. The
requirement that reference-only facts and measured facts stay separated, and that
a module capability is never presented as a measured link speed or throughput, is
met by this document together with the measured tables above.

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
cache can only lower it by the measured hit rate. The comparison is now concrete:
the negotiated link is Gen2 x4, measured host DMA peaks are ≈1.32–1.41 GiB/s H2C
and ≈0.97–1.09 GiB/s C2H at 127,660,416–8,388,608-byte transfers, far below the
≈6.9 GB/s streaming requirement, so cold experts cannot be streamed from host
memory at the target and must run from resident device data or on the CPU. 10 tok/s
also needs at least 7.51 effective parallel workers (750.6 ms / 100 ms) even if
bandwidth were free; the host has 88 logical CPUs but the device-side memory and
link are insufficient for streaming.

Conclusion for #6 (hardware-free part): on the declared 4 GiB envelope the full
mixed model cannot reside (68.0 GB known storage; 35.45 GB experts; 28.8 GB PLE
table). With all persistent weights resident, ctx-4,096 leaves a ≈0.99 GB ceiling
for 655 max-size expert slots before the declared 180,355,072-byte reserve,
2.67 % of expert tensors and 1.36 token-equivalents of routed applications without
reuse, so the CPU cold path can never be assumed away. The projection is that a
parallel CPU cold-expert path is arithmetically capable of a ≈13 tok/s order of
magnitude at 10 workers, but that figure is conditional on unmeasured bandwidth,
dense/attention time and sync, and must not be published as a result. The Phase 1–4
device facts are measured (§6, “Measured results”), including the authorized
write/read-back of the full 4 GiB host-DMA window, but the projection still assumes
unmeasured device compute and wall time: no end-to-end model execution exists, so
no throughput claim follows and the target stays unmet. #6 stays open on the
remaining evidence gap — bitstream provenance — and on unmeasured device-side DRAM
performance, logical interleave and real allocator placement; its former Blocked
condition (the allocator/bank-local capacity authorization) is discharged. The
reference-only conclusions above are final for this document.

## 8. Limits

- Device identity/capacity, negotiated PCIe, DMA rates, host NUMA placement and the
  authorized synthetic host-DMA write/read-back of the full 4 GiB window were
  measured on the reference host on 2026-11-23 (§6). No logical-interleave or
  device-side DDR exercise, device allocator run, bitstream, model or wall-time
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