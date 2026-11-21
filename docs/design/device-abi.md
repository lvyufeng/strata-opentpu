# Proposed Strata ↔ openTPU device ABI, revision 0

**Status: design plus executable conformance model; not an implemented transport,
FPGA backend, inference path, compatibility certificate or performance result.**
Tracked by [central Issue #7](https://github.com/lvyufeng/strata-opentpu/issues/7).
The [stdlib conformance model](../../runtime/abi_contract.py) checks host-side
protocol invariants using synthetic events. It never opens a device, pins memory,
performs DMA, converts tensors, executes a model or restores checkpoint bytes.
The [adversarial tests](../../tests/test_abi_contract.py) are specification tests,
not RTL simulation, hardware validation or numerical inference tests.

## Ownership and source evidence

Locked inputs are Strata `3142b66b96e9c0e775e30f620b11b191a1427c2a` and openTPU
`d84db55712aba0d40811d7762cb052f05d55b7e5`. These are inspected source inputs,
not a validated inference combination. Existing implementations stay in their
forks: Strata owns model execution, routing, CPU kernels, hot/cold expert cache,
MTP verification/acceptance and its FPGA backend. openTPU owns ISA/RTL, packed
kernel execution, device runtime, DMA and synchronization. This repository owns
version/profile orchestration, contracts and evidence; it must not grow a second
model engine or board driver.

| Inspected existing mechanism | Exact locked source | Integration gap / proposed requirement |
| --- | --- | --- |
| Board host has XDMA transport, logical channel mapping and an Engine backend; it is not only the simulator runtime | [openTPU board overview:1–35](https://github.com/lvyufeng/openTPU/blob/d84db55712aba0d40811d7762cb052f05d55b7e5/opentpu/host/board.py#L1-L35) | Reuse board/runtime in openTPU; no claim that its Engine backend executes the complete Strata model |
| Configuration identity hash and target/platform/variant checks | [check_hash/check_target:2240–2305](https://github.com/lvyufeng/openTPU/blob/d84db55712aba0d40811d7762cb052f05d55b7e5/opentpu/host/board.py#L2240-L2305) | New integration version negotiation and exact deployment artifact identity; older/zero CFG_HASH is not sufficient proof |
| HALTED is not DRAM write visibility; wait explicitly checks WR_IDLE | [Board.wait:1850–1876](https://github.com/lvyufeng/openTPU/blob/d84db55712aba0d40811d7762cb052f05d55b7e5/opentpu/host/board.py#L1850-L1876) | Adapter completion must also cover all relevant H2D/D2H and host visibility, not just core halt |
| Prior holder quiescence, CTRL reset and QUIET drain; abort suppresses stop errors | [quiesce/stop/abort:1177–1243](https://github.com/lvyufeng/openTPU/blob/d84db55712aba0d40811d7762cb052f05d55b7e5/opentpu/host/board.py#L1177-L1243) | A timeout exception alone is not successful drain evidence; recovery must preserve uncertainty |
| Simulator-facing Weight takes FP32 `[N,K]` and enumerated MXU formats; DRAM alignment 128 | [runtime:25–69](https://github.com/lvyufeng/openTPU/blob/d84db55712aba0d40811d7762cb052f05d55b7e5/opentpu/runtime.py#L25-L69) | Not an API to consume arbitrary Strata GGUF bytes; alignment is negotiated, not universally inferred from this constant |
| Low-bit formats have distinct element/scale/block semantics | [quant format definitions:1–26](https://github.com/lvyufeng/openTPU/blob/d84db55712aba0d40811d7762cb052f05d55b7e5/opentpu/quant.py#L1-L26) | Explicit packed signatures; no name/bit-width equivalence |
| Session owns residual, QSA, GDN and expert handoff | [Strata session:1–9,29–48](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/session.hpp#L1-L48) | Opaque Strata-owned session/checkpoint inventory, not duplicated numeric state |
| QSA KV variants, paged/streamed/ring storage, indexer and fixed-address pinned graph inputs | [QsaState:203–273](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/layer.hpp#L203-L273) | Page maps, scales, history and captured addresses have lifetimes beyond a single submission |
| Verify appends speculative KV/indexer rows, defers GDN, then commits accepted prefix and restores/replays state | [verify state contract:13–18](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/verify.hpp#L13-L18) | Candidate-ready completion is never acceptance; preserve Strata's accepted-prefix semantics |
| Commit can return before its graph; cross-stream/host readers require wait_commit | [async commit:324–331](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/verify.hpp#L324-L331) | Committed checkpoint publication/fallback requires the actual commit fence, not host enqueue success |
| Expert exchange ownership flips only when copies land, with additional asynchronous staging paths | [exchange:820–860](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/expert_source.hpp#L820-L860) | Preserve cache generations and copy dependencies; do not publish a new residency table over old bytes |

These anchors establish the inspected mechanisms, not a global transaction API
already present in either fork. The transaction, epoch, timeout/fallback and wire
requirements below are **proposed integration requirements**.

## 1. Negotiation and identity (fail closed)

Before any writable allocation or submission, exchange:

- Integration ABI major/minor; incompatible major fails. Device minor must cover
  the requested host minor. New optional capabilities never silently change old
  semantics. Unknown required flags/operations/formats fail. Wire encoding,
  reserved fields and backwards compatibility need a reviewed fork implementation.
- Exact target/variant and board/device identity, register-map version, CFG_HASH,
  build ID, reviewed bitstream SHA256, host runtime and locked source revisions.
  CFG_HASH is configuration identity, not a cryptographic bitstream digest.
  Identity readback and artifact provenance are separate checks. The mock compares
  supplied values only: **it does not identify or authenticate hardware**.
- Operations, exact format/layout/numerics signatures, supported state components,
  address/offset alignment, memory domains, allocation/range/queue limits,
  synchronization/reset/drain facilities and error reporting. Absence is not
  support. MTP must stay off unless the complete state/commit contract is available.
- Numeric manifest: rounding/saturation/FTZ, accumulation precision, scale encoding,
  reduction ordering and permitted error metrics. Capabilities belong in the
  respective forks; the integration consumes a descriptor, not copied kernels.

The Python model requires target/configuration/bitstream match and compatible ABI,
valid alignment/limits, operation allowlisting and exact Format equality. It does
not encode a wire structure, probe register maps, negotiate a real device, or
compare software revisions. It implements host ABI 1.0 semantics only, even when
the supplied device offers a newer compatible minor. Its identity strings are
synthetic in tests. `max_buffer_bytes` limits each allocation, **not aggregate
residency**: this model does not validate a 4 GiB device-memory budget, available
physical memory, pinned-host memory quota or cache placement feasibility (Issue #6).

## 2. Buffers, ownership and DMA lifetime

Opaque handles contain connection identity, device epoch and monotonically unique
allocation generation. Raw pointer/address integers must not cross this boundary
as ownership tokens. After release/reset, old handles must fail; another connection
cannot supply a valid local handle. Real code additionally owns allocation bounds,
IOMMU/pinning/mapping lifetimes and alias detection.

A descriptor specifies memory domain, capacity, access permissions and alignment.
A tensor binding adds exact format, shape, offset and read/write/readwrite access;
strides/layout must be explicit. Validate overflow, nonempty dimensions, last-axis
block divisibility, packed byte size, offset alignment and bounds **before** queueing.
The model supports contiguous packed rows only, arbitrary-precision byte arithmetic,
and exclusive whole-buffer reservations (even read/read sharing is serialized).
It does not model suballocations, arbitrary strides or physical aliases.

Submission retains **both source and destination** for the full H2D/D2H transfer,
and all compute operands until terminal accesses drain. Host mutation, host read
of a pending D2H result, remapping, release, expert-slot reuse and residency-table
publication cannot race those retained references. A real read-only input may be
shared only with tracked reader dependencies. Captured graph addresses and borrowed
RoPE/cache tables can outlive command completion and require separate leases in
Strata; the mock does not simulate those captures or leases.

H2D/D2H are byte-preserving transfers, not implicit quantization/transposition.
Device compute consumes device bindings. A completion fence releases the command's
reservation only after hardware and DMA accesses stop and the promised domain has
visibility. Core halt, queued transfer, posted store and cancellation request are
not sufficient. CPU access to device memory is never implied.

## 3. Format and quantization contract

Each packed format identifies the complete encoding: element codes, block/group
size, block bytes including scales/metadata, scale encoding, row/column order,
interleave/shard rules, byte order, layout revision and numeric semantics. Mixed
GGUF tensor types must be checked **per tensor**, not only by model-pack label.

**IQ2_XS is not FP4. Q2_0 is not INT4. IQ3_S is not another spelling of FP4.**
The requested primary IQ2_XS, speed Q2_0 and quality IQ3_S paths require either
exact fork support or an explicitly approved, measured conversion experiment.
Never reinterpret packed bytes because their bit-width is similar. Unsupported
formats fail before DMA; CPU execution of supported original bytes remains a
Strata decision only after safe fallback. Conversion changes storage, accuracy,
transfer volume and throughput; it needs its own correctness evidence. The
mock ships **no real format capability table**. Tests use F32 metadata and a
clearly synthetic FP4 descriptor, not a certified openTPU layout.

## 4. Submit, completion, fences and errors

A command carries connection/epoch/sequence identity, operation, retained bindings,
optional transaction/base version and explicit predecessor fences. Validate all
fields atomically before acquiring refs/queueing. Failed preconditions must not
partially submit work. Queue-full is a retryable pre-submit condition, not an
unknown device execution result. Exact wire/status numeric values remain TBD.

Completion echoes epoch/sequence and reports success, terminal error or acknowledged
cancellation, with transport/device diagnostic details. Duplicate, foreign, stale
or unknown completions must fail without releasing current allocations. Successful
fences are prerequisites for dependent consumers; a failed fence cannot satisfy a
normal data dependency. In the model `complete(..., visible=True)` is **injected
proof** that all accesses are drained and visible, including error/cancel paths.
It is not an implemented fence. Until such proof arrives, refs remain pinned.

Real errors need distinct categories: negotiation/unsupported descriptor, host
resource exhaustion, DMA failure, device execution/numeric error, wait timeout,
request-cancel pending, cancellation acknowledged, quiescence/reset failure and
state reconstruction failure. The model intentionally reduces host violations to
`ContractError` and terminal outcomes to `ok/error/cancelled`.

## 5. Session state, MTP accepted frontier and cache coherence

The [operator and state audit](../audits/strata-execution.md) is the source-backed
constraint inventory for which operations/state may be offloaded; this ABI does
not supersede it or certify placement. Strata is the sole authority for the committed token frontier and verification.
A transaction references a session, committed version, base frontier, checkpoint
identity, candidate row count and complete state inventory: KV, QSA indexer,
DeltaNet recurrent and convolution state, n-gram/PLE token history, residual/draft
state, and sampler seed/draw counter/token and repetition history. Sampler state
may remain host/Strata-owned; logical inventory does not mandate a device copy.
Rejected speculative sampling must not advance committed RNG/history except as
specified by Strata's acceptance algorithm. Actual persistent vs scratch ownership
is decided by Strata's execution path; inventory does not require copying every
scratch buffer into a checkpoint.
The mock uses opaque identities and one active transaction per session, not tensor
contents or checkpoint serialization.

1. Begin from a valid committed checkpoint. All state-affecting transfers and
   compute must join its transaction. Stateless unrelated compute must not mutate
   session state. A real adapter must enforce this association; the mock cannot
   inspect tensor meaning or detect a caller falsely declaring work stateless.
2. Speculative work produces candidate rows only. Device/DMA completion makes them
   visible but **does not emit accepted tokens or advance committed state**.
3. Strata verifies drafts with the original route/top-k semantics. Commit accepts
   a prefix `1 <= accepted_rows <= candidate_rows`, never arbitrary suffix rows.
   Zero accepted advancement uses rollback in this model; a real verifier's input
   row/bonus-token counting must be mapped explicitly, not inferred from this name.
4. Retire all related transfers, compute and Strata's commit/replay graph; restore
   rejected indexer/history tails and handle KV overwrite-before-read semantics.
   Publish one coherent new checkpoint/version/frontier only after that commit
   fence. The mock's `commit(..., checkpoint)` treats the supplied checkpoint as
   injected evidence that this work completed; it does **not** run that graph.
5. Rollback requires quiescence, then Strata restores/reconstructs the base state.
   The model's rollback is a logical injected restore, not a byte copy. Cache slot
   generation and resident tables must agree with landed expert bytes. Existing
   exchange/adaptive-cache mechanisms remain in Strata, not duplicated here.

No partial token/state publication on an uncertain failure. Checkpoint ownership
and cross-stream events must cover QSA/indexer, GDN replay, PLE/ngram context and
MTP drafter/catch-up state together. Real atomicity/recovery is unimplemented.

## 6. Timeout, cancellation, reset and CPU fallback

A wait deadline expiring says only that the host stopped waiting. Likewise a cancel
request says nothing about device acknowledgement. Retain refs and quarantine the
candidate. The model poisons a transaction on timeout/request-cancel even if a late
success arrives, requiring drained rollback rather than guessing which writes won.
A terminal error/cancelled completion also requires rollback before reuse.

Recovery choices:

- Receive matching terminal visibility/drain evidence; restore the base checkpoint,
  then permit Strata CPU fallback. No CPU and FPGA path may race writes into the
  same session/cache. Fallback must preserve routing, all experts and accepted-token
  accounting; it must not masquerade as a successful FPGA command.
- If terminal completion cannot be obtained, obtain operation-specific authorization
  where required, stop the core **and DMA engines**, verify all old writes drained,
  then advance the epoch and invalidate device handles/fences/transactions/state.
  Re-identify/re-negotiate and restore/reconstruct a checkpoint before use. A host
  epoch counter increment alone is not a reset or quiescence proof.
- If stop/drain/reset/restore fails, remain quarantined; do not launch CPU fallback,
  recycle memory or continue the session. Report a concrete recovery failure.

`reset(proven_quiescent=True)` injects external global quiescence proof, clears all
handles and invalidates every session. `restore_checkpoint` injects reconstruction
proof. These booleans/strings are not production security or safety mechanisms.
The model's fallback check conservatively requires **all commands** drained, even
unrelated ones, and no active transaction. A fork implementation can relax this
only with proven isolated ownership. Actual openTPU stop/wait code is useful
implementation evidence, but timeout/abort exceptions do not establish that every
required safety condition succeeded.

## 7. Reference resource and placement constraints (not measured)

The [reference board descriptor](../../configs/boards/kintex7-reference.json)
records `xc7k480tffg1156-2`, 4 GiB DDR3 and Gen2x8 endpoint capability with unknown
device identity and programming transport. The host reference
is dual E5-2696v4, 44 physical/88 logical threads, AVX2/FMA and 1 TiB RAM.
These are planning inputs, not hardware observations made by this work. A reported
Gen2x8 capability and historical Gen2x4/H2D approximately 1.39 GB/s are not current
link or bandwidth measurements. Do not insert them as negotiated performance.

Before a placement experiment, calculate and enforce the **aggregate** device
budget: resident expert bytes + activation/workspace + KV/indexer/state + DMA
staging + alignment/reserved firmware allocations <= usable device DRAM, not just
nominal 4 GiB. Admit each mixed-format tensor by its real packed size. Host cache
and pinned staging have separate quotas; unbounded pinning is not justified by
1 TiB capacity. The current model enforces per-buffer bounds only and cannot
certify that a placement fits. Keep this gate in the budget/placement workstream.

NUMA placement must identify the actual PCIe root/socket, place pinned DMA staging
and transfer worker there, and assign CPU expert workers/cache pages explicitly.
Remote-socket traffic and page migrations belong in wall-time accounting. Never
infer locality from CPU count or silently reduce top10/skip experts to fit. Use
full routing and expert-skip=0; separate MTP-off/on experiments and accepted output.
Any >10 tok/s target needs measured complete decode wall time including CPU work,
H2D/D2H, fences, MTP verification/commit, cache misses and fallback. Device-only
rates or bandwidth arithmetic cannot establish it. No NUMA affinity, allocations
of this size, link probing, device access or throughput measurement occur here.

## Validation and remaining implementation gates

Run the hardware-free specification suite from the repository root:

```sh
python3 -m unittest discover -s tests -p 'test_abi_contract.py' -v
python3 -m unittest discover -s tests -v
```

Adversarial coverage: incompatible ABI/target/config/digest, unknown exact formats,
IQ2_XS/FP4 non-aliasing, wrong scale/layout/numerics, access/alignment/range failures,
DMA source/destination retention, in-flight release/reuse, stale/foreign/duplicate
completions, missing/failed visibility fences, MTP accepted-prefix bounds and no
completion-driven commit, incomplete state inventory, late success after timeout,
cancel request vs acknowledgement, epoch invalidation, required restore and unsafe
CPU fallback. All tests are stdlib-only synthetic protocol events.

Still required in reviewed fork work: concrete binary/API encoding and capability
implementation, actual handles/pinning/DMA fences, state component mapping and
checkpoint/restore proofs, numerical/layout golden vectors for every mixed GGUF
format, whole-model CPU correctness, FPGA operator correctness, hardware reset and
failure injection, cache/MTP equivalence and end-to-end wall-time measurements.
No hardware/model access, installation, upstream scripts or fork edits are needed
or authorized by these tests. There is no inference-ready claim or >10 tok/s result.
