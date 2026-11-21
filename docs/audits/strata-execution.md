# Strata whole-execution audit — Issue #4

**Status: unconfirmed static analysis.** This is a source audit, not a working CPU+FPGA engine, model validation, hardware result or throughput claim.

## Inputs and scope

- Strata fork: exact SHA `3142b66b96e9c0e775e30f620b11b191a1427c2a`, as recorded in the [source lock](../../dependencies.lock.json).
- Integration openTPU SHA: `d84db55712aba0d40811d7762cb052f05d55b7e5`. This audit does not certify openTPU implementation support; no openTPU source assertions are made.
- Machine-readable operator/format evidence: [audit manifest](../../data/strata-execution-audit.json). Each operator records placement, shape, layout, lifetime, synchronization, state and exact pinned line anchors.
- Scope: compiled Flash-Next text trunk, prefill, single-token decode, CPU cold experts, resident expert/cache interaction, PLE, head/sampling, MTP, verification and accepted-prefix commit. Different model families, multimedia and every optional backend/flag combination are not claimed exhaustive.
- No model artifacts were downloaded or inspected. No dependency source was modified. No build, upstream setup script, inference, device operation or benchmark was run for this audit.
- Validation performed: both source-anchor checker modes passed (25 operator entries, 87 anchors including format rows), including exact pinned HEAD/clean checkout and source bounds; scoped whitespace check passed. This is citation consistency only, not semantic or runtime validation.

## Conclusion and preserved reuse boundaries

Strata is not just a CPU expert library. The current execution path uses the GPU for the dense/stateful trunk, routing, shared experts, hot experts, head, sampling, prefill, MTP and verification. CPU expert implementations **do not imply complete CPU-only decode**, and CPU-only kernel-test targets do not change that conclusion. Replacing only routed expert compute cannot make this a CPU+FPGA engine.

Keep Strata's hot-resident/CPU-cold split, profile/adaptive cache, native pack loaders, state ownership and MTP acceptance logic. Extend the Strata fork for model/backend/NUMA work. Implement low-bit RTL/ISA, device runtime, DMA and synchronization in the openTPU fork. This integration repository should hold version locks, orchestration and evidence, not a duplicate model engine.

The non-expert backend gap is a required central follow-up after Issue #4. Use the operator inventory below as its acceptance checklist. For **each** operator, assign CPU/FPGA/retained-GPU placement, implement or explicitly reject it, preserve numerical and state semantics, and demonstrate parity before performance. This includes cheap-looking norms, gates, RoPE, sparse selection, PLE history, final mix and sampling, not just GEMV. If GPU is excluded, the current verifier and MTP dependencies need an explicit replacement plan; they cannot be enabled by changing a configuration boolean.

## Geometry and dispatch are contracts, not observed model metadata

Default geometry is hidden 2560, vocabulary 248320, 48 layers (QSA layers 3,7,...47; 36 GDN), four hyperconnections/rank320, 512 experts/top10, FFN640/shared640, RMS epsilon1e-6. GDN uses S128, 16 key heads/48 value heads, convolution width4/channels10240, value dimension6144. QSA uses24 query/2 KV heads,width256,RoPE64,indexer4x128,block4,budget2048. Alternative compiled geometry exists. Validate the actual model header against the selected build; do not infer it from a pack name.

## Whole-path operator inventory

All devices below describe current source placement, **not proposed FPGA placement**. “Persistent” means sequence/model lifetime, not permission to retain rejected speculative state.

### geometry: Compiled Flash-Next geometry

- **Placement:** host validation / GPU kernel contract.
- **Shape:** 48 layers: 36 GDN, 12 QSA; H=2560; vocab=248320; experts=512, top-k=10, FF=640; hyperconnections=4, rank=320.
- **Layout:** QSA layers 3,7,...47; configurable build geometry is not proof of an inspected model.
- **Lifetime:** Model/build lifetime.
- **Ordering / synchronization:** Validate actual GGUF geometry against chosen compiled geometry before allocation.
- **State / constraints:** Dimensions are defaults, not observed model metadata.
- **Pinned evidence:** [geometry.hpp:20–89](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/geometry.hpp#L20-L89).

### embedding: Token embedding gather

- **Placement:** GPU; native embedding may read mapped host memory.
- **Shape:** token id -> H=2560.
- **Layout:** Native GGUF gather or canonical S2/S4/S8 planes; canonical IQ4_NL explicitly rejected; native placement configurable.
- **Lifetime:** Weights persistent; output per token/window.
- **Ordering / synchronization:** Gather on ordered stream; retain mapped host storage until GPU reads complete.
- **State / constraints:** Read-only weights; writes token activation.
- **Pinned evidence:** [layer.cpp:1336–1388](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L1336-L1388); [native_head.hpp:50–68](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/native_head.hpp#L50-L68).

### hyperconnection: RMS normalization, gated residual read/write and injection

- **Placement:** GPU.
- **Shape:** R=[4,2560], mixed=[2560], low rank=320.
- **Layout:** Float residual; F32 norm; BF16-in-F32 down/up/inject forms checked; fused path may defer prior FFN write.
- **Lifetime:** R persistent; mixed and inject survive pre -> post; scratch shared in ordered execution.
- **Ordering / synchronization:** Pre/attention/FFN/post must remain ordered; do not reuse mixed/inject while CPU experts are in flight.
- **State / constraints:** Updates multi-stream R; fused deferred write is part of state frontier.
- **Pinned evidence:** [layer.cpp:1534–1587](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L1534-L1587); [layer.cpp:1622–1640](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L1622-L1640).

### gdn-projection: GDN normalization/projections and activation quantization

- **Placement:** GPU.
- **Shape:** H2560 -> QKV10240; q2048,k2048,v6144; alpha/beta48; output6144 ->2560.
- **Layout:** Per-tensor native/canonical activation choice; Q8_K and Q8_0 canonical scratch can both be needed; BF16 native path has distinct rounding.
- **Lifetime:** Dense weights persistent; projection scratch per layer call/window.
- **Ordering / synchronization:** Ordered projection -> convolution -> recurrence -> output; retain recorded verifier inputs for accepted-prefix replay.
- **State / constraints:** Writes transient q/k/v, alpha/beta, z.
- **Pinned evidence:** [layer.cpp:251–315](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L251-L315); [layer.cpp:320–345](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L320-L345); [layer.cpp:371–383](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L371-L383).

### gdn-convolution: Causal convolution and SiLU

- **Placement:** GPU.
- **Shape:** 10240 channels, conv width4; split q/k/v only after convolution.
- **Layout:** Persistent C*(d_conv-1) history; fused native preconditions include conv4 and S128; legacy D2D/SiLU fallback.
- **Lifetime:** One convolution history per GDN layer; output scratch reused only after consumers.
- **Ordering / synchronization:** Convolution before split and q/k L2 normalization; verifier must advance only accepted history.
- **State / constraints:** Mutates convolution history on token path; speculative window commit has separate rules.
- **Pinned evidence:** [layer.cpp:275–311](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L275-L311); [layer.hpp:60–95](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/layer.hpp#L60-L95); [verify.hpp:13–18](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/verify.hpp#L13-L18).

### gdn-recurrence: Gated DeltaNet recurrence

- **Placement:** GPU.
- **Shape:** S128; 16 key heads,48 value heads; state=128*48*128 float elements/layer.
- **Layout:** Transposed internal state (S,h_v,S), not reference (S,S,h_v); q/k L2 normalization, not v.
- **Lifetime:** 36 distinct persistent recurrent states; common scratch does not mean shared state.
- **Ordering / synchronization:** Single-token recurrence ordered; verifier defers recurrent mutation except eligible self-commit path.
- **State / constraints:** Updates GDN recurrent state; accepted-prefix replay is required.
- **Pinned evidence:** [layer.hpp:8–32](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/layer.hpp#L8-L32); [layer.hpp:60–95](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/layer.hpp#L60-L95); [layer.cpp:296–352](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L296-L352); [session.cpp:808–815](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/session.cpp#L808-L815); [verify.hpp:13–18](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/verify.hpp#L13-L18).

### gdn-output-gate: GDN output RMS norm, norm weight, sigmoid gate and output projection

- **Placement:** GPU.
- **Shape:** o/z/y6144 -> H2560.
- **Layout:** rms_norm(o)*ssm_norm*sigmoid(z); this output gate is NOT SiLU.
- **Lifetime:** Transient gated output and quantization scratch.
- **Ordering / synchronization:** Recurrence output must be ready before gate and projection.
- **State / constraints:** No separate persistent state beyond preceding GDN state.
- **Pinned evidence:** [layer.cpp:360–383](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L360-L383).

### qsa-projection: QSA query/gate, KV and indexer projections, norm and RoPE

- **Placement:** GPU.
- **Shape:** 24 query heads x256;2 KV heads x256; query+gate=24*2*256; indexer4x128; RoPE64.
- **Layout:** Query/gate half-split with 2D copy; Q/K normalization and RoPE; weight type chosen per tensor.
- **Lifetime:** Projection scratch per call; RoPE tables persistent/possibly borrowed.
- **Ordering / synchronization:** Projection and rotation before KV/index append and attention.
- **State / constraints:** Advances KV/index state via following operators, not free-standing matmul.
- **Pinned evidence:** [layer.cpp:1240–1263](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L1240-L1263); [layer.hpp:203–211](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/layer.hpp#L203-L211); [layer.hpp:246–252](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/layer.hpp#L246-L252); [geometry.hpp:38–46](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/geometry.hpp#L38-L46).

### kv-storage: QSA KV append, streaming and page resolution

- **Placement:** GPU + optional authoritative host pools.
- **Shape:** [page][kv_head][page_size][head_dim];2 KV heads x256.
- **Layout:** FP16; INT8 + FP16 scale/64; rotated Q4_0 144 B/cell/head; hybrid K8V4 816 B/cell; hybrid mode0 only, MTP ring remains INT8.
- **Lifetime:** Persistent per QSA layer/sequence; pinned staging and page mappings outlive asynchronous reads.
- **Ordering / synchronization:** Resolve selected blocks before attention; streamed mode host owns authoritative bytes; elastic growth requires no concurrent device work.
- **State / constraints:** Mode0 all VRAM; mode1 host+VRAM slots; mode2 block%n_slots ring+host resume; rejected speculative cells overwritten before reuse.
- **Pinned evidence:** [layer.hpp:203–249](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/layer.hpp#L203-L249); [layer.hpp:263–286](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/layer.hpp#L263-L286); [layer.hpp:288–299](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/layer.hpp#L288-L299); [layer.cpp:1275–1305](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L1275-L1305); [verify.hpp:13–18](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/verify.hpp#L13-L18).

### indexer: Indexer key append/pooling, scoring and sparse top-k selection

- **Placement:** GPU.
- **Shape:** query4x128; one pooled key/4 cells; budget2048 cells; tail[3,128].
- **Layout:** Persistent float tail/pooled rows, block position and constant dead key; capacity-aware graph grids; native and fallback append/selection.
- **Lifetime:** Tail, pooled index and block cursor persist across tokens.
- **Ordering / synchronization:** Append -> score/top-k -> resolve -> gather/attention; commit restores tail then re-appends accepted keys.
- **State / constraints:** Speculative appends are not committed history; restore rejected-tail contamination.
- **Pinned evidence:** [layer.hpp:244–249](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/layer.hpp#L244-L249); [layer.cpp:1240–1277](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L1240-L1277); [verify.hpp:13–21](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/verify.hpp#L13-L21).

### qsa-attention: Sparse attention, optional rotation, sigmoid gate and output projection

- **Placement:** GPU.
- **Shape:** 24x256 attention output=6144 ->2560.
- **Layout:** Fast pool attention or gathered FP16/INT8/Q4/hybrid paths; optional FWHT and inverse; short native adapter ctx<=256; native F32 versus canonical FP64 sigmoid gate.
- **Lifetime:** Attention workspace per call/window; KV persistent.
- **Ordering / synchronization:** Selected block resolution before read; inverse rotation before gated output projection.
- **State / constraints:** Reads visible KV prefix; writes transient attention/output.
- **Pinned evidence:** [layer.cpp:1275–1321](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L1275-L1321).

### router: MoE router, softmax, stable top-k and renormalization

- **Placement:** GPU -> mapped pinned host publication.
- **Shape:** 2560 ->512 logits; top10 ids and weights.
- **Layout:** BF16 router; softmax over all experts, descending stable ties by index; native specialization512/top10.
- **Lifetime:** Weights persistent; activation/ids/weights in mapped doorbell until host pool consumes.
- **Ordering / synchronization:** GPU publishes x/ids/weights then rings sequence last on ordered stream; host load fence precedes reads.
- **State / constraints:** Full routing required: skip=0, all top10; thresholds/skip paths are not acceptance defaults.
- **Pinned evidence:** [layer.cpp:405–444](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L405-L444); [layer.cpp:1326–1334](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L1326-L1334).

### shared-expert: Shared expert gate/up, SiLU multiplication, down and scalar sigmoid gate

- **Placement:** GPU.
- **Shape:** 2560 ->640 ->2560, plus scalar gate.
- **Layout:** Each gate/up/down tensor selects its own form/planes and BF16/Q8_K/Q8_0 activation; scalar gate BF16 form required.
- **Lifetime:** Weights persistent; one workspace reused in stream order.
- **Ordering / synchronization:** Can run after doorbell publication while CPU cold experts execute; must finish before final combine.
- **State / constraints:** No independent recurrent state.
- **Pinned evidence:** [layer.cpp:468–507](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L468-L507); [layer.cpp:1584–1587](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L1584-L1587).

### cpu-experts: CPU cold routed expert gate/up, SiLU product and down

- **Placement:** CPU host-RAM kernels / ggml fallback.
- **Shape:** Each selected expert:2560 ->640 ->2560; multi-token groups reuse weights.
- **Layout:** Native GGUF types retain metadata; IQ AVX2 and AVX512 are distinct runtime tiers; activation quantizer follows native format; not all kernels share rounding.
- **Lifetime:** Host expert blobs persistent or staged; stage buffers claimed through read joins and all compute batches.
- **Ordering / synchronization:** Pool partitions rows, not partial row sums; cold-read overlap must join before releasing buffers.
- **State / constraints:** Only routed cold experts; this is NOT a complete CPU-only decode engine.
- **Pinned evidence:** [native_expert.cpp:559–617](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cpu/native_expert.cpp#L559-L617); [native_expert.cpp:648–707](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cpu/native_expert.cpp#L648-L707); [expert_source.cpp:5564–5629](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/expert_source.cpp#L5564-L5629).

### resident-experts-combine: Hot resident expert execution and weighted MoE/shared reduction

- **Placement:** GPU + CPU result transport.
- **Shape:** top10 outputs ->2560; GPU hits and CPU misses.
- **Layout:** Residency lookup selects hit slots; GPU can zero hit rows while mapped CPU outputs carry misses; weighted combination remains GPU-owned.
- **Lifetime:** Cache slots persistent; miss output and staging scratch survive until GPU consumer is done.
- **Ordering / synchronization:** Launch hits before CPU pool; wait/publish miss completion before combine; shared expert can overlap but must join.
- **State / constraints:** Numerical order and exact mode depend on path; do not replace with unvalidated host summation.
- **Pinned evidence:** [session.cpp:726–797](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/session.cpp#L726-L797); [expert_source.cpp:5564–5597](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/expert_source.cpp#L5564-L5597); [layer.cpp:1622–1640](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L1622-L1640).

### expert-cache: Profile hot-tier fill and adaptive expert residency

- **Placement:** GPU cache; host expert source/control.
- **Shape:** Slots indexed by layer/expert; native blob sizes may differ.
- **Layout:** Residency/table bookkeeping is not expert compute; device slots contain exact expert bytes; per-layer admission option does not imply eviction.
- **Lifetime:** Persistent across tokens; captured pointers must remain stable.
- **Ordering / synchronization:** Async fill caller-orders copy and reader; profile startup blocking/queued fills need explicit synchronization; verify_slot compares actual bytes.
- **State / constraints:** Reuse Strata profile/residency/adaptive machinery; FPGA cache lifecycle must preserve publication and ownership.
- **Pinned evidence:** [expert_cache.hpp:19–55](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/expert_cache.hpp#L19-L55); [expert_cache.hpp:180–217](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/expert_cache.hpp#L180-L217); [verify.hpp:20–21](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/verify.hpp#L20-L21).

### ple: PLE n-gram hash/lookup, row transfer and history update

- **Placement:** Host/SSD-or-OS-cache table + GPU consumption.
- **Shape:** Layer1 injection; token/history hash selects embedding rows; history channels use row+NG_HIST*channel.
- **Layout:** Host lookup outside capture; pinned row collected then asynchronous H2D; strided history layout.
- **Lifetime:** Table persistent; row buffer cannot be recycled before transfer completes; history per session.
- **Ordering / synchronization:** Issue previous lookup must be collected; layer1 PLE before gated residual; caller advances history exactly once/token.
- **State / constraints:** Token history and GPU history are commit/rollback state, not just lookup scratch.
- **Pinned evidence:** [layer.cpp:1473–1528](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L1473-L1528); [layer.cpp:1593–1619](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L1593-L1619); [verify.hpp:13–18](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/verify.hpp#L13-L18).

### head: Final gated-residual mix, norm and vocabulary projection

- **Placement:** GPU.
- **Shape:** R[4,2560] ->2560 ->vocab248320.
- **Layout:** F32 output_hc_norm; BF16-in-F32 down/up; GR read ->Q8_K ->quantized head GEMV; optional native head.
- **Lifetime:** Head weights persistent; logits per token/window row; shared scratch ordered.
- **Ordering / synchronization:** Finish last FFN residual materialization before head; logits ready before sampler.
- **State / constraints:** Reads final residual; writes logits.
- **Pinned evidence:** [layer.cpp:1391–1443](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L1391-L1443); [layer.cpp:1622–1640](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L1622-L1640); [native_head.hpp:1–49](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/native_head.hpp#L1-L49).

### sampling: Penalties, top-k/top-p/min-p, temperature, greedy or stochastic pick

- **Placement:** GPU kernel, host-callable launch.
- **Shape:** logits[T,vocab] ->token_ids[T]; histories[T,history_len].
- **Layout:** Device pointers for logits/out/history; filter order specified; Philox seed/counter per row; greedy<=0 temperature.
- **Lifetime:** History/counter persistent across requests as configured; per-window history rows differ.
- **Ordering / synchronization:** Host must await device completion before consuming ids; speculative rows require their own histories.
- **State / constraints:** Sampler draw counter/seed and history must be part of reproducibility/checkpoint inventory.
- **Pinned evidence:** [sampler.hpp:1–45](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/kernels/sampler.hpp#L1-L45); [sampler.hpp:106–110](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/kernels/sampler.hpp#L106-L110).

### token-execution: Single-token graph, CPU doorbell and layer barriers

- **Placement:** GPU graph + CPU orchestration.
- **Shape:** 48 layer sequence handshakes per token.
- **Layout:** Mapped x/ids/weights, sequence and flag; shared GPU scratch with per-layer persistent state.
- **Lifetime:** Graph captures pointers; per-token buffers retained until stream completion.
- **Ordering / synchronization:** Graph launch; poll sequence + stream query; load fence; CPU pool; seq_cst/store fence; publish flag; final stream sync. Timeout reports failure, not proven quiescence.
- **State / constraints:** 20s layer wait timeout is not a safe FPGA cancellation/rollback protocol.
- **Pinned evidence:** [session.cpp:929–983](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/session.cpp#L929-L983); [layer.cpp:1326–1334](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L1326-L1334).

### prefill: Chunked prompt through all operators, grouped experts and state handoff

- **Placement:** GPU tensor-core GEMMs + host/pinned copy stream; optional CPU share.
- **Shape:** Chunk[pos0,pos0+n) through48 layers; recurrences walk chunk.
- **Layout:** Requires native GGUF blocks; quantized projections dequantized to FP16, BF16 as-is; grouped resident experts or host->pinned-ring stream.
- **Lifetime:** Borrowed expert rings and streams remain live to completion; persistent state handed to decode.
- **Ordering / synchronization:** Cross-stream copy/compute dependencies; all streams idle before borrowed ring returned; native-only prerequisite.
- **State / constraints:** Leaves GDN/conv, KV/index and PLE at prompt frontier; draft prefill precision can change drafts, not authorize target-logit drift.
- **Pinned evidence:** [prefill.hpp:1–10](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/prefill/prefill.hpp#L1-L10); [prefill.hpp:32–44](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/prefill/prefill.hpp#L32-L44); [prefill.hpp:60–79](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/prefill/prefill.hpp#L60-L79); [prefill.hpp:175–177](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/prefill/prefill.hpp#L175-L177).

### mtp: MTP embedding/hidden projections, own QSA/MoE/head and draft catch-up

- **Placement:** GPU; currently all draft experts VRAM.
- **Shape:** Residual_i +token_i+1 predicts token_i+2; own512/top10/shared expert.
- **Layout:** Large draft projections Q8_0; dense attention, no target indexer mutation; optional draft Q4; own KV/ring with host resume.
- **Lifetime:** Separate draft KV and residual; may borrow main embedding/head/windowR; lent expert snapshot preserves captured addresses.
- **Ordering / synchronization:** Reserve before inflight; sync=false caller orders stream; expert lending refuses decode while lent; restore/catch-up accepted prefix.
- **State / constraints:** Approximate drafts are proposals only; verifier chooses emitted tokens; rejected draft KV must not become target state.
- **Pinned evidence:** [mtp.hpp:1–19](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/mtp.hpp#L1-L19); [mtp.hpp:46–49](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/mtp.hpp#L46-L49); [mtp.hpp:64–64](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/mtp.hpp#L64-L64); [mtp.hpp:88–144](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/mtp.hpp#L88-L144).

### verify-window: Batched target verification through every trunk operator

- **Placement:** GPU graph + grouped CPU cold experts.
- **Shape:** T consecutive rows through48 layers; last accepted token plusT-1 drafts.
- **Layout:** Requires native GPU projections, fused GR/GDN, fast attention/selection, native indexer, profile-filled VRAM tier and device residency table.
- **Lifetime:** Window inputs/replay buffers and snapshots survive until prefix commit completes.
- **Ordering / synchronization:** Per-token arithmetic equivalence is implementation intent requiring tests, not a proven result from citations.
- **State / constraints:** Speculative KV/index appends; deferred GDN/conv; snapshots include index tail and PLE; eligible single-token self-commit exception.
- **Pinned evidence:** [verify.hpp:3–21](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/verify.hpp#L3-L21); [verify.cpp:4143–4153](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/verify.cpp#L4143-L4153).

### prefix-commit: Accepted-prefix replay/restore and asynchronous commit completion

- **Placement:** GPU replay + host PLE advancement.
- **Shape:** 1 <=n_keep<=last_t; reject suffix.
- **Layout:** Replays accepted GDN/conv; restores/re-appends index tail; rejected KV overwritten before reuse.
- **Lifetime:** Snapshot/replay/commit parameter buffers retained until completion; no early buffer reuse.
- **Ordering / synchronization:** Same-stream next window is ordered; other readers call wait_commit/event synchronization; synchronous path stream-syncs.
- **State / constraints:** Visibility of candidate/result is not commit. Existing internal API is not opaque versioned FPGA transaction ABI.
- **Pinned evidence:** [verify.hpp:13–18](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/verify.hpp#L13-L18); [verify.cpp:4139–4220](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/verify.cpp#L4139-L4220).

### cpu-isa: CPU expert build and runtime ISA selection

- **Placement:** x86 CPU; non-x86 portable fallback differs.
- **Shape:** AVX2/FMA/F16C expert objects alongside separate AVX512 and optional AVX1 units.
- **Layout:** Per-source compiler flags; CPUID FMA/OSXSAVE/AVX/F16C, XCR0 XMM/YMM and AVX2 gates; optional AVX-VNNI function targets do not raise AVX2 floor.
- **Lifetime:** Build-selected ggml baseline affects fallback arithmetic; runtime force flags cannot make unsafe global compiler flags safe.
- **Ordering / synchronization:** Gate before dispatch; AVX512 VNNI IQ path separate; AVX2 host does not need AVX512 merely because objects exist.
- **State / constraints:** Dual E5-2696v4 is an AVX2/FMA reference target, not a measured build or throughput result; CPU test targets do not establish CPU-only decode.
- **Pinned evidence:** [cpu.cmake:193–235](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/cmake/backends/cpu.cmake#L193-L235); [cpu.cmake:259–328](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/cmake/backends/cpu.cmake#L259-L328); [expert_layout.cpp:188–239](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cpu/expert_layout.cpp#L188-L239); [native_expert.cpp:567–594](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cpu/native_expert.cpp#L567-L594); [portable.cpp:24–39](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cpu/portable.cpp#L24-L39).

## Quantization and mixed tensor support

The primary/default audit target is IQ2_XS. Q2_0 is the speed comparison and IQ3_S the quality comparison. These names identify pack families, not a promise that every matrix uses that type. Preserve original type IDs and row/block metadata. Effective bits below include block scale/metadata but not artifact headers or alignment. A byte count alone does not identify the format (for example IQ3_S and Q3_K both use110 bytes/256 values).

### IQ2_XS

- **Type/layout:** GGML type17; 74 bytes/256 values (2.3125 effective bits/weight).
- **Encoding:** FP16 d + uint16 grid/sign words + packed 4-bit scales; nonlinear integer codebook, not FP4.
- **CPU:** Native AVX2 IQ256 support, runtime AVX2/FMA/F16C gates and group-size/arithmetic gates; AVX512 tier optional; ggml singleton/fallback depends on build.
- **GPU:** native_mmvq supported via iq_mmvq; grouped/prefill dispatch still requires compatible pack/layout and path validation.
- **FPGA:** No verified IQ decoder/MAC/runtime path at this pin pair; LUT/sign/scale semantics and rounding need a dedicated implementation.
- **Pinned evidence:** [strata-quant-common.h:379–385](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/strata-quant-common.h#L379-L385); [iq_avx2.cpp:1–40](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cpu/iq_avx2.cpp#L1-L40); [native_expert.cpp:567–594](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cpu/native_expert.cpp#L567-L594); [native_mmvq.cu:3528–3592](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cuda/native_mmvq.cu#L3528-L3592).

### Q2_0

- **Type/layout:** GGML type42; 18 bytes/64 values (2.25 effective bits/weight).
- **Encoding:** FP16 d +16 packed 2-bit code bytes, fixed integer offset; native GGUF block is not canonical S2 blob.
- **CPU:** Native q2_native_kernels(type42) accepts runtime AVX2 or AVX1 tier; AVX2 object uses FMA/F16C; bitplane opt-in changes last bits; portable canonical custom Q2 blob refused.
- **GPU:** native_q2_0_mmvq supported,64 elements/18B. Native and canonical representations cannot be interchanged by name.
- **FPGA:** No verified implementation; specify code ordering, signed/offset arithmetic, scale rounding and activation contract.
- **Pinned evidence:** [expert_layout.hpp:84–102](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/kernels/cpu/expert_layout.hpp#L84-L102); [q2_avx2.cpp:1–105](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cpu/q2_avx2.cpp#L1-L105); [portable.cpp:24–39](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cpu/portable.cpp#L24-L39); [native_mmvq.cu:3535–3589](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cuda/native_mmvq.cu#L3535-L3589).

### IQ3_S

- **Type/layout:** GGML type21; 110 bytes/256 values (3.4375 effective bits/weight).
- **Encoding:** FP16 d +qs/qh grid bits +signs +4-bit scales; not Q3_K despite equal110B/256 storage.
- **CPU:** AVX2 IQ256 and optional AVX512 IQ512 paths with dispatch, multi-token thresholds and singleton parity/arithmetic restrictions; ggml fallback.
- **GPU:** native_mmvq supported via iq_mmvq; native support is not proof every tensor/operator uses that path.
- **FPGA:** No verified nonlinear grid/sign decoder; IQ3_S exact semantics required independently of IQ2_XS.
- **Pinned evidence:** [strata-quant-common.h:403–412](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/strata-quant-common.h#L403-L412); [iq_avx2.cpp:1–40](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cpu/iq_avx2.cpp#L1-L40); [native_expert.cpp:648–707](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cpu/native_expert.cpp#L648-L707); [native_mmvq.cu:3528–3592](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cuda/native_mmvq.cu#L3528-L3592).

### Mixed GGUF and engine tensor forms

- **Encoding:** Pack labels are primary quantizers, not per-tensor inventories. Preserve tensor type, dimensions, row bytes, planes, offsets, codebook, scales and any global tail scales.
- **CPU:** Native per-layer/per-role metadata selects activation/dot kernels; fallback requires valid linked/own traits and correct build baseline.
- **GPU:** BF16/F32 norms/router/hyperconnections plus mixed native quantized projections/shared/head/experts. Native MMVQ list is operator-specific, not universal embedding/prefill/cache support.
- **FPGA:** FP4/NVFP4/MXFP4 primitives are not IQ2_XS/IQ3_S/Q2_0 implementations. Do not reinterpret or silently requantize.
- **Pinned evidence:** [weights.hpp:1–69](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/core/weights.hpp#L1-L69); [layer.cpp:468–507](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/layer.cpp#L468-L507); [native_dense.cpp:134–209](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/core/native_dense.cpp#L134-L209); [native_expert.cpp:710–729](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/kernels/cpu/native_expert.cpp#L710-L729); [types.hpp:9–39](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/types.hpp#L9-L39).

### CPU build and runtime limits

The implementation compiles IQ/KQ/NVFP4 AVX2 units with AVX2/FMA/F16C flags and separate AVX512 units; Q2 has an AVX2 object and optional AVX1 object. The runtime checks CPUID plus OSXSAVE/XCR0, not merely a model name or historical AVX512 documentation. Optional AVX-VNNI is function-targeted and independently runtime gated; it is not required on a Broadwell-class AVX2/FMA host. The ggml baseline and native-versus-floor build affect fallback arithmetic. Multi-token thresholds and exact/window-invariant modes may select different kernels. Therefore “AVX2 supported” does not prove byte parity for a particular model/build or aggregate speed. See the cpu-isa and cpu-experts anchors above.

Reference host dual E5-2696 v4 (44 physical/88 logical cores, AVX2/FMA,1TiB RAM) is a planning target, **not hardware used for a measurement here**. No build/runtime/kernel-parity test was executed on it. Portable/non-x86 stubs explicitly refuse some canonical custom Q2 blobs; native GGUF support must not be generalized to those blobs.

## Verification, rollback and ownership

The source requires native GPU projections, fused GR/GDN, fast attention and selection, native indexer, and a **profile-filled VRAM expert tier with its residency table on device**. Batched verification is not an independent CPU backend. Its bitwise single-token equivalence is an implementation contract to test, not a result established by reading comments.

A speculative window appends KV/index keys and records GDN/conv replay inputs. Prefix commit replays accepted recurrence/convolution, restores and re-appends the index tail, and restores/advances PLE history. Rejected KV is overwritten before a later valid reader sees those positions. Eligible single-token windows self-commit, except the guarded QFUSE case. Shared scratch, final residual rows, sampler histories and MTP catch-up are also part of the frontier. Treating only expert outputs as transaction state is insufficient.

Async commit is safe for the next operation on the same ordered stream. Other readers need the commit event/wait. Candidate-ready, DMA visibility and committed-state publication are different milestones. Current graph/doorbell timeout handling reports failure; it does not prove a device has stopped touching memory. A future adapter needs an opaque session/transaction identity, monotonic committed version/frontier, immutable checkpoint inventory, one active mutation transaction, completion fences, quiescent rollback, timeout quarantine until completion or proven reset+new epoch, and CPU fallback only after quiescence and rollback. These are **future ABI requirements**, not claimed existing public APIs. Strata remains authoritative owner of model/session state.

MTP has its own GPU KV/draft residual and currently assumes all draft experts in VRAM. Lending must retain captured addresses and block decode while lent. Q8_0/Q4 draft approximation can change acceptance/cost; it does not authorize approximation of the target verifier. Report MTP off/on separately, counting only accepted output tokens.

## Missing actual tensor metadata and required next evidence

Before planning a memory map or hardware kernel set, obtain an explicitly authorized local model inventory without silently downloading weights. Record:

1. Exact pack repository/revision and every shard filename, byte size and hash; license and redistribution restrictions.
2. Actual GGUF geometry, tensor names, type IDs, dimensions/strides, block geometry, row bytes, shard/offset/alignment and total storage.
3. Separate gate/up/down types and offsets for routed/shared experts; per-layer variations; any codebook, scale/offset planes and NVFP4-style global tail scales.
4. Forms of embedding/head, router, GR down/up/inject, norms, GDN parameters/convolution and PLE; engine conversions (BF16/F32/Q8_0) and any auxiliary MTP artifacts.
5. Context/KV format and mode, index/conv/recurrent/history snapshots, pinned buffers, graph scratch and expert slot budget; include host duplicates and draft state. Do not estimate every tensor as “2-bit” or every byte as reusable FPGA RAM.
6. Actual selected CPU build/compiler/ISA floor, runtime dispatch, environment flags, NUMA placement and numerical mode. Run block dequant/dot, operator and accepted-prefix parity for actual types and boundary shapes.

No complete tensor-level support/memory matrix is possible from these source defaults alone. Missing metadata is an explicit unknown, not a reason to relabel IQ as FP4 or claim a supported path.

## Acceptance and what was not tested

Correctness precedes speed. Retain full routing, top10 and expert skip=0. Separate prefill/TTFT/decode and MTP off/on. Compare target logits/token outputs and recurrent/KV/index/PLE frontiers; inject rejection, delayed completion, timeout and fallback cases. For >10 tok/s, count accepted output tokens and full wall time including CPU, host reads, DMA/PCIe, synchronization, commit and any draft cost. Device-only rates or historical upstream measurements do not satisfy this goal.

Reference FPGA xc7k480tffg1156-2,4GiB DDR3 and PCIe Gen2x8 capability (historical Gen2x4/H2C around1.39GB/s) are planning context only, not newly checked board identity, link width or measured transfer results. No bitstream, FPGA IQ primitive, endpoint, ABI, model decode or throughput was verified by this audit.

A source-anchor checker may confirm lock equality, clean exact-HEAD checkout, line bounds and optional literal excerpts. It cannot validate semantic interpretation, compilation, artifact compatibility, inference or timing. Bootstrap unit tests are not inference tests.

### Explicit limits

- Unconfirmed static analysis only. Source bounds/SHA validation is not semantic correctness, a build, model execution or a measured performance result.
- No actual model GGUF/header/tensor inventory was supplied or inspected. IQ2_XS primary, Q2_0 speed and IQ3_S quality are intended pack categories only; no exact tensor type map, shard hashes, artifact lengths, alignment, auxiliary forms or model memory budget is established.
- No CPU+FPGA complete decode backend is established by these sources. Native CPU expert kernels and CPU-only unit-test targets do not implement embedding, GR, GDN/QSA, PLE, router/shared/head/sampler, prefill, MTP or verifier scheduling on a CPU-only engine.
- Non-expert GPU replacement is a required central follow-up: placement/implementation/parity for every operator in entries, including stateful recurrence, KV/index storage, normalization/activation, host lookup, head/sampling and orchestration. Replacing only the expert accelerator leaves the GPU dependency intact.
- Verifier explicitly depends on native GPU projections, fused GR/GDN, fast attention/selection, native indexer and profile-filled VRAM resident expert tier/device residency table. CPU+FPGA must provide equivalent contracts or retain GPU; MTP also assumes VRAM experts and own GPU KV.
- No openTPU source claim or hardware compatibility is certified here. Strata pin is audited; openTPU pin is integration context only. No FPGA IQ support, DMA coherence, usable endpoint, programming, bitstream, ABI compatibility or throughput is verified.
- An opaque versioned transaction ABI, candidate-versus-commit separation, quiescent rollback, timeout quarantine/reset epoch and safe CPU fallback are requirements for future adapters, not demonstrated Strata public APIs.
- Broader optional model families, vision/audio, remote multi-GPU implementations and every environment-variable variant were not exhaustively audited. Scope is default compiled Flash-Next text prefill/decode/MTP target path and its source-visible alternatives.
