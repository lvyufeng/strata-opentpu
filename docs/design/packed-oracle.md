# Supplied-packed Q2_0/Q8_0 oracle: synthetic source preparation

## Scope and ownership

[Central subtask #17](https://github.com/lvyufeng/strata-opentpu/issues/17) is a
narrow preparatory child of the
[IQ/low-bit primitives task #8](https://github.com/lvyufeng/strata-opentpu/issues/8).
It supplies an opt-in, standard-library-only scalar reference for **already packed
bytes**, independently generated synthetic goldens, attribution and a public
source-test gate. It is not an IQ implementation or a quantizer.

The math, vector generator and scalar tests belong to the
[openTPU capability fork](https://github.com/lvyufeng/openTPU), in the standalone
module `tools.packed_oracle`. This integration repository owns only the
[receipt](../../configs/contracts/opentpu-packed-oracle.json),
[receipt/integrity checker](../../scripts/check_packed_oracle.py), orchestration
boundary and central acceptance evidence. It does not implement a second decode
or dot algorithm. Strata keeps its existing model engine, hot-expert residency,
cold CPU experts, cache and MTP; see the [architecture](../architecture.md).

Accepting this preparation does **not** accept or unblock #8, the
[Strata backend task #9](https://github.com/lvyufeng/strata-opentpu/issues/9), or the
[full-model roadmap #1](https://github.com/lvyufeng/strata-opentpu/issues/1).
Actual mixed-GGUF metadata, IQ2_XS/IQ3_S codebooks, native CPU comparison, complete
expert branches, RTL/resource/timing/device validation and whole-decode evidence
remain separate requirements. There are no model weights, native builds, ISA or
RTL adapters, DMA operations, device access or performance measurements here.

## Frozen source identity and wire contract

The [bootstrap lock](../../dependencies.lock.json) continues to select Strata
`3142b66b96e9c0e775e30f620b11b191a1427c2a` and openTPU base
`d84db55712aba0d40811d7762cb052f05d55b7e5`. The independently reviewed oracle
commit is recorded separately in the receipt; it is not a bootstrap lock bump or
an inference-compatible profile.

Relevant pinned Strata sources are
[decode:25–43/139–153](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/quant/dequant.cpp),
[generic dot:22–63/203–226](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/quant/dot_generic.cpp)
and [block declarations:195–201/266–272](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/strata-quant-common.h).
The receipt freezes those six line ranges, source revisions, the ggml origin
revision and complete MIT-notice digests. Source citations establish the selected
layout/formula, not native compiler or SIMD parity.

| Format | GGML type | Elements/block | Bytes/block | Payload after little-endian binary16 scale |
| --- | ---: | ---: | ---: | --- |
| Q2_0 | 42 | 64 | 18 | 16 bytes of adjacent 2-bit codes |
| Q8_0 | 8 | 32 | 34 | 32 signed int8 values, including -128 |

For Q2_0 element `j`, the code comes from byte `j // 4`, shifted by
`(j % 4) * 2`; codes 0, 1, 2, 3 map to -1, 0, +1, +2. Decode is scale times
that integer. Q8_0 decode is scale times the signed byte. Q2_0 is not Q2_K or
FP4; packed Q8_0 is not Strata's planar native activation representation.

Both dot variants accept supplied Q8_0 activation blocks. One Q2_0 block pairs
with two Q8_0 blocks in half-0, half-1 order; one Q8_0 weight block pairs with one
Q8_0 activation block. Inputs require exact packed lengths and positive block
multiples, at most 65,536 elements. Booleans are not dimensions. No implicit
padding, layout conversion, activation quantization or model loading occurs.

Finite half scales include negative values, signed zero, subnormals and maximum
finite values. Every exponent-31 half encoding (NaN or infinity) is rejected.

## Selected arithmetic, not native parity

The arithmetic identifier is `strata-generic-f32-rne-separate-v1`. It explicitly
selects exact half-to-binary32 widening, round-to-nearest-even binary32 after each
multiply and add, fixed block order, no FMA, no reassociation and no flush-to-zero.
The row accumulator starts at positive zero. Signed-zero and gradual-underflow
semantics are part of the contract; output comparisons use exact FP32 bit strings.

With `r32` denoting one binary32 RNE rounding and `dw`/`da` the widened scales:

```text
Q2_0, each weight block:
    subtotal = +0.0f
    for activation half in [0, 1]:
        intdot = sum((code - 1) * signed_activation)
        subtotal = r32(subtotal + r32(da * intdot))
    row = r32(row + r32(dw * subtotal))

Q8_0, each weight block:
    intdot = sum(signed_weight * signed_activation)
    factor = r32(dw * da)
    row = r32(row + r32(intdot * factor))
```

The exact integer subdot bounds are [-8192, 8128] for each Q2_0 half and
[-520192, 524288] for Q8_0. Their int32 values convert exactly to binary32.
Decoding both rows to floats and then taking a floating dot is **not** a substitute
for this grouped integer/scale order. The frozen tests include row-order,
half-subtotal, ties-even and separate-versus-fused counterexamples.

This contract describes one reproducible scalar reference. A native compiler may
contract operations or use ISA-specific accumulation/rounding; native/SIMD parity
needs a separately controlled build and elementwise comparison. No quantizer or
ActQ path is tested, and no claim is made about IQ formats or real mixed tensors.

## Independently generated synthetic evidence

The frozen fixture has **146 cases: 9 decode and 137 dot**, serialized as 123,947
bytes of JSON containing packed hex, expected integer subdots and exact FP32 bits.
Its generator derives expected values with independent exact Fraction/integer
arithmetic, not by recording the scalar implementation's outputs. Generator
independence is tested by making DUT decode/dot calls fail while generating.

```text
fixture_id: q2-q8-supplied-packed-v1
canonical payload SHA256:
59a84c7e1884e50a31798cee3a8f171229e26b361d09d80b0f57ddc0c3c04efa
complete serialized file SHA256:
709d690afbde91789286a0b991d3dca35f13fb5eaa4e6ef505690ed1295cc4bd
```

The canonical digest excludes only the top-level `payload_sha256` field and uses
sorted-key, compact ASCII JSON with non-finite JSON values disallowed. The file
digest additionally pins whitespace and the final newline. The receipt records
both digests, counts, formats, source provenance, exclusions, notice hashes,
independent-review decisions and the exact oracle commit/public CI run.

A successful oracle `verify` reports `synthetic: true`, `verified: true`, zero
mismatches and `device_ready: false` / `inference_ready: false`. The corpus and
boundary tests are evidence for this supplied-packed scalar scope only. They are
not real-model coverage, target-host/NUMA/native-CPU, RTL/FPGA, DMA, complete
GU→requantization→down, correctness of a full decode, or throughput evidence.

## Integration checker modes and trust boundary

Run the [checker](../../scripts/check_packed_oracle.py) from the integration root:

```sh
python3 scripts/check_packed_oracle.py
python3 scripts/check_packed_oracle.py --strict-receipt
python3 scripts/check_packed_oracle.py --fixture /path/to/supplied-fixture.json
```

Default mode validates receipt schema and consistency against the bootstrap lock,
without reading dependency sources or running an oracle. `--strict-receipt`
requires all recorded acceptance fields to be complete. `--fixture` additionally
checks JSON schema, provenance, packed lengths, integer bounds, counts and digests;
it deliberately does **not** perform decode/dot or certify mathematical truth.
Neither mode verifies the truth of recorded review/CI assertions. Acceptance
requires a human/Agent review of the actual commit and GitHub run, not just JSON.

Only an explicit opt-in invokes source code from a pre-existing checkout:

```sh
python3 scripts/check_packed_oracle.py --strict-receipt \
  --oracle-root /path/to/reviewed-opentpu
```

This first requires a complete receipt, nonsymlink independent Git root, exact
recorded HEAD and fork origin, tracked-byte/index identity, no extra files
(including ignored files), base ancestry, complete notice digests and pinned
fixture integrity.
The only source command is fixed in code:

```sh
python -B -E -S -m tools.packed_oracle verify tools/packed_oracle/data/q2_q8_synthetic.json
```

The actual interpreter is the checker's `sys.executable`; the checkout is its
working directory. The module, operation and fixture path are constants, never
commands or URLs from JSON. Before and after execution, the checker compares the
recorded commit tree with NUL-delimited index metadata, rejects assume-unchanged,
skip-worktree, sparse/unmerged entries, symlinks and gitlinks, and streams **all
tracked files' actual Git blob hashes and executable modes** against that tree.
This includes parent packages and possible local import shadows. Extra ignored
and untracked files are rejected; HEAD is checked again after execution.
Git status and clean/smudge filters are not used as content-integrity evidence.
Git/Python/loader environment overrides are stripped, Git hooks/fsmonitor are
disabled and the fixed subprocess has a timeout. Effective partial-clone/promisor
configuration is rejected before tree-object lookup, and Git lazy fetching is
disabled. No clone, fetch, install, native build, model acquisition or device
procedure is implemented by the checker.

**This is execution of trusted, reviewed source, not an OS/hardware or resource
sandbox.** Environment cleanup does not confine a malicious interpreter/module or
prevent arbitrary I/O, imports, network/device operations, races or external writes.
Captured output is bounded only after allocation. Use a checkout and interpreter
you trust; source-review and operation authorization remain necessary. Do not use
this flag as a mechanism to run arbitrary unreviewed repositories.

JSON is limited to 8 MiB, nesting depth 64, numeric literals of 64 characters and
256 fixture cases; duplicate keys, non-finite numbers and invalid UTF-8 fail closed.
Exit 0 means the selected checks passed (default mode may report an incomplete
receipt); exit 1 means `--strict-receipt` found missing fields; exit 2 means malformed,
unsupported, unsafe-source or failed fixed-source verification. Errors use a stable
JSON code/message on stderr, without raw subprocess output or private paths.

## CI, licensing and acceptance

The openTPU workflow `Packed-byte oracle checks` tests Python 3.11 and 3.14,
verifies the frozen corpus and checks byte-identical regeneration. It performs no
native build, model download or device operation. Accept only the exact recorded
oracle commit's successful public **push** run, with both matrix jobs passing.
The integration [Bootstrap workflow](../../.github/workflows/bootstrap.yml) tests
its own checker and validates `--strict-receipt` only: it never fetches or executes
the oracle and does not re-prove the fork's arithmetic.

The adapted mathematical files retain MIT terms with full Strata and embedded
ggml notices plus a modification record; original openTPU scaffolding remains
Apache-2.0. This is source-derived adaptation, not a clean-room or relicensing
claim. The receipt pins the two complete source-notice hashes. Review the
capability fork's attribution/modification record and license closure before
publication; software notices do not grant rights to model weights or vendor IP.

Follow the [development contract](../development.md): independently review math
and scope/license, run both local Python matrices and final diffs, publish exact
fork/integration commits and reciprocal central links, verify both public CI runs,
then close only #17 and synchronize Project Done at this synthetic scope.
Keep #8/#9/#1 open with their real remaining gates. Neither this receipt nor CI
changes the [bootstrap profile](../../configs/profiles/bootstrap.json): execution
of build/program/smoke-test/launch/benchmark remains unsupported.
