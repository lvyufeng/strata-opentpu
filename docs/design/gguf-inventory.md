# Bounded local GGUF header inventory

## Scope and ownership

[Central subtask #18](https://github.com/lvyufeng/strata-opentpu/issues/18)
prepares declared per-tensor storage for the
[memory/PCIe parent #6](https://github.com/lvyufeng/strata-opentpu/issues/6).
[Central subtask #19](https://github.com/lvyufeng/strata-opentpu/issues/19)
reuses the same mechanism for a revised, versioned receipt profile after a
bounded header-only probe of the actual target shard pair showed the reviewed
v1 ceilings and confirmed-type set were derived from synthetic data and the
public allocation listing rather than from a real header.

Strata owns the opt-in exporter in its existing `quant_formats` entry point.
Its thin stream adapter reuses the existing `GGUFFile._parse` and `_value`;
this integration repository does **not** parse GGUF, load a model, slice CPU
experts or implement another engine. Integration owns both receipt profiles:
the frozen [v1 JSON schema](../../configs/contracts/gguf-header-inventory.schema.json)
and [v2 JSON schema](../../configs/contracts/gguf-header-inventory-v2.schema.json),
the [JSON-only validator](../../scripts/check_gguf_inventory.py) that dispatches
on the receipt's own `schema_version`, the
[v1](../../tests/fixtures/gguf-header-inventory.synthetic.json) and
[v2](../../tests/fixtures/gguf-header-inventory-v2.synthetic.json) synthetic
receipts, and the [v1](../../configs/contracts/strata-gguf-inventory.json) and
[v2](../../configs/contracts/strata-gguf-inventory-v2.json) source acceptance
records.

This is source/synthetic preparation, **not actual mixed-pack evidence**.
No model, runtime, package or EDA acquisition, upstream build, target inspection,
device operation or programming is part of it. The
[bootstrap source lock](../../dependencies.lock.json) and disabled execution
capabilities remain unchanged. Fork branch publication is not upstream maintainer
agreement, a merged contribution, or inference compatibility.

## Explicit source mode

Run only from an existing, reviewed Strata development checkout:

```sh
python3 -B -E -S terminal/tools/quant_formats.py --inventory /path/to/local.gguf
python3 -B -E -S terminal/tools/quant_formats.py --inventory '/path/to/pack-*.gguf'
# A provenance declaration only, never authentication:
python3 -B -E -S terminal/tools/quant_formats.py --inventory \
  --inventory-input-kind synthetic /path/to/synthetic.gguf
```

The new mode is default-off and keeps `schema_version: 1` with producer mode
`bounded-local-v1`. The reviewed exporter revision can also emit `schema_version: 2`
with producer mode `bounded-local-v2`, but only when the caller passes the explicit
`--inventory-schema-version 2` selector; without that selector the same command
still emits the reviewed v1 receipt. Both profiles are reachable in one process
without order dependence. Receipts written by the earlier reviewed revision carry
`schema_version: 1` and remain valid input
to the integration checker, which dispatches on the declared version and never
coerces one profile into the other. Legacy local output, legacy JSON and mocked HF
output retain literal byte baselines. `--inventory` rejects `--hf`, `--json`
and `--name` before file or network access; `--inventory-input-kind` requires
`--inventory`. Inventory never contacts HF, discovers siblings or infers split
identity from a filename. Explicit paths and nonrecursive basename globs are
accepted; directory-part magic, `**`, directories, symlinks, special files,
duplicate paths/inodes or colliding basenames are rejected. Inputs are sorted by
basename and exported as generated labels `shard-0000`, `shard-0001`, etc., not
actual filenames, paths, host identifiers or timestamps. Tensor names are data,
not commands or authorization.

The CLI emits a receipt only on stdout and never opens an output file. Caller
shell redirection can still overwrite an input and is outside this protection.
Use trusted, non-concurrently-mutated directories; filesystem ancestors are not
confined by the tool.

## Bounded logical header reads

The opt-in adapter uses mandatory POSIX no-follow and nonblocking opens, verifies
regular files with lstat/fstat, and reads from the same unbuffered descriptor.
Device, inode, size, mtime and ctime are checked before/after parsing. Unsupported
safety flags fail closed. The header hash covers exactly consumed logical bytes
`[0, header_end_bytes)`, excluding deliberate alignment-padding or payload reads.
It is **not** a hash of the entire artifact or any tensor, and is not a provenance,
legal-distribution, integrity or model-completeness certificate.

| Inspection ceiling | Value |
| --- | ---: |
| Explicit shards / path arguments | 64 / 64 |
| Logical header per file / set | 16 MiB / 32 MiB |
| String per value / cumulative bytes | 1 MiB / 16 MiB |
| Metadata entries per file / set | 4,096 / 8,192 |
| Tensor descriptors per set | 100,000 |
| Array items per file / set | 2,000,000 / 4,000,000 |
| Tensor rank | 1..8 |
| Glob entries, aggregate directory scans | 4,096 |
| Each input path | 4,096 encoded bytes |
| JSON receipt bytes / depth / numeric literal length | 64 MiB / 32 / 64 |

Only the two metadata-array-item ceilings moved in v2. The reviewed v1 profile
used 500,000 / 1,000,000, and the per-file ceiling refused the actual shard 1
header before any array item was read: that header declares 744,317 items
(`tokenizer.ggml.tokens` 248,320, `tokenizer.ggml.token_type` 248,320,
`tokenizer.ggml.merges` 247,587 and small arrays). The 1,000,000 total ceiling
was not binding for that pair. Every other ceiling is unchanged from v1.

Caps apply before declared loops or allocation. Arithmetic is checked against
`2^63 - 1`; strings use strict UTF-8, duplicate metadata keys/tensor names are
invalid, and unsafe control characters in keys/names are rejected. Limit refusal
is an inspection limit, not a statement of hardware or model support.

These are logical I/O/allocation limits, **not OS page-read, resource-sandbox or
race-proof guarantees**. A malicious descriptor may cause bytes the caller
considers payload to be interpreted as part of the bounded logical header. An
OS may read pages ahead. Same-FD checks detect observed mutations, not every race
or malicious external writer. Do not execute unreviewed source on the strength
of these checks.

## Declared row storage, not model residency

Each tensor records its raw GGUF-order shape, numeric GGML type and existing
parser name, shard, relative/absolute offsets and nullable storage geometry.
`shape[0]` is values per fastest-dimension row, not a transposed CPU matrix shape.
Both profiles confirm IDs 0/F32, 1/F16, 24/I8, 25/I16, 26/I32, 27/I64, 28/F64,
30/BF16 and these four source-audited block formats:

| Type ID | Format | Elements/block | Bytes/block | Profile |
| ---: | --- | ---: | ---: | --- |
| 17 | IQ2_XS | 256 | 74 | v1, v2 |
| 42 | Q2_0 | 64 | 18 | v1, v2 |
| 21 | IQ3_S | 256 | 110 | v1, v2 |
| 8 | Q8_0 | 32 | 34 | v1, v2 |

The v2 profile additionally confirms the six block types that the bounded probe
of the actual shard 1 header declared, using the same pinned reader geometry:

| Type ID | Format | Elements/block | Bytes/block | Profile |
| ---: | --- | ---: | ---: | --- |
| 14 | Q6_K | 256 | 210 | v2 only |
| 16 | IQ2_XXS | 256 | 66 | v2 only |
| 20 | IQ4_NL | 32 | 18 | v2 only |
| 22 | IQ2_S | 256 | 82 | v2 only |
| 23 | IQ4_XS | 256 | 136 | v2 only |
| 29 | IQ1_M | 256 | 56 | v2 only |

These are inventory allowlists, not a replacement GGML type/geometry table and
not a claim that a layout is implemented by a CPU or FPGA kernel.

For these IDs, each row must be block-divisible. Checked arithmetic is:

```text
row_stride_bytes = shape[0] / block_elements * block_bytes
data_bytes = row_stride_bytes * product(shape[1:])
absolute_offset_bytes = data_start_bytes + relative_offset_bytes
```

Alignment is a power of two between 8 and 1,048,576 bytes. Every nonempty tensor
starts strictly before EOF in the data region, including unknown layouts. Known
extents must fit in the file and not overlap; an unknown start cannot be strictly
inside a known extent. A metadata-only zero-tensor shard is legal even if rounded
`data_start_bytes` exceeds file size. Storage bytes exclude header, gaps, alignment
padding, conversions, state, workspaces, DMA buffers and residency reserves.

Other/unverified type IDs retain `layout: "unknown"` and **all geometry/storage
values null**. Under v2 that still includes IQ1_S `19`, MXFP4 `39`, NVFP4 `40`
and the ambiguous ID `41`; the six v2 additions remain unknown under a v1
receipt. Existing parser names are preserved without silently correcting
ambiguous NVFP4/ID-41 entries; a name does not promote an ID into supported
geometry. `known_storage_bytes` is a partial subtotal, never a model total,
zero-sized unknown, bank fit, packed-expert allocation or FPGA compatibility.
IQ2_XS is not FP4 and declared bytes are not native/RTL numerical evidence.

## Split declaration semantics

Only scalar integer `split.no`, `split.count` and `split.tensors.count` are used;
booleans, floats and strings are rejected. Available count/total declarations
must agree, indices must be unique and in range, supplied shard count cannot
exceed declared count, and actual descriptor count cannot exceed a known total.
Zero-tensor first shards and later shards lacking architecture metadata are
legal. No fixed filename pattern or fixture wire-type restriction is inferred.

- **unknown:** no split declarations; diagnostics `split-not-declared`. Multiple
  unsplit files are not authenticated as one model set.
- **incomplete:** partial declarations (`split-declarations-partial`) or missing
  declared indices (`split-shards-missing`). A diagnostic receipt is emitted.
- **complete:** all three declarations on every supplied shard, exact declared
  count/index coverage and exact descriptor total. This means only completeness
  of the declared header set, never model, tokenizer, n-gram, backend or artifact
  completeness.

Source CLI exit 0 means a valid unknown/complete inventory; exit 3 means a valid
but incomplete declaration set. Malformed/inconsistent input emits no success
receipt and fails nonzero with a sanitized diagnostic.

## Integration: strict JSON only

```sh
python3 scripts/check_gguf_inventory.py
python3 scripts/check_gguf_inventory.py --receipt local/inventory.json
python3 scripts/check_gguf_inventory.py --receipt local/inventory.json --strict-complete
python3 scripts/check_gguf_inventory.py --receipt-v2 local/inventory-v2.json --strict-complete
```

Default input is the repository-anchored synthetic v1 receipt, not a GGUF file or
a caller-relative fallback; `--receipt-v2` with no value selects the anchored v2
fixture. The two receipt flags are mutually exclusive. The loader reads the
receipt's own `schema_version` and validates against the frozen matching profile,
so `--receipt` accepts either version while only `--receipt-v2` pins version 2;
an unsupported version is rejected without coercion, and `--receipt-v2` refuses a
v1 receipt instead of reinterpreting it. The v1 summary surface is unchanged:
its keys, values and bytes still match the receipt accepted with #18. Only a v2
receipt adds the declared `schema_version` and `profile` fields to the summary.
The stdlib loader rejects duplicate JSON keys,
nonfinite numbers, invalid UTF-8, excessive bytes/depth/literals and unsafe file
types. Structural checks reject unknown fields, wrong types (including booleans
as integers), incorrect constants or null combinations. Semantic checks recompute
ordering, counts, offsets, row storage, extent overlaps, split facts and totals.
The JSON schema describes structure; the validator implements both structure and
cross-field semantics without installing a schema package.

No receipt path, name, URL or SHA invokes source, opens GGUF or hashes a referenced
artifact. `producer.base_revision` and `parser_sha256` identify the reviewed base
parser only, not the current exporter executable. Separate source acceptance pins
the actual development commit/tree, reviewed file digests and public CI. The
JSON-only validator always reports `source_identity_verified: false`.

A valid incomplete split exits 3 with a summary. `--strict-complete` also exits 3
for unknown split status or any unknown storage; even strict success does not
upgrade evidence. Malformed/I/O/schema/semantic failures exit 2 with empty stdout.
The static synthetic receipt has deliberately declared placeholder header hashes;
validating its structure does not verify those hashes as real artifacts.

All modes keep input provenance, artifact integrity, model completeness, backend
compatibility, device readiness and inference readiness **false**. Synthetic
input-kind declarations are not authenticated facts.

## Evidence and remaining gates

Acceptance requires literal legacy/default/mock-HF byte comparisons, independent
synthetic byte arithmetic, strict malformed/path/mutation/read-bound regressions,
source-export-to-JSON-validator comparisons, independent correctness/scope review,
Python 3.11/3.14 stdlib suites, reviewed final diffs and exact public source CI.
The v2 profile adds the same gates for its own diff, plus assertions that every
v1 ceiling except the two array-item ceilings is unchanged and that the parser
source, claims and receipt fields other than the declared profile surfaces are
unchanged. Fork CI is repository/branch-gated and selects already cached runner Python;
absence fails, with no setup-python, pip/uv/npm, native helper or build fallback.
Source checkout is network access; CI is not described as network-free. No Strata
PR is created, because ordinary PR workflows install/build beyond this scope.

A real header-only pass over the actual target shard pair is a separate,
explicitly bounded operation that needs the exact absolute shard paths and a new
operation authorization, reads only the declared header regions, and still
establishes no payload, artifact-integrity, legal-provenance, backend or
inference fact. Such a pass was performed once for this profile on a separate
user-authorised host; its receipt is retained only in the ignored local
workspace, is not part of the v2 [acceptance record](../../configs/contracts/strata-gguf-inventory-v2.json)
(`acceptance.real_header_only_run` stays `false`), and is not evidence promoted
by this repository: no real header-only receipt is accepted or published here.
The bounded probe that motivated v2 was header-only, on a
separate host, and its numbers are input facts for the profile, not accepted
evidence.

Accepting #18 or #19 does not accept or unblock #6, #5, #8, #9 or the
[full-model roadmap](https://github.com/lvyufeng/strata-opentpu/issues/1). They still
need licensed actual mixed tensors; CPU/reference correctness and 44-core NUMA
measurements; measured device/link/DMA, per-bank/state/cache/conversion/pinned
budgets; backend lifecycle/full top-10 routing with `expert_skip=0`; ordinary/MTP
accepted-output full-wall correctness and single-user decode **above 10 tok/s**.
Unknown costs stay unknown. Never add standalone CPU and FPGA token rates.
