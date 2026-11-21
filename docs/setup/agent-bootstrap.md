# Agent setup: bootstrap now, verified execution later

Read [AGENTS.md](../../AGENTS.md), the [architecture](../architecture.md), and the
[upstream policy](../upstreams.md) before acting. A single clone is the intended
entry point; a working full inference installation is a future deliverable.

## Current supported operations

Use Python 3.11+ and Git from the local environment. The CLI is
[scripts/sop.py](../../scripts/sop.py), with the
[bootstrap profile](../../configs/profiles/bootstrap.json) as the current profile.
Run from the repository root:

```sh
python3 scripts/sop.py doctor
python3 scripts/sop.py bootstrap
```

`doctor` inspects local prerequisites/configuration without installing or
programming anything. It is not a hardware smoke test and cannot certify model
or FPGA readiness. `bootstrap` prints the locked source-acquisition plan without
network or write side effects. Review its repositories, commits, and destination.

If the user explicitly authorizes fetching those dependency sources:

```sh
python3 scripts/sop.py bootstrap --apply
```

This obtains the exact fork commits from
[dependencies.lock.json](../../dependencies.lock.json) under ignored
`third_party/`, without recursively initializing submodules. Acquired checkouts
are detached, clean inputs; use separate development branches/worktrees for fork
changes. Existing wrong or dirty checkouts are rejected, not reset or cleaned.
Explain any conflict and request a user-chosen remedy rather than overwriting,
deleting, or moving the checkout. It does not build dependencies, install tools,
download models, load a bitstream, write flash, or start inference. Do not
substitute unpinned upstream or fork branch tips if acquisition fails; diagnose
the lock/source problem and report it.

Run the hardware-free integration checks and report their actual result:

```sh
python3 -m unittest discover -s tests -v
```

Passing these checks verifies metadata/setup safeguards, not model inference.

Select a profile with `--profile`, for example:

```sh
python3 scripts/sop.py doctor --profile configs/profiles/bootstrap.json
python3 scripts/sop.py bootstrap --profile configs/profiles/bootstrap.json
```

Read CLI help before using another profile. A profile name or successful clone
is not evidence that the combination is inference-ready.

## Planning interfaces, not working execution

The following are plans only in the bootstrap profile:

```sh
python3 scripts/sop.py build
python3 scripts/sop.py program
python3 scripts/sop.py smoke-test
python3 scripts/sop.py launch
python3 scripts/sop.py benchmark
```

Default execution describes future work; `--apply` rejects these operations
until a supported profile and real adapters exist. Do not bypass that refusal
with guessed shell commands. No build/install, board communication, programming,
model inference, or performance test is implemented by these placeholders.

## Authorization boundaries

- `bootstrap --apply` permits the specified dependency Git acquisition only.
- Installing tools/packages, downloading model weights, and hardware programming
  each need explicit authorization for the operation. A request to inspect or
  plan setup is not permission to perform them.
- Volatile FPGA bitstream loading is separate from persistent flash writing.
  Persistent flash support is not implemented; never infer permission for it
  from permission to load a bitstream.
- Do not change host network settings, flash, existing repositories, or hardware
  as a workaround for an unsupported profile or failed check.
- Keep secrets, private addresses, host-specific paths, weights, bitstreams,
  vendor IP/licenses, and unsanitized logs out of Git.

## Future verified-profile procedure

Only once a profile explicitly supports real operations should an Agent:

1. Check exact fork revisions, board identity, toolchains, and supported commands.
2. Check bitstream SHA256 and software/firmware ABI; check model revision, actual
   quantization formats, shard checksums, and licenses.
3. Obtain explicit authorization for each install/download/program operation.
4. Build or obtain an allowed compatible artifact, validate the selected target,
   then perform volatile loading if authorized.
5. Run real device/DMA/operator smoke tests. A passing device test alone does not
   establish full model correctness.
6. Launch only an inference-ready combination, then validate output and use the
   [benchmark protocol](../../benchmarks/README.md) for measurements.
7. Report exact versions, commands applied, checks passed/failed, remaining gaps,
   and sanitized evidence rather than claiming success from source acquisition.

Today the appropriate result is a bootstrap inspection/acquisition report, with
all missing build/hardware/inference capabilities stated explicitly.
