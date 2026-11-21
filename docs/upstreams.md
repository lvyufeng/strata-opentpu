# Upstreams, forks, and licensing

## Three repositories, two kinds of metadata

[ADR 0002](adr/0002-three-repository-integration.md) selects separate forks for
capability development and this independent repository for integration.

| Upstream | Capability fork | Role | Upstream license |
| --- | --- | --- | --- |
| [Strata](https://github.com/Niko1221/Strata) | [lvyufeng/Strata](https://github.com/lvyufeng/Strata) | Existing model engine, CPU kernels, cache/MTP; planned FPGA backend and scheduling changes | MIT |
| [openTPU](https://github.com/FeSens/openTPU) | [lvyufeng/openTPU](https://github.com/lvyufeng/openTPU) | FPGA RTL/ISA, low-bit compute, device runtime and transport | Apache-2.0 |

- [upstreams.json](../upstreams.json) preserves original bootstrap reference
  commits and historical integration status. It is not an executable build lock.
- [dependencies.lock.json](../dependencies.lock.json) is the source-acquisition
  lock for the forks. Exact commits do not by themselves prove build, hardware,
  or inference compatibility.
- Profiles select a compatible combination. The
  [bootstrap profile](../configs/profiles/bootstrap.json) is not inference-ready.
  Future verified profiles also pin board/toolchain, bitstream SHA256 and ABI,
  model revision/checksums, and runtime parameters.

Both forks have an `fpga-integration` development branch starting at their locked
upstream commit; fork `main` is unchanged. The current source lock still selects
the exact upstream SHAs, not floating development tips. Bootstrap checkouts are
clean detached inputs, not development workspaces. Make capability changes in
explicit branches/worktrees and review commits before updating the lock.

Upstream updates are reviewed and locked deliberately, not fetched as floating
branch tips. Keep generic improvements suitable for upstream contributions;
retain experimental backend/hardware changes in the respective forks. Maintain
clear modification history rather than duplicating implementation in this repo.

## Source acquisition and distribution

The [Agent setup guide](setup/agent-bootstrap.md) describes plan-first operation.
`bootstrap --apply` authorizes Git acquisition of the locked dependency sources
into ignored `third_party/`; it does not authorize installs, model downloads,
hardware programming, flash writes, or network configuration changes.

Audit complete dependency closure, embedded third-party code, and build tools.
Strata may use its own CPU quantization layer as well as ggml-derived code; do
not assume every path is simply ggml-cpu. Before distributing source or binaries:

1. Record upstream/fork commits, relevant files, and modifications.
2. Preserve Strata copyright/MIT, openTPU Apache-2.0, embedded licenses, and any
   relevant NOTICE files; mark modifications as required.
3. Check obligations for linked/vendored dependencies and released artifacts.
4. Keep original Apache-2.0 integration code distinct: it does not relicense
   either fork, model weights, or vendor IP.

Model licenses require separate review; quantized-repository YAML alone is not
sufficient. Tool licenses, credentials, restricted vendor IP, private host data,
weights, and unreviewed build products must not enter Git. Bitstream release
artifacts are a future option only after distribution-rights review; do not
assume a tool-generated artifact is unrestricted.

The projects are unofficial and not upstream-endorsed. The target FPGA backend
and combined model execution are not implemented by the bootstrap scaffold.
