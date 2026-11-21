# Configuration contracts

- [Bootstrap profile](profiles/bootstrap.json): only host inspection and locked source acquisition are implemented.
- [Kintex-7 reference](boards/kintex7-reference.json): expected target characteristics, not verified device identity or a programming configuration.
- [IQ2_XS model reference](models/qwen38-flash-next-iq2-xs-reference.json): shortlist only, no pinned downloadable files.
- [Dependency lock](../dependencies.lock.json): exact fork commits, **not** a validated inference combination.

Current JSON files are data, not shell commands. Setting a capability to `true`
does not implement it or authorize operations. The CLI rejects unsupported
execution even when a profile claims otherwise.

A future runnable profile must identify both dependency commits, supported host
and toolchain, board/device checks, bitstream URL + SHA256 + RTL/runtime ABI,
model revision + file checksums, memory/layout configuration and validation
results. Do not expose credentials, host addresses or local machine paths here;
keep local overrides under ignored `local/` and sanitize reports.

Do not point a reference-only file at hardware and treat it as ready to run.
See [Agent setup](../docs/setup/agent-bootstrap.md) for the staged handoff.
