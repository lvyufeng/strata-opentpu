# Configuration contracts

- [Bootstrap profile](profiles/bootstrap.json): only host inspection and locked source acquisition are implemented.
- [Kintex-7 reference](boards/kintex7-reference.json): expected target characteristics, not verified device identity or a programming configuration.
- [IQ2_XS model reference](models/qwen38-flash-next-iq2-xs-reference.json): shortlist only, no pinned downloadable files.
- [Dependency lock](../dependencies.lock.json): exact fork commits, **not** a validated inference combination.
- [Offline resource-budget reference](budgets/kintex7-qwen38-reference.json): per-bank/NUMA inventory and transfer inputs with explicit unknowns, not an allocation plan or measured feasibility result. See the [budget contract](../docs/design/resource-budget.md).
- [Partial-tensor budget schema v2](contracts/resource-budget-v2.schema.json): descriptive JSON schema for per-expert sub-slice selection and residency; the standard-library [v2 checker](../scripts/resource_budget_v2.py) is authoritative and the frozen v1 files are unchanged. See the [v2 contract](../docs/design/resource-budget-v2.md).
- [Partial-tensor budget acceptance record](contracts/strata-resource-budget-v2.json): pinned validator/schema/fixture digests, synthetic fixture matrix and explicit scope exclusions. It is not a hot/cold placement plan, measured host/device result or inference-readiness record.
- [GGUF header receipt schema](contracts/gguf-header-inventory.schema.json): exact structure with nullable unknown layouts; [JSON-only validation](../scripts/check_gguf_inventory.py) also checks arithmetic and declared split semantics. The [synthetic fixture](../tests/fixtures/gguf-header-inventory.synthetic.json) is not an actual model or authenticated artifact.
- [Strata inventory source record](contracts/strata-gguf-inventory.json): reviewed exporter commit/tree, limits and source-test evidence, separate from the bootstrap lock and runtime/inference compatibility. See the [inventory design](../docs/design/gguf-inventory.md).

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
