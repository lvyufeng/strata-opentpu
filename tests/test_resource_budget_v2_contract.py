"""Governance checks for the v2 partial-tensor residency budget contract.

These tests never authenticate an input, a model artifact or a measurement.
They pin the exact bytes of the accepted v2 work as literals so that a silent
repoint of the record, schema, validator or fixtures must appear as an explicit
reviewed diff in both the record and this test. The frozen v1 baseline is
re-asserted here because v2 imports those helpers read-only.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/strata-resource-budget-v2.json"
SCHEMA = ROOT / "configs/contracts/resource-budget-v2.schema.json"
VALIDATOR = ROOT / "scripts/resource_budget_v2.py"
DESIGN = ROOT / "docs/design/resource-budget-v2.md"
TEST_MODULE = ROOT / "tests/test_resource_budget_v2.py"
FIXTURES = ROOT / "tests/fixtures"

CENTRAL_ISSUE = "https://github.com/lvyufeng/strata-opentpu/issues/20"
PARENT_ISSUE = "https://github.com/lvyufeng/strata-opentpu/issues/6"
FROZEN_V1_ISSUE = "https://github.com/lvyufeng/strata-opentpu/issues/15"

# Recorded evidence for the reviewed source commit. The head SHA is the commit
# that contains the pinned artifacts, not the later evidence-only commit.
SOURCE_COMMIT = "5caa745f9ead24e5bb30f2911ecd736dff5c3ebc"
CI_RUN_URL = "https://github.com/lvyufeng/strata-opentpu/actions/runs/45005116145"
CI_RUN_ATTEMPT = 1
REVIEW_RECORD = (
    "https://github.com/lvyufeng/strata-opentpu/issues/20#issuecomment-7076847921"
)

# Literals, not values derived from the record under test.
VALIDATOR_SHA256 = "e362eca8cd83feb3dc2270dbd17805300b4fc09b2752c0d424fa14b048bca232"
SCHEMA_SHA256 = "f45464d6ffd13330b3f6d220b25b4100d046de0701ba2672c6191fe10d0f013d"
TEST_MODULE_SHA256 = "24f971d601c84a9b291097bd8e951437316b2f4341bbd73d7d6207b884c43d90"
DESIGN_SHA256 = "8235546c48bd25d7c143d7776343dae97fed912447379e4e8ca5658ac81e872d"
V1_FROZEN_SHA256 = {
    "scripts/resource_budget.py":
        "64e012a09a1bba6151f4b606d160af9523901442ff79868982cb79536fa57f57",
    "configs/budgets/kintex7-qwen38-reference.json":
        "35006e3140392fbae6ca6435fc27fb0870ad9e29e54db1b2e44d21ef71b5515a",
    "tests/test_resource_budget.py":
        "e126fa536794850820acb4c6a46b152ff03891d877e1792105d336ddfecde41b",
    "docs/design/resource-budget.md":
        "83ef230c8cfd74d3312e9ad3df47ad3a863d0a572515e720a21d91ce5d41e608",
}
# name -> bytes, sha256, non-strict exit, strict exit, status, complete, fit scope
FIXTURE_EXPECTATIONS = {
    "full-coverage": (
        17892,
        "55c63d84605b8aab7ed7bb407ffc1a84d3e56b3a0a43d7699d8297eb7dedd80c",
        0, 0, "fit", True, "complete",
    ),
    "overlap-reject": (
        12843,
        "4fe71bc93283a60729b28f934fa28f094e8f2b278388deea2723fa3615a88cfa",
        2, 2, None, None, None,
    ),
    "partial-hot-subset": (
        15044,
        "288135f748f3fb20aa84c669b288677420a8556d2d661cc8dade399ea700746d",
        0, 1, "unknown", False, "partial_coverage",
    ),
    "range-out-of-bounds-reject": (
        11655,
        "b991f6af2310fe878eeb07c68d0aed02fa0985315e82f8f3f86657693bb0eed7",
        2, 2, None, None, None,
    ),
    "real-shape-split": (
        18137,
        "dce70ded6268cc2d5dc36c583eb6698aa3f226e043f957202fff599a2c5ff549",
        0, 0, "fit", True, "complete",
    ),
    "unknown-selection": (
        14725,
        "35a54ee445ac61e6930e63a53a4a48e115b1673ff9544592436ef9d74b0a0576",
        0, 1, "unknown", False, "partial_coverage",
    ),
}
V2_TEST_METHODS = 110
LIMITS = {
    "max_selection_items": 4096,
    "max_selections": 4096,
    "max_selections_per_tensor": 64,
    "max_tensor_records": 4096,
    "max_total_selected_items": 1000000,
    "max_expert_count": 1000000,
}
# Shapes observed in the real v2 GGUF header inventory of shard 1. Only the
# shapes are reused; every source in the fixture is synthetic.
REAL_ROLES = {
    "ffn_gate_exps": [2560, 640],
    "ffn_up_exps": [2560, 640],
    "ffn_down_exps": [640, 2560],
}
REPORT_FALSE_FLAGS = (
    "input_provenance_verified",
    "device_ready",
    "inference_ready",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_record() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def run_cli(path: Path, strict: bool = False):
    command = [sys.executable, str(VALIDATOR)]
    if strict:
        command.append("--strict")
    command.append(str(path))
    return subprocess.run(command, capture_output=True, text=True, check=False)


def load_budget_modules() -> tuple[object, object]:
    scripts = str(ROOT / "scripts")
    sys.path.insert(0, scripts)
    try:
        import resource_budget  # frozen v1 helpers, imported read-only
        import resource_budget_v2
    finally:
        sys.path.remove(scripts)
    return resource_budget, resource_budget_v2


class ResourceBudgetV2ContractTests(unittest.TestCase):
    def test_record_is_coordinated_and_pessimistic(self):
        record = load_record()
        self.assertEqual(record["schema_version"], 2)
        self.assertEqual(record["central_issue"], CENTRAL_ISSUE)
        self.assertEqual(record["parent_issue"], PARENT_ISSUE)
        self.assertEqual(record["receipt_schema"],
                         "configs/contracts/resource-budget-v2.schema.json")
        self.assertEqual(record["validator"], "scripts/resource_budget_v2.py")
        self.assertIs(record["bootstrap_lock_changed"], False)
        self.assertTrue((ROOT / record["receipt_schema"]).is_file())
        self.assertTrue((ROOT / record["validator"]).is_file())
        self.assertTrue(record["evidence_scope"])
        self.assertTrue(record["scope_exclusions"])
        self.assertEqual(set(record["claims"]), {
            "input_provenance_verified",
            "artifact_integrity_verified",
            "model_completeness_verified",
            "backend_compatibility_verified",
            "device_ready",
            "inference_ready",
            "parents_accepted",
        })
        self.assertTrue(all(value is False for value in record["claims"].values()))
        acceptance = record["acceptance"]
        # Every evidence value is pinned to a real public run and an independent
        # review record, not merely asserted as true.
        self.assertEqual(
            acceptance["kind"], "reviewed-synthetic-declaration-arithmetic")
        self.assertIs(acceptance["independent_correctness_review"], True)
        self.assertIs(acceptance["independent_scope_review"], True)
        self.assertRegex(
            acceptance["review_record"],
            r"^https://github\.com/lvyufeng/strata-opentpu/issues/20"
            r"#issuecomment-\d+$")
        self.assertEqual(acceptance["review_record"], REVIEW_RECORD)
        evidence = acceptance["ci"]
        self.assertRegex(
            evidence["url"],
            r"^https://github\.com/lvyufeng/strata-opentpu/actions/runs/\d+$")
        self.assertEqual(evidence["url"], CI_RUN_URL)
        self.assertRegex(evidence["head_sha"], r"^[0-9a-f]{40}$")
        self.assertEqual(evidence["head_sha"], SOURCE_COMMIT)
        self.assertEqual(evidence["event"], "push")
        self.assertEqual(evidence["branch"], "main")
        self.assertIsInstance(evidence["run_attempt"], int)
        self.assertGreaterEqual(evidence["run_attempt"], 1)
        self.assertEqual(evidence["run_attempt"], CI_RUN_ATTEMPT)
        self.assertEqual(evidence["conclusion"], "success")
        self.assertIs(acceptance["frozen_v1_baseline_unchanged"], True)
        self.assertIs(acceptance["schema_is_descriptive_only"], True)
        self.assertIs(record["provenance_boundaries"]["fixtures_are_synthetic"], True)
        for key, value in record["provenance_boundaries"].items():
            if isinstance(value, bool):
                self.assertIn(value, (True, False), key)

    def test_frozen_v1_baseline_is_pinned_independently(self):
        record = load_record()
        baseline = record["frozen_baseline"]
        self.assertEqual(baseline["schema_version"], 1)
        self.assertEqual(baseline["central_issue"], FROZEN_V1_ISSUE)
        self.assertIs(baseline["additive_changes_only"], True)
        self.assertEqual(baseline["files"], V1_FROZEN_SHA256)
        for relative, expected in V1_FROZEN_SHA256.items():
            self.assertEqual(digest(ROOT / relative), expected, relative)

    def test_pinned_artifacts_and_fixtures_match_recorded_digests(self):
        record = load_record()
        self.assertEqual(digest(VALIDATOR), VALIDATOR_SHA256)
        self.assertEqual(record["validator_sha256"], VALIDATOR_SHA256)
        self.assertEqual(digest(SCHEMA), SCHEMA_SHA256)
        self.assertEqual(record["receipt_schema_sha256"], SCHEMA_SHA256)
        self.assertEqual(digest(TEST_MODULE), TEST_MODULE_SHA256)
        self.assertEqual(digest(DESIGN), DESIGN_SHA256)
        self.assertEqual(record["design_document"], "docs/design/resource-budget-v2.md")
        self.assertEqual(record["design_document_sha256"], DESIGN_SHA256)
        self.assertEqual(record["test_module_sha256"], TEST_MODULE_SHA256)
        tree = ast.parse(TEST_MODULE.read_text(encoding="utf-8"))
        methods = sum(
            1 for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        )
        self.assertEqual(methods, V2_TEST_METHODS)
        self.assertEqual(record["acceptance"]["resource_budget_v2_test_methods"],
                         V2_TEST_METHODS)
        self.assertEqual(set(record["fixtures"]), set(FIXTURE_EXPECTATIONS))
        for name, (size, sha, *_) in FIXTURE_EXPECTATIONS.items():
            entry = record["fixtures"][name]
            path = ROOT / entry["path"]
            self.assertEqual(path.name, f"resource-budget-v2-{name}.json")
            self.assertEqual(entry["bytes"], size, name)
            self.assertEqual(entry["sha256"], sha, name)
            self.assertEqual(path.stat().st_size, size, name)
            self.assertEqual(digest(path), sha, name)

    def test_schema_vocabulary_matches_enforced_categories(self):
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        frozen, v2 = load_budget_modules()
        self.assertIs(v2.CATEGORIES, frozen.CATEGORIES)
        definitions = schema["$defs"]
        self.assertEqual(set(definitions["memoryCoverage"]["required"]),
                         set(frozen.CATEGORIES))
        self.assertEqual(len(definitions["memoryCoverage"]["required"]),
                         len(frozen.CATEGORIES))
        self.assertEqual(set(definitions["allocation"]["properties"]["category"]["enum"]),
                         set(frozen.CATEGORIES))
        self.assertEqual(set(definitions["wallCoverage"]["required"]),
                         set(frozen.WALL_CATEGORIES))
        for scope, definition in (
            ("layer", "stageCoverage"),
            ("non_expert", "nonExpertCoverage"),
            ("mtp", "mtpCoverage"),
        ):
            self.assertEqual(set(definitions[definition]["required"]),
                             set(frozen.STAGE_CATEGORIES[scope]), scope)
        self.assertEqual(set(schema["required"]),
                         {"schema_version", "id", "input_kind", "selection_contract",
                          "model", "memory", "wall"})
        self.assertEqual(
            definitions["selectionContract"]["required"],
            ["convention", "expert_dimension", "expert_count", "roles"])
        self.assertEqual(definitions["tensor"]["properties"]["role"]["enum"],
                         list(v2.ROLES))
        self.assertEqual(definitions["selection"]["properties"]["mode"]["enum"],
                         list(v2.SELECTION_MODES))
        self.assertEqual(definitions["stage"]["properties"]["scope"]["enum"],
                         ["layer", "non_expert", "mtp"])
        self.assertEqual(definitions["operation"]["properties"]["kind"]["enum"],
                         ["cpu", "fpga", "host", "sync", "merge", "h2d", "d2h"])
        self.assertEqual(record_limits := load_record()["limits"], LIMITS)
        self.assertEqual(record_limits["max_selection_items"],
                         v2.MAX_SELECTION_ITEMS)
        self.assertEqual(record_limits["max_selections"], v2.MAX_SELECTIONS)
        self.assertEqual(record_limits["max_selections_per_tensor"],
                         v2.MAX_SELECTIONS_PER_TENSOR)
        self.assertEqual(record_limits["max_tensor_records"],
                         v2.MAX_TENSOR_RECORDS)
        self.assertEqual(record_limits["max_total_selected_items"],
                         v2.MAX_TOTAL_SELECTED_ITEMS)
        self.assertEqual(record_limits["max_expert_count"], v2.MAX_EXPERT_COUNT)
        self.assertEqual(schema["$schema"],
                         "https://json-schema.org/draft/2020-12/schema")

    def test_validator_is_stdlib_only_and_never_reads_artifacts(self):
        source = VALIDATOR.read_text(encoding="utf-8")
        tree = ast.parse(source)
        allowed = set(sys.stdlib_module_names) | {"resource_budget"}
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                if node.module:
                    imported.add(node.module.split(".")[0])
        self.assertTrue(imported)
        self.assertLessEqual(imported, allowed)
        forbidden = {"subprocess", "socket", "urllib", "http", "requests",
                     "huggingface_hub", "numpy", "jsonschema"}
        self.assertFalse(imported & forbidden)
        for marker in ("gguf", "jsonschema", "Draft202012Validator",
                       ".schema.json", "model.gguf"):
            self.assertNotIn(marker, source.lower(), marker)

    def test_fixture_matrix_exit_codes_and_declaration_only_reports(self):
        for name, (_, _, exit_code, strict_exit, status, complete,
                   fit_scope) in FIXTURE_EXPECTATIONS.items():
            path = FIXTURES / f"resource-budget-v2-{name}.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(document["input_kind"], "synthetic", name)
            result = run_cli(path)
            self.assertEqual(result.returncode, exit_code, name + result.stderr)
            if status is None:
                self.assertEqual(result.stdout, "", name)
                self.assertIn("resource budget v2 input error:", result.stderr)
                self.assertEqual(run_cli(path, strict=True).returncode, 2, name)
                continue
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], status, name)
            self.assertIs(report["complete"], complete, name)
            self.assertEqual(report["fit_scope"], fit_scope, name)
            self.assertIs(report["analytical_only"], True)
            for flag in REPORT_FALSE_FLAGS:
                self.assertIs(report[flag], False, f"{name}:{flag}")
            strict = run_cli(path, strict=True)
            self.assertEqual(strict.returncode, strict_exit, name)
            if strict_exit == 0:
                self.assertIs(json.loads(strict.stdout)["strict_pass"], True)
            else:
                self.assertIs(json.loads(strict.stdout)["strict_pass"], False)

    def test_real_shape_fixture_binds_the_declared_512_expert_axis(self):
        record = load_record()
        selection = record["selection_contract"]
        self.assertEqual(selection["convention"], "last_dimension")
        self.assertEqual(selection["expert_dimension"], 2)
        self.assertEqual(selection["expert_count"], 512)
        self.assertEqual(selection["roles"], REAL_ROLES)
        path = FIXTURES / "resource-budget-v2-real-shape-split.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        fixture_contract = document["selection_contract"]
        self.assertEqual(fixture_contract,
                         {key: selection[key] for key in fixture_contract})
        expected_shapes = {
            "ffn_gate_exps": [2560, 640, 512],
            "ffn_up_exps": [2560, 640, 512],
            "ffn_down_exps": [640, 2560, 512],
        }
        roles = set()
        for tensor in document["memory"]["tensors"]:
            roles.add(tensor["role"])
            self.assertEqual(tensor["packed"]["shape"],
                             expected_shapes[tensor["role"]], tensor["id"])
            self.assertEqual(tensor["source"]["kind"], "synthetic", tensor["id"])
            self.assertTrue(tensor["source"]["artifact"].endswith(".synthetic"))
        self.assertEqual(roles, set(REAL_ROLES))
        report = json.loads(run_cli(path).stdout)
        coverage = report["selection_coverage"]
        self.assertIs(coverage["full_coverage"], True)
        self.assertEqual(coverage["declared_items"], coverage["accounted_items"])
        self.assertEqual(coverage["expert_axis_size"], 512)
        self.assertTrue(record["provenance_boundaries"]["hot_cold_split_is_hypothetical"])


if __name__ == "__main__":
    unittest.main()