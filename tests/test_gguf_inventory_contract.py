"""Governance and cached-only CI checks; never import or execute fork source."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/strata-gguf-inventory.json"
CONTRACT_V2 = ROOT / "configs/contracts/strata-gguf-inventory-v2.json"
WORKFLOW = ROOT / ".github/workflows/bootstrap.yml"
BASE = "3142b66b96e9c0e775e30f620b11b191a1427c2a"
PARSER_SHA = "c84a0f88c6c75a588cb7de74abd23b2e6387c92c58e088012aedef5bd3c6c431"
# The accepted v2 development commit and the exact bytes reviewed there. These are
# literals, not derived from the record under test: a silent repoint, a raised test
# count or a file edit must be an explicit, reviewed diff in both places.
V2_REVISION = "97affb754d298b2cce1fc7b12338797ac155aa9b"
V2_TREE = "e87f198acd761b9e7b855b90b9aeee992cee07fb"
V2_CI_URL = "https://github.com/lvyufeng/Strata/actions/runs/44990573044"
V2_FILE_SHA256 = {
    "terminal/tools/gguf_inventory.py": "a0b242b2491f5aac76a9e07fd4039ccfe62e8d3705120459aa6f3ad9d1ad4c05",
    "terminal/tools/quant_formats.py": "2de1ed0e085260a0dfb7a6369003a7b6a1efee7b4686005d5da0e5d39ccb338d",
    "terminal/tools/test_gguf_inventory.py": "d8d4a670ed1ffcbfa6b31e11cf9b6ac13619e81cf5dfaed65fa80d61db6bcf22",
    "docs/QUANTS.md": "b9204eb8783cd37171ea009eaa2f9a4ee883dd3170c31c8bebcfe696ce5ecf69",
    ".github/workflows/gguf-inventory.yml": "6a2bc652979a3da1f8c3da87e876f9f60e7cef8e97e7462a6877b39983453e3a",
}


class InventorySourceContractTests(unittest.TestCase):
    def test_source_receipt_is_separate_from_bootstrap_and_runtime(self):
        record = json.loads(CONTRACT.read_text(encoding="utf-8"))
        lock = json.loads((ROOT / "dependencies.lock.json").read_text(encoding="utf-8"))
        self.assertEqual(record["central_issue"], "https://github.com/lvyufeng/strata-opentpu/issues/18")
        self.assertEqual(record["parent_issue"], "https://github.com/lvyufeng/strata-opentpu/issues/6")
        self.assertEqual(record["source"]["base_revision"], BASE)
        self.assertEqual(lock["dependencies"]["strata"]["revision"], BASE)
        self.assertEqual(lock["dependencies"]["opentpu"]["revision"],
                         "d84db55712aba0d40811d7762cb052f05d55b7e5")
        self.assertIs(record["bootstrap_lock_changed"], False)
        self.assertEqual(record["source"]["parser"]["sha256"], PARSER_SHA)
        self.assertIs(record["source"]["parser"]["changed"], False)
        self.assertEqual(record["source"]["parser"]["reuse"],
                         ["GGUFFile._parse", "GGUFFile._value"])
        self.assertIsNone(record["source"]["pull_request"])
        self.assertIs(record["source"]["upstream_scope_agreement"], False)
        self.assertIs(record["source"]["merged"], False)
        self.assertTrue(record["claims"])
        self.assertTrue(all(value is False for value in record["claims"].values()))
        self.assertTrue((ROOT / record["receipt_schema"]).is_file())
        self.assertTrue((ROOT / record["validator"]).is_file())

    def test_no_provenance_integrity_or_confinement_promotion(self):
        record = json.loads(CONTRACT.read_text(encoding="utf-8"))
        bounds = record["provenance_boundaries"]
        self.assertIs(bounds["input_kind_is_declaration"], True)
        self.assertIs(bounds["producer_identifies_base_parser_not_current_executable"], True)
        self.assertEqual(bounds["header_sha256_scope"], "consumed-logical-header-only")
        self.assertIsNone(bounds["artifact_sha256"])
        self.assertIsNone(bounds["tensor_sha256"])
        self.assertIs(bounds["payload_read"], False)
        self.assertEqual(bounds["split_complete_means"], "declared-header-set-only")
        self.assertEqual(bounds["unknown_storage_means"], "null-not-zero-fit-or-backend-supported")
        for name in ("logical_limits_are_os_sandbox", "same_fd_checks_are_race_proof",
                     "json_validator_executes_source", "json_validator_opens_referenced_artifacts",
                     "json_validator_verifies_source_identity"):
            self.assertIs(bounds[name], False)

    def test_acceptance_cannot_be_inferred_from_pending_placeholders(self):
        record = json.loads(CONTRACT.read_text(encoding="utf-8"))
        accepted = record["acceptance"]
        self.assertEqual(accepted["local_python_versions"], ["3.11", "3.14"])
        if accepted["kind"] == "pending-source-review-and-public-ci":
            self.assertIsNone(record["source"]["revision"])
            self.assertIsNone(record["source"]["tree"])
            self.assertEqual(record["source"]["file_sha256"], {})
            self.assertIsNone(accepted["ci"])
            for name in ("independent_correctness_review", "independent_scope_review",
                         "literal_legacy_and_mock_hf_byte_baselines",
                         "synthetic_source_to_json_validator"):
                self.assertIs(accepted[name], False)
        else:
            self.assertEqual(accepted["kind"], "reviewed-synthetic-source-preparation")
            self.assertRegex(record["source"]["revision"], r"^[0-9a-f]{40}$")
            self.assertNotEqual(record["source"]["revision"], BASE)
            self.assertRegex(record["source"]["tree"], r"^[0-9a-f]{40}$")
            self.assertEqual(set(record["source"]["file_sha256"]), {
                "terminal/tools/gguf_inventory.py", "terminal/tools/quant_formats.py",
                "terminal/tools/test_gguf_inventory.py", "docs/QUANTS.md",
                ".github/workflows/gguf-inventory.yml",
            })
            for digest in record["source"]["file_sha256"].values():
                self.assertRegex(digest, r"^[0-9a-f]{64}$")
            for name in ("independent_correctness_review", "independent_scope_review",
                         "literal_legacy_and_mock_hf_byte_baselines",
                         "synthetic_source_to_json_validator"):
                self.assertIs(accepted[name], True)
            for name in ("source_test_count_per_python", "e2e_check_count_per_python"):
                self.assertIs(type(accepted[name]), int)
                self.assertGreater(accepted[name], 0)
            ci = accepted["ci"]
            self.assertRegex(ci["url"], r"^https://github\.com/lvyufeng/Strata/actions/runs/[0-9]+$")
            self.assertEqual(ci["head_sha"], record["source"]["revision"])
            self.assertEqual(ci["event"], "push")
            self.assertEqual(ci["conclusion"], "success")
            self.assertEqual(ci["python_versions"], ["3.11", "3.14"])
            self.assertEqual(ci["branch"], "bounded-gguf-inventory")
            self.assertEqual(ci["run_attempt"], 1)

    def test_documentation_tracks_source_only_child_and_remaining_gates(self):
        for name in ("README.md", "docs/development.md", "docs/design/gguf-inventory.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("https://github.com/lvyufeng/strata-opentpu/issues/18", text)
        design = (ROOT / "docs/design/gguf-inventory.md").read_text(encoding="utf-8")
        for phrase in ("not actual mixed-pack evidence", "not OS page-read", "race-proof",
                       "shape[0]", "null", "--strict-complete", "source_identity_verified: false",
                       "expert_skip=0", "Never add standalone CPU and FPGA token rates"):
            self.assertIn(phrase, design)
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("python scripts/check_gguf_inventory.py", workflow)
        self.assertIn("JSON only, no source execution", workflow)
        self.assertNotIn("actions/setup-python", workflow)
        self.assertNotIn("pip install", workflow)
        self.assertNotIn("uv ", workflow)
        self.assertNotIn("npm ", workflow)
        self.assertIn("python-version: ['3.11', '3.14']", workflow)


class InventoryV2SourceContractTests(unittest.TestCase):
    def test_v2_source_receipt_is_separate_from_bootstrap_and_runtime(self):
        record = json.loads(CONTRACT_V2.read_text(encoding="utf-8"))
        lock = json.loads((ROOT / "dependencies.lock.json").read_text(encoding="utf-8"))
        self.assertEqual(record["schema_version"], 2)
        self.assertEqual(record["central_issue"],
                         "https://github.com/lvyufeng/strata-opentpu/issues/19")
        self.assertEqual(record["parent_issue"],
                         "https://github.com/lvyufeng/strata-opentpu/issues/6")
        self.assertEqual(record["source"]["base_revision"], BASE)
        self.assertEqual(lock["dependencies"]["strata"]["revision"], BASE)
        self.assertEqual(lock["dependencies"]["opentpu"]["revision"],
                         "d84db55712aba0d40811d7762cb052f05d55b7e5")
        self.assertIs(record["bootstrap_lock_changed"], False)
        self.assertRegex(record["source"]["revision"], r"^[0-9a-f]{40}$")
        self.assertNotEqual(record["source"]["revision"], BASE)
        self.assertRegex(record["source"]["tree"], r"^[0-9a-f]{40}$")
        self.assertEqual(set(record["source"]["file_sha256"]), {
            "terminal/tools/gguf_inventory.py", "terminal/tools/quant_formats.py",
            "terminal/tools/test_gguf_inventory.py", "docs/QUANTS.md",
            ".github/workflows/gguf-inventory.yml",
        })
        for digest in record["source"]["file_sha256"].values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(record["source"]["parser"]["sha256"], PARSER_SHA)
        self.assertIs(record["source"]["parser"]["changed"], False)
        self.assertEqual(record["source"]["parser"]["reuse"],
                         ["GGUFFile._parse", "GGUFFile._value"])
        self.assertEqual(record["source"]["license"], "MIT")
        self.assertIsNone(record["source"]["pull_request"])
        self.assertIs(record["source"]["upstream_scope_agreement"], False)
        self.assertIs(record["source"]["merged"], False)
        self.assertTrue(record["claims"])
        self.assertTrue(all(value is False for value in record["claims"].values()))
        self.assertTrue((ROOT / record["receipt_schema"]).is_file())
        self.assertTrue((ROOT / record["validator"]).is_file())

    def test_v2_raises_only_the_two_array_item_ceilings(self):
        v1 = json.loads(CONTRACT.read_text(encoding="utf-8"))
        v2 = json.loads(CONTRACT_V2.read_text(encoding="utf-8"))
        self.assertEqual(set(v2["limits"]), set(v1["limits"]))
        raised = {name for name in v1["limits"] if v1["limits"][name] != v2["limits"][name]}
        self.assertEqual(raised, {"max_array_items_per_file", "max_array_items_total"})
        self.assertEqual((v1["limits"]["max_array_items_per_file"],
                          v2["limits"]["max_array_items_per_file"]), (500000, 2000000))
        self.assertEqual((v1["limits"]["max_array_items_total"],
                          v2["limits"]["max_array_items_total"]), (1000000, 4000000))
        changes = v2["profile_changes"]
        self.assertEqual(changes["from_schema_version"], 1)
        self.assertEqual(changes["to_schema_version"], 2)
        self.assertEqual(changes["producer_mode"], "bounded-local-v2")
        self.assertEqual(changes["raised_limits"]["max_array_items_per_file"],
                         {"v1": 500000, "v2": 2000000})
        self.assertEqual(changes["raised_limits"]["max_array_items_total"],
                         {"v1": 1000000, "v2": 4000000})
        self.assertEqual(changes["binding_v1_limit"], "max_array_items_per_file")
        self.assertEqual(changes["measured_array_items_in_one_file"], 744317)
        self.assertEqual(changes["added_confirmed_type_ids"], [14, 16, 20, 22, 23, 29])
        self.assertEqual(changes["still_unconfirmed_type_ids"], [19, 39, 40, 41])

    def test_v2_acceptance_is_reviewed_and_ci_is_public(self):
        record = json.loads(CONTRACT_V2.read_text(encoding="utf-8"))
        accepted = record["acceptance"]
        self.assertEqual(accepted["kind"], "reviewed-synthetic-source-preparation")
        self.assertEqual(accepted["local_python_versions"], ["3.11", "3.14"])
        self.assertEqual(record["source"]["revision"], V2_REVISION)
        self.assertEqual(record["source"]["tree"], V2_TREE)
        self.assertEqual(record["source"]["file_sha256"], V2_FILE_SHA256)
        for name in ("independent_correctness_review", "independent_scope_review",
                     "literal_legacy_and_mock_hf_byte_baselines",
                     "synthetic_source_to_json_validator"):
            self.assertIs(accepted[name], True)
        self.assertEqual(accepted["review_record"],
                         "https://github.com/lvyufeng/strata-opentpu/issues/19#issuecomment-7075409283")
        self.assertIs(accepted["v1_receipt_surface_unchanged"], True)
        self.assertIs(accepted["real_header_only_run"], False)
        for name in ("source_test_count_per_python",
                     "integration_receipt_test_count_per_python",
                     "integration_receipt_subtest_count_per_python"):
            self.assertIs(type(accepted[name]), int)
            self.assertGreater(accepted[name], 0)
        self.assertEqual(accepted["source_test_count_per_python"], 87)
        self.assertEqual(accepted["integration_receipt_test_count_per_python"], 70)
        self.assertEqual(accepted["integration_receipt_subtest_count_per_python"], 166)
        ci = accepted["ci"]
        self.assertEqual(ci["url"], V2_CI_URL)
        self.assertEqual(ci["head_sha"], V2_REVISION)
        self.assertEqual(ci["event"], "push")
        self.assertEqual(ci["conclusion"], "success")
        self.assertEqual(ci["python_versions"], ["3.11", "3.14"])
        self.assertEqual(ci["branch"], "bounded-gguf-inventory")
        self.assertEqual(ci["run_attempt"], 1)
        self.assertEqual(ci["test_count_per_python"],
                         accepted["source_test_count_per_python"])

    def test_v2_boundaries_do_not_promote_provenance_or_integrity(self):
        record = json.loads(CONTRACT_V2.read_text(encoding="utf-8"))
        bounds = record["provenance_boundaries"]
        self.assertIs(bounds["input_kind_is_declaration"], True)
        self.assertIs(bounds["producer_identifies_base_parser_not_current_executable"], True)
        self.assertEqual(bounds["header_sha256_scope"], "consumed-logical-header-only")
        self.assertIsNone(bounds["artifact_sha256"])
        self.assertIsNone(bounds["tensor_sha256"])
        self.assertIs(bounds["payload_read"], False)
        self.assertEqual(bounds["split_complete_means"], "declared-header-set-only")
        self.assertEqual(bounds["unknown_storage_means"],
                         "null-not-zero-fit-or-backend-supported")
        for name in ("logical_limits_are_os_sandbox", "same_fd_checks_are_race_proof",
                     "json_validator_executes_source",
                     "json_validator_opens_referenced_artifacts",
                     "json_validator_verifies_source_identity"):
            self.assertIs(bounds[name], False)
        self.assertEqual(record["evidence_scope"],
                         "source-and-synthetic-header-preparation")
        local = bounds["local_real_header_receipt"]
        self.assertEqual(local["retained"], "ignored-local-workspace-only")
        self.assertIs(local["committed"], False)
        self.assertIs(local["accepted_evidence"], False)
        self.assertRegex(local["sha256"], r"^[0-9a-f]{64}$")
        self.assertIs(type(local["bytes"]), int)
        self.assertEqual(local["schema_version"], 2)
        self.assertEqual(local["input_kind"], "local-unverified")
        self.assertEqual(local["reported_totals"], {
            "tensor_count": 1224,
            "known_storage_bytes": 68015068160,
            "unknown_storage_tensors": 0,
        })

    def test_v2_documentation_records_the_profile_delta_and_gates(self):
        design = (ROOT / "docs/design/gguf-inventory.md").read_text(encoding="utf-8")
        for phrase in ("issues/19", "--receipt-v2", "2,000,000", "4,000,000",
                       "744,317", "IQ1_M", "bounded-local-v2",
                       "still includes IQ1_S", "not actual mixed-pack evidence",
                       "real header-only receipt"):
            self.assertIn(phrase, design)
        self.assertTrue((ROOT / "configs/contracts/gguf-header-inventory-v2.schema.json").is_file())
        self.assertTrue((ROOT / "tests/fixtures/gguf-header-inventory-v2.synthetic.json").is_file())
        self.assertTrue((ROOT / "configs/contracts/strata-gguf-inventory.json").is_file())


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "cached locator uses POSIX bash")
class CachedPythonSelectionTests(unittest.TestCase):
    @staticmethod
    def locator_script():
        lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
        start = next(i for i, line in enumerate(lines)
                     if "Select preinstalled cached Python" in line)
        run = next(i for i in range(start, len(lines)) if lines[i] == "        run: |")
        script = []
        for line in lines[run + 1:]:
            if not line.startswith("          "):
                break
            script.append(line[10:])
        return "\n".join(script) + "\n"

    def locate(self, directory, minor):
        env = dict(os.environ, RUNNER_TOOL_CACHE=str(directory / "cache"),
                   PYTHON_MINOR=minor, GITHUB_PATH=str(directory / "github-path"))
        return subprocess.run(["bash", "-c", self.locator_script()], env=env,
                              cwd=directory, capture_output=True, text=True, timeout=10)

    def test_absent_cache_fails_without_installing_or_creating_path(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            result = self.locate(directory, "3.11")
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("installation is not permitted", result.stderr)
            self.assertFalse((directory / "github-path").exists())
            self.assertEqual(list(directory.iterdir()), [])

    def test_cached_matching_interpreter_is_selected_without_fallback(self):
        minor = ".".join(map(str, sys.version_info[:2]))
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            bindir = directory / "cache" / "Python" / (minor + ".99") / "x64" / "bin"
            bindir.mkdir(parents=True)
            (bindir / "python").symlink_to(sys.executable)
            result = self.locate(directory, minor)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((directory / "github-path").read_text(encoding="utf-8"), str(bindir) + "\n")
            self.assertIn("Python " + minor + ".", result.stdout)

    def test_cache_directory_label_cannot_substitute_wrong_interpreter_version(self):
        minor = ".".join(map(str, sys.version_info[:2]))
        incorrect = "3.14" if minor == "3.11" else "3.11"
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            bindir = directory / "cache" / "Python" / (incorrect + ".99") / "x64" / "bin"
            bindir.mkdir(parents=True)
            (bindir / "python").symlink_to(sys.executable)
            result = self.locate(directory, incorrect)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("installation is not permitted", result.stderr)
            self.assertFalse((directory / "github-path").exists())


if __name__ == "__main__":
    unittest.main()
