# Copyright (c) 2026 lvyufeng and contributors
# SPDX-License-Identifier: Apache-2.0
"""Hardware-free receipt/integrity tests; the scalar math lives in openTPU."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import check_packed_oracle as checker

ROOT = Path(__file__).resolve().parents[1]


def contract_data():
    return json.loads(checker.DEFAULT_CONTRACT.read_text(encoding="utf-8"))


def lock_data():
    return json.loads((ROOT / "dependencies.lock.json").read_text(encoding="utf-8"))


def tiny_fixture():
    """Schema-only fixture; no integration copy of the quantization algorithm."""
    contract = contract_data()
    document = {
        "schema_version": 1, "fixture_id": "q2-q8-supplied-packed-v1", "synthetic": True,
        "arithmetic_id": checker.ARITHMETIC_ID, "formats": deepcopy(checker.FORMATS),
        "provenance": {
            "strata": {**contract["sources"]["strata"], "anchors": deepcopy(checker.ANCHORS)},
            "opentpu_base": deepcopy(contract["sources"]["opentpu_base"]),
            **deepcopy({key: value for key, value in checker.PROVENANCE.items() if key != "strata_anchors"}),
        },
        "cases": [
            {"id": "schema-q2", "operation": "decode", "format": "Q2_0", "elements": 64,
             "payload_hex": "003c" + "55" * 16, "expected_fp32_bits": ["0x00000000"] * 64},
            {"id": "schema-q8", "operation": "dot", "weight_format": "Q8_0", "activation_format": "Q8_0", "elements": 32,
             "weight_hex": "003c" + "00" * 32, "activation_hex": "003c" + "00" * 32,
             "expected_fp32_bits": "0x00000000", "expected_integer_subdots": [[0]]},
        ],
    }
    document["payload_sha256"] = checker.canonical_digest(document)
    return document


def serialize(document):
    return (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def complete_contract(document=None):
    contract = contract_data()
    document = tiny_fixture() if document is None else document
    contract["fixture"].update({"payload_sha256": document["payload_sha256"],
                                "file_sha256": hashlib.sha256(serialize(document)).hexdigest(),
                                "case_count": len(document["cases"]),
                                "decode_cases": sum(case["operation"] == "decode" for case in document["cases"]),
                                "dot_cases": sum(case["operation"] == "dot" for case in document["cases"])})
    contract["oracle"]["revision"] = "a" * 40
    contract["acceptance"]["independent_math_review"] = True
    contract["acceptance"]["independent_scope_license_review"] = True
    contract["acceptance"]["ci"].update({"url": "https://github.com/lvyufeng/openTPU/actions/runs/123456789",
                                         "head_sha": "a" * 40, "conclusion": "success"})
    return contract


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.contract = complete_contract()
        self.lock = lock_data()

    def assert_invalid(self, document, lock=None):
        with self.assertRaises(checker.ContractError):
            checker.validate(document, self.lock if lock is None else lock)

    def test_checked_in_receipt_is_consistent_without_source_execution(self):
        with patch.object(checker.subprocess, "run", side_effect=AssertionError("metadata must not execute")):
            report = checker.validate(contract_data(), self.lock)
        self.assertFalse(report["oracle_executed"])
        self.assertFalse(report["source_identity_checked"])
        self.assertFalse(report["synthetic_vectors_verified"])
        self.assertFalse(report["device_ready"])
        self.assertFalse(report["inference_ready"])

    def test_complete_receipt_is_only_recorded_metadata(self):
        report = checker.validate(self.contract, self.lock)
        self.assertTrue(report["receipt_complete"])
        self.assertEqual(report["missing"], [])
        self.assertFalse(report["oracle_executed"])
        self.assertIn("truth", report["note"])

    def test_unknown_data_stays_incomplete(self):
        self.contract["fixture"]["file_sha256"] = None
        self.contract["acceptance"]["independent_math_review"] = False
        report = checker.validate(self.contract, self.lock)
        self.assertFalse(report["receipt_complete"])
        self.assertIn("fixture.file_sha256", report["missing"])
        self.assertIn("acceptance.independent_math_review", report["missing"])

    def test_fields_cannot_enable_execution_or_readiness(self):
        for field in ("device_ready", "inference_ready", "command", "download_url"):
            with self.subTest(field=field):
                changed = deepcopy(self.contract)
                changed[field] = True
                self.assert_invalid(changed)

    def test_boolean_not_integer_schema_or_layout(self):
        for route in (("schema_version",), ("formats", "Q2_0", "ggml_type"), ("limits", "max_cases"), ("fixture", "case_count")):
            with self.subTest(route=route):
                changed = deepcopy(self.contract)
                cursor = changed
                for key in route[:-1]:
                    cursor = cursor[key]
                cursor[route[-1]] = True
                self.assert_invalid(changed)

    def test_unknown_or_confused_formats_rejected(self):
        for value in ("IQ2_XS", "Q2_K", "FP4"):
            with self.subTest(value=value):
                changed = deepcopy(self.contract)
                changed["formats"][value] = changed["formats"].pop("Q2_0")
                self.assert_invalid(changed)

    def test_exact_base_identities_must_match_lock(self):
        for source in ("strata", "opentpu_base"):
            for field, value in (("revision", "b" * 40), ("repository", "https://github.com/other/repo")):
                with self.subTest(source=source, field=field):
                    changed = deepcopy(self.contract)
                    changed["sources"][source][field] = value
                    self.assert_invalid(changed)

    def test_lock_must_be_bootstrap_and_not_inference(self):
        for field, value in (("purpose", "inference"), ("compatibility_status", "validated"), ("schema_version", True), ("dependencies", [])):
            with self.subTest(field=field):
                lock = deepcopy(self.lock)
                lock[field] = value
                self.assert_invalid(self.contract, lock)

    def test_revision_and_digest_are_exact_lowercase_hex(self):
        for value in ("A" * 40, "a" * 39, "latest", 123, True):
            with self.subTest(value=value):
                changed = deepcopy(self.contract)
                changed["oracle"]["revision"] = value
                self.assert_invalid(changed)
        for value in ("A" * 64, "a" * 63, "unknown", True):
            with self.subTest(value=value):
                changed = deepcopy(self.contract)
                changed["fixture"]["payload_sha256"] = value
                self.assert_invalid(changed)

    def test_original_base_does_not_contain_oracle(self):
        self.contract["oracle"]["revision"] = self.contract["sources"]["opentpu_base"]["revision"]
        self.contract["acceptance"]["ci"]["head_sha"] = self.contract["oracle"]["revision"]
        self.assert_invalid(self.contract)

    def test_ci_cannot_point_to_other_commit_or_repository(self):
        for field, value in (("head_sha", "b" * 40), ("url", "https://github.com/other/openTPU/actions/runs/12"),
                             ("conclusion", "failure"), ("workflow", "unrelated tests"), ("event", "workflow_dispatch"),
                             ("python_versions", ["3.11"])):
            with self.subTest(field=field):
                changed = deepcopy(self.contract)
                changed["acceptance"]["ci"][field] = value
                self.assert_invalid(changed)

    def test_path_and_module_cannot_inject_other_commands(self):
        for route, value in ((("fixture", "path"), "../../outside.json"), (("oracle", "module"), "opentpu.rtlsim"),
                             (("oracle", "workflow_path"), "/tmp/other.yml")):
            with self.subTest(route=route):
                changed = deepcopy(self.contract)
                cursor = changed[route[0]]
                cursor[route[1]] = value
                self.assert_invalid(changed)

    def test_counts_bounded_and_consistent(self):
        for key, value in (("case_count", 257), ("decode_cases", 0), ("dot_cases", -1), ("case_count", 3)):
            with self.subTest(key=key, value=value):
                changed = deepcopy(self.contract)
                changed["fixture"][key] = value
                self.assert_invalid(changed)

    def test_full_license_and_independent_provenance_cannot_be_omitted(self):
        for field in ("licenses", "provenance"):
            changed = deepcopy(self.contract)
            changed[field] = {} if field == "provenance" else []
            self.assert_invalid(changed)
        changed = deepcopy(self.contract)
        changed["provenance"]["expected_derivation"] = "generated by DUT"
        self.assert_invalid(changed)

    def test_review_fields_are_explicit_boolean(self):
        for value in (1, "true", None):
            self.contract["acceptance"]["independent_math_review"] = value
            self.assert_invalid(self.contract)


class FixtureMetadataTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tiny_fixture()
        self.contract = complete_contract(self.fixture)

    def check_bad(self, document, resign=True):
        if resign and type(document) is dict:
            document["payload_sha256"] = checker.canonical_digest(document)
        with self.assertRaises(checker.ContractError):
            checker.validate_fixture(document, self.contract)

    def test_valid_integrity_and_counts_without_arithmetic(self):
        with patch.object(checker.subprocess, "run", side_effect=AssertionError("fixture must not execute")):
            report = checker.validate_fixture(self.fixture, self.contract, hashlib.sha256(serialize(self.fixture)).hexdigest())
        self.assertTrue(report["fixture_metadata_checked"])
        self.assertEqual(report["case_count"], 2)
        self.assertNotIn("synthetic_vectors_verified", report)

    def test_payload_and_file_digest_pin(self):
        changed = deepcopy(self.fixture)
        changed["cases"][0]["payload_hex"] = "003c" + "aa" * 16
        self.check_bad(changed, resign=False)
        self.check_bad(changed, resign=True)
        with self.assertRaises(checker.ContractError):
            checker.validate_fixture(self.fixture, self.contract, "b" * 64)

    def test_provenance_identity_and_scope_cannot_be_changed(self):
        for route, value in ((("strata", "revision"), "b" * 40), (("generator", "id"), "DUT"),
                             (("strata", "anchors"), [])):
            with self.subTest(route=route):
                changed = deepcopy(self.fixture)
                changed["provenance"][route[0]][route[1]] = value
                self.check_bad(changed)
        changed = deepcopy(self.fixture)
        changed["provenance"]["scope_exclusions"] = []
        self.check_bad(changed)

    def test_extra_missing_or_wrong_schema_fields(self):
        for field in ("schema_version", "synthetic", "cases"):
            changed = deepcopy(self.fixture)
            del changed[field]
            self.check_bad(changed)
        changed = deepcopy(self.fixture)
        changed["device_ready"] = True
        self.check_bad(changed)
        changed = deepcopy(self.fixture)
        changed["schema_version"] = True
        self.check_bad(changed)

    def test_ids_unique_and_bounded(self):
        for value in ("schema-q2", "../path", "", "a" * 200, [], 1):
            with self.subTest(value=value):
                changed = deepcopy(self.fixture)
                changed["cases"][1]["id"] = value
                self.check_bad(changed)

    def test_dimensions_block_multiple_and_bounded(self):
        for value in (True, 0, -64, 63, 65537, 64.0, "64"):
            changed = deepcopy(self.fixture)
            changed["cases"][0]["elements"] = value
            self.check_bad(changed)

    def test_payload_contiguous_canonical_hex_with_no_trailing_or_missing(self):
        for value in ("00", self.fixture["cases"][0]["payload_hex"] + "00", "00 3c" + "55" * 16,
                      "003C" + "55" * 16, "003c" + "gg" * 16, None):
            changed = deepcopy(self.fixture)
            changed["cases"][0]["payload_hex"] = value
            self.check_bad(changed)

    def test_expected_bits_exact_lowercase_and_count(self):
        for value in (["0x0"] * 64, ["0x00000000"] * 63, [True] * 64, "0x00000000", ["0X00000000"] * 64):
            changed = deepcopy(self.fixture)
            changed["cases"][0]["expected_fp32_bits"] = value
            self.check_bad(changed)

    def test_integer_subdots_have_exact_shape_integer_and_bounds(self):
        for value in ([[True]], [], [[0, 0]], [[524289]], [[-520193]], [[0.0]]):
            changed = deepcopy(self.fixture)
            changed["cases"][1]["expected_integer_subdots"] = value
            self.check_bad(changed)

    def test_q2_integer_subdots_contract(self):
        changed = deepcopy(self.fixture)
        changed["cases"][1].update({"weight_format": "Q2_0", "elements": 64,
                                    "weight_hex": "003c" + "55" * 16,
                                    "activation_hex": ("003c" + "00" * 32) * 2,
                                    "expected_integer_subdots": [[-8192, 8128]]})
        changed["payload_sha256"] = checker.canonical_digest(changed)
        contract = complete_contract(changed)
        checker.validate_fixture(changed, contract)
        changed["cases"][1]["expected_integer_subdots"] = [[-8193, 8128]]
        self.check_bad(changed)

    def test_unsupported_format_activation_operation(self):
        for key, value in (("weight_format", "IQ2_XS"), ("activation_format", "Q8_K"), ("operation", "quantize")):
            changed = deepcopy(self.fixture)
            changed["cases"][1][key] = value
            self.check_bad(changed)

    def test_case_counts_and_pinned_canonical_digest(self):
        for cases in ([], self.fixture["cases"][:1], self.fixture["cases"] * 129):
            changed = deepcopy(self.fixture)
            changed["cases"] = cases
            self.check_bad(changed)

    def test_canonical_digest_not_dependent_on_serialized_key_order(self):
        alternate = dict(reversed(list(self.fixture.items())))
        self.assertEqual(checker.canonical_digest(self.fixture), checker.canonical_digest(alternate))

    def test_fixture_metadata_does_not_certify_expected_arithmetic_truth(self):
        changed = deepcopy(self.fixture)
        changed["cases"][1]["expected_fp32_bits"] = "0x3f800000"
        changed["payload_sha256"] = checker.canonical_digest(changed)
        report = checker.validate_fixture(changed, complete_contract(changed))
        self.assertTrue(report["fixture_metadata_checked"])
        self.assertNotIn("verified", report)


class SourceExecutionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.checkout = Path(self.directory.name).resolve()
        self.fixture = tiny_fixture()
        self.contract = complete_contract(self.fixture)
        self.lock = lock_data()
        self.notice_data = [b"full notice A", b"full notice B"]
        self.fake_licenses = deepcopy(checker.LICENSES)
        for notice, content in zip(self.fake_licenses, self.notice_data):
            notice["sha256"] = hashlib.sha256(content).hexdigest()
            path = self.checkout / notice["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.contract["licenses"] = deepcopy(self.fake_licenses)
        path = self.checkout / checker.FIXTURE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(serialize(self.fixture))
        for relative in ("tools/packed_oracle/__init__.py", "tools/packed_oracle/__main__.py", "tools/packed_oracle/formats.py", "tools/packed_oracle/vectors.py", checker.WORKFLOW_PATH):
            path = self.checkout / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic source fixture, not executed\n", encoding="utf-8")
        self.git_overrides = {}
        self.calls = []
        self.tree_records, self.index_records = [], []
        for path in sorted(self.checkout.rglob("*")):
            if path.is_file():
                data = path.read_bytes()
                digest = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data, usedforsecurity=False).hexdigest()
                relative = path.relative_to(self.checkout).as_posix()
                self.tree_records.append(f"100644 blob {digest}\t{relative}\0")
                self.index_records.append(f"H 100644 {digest} 0\t{relative}\0")

    def fake_git(self, checkout, *arguments):
        self.calls.append(arguments)
        if arguments in self.git_overrides:
            return self.git_overrides[arguments]
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(self.checkout)
        if arguments == ("rev-parse", "HEAD"):
            return "a" * 40
        if arguments == ("config", "--local", "--get-all", "remote.origin.url"):
            return "https://github.com/lvyufeng/openTPU"
        if arguments == ("ls-tree", "-r", "--full-tree", "-z", "a" * 40):
            return "".join(self.tree_records)
        if arguments == ("ls-files", "--stage", "-v", "-z"):
            return "".join(self.index_records)
        return ""

    def oracle_report(self):
        return {"schema_version": 1, "fixture_id": self.fixture["fixture_id"], "arithmetic_id": checker.ARITHMETIC_ID,
                "payload_sha256": self.fixture["payload_sha256"], "case_count": 2, "decode_cases": 1, "dot_cases": 1,
                "synthetic": True, "verified": True, "mismatches": [], "device_ready": False, "inference_ready": False}

    def execute(self, report=None):
        report = self.oracle_report() if report is None else report
        with patch.object(checker, "LICENSES", self.fake_licenses), patch.object(checker, "_git", side_effect=self.fake_git), \
             patch.object(checker, "_run", return_value=serialize(report)) as run:
            result = checker.verify_oracle(self.contract, self.lock, self.checkout)
        return result, run

    def test_explicit_clean_pinned_source_uses_only_fixed_command(self):
        result, run = self.execute()
        command = run.call_args.args[0]
        self.assertEqual(command, [sys.executable, "-B", "-E", "-S", "-m", "tools.packed_oracle", "verify", checker.FIXTURE_PATH])
        self.assertTrue(result["source_identity_checked"])
        self.assertTrue(result["synthetic_vectors_verified"])
        self.assertFalse(result["device_ready"])
        self.assertFalse(result["inference_ready"])
        self.assertIn(("merge-base", "--is-ancestor", self.contract["sources"]["opentpu_base"]["revision"], "a" * 40), self.calls)

    def test_incomplete_receipt_refuses_before_git(self):
        self.contract["fixture"]["file_sha256"] = None
        with patch.object(checker, "LICENSES", self.fake_licenses), patch.object(checker, "_git") as git:
            with self.assertRaises(checker.ContractError) as failure:
                checker.verify_oracle(self.contract, self.lock, self.checkout)
        self.assertEqual(failure.exception.code, "incomplete_receipt")
        git.assert_not_called()

    def test_wrong_head_origin_top_and_dirty_or_extra_files_refuse(self):
        for arguments, response in ((("rev-parse", "HEAD"), "b" * 40),
                                    (("rev-parse", "--show-toplevel"), str(self.checkout.parent)),
                                    (("config", "--local", "--get-all", "remote.origin.url"), "https://github.com/other/repo"),
                                    (("ls-files", "--stage", "-v", "-z"), "h 100644 " + "a" * 40 + " 0\ttools.py\0"),
                                    (("ls-files", "--others", "-z"), "tools.py\0")):
            with self.subTest(arguments=arguments):
                self.git_overrides = {arguments: response}
                with self.assertRaises(checker.ContractError):
                    self.execute()

    def test_missing_notice_and_symlinked_source_refuse(self):
        path = self.checkout / "tools/packed_oracle/formats.py"
        renamed = path.with_suffix(".txt")
        renamed.write_text("original synthetic fixture", encoding="utf-8")
        path.unlink()
        path.symlink_to(renamed)
        with self.assertRaises(checker.ContractError):
            self.execute()

    def test_modified_notice_digest_refuses(self):
        path = self.checkout / self.fake_licenses[0]["path"]
        path.write_bytes(b"shortened notice")
        with self.assertRaises(checker.ContractError):
            self.execute()

    def test_pinned_fixture_file_bytes_refuse_even_if_json_equivalent(self):
        path = self.checkout / checker.FIXTURE_PATH
        path.write_bytes(serialize(self.fixture) + b"\n")
        with self.assertRaises(checker.ContractError):
            self.execute()

    def test_claimed_ready_mismatch_extra_fields_and_wrong_counts_refuse(self):
        for key, value in (("verified", False), ("device_ready", True), ("inference_ready", True),
                           ("case_count", 3), ("mismatches", [{"case_id": "schema-q8"}]), ("schema_version", True),
                           ("payload_sha256", "b" * 64), ("native_correctness", True)):
            with self.subTest(key=key):
                report = self.oracle_report()
                report[key] = value
                with self.assertRaises(checker.ContractError):
                    self.execute(report)

    def test_post_execution_tracked_bytes_and_index_flags_rechecked(self):
        for change in ("bytes", "index", "HEAD", "extra"):
            with self.subTest(change=change):
                self.git_overrides = {}
                path = self.checkout / "tools/packed_oracle/__main__.py"
                original = path.read_bytes()
                def mutate(command, cwd, *, timeout):
                    if change == "bytes":
                        path.write_bytes(b"# mutated synthetic source\n")
                    elif change == "index":
                        self.git_overrides[("ls-files", "--stage", "-v", "-z")] = "".join(self.index_records).replace("H ", "h ", 1)
                    elif change == "HEAD":
                        self.git_overrides[("rev-parse", "HEAD")] = "b" * 40
                    else:
                        self.git_overrides[("ls-files", "--others", "-z")] = "hidden.ignored\0"
                    return serialize(self.oracle_report())
                with patch.object(checker, "LICENSES", self.fake_licenses), \
                     patch.object(checker, "_git", side_effect=self.fake_git), patch.object(checker, "_run", side_effect=mutate):
                    with self.assertRaises(checker.ContractError):
                        checker.verify_oracle(self.contract, self.lock, self.checkout)
                path.write_bytes(original)

    def test_tracked_listing_rejects_unsafe_paths_modes_duplicates_and_flags(self):
        digest = "a" * 40
        for raw, index in ((f"S 100644 {digest} 0\tx.py\0", True),
                           (f"h 100644 {digest} 0\tx.py\0", True),
                           (f"H 100644 {digest} 2\tx.py\0", True),
                           (f"160000 commit {digest}\tsubmodule\0", False),
                           (f"120000 blob {digest}\tlink.py\0", False),
                           (f"100644 blob {digest}\t../escape.py\0", False),
                           (f"100644 blob {digest}\t/absolute.py\0", False),
                           (f"100644 blob {digest}\tx.py\0" * 2, False),
                           (f"100644 blob {digest}\tx.py", False)):
            with self.subTest(raw=raw):
                with self.assertRaises(checker.ContractError):
                    checker._tracked_records(raw, index=index)

    def test_inherited_import_and_git_settings_are_not_used(self):
        with patch.dict(os.environ, {"PYTHONPATH": "/private/evil", "GIT_CONFIG_COUNT": "1", "GIT_DIR": "/wrong", "DYLD_INSERT_LIBRARIES": "/private/lib"}):
            env = checker._clean_environment()
        for key in ("PYTHONPATH", "GIT_CONFIG_COUNT", "GIT_DIR", "DYLD_INSERT_LIBRARIES"):
            self.assertNotIn(key, env)
        self.assertEqual(env["GIT_OPTIONAL_LOCKS"], "0")
        self.assertEqual(env["GIT_NO_REPLACE_OBJECTS"], "1")
        self.assertEqual(env["GIT_NO_LAZY_FETCH"], "1")

    def test_fixed_runner_failure_timeout_and_oversized_output(self):
        for failure in (OSError("private path"), subprocess.TimeoutExpired("private command", 1)):
            with patch.object(checker.subprocess, "run", side_effect=failure):
                with self.assertRaises(checker.ContractError):
                    checker._run(["git", "status"], self.checkout)
        for returncode, stdout in ((1, b"private path"), (0, b"x" * (checker.LIMITS["max_fixture_bytes"] + 1))):
            with patch.object(checker.subprocess, "run", return_value=subprocess.CompletedProcess([], returncode, stdout, b"")):
                with self.assertRaises(checker.ContractError):
                    checker._run(["git", "status"], self.checkout)


class RealGitIntegrityTests(unittest.TestCase):
    """Use only disposable local repositories; never execute their Python files."""
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.checkout = Path(self.directory.name).resolve()
        self.git("init", "-q")
        for key, value in (("user.name", "Synthetic Test"), ("user.email", "synthetic@example.invalid"),
                           ("commit.gpgsign", "false"), ("core.autocrlf", "false"), ("core.filemode", "true")):
            self.git("config", "--local", key, value)
        self.main = self.checkout / "tools/packed_oracle/__main__.py"
        self.main.parent.mkdir(parents=True)
        self.main.write_bytes(b"# reviewed fixture, never executed\n")
        for name in ("tools/__init__.py", "json.py", "tab\tname.txt", "newline\nname.txt"):
            (self.checkout / name).write_bytes(b"# tracked integrity fixture\n")
        (self.checkout / ".gitignore").write_bytes(b"*.ignored\n")
        self.git("add", "--all")
        self.git("commit", "-q", "--no-verify", "-m", "synthetic integrity fixture")
        self.revision = self.git("rev-parse", "HEAD")

    def git(self, *arguments):
        return checker._git(self.checkout, *arguments)

    def assert_unsafe(self):
        with self.assertRaises(checker.ContractError) as failure:
            checker._check_tracked_bytes(self.checkout, self.revision)
        self.assertEqual(failure.exception.code, "unsafe_source")

    def test_ordinary_clean_tree_and_nul_delimited_unusual_paths(self):
        checker._check_tracked_bytes(self.checkout, self.revision)
        self.assertEqual(self.git("rev-parse", "HEAD"), self.revision)

    def test_assume_unchanged_hidden_same_length_edit_rejected(self):
        self.git("update-index", "--assume-unchanged", "--", "tools/packed_oracle/__main__.py")
        self.main.write_bytes(b"# unreviewed source never executed\n")
        self.assertEqual(self.git("status", "--porcelain=v1"), "")
        self.assert_unsafe()

    def test_skip_worktree_hidden_edit_rejected(self):
        self.git("update-index", "--skip-worktree", "--", "tools/packed_oracle/__main__.py")
        self.main.write_bytes(b"# hidden unreviewed source\n")
        self.assertEqual(self.git("status", "--porcelain=v1"), "")
        self.assert_unsafe()

    def test_staged_edit_index_differs_from_commit(self):
        self.main.write_bytes(b"# staged unreviewed fixture\n")
        self.git("add", "--", "tools/packed_oracle/__main__.py")
        self.assert_unsafe()

    def test_same_length_edit_with_restored_timestamps_rejected(self):
        original = self.main.read_bytes()
        previous = self.main.stat()
        changed = original.replace(b"reviewed", b"evilcode")
        self.assertNotEqual(changed, original)
        self.assertEqual(len(changed), len(original))
        self.main.write_bytes(changed)
        os.utime(self.main, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        self.assertEqual(self.main.stat().st_size, previous.st_size)
        self.assertEqual(self.main.stat().st_mtime_ns, previous.st_mtime_ns)
        self.assert_unsafe()

    def test_tracked_byte_checks_do_not_invoke_local_clean_filter(self):
        # The sentinel command is configured only in this disposable repository.
        attributes = self.checkout / ".gitattributes"
        attributes.write_bytes(b"*.py filter=sentinel\n")
        self.git("add", "--", ".gitattributes")
        self.git("commit", "-q", "--no-verify", "-m", "synthetic attribute fixture")
        self.revision = self.git("rev-parse", "HEAD")
        self.git("config", "--local", "filter.sentinel.clean", "touch filter-called.ignored; cat")
        sentinel = self.checkout / "filter-called.ignored"
        checker._check_tracked_bytes(self.checkout, self.revision)
        self.main.write_bytes(b"# modified synthetic fixture\n")
        self.assert_unsafe()
        self.assertFalse(sentinel.exists())
        # A deliberate test-only hash-object --path control proves the filter
        # really is active. The production checker never runs this command.
        self.git("hash-object", "--path=tools/packed_oracle/__main__.py", "tools/packed_oracle/__main__.py")
        self.assertTrue(sentinel.is_file())

    def test_partial_clone_settings_refuse_before_object_lookup(self):
        for key in ("extensions.partialClone", "remote.origin.promisor", "remote.origin.partialclonefilter"):
            with self.subTest(key=key):
                self.git("config", "--local", key, "synthetic")
                self.assert_unsafe()
                self.git("config", "--local", "--unset", key)

    def test_parent_package_and_local_stdlib_shadow_changes_rejected(self):
        for name in ("tools/__init__.py", "json.py"):
            with self.subTest(path=name):
                path = self.checkout / name
                original = path.read_bytes()
                path.write_bytes(b"# unreviewed parent or import shadow\n")
                self.assert_unsafe()
                path.write_bytes(original)

    def test_ignored_and_untracked_extras_rejected(self):
        for name in ("hidden.ignored", "tools.py"):
            with self.subTest(path=name):
                path = self.checkout / name
                path.write_bytes(b"# extra fixture, never executed\n")
                self.assert_unsafe()
                path.unlink()

    def test_executable_mode_change_rejected(self):
        self.main.chmod(0o755)
        self.assert_unsafe()

    def test_symlink_rejected_even_when_tracked(self):
        path = self.checkout / "symlink.py"
        path.symlink_to("json.py")
        self.git("add", "--", "symlink.py")
        self.git("commit", "-q", "--no-verify", "-m", "synthetic symlink fixture")
        self.revision = self.git("rev-parse", "HEAD")
        self.assert_unsafe()


class JsonCliAndDocsTests(unittest.TestCase):
    def test_duplicate_nonfinite_and_malformed_json_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":Infinity}', b'\xff', b'{', b'[' * 2000):
            with self.subTest(raw=raw[:40]):
                with self.assertRaises(checker.ContractError):
                    checker.parse_json(raw)

    def test_json_depth_boundary_counts_only_unquoted_syntax(self):
        nested = "[" * 64 + "0" + "]" * 64
        checker.parse_json(nested.encode("ascii"))
        with self.assertRaises(checker.ContractError) as failure:
            checker.parse_json(("[" + nested + "]").encode("ascii"))
        self.assertEqual(failure.exception.code, "invalid_json")
        value = '[{"brackets":"]}\\\"\\\\[{"}]' * 80
        self.assertEqual(checker.parse_json(json.dumps({"quoted": value}).encode("utf-8")), {"quoted": value})

    def test_json_number_literal_boundary_and_finite_float(self):
        self.assertEqual(checker.parse_json(b"1" * 64), int("1" * 64))
        checker.parse_json(b"0." + b"1" * 62)
        for raw in (b"1" * 65, b"0." + b"1" * 63, b"1e999", b"-Infinity"):
            with self.subTest(raw=raw[:20]):
                with self.assertRaises(checker.ContractError) as failure:
                    checker.parse_json(raw)
                self.assertEqual(failure.exception.code, "invalid_json")

    def test_input_limits_and_io_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_bytes(b"{} " * 20)
            with self.assertRaises(checker.ContractError):
                checker.read_json(path, limit=4)
            with self.assertRaises(checker.ContractError):
                checker.read_json(Path(directory))

    def test_default_cli_runs_from_other_cwd_without_upstreams(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-B", "-S", str(ROOT / "scripts/check_packed_oracle.py")], cwd=directory,
                                    capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["oracle_executed"])
        self.assertFalse(report["synthetic_vectors_verified"])

    def test_cli_strict_incomplete_exits_one_with_json_report(self):
        contract = complete_contract()
        contract["fixture"]["file_sha256"] = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_bytes(serialize(contract))
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = checker.main(["--contract", str(path), "--strict-receipt"])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(stdout.getvalue())["receipt_complete"])

    def test_cli_malformed_and_arguments_exit_two_with_stable_stderr(self):
        for arguments in (["--unknown-option"], ["--contract", "/not/an/existing/input.json"]):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = checker.main(arguments)
            self.assertEqual(code, 2)
            error = json.loads(stderr.getvalue())
            self.assertIn("code", error)
            self.assertNotIn("/not/an", error["message"])

    def test_cli_fixture_metadata_does_not_execute_oracle(self):
        fixture = tiny_fixture()
        contract = complete_contract(fixture)
        with tempfile.TemporaryDirectory() as directory:
            cp, fp = Path(directory) / "contract.json", Path(directory) / "fixture.json"
            cp.write_bytes(serialize(contract))
            fp.write_bytes(serialize(fixture))
            stdout = io.StringIO()
            with redirect_stdout(stdout), patch.object(checker.subprocess, "run", side_effect=AssertionError("metadata execution")):
                code = checker.main(["--contract", str(cp), "--fixture", str(fp), "--strict-receipt"])
        self.assertEqual(code, 0)
        report = json.loads(stdout.getvalue())
        self.assertTrue(report["fixture_metadata_checked"])
        self.assertFalse(report["oracle_executed"])

    def test_docs_ci_and_bootstrap_boundary(self):
        url = "https://github.com/lvyufeng/strata-opentpu/issues/17"
        for relative in ("README.md", "docs/development.md", "docs/design/packed-oracle.md"):
            self.assertIn(url, (ROOT / relative).read_text(encoding="utf-8"))
        workflow = (ROOT / ".github/workflows/bootstrap.yml").read_text(encoding="utf-8")
        self.assertIn("python scripts/check_packed_oracle.py --strict-receipt", workflow)
        self.assertNotIn("--oracle-root", workflow)
        self.assertNotIn("--fixture", workflow)
        profile = json.loads((ROOT / "configs/profiles/bootstrap.json").read_text(encoding="utf-8"))
        self.assertEqual({name for name, enabled in profile["capabilities"].items() if enabled}, {"doctor", "bootstrap"})
        self.assertEqual(lock_data()["dependencies"]["opentpu"]["revision"], "d84db55712aba0d40811d7762cb052f05d55b7e5")


if __name__ == "__main__":
    unittest.main()
