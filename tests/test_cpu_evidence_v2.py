"""Acceptance tests for CPU evidence protocol v2.

These tests exercise the v2 checker against the three real routed-expert
gate/up formats recorded in the audited GGUF header inventory.  They are
structural checks only: no payload is read, no host is probed, and nothing
here is measured evidence or an inference certificate.
"""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2_CHECKER = ROOT / "scripts" / "check_cpu_evidence_v2.py"
V2_TEMPLATE = ROOT / "benchmarks" / "templates" / "cpu-cold-expert-v2.json"
V1_CHECKER = ROOT / "scripts" / "check_cpu_evidence.py"
V1_TEMPLATE = ROOT / "benchmarks" / "templates" / "cpu-cold-expert.json"
LOCK_PATH = ROOT / "dependencies.lock.json"

V1_PROTOCOL = "strata-cpu-cold-expert-v1"
V1_FORMATS = ("IQ2_XS", "Q2_0", "IQ3_S")
V1_TYPE_INFO = {
    "IQ2_XS": (17, 256, 74, "Q8_K"),
    "Q2_0": (42, 64, 18, "Q8_0"),
    "IQ3_S": (21, 256, 110, "Q8_K"),
    "Q8_0": (8, 32, 34, "Q8_0"),
}
V2_ONLY_TYPE_INFO = {
    "IQ2_S": (22, 256, 82, "Q8_K"),
    "IQ2_XXS": (16, 256, 66, "Q8_K"),
    "IQ1_M": (29, 256, 56, "Q8_K"),
}
# sha256 / line counts of the four frozen v1 files as reviewed before v2.
FROZEN_V1_FILES = {
    "scripts/check_cpu_evidence.py": (
        936,
        "d727806ba677563cd4bfa267bebec9d04c67c4b1b048da04f1405678c318826a",
    ),
    "benchmarks/cpu-cold-expert-protocol.md": (
        355,
        "5a7ca9ac1fda336bcbc2286e0526379753878bcda096fa1b708cd4083514dce9",
    ),
    "tests/test_cpu_evidence.py": (
        875,
        "704dc4b9fb801f8c2070d4ad07cd07a4d6ce327dc0e32c271d95f17ffd8830dd",
    ),
    "benchmarks/templates/cpu-cold-expert.json": (
        136,
        "522150850c473692d001ec9e9e2360d8c0cb6c704b320e3e731206af9c7ffeb2",
    ),
}
# Real expert gate/up block geometry: 2560 columns of 256-value blocks.
PROFILE_GEOMETRY = {
    "iq2-s-layer0-mixed-role": ("IQ2_S", 820, 524800, 1049600, 1510400),
    "iq2-xxs-layer1-mixed-role": ("IQ2_XXS", 660, 422400, 844800, 1305600),
    "iq1-m-layer8-mixed-role": ("IQ1_M", 560, 358400, 716800, 1177600),
}
FRAGMENT_WIDTH = "projection columns are not whole packed weight blocks"
FRAGMENT_DERIVED = "does not match derived value"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cpu = load_module(V2_CHECKER, "cpu_evidence_v2_under_test")
TEMPLATE = cpu.load_json(V2_TEMPLATE)
LOCK = cpu.load_json(LOCK_PATH)


def load_v1_fixture():
    """Load the frozen v1 test module and reuse its reviewed synthetic fixture.

    The v1 module owns the canonical complete 237-case synthetic constructor, so
    reusing it keeps this regression aligned with v1's strict-mode proof instead
    of forking ~90 lines of fixture. Reuse is safe: the v1 fixture loads the
    frozen v1 checker into a private module instance that is never registered in
    ``sys.modules``, and the v2 checker under test is a second such instance with
    its own rebindings; neither can mutate the other.
    """
    try:
        import test_cpu_evidence as fixture
    except ImportError:  # e.g. pytest importlib import mode without tests/ on sys.path
        fixture = load_module(
            ROOT / "tests" / "test_cpu_evidence.py", "_cpu_evidence_v1_fixture_for_v2"
        )
    return fixture


V1_FIXTURE = load_v1_fixture()


def key_shape(value):
    """Ordered key skeleton, so field order is compared too.

    Leaf types are ignored because the v1 template is a null skeleton while
    the v2 template carries real header geometry.
    """
    if isinstance(value, dict):
        return tuple((key, key_shape(item)) for key, item in value.items())
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return ("list-of", key_shape(value[0]))
    return "value"


def retag(role, fmt):
    """Declare a role as fmt with self-consistent type metadata."""
    type_id, block_values, block_bytes, activation = cpu.TYPE_INFO[fmt]
    role["format"] = fmt
    role["type_id"] = type_id
    role["block_values"] = block_values
    role["block_bytes"] = block_bytes
    role["native_vec_dot_activation"] = activation


def complete_v2_report():
    """Retarget v1's reviewed complete 237-case fixture onto the v2 formats.

    v1's ``synthetic_report()`` is a v1-protocol report (v1 ``cpu`` constants and
    paths), so it cannot be imported as-is.  This builder keeps its provenance,
    host, build, placement, timing and full-case shape unchanged, and only
    retargets what section 2 of the protocol changes: protocol, matrix formats,
    profile ids/role type metadata/derived geometry, case target formats,
    dispatch activation names, and block/dot control geometry and checks.
    """
    report = V1_FIXTURE.synthetic_report()
    report["protocol"] = cpu.PROTOCOL
    report["matrix"]["formats"] = list(cpu.FORMATS)
    remap = dict(zip(V1_FORMATS, cpu.FORMATS))
    id_map = {}
    for profile in report["profiles"]:
        fmt = remap[profile["target_format"]]
        old_id = profile["id"]
        new_id = f"synthetic-{fmt.lower()}-selected-mixed-role"
        id_map[old_id] = new_id
        profile["id"] = new_id
        profile["target_format"] = fmt
        offset = 0
        for name in cpu.ROLES:
            role = profile["roles"][name]
            role_fmt = role["format"] if name == "down" else fmt
            type_id, block_values, block_bytes, activation = cpu.TYPE_INFO[role_fmt]
            rows, columns = role["shape"]
            role.update(
                tensor_name=f"SYNTHETIC.{new_id}.{name}",
                format=role_fmt,
                type_id=type_id,
                block_values=block_values,
                block_bytes=block_bytes,
                row_bytes=columns // block_values * block_bytes,
                offset_bytes=offset,
                native_vec_dot_activation=activation,
            )
            offset += rows * role["row_bytes"]
        profile["blob_bytes"] = offset
    for index, case in enumerate(report["cases"]):
        fmt = remap[case["target_format"]]
        activation = cpu.TYPE_INFO[fmt][3]
        case["id"] = f"SYNTHETIC-v2-{index:03d}"
        case["target_format"] = fmt
        case["dispatch"]["gate_up_activation"] = activation
        if case["unit"] == "full":
            case["profile_id"] = id_map[case["profile_id"]]
            continue
        case["dispatch"]["down_activation"] = activation
        _, block_values, block_bytes, _ = cpu.TYPE_INFO[fmt]
        rows = len(cpu.SCENARIOS)
        case["control"].update(
            columns=block_values,
            rows=rows,
            row_bytes=block_bytes,
            packed_bytes=rows * block_bytes,
        )
        shape = [rows, block_values] if case["unit"] == "block" else [case["batch_size"], rows]
        case["checks"] = [V1_FIXTURE.numerical_check(case["unit"], shape)]
    return report


COMPLETE = complete_v2_report()


class V2EvidenceTest(unittest.TestCase):
    def report(self):
        return copy.deepcopy(TEMPLATE)

    def profile(self, index):
        return copy.deepcopy(TEMPLATE["profiles"][index])

    def check_profiles(self, profiles, kind="partial"):
        check = cpu.Checker()
        cpu._validate_profiles(check, profiles, kind)
        return check

    def assert_rejected(self, evidence, fragment, mode="partial"):
        with self.assertRaises(cpu.EvidenceError) as caught:
            cpu.validate(evidence, LOCK, mode)
        self.assertIn(fragment, str(caught.exception))

    # -- protocol/format contract -------------------------------------

    def test_protocol_and_schema_identity(self):
        self.assertEqual(cpu.PROTOCOL, "strata-cpu-cold-expert-v2")
        self.assertEqual(TEMPLATE["protocol"], cpu.PROTOCOL)
        self.assertEqual(TEMPLATE["schema_version"], 1)

    def test_formats_are_the_three_real_expert_types(self):
        self.assertEqual(cpu.FORMATS, ("IQ2_S", "IQ2_XXS", "IQ1_M"))
        self.assertEqual(tuple(TEMPLATE["matrix"]["formats"]), cpu.FORMATS)
        self.assertEqual(len(cpu.FORMATS), len(set(cpu.FORMATS)))

    def test_new_type_info_entries(self):
        for fmt, expected in V2_ONLY_TYPE_INFO.items():
            self.assertEqual(cpu.TYPE_INFO[fmt], expected)

    def test_inherited_type_info_is_complete_and_unchanged(self):
        expected = dict(V1_TYPE_INFO)
        expected.update(V2_ONLY_TYPE_INFO)
        self.assertEqual(cpu.TYPE_INFO, expected)

    def test_expected_case_count_is_237(self):
        matrix = copy.deepcopy(TEMPLATE["matrix"])
        matrix["worker_nodes"] = [0, 1]
        expected = cpu._expected_keys(matrix)
        self.assertEqual(len(expected), 237)
        self.assertEqual(
            sum(1 for key in expected if key[0] == "block"), 9
        )
        self.assertEqual(
            sum(1 for key in expected if key[0] == "dot"), 36
        )
        self.assertEqual(
            sum(1 for key in expected if key[0] == "full"), 192
        )

    def test_template_field_set_matches_v1_template(self):
        v1 = cpu.load_json(V1_TEMPLATE)
        self.assertEqual(key_shape(self.report()), key_shape(v1))

    def test_template_is_partial_with_empty_cases(self):
        data = self.report()
        self.assertEqual(data["evidence_kind"], "partial")
        self.assertEqual(data["cases"], [])
        self.assertEqual(len(data["profiles"]), 3)
        self.assertEqual(
            [profile["target_format"] for profile in data["profiles"]],
            list(cpu.FORMATS),
        )

    # -- profile geometry ---------------------------------------------

    def test_template_passes_default_mode(self):
        result = cpu.validate(self.report(), LOCK, "partial")
        self.assertEqual(result["protocol"], cpu.PROTOCOL)
        self.assertEqual(result["validation_mode"], "partial")
        self.assertTrue(result["mode_satisfied"])
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["schema_complete"])
        self.assertFalse(result["measured_evidence_complete"])
        self.assertFalse(result["declared_cpu_correctness_pass"])
        self.assertEqual(result["cases_present"], 0)
        self.assertIsNone(result["cases_required"])
        self.assertIn("coverage.derived_matrix", result["missing_fields"])
        self.assertEqual(result["missing_cases"], [])
        self.assertTrue(result["structural_consistency_only"])

    def geometry_case(self, index):
        profile = self.profile(index)
        target, gate_row, up_offset, down_offset, blob_bytes = PROFILE_GEOMETRY[
            profile["id"]
        ]
        self.assertEqual(profile["target_format"], target)
        self.assertEqual(profile["n_embd"], 2560)
        self.assertEqual(profile["n_ff"], 640)
        gate = profile["roles"]["gate"]
        up = profile["roles"]["up"]
        down = profile["roles"]["down"]
        self.assertEqual(gate["row_bytes"], gate_row)
        self.assertEqual(up["row_bytes"], gate_row)
        self.assertEqual(gate["offset_bytes"], 0)
        self.assertEqual(up["offset_bytes"], up_offset)
        self.assertEqual(down["offset_bytes"], down_offset)
        self.assertEqual(profile["blob_bytes"], blob_bytes)
        # Independent derivation, not just a restated constant.
        self.assertEqual(gate_row, (2560 // 256) * gate["block_bytes"])
        self.assertEqual(up_offset, gate_row * 640)
        self.assertEqual(down_offset, 2 * up_offset)
        self.assertEqual(blob_bytes, down_offset + down["row_bytes"] * 2560)
        check = self.check_profiles([profile])
        self.assertEqual(
            {item for item in check.missing if item.startswith("profiles.target_format=")},
            {
                "profiles.target_format=" + fmt
                for fmt in cpu.FORMATS
                if fmt != target
            },
        )
        drifted = copy.deepcopy(profile)
        drifted["blob_bytes"] += 1
        with self.assertRaises(cpu.EvidenceError) as caught:
            self.check_profiles([drifted])
        self.assertIn(FRAGMENT_DERIVED, str(caught.exception))

    def test_iq2_s_profile_geometry(self):
        self.geometry_case(0)

    def test_iq2_xxs_profile_geometry(self):
        self.geometry_case(1)

    def test_iq1_m_profile_geometry(self):
        self.geometry_case(2)

    def test_every_profile_down_role_is_q2_0_at_640(self):
        for profile in self.report()["profiles"]:
            down = profile["roles"]["down"]
            self.assertEqual(down["format"], "Q2_0")
            self.assertEqual(down["type_id"], 42)
            self.assertEqual(down["block_values"], 64)
            self.assertEqual(down["block_bytes"], 18)
            self.assertEqual(down["row_bytes"], 180)
            self.assertEqual(down["shape"], [2560, 640])
            self.assertEqual(down["native_vec_dot_activation"], "Q8_0")
            for name in ("gate", "up"):
                self.assertEqual(
                    profile["roles"][name]["native_vec_dot_activation"], "Q8_K"
                )

    def test_down_width_640_rejects_v2_iq_targets(self):
        for index, profile in enumerate(self.report()["profiles"]):
            retag(profile["roles"]["down"], profile["target_format"])
            with self.assertRaises(cpu.EvidenceError) as caught:
                self.check_profiles([profile])
            self.assertIn(FRAGMENT_WIDTH, str(caught.exception), index)

    def test_down_width_640_rejects_v1_iq_formats(self):
        for fmt in ("IQ2_XS", "IQ3_S"):
            profile = self.profile(0)
            retag(profile["roles"]["down"], fmt)
            with self.assertRaises(cpu.EvidenceError) as caught:
                self.check_profiles([profile])
            self.assertIn(FRAGMENT_WIDTH, str(caught.exception), fmt)

    def test_down_padded_width_claim_is_rejected(self):
        profile = self.profile(0)
        # A 704-column padded row (704 % 64 == 0) is still not this profile.
        profile["roles"]["down"]["shape"] = [2560, 704]
        with self.assertRaises(cpu.EvidenceError) as caught:
            self.check_profiles([profile])
        self.assertIn(FRAGMENT_DERIVED, str(caught.exception))
        padded = self.profile(0)
        padded["roles"]["down"]["shape"] = [2560, 704]
        padded["roles"]["down"]["row_bytes"] = 198
        with self.assertRaises(cpu.EvidenceError) as caught:
            self.check_profiles([padded])
        self.assertIn(FRAGMENT_DERIVED, str(caught.exception))

    def test_relabelled_gate_format_is_rejected(self):
        profile = self.profile(0)
        profile["roles"]["gate"]["format"] = "IQ1_M"
        with self.assertRaises(cpu.EvidenceError) as caught:
            self.check_profiles([profile])
        self.assertIn(FRAGMENT_DERIVED, str(caught.exception))
        swapped = self.profile(0)
        swapped["roles"]["gate"]["format"] = "IQ2_S"
        swapped["roles"]["gate"]["type_id"] = 29
        with self.assertRaises(cpu.EvidenceError) as caught:
            self.check_profiles([swapped])
        self.assertIn(FRAGMENT_DERIVED, str(caught.exception))

    def test_unknown_format_is_rejected(self):
        for fmt in ("IQ3_XXS", "Q4_K", "TQ2_T"):
            profile = self.profile(0)
            profile["target_format"] = fmt
            with self.assertRaises(cpu.EvidenceError) as caught:
                self.check_profiles([profile])
            self.assertIn("expected one of", str(caught.exception), fmt)
        profile = self.profile(0)
        profile["roles"]["gate"]["format"] = "Q4_K"
        with self.assertRaises(cpu.EvidenceError) as caught:
            self.check_profiles([profile])
        self.assertIn("expected one of", str(caught.exception))

    def test_v2_target_format_outside_v2_is_rejected(self):
        data = self.report()
        data["profiles"][0]["target_format"] = "IQ2_XS"
        self.assert_rejected(data, "expected one of")

    def test_matrix_formats_must_equal_pinned_formats(self):
        data = self.report()
        data["matrix"]["formats"] = ["IQ2_S", "Q2_0", "IQ3_S"]
        self.assert_rejected(data, "must contain exactly")

    def test_missing_target_profile_is_incomplete(self):
        data = self.report()
        data["profiles"] = data["profiles"][:2]
        result = cpu.validate(data, LOCK, "partial")
        self.assertEqual(result["status"], "incomplete")
        self.assertIn(
            "profiles.target_format=IQ1_M", result["missing_fields"]
        )
        strict = cpu.validate(data, LOCK, "strict-complete")
        self.assertFalse(strict["mode_satisfied"])

    # -- honesty and cross-protocol behaviour --------------------------

    def test_strict_modes_reject_partial_template(self):
        for mode in ("strict-complete", "strict-accept"):
            result = cpu.validate(self.report(), LOCK, mode)
            self.assertEqual(result["validation_mode"], mode)
            self.assertFalse(result["mode_satisfied"], mode)
            self.assertFalse(result["schema_complete"], mode)
        with self.assertRaises(cpu.EvidenceError):
            cpu.validate(self.report(), LOCK, "accept")

    def test_honesty_flags_never_flip(self):
        for mode in ("partial", "strict-complete", "strict-accept"):
            result = cpu.validate(self.report(), LOCK, mode)
            self.assertIs(result["structural_consistency_only"], True)
            self.assertIs(result["measurements_independently_verified"], False)
            self.assertIs(result["inference_ready"], False)
            self.assertEqual(
                result["note"],
                "Checks declared JSON consistency only; no experiment "
                "authorization, authenticity, model quality or "
                "decode-throughput claim.",
            )

    def test_v2_rejects_v1_report(self):
        v1_report = cpu.load_json(V1_TEMPLATE)
        with self.assertRaises(cpu.EvidenceError) as caught:
            cpu.validate(v1_report, LOCK, "partial")
        self.assertIn("unsupported CPU evidence protocol/schema", str(caught.exception))

    def test_v1_rejects_v2_report_without_contamination(self):
        v1 = load_module(V1_CHECKER, "_strata_v1_core_probe")
        self.assertEqual(v1.PROTOCOL, V1_PROTOCOL)
        self.assertEqual(v1.FORMATS, V1_FORMATS)
        self.assertEqual(v1.TYPE_INFO, V1_TYPE_INFO)
        self.assertIsNot(cpu._impl, v1)
        self.assertEqual(cpu._impl.PROTOCOL, cpu.PROTOCOL)
        self.assertNotIn("_strata_cpu_evidence_v1_core_for_v2", sys.modules)
        with self.assertRaises(v1.EvidenceError) as caught:
            v1.validate(cpu.load_json(V2_TEMPLATE), LOCK, "partial")
        self.assertIn("unsupported CPU evidence protocol/schema", str(caught.exception))

    def test_frozen_v1_files_unchanged(self):
        for relative, (lines, digest) in FROZEN_V1_FILES.items():
            path = ROOT / relative
            payload = path.read_bytes()
            self.assertEqual(
                hashlib.sha256(payload).hexdigest(), digest, relative
            )
            self.assertEqual(payload.count(b"\n"), lines, relative)

    # -- complete 237-case matrix (end-to-end regression) ---------------

    def test_complete_synthetic_matrix_is_schema_complete_but_not_measured(self):
        result = cpu.validate(copy.deepcopy(COMPLETE), LOCK, "partial")
        self.assertEqual(result["protocol"], cpu.PROTOCOL)
        self.assertEqual(result["status"], "schema-complete")
        self.assertTrue(result["schema_complete"])
        self.assertEqual(result["cases_present"], 237)
        self.assertEqual(result["cases_required"], 237)
        self.assertEqual(result["missing_cases"], [])
        self.assertEqual(result["missing_fields"], [])
        self.assertEqual(result["evidence_kind"], "synthetic")
        self.assertFalse(result["measured_evidence_complete"])
        self.assertFalse(result["declared_cpu_correctness_pass"])
        self.assertFalse(result["measurements_independently_verified"])
        self.assertFalse(result["inference_ready"])
        for mode in ("strict-complete", "strict-accept"):
            self.assertFalse(
                cpu.validate(copy.deepcopy(COMPLETE), LOCK, mode)["mode_satisfied"],
                mode,
            )

    def test_complete_synthetic_matrix_exits_one_under_both_strict_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complete-v2.json"
            path.write_text(json.dumps(COMPLETE), encoding="utf-8")
            code, out, _ = self.run_cli([str(path)])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(out)["schema_complete"])
            for flag in ("--strict-complete", "--strict-accept"):
                code, out, _ = self.run_cli([flag, str(path)])
                self.assertEqual(code, 1, flag)
                self.assertFalse(json.loads(out)["mode_satisfied"], flag)

    def test_synthetic_as_measured_is_rejected(self):
        report = copy.deepcopy(COMPLETE)
        report["evidence_kind"] = "measured"
        with self.assertRaisesRegex(cpu.EvidenceError, "synthetic model fixture"):
            cpu.validate(report, LOCK, "strict-complete")

    def test_schema_legal_authoring_violations_still_validate(self):
        # Guards the v1/v2 "schema-legal is not evidence-valid" boundary stated in
        # section 3 and section 8 of the v2 protocol: the offline checker enforces
        # derived consistency only, so substituting Q8_0 for the real Q2_0 down
        # role and recording an unaudited 1280x320 geometry stay structurally
        # consistent and exit 0 in partial mode. This test documents the boundary;
        # it does not make either report valid evidence or a protocol target.
        q8 = copy.deepcopy(TEMPLATE)
        profile = q8["profiles"][0]
        down = profile["roles"]["down"]
        down.update(
            format="Q8_0",
            type_id=8,
            block_values=32,
            block_bytes=34,
            row_bytes=640 // 32 * 34,
            native_vec_dot_activation="Q8_0",
        )
        profile["blob_bytes"] = down["offset_bytes"] + down["row_bytes"] * profile["n_embd"]

        drift = copy.deepcopy(TEMPLATE)
        profile = drift["profiles"][0]
        n_embd, n_ff = 1280, 320
        profile.update(n_embd=n_embd, n_ff=n_ff)
        gate, up, down = (profile["roles"][name] for name in ("gate", "up", "down"))
        gu_row_bytes = n_embd // 256 * 82
        gate.update(shape=[n_ff, n_embd], row_bytes=gu_row_bytes)
        up.update(
            shape=[n_ff, n_embd],
            row_bytes=gu_row_bytes,
            offset_bytes=gu_row_bytes * n_ff,
        )
        down.update(
            shape=[n_embd, n_ff],
            row_bytes=n_ff // 64 * 18,
            offset_bytes=2 * gu_row_bytes * n_ff,
        )
        profile["blob_bytes"] = down["offset_bytes"] + down["row_bytes"] * n_embd

        for report in (q8, drift):
            result = cpu.validate(report, LOCK, "partial")
            self.assertTrue(result["mode_satisfied"])
            self.assertEqual(result["evidence_kind"], "partial")

    def test_mocked_measured_seam_passes_strict_accept_without_evidence(self):
        report = copy.deepcopy(COMPLETE)
        report["evidence_kind"] = "measured"
        profiles = {profile["id"]: profile for profile in report["profiles"]}
        targets = {profile["target_format"]: profile for profile in report["profiles"]}
        # EXPLICIT TEST SEAM: v2's validate() delegates to the private v1 module
        # instance, so only the profile/case declaration gates are replaced there.
        # No fabricated values are reclassified as actual measurements.
        with mock.patch.object(
            cpu._impl, "_validate_profiles", return_value=(profiles, targets)
        ), mock.patch.object(cpu._impl, "_validate_case", return_value=("pass", True)):
            result = cpu.validate(report, LOCK, "strict-accept")
        self.assertTrue(result["mode_satisfied"])
        self.assertTrue(result["schema_complete"])
        self.assertTrue(result["measured_evidence_complete"])
        self.assertTrue(result["declared_cpu_correctness_pass"])
        self.assertEqual(result["cases_present"], 237)
        self.assertFalse(result["measurements_independently_verified"])
        self.assertFalse(result["inference_ready"])

    # -- CLI -----------------------------------------------------------

    def run_cli(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cpu.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_cli_default_and_strict_exit_codes(self):
        code, out, _ = self.run_cli([])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["validation_mode"], "partial")
        for flag in ("--strict-complete", "--strict-accept"):
            code, out, _ = self.run_cli([flag])
            self.assertEqual(code, 1, flag)
            self.assertFalse(json.loads(out)["mode_satisfied"], flag)

    def test_cli_both_strict_flags_exit_two(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_cli(["--strict-complete", "--strict-accept"])
        self.assertEqual(caught.exception.code, 2)

    def test_cli_malformed_input_exits_one_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"schema_version": 1, "schema_version": 2}', encoding="utf-8")
            with self.assertRaises(SystemExit) as caught:
                self.run_cli([str(path)])
            self.assertEqual(caught.exception.code, 1)
            broken = Path(directory) / "broken.json"
            broken.write_text("{not json", encoding="utf-8")
            with self.assertRaises(SystemExit) as caught:
                self.run_cli([str(broken)])
            self.assertEqual(caught.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
