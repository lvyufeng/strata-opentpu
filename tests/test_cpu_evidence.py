# Copyright (c) 2026 lvyufeng and contributors
# SPDX-License-Identifier: Apache-2.0
"""Offline schema tests. ALL fixture values/hashes/timings are fabricated.

The validator is not a benchmark; these tests never build or import Strata,
probe the host, extract weights, allocate NUMA pages, or run any CPU kernel.
The measured-declaration branch is tested separately and mocked explicitly.
"""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cpu_evidence", ROOT / "scripts/check_cpu_evidence.py")
cpu = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cpu)
TEMPLATE = ROOT / "benchmarks/templates/cpu-cold-expert.json"


def fake_hash(label):
    return hashlib.sha256(("SYNTHETIC-NOT-MEASURED:" + label).encode()).hexdigest()


def numerical_check(stage, shape):
    # Identical vectors exercise schema/hash/elementwise machinery only. This
    # deliberately proves NOTHING about Strata numerical correctness.
    values = [0.0, -0.25, 0.5, 1.0] * ((__import__("math").prod(shape) + 3) // 4)
    values = values[:__import__("math").prod(shape)]
    return {
        "stage": stage, "criterion": "elementwise-abs-plus-rel",
        "atol": 0.00001, "rtol": 0.00001, "shape": shape,
        "actual": values.copy(), "reference": values.copy(),
        "actual_sha256": cpu.vector_sha256(values),
        "reference_sha256": cpu.vector_sha256(values),
        "reference_identity": {"kind": "strata-scalar-reference", "revision": cpu.STRATA_REVISION, "artifact_sha256": fake_hash("reference")},
        "failure_count": 0, "passed": True,
    }


def placement(threads, node, policy):
    memory_nodes = [node] if policy in ("local", "contention") else [1 - node] if policy == "remote" else [0, 1]
    worker_cpus = [2 * node + i for i in range(threads)]
    active = policy == "contention"
    return {
        "requested": {"cpu_ids": worker_cpus, "memory_nodes": memory_nodes, "memory_policy": policy, "first_touch": "SYNTHETIC fixture only"},
        "observed": {
            "workers": [{"thread": i, "effective_cpu_ids": [worker_cpus[i]], "sampled_cpu_ids": [worker_cpus[i]]} for i in range(threads)],
            "memory_policy": policy,
            "page_counts": [{"node": item, "pages": 10} for item in memory_nodes],
            "total_pages": 10 * len(memory_nodes),
            "first_touch": {"cpu_ids": [2 * item for item in memory_nodes], "node_ids": memory_nodes, "pages_touched": 10 * len(memory_nodes), "evidence_sha256": fake_hash("touch")},
            "contention": {"active": active, "cpu_ids": worker_cpus if active else [], "memory_nodes": memory_nodes if active else [], "description": "SYNTHETIC competing traffic" if active else None, "evidence_sha256": fake_hash("contention") if active else None},
            "affinity_sha256": fake_hash("affinity"), "memory_sha256": fake_hash("memory"),
        },
    }


def synthetic_report():
    """Complete, unmistakably synthetic full mandatory matrix fixture."""
    report = cpu.load_json(TEMPLATE)
    report["evidence_kind"] = "synthetic"
    report["notes"] = ["ALL values are fabricated for unit tests, not host or CPU measurements."]
    report["provenance"] = {"experiment_id": "SYNTHETIC-UNIT-TEST", "observed_at": "SYNTHETIC-NO-EXPERIMENT", "operator_label": "SYNTHETIC", "producer": "tests/test_cpu_evidence.py fixture", "raw_artifact_sha256": fake_hash("raw")}
    report["build"] = {
        "compiler": "SYNTHETIC clang", "compiler_version": "SYNTHETIC", "flags": ["-O2"],
        "avx2_translation_unit_flags": ["-mavx2", "-mfma", "-mf16c"],
        "quant_backend": "strata-own", "ggml_revision": None, "compiled_isa_floor": "avx2",
        "binary_sha256": fake_hash("binary"), "build_log_sha256": fake_hash("build"), "environment": {},
    }
    report["host"] = {
        "label": "SYNTHETIC dual-NUMA", "architecture": "x86_64", "os": "SYNTHETIC",
        "sockets": 2, "numa_nodes": [0, 1],
        "topology": [{"cpu": i, "core": i, "socket": i // 2, "node": i // 2} for i in range(4)],
        "features": dict.fromkeys(cpu.FEATURES, True),
        "cpuid_1_ecx": sum(1 << bit for bit in (12, 27, 28, 29)), "cpuid_7_ebx": 1 << 5,
        "xgetbv_executed": True, "xcr0": 6, "avx2_usable": True,
        "feature_probe_sha256": fake_hash("features"), "topology_sha256": fake_hash("topology"),
    }
    report["matrix"]["batch_sizes"] = [1, 2]
    report["matrix"]["worker_nodes"] = [0, 1]
    # Small valid geometry keeps schema tests fast. Q8_0 down is auxiliary,
    # explicitly NOT asserted to be the real model's down tensor.
    for profile in report["profiles"]:
        profile.update({"data_origin": "synthetic-fixture", "model_id": "SYNTHETIC-NOT-A-MODEL", "model_sha256": fake_hash("model"), "tensor_metadata_sha256": fake_hash("metadata"), "layer": 0, "expert": 0, "n_embd": 256, "n_ff": 64, "blob_sha256": fake_hash(profile["id"]), "blob_bytes": 1})
        offset = 0
        for role in cpu.ROLES:
            row = profile["roles"][role]
            fmt = profile["target_format"] if role != "down" else "Q8_0"
            type_id, block_values, block_bytes, activation = cpu.TYPE_INFO[fmt]
            rows, columns = (256, 64) if role == "down" else (64, 256)
            row_bytes = columns // block_values * block_bytes
            row.update({"tensor_name": f"SYNTHETIC.{profile['id']}.{role}", "format": fmt, "type_id": type_id, "shape": [rows, columns], "block_values": block_values, "block_bytes": block_bytes, "row_bytes": row_bytes, "offset_bytes": offset, "tensor_sha256": fake_hash(profile["id"] + role), "native_vec_dot_activation": activation})
            offset += rows * row_bytes
        profile["blob_bytes"] = offset
    by_target = {p["target_format"]: p for p in report["profiles"]}
    for index, key in enumerate(sorted(cpu._expected_keys(report["matrix"]))):
        unit, fmt, role, batch, threads, cache, policy, node = key
        profile = by_target[fmt]
        native_activation = cpu.TYPE_INFO[fmt][3]
        activation = "ActQ-legacy" if fmt == "Q2_0" else native_activation
        checks = []
        control = None
        trace = None
        if unit == "full":
            checks = [numerical_check(stage, [batch, 64 if stage != "down" else 256]) for stage in ("gate", "up", "hidden", "down")]
        else:
            _, block_values, block_bytes, _ = cpu.TYPE_INFO[fmt]
            control = {"data_origin": "generated-controls", "scenarios": list(cpu.SCENARIOS), "rows": 4, "columns": block_values, "row_bytes": block_bytes, "packed_bytes": 4 * block_bytes}
            checks = [numerical_check(unit, [4, block_values] if unit == "block" else [batch, 4])]
        input_hash = fake_hash(f"input-{index}")
        if unit == "full":
            trace = {"stages": list(cpu.STAGES), "x_input_sha256": input_hash, "x_quantized_sha256": fake_hash("xquant"), "gate_input_sha256": fake_hash("xquant"), "up_input_sha256": fake_hash("xquant"), "gate_output_sha256": checks[0]["actual_sha256"], "up_output_sha256": checks[1]["actual_sha256"], "silu_gate_input_sha256": checks[0]["actual_sha256"], "silu_up_input_sha256": checks[1]["actual_sha256"], "hidden_output_sha256": checks[2]["actual_sha256"], "down_quantization_input_sha256": checks[2]["actual_sha256"], "h_quantized_sha256": fake_hash("hquant"), "down_input_sha256": fake_hash("hquant"), "down_output_sha256": checks[3]["actual_sha256"], "synchronized": True}
        raw = [100, 120, 110, 140, 130]
        starts = [1000 * i for i in range(5)]
        timing = {"clock": "monotonic-wall", "scope": "full-branch-with-quantization-and-synchronization" if unit == "full" else unit, "stages": list(cpu.STAGES) if unit == "full" else [unit], "warmup_wall_ns": [150], "raw_start_ns": starts, "raw_end_ns": [start + duration for start, duration in zip(starts, raw)], "raw_wall_ns": raw, "median_ns": statistics.median(raw), "p95_ns": 140, "variance_ns2": statistics.pvariance(raw), "cache_preparation": "SYNTHETIC warm/cold preparation", "cache_evidence_sha256": fake_hash("cache")}
        report["cases"].append({"id": f"SYNTHETIC-{index}", "evidence_kind": "synthetic", "unit": unit, "target_format": fmt, "role": role, "batch_size": batch, "threads": threads, "cache_state": cache, "numa_policy": policy, "worker_node": node, "profile_id": profile["id"] if unit == "full" else None, "seed": 7, "input_sha256": input_hash, "packed_sha256": profile["blob_sha256"] if unit == "full" else fake_hash(f"packed-{index}"), "control": control, "dispatch": {"arithmetic": "default", "isa": "avx2", "runtime_usable": True, "gate_up_activation": activation, "down_activation": "Q8_0" if unit == "full" else activation, "implementation": "SYNTHETIC-NOT-EXECUTED", "environment": {}}, "placement": placement(threads, node, policy), "attempt": {"started": True, "exit_code": 0, "elapsed_wall_ns": 1000, "observation_sha256": fake_hash("attempt")}, "outcome": "pass", "reason": None, "checks": checks, "trace": trace, "timing": timing})
    return report


def first_full(report):
    return next(case for case in report["cases"] if case["unit"] == "full")


def synthetic_down_format(report, target, down_format):
    """Repack declared fixture geometry only; this never creates real weights."""
    profile = next(item for item in report["profiles"] if item["target_format"] == target)
    row = profile["roles"]["down"]
    type_id, block_values, block_bytes, activation = cpu.TYPE_INFO[down_format]
    columns = profile["n_ff"]
    if columns % block_values:
        raise ValueError("synthetic down columns must contain whole blocks")
    row_bytes = columns // block_values * block_bytes
    row.update(format=down_format, type_id=type_id, block_values=block_values,
               block_bytes=block_bytes, row_bytes=row_bytes, native_vec_dot_activation=activation,
               tensor_sha256=fake_hash(profile["id"] + ":down:" + down_format))
    profile["blob_bytes"] = row["offset_bytes"] + row_bytes * profile["n_embd"]
    profile["blob_sha256"] = fake_hash(profile["id"] + ":blob:" + down_format)
    for case in report["cases"]:
        if case["unit"] == "full" and case["profile_id"] == profile["id"]:
            case["packed_sha256"] = profile["blob_sha256"]
            case["dispatch"]["down_activation"] = "ActQ-legacy" if down_format == "Q2_0" else activation


def failed_check(case, element=0):
    item = case["checks"][0]
    item["actual"][element] += 1
    item["actual_sha256"] = cpu.vector_sha256(item["actual"])
    item["failure_count"] = 1
    item["passed"] = False
    if case["trace"] is not None:
        case["trace"]["gate_output_sha256"] = item["actual_sha256"]
        case["trace"]["silu_gate_input_sha256"] = item["actual_sha256"]


class CpuEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = cpu.load_json(ROOT / "dependencies.lock.json")
        cls.synthetic = synthetic_report()

    def report(self):
        return copy.deepcopy(self.synthetic)

    def validate(self, report, mode="partial"):
        return cpu.validate(report, self.lock, mode)

    def test_default_partial_template_is_honest_and_inert(self):
        result = self.validate(cpu.load_json(TEMPLATE))
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(result["mode_satisfied"])
        self.assertFalse(result["schema_complete"])
        self.assertFalse(result["measured_evidence_complete"])
        self.assertFalse(result["declared_cpu_correctness_pass"])
        self.assertFalse(result["inference_ready"])
        self.assertFalse(result["measurements_independently_verified"])

    def test_partial_template_fails_both_strict_modes(self):
        for mode in ("strict-complete", "strict-accept"):
            with self.subTest(mode=mode):
                self.assertFalse(self.validate(cpu.load_json(TEMPLATE), mode)["mode_satisfied"])

    def test_complete_synthetic_is_not_measured_or_correctness_evidence(self):
        result = self.validate(self.report())
        self.assertTrue(result["schema_complete"])
        self.assertEqual(result["cases_present"], 237)
        self.assertEqual(result["cases_required"], 237)
        self.assertFalse(result["measured_evidence_complete"])
        self.assertFalse(result["declared_cpu_correctness_pass"])
        for mode in ("strict-complete", "strict-accept"):
            self.assertFalse(self.validate(self.report(), mode)["mode_satisfied"])

    def test_synthetic_as_measured_is_rejected(self):
        report = self.report()
        report["evidence_kind"] = "measured"
        with self.assertRaisesRegex(cpu.EvidenceError, "synthetic model fixture"):
            self.validate(report, "strict-complete")

    def test_declared_measured_gate_is_mocked_not_a_cpu_measurement(self):
        # EXPLICIT TEST SEAM: replace profile/case validators with declaration
        # gates, not fabricated values reclassified as actual measurements.
        report = self.report()
        report["evidence_kind"] = "measured"
        profiles = {p["id"]: p for p in report["profiles"]}
        targets = {p["target_format"]: p for p in report["profiles"]}
        with mock.patch.object(cpu, "_validate_profiles", return_value=(profiles, targets)), mock.patch.object(cpu, "_validate_case", return_value=("pass", True)):
            result = self.validate(report, "strict-accept")
        self.assertTrue(result["mode_satisfied"])
        self.assertTrue(result["measured_evidence_complete"])
        self.assertFalse(result["measurements_independently_verified"])
        self.assertFalse(result["inference_ready"])

    def test_complete_reject_is_never_correctness_pass(self):
        report = self.report()
        case = first_full(report)
        failed_check(case)
        case.update(outcome="reject", reason="SYNTHETIC observed elementwise failure")
        result = self.validate(report)
        self.assertTrue(result["schema_complete"])
        self.assertEqual(result["outcomes"]["reject"], 1)
        self.assertFalse(result["declared_cpu_correctness_pass"])

    def test_declared_measured_reject_complete_vs_accept_is_mocked(self):
        report = self.report()
        report["evidence_kind"] = "measured"
        profiles = {p["id"]: p for p in report["profiles"]}
        with mock.patch.object(cpu, "_validate_profiles", return_value=(profiles, {})), mock.patch.object(cpu, "_validate_case", return_value=("reject", False)):
            complete = self.validate(report, "strict-complete")
            accepted = self.validate(report, "strict-accept")
        self.assertTrue(complete["mode_satisfied"])
        self.assertFalse(accepted["mode_satisfied"])
        self.assertFalse(complete["declared_cpu_correctness_pass"])
        self.assertFalse(complete["measurements_independently_verified"])

    def test_complete_error_needs_actual_attempt_observations(self):
        report = self.report()
        case = first_full(report)
        case.update(outcome="error", reason="SYNTHETIC guarded refusal", checks=[], trace=None, timing=None)
        case["attempt"]["exit_code"] = 2
        result = self.validate(report)
        self.assertTrue(result["schema_complete"])
        self.assertFalse(result["declared_cpu_correctness_pass"])
        case["attempt"]["observation_sha256"] = None
        self.assertFalse(self.validate(report)["schema_complete"])
        case["attempt"]["started"] = False
        with self.assertRaisesRegex(cpu.EvidenceError, "unattempted"):
            self.validate(report)

    def test_error_cannot_waive_host_build_model_or_placement(self):
        report = self.report()
        for case in report["cases"]:
            case.update(outcome="error", reason="SYNTHETIC refusal", checks=[], trace=None, timing=None)
            case["attempt"]["exit_code"] = 2
        self.assertTrue(self.validate(report)["schema_complete"])
        for section, key in (("host", "feature_probe_sha256"), ("build", "binary_sha256")):
            candidate = copy.deepcopy(report)
            candidate[section][key] = None
            self.assertFalse(self.validate(candidate)["schema_complete"])
        report["profiles"][0]["tensor_metadata_sha256"] = None
        first_full(report)["placement"]["observed"]["memory_sha256"] = None
        self.assertFalse(self.validate(report)["schema_complete"])

    def test_empty_error_reason_and_false_error_exit_fail(self):
        for reason, exit_code in (("", 2), ("SYNTHETIC error", 0)):
            report = self.report()
            case = first_full(report)
            case.update(outcome="error", reason=reason, checks=[], trace=None, timing=None)
            case["attempt"]["exit_code"] = exit_code
            with self.subTest(reason=reason, exit_code=exit_code), self.assertRaises(cpu.EvidenceError):
                self.validate(report)

    def test_error_cannot_claim_numerical_or_timing_pass(self):
        report = self.report()
        first_full(report).update(outcome="error", reason="SYNTHETIC")
        first_full(report)["attempt"]["exit_code"] = 1
        with self.assertRaisesRegex(cpu.EvidenceError, "no claimed"):
            self.validate(report)

    def test_missing_and_duplicate_matrix_coverage(self):
        report = self.report()
        report["cases"].pop()
        result = self.validate(report)
        self.assertFalse(result["schema_complete"])
        self.assertEqual(len(result["missing_cases"]), 1)
        report = self.report()
        duplicate = copy.deepcopy(report["cases"][0])
        duplicate["id"] = "SYNTHETIC-new-id-same-matrix"
        report["cases"].append(duplicate)
        with self.assertRaisesRegex(cpu.EvidenceError, "duplicate matrix"):
            self.validate(report)

    def test_duplicate_case_and_profile_ids(self):
        report = self.report()
        report["cases"][1]["id"] = report["cases"][0]["id"]
        with self.assertRaisesRegex(cpu.EvidenceError, "duplicate case id"):
            self.validate(report)
        report = self.report()
        report["profiles"][1]["id"] = report["profiles"][0]["id"]
        with self.assertRaisesRegex(cpu.EvidenceError, "duplicate profile id"):
            self.validate(report)

    def test_cannot_shrink_fixed_format_role_or_numa_matrix(self):
        for axis in ("formats", "roles", "numa_policies", "cache_states"):
            report = self.report()
            report["matrix"][axis].pop()
            with self.subTest(axis=axis), self.assertRaisesRegex(cpu.EvidenceError, "exactly"):
                self.validate(report)

    def test_single_token_small_batch_threads_and_two_numa_nodes_required(self):
        for axis, value in (("batch_sizes", [1]), ("thread_counts", [1]), ("worker_nodes", [0])):
            report = self.report()
            report["matrix"][axis] = value
            with self.subTest(axis=axis), self.assertRaises(cpu.EvidenceError):
                self.validate(report)

    def test_conflicting_types_shapes_rows_offsets_and_activation(self):
        changes = (("type_id", 21), ("block_bytes", 110), ("row_bytes", 999), ("offset_bytes", 5), ("shape", [64, 512]), ("native_vec_dot_activation", "Q8_0"))
        for key, value in changes:
            report = self.report()
            report["profiles"][0]["roles"]["gate"][key] = value
            with self.subTest(key=key), self.assertRaises(cpu.EvidenceError):
                self.validate(report)

    def test_iq_down_width640_is_rejected_not_fake_packed(self):
        report = self.report()
        profile = report["profiles"][0]
        profile["n_ff"] = 640
        profile["roles"]["gate"]["shape"] = [640, 256]
        profile["roles"]["up"]["shape"] = [640, 256]
        profile["roles"]["down"].update(format="IQ2_XS", type_id=17, block_values=256, block_bytes=74, native_vec_dot_activation="Q8_K", shape=[256, 640])
        with self.assertRaisesRegex(cpu.EvidenceError, "not whole packed"):
            self.validate(report)

    def test_ggml_reference_sha_must_match_reviewed_build_identity(self):
        report = self.report()
        identity = report["cases"][0]["checks"][0]["reference_identity"]
        identity.update(kind="ggml-pinned-reference", revision="b" * 40)
        self.assertFalse(self.validate(report)["schema_complete"])
        report["build"]["ggml_revision"] = "b" * 40
        self.assertTrue(self.validate(report)["schema_complete"])
        identity["revision"] = "c" * 40
        with self.assertRaisesRegex(cpu.EvidenceError, "revision"):
            self.validate(report)

    def test_reference_shape_counter_and_hash_cannot_lie(self):
        for key, value in (("shape", [4, 999]), ("failure_count", 1), ("passed", False), ("actual_sha256", fake_hash("wrong"))):
            report = self.report()
            report["cases"][0]["checks"][0][key] = value
            with self.subTest(key=key), self.assertRaises(cpu.EvidenceError):
                self.validate(report)

    def test_elementwise_abs_plus_rel_not_global_maxima_or(self):
        report = self.report()
        case = report["cases"][0]
        item = case["checks"][0]
        item["reference"][0] = 0.0
        item["actual"][0] = 0.01
        item["reference"][1] = 1000000.0
        item["actual"][1] = 1000000.0
        item["atol"] = 0.001
        item["rtol"] = 0.0001
        item["actual_sha256"] = cpu.vector_sha256(item["actual"])
        item["reference_sha256"] = cpu.vector_sha256(item["reference"])
        # Global max relative error normalized by a large reference would look
        # small, but element 0 violates its own atol+rtol*abs(reference) bound.
        with self.assertRaisesRegex(cpu.EvidenceError, "failure_count"):
            self.validate(report)
        item.update(failure_count=1, passed=False)
        case.update(outcome="reject", reason="SYNTHETIC element0 fails")
        self.assertTrue(self.validate(report)["schema_complete"])

    def test_abs_and_rel_allowance_are_added_per_element(self):
        report = self.report()
        item = report["cases"][0]["checks"][0]
        item["atol"], item["rtol"] = 0.01, 0.01
        item["reference"][0], item["actual"][0] = 1.0, 1.015
        item["actual_sha256"] = cpu.vector_sha256(item["actual"])
        item["reference_sha256"] = cpu.vector_sha256(item["reference"])
        self.assertTrue(self.validate(report)["schema_complete"])

    def test_false_pass_and_false_reject_are_contradictions(self):
        report = self.report()
        failed_check(report["cases"][0])
        with self.assertRaisesRegex(cpu.EvidenceError, "pass contradicts"):
            self.validate(report)
        report = self.report()
        first_full(report).update(outcome="reject", reason="SYNTHETIC made-up rejection")
        with self.assertRaisesRegex(cpu.EvidenceError, "reject requires"):
            self.validate(report)

    def test_full_branch_dataflow_cannot_use_independent_random_h(self):
        for name in ("gate_input_sha256", "up_input_sha256", "silu_gate_input_sha256", "silu_up_input_sha256", "down_quantization_input_sha256", "down_input_sha256", "down_output_sha256"):
            report = self.report()
            first_full(report)["trace"][name] = fake_hash("independent")
            with self.subTest(name=name), self.assertRaises(cpu.EvidenceError):
                self.validate(report)

    def test_missing_full_check_or_synchronization_is_not_complete(self):
        report = self.report()
        case = first_full(report)
        case["checks"].pop()
        case.update(outcome="reject", reason="SYNTHETIC incomplete checks")
        self.assertFalse(self.validate(report)["schema_complete"])
        report = self.report()
        first_full(report)["trace"]["synchronized"] = False
        with self.assertRaisesRegex(cpu.EvidenceError, "synchronization"):
            self.validate(report)

    def test_raw_wall_statistics_and_boundaries_recomputed(self):
        for key, value in (("median_ns", 999), ("p95_ns", 999), ("variance_ns2", 999), ("clock", "cpu-time"), ("scope", "gu-plus-independent-down"), ("raw_end_ns", [1, 2, 3, 4, 5])):
            report = self.report()
            first_full(report)["timing"][key] = value
            with self.subTest(key=key), self.assertRaises(cpu.EvidenceError):
                self.validate(report)

    def test_warmups_repeat_count_and_missing_cache_observation(self):
        report = self.report()
        first_full(report)["timing"]["warmup_wall_ns"] = [1, 2]
        with self.assertRaises(cpu.EvidenceError):
            self.validate(report)
        report = self.report()
        first_full(report)["timing"]["cache_evidence_sha256"] = None
        self.assertFalse(self.validate(report)["schema_complete"])
        report = self.report()
        next(c for c in report["cases"] if c["cache_state"] == "cold")["timing"]["cache_preparation"] = "none"
        with self.assertRaises(cpu.EvidenceError):
            self.validate(report)

    def test_requested_numa_affinity_does_not_replace_observed(self):
        report = self.report()
        case = first_full(report)
        case["placement"]["observed"]["workers"] = None
        case.update(outcome="reject", reason="SYNTHETIC missing affinity observation")
        self.assertFalse(self.validate(report)["schema_complete"])
        report = self.report()
        case = first_full(report)
        case["placement"]["observed"]["first_touch"]["evidence_sha256"] = None
        self.assertFalse(self.validate(report)["schema_complete"])

    def test_wrong_effective_affinity_can_be_observed_reject_not_pass(self):
        report = self.report()
        case = next(c for c in report["cases"] if c["unit"] == "full" and c["worker_node"] == 0)
        worker = case["placement"]["observed"]["workers"][0]
        worker["effective_cpu_ids"], worker["sampled_cpu_ids"] = [2], [2]
        with self.assertRaisesRegex(cpu.EvidenceError, "pass contradicts"):
            self.validate(report)
        case.update(outcome="reject", reason="SYNTHETIC effective affinity is remote")
        self.assertTrue(self.validate(report)["schema_complete"])

    def test_page_counts_first_touch_and_remote_local_interleave_observations(self):
        for mutation in ("page-count", "touch-cpu", "remote-is-local", "interleave-is-one-node"):
            report = self.report()
            policy = "remote" if mutation == "remote-is-local" else "interleave" if mutation == "interleave-is-one-node" else "local"
            case = next(c for c in report["cases"] if c["unit"] == "full" and c["numa_policy"] == policy and c["worker_node"] == 0)
            observed = case["placement"]["observed"]
            if mutation == "page-count":
                observed["total_pages"] += 1
            elif mutation == "touch-cpu":
                observed["first_touch"]["cpu_ids"] = [999]
            else:
                observed["page_counts"] = [{"node": 0, "pages": 10}]
                observed["total_pages"] = 10
                observed["first_touch"].update(cpu_ids=[0], node_ids=[0], pages_touched=10)
            with self.subTest(mutation=mutation), self.assertRaises(cpu.EvidenceError):
                self.validate(report)

    def test_coherent_memory_placement_mismatch_is_reject_not_success(self):
        for requested_policy, achieved_policy in (("local", "remote"), ("remote", "local"), ("interleave", "local")):
            report = self.report()
            case = next(c for c in report["cases"] if c["unit"] == "full" and c["numa_policy"] == requested_policy and c["worker_node"] == 0)
            observed = case["placement"]["observed"]
            achieved = placement(case["threads"], 0, achieved_policy)["observed"]
            for name in ("memory_policy", "page_counts", "total_pages", "first_touch"):
                observed[name] = achieved[name]
            with self.subTest(requested=requested_policy, achieved=achieved_policy), self.assertRaisesRegex(cpu.EvidenceError, "pass contradicts"):
                self.validate(report)
            case.update(outcome="reject", reason="SYNTHETIC achieved residency differs from the requested strategy")
            self.assertTrue(self.validate(report)["schema_complete"])

    def test_requested_plan_cannot_contradict_its_matrix_even_with_reject(self):
        mutations = (("local", "memory_policy", "remote"), ("local", "cpu_ids", [2]),
                     ("local", "memory_nodes", [1]), ("remote", "memory_nodes", [0]),
                     ("interleave", "memory_nodes", [0]))
        for policy, name, value in mutations:
            report = self.report()
            case = next(c for c in report["cases"] if c["unit"] == "full" and c["numa_policy"] == policy and c["worker_node"] == 0)
            case["placement"]["requested"][name] = value
            case.update(outcome="reject", reason="SYNTHETIC reason cannot repair a contradictory request")
            with self.subTest(policy=policy, field=name), self.assertRaisesRegex(cpu.EvidenceError, "requested"):
                self.validate(report)

    def test_internal_observation_contradictions_cannot_be_reject(self):
        for mutation in ("sample-outside-mask", "touch-exceeds-total", "policy-contradicts-pages", "touch-nodes-contradict-cpus"):
            report = self.report()
            case = next(c for c in report["cases"] if c["unit"] == "full" and c["numa_policy"] == "local" and c["worker_node"] == 0)
            observed = case["placement"]["observed"]
            if mutation == "sample-outside-mask":
                observed["workers"][0]["sampled_cpu_ids"] = [2]
            elif mutation == "touch-exceeds-total":
                observed["first_touch"]["pages_touched"] = observed["total_pages"] + 1
            elif mutation == "policy-contradicts-pages":
                observed["page_counts"] = [{"node": 1, "pages": 10}]
                observed["first_touch"].update(cpu_ids=[2], node_ids=[1])
            else:
                observed["first_touch"]["node_ids"] = [1]
            case.update(outcome="reject", reason="SYNTHETIC reason cannot repair contradictory observations")
            with self.subTest(mutation=mutation), self.assertRaises(cpu.EvidenceError):
                self.validate(report)

    def test_partial_first_touch_and_remote_competition_are_honest_rejects(self):
        for mutation in ("partial-touch", "remote-competition"):
            report = self.report()
            case = next(c for c in report["cases"] if c["unit"] == "full" and c["numa_policy"] == "contention" and c["worker_node"] == 0)
            observed = case["placement"]["observed"]
            if mutation == "partial-touch":
                observed["first_touch"]["pages_touched"] -= 1
            else:
                observed["contention"]["cpu_ids"] = [2]
            with self.subTest(mutation=mutation), self.assertRaisesRegex(cpu.EvidenceError, "pass contradicts"):
                self.validate(report)
            case.update(outcome="reject", reason="SYNTHETIC requested placement gate failed")
            self.assertTrue(self.validate(report)["schema_complete"])

    def test_contention_needs_effective_competing_workload(self):
        report = self.report()
        case = next(c for c in report["cases"] if c["numa_policy"] == "contention")
        case["placement"]["observed"]["contention"] = {"active": False, "cpu_ids": [], "memory_nodes": [], "description": None, "evidence_sha256": None}
        with self.assertRaisesRegex(cpu.EvidenceError, "pass contradicts"):
            self.validate(report)
        case.update(outcome="reject", reason="SYNTHETIC contention failed to start")
        self.assertTrue(self.validate(report)["schema_complete"])

    def test_cpuid_xcr0_and_osxsave_are_actual_separate_gates(self):
        report = self.report()
        report["host"]["xcr0"] = 2
        with self.assertRaisesRegex(cpu.EvidenceError, "avx2_usable"):
            self.validate(report)
        report = self.report()
        report["host"]["features"]["osxsave"] = False
        report["host"]["cpuid_1_ecx"] &= ~(1 << 27)
        with self.assertRaisesRegex(cpu.EvidenceError, "OSXSAVE"):
            self.validate(report)
        report = self.report()
        report["host"]["cpuid_7_ebx"] = 0
        with self.assertRaisesRegex(cpu.EvidenceError, "features.avx2"):
            self.validate(report)

    def test_cpuid_and_xcr0_register_width_boundaries(self):
        report = self.report()
        report["host"].update(cpuid_1_ecx=2**32 - 1, cpuid_7_ebx=2**32 - 1, xcr0=2**64 - 1)
        self.assertTrue(self.validate(report)["schema_complete"])
        for field, value, message in (("cpuid_1_ecx", 2**32, "uint32"),
                                      ("cpuid_7_ebx", 2**32, "uint32"),
                                      ("xcr0", 2**64, "uint64")):
            report = self.report()
            report["host"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(cpu.EvidenceError, message):
                self.validate(report)

    def test_guarded_isa_refusal_is_error_evidence_without_executing_xgetbv(self):
        report = self.report()
        report["host"]["features"]["osxsave"] = False
        report["host"]["cpuid_1_ecx"] &= ~(1 << 27)
        report["host"].update(xgetbv_executed=False, xcr0=None, avx2_usable=False)
        for case in report["cases"]:
            case["dispatch"]["runtime_usable"] = False
            case.update(outcome="error", reason="SYNTHETIC guarded ISA refusal", checks=[], trace=None, timing=None)
            case["attempt"]["exit_code"] = 2
        self.assertTrue(self.validate(report)["schema_complete"])
        report["host"]["xcr0"] = 0
        with self.assertRaisesRegex(cpu.EvidenceError, "fabricated zero"):
            self.validate(report)

    def test_isa_caps_do_not_downgrade_compile_floor(self):
        report = self.report()
        report["build"]["compiled_isa_floor"] = "avx512"
        report["build"]["environment"] = {"STRATA_FORCE_ISA": "avx"}
        with self.assertRaisesRegex(cpu.EvidenceError, "compiled_isa_floor"):
            self.validate(report)
        report = self.report()
        report["build"]["avx2_translation_unit_flags"] = ["-mavx2"]
        with self.assertRaisesRegex(cpu.EvidenceError, "FMA/F16C"):
            self.validate(report)

    def test_report_wide_arithmetic_choices_must_agree_including_reject_error(self):
        for outcome in ("pass", "reject", "error"):
            report = self.report()
            case = report["cases"][0]
            if outcome == "reject":
                failed_check(case)
                case.update(outcome="reject", reason="SYNTHETIC numerical rejection")
            elif outcome == "error":
                case.update(outcome="error", reason="SYNTHETIC attempted error", checks=[], trace=None, timing=None)
                case["attempt"]["exit_code"] = 2
            self.assertTrue(self.validate(report)["schema_complete"])
            case["dispatch"]["arithmetic"] = "window-invariant"
            with self.subTest(outcome=outcome), self.assertRaisesRegex(cpu.EvidenceError, "report-wide arithmetic"):
                self.validate(report)

    def test_unknown_arithmetic_is_missing_not_default(self):
        report = self.report()
        for case in report["cases"]:
            case["dispatch"]["arithmetic"] = "window-invariant"
        self.assertTrue(self.validate(report)["schema_complete"])
        report["cases"][0]["dispatch"]["arithmetic"] = None
        result = self.validate(report)
        self.assertFalse(result["schema_complete"])
        self.assertIn("cases[0].dispatch.arithmetic", result["missing_fields"])
        report["cases"][1]["dispatch"]["arithmetic"] = "default"
        with self.assertRaisesRegex(cpu.EvidenceError, "report-wide arithmetic"):
            self.validate(report)

    def test_q2_activation_must_agree_across_cases_units_and_roles(self):
        for unit, role, slot in (("block", "gate", "gate_up_activation"),
                                 ("block", "down", "down_activation"),
                                 ("dot", "up", "gate_up_activation"),
                                 ("dot", "down", "down_activation"),
                                 ("full", "all", "gate_up_activation")):
            report = self.report()
            case = next(item for item in report["cases"] if item["target_format"] == "Q2_0" and item["unit"] == unit and item["role"] == role)
            case["dispatch"][slot] = "ActQ-oracle-Q8_0"
            with self.subTest(unit=unit, role=role, slot=slot), self.assertRaisesRegex(cpu.EvidenceError, "report-wide Q2_0 activation"):
                self.validate(report)

    def test_same_case_q2_slots_must_agree_for_controls_and_full_actual_roles(self):
        for unit in ("block", "dot", "full"):
            report = self.report()
            if unit == "full":
                synthetic_down_format(report, "Q2_0", "Q2_0")
            case = next(item for item in report["cases"] if item["target_format"] == "Q2_0" and item["unit"] == unit)
            case["dispatch"]["down_activation"] = "ActQ-oracle-Q8_0"
            # Isolate one case: this must fail within its two Q2 slots, not just
            # because another matrix entry chose a different process flag.
            report["cases"] = [case]
            with self.subTest(unit=unit), self.assertRaisesRegex(cpu.EvidenceError, "report-wide Q2_0 activation"):
                self.validate(report)

    def test_q2_activation_consistency_includes_reject_error(self):
        for outcome in ("reject", "error"):
            report = self.report()
            case = next(item for item in report["cases"] if item["target_format"] == "Q2_0" and item["unit"] == "full")
            if outcome == "reject":
                failed_check(case)
                case.update(outcome="reject", reason="SYNTHETIC numerical rejection")
            else:
                case.update(outcome="error", reason="SYNTHETIC attempted error", checks=[], trace=None, timing=None)
                case["attempt"]["exit_code"] = 2
            self.assertTrue(self.validate(report)["schema_complete"])
            case["dispatch"]["gate_up_activation"] = "ActQ-oracle-Q8_0"
            with self.subTest(outcome=outcome), self.assertRaisesRegex(cpu.EvidenceError, "report-wide Q2_0 activation"):
                self.validate(report)

    def test_q2_activation_uses_actual_down_format_not_target_label(self):
        report = self.report()
        synthetic_down_format(report, "IQ2_XS", "Q2_0")
        self.assertTrue(self.validate(report)["schema_complete"])
        case = next(item for item in report["cases"] if item["unit"] == "full" and item["target_format"] == "IQ2_XS")
        self.assertEqual(case["dispatch"]["gate_up_activation"], "Q8_K")
        case["dispatch"]["down_activation"] = "ActQ-oracle-Q8_0"
        with self.assertRaisesRegex(cpu.EvidenceError, "report-wide Q2_0 activation"):
            self.validate(report)

    def test_uniform_oracle_choice_and_legal_mixed_roles_remain_schema_complete(self):
        report = self.report()
        synthetic_down_format(report, "IQ2_XS", "Q2_0")
        synthetic_down_format(report, "Q2_0", "Q2_0")
        for case in report["cases"]:
            for slot in ("gate_up_activation", "down_activation"):
                if case["dispatch"][slot] == "ActQ-legacy":
                    case["dispatch"][slot] = "ActQ-oracle-Q8_0"
        result = self.validate(report)
        self.assertTrue(result["schema_complete"])
        self.assertFalse(result["measured_evidence_complete"])
        self.assertFalse(result["inference_ready"])
        self.assertTrue(any(item["dispatch"]["gate_up_activation"] != item["dispatch"]["down_activation"] for item in report["cases"] if item["unit"] == "full"))

    def test_unknown_activation_is_missing_not_a_process_default(self):
        report = self.report()
        q2_cases = [item for item in report["cases"] if item["target_format"] == "Q2_0"]
        for case in q2_cases:
            for slot in ("gate_up_activation", "down_activation"):
                if case["dispatch"][slot] == "ActQ-legacy":
                    case["dispatch"][slot] = "ActQ-oracle-Q8_0"
        case = next(item for item in q2_cases if item["unit"] == "full")
        case["dispatch"]["gate_up_activation"] = None
        result = self.validate(report)
        self.assertFalse(result["schema_complete"])
        index = report["cases"].index(case)
        self.assertIn(f"cases[{index}].dispatch.gate_up_activation", result["missing_fields"])
        case["dispatch"]["gate_up_activation"] = "ActQ-legacy"
        with self.assertRaisesRegex(cpu.EvidenceError, "report-wide Q2_0 activation"):
            self.validate(report)

    def test_unknown_profile_id_does_not_substitute_target_for_down_format(self):
        report = self.report()
        case = next(item for item in report["cases"] if item["unit"] == "full" and item["target_format"] == "IQ2_XS")
        case["profile_id"] = None
        case["dispatch"]["gate_up_activation"] = "Q8_0"
        case["dispatch"]["down_activation"] = "ActQ-oracle-Q8_0"
        result = self.validate(report)
        self.assertFalse(result["schema_complete"])
        index = report["cases"].index(case)
        self.assertIn(f"cases[{index}].profile_id", result["missing_fields"])

    def test_unknown_actual_down_format_stays_incomplete_without_invented_mapping(self):
        report = self.report()
        profile = next(item for item in report["profiles"] if item["target_format"] == "IQ2_XS")
        profile["roles"]["down"]["format"] = None
        for case in report["cases"]:
            if case["unit"] == "full" and case["profile_id"] == profile["id"]:
                case["dispatch"]["down_activation"] = "ActQ-oracle-Q8_0"
        result = self.validate(report)
        self.assertFalse(result["schema_complete"])
        index = report["profiles"].index(profile)
        self.assertIn(f"profiles[{index}].roles.down.format", result["missing_fields"])

    def test_q2_actual_pool_activation_not_native_metadata_label(self):
        report = self.report()
        case = next(c for c in report["cases"] if c["target_format"] == "Q2_0")
        case["dispatch"]["gate_up_activation"] = "Q8_0"
        with self.assertRaisesRegex(cpu.EvidenceError, "actual native/pool"):
            self.validate(report)

    def test_nonfinite_bool_negative_numbers_and_unknown_fields_fail_closed(self):
        for value in (float("nan"), float("inf"), True, -1):
            report = self.report()
            report["cases"][0]["timing"]["raw_wall_ns"][0] = value
            with self.subTest(value=value), self.assertRaises(cpu.EvidenceError):
                self.validate(report)
        report = self.report()
        report["cases"][0]["checks"][0]["actual"][0] = True
        with self.assertRaises(cpu.EvidenceError):
            self.validate(report)
        report = self.report()
        report["cases"][0]["global_max_error_pass"] = True
        with self.assertRaisesRegex(cpu.EvidenceError, "unknown"):
            self.validate(report)

    def test_huge_integer_fields_and_vector_hash_fail_as_evidence_errors(self):
        huge = 10**1000
        for location in ("sample", "tolerance", "timestamp", "seed"):
            report = self.report()
            case = report["cases"][0]
            if location == "sample":
                case["checks"][0]["actual"][0] = huge
            elif location == "tolerance":
                case["checks"][0]["atol"] = huge
            elif location == "timestamp":
                case["timing"]["raw_start_ns"][0] = huge
            else:
                case["seed"] = huge
            with self.subTest(location=location), self.assertRaisesRegex(cpu.EvidenceError, "overflowing"):
                self.validate(report)
        with self.assertRaises(cpu.EvidenceError):
            cpu.vector_sha256([huge])

    def test_elementwise_float_and_integer_arithmetic_overflow_fail_closed(self):
        for actual, reference, rtol in (
            (1e308, -1e308, 0.0),
            (10**308, -(10**308), 0),
            (1e308, 1e308, 1e308),
            (10**308, 10**308, 10**308),
        ):
            report = self.report()
            item = report["cases"][0]["checks"][0]
            item["actual"][0], item["reference"][0], item["rtol"] = actual, reference, rtol
            item["actual_sha256"] = cpu.vector_sha256(item["actual"])
            item["reference_sha256"] = cpu.vector_sha256(item["reference"])
            with self.subTest(actual=actual, reference=reference, rtol=rtol), self.assertRaisesRegex(cpu.EvidenceError, "overflow in numerical"):
                self.validate(report)

    def test_raw_wall_variance_overflow_is_evidence_error_and_cli1(self):
        report = self.report()
        timing = report["cases"][0]["timing"]
        raw = [1, 10**200, 1, 1, 1]
        starts, ends, cursor = [], [], 0
        for duration in raw:
            starts.append(cursor)
            cursor += duration
            ends.append(cursor)
        timing.update(raw_wall_ns=raw, raw_start_ns=starts, raw_end_ns=ends,
                      median_ns=None, p95_ns=None, variance_ns2=None)
        with self.assertRaisesRegex(cpu.EvidenceError, "overflow in recomputed"):
            self.validate(report)
        with tempfile.TemporaryDirectory(prefix="cpu-evidence-overflow-") as temporary:
            path = Path(temporary) / "synthetic.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            error = io.StringIO()
            with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                cpu.main([str(path)])
            self.assertEqual(raised.exception.code, 1)
            self.assertIn("validation failed", error.getvalue())
            self.assertNotIn("Traceback", error.getvalue())

    def test_huge_json_integer_parse_and_cli_fail_without_traceback(self):
        with tempfile.TemporaryDirectory(prefix="cpu-evidence-integers-") as temporary:
            path = Path(temporary) / "synthetic.json"
            for digits in (1000, 10000):
                path.write_text('{"synthetic": ' + "9" * digits + "}", encoding="utf-8")
                with self.subTest(digits=digits), self.assertRaisesRegex(cpu.EvidenceError, "integer"):
                    cpu.load_json(path)
                error = io.StringIO()
                with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                    cpu.main([str(path)])
                self.assertEqual(raised.exception.code, 1)
                self.assertNotIn("Traceback", error.getvalue())

    def test_full_routing_skip0_mtp_off_and_pinned_source(self):
        for section, key, value in (("routing", "top_k", 5), ("routing", "expert_skip", 1), ("routing", "mtp", "on"), ("source", "revision", "a" * 40)):
            report = self.report()
            report[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(cpu.EvidenceError):
                self.validate(report)

    def test_json_input_text_is_inert_even_when_it_looks_executable(self):
        report = self.report()
        command = "$(touch /tmp/CPU-EVIDENCE-MUST-NOT-EXECUTE); https://untrusted.invalid/run"
        report["notes"] = [command]
        report["build"]["environment"] = {"STRATA_Q2_BITPLANE": "1", "INERT_TEST": command}
        for case in report["cases"]:
            case["dispatch"]["environment"] = report["build"]["environment"].copy()
        with mock.patch("subprocess.run", side_effect=AssertionError("must not execute")), mock.patch("urllib.request.urlopen", side_effect=AssertionError("must not fetch")):
            self.assertTrue(self.validate(report)["schema_complete"])

    def test_json_duplicate_keys_and_nonstandard_constants_rejected(self):
        with tempfile.TemporaryDirectory(prefix="cpu-evidence-tests-") as temporary:
            path = Path(temporary) / "synthetic.json"
            for text in ('{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}', '{"x": 1e999}'):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(cpu.EvidenceError):
                    cpu._finite_tree(cpu.load_json(path))

    def test_cli_default_and_strict_modes_match_protocol(self):
        for arguments, expected in (([], 0), (["--strict-complete"], 1), (["--strict-accept"], 1)):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cpu.main(arguments)
            self.assertEqual(code, expected)
            self.assertFalse(json.loads(output.getvalue())["inference_ready"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            cpu.main(["--strict-complete", "--strict-accept"])
        self.assertEqual(raised.exception.code, 2)

    def test_cli_only_reads_explicit_report_and_integration_lock(self):
        with tempfile.TemporaryDirectory(prefix="cpu-evidence-cli-") as temporary:
            path = Path(temporary) / "synthetic.json"
            report = cpu.load_json(TEMPLATE)
            report["notes"] = [str(Path(temporary) / "NEVER-READ-OR-WRITE")]
            path.write_text(json.dumps(report), encoding="utf-8")
            with mock.patch.object(cpu, "load_json", wraps=cpu.load_json) as loader, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cpu.main([str(path)]), 0)
            self.assertEqual([Path(call.args[0]) for call in loader.call_args_list], [path, ROOT / "dependencies.lock.json"])
            self.assertFalse((Path(temporary) / "NEVER-READ-OR-WRITE").exists())

    def test_cli_malformed_input_is_exit1_not_traceback(self):
        with tempfile.TemporaryDirectory(prefix="cpu-evidence-cli-") as temporary:
            path = Path(temporary) / "synthetic.json"
            path.write_text('{"invalid": true}', encoding="utf-8")
            error = io.StringIO()
            with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                cpu.main([str(path)])
            self.assertEqual(raised.exception.code, 1)
            self.assertIn("validation failed", error.getvalue())
            self.assertNotIn("Traceback", error.getvalue())


if __name__ == "__main__":
    unittest.main()
