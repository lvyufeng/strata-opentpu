# Copyright (c) 2026 lvyufeng and contributors
# SPDX-License-Identifier: Apache-2.0
"""Synthetic schema-v2 selection/residency arithmetic tests.

Every input here is a committed synthetic declaration. These tests never read a
packed model tensor, authenticate an artifact hash, probe a host/device, run an
engine or measure performance. A passing suite is arithmetic evidence only; it
does not establish device-ready or inference-ready status.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "resource_budget_v2.py"
FIXTURES = ROOT / "tests" / "fixtures"

SPEC = importlib.util.spec_from_file_location("resource_budget_v2", SCRIPT)
budget = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(budget)

ACCEPTED = ("partial-hot-subset", "full-coverage", "unknown-selection",
            "real-shape-split")
REJECTED = ("overlap-reject", "range-out-of-bounds-reject")


def fixture(name):
    path = FIXTURES / f"resource-budget-v2-{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def clone(name):
    return copy.deepcopy(fixture(name))


def tensor(document, tensor_id):
    for item in document["memory"]["tensors"]:
        if item["id"] == tensor_id:
            return item
    raise AssertionError(f"no tensor {tensor_id}")


def allocation(document, allocation_id):
    for item in document["memory"]["allocations"]:
        if item["id"] == allocation_id:
            return item
    raise AssertionError(f"no allocation {allocation_id}")


def stage(document, stage_id):
    for item in document["wall"]["stages"]:
        if item["id"] == stage_id:
            return item
    raise AssertionError(f"no stage {stage_id}")


def operation(document, stage_id, branch, operation_id):
    for item in stage(document, stage_id)[branch]:
        if item["id"] == operation_id:
            return item
    raise AssertionError(f"no operation {operation_id}")


def run_cli(*arguments):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, arguments)],
                          capture_output=True, text=True, check=False)


class BudgetCase(unittest.TestCase):
    def assert_rejected(self, document, pattern):
        with self.assertRaises(budget.BudgetError) as caught:
            budget.analyze(document)
        self.assertRegex(str(caught.exception), pattern)
        return caught.exception


class AcceptedFixtureMatrixTests(BudgetCase):
    def test_accepted_fixture_matrix(self):
        expectations = {
            "partial-hot-subset": {
                "status": "unknown", "complete": False, "strict_pass": False,
                "fit_scope": "partial_coverage", "coverage": "partial",
                "memory": "unknown", "wall": "unknown",
            },
            "full-coverage": {
                "status": "fit", "complete": True, "strict_pass": True,
                "fit_scope": "complete", "coverage": "complete",
                "memory": "fit", "wall": "estimated",
            },
            "unknown-selection": {
                "status": "unknown", "complete": False, "strict_pass": False,
                "fit_scope": "partial_coverage", "coverage": "unknown",
                "memory": "unknown", "wall": "unknown",
            },
            "real-shape-split": {
                "status": "fit", "complete": True, "strict_pass": True,
                "fit_scope": "complete", "coverage": "complete",
                "memory": "fit", "wall": "estimated",
            },
        }
        for name, expected in expectations.items():
            with self.subTest(fixture=name):
                report = budget.analyze(fixture(name))
                self.assertEqual(report["status"], expected["status"])
                self.assertIs(report["complete"], expected["complete"])
                self.assertIs(report["strict_pass"], expected["strict_pass"])
                self.assertEqual(report["fit_scope"], expected["fit_scope"])
                self.assertEqual(report["selection_coverage"]["status"],
                                 expected["coverage"])
                self.assertEqual(report["memory"]["status"], expected["memory"])
                self.assertEqual(report["wall"]["status"], expected["wall"])

    def test_rejected_fixture_matrix(self):
        messages = {
            "overlap-reject": "overlapping expert selections",
            "range-out-of-bounds-reject": "contiguous range exceeds",
        }
        for name, pattern in messages.items():
            with self.subTest(fixture=name):
                self.assert_rejected(fixture(name), pattern)

    def test_reports_never_claim_readiness(self):
        for name in ACCEPTED:
            with self.subTest(fixture=name):
                report = budget.analyze(fixture(name))
                self.assertIs(report["analytical_only"], True)
                self.assertIs(report["input_provenance_verified"], False)
                self.assertIs(report["device_ready"], False)
                self.assertIs(report["inference_ready"], False)
                self.assertIs(report["wall"]["estimate_is_measurement"], False)
                self.assertEqual(len(report["notes"]), 6)
                for note in report["notes"]:
                    self.assertIsInstance(note, str)
                    self.assertTrue(note.strip())

    def test_partial_hot_subset_keeps_conservative_bounds(self):
        report = budget.analyze(fixture("partial-hot-subset"))
        selection = report["selection_coverage"]
        self.assertEqual(selection["expert_axis_size"], 512)
        self.assertEqual(selection["declared_items"], 3 * 512)
        self.assertEqual(selection["accounted_items"], 768)
        self.assertEqual(selection["unaccounted_items"], 768)
        self.assertFalse(selection["full_coverage"])
        self.assertEqual(selection["status"], "partial")
        self.assertEqual(selection["fit_scope"], "partial_coverage")
        by_id = {item["tensor_id"]: item for item in selection["tensors"]}
        gate = by_id["hot-gate-layer0"]
        self.assertEqual(gate["accounted_items"], 128)
        self.assertEqual(gate["coverage"], "partial")
        self.assertEqual(gate["packed_selected_bytes"], 60620800)
        self.assertEqual(gate["packed_total_bytes"], 242483200)
        down = by_id["cold-down-layer0"]
        self.assertEqual(down["coverage"], "complete")
        self.assertEqual(down["packed_selected_bytes"], 1101004800)
        memory = report["memory"]
        self.assertEqual(memory["device_total"]["known_used_lower_bound_bytes"],
                         121241600)
        self.assertEqual(memory["device_total"]["headroom_upper_bound_bytes"],
                         12976128)
        self.assertEqual(memory["host_total"]["known_used_lower_bound_bytes"],
                         1235222528)
        self.assertIsNone(memory["host_total"]["used_bytes"])
        self.assertAlmostEqual(report["wall"]["known_lower_bound_milliseconds"],
                               28.74693333333333, places=9)
        self.assertIsNone(report["wall"]["estimated_accepted_tokens_per_second"])
        self.assertEqual(report["wall"]["accepted_tokens"], 32)
        self.assertEqual(report["unknown_inputs"], [
            "memory.tensors[0].selection: expert-axis coverage is partial "
            "(128 of 512 items)",
            "memory.tensors[1].selection: expert-axis coverage is partial "
            "(128 of 512 items)",
            "wall.residency: complete fitting allocation inventory required for "
            "a wall estimate",
        ])

    def test_full_coverage_charges_selected_extents(self):
        report = budget.analyze(fixture("full-coverage"))
        tensors = {item["id"]: item for item in report["memory"]["tensors"]}
        self.assertEqual(tensors["hot-gate-layer0"]["packed_bytes"], 96)
        self.assertEqual(tensors["hot-gate-layer0"]["resident_bytes"], 48)
        self.assertEqual(tensors["cold-gate-layer0"]["resident_bytes"], 48)
        self.assertEqual(tensors["hot-up-layer0"]["resident_bytes"], 40)
        self.assertEqual(tensors["cold-up-layer0"]["resident_bytes"], 40)
        self.assertEqual(tensors["cold-down-layer0"]["packed_bytes"], 96)
        self.assertEqual(tensors["cold-down-layer0"]["resident_bytes"], 96 + 16)
        allocations = {item["id"]: item for item in report["memory"]["allocations"]}
        self.assertEqual(allocations["hot-experts"]["bytes"], 48 + 40)
        self.assertEqual(allocations["hot-experts"]["aligned_bytes"], 128)
        self.assertEqual(allocations["cold-experts"]["bytes"], 48 + 40 + 96 + 16)
        self.assertEqual(allocations["cold-experts"]["aligned_bytes"], 256)
        self.assertEqual(report["memory"]["device_total"]["used_bytes"], 128)
        self.assertEqual(report["memory"]["host_total"]["used_bytes"], 4416 + 64)
        self.assertAlmostEqual(report["wall"]["milliseconds"],
                               8.540014666666666, places=9)
        self.assertAlmostEqual(
            report["wall"]["estimated_accepted_tokens_per_second"],
            3747.0661642891796, places=6)

    def test_unknown_selection_charges_only_declared_extents(self):
        report = budget.analyze(fixture("unknown-selection"))
        tensors = {item["id"]: item for item in report["memory"]["tensors"]}
        unknown = tensors["unknown-gate-layer0"]
        self.assertEqual(unknown["resident_status"], "unknown")
        self.assertIsNone(unknown["resident_bytes"])
        self.assertEqual(unknown["packed_bytes"], 96)
        allocations = {item["id"]: item for item in report["memory"]["allocations"]}
        self.assertEqual(allocations["hot-experts"]["bytes"], 40)
        self.assertEqual(allocations["cold-experts"]["bytes"], 96)
        self.assertEqual(report["memory"]["device_total"]
                         ["known_used_lower_bound_bytes"], 64)
        paths = report["unknown_inputs"]
        self.assertIn("memory.tensors[0].resident.data_bytes: unknown selection "
                      "extent is not charged", paths)
        self.assertIn("memory.tensors[0].selection: expert-axis coverage is "
                      "unknown; a mode 'unknown' selection is declared", paths)
        self.assertIn("wall.residency: complete fitting allocation inventory "
                      "required for a wall estimate", paths)
        self.assertAlmostEqual(report["wall"]["known_lower_bound_milliseconds"],
                               8.540006666666667, places=9)

    def test_real_shape_split_matches_reviewed_axis_sizes(self):
        report = budget.analyze(fixture("real-shape-split"))
        contract = report["selection_contract"]
        self.assertEqual(contract["expert_count"], 512)
        self.assertEqual(contract["expert_dimension"], 2)
        self.assertEqual(contract["roles"]["ffn_gate_exps"], [2560, 640])
        self.assertEqual(contract["roles"]["ffn_up_exps"], [2560, 640])
        self.assertEqual(contract["roles"]["ffn_down_exps"], [640, 2560])
        selection = report["selection_coverage"]
        self.assertEqual(selection["expert_axis_size"], 512)
        self.assertEqual(selection["declared_items"], 3 * 512)
        self.assertEqual(selection["accounted_items"], 3 * 512)
        tensors = {item["id"]: item for item in report["memory"]["tensors"]}
        # rows-per-item * selected items * row stride, with the reviewed packed
        # row strides, never a dequantized float width.
        self.assertEqual(tensors["hot-gate-layer0"]["packed_bytes"], 242483200)
        self.assertEqual(tensors["hot-gate-layer0"]["resident_bytes"], 60620800)
        self.assertEqual(tensors["cold-gate-layer0"]["resident_bytes"], 181862400)
        self.assertEqual(tensors["hot-up-layer0"]["resident_bytes"], 60620800)
        self.assertEqual(tensors["cold-up-layer0"]["resident_bytes"], 181862400)
        self.assertEqual(tensors["cold-down-layer0"]["packed_bytes"], 1101004800)
        self.assertAlmostEqual(report["wall"]["milliseconds"],
                               28.74693333333333, places=9)
        self.assertAlmostEqual(
            report["wall"]["estimated_accepted_tokens_per_second"],
            1113.1622155638631, places=6)

    def test_declared_fixtures_are_synthetic_and_offline(self):
        for name in ACCEPTED + REJECTED:
            with self.subTest(fixture=name):
                document = fixture(name)
                self.assertEqual(document["input_kind"], "synthetic")
                for record in document["memory"]["tensors"]:
                    source = record["source"]
                    self.assertEqual(source["kind"], "synthetic")
                    self.assertTrue(source["artifact"].endswith(".synthetic"))
                    self.assertEqual(len(source["sha256"]), 64)
                    int(source["sha256"], 16)
                text = json.dumps(document)
                for marker in ("http://", "https://", "/Users/", "~/",
                               "modelscope", "huggingface"):
                    self.assertNotIn(marker, text)


class SchemaGateTests(BudgetCase):
    def test_schema_version_must_be_integer_two(self):
        for value in (1, 3, "2", 2.0, True, None):
            with self.subTest(value=value):
                document = clone("full-coverage")
                document["schema_version"] = value
                self.assert_rejected(document, "schema_version: expected integer 2")

    def test_top_level_fields_are_closed(self):
        document = clone("full-coverage")
        document["artifact_path"] = "weights.gguf"
        self.assert_rejected(document, "budget: unknown fields")

    def test_input_kind_is_bounded(self):
        document = clone("full-coverage")
        document["input_kind"] = "measured"
        self.assert_rejected(document, "input_kind: expected reference, synthetic "
                                       "or declared")

    def test_budget_id_is_required(self):
        document = clone("full-coverage")
        document["id"] = ""
        self.assert_rejected(document, "budget.id: expected nonempty identifier")

    def test_model_must_keep_full_top10_and_no_skip(self):
        for key, value, pattern in (
                ("routed_top_k", 9, "model.routed_top_k: full top10"),
                ("expert_skip", 1, "model.expert_skip: full top10"),
                ("shared_expert", False, "model.shared_expert: shared expert")):
            with self.subTest(field=key):
                document = clone("full-coverage")
                document["model"][key] = value
                self.assert_rejected(document, pattern)

    def test_model_dimensions_must_be_positive(self):
        for key in ("layers", "context_tokens", "sessions"):
            with self.subTest(field=key):
                document = clone("full-coverage")
                document["model"][key] = 0
                self.assert_rejected(document, f"model.{key}: expected positive u64")

    def test_required_sections_are_present(self):
        for key in ("model", "memory", "wall", "selection_contract"):
            with self.subTest(section=key):
                document = clone("full-coverage")
                del document[key]
                with self.assertRaises(budget.BudgetError):
                    budget.analyze(document)

    def test_selection_contract_requires_last_dimension_convention(self):
        document = clone("full-coverage")
        document["selection_contract"]["convention"] = "first_dimension"
        self.assert_rejected(document, "convention: expected last_dimension")

    def test_selection_contract_requires_positive_expert_axis(self):
        for field, value in (("expert_dimension", 0), ("expert_dimension", -1),
                             ("expert_dimension", "2"), ("expert_count", 0),
                             ("expert_count", 1_000_001)):
            with self.subTest(field=field, value=value):
                document = clone("full-coverage")
                document["selection_contract"][field] = value
                self.assert_rejected(
                    document, f"selection_contract.{field}: expected positive integer")

    def test_selection_contract_requires_every_role_shape(self):
        document = clone("full-coverage")
        del document["selection_contract"]["roles"]["ffn_down_exps"]
        self.assert_rejected(document, r"roles\.ffn_down_exps: every declared "
                                       r"expert role needs a shape")

    def test_selection_contract_role_rank_must_match_dimension(self):
        document = clone("full-coverage")
        document["selection_contract"]["roles"]["ffn_gate_exps"] = [4, 3, 1]
        self.assert_rejected(document, r"roles\.ffn_gate_exps: expected 2 "
                                       r"dimensions before the expert axis")

    def test_selection_contract_role_extents_must_be_positive(self):
        document = clone("full-coverage")
        document["selection_contract"]["roles"]["ffn_up_exps"] = [5, 0]
        self.assert_rejected(document, r"roles\.ffn_up_exps\[1\]: expected "
                                       r"positive integer")

    def test_selection_contract_rejects_short_role_aliases(self):
        document = clone("full-coverage")
        document["selection_contract"]["roles"] = {"gate": [4, 3], "up": [5, 2],
                                                   "down": [3, 4]}
        self.assert_rejected(document, "selection_contract.roles: unknown fields")


class SelectionArithmeticTests(BudgetCase):
    def parse(self, selection, axis_size=8):
        return budget.selection_items(selection, "sel", axis_size)

    def test_range_selection_charges_contiguous_items(self):
        parsed = self.parse({"id": "s", "mode": "range", "dimension": 2,
                             "axis_size": 8, "start": 2, "count": 4})
        self.assertEqual(parsed["start"], 2)
        self.assertEqual(parsed["count"], 4)
        self.assertEqual(list(parsed["items"]), [2, 3, 4, 5])
        self.assertIsNone(parsed["indices"])

    def test_indexed_selection_charges_named_items(self):
        parsed = self.parse({"id": "s", "mode": "indices", "dimension": 2,
                             "axis_size": 8, "count": 3,
                             "indices": [1, 4, 7]})
        self.assertEqual(parsed["indices"], [1, 4, 7])
        self.assertEqual(parsed["items"], [1, 4, 7])
        self.assertIsNone(parsed["start"])

    def test_unknown_selection_carries_no_extent(self):
        parsed = self.parse({"id": "s", "mode": "unknown", "dimension": 2,
                             "axis_size": 8, "reason": "not mapped yet"})
        self.assertIsNone(parsed["count"])
        self.assertIsNone(parsed["items"])
        self.assertEqual(parsed["reason"], "not mapped yet")

    def test_unknown_selection_cannot_declare_extent(self):
        for key, value in (("start", 0), ("count", 1), ("indices", [0])):
            with self.subTest(field=key):
                selection = {"id": "s", "mode": "unknown", "dimension": 2,
                             "axis_size": 8, "reason": "mapping pending",
                             key: value}
                with self.assertRaises(budget.BudgetError) as caught:
                    self.parse(selection)
                self.assertIn(f"unknown selection cannot declare {key}",
                              str(caught.exception))

    def test_unknown_selection_requires_a_reason(self):
        selection = {"id": "s", "mode": "unknown", "dimension": 2, "axis_size": 8}
        with self.assertRaises(budget.BudgetError) as caught:
            self.parse(selection)
        self.assertIn("sel.reason: expected nonempty identifier",
                      str(caught.exception))

    def test_range_selection_rejects_axis_overflow(self):
        for start, count in ((6, 4), (8, 1), (0, 9)):
            with self.subTest(start=start, count=count):
                selection = {"id": "s", "mode": "range", "dimension": 2,
                             "axis_size": 8, "start": start, "count": count}
                with self.assertRaises(budget.BudgetError) as caught:
                    self.parse(selection)
                self.assertIn("contiguous range exceeds the declared expert axis "
                              "of 8", str(caught.exception))

    def test_range_selection_rejects_index_list(self):
        selection = {"id": "s", "mode": "range", "dimension": 2, "axis_size": 8,
                     "start": 0, "count": 1, "indices": [0]}
        with self.assertRaises(budget.BudgetError) as caught:
            self.parse(selection)
        self.assertIn("contiguous selection cannot declare indices",
                      str(caught.exception))

    def test_range_selection_rejects_negative_start(self):
        selection = {"id": "s", "mode": "range", "dimension": 2, "axis_size": 8,
                     "start": -1, "count": 1}
        with self.assertRaises(budget.BudgetError) as caught:
            self.parse(selection)
        self.assertIn("sel.start: expected nonnegative integer",
                      str(caught.exception))

    def test_indexed_selection_requires_sorted_unique_items(self):
        cases = {
            "duplicate": ([1, 1], "strictly increasing"),
            "descending": ([3, 2], "strictly increasing"),
            "outside": ([1, 8], "outside the declared expert axis of 8"),
            "negative": ([0, -1], "expected nonnegative integer"),
            "string": ([0, "1"], "expected nonnegative integer"),
        }
        for name, (indices, pattern) in cases.items():
            with self.subTest(case=name):
                selection = {"id": "s", "mode": "indices", "dimension": 2,
                             "axis_size": 8, "count": len(indices),
                             "indices": indices}
                with self.assertRaises(budget.BudgetError) as caught:
                    self.parse(selection)
                self.assertIn(pattern, str(caught.exception))

    def test_indexed_selection_rejects_empty_or_oversized_lists(self):
        cases = {
            "empty": ({"count": 1, "indices": []}, "cannot be empty"),
            "oversized": ({"count": budget.MAX_SELECTION_ITEMS,
                           "indices": list(range(budget.MAX_SELECTION_ITEMS + 1))},
                          "exceeds the identifying bound"),
        }
        for name, (extent, pattern) in cases.items():
            with self.subTest(case=name):
                selection = {"id": "s", "mode": "indices", "dimension": 2,
                             "axis_size": budget.MAX_SELECTION_ITEMS + 1,
                             **extent}
                with self.assertRaises(budget.BudgetError) as caught:
                    self.parse(selection, budget.MAX_SELECTION_ITEMS + 1)
                self.assertIn(pattern, str(caught.exception))

    def test_indexed_selection_count_matches_indices(self):
        selection = {"id": "s", "mode": "indices", "dimension": 2, "axis_size": 8,
                     "count": 2, "indices": [1, 2, 3]}
        with self.assertRaises(budget.BudgetError) as caught:
            self.parse(selection)
        self.assertIn("sel.count: differs from the identifying indices",
                      str(caught.exception))

    def test_indexed_selection_rejects_range_start(self):
        selection = {"id": "s", "mode": "indices", "dimension": 2, "axis_size": 8,
                     "count": 1, "indices": [1], "start": 1}
        with self.assertRaises(budget.BudgetError) as caught:
            self.parse(selection)
        self.assertIn("indexed selection cannot declare start",
                      str(caught.exception))

    def test_selection_count_is_bounded(self):
        cases = {
            "zero": (0, "sel.count: expected positive integer bounded by"),
            "oversized": (budget.MAX_SELECTION_ITEMS + 1,
                          "sel.count: expected positive integer bounded by"),
            "boolean": (True, "sel.count: expected positive integer bounded by"),
        }
        for name, (count, pattern) in cases.items():
            with self.subTest(case=name):
                selection = {"id": "s", "mode": "range", "dimension": 2,
                             "axis_size": budget.MAX_SELECTION_ITEMS + 1,
                             "start": 0, "count": count}
                with self.assertRaises(budget.BudgetError) as caught:
                    self.parse(selection, budget.MAX_SELECTION_ITEMS + 1)
                self.assertIn(pattern, str(caught.exception))

    def test_selection_requires_identity_mode_and_axis(self):
        base = {"id": "s", "mode": "range", "dimension": 2, "axis_size": 8,
                "start": 0, "count": 1}
        for key in ("id", "mode", "dimension", "axis_size"):
            with self.subTest(field=key):
                selection = dict(base)
                del selection[key]
                with self.assertRaises(budget.BudgetError):
                    self.parse(selection)

    def test_selected_row_bytes_uses_rows_per_item(self):
        self.assertEqual(budget.selected_row_bytes([4, 3, 8], 2, 4, 4), 48)
        self.assertEqual(budget.selected_row_bytes([5, 2, 8], 2, 4, 5), 40)
        self.assertEqual(budget.selected_row_bytes([3, 4, 8], 2, 8, 3), 96)
        self.assertEqual(budget.selected_row_bytes([2560, 640, 512], 2, 128,
                                                   740), 60620800)
        self.assertEqual(budget.selected_row_bytes([2560, 640, 512], 2, 512,
                                                   740), 242483200)
        self.assertEqual(budget.selected_row_bytes([640, 2560, 512], 2, 512,
                                                   840), 1101004800)
        self.assertEqual(budget.selected_row_bytes([7, 8], 1, 5, 3), 15)

    def test_selected_row_bytes_rejects_arithmetic_overflow(self):
        with self.assertRaises(budget.BudgetError) as caught:
            budget.selected_row_bytes([2, budget.U64_MAX, 2], 2, 2, 2)
        self.assertIn("overflow", str(caught.exception))


class PackedMetadataTests(BudgetCase):
    def test_duplicate_tensor_records_are_rejected(self):
        document = clone("full-coverage")
        duplicate = copy.deepcopy(tensor(document, "hot-gate-layer0"))
        duplicate["id"] = "hot-gate-layer0"
        document["memory"]["tensors"].append(duplicate)
        self.assert_rejected(document, "duplicate tensor id")

    def test_selection_ids_are_unique(self):
        document = clone("full-coverage")
        tensor(document, "cold-gate-layer0")["selection"]["id"] = "hot-gate-range"
        self.assert_rejected(document, "selection.id: duplicate selection id")

    def test_packed_rank_must_keep_the_expert_axis(self):
        document = clone("full-coverage")
        tensor(document, "hot-gate-layer0")["packed"]["shape"] = [4, 3, 8, 1]
        self.assert_rejected(document, "packed.shape: expected rank 3")

    def test_packed_expert_axis_must_match_the_contract(self):
        document = clone("full-coverage")
        tensor(document, "hot-gate-layer0")["packed"]["shape"] = [4, 3, 7]
        self.assert_rejected(document,
                             "does not match selection_contract.expert_count 8")

    def test_packed_prefix_must_match_the_role_shape(self):
        document = clone("full-coverage")
        tensor(document, "hot-up-layer0")["packed"]["shape"] = [5, 3, 8]
        self.assert_rejected(document, "prefix does not match")

    def test_packed_row_must_be_integral_blocks(self):
        document = clone("full-coverage")
        packed = tensor(document, "cold-down-layer0")["packed"]
        packed["shape"] = [3, 4, 8]
        packed["block_values"] = 5
        packed["block_bytes"] = 1
        packed["row_stride_bytes"] = 3
        packed["data_bytes"] = 96
        self.assert_rejected(document, "row shape is not an integral number of")

    def test_packed_row_stride_must_cover_blocks(self):
        document = clone("full-coverage")
        packed = tensor(document, "hot-gate-layer0")["packed"]
        packed["block_values"] = 2
        packed["row_stride_bytes"] = 1
        self.assert_rejected(document, "row stride omits packed blocks/scales")

    def test_packed_data_bytes_must_agree_with_rows(self):
        document = clone("full-coverage")
        tensor(document, "hot-gate-layer0")["packed"]["data_bytes"] = 100
        self.assert_rejected(document, "packed data bytes differ from "
                                       "row-stride/shape")

    def test_packed_external_scales_are_not_charged(self):
        document = clone("full-coverage")
        tensor(document, "hot-gate-layer0")["packed"]["external_scale_bytes"] = 4
        self.assert_rejected(document, "external_scale_bytes: embedded scales are")

    def test_selection_dimension_must_be_the_last_axis(self):
        document = clone("full-coverage")
        tensor(document, "hot-gate-layer0")["selection"]["dimension"] = 1
        self.assert_rejected(document, "selection.dimension: expected the declared "
                                       "expert axis 2")

    def test_selection_axis_must_match_the_packed_axis(self):
        document = clone("full-coverage")
        selection = tensor(document, "hot-gate-layer0")["selection"]
        selection["axis_size"] = 7
        selection["count"] = 4
        self.assert_rejected(document, "selection.axis_size: does not match packed")

    def test_selection_mode_is_bounded(self):
        document = clone("full-coverage")
        tensor(document, "hot-gate-layer0")["selection"]["mode"] = "sparse"
        self.assert_rejected(document, "selection.mode: expected range, indices "
                                       "or unknown")

    def test_packed_metadata_must_repeat_across_selections(self):
        document = clone("full-coverage")
        resident = tensor(document, "cold-gate-layer0")["resident"]
        resident["status"] = "converted"
        resident["evidence"] = {
            "kind": "synthetic",
            "detail": "this test declares a converted resident copy",
        }
        tensor(document, "cold-gate-layer0")["packed"]["dtype"] = "synthetic-Q2_0-block"
        self.assert_rejected(document, "conflicting packed dtype for source tensor")

    def test_one_source_tensor_cannot_change_layer_or_role(self):
        document = clone("full-coverage")
        tensor(document, "cold-gate-layer0")["layer"] = 1
        self.assert_rejected(document, "conflicting layer/role mapping")

    def test_source_provenance_is_checked_per_record(self):
        document = clone("full-coverage")
        tensor(document, "hot-gate-layer0")["source"]["sha256"] = "ABC"
        self.assert_rejected(document, "source.sha256: expected 64 hexadecimal")

    def test_source_kind_is_bounded(self):
        document = clone("full-coverage")
        tensor(document, "hot-gate-layer0")["source"]["kind"] = "file"
        self.assert_rejected(document, "source.kind: expected tensor_metadata or "
                                       "synthetic")

    def test_role_must_be_a_reviewed_expert_projection(self):
        document = clone("full-coverage")
        tensor(document, "hot-gate-layer0")["role"] = "ffn_gate_inp"
        self.assert_rejected(document, "role: expected one of")


class ResidentMetadataTests(BudgetCase):
    def test_native_resident_must_match_packed_dtype(self):
        document = clone("full-coverage")
        tensor(document, "hot-gate-layer0")["resident"]["dtype"] = "synthetic-Q2_0-block"
        self.assert_rejected(document, "native resident dtype/layout differs")

    def test_native_resident_must_match_packed_layout(self):
        document = clone("full-coverage")
        tensor(document, "hot-gate-layer0")["resident"]["layout"] = "row-major"
        self.assert_rejected(document, "native resident dtype/layout differs")

    def test_native_resident_must_equal_the_selected_bytes(self):
        document = clone("full-coverage")
        tensor(document, "hot-gate-layer0")["resident"]["data_bytes"] = 52
        self.assert_rejected(document, "resident.data_bytes: native sub-slice must")

    def test_native_resident_cannot_hold_external_scales(self):
        document = clone("full-coverage")
        tensor(document, "hot-gate-layer0")["resident"]["scale_bytes"] = 8
        self.assert_rejected(document, "resident.scale_bytes: native resident")

    def test_native_resident_cannot_retain_the_packed_source(self):
        document = clone("full-coverage")
        tensor(document, "hot-gate-layer0")["resident"]["source_retained"] = True
        self.assert_rejected(document, "native resident cannot retain the packed")

    def test_converted_resident_requires_evidence(self):
        document = clone("full-coverage")
        del tensor(document, "cold-down-layer0")["resident"]["evidence"]
        self.assert_rejected(document, "resident.evidence: converted resident")

    def test_converted_resident_requires_source_retained_flag(self):
        document = clone("full-coverage")
        del tensor(document, "cold-down-layer0")["resident"]["source_retained"]
        self.assert_rejected(document, "resident.source_retained: required for a")

    def test_resident_status_is_bounded(self):
        document = clone("full-coverage")
        tensor(document, "hot-gate-layer0")["resident"]["status"] = "pinned"
        self.assert_rejected(document, "resident.status: expected native, "
                                       "converted or unknown")

    def test_unknown_resident_is_never_charged(self):
        document = clone("unknown-selection")
        report = budget.analyze(document)
        unknown = {item["id"]: item for item in report["memory"]["tensors"]}
        self.assertIsNone(unknown["unknown-gate-layer0"]["resident_bytes"])
        self.assertIn("memory.tensors[0].resident.status",
                      report["unknown_inputs"])
        self.assertIn("memory.tensors[0].resident.source_retained",
                      report["unknown_inputs"])

    def test_converted_resident_cannot_be_negative_or_implicit(self):
        document = clone("full-coverage")
        tensor(document, "cold-down-layer0")["resident"]["overhead_bytes"] = -1
        self.assert_rejected(document, "resident.overhead_bytes: expected "
                                       "nonnegative u64")


class AllocationTests(BudgetCase):
    def test_weight_allocation_requires_resident_or_packed_source(self):
        document = clone("full-coverage")
        allocation(document, "hot-experts")["tensor_storage"] = None
        self.assert_rejected(document, "tensor_storage: expected resident or "
                                       "packed_source")

    def test_weight_allocation_must_match_tensor_storage(self):
        document = clone("full-coverage")
        allocation(document, "hot-experts")["bytes"] = 96
        self.assert_rejected(document, "weight allocation differs from selected "
                                       "tensor storage bytes")

    def test_packed_source_allocation_requires_a_retained_conversion(self):
        document = clone("full-coverage")
        cold = allocation(document, "cold-experts")
        cold["tensor_storage"] = "packed_source"
        cold["bytes"] = 272
        cold["placements"][0]["bytes"] = 272
        self.assert_rejected(document, "packed_source copy requires converted "
                                       "tensors with source_retained=true")

    def test_retained_source_copy_needs_its_own_allocation(self):
        document = clone("full-coverage")
        tensor(document, "cold-down-layer0")["resident"]["source_retained"] = True
        self.assert_rejected(document, "retained packed sources need separately "
                                       "placed and charged allocations")

    def test_conversions_need_explicit_workspace(self):
        document = clone("full-coverage")
        document["memory"]["allocations"] = [
            item for item in document["memory"]["allocations"]
            if item["id"] != "conversion-workspace"]
        self.assert_rejected(document, "conversions need explicit "
                                       "conversion_workspace")

    def test_non_weight_allocations_cannot_reference_tensors(self):
        document = clone("full-coverage")
        allocation(document, "conversion-workspace")["tensor_storage"] = "resident"
        self.assert_rejected(document, "tensor references/storage are reserved "
                                       "for weight allocations")

    def test_down_projection_cannot_be_charged_as_a_hot_expert(self):
        document = clone("full-coverage")
        hot = allocation(document, "hot-experts")
        hot["tensor_ids"] = ["cold-down-layer0"]
        self.assert_rejected(document, "ffn_down_exps selections must be charged "
                                       "to cold_experts")

    def test_undeclared_tensor_reference_is_rejected(self):
        document = clone("full-coverage")
        allocation(document, "hot-experts")["tensor_ids"].append("missing-tensor")
        self.assert_rejected(document, "undeclared tensor reference")

    def test_duplicate_tensor_reference_is_rejected(self):
        document = clone("full-coverage")
        hot = allocation(document, "hot-experts")
        hot["tensor_ids"] = ["hot-gate-layer0", "hot-gate-layer0"]
        self.assert_rejected(document, "invalid or duplicate tensor references")

    def test_every_resident_tensor_needs_a_weight_allocation(self):
        document = clone("full-coverage")
        del document["memory"]["tensors"][0]
        self.assert_rejected(document, "undeclared tensor reference")

    def test_unallocated_tensor_metadata_is_rejected(self):
        document = clone("full-coverage")
        orphan = copy.deepcopy(tensor(document, "cold-gate-layer0"))
        orphan["id"] = "orphan-gate-layer0"
        orphan["selection"]["id"] = "orphan-range"
        orphan["source"]["artifact"] = "orphan.synthetic"
        document["memory"]["tensors"].append(orphan)
        self.assert_rejected(document, "unused metadata")

    def test_allocations_cannot_be_unassigned(self):
        document = clone("full-coverage")
        allocation(document, "hot-experts")["placements"] = []
        self.assert_rejected(document, "allocations cannot be unassigned")

    def test_placement_resources_must_be_declared_and_unique(self):
        document = clone("full-coverage")
        allocation(document, "hot-experts")["placements"][0]["resource"] = "ddr9"
        self.assert_rejected(document, "unknown or duplicate resource placement")

    def test_placement_bytes_must_sum_to_the_allocation(self):
        document = clone("full-coverage")
        allocation(document, "hot-experts")["placements"][0]["bytes"] = 80
        self.assert_rejected(document, "explicit placement byte counts do not "
                                       "match allocation")

    def test_placement_alignment_must_be_a_power_of_two(self):
        document = clone("full-coverage")
        allocation(document, "hot-experts")["placements"][0]["alignment_bytes"] = 3
        self.assert_rejected(document, "alignment_bytes: must be a power of two")

    def test_one_allocation_cannot_span_device_and_host(self):
        document = clone("full-coverage")
        hot = allocation(document, "hot-experts")
        hot["placements"] = [
            {"resource": "ddr0", "bytes": 88, "alignment_bytes": 64},
            {"resource": "node0", "bytes": 0, "alignment_bytes": 64},
        ]
        self.assert_rejected(document, "use separate allocations for host/device")

    def test_pinned_flag_is_reserved_for_host_allocations(self):
        document = clone("full-coverage")
        allocation(document, "hot-experts")["pinned"] = True
        self.assert_rejected(document, "pinned quotas apply to host allocations only")

    def test_host_pinned_allocations_are_charged_to_the_quota(self):
        document = clone("full-coverage")
        allocation(document, "conversion-workspace")["pinned"] = True
        report = budget.analyze(document)
        self.assertEqual(report["memory"]["status"], "over_capacity")
        self.assertEqual(report["memory"]["host_total"]["pinned"]["status"],
                         "over_capacity")
        uses = report["memory"]["host_total"]["pinned"]
        self.assertEqual(uses["known_used_lower_bound_bytes"], 4096)
        self.assertEqual(uses["used_bytes"], 4096)
        self.assertEqual(uses["headroom_bytes"], -4096)
        self.assertEqual(uses["headroom_upper_bound_bytes"], -4096)
        self.assertEqual(uses["capacity_bytes"], 0)

    def test_host_reserve_and_fragmentation_are_explicit(self):
        document = clone("full-coverage")
        document["memory"]["host"]["fragmentation_bytes"] = 500
        report = budget.analyze(document)
        self.assertEqual(report["memory"]["host_total"]["used_bytes"],
                         4416 + 64 + 500)

    def test_resources_must_be_explicit(self):
        cases = (
            ("device", "banks",
             "memory.ddr_bank: must explicitly list resources"),
            ("host", "numa_nodes",
             "memory.host_numa: must explicitly list resources"),
        )
        for section, key, pattern in cases:
            with self.subTest(section=section):
                document = clone("full-coverage")
                document["memory"][section][key] = []
                self.assert_rejected(document, pattern)

    def test_resource_capacity_must_be_positive(self):
        document = clone("full-coverage")
        document["memory"]["device"]["banks"][0]["capacity_bytes"] = 0
        self.assert_rejected(document, "capacity_bytes: expected positive u64")

    def test_resource_ids_are_unique(self):
        document = clone("full-coverage")
        document["memory"]["device"]["banks"].append(
            {"id": "ddr0", "capacity_bytes": 1024, "reserve_bytes": 0,
             "fragmentation_bytes": 0})
        self.assert_rejected(document, "duplicate resource id")

    def test_absent_coverage_cannot_hide_an_allocation(self):
        document = clone("full-coverage")
        document["memory"]["coverage"]["conversion_workspace"] = {
            "status": "absent", "reason": "no conversion workspace"}
        self.assert_rejected(document, "allocation in a declared absent category")

    def test_complete_coverage_needs_an_allocation(self):
        document = clone("full-coverage")
        document["memory"]["coverage"]["workspace"] = {
            "status": "complete", "reason": "declared workspace"}
        self.assert_rejected(document, "workspace: complete needs an allocation")

    def test_absent_coverage_needs_a_reason(self):
        document = clone("full-coverage")
        del document["memory"]["coverage"]["workspace"]["reason"]
        self.assert_rejected(document, "coverage.workspace.reason")

    def test_unknown_coverage_is_not_free(self):
        document = clone("full-coverage")
        document["memory"]["coverage"]["workspace"] = {
            "status": "unknown", "reason": "size not reviewed"}
        report = budget.analyze(document)
        self.assertEqual(report["memory"]["status"], "unknown")
        self.assertIs(report["complete"], False)
        self.assertIs(report["strict_pass"], False)
        self.assertIn("memory.coverage.workspace", report["unknown_inputs"])


class WallAccountingTests(BudgetCase):
    def test_accepted_window_must_have_positive_tokens(self):
        for value in (0, -1, None, "32", True):
            with self.subTest(value=value):
                document = clone("full-coverage")
                if value is None:
                    del document["wall"]["accepted_tokens"]
                else:
                    document["wall"]["accepted_tokens"] = value
                self.assert_rejected(document, "wall.accepted_tokens: expected a "
                                               "positive accepted-output token count")

    def test_only_accepted_output_tokens_can_be_counted(self):
        document = clone("full-coverage")
        document["wall"]["count_basis"] = "tokens"
        self.assert_rejected(document, "wall.count_basis: expected "
                                       "accepted_output_tokens")

    def test_wall_mode_is_bounded(self):
        document = clone("full-coverage")
        document["wall"]["mode"] = "prefill"
        self.assert_rejected(document, "wall.mode: expected mtp_off or mtp_on")

    def test_mtp_off_requires_an_absent_verify_commit(self):
        document = clone("full-coverage")
        document["wall"]["coverage"]["mtp_verify_commit"] = {
            "status": "complete", "reason": "declared"}
        self.assert_rejected(document, "MTP-off must explicitly declare absent")

    def test_mtp_on_requires_a_verification_stage(self):
        document = clone("full-coverage")
        document["wall"]["mode"] = "mtp_on"
        document["wall"]["coverage"]["mtp_verify_commit"] = {
            "status": "complete", "reason": "declared"}
        with self.assertRaises(budget.BudgetError) as caught:
            budget.analyze(document)
        self.assertIn("wall.coverage.mtp_verify_commit: complete needs a mapped "
                      "covering operation", str(caught.exception))

    def test_stage_layers_must_match_the_model(self):
        document = clone("full-coverage")
        document["model"]["layers"] = 2
        report = budget.analyze(document)
        self.assertEqual(report["wall"]["status"], "unknown")
        self.assertIn("wall.stages: layer repeats differ from model.layers",
                      report["unknown_inputs"])

    def test_non_expert_path_is_mandatory(self):
        document = clone("full-coverage")
        document["wall"]["stages"] = [
            item for item in document["wall"]["stages"]
            if item["scope"] != "non_expert"]
        document["wall"]["coverage"]["sampling"] = {
            "status": "absent", "reason": "no sampled head work declared"}
        report = budget.analyze(document)
        self.assertEqual(report["wall"]["status"], "unknown")
        self.assertIn("wall.stages: non-expert/head/sampling path required",
                      report["unknown_inputs"])

    def test_layer_stage_needs_expert_branches(self):
        document = clone("full-coverage")
        layer = stage(document, "decode-layer")
        layer["cpu"] = []
        layer["fpga"] = []
        layer["coverage"]["cold_expert"] = {
            "status": "absent", "reason": "declared absent"}
        layer["coverage"]["hot_expert"] = {
            "status": "absent", "reason": "declared absent"}
        with self.assertRaises(budget.BudgetError) as caught:
            budget.analyze(document)
        self.assertIn("full top10 cannot omit both cold and hot expert "
                      "computation", str(caught.exception))

    def test_overlap_claims_need_independence_evidence(self):
        document = clone("full-coverage")
        del stage(document, "decode-layer")["overlap"]["evidence"]
        report = budget.analyze(document)
        self.assertEqual(report["wall"]["status"], "unknown")
        self.assertIn("wall.stages[0].overlap: unverified independence",
                      report["unknown_inputs"])

    def test_transfer_operations_need_bytes_and_submissions(self):
        document = clone("full-coverage")
        h2d = operation(document, "decode-layer", "before", "layer-h2d")
        h2d["submissions"] = 0
        self.assert_rejected(document, "nonzero transfer requires submissions")

    def test_operations_need_a_known_kind(self):
        document = clone("full-coverage")
        operation(document, "decode-layer", "before", "layer-fixed")["kind"] = "gpu"
        self.assert_rejected(document, "kind: unsupported operation")

    def test_operation_timing_must_be_explicit_or_unknown(self):
        document = clone("full-coverage")
        fixed = operation(document, "decode-layer", "before", "layer-fixed")
        del fixed["milliseconds"]
        report = budget.analyze(document)
        self.assertEqual(report["wall"]["status"], "unknown")
        self.assertIn("wall.stages[0].before[0].milliseconds",
                      report["unknown_inputs"])
        self.assertEqual(report["wall"]["known_lower_bound_milliseconds"],
                         6.540014666666667)
        self.assertIsNone(
            report["wall"]["estimated_accepted_tokens_per_second"])

    def test_mtp_stage_is_rejected_in_mtp_off(self):
        document = clone("full-coverage")
        absent = {"status": "absent", "reason": "no MTP work in this test"}
        complete = {"status": "complete", "reason": "declared"}
        drafting = {
            "id": "decode-mtp",
            "scope": "mtp",
            "repeat": 1,
            "coverage": {
                "draft": complete,
                "verification": complete,
                "commit_rollback": complete,
                "state": complete,
                "transfers": absent,
                "numa_traffic": absent,
                "sync_merge": complete,
            },
            "overlap": {"mode": "serial"},
            "before": [],
            "cpu": [],
            "fpga": [],
            "after": [],
        }
        document["wall"]["stages"].append(drafting)
        self.assert_rejected(document, "MTP stage in MTP-off experiment")

    def test_missing_stages_keep_the_wall_unknown(self):
        document = clone("full-coverage")
        document["wall"]["stages"] = []
        for category in ("non_expert", "transfers", "sync_merge", "sampling"):
            document["wall"]["coverage"][category] = {
                "status": "absent", "reason": "no stage declared by this test"}
        report = budget.analyze(document)
        self.assertEqual(report["wall"]["status"], "unknown")
        self.assertIsNone(report["wall"]["milliseconds"])
        self.assertIsNone(report["wall"]["estimated_accepted_tokens_per_second"])
        self.assertEqual(report["unknown_inputs"], [
            "wall.stages: complete decode path required",
            "wall.stages: layer repeats differ from model.layers",
            "wall.stages: non-expert/head/sampling path required",
        ])


class CliTests(BudgetCase):
    def accepted_path(self, name):
        return FIXTURES / f"resource-budget-v2-{name}.json"

    def test_accepted_fixtures_exit_zero_and_emit_json(self):
        for name in ACCEPTED:
            with self.subTest(fixture=name):
                result = run_cli(self.accepted_path(name))
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertIs(report["analytical_only"], True)
                self.assertNotIn("NaN", result.stdout)

    def test_strict_exit_codes_follow_strict_pass(self):
        strict = {
            "partial-hot-subset": 1,
            "full-coverage": 0,
            "unknown-selection": 1,
            "real-shape-split": 0,
        }
        for name, returncode in strict.items():
            with self.subTest(fixture=name):
                result = run_cli(self.accepted_path(name), "--strict")
                self.assertEqual(result.returncode, returncode, result.stderr)

    def test_rejected_fixtures_exit_two_without_a_report(self):
        for name in REJECTED:
            with self.subTest(fixture=name):
                result = run_cli(self.accepted_path(name))
                self.assertEqual(result.returncode, 2)
                self.assertIn("resource budget v2 input error:",
                              result.stderr)
                self.assertEqual(result.stdout, "")

    def test_invalid_json_is_an_input_error(self):
        cases = {
            "truncated": '{"schema_version": 2',
            "duplicate-key": '{"a": 1, "a": 2}',
            "nonfinite": '{"schema_version": NaN}',
            "array": "[]",
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, text in cases.items():
                with self.subTest(case=name):
                    path = Path(directory) / f"{name}.json"
                    path.write_text(text, encoding="utf-8")
                    result = run_cli(path)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("resource budget v2 input error:",
                                  result.stderr)

    def test_missing_input_is_an_input_error(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_cli(Path(directory) / "missing.json")
        self.assertEqual(result.returncode, 2)
        self.assertIn("resource budget v2 input error:", result.stderr)

    def test_one_input_argument_is_required(self):
        result = run_cli()
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()