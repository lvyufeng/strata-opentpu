# Copyright (c) 2026 lvyufeng and contributors
# SPDX-License-Identifier: Apache-2.0
"""Source-free receipt checks: no real GGUF weights, Strata imports or devices.

The synthetic receipt's all-zero/all-one header hashes are deliberately inert
placeholders. Every arithmetic expectation is stated here independently; tests
never import a type table or parser from a development checkout.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from scripts import check_gguf_inventory as checker

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/gguf-header-inventory.synthetic.json"
SCHEMA = ROOT / "configs/contracts/gguf-header-inventory.schema.json"
FALSE_CLAIMS = ("input_provenance_verified", "artifact_integrity_verified", "model_completeness_verified",
                "backend_compatibility_verified", "device_ready", "inference_ready")
# Independently stated subset, not generated from the checker's implementation.
GEOMETRY = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34),
            17: ("IQ2_XS", 256, 74), 21: ("IQ3_S", 256, 110), 24: ("I8", 1, 1),
            25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8),
            28: ("F64", 1, 8), 30: ("BF16", 1, 2), 42: ("Q2_0", 64, 18)}


def fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def replace_path(document, path, value):
    parent = document
    for component in path[:-1]:
        parent = parent[component]
    parent[path[-1]] = value


def single_tensor(type_id=0, name="test.tensor", shape=None):
    document = fixture()
    shard = document["shards"][0]
    shard.update(file_size_bytes=4096, header_end_bytes=64, data_start_bytes=64, tensor_count=1)
    shard["split"] = {"index": 0, "count": 1, "total_tensors": 1}
    type_name, elements, block_bytes = GEOMETRY[type_id]
    shape = [elements * 2, 3] if shape is None else shape
    rows = 1
    for dimension in shape[1:]:
        rows *= dimension
    stride = shape[0] // elements * block_bytes
    tensor = {"name": name, "shard": "shard-0000", "shape": shape,
              "ggml_type_id": type_id, "ggml_type_name": type_name,
              "relative_offset_bytes": 0, "absolute_offset_bytes": 64,
              "storage": {"layout": "gguf-row-blocks", "block_elements": elements,
                          "block_bytes": block_bytes, "row_stride_bytes": stride, "data_bytes": stride * rows},
              "tensor_sha256": None}
    document["shards"] = [shard]
    document["tensors"] = [tensor]
    document["split"] = {"status": "complete", "declared_shards": 1, "declared_tensors": 1,
                         "present_indices": [0], "missing_indices": [], "diagnostics": []}
    document["totals"] = {"tensor_count": 1, "known_storage_bytes": stride * rows, "unknown_storage_tensors": 0}
    return document


def unknown_tensor(type_id=999, type_name="type999"):
    document = single_tensor()
    document["tensors"][0].update(ggml_type_id=type_id, ggml_type_name=type_name, shape=[7, 3])
    document["tensors"][0]["storage"] = {"layout": "unknown", "block_elements": None, "block_bytes": None,
                                         "row_stride_bytes": None, "data_bytes": None}
    document["totals"].update(known_storage_bytes=0, unknown_storage_tensors=1)
    return document


def missing_shard():
    document = fixture()
    document["shards"] = document["shards"][:1]
    document["tensors"] = document["tensors"][:3]
    document["totals"] = {"tensor_count": 3, "known_storage_bytes": 660, "unknown_storage_tensors": 1}
    document["split"] = {"status": "incomplete", "declared_shards": 2, "declared_tensors": 5,
                         "present_indices": [0], "missing_indices": [1], "diagnostics": ["split-shards-missing"]}
    return document


def no_split(document):
    for shard in document["shards"]:
        shard["split"] = {"index": None, "count": None, "total_tensors": None}
    document["split"] = {"status": "unknown", "declared_shards": None, "declared_tensors": None,
                         "present_indices": [], "missing_indices": [], "diagnostics": ["split-not-declared"]}
    return document


class InventorySemanticTests(unittest.TestCase):
    def assert_invalid(self, document):
        with self.assertRaises(checker.InventoryError) as caught:
            checker.validate(document)
        self.assertEqual(caught.exception.code, "invalid_receipt")
        return caught.exception

    def test_static_mixed_fixture_counts_and_raw_row_arithmetic(self):
        document = fixture()
        result = checker.validate(document)
        self.assertEqual([tensor["ggml_type_id"] for tensor in document["tensors"]], [17, 42, 999, 21, 8])
        self.assertEqual([tensor["storage"]["data_bytes"] for tensor in document["tensors"]],
                         [74 * 2 * 3, 2 * 18 * 2 * 3, None, 2 * 110 * 2, 3 * 34 * 2])
        self.assertEqual(result["totals"], {"tensor_count": 5, "known_storage_bytes": 1304, "unknown_storage_tensors": 1})
        self.assertEqual(result["split_status"], "complete")
        self.assertTrue(result["valid"])
        self.assertFalse(result["header_declarations_complete"])
        self.assertIn("unverified declarations", result["note"])
        self.assertIn("subtotal", result["note"])
        for key in FALSE_CLAIMS + ("source_identity_verified",):
            self.assertIs(result[key], False)

    def test_all_confirmed_scalar_and_packed_geometries(self):
        self.assertEqual(set(checker.CONFIRMED_TYPES), set(GEOMETRY))
        for type_id, (type_name, elements, block_bytes) in GEOMETRY.items():
            with self.subTest(type_id=type_id):
                document = single_tensor(type_id)
                result = checker.validate(document)
                self.assertEqual(document["tensors"][0]["ggml_type_name"], type_name)
                self.assertEqual(result["totals"]["known_storage_bytes"], 6 * block_bytes)
                self.assertTrue(result["header_declarations_complete"])

    def test_rank_one_and_max_rank_are_valid(self):
        for shape in ([256], [256, 1, 1, 1, 1, 1, 1, 1]):
            with self.subTest(shape=shape):
                self.assertEqual(checker.validate(single_tensor(17, shape=shape))["totals"]["known_storage_bytes"], 74)

    def test_raw_shape_zero_dimension_rank_bool_and_divisibility_reject(self):
        for shape in ([], [256] * 9, [0, 2], [-1, 2], [True, 2], [256.0, 2], ["256", 2], [128, 2]):
            with self.subTest(shape=shape):
                document = single_tensor(17)
                document["tensors"][0]["shape"] = shape
                self.assert_invalid(document)
        document = single_tensor(17)
        document["tensors"][0]["shape"] = [128, 2]  # Total is divisible, raw fastest row is not.
        self.assert_invalid(document)

    def test_geometry_name_and_null_aliases_reject(self):
        for path, value in ((("tensors", 0, "ggml_type_name"), "IQ3_S"),
                            (("tensors", 0, "storage", "layout"), "unknown"),
                            (("tensors", 0, "storage", "block_elements"), 128),
                            (("tensors", 0, "storage", "block_bytes"), 110),
                            (("tensors", 0, "storage", "row_stride_bytes"), 75),
                            (("tensors", 0, "storage", "data_bytes"), 148),
                            (("tensors", 0, "storage", "data_bytes"), None)):
            with self.subTest(path=path, value=value):
                document = single_tensor(17, shape=[256])
                replace_path(document, path, value)
                self.assert_invalid(document)
        document = single_tensor(24, shape=[1])
        document["tensors"][0]["storage"]["block_elements"] = True
        self.assert_invalid(document)
        document["tensors"][0]["storage"]["block_elements"] = 1.0
        self.assert_invalid(document)

    def test_unknown_nvf_and_unconfirmed_ids_stay_null_even_known_name(self):
        for type_id, type_name in ((999, "Q8_0"), (40, "NVFP4"), (41, "Q4_0_8_8"), (2, "Q4_0"), (4294967295, "type4294967295")):
            with self.subTest(type_id=type_id):
                document = unknown_tensor(type_id, type_name)
                result = checker.validate(document)
                self.assertEqual(result["totals"]["unknown_storage_tensors"], 1)
                self.assertFalse(result["header_declarations_complete"])
        for key in ("block_elements", "block_bytes", "row_stride_bytes", "data_bytes"):
            for value in (0, False, "unknown", 1):
                with self.subTest(key=key, value=value):
                    document = unknown_tensor()
                    document["tensors"][0]["storage"][key] = value
                    self.assert_invalid(document)
        document = unknown_tensor()
        document["tensors"][0]["storage"] = {"layout": "gguf-row-blocks", "block_elements": 1,
                                             "block_bytes": 1, "row_stride_bytes": 7, "data_bytes": 21}
        self.assert_invalid(document)

    def test_declared_input_kind_and_hashes_do_not_authenticate(self):
        for input_kind in ("synthetic", "local-unverified"):
            document = single_tensor()
            document["input_kind"] = input_kind
            document["shards"][0]["header_sha256"] = "f" * 64
            result = checker.validate(document)
            self.assertEqual(result["input_kind"], input_kind)
            self.assertTrue(result["header_declarations_complete"])
            for key in FALSE_CLAIMS + ("source_identity_verified",):
                self.assertIs(result[key], False)

    def test_fixed_producer_limits_and_claims_exact(self):
        for key in checker.PRODUCER:
            document = fixture()
            document["producer"][key] += "x"
            self.assert_invalid(document)
        for key in checker.LIMITS:
            document = fixture()
            document["limits"][key] += 1
            self.assert_invalid(document)
        for key in checker.CLAIMS:
            for value in (not checker.CLAIMS[key], int(checker.CLAIMS[key]), None):
                document = fixture()
                document["claims"][key] = value
                self.assert_invalid(document)
        for value in (True, 1.0, "1", None, 2):
            document = fixture()
            document["schema_version"] = value
            self.assert_invalid(document)

    def test_unknown_and_missing_fields_rejected_at_every_object(self):
        for path in ((), ("producer",), ("limits",), ("claims",), ("totals",), ("split",),
                     ("shards", 0), ("shards", 0, "split"), ("tensors", 0), ("tensors", 0, "storage")):
            for missing in (False, True):
                with self.subTest(path=path, missing=missing):
                    document = fixture()
                    target = document
                    for component in path:
                        target = target[component]
                    if missing:
                        del target[next(iter(target))]
                    else:
                        target["artifact_path"] = "never-open-this-model.gguf"
                    self.assert_invalid(document)

    def test_objects_and_sequences_have_strict_json_types(self):
        for path in (("shards",), ("tensors",), ("producer",), ("limits",), ("claims",), ("split",),
                     ("totals",), ("tensors", 0, "storage"), ("shards", 0, "split")):
            for value in (None, "object", 1, False):
                document = fixture()
                replace_path(document, path, value)
                self.assert_invalid(document)
        for document in (None, [], "receipt", True, {1: "not a string key"}):
            self.assert_invalid(document)

    def test_no_artifact_or_source_or_schema_access_during_validation(self):
        document = fixture()
        # Allowed strings may look like paths, commands or URLs. They remain data.
        document["tensors"][2]["name"] = "https://invalid.example/run?x=1;$(exec)"
        document["tensors"][2]["ggml_type_name"] = "/private/no-model.gguf"
        with patch("builtins.open", side_effect=AssertionError("unexpected file read")), \
             patch.object(checker.os, "open", side_effect=AssertionError("unexpected os.open")), \
             patch.object(Path, "open", side_effect=AssertionError("unexpected Path.open")):
            self.assertTrue(checker.validate(document)["valid"])

    def test_safe_unicode_names_and_global_duplicate_names(self):
        document = single_tensor(name="權重.Δ")
        self.assertTrue(checker.validate(document)["valid"])
        for name in ("", "a\n", "a\x7f", "a\x85", "a\u200d", "a\ud800", "a\udfff"):
            with self.subTest(name=repr(name)):
                document = single_tensor(name=name)
                self.assert_invalid(document)
        document = fixture()
        document["tensors"][3]["name"] = document["tensors"][0]["name"]
        self.assert_invalid(document)  # Duplicate even across shards.

    def test_type_id_and_safe_type_name_bounds(self):
        for value in (-1, 4294967296, True, 17.0, "17", None):
            document = fixture()
            document["tensors"][0]["ggml_type_id"] = value
            self.assert_invalid(document)
        for value in (None, "", "\u200e", "x\x00", "x\ud800"):
            document = unknown_tensor()
            document["tensors"][0]["ggml_type_name"] = value
            self.assert_invalid(document)

    def test_string_caps_are_encoded_bytes_with_total_name_bound(self):
        document = single_tensor(name="x" * 1048576)
        self.assertTrue(checker.validate(document)["valid"])
        document["tensors"][0]["name"] = "é" * 524289  # <=1 Mi characters, >1 Mi encoded bytes.
        self.assert_invalid(document)
        document = fixture()
        with patch.dict(checker.LIMITS, max_string_bytes_total=10):
            document["limits"]["max_string_bytes_total"] = 10
            self.assert_invalid(document)
        document = unknown_tensor(type_name="x" * 1048577)
        self.assert_invalid(document)

    def test_labels_names_and_tensor_order_are_canonical(self):
        document = fixture()
        document["shards"].reverse()
        self.assert_invalid(document)
        for label in ("shard-0001", "shard-0", "private.gguf", None):
            document = fixture()
            document["shards"][0]["label"] = label
            self.assert_invalid(document)
        document = fixture()
        document["tensors"][0], document["tensors"][1] = document["tensors"][1], document["tensors"][0]
        self.assert_invalid(document)
        document = fixture()
        document["tensors"][0]["shard"] = "shard-0002"
        self.assert_invalid(document)

    def test_file_headers_and_alignment_caps(self):
        for path, value in ((("shards", 0, "file_size_bytes"), 23),
                            (("shards", 0, "file_size_bytes"), 399),
                            (("shards", 0, "file_size_bytes"), 9223372036854775808),
                            (("shards", 0, "header_end_bytes"), 0),
                            (("shards", 0, "header_end_bytes"), 16777217),
                            (("shards", 0, "data_start_bytes"), 400),
                            (("shards", 0, "alignment_bytes"), 7),
                            (("shards", 0, "alignment_bytes"), 24),
                            (("shards", 0, "alignment_bytes"), 2097152),
                            (("shards", 0, "gguf_version"), 2),
                            (("shards", 0, "gguf_version"), 3.0)):
            document = fixture()
            replace_path(document, path, value)
            self.assert_invalid(document)
        document = fixture()
        with patch.dict(checker.LIMITS, max_header_bytes_total=500):
            document["limits"]["max_header_bytes_total"] = 500
            self.assert_invalid(document)
        for value in ("A" * 64, "1" * 63, "g" * 64, "1" * 64 + "\n", None):
            document = fixture()
            document["shards"][0]["header_sha256"] = value
            self.assert_invalid(document)

    def test_every_permitted_alignment_with_consistent_offsets(self):
        for alignment in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
                          16384, 32768, 65536, 131072, 262144, 524288, 1048576):
            document = single_tensor()
            start = (64 + alignment - 1) // alignment * alignment
            document["shards"][0].update(alignment_bytes=alignment, data_start_bytes=start, file_size_bytes=start + 128)
            document["tensors"][0]["absolute_offset_bytes"] = start
            self.assertTrue(checker.validate(document)["valid"])

    def test_counts_totals_and_tensor_limit_are_not_advisory(self):
        for path, value in ((("shards", 0, "tensor_count"), 4),
                            (("shards", 0, "tensor_count"), True),
                            (("totals", "tensor_count"), 6),
                            (("totals", "known_storage_bytes"), 1305),
                            (("totals", "unknown_storage_tensors"), 0),
                            (("totals", "unknown_storage_tensors"), True)):
            document = fixture()
            replace_path(document, path, value)
            self.assert_invalid(document)
        document = fixture()
        with patch.dict(checker.LIMITS, max_tensors_total=4):
            document["limits"]["max_tensors_total"] = 4
            self.assert_invalid(document)
        document = fixture()
        document["shards"] = []
        self.assert_invalid(document)
        document = fixture()
        document["shards"] = document["shards"] * 33
        self.assert_invalid(document)

    def test_metadata_only_shard_allows_data_start_beyond_eof(self):
        document = single_tensor()
        document["shards"][0].update(file_size_bytes=24, header_end_bytes=24, data_start_bytes=32, tensor_count=0)
        document["shards"][0]["split"]["total_tensors"] = 0
        document["tensors"] = []
        document["totals"] = {"tensor_count": 0, "known_storage_bytes": 0, "unknown_storage_tensors": 0}
        document["split"]["declared_tensors"] = 0
        self.assertTrue(checker.validate(document)["header_declarations_complete"])
        document["shards"][0]["tensor_count"] = 1
        self.assert_invalid(document)

    def test_zero_tensor_first_shard_in_two_shard_split(self):
        document = fixture()
        document["shards"][0].update(file_size_bytes=24, header_end_bytes=24, data_start_bytes=32, tensor_count=0)
        document["tensors"] = document["tensors"][3:]
        for shard in document["shards"]:
            shard["split"]["total_tensors"] = 2
        document["split"]["declared_tensors"] = 2
        document["totals"] = {"tensor_count": 2, "known_storage_bytes": 644, "unknown_storage_tensors": 0}
        self.assertTrue(checker.validate(document)["header_declarations_complete"])

    def test_offsets_data_region_and_unknown_eof_reject(self):
        for relative, absolute in ((1, 65), (32, 64), (-32, 32), (0, 0), (4096, 4160)):
            document = unknown_tensor()
            document["tensors"][0].update(relative_offset_bytes=relative, absolute_offset_bytes=absolute)
            self.assert_invalid(document)
        for document in (unknown_tensor(), single_tensor()):
            document["tensors"][0].update(relative_offset_bytes=4032, absolute_offset_bytes=4096)
            self.assert_invalid(document)  # EOF is not a permitted data start, even unknown.
        document = single_tensor()
        document["shards"][0]["file_size_bytes"] = 87  # F32 [2,3] requires 24 bytes [64,88).
        self.assert_invalid(document)
        document["shards"][0]["file_size_bytes"] = 88
        self.assertTrue(checker.validate(document)["valid"])

    def test_integer_product_row_bytes_offset_and_subtotal_overflows(self):
        document = unknown_tensor()
        document["tensors"][0]["shape"] = [9223372036854775807, 2]
        self.assert_invalid(document)
        document["tensors"][0]["shape"] = [9223372036854775808]
        self.assert_invalid(document)
        document = single_tensor(0)
        document["tensors"][0]["shape"] = [2305843009213693952]
        self.assert_invalid(document)  # Shape product fits, 4-byte row does not.
        document = unknown_tensor()
        document["tensors"][0].update(relative_offset_bytes=9223372036854775776,
                                     absolute_offset_bytes=9223372036854775807)
        self.assert_invalid(document)
        document = single_tensor(24)
        tensor = document["tensors"][0]
        tensor.update(shape=[9223372036854775807])
        tensor["storage"].update(row_stride_bytes=9223372036854775807, data_bytes=9223372036854775807)
        document["shards"][0]["file_size_bytes"] = 9223372036854775807
        document["totals"]["known_storage_bytes"] = 9223372036854775807
        self.assert_invalid(document)  # Start + known bytes overflows even though bytes fit.
        # Independent shards have valid individual extents but overflowing subtotal.
        document = single_tensor(24)
        shard = document["shards"][0]
        tensor = document["tensors"][0]
        shard.update(file_size_bytes=9223372036854775807)
        shard["split"].update(count=2, total_tensors=2)
        tensor["shape"] = [4611686018427387904]
        tensor["storage"].update(row_stride_bytes=4611686018427387904, data_bytes=4611686018427387904)
        second_shard, second_tensor = deepcopy(shard), deepcopy(tensor)
        second_shard["label"] = "shard-0001"
        second_shard["split"]["index"] = 1
        second_tensor.update(shard="shard-0001", name="z.tensor")
        document["shards"].append(second_shard)
        document["tensors"].append(second_tensor)
        document["split"].update(declared_shards=2, declared_tensors=2, present_indices=[0, 1])
        document["totals"].update(tensor_count=2, known_storage_bytes=9223372036854775808)
        self.assert_invalid(document)

    def test_known_intervals_sort_by_offsets_not_tensor_names(self):
        document = fixture()
        document["tensors"][0].update(relative_offset_bytes=224, absolute_offset_bytes=640)
        document["tensors"][1].update(relative_offset_bytes=0, absolute_offset_bytes=416)
        self.assertTrue(checker.validate(document)["valid"])
        document = fixture()
        document["tensors"][1].update(relative_offset_bytes=416, absolute_offset_bytes=832)
        self.assert_invalid(document)

    def test_unknown_inside_known_rejects_but_boundary_does_not_assert_size(self):
        document = fixture()
        document["tensors"][2].update(relative_offset_bytes=32, absolute_offset_bytes=448)
        self.assert_invalid(document)
        document["tensors"][2].update(relative_offset_bytes=0, absolute_offset_bytes=416)
        self.assertTrue(checker.validate(document)["valid"])  # Unknown remains size-null.
        # Known end and unknown start exactly adjacent and aligned.
        document = fixture()
        document["tensors"][0]["ggml_type_id"] = 1
        document["tensors"][0]["ggml_type_name"] = "F16"
        document["tensors"][0]["shape"] = [16, 1]
        document["tensors"][0]["storage"] = {"layout": "gguf-row-blocks", "block_elements": 1, "block_bytes": 2,
                                             "row_stride_bytes": 32, "data_bytes": 32}
        document["tensors"][2].update(relative_offset_bytes=32, absolute_offset_bytes=448)
        document["totals"]["known_storage_bytes"] = 892
        self.assertTrue(checker.validate(document)["valid"])

    def test_all_null_split_is_unknown_never_filename_inferred(self):
        document = no_split(fixture())
        result = checker.validate(document)
        self.assertEqual(result["split_status"], "unknown")
        self.assertFalse(result["header_declarations_complete"])
        self.assertEqual(result["split"]["diagnostics"], ["split-not-declared"])

    def test_known_missing_shard_and_partial_declarations(self):
        document = missing_shard()
        result = checker.validate(document)
        self.assertEqual(result["split_status"], "incomplete")
        self.assertEqual(result["split"]["missing_indices"], [1])
        document = fixture()
        document["shards"][1]["split"]["total_tensors"] = None
        document["split"].update(status="incomplete", diagnostics=["split-declarations-partial"])
        self.assertEqual(checker.validate(document)["split_status"], "incomplete")
        document["shards"][1]["split"]["index"] = None
        document["split"].update(present_indices=[0], missing_indices=[1],
                                 diagnostics=["split-declarations-partial", "split-shards-missing"])
        self.assertEqual(checker.validate(document)["split"]["diagnostics"],
                         ["split-declarations-partial", "split-shards-missing"])
        document["split"]["diagnostics"].reverse()
        self.assert_invalid(document)

    def test_partial_only_indices_or_only_total_need_no_guessed_count(self):
        for field, value, indices, total in (("index", 0, [0], None), ("total_tensors", 1, [], 1)):
            document = no_split(single_tensor())
            document["shards"][0]["split"][field] = value
            document["split"].update(status="incomplete", declared_tensors=total, present_indices=indices,
                                     diagnostics=["split-declarations-partial"])
            self.assertEqual(checker.validate(document)["split_status"], "incomplete")

    def test_split_indices_need_not_follow_input_basename_order(self):
        document = fixture()
        document["shards"][0]["split"]["index"] = 1
        document["shards"][1]["split"]["index"] = 0
        self.assertEqual(checker.validate(document)["split_status"], "complete")

    def test_split_scalar_bounds_strict_types_and_disagreement(self):
        for key, values in (("index", (-1, 64, True, 0.0, "0")),
                            ("count", (0, 65, True, 2.0, "2")),
                            ("total_tensors", (-1, 100001, True, 5.0, "5"))):
            for value in values:
                document = fixture()
                document["shards"][0]["split"][key] = value
                self.assert_invalid(document)
        for key, value in (("index", 0), ("count", 3), ("total_tensors", 6)):
            document = fixture()
            document["shards"][1]["split"][key] = value
            self.assert_invalid(document)
        document = fixture()
        document["shards"][0]["split"]["count"] = 1
        self.assert_invalid(document)

    def test_actual_total_lower_complete_invalid_but_missing_may_be_lower(self):
        document = fixture()
        for shard in document["shards"]:
            shard["split"]["total_tensors"] = 6
        document["split"]["declared_tensors"] = 6
        self.assert_invalid(document)
        document = missing_shard()
        document["shards"][0]["split"]["total_tensors"] = 6
        document["split"]["declared_tensors"] = 6
        self.assertTrue(checker.validate(document)["valid"])
        document["shards"][0]["split"]["total_tensors"] = 2
        document["split"]["declared_tensors"] = 2
        self.assert_invalid(document)

    def test_split_summary_facts_sorted_indices_types_and_diagnostics_exact(self):
        for key, value in (("status", "unknown"), ("declared_shards", 3), ("declared_tensors", 5.0),
                           ("present_indices", [1, 0]), ("present_indices", [False, 1]),
                           ("missing_indices", [1]), ("diagnostics", ["split-not-declared"])):
            document = fixture()
            document["split"][key] = value
            self.assert_invalid(document)
        document = missing_shard()
        document["split"]["missing_indices"] = [True]
        self.assert_invalid(document)


class _ObservedRawReader:
    """Test-only read probe; owns exactly the real receipt descriptor it wraps."""

    def __init__(self, stream, reader):
        self.stream, self.reader = stream, reader
        self.descriptor = stream.fileno()
        self.requests = []

    def read(self, size):
        self.requests.append(size)
        return self.reader(self.stream, size)

    def fileno(self):
        return self.stream.fileno()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.stream.__exit__(*args)


class InventoryLoaderTests(unittest.TestCase):
    def assert_bad_json(self, data, code="invalid_json"):
        with self.assertRaises(checker.InventoryError) as caught:
            checker.loads_receipt(data)
        self.assertEqual(caught.exception.code, code)

    def test_valid_utf8_load_and_receipt_regular_file(self):
        document = checker.load_receipt(FIXTURE)
        self.assertEqual(checker.validate(document)["totals"]["known_storage_bytes"], 1304)
        self.assertEqual(checker.loads_receipt(b'{"unicode":"\xe6\xac\x8a"}'), {"unicode": "權"})

    def test_json_duplicate_keys_nested_constants_and_utf8_reject(self):
        for data in (b'{"a":1,"a":2}', b'{"a":{"b":1,"b":2}}', b'{"a":NaN}', b'{"a":Infinity}',
                     b'{"a":-Infinity}', b'{"a":1e999}', b'{"a":"\xff"}', b'{"a":"\\ud800"}',
                     b'{"\\ud800":1}', b'\xef\xbb\xbf{}', b'{"a":', b'{}{}', b'{"a":"bad\x00"}'):
            with self.subTest(data=data):
                self.assert_bad_json(data)

    def test_literal_length_and_depth_limits_apply_before_parse(self):
        self.assertEqual(checker.loads_receipt(b"1" * 64), int("1" * 64))
        for data in (b"1" * 65, b"0." + b"1" * 63, b"-" + b"1" * 64):
            self.assert_bad_json(data)
        checker.loads_receipt(b"[" * 32 + b"0" + b"]" * 32)
        self.assert_bad_json(b"[" * 33 + b"0" + b"]" * 33)
        self.assertEqual(checker.loads_receipt(b'{"x":"[ { \\" } ]"}'), {"x": '[ { " } ]'})

    def test_max_bytes_bounds_without_large_allocation(self):
        with patch.object(checker, "MAX_JSON_BYTES", 2):
            self.assertEqual(checker.loads_receipt(b"{}"), {})
            self.assert_bad_json(b"{} ", "receipt_too_large")
            with self.assertRaises(checker.InventoryError) as caught:
                checker.load_receipt(FIXTURE)
            self.assertEqual(caught.exception.code, "receipt_too_large")

    def test_float_integer_bool_rejected_by_semantics(self):
        for value in (True, 1.0, 1e0):
            document = fixture()
            document["schema_version"] = value
            decoded = checker.loads_receipt(json.dumps(document).encode("utf-8"))
            with self.assertRaises(checker.InventoryError):
                checker.validate(decoded)
        document = fixture()
        document["tensors"][0]["shape"][0] = 9223372036854775808
        with self.assertRaises(checker.InventoryError):
            checker.validate(checker.loads_receipt(json.dumps(document).encode("utf-8")))

    def test_regular_file_flags_unbuffered_reads_and_same_fd(self):
        real_open, real_fdopen, real_fstat = checker.os.open, checker.os.fdopen, checker.os.fstat
        with patch.object(checker.os, "open", wraps=real_open) as opened, \
             patch.object(checker.os, "fdopen", wraps=real_fdopen) as stream, \
             patch.object(checker.os, "fstat", wraps=real_fstat) as fstat_calls:
            checker.load_receipt(FIXTURE)
        self.assertEqual(opened.call_count, 1)
        flags = opened.call_args.args[1]
        self.assertTrue(flags & os.O_NOFOLLOW)
        self.assertTrue(flags & os.O_NONBLOCK)
        self.assertEqual(stream.call_args.kwargs, {"buffering": 0})
        self.assertEqual(len(fstat_calls.call_args_list), 2)
        self.assertEqual(fstat_calls.call_args_list[0].args, fstat_calls.call_args_list[1].args)

    def test_short_reads_retry_with_bounded_positive_requests_and_close(self):
        data = b'{"receipt":"short"}'
        probes = []
        real_fdopen, real_fstat = checker.os.fdopen, checker.os.fstat

        def observed(*args, **kwargs):
            probe = _ObservedRawReader(real_fdopen(*args, **kwargs),
                                       lambda stream, size: stream.read(min(size, 7)))
            probes.append(probe)
            return probe

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_bytes(data)
            with patch.object(checker, "MAX_JSON_BYTES", len(data)), \
                 patch.object(checker.os, "fdopen", side_effect=observed):
                self.assertEqual(checker.load_receipt(path), {"receipt": "short"})
        self.assertEqual(len(probes), 1)
        self.assertGreater(len(probes[0].requests), 2)
        self.assertEqual(probes[0].requests[0], len(data) + 1)
        self.assertEqual(probes[0].requests[-1], 1)
        self.assertTrue(all(0 < size <= len(data) + 1 for size in probes[0].requests))
        with self.assertRaises(OSError):
            real_fstat(probes[0].descriptor)

    def test_raw_read_failure_or_premature_eof_sanitizes_and_closes(self):
        real_fdopen, real_fstat = checker.os.fdopen, checker.os.fstat

        def failed_read(_stream, _size):
            raise OSError("private-model-path")

        for reader, code in ((failed_read, "receipt_access"),
                             (lambda _stream, _size: b"", "receipt_changed")):
            probes = []

            def observed(*args, **kwargs):
                probe = _ObservedRawReader(real_fdopen(*args, **kwargs), reader)
                probes.append(probe)
                return probe

            with patch.object(checker.os, "fdopen", side_effect=observed):
                with self.assertRaises(checker.InventoryError) as caught:
                    checker.load_receipt(FIXTURE)
            self.assertEqual(caught.exception.code, code)
            self.assertNotIn("private-model-path", str(caught.exception))
            self.assertEqual(len(probes), 1)
            with self.assertRaises(OSError):
                real_fstat(probes[0].descriptor)

    def test_raw_read_growth_is_bounded_to_cap_plus_one_and_closes(self):
        data = b"{}"
        probes = []
        real_fdopen, real_fstat = checker.os.fdopen, checker.os.fstat

        def observed(*args, **kwargs):
            probe = _ObservedRawReader(real_fdopen(*args, **kwargs),
                                       lambda _stream, size: b"x" * size)
            probes.append(probe)
            return probe

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_bytes(data)
            with patch.object(checker, "MAX_JSON_BYTES", len(data)), \
                 patch.object(checker.os, "fdopen", side_effect=observed):
                with self.assertRaises(checker.InventoryError) as caught:
                    checker.load_receipt(path)
        self.assertEqual(caught.exception.code, "receipt_too_large")
        self.assertEqual(probes[0].requests, [len(data) + 1])
        with self.assertRaises(OSError):
            real_fstat(probes[0].descriptor)

    def test_fdopen_failure_closes_untransferred_descriptor(self):
        real_open, real_fstat = checker.os.open, checker.os.fstat
        descriptors = []

        def observed(*args, **kwargs):
            descriptor = real_open(*args, **kwargs)
            descriptors.append(descriptor)
            return descriptor

        with patch.object(checker.os, "open", side_effect=observed), \
             patch.object(checker.os, "fdopen", side_effect=OSError("private-host-details")):
            with self.assertRaises(checker.InventoryError) as caught:
                checker.load_receipt(FIXTURE)
        self.assertEqual(caught.exception.code, "receipt_access")
        self.assertNotIn("private-host-details", str(caught.exception))
        self.assertEqual(len(descriptors), 1)
        with self.assertRaises(OSError):
            real_fstat(descriptors[0])

    def test_safe_flags_unavailable_reject_before_read(self):
        for flag in ("O_NOFOLLOW", "O_NONBLOCK"):
            with patch.object(checker.os, flag, 0), patch.object(checker.os, "lstat") as inspected:
                with self.assertRaises(checker.InventoryError) as caught:
                    checker.load_receipt(FIXTURE)
            self.assertEqual(caught.exception.code, "unsupported_file_safety")
            inspected.assert_not_called()

    def test_directory_symlink_fifo_device_and_missing_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            link = base / "symlink.json"
            link.symlink_to(FIXTURE)
            paths = [base, link, base / "missing.json"]
            if hasattr(os, "mkfifo"):
                fifo = base / "fifo.json"
                os.mkfifo(fifo)
                paths.append(fifo)
            if Path("/dev/null").exists():
                paths.append(Path("/dev/null"))
            for path in paths:
                with self.subTest(kind=path.name):
                    with self.assertRaises(checker.InventoryError):
                        checker.load_receipt(path)

    def test_read_only_receipt_and_path_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_bytes(FIXTURE.read_bytes())
            path.chmod(stat.S_IRUSR)
            self.assertTrue(checker.validate(checker.load_receipt(path))["valid"])
        with patch.object(checker, "MAX_PATH_BYTES", 8), patch.object(checker.os, "lstat") as inspected:
            with self.assertRaises(checker.InventoryError) as caught:
                checker.load_receipt("x" * 9)
            self.assertEqual(caught.exception.code, "unsafe_receipt")
            inspected.assert_not_called()

    def test_stat_change_before_or_after_read_closes_descriptor(self):
        real_fstat, real_open = checker.os.fstat, checker.os.open
        for change_call in (1, 2):
            seen = []
            def changed(descriptor):
                seen.append(descriptor)
                info = real_fstat(descriptor)
                if len(seen) != change_call:
                    return info
                class ChangedStat:
                    st_dev = info.st_dev
                    st_ino = info.st_ino
                    st_mode = info.st_mode
                    st_size = info.st_size
                    st_mtime_ns = info.st_mtime_ns + 1
                    st_ctime_ns = info.st_ctime_ns
                return ChangedStat()
            with patch.object(checker.os, "open", wraps=real_open) as opened, \
                 patch.object(checker.os, "fstat", side_effect=changed):
                with self.assertRaises(checker.InventoryError) as caught:
                    checker.load_receipt(FIXTURE)
            self.assertEqual(caught.exception.code, "receipt_changed")
            self.assertEqual(opened.call_count, 1)
            with self.assertRaises(OSError):
                real_fstat(seen[0])

    def test_malformed_json_rejects_and_access_errors_are_sanitized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private-host-secret.json"
            path.write_bytes(b'{"duplicate":0,"duplicate":1}')
            with self.assertRaises(checker.InventoryError) as caught:
                checker.load_receipt(path)
            self.assertNotIn(str(path), str(caught.exception))
        with patch.object(checker.os, "open", side_effect=OSError("private-host-address")):
            with self.assertRaises(checker.InventoryError) as caught:
                checker.load_receipt(FIXTURE)
        self.assertEqual(caught.exception.code, "receipt_access")
        self.assertNotIn("private-host-address", str(caught.exception))


class InventoryCliTests(unittest.TestCase):
    def execute(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = checker.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def receipt_cli(self, document, extra=()):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text(json.dumps(document, ensure_ascii=True, allow_nan=False), encoding="utf-8")
            return self.execute(["--receipt", str(path), *extra])

    def test_default_complete_unknown_layout_is_valid_zero_exit(self):
        code, out, error = self.execute([])
        report = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(error, "")
        self.assertEqual(report["split_status"], "complete")
        self.assertEqual(report["totals"]["unknown_storage_tensors"], 1)
        self.assertIs(report["strict_complete_requested"], False)
        self.assertIs(report["valid"], True)

    def test_strict_complete_unknown_layout_returns_three_with_valid_summary(self):
        code, out, error = self.execute(["--strict-complete"])
        self.assertEqual(code, 3)
        self.assertEqual(error, "")
        self.assertTrue(json.loads(out)["valid"])
        self.assertFalse(json.loads(out)["header_declarations_complete"])

    def test_strict_complete_all_known_passes_but_never_promotes_claims(self):
        code, out, error = self.receipt_cli(single_tensor(17), ["--strict-complete"])
        self.assertEqual(code, 0)
        self.assertEqual(error, "")
        report = json.loads(out)
        self.assertTrue(report["header_declarations_complete"])
        for key in FALSE_CLAIMS + ("source_identity_verified",):
            self.assertIs(report[key], False)

    def test_incomplete_split_three_even_without_strict(self):
        for extra in ((), ("--strict-complete",)):
            code, out, error = self.receipt_cli(missing_shard(), extra)
            self.assertEqual(code, 3)
            self.assertEqual(error, "")
            report = json.loads(out)
            self.assertTrue(report["valid"])
            self.assertEqual(report["split"]["diagnostics"], ["split-shards-missing"])

    def test_unknown_split_default_zero_strict_three(self):
        document = no_split(single_tensor())
        for extra, expected in (((), 0), (("--strict-complete",), 3)):
            code, out, error = self.receipt_cli(document, extra)
            self.assertEqual(code, expected)
            self.assertEqual(error, "")
            self.assertEqual(json.loads(out)["split_status"], "unknown")

    def test_malformed_receipt_two_empty_stdout(self):
        document = fixture()
        document["claims"]["inference_ready"] = True
        code, out, error = self.receipt_cli(document)
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("invalid_receipt", error)
        self.assertNotIn("receipt.json", error)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private-secret.json"
            path.write_bytes(b"NaN")
            code, out, error = self.execute(["--receipt", str(path)])
            self.assertEqual(code, 2)
            self.assertEqual(out, "")
            self.assertIn("invalid_json", error)
            self.assertNotIn("private-secret", error)

    def test_arguments_reject_unknown_modes_or_missing_path(self):
        for arguments in (["--hf", "secret"], ["--receipt"], ["--unknown-private-path"], ["unexpected.json"]):
            code, out, error = self.execute(arguments)
            self.assertEqual(code, 2)
            self.assertEqual(out, "")
            self.assertIn("invalid_arguments", error)
            self.assertNotIn("secret", error)
            self.assertNotIn("private-path", error)


class InventorySchemaTests(unittest.TestCase):
    def test_schema_is_strict_json_and_draft_2020_12(self):
        schema = checker.load_receipt(SCHEMA)
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(set(schema["required"]), set(fixture()))
        self.assertEqual(set(schema["properties"]), set(fixture()))

    def test_each_object_definition_closed_and_required_properties_exact(self):
        schema = checker.load_receipt(SCHEMA)
        definitions = schema["$defs"]
        objects = ("producer", "limits", "shardSplit", "shard", "unknownStorage", "rowStorage", "tensor", "splitSummary", "totals", "claims")
        for name in objects:
            with self.subTest(definition=name):
                definition = definitions[name]
                self.assertEqual(definition["type"], "object")
                self.assertIs(definition["additionalProperties"], False)
                self.assertEqual(set(definition["required"]), set(definition["properties"]))
        for key, value in fixture()["limits"].items():
            self.assertEqual(definitions["limits"]["properties"][key], {"type": "integer", "const": value})
        for key, value in fixture()["producer"].items():
            self.assertEqual(definitions["producer"]["properties"][key], {"type": "string", "const": value})
        for key, value in fixture()["claims"].items():
            self.assertEqual(definitions["claims"]["properties"][key], {"type": "boolean", "const": value})

    def test_schema_null_geometry_and_confirmed_ids_match_frozen_subset(self):
        definitions = checker.load_receipt(SCHEMA)["$defs"]
        for key in ("block_elements", "block_bytes", "row_stride_bytes", "data_bytes"):
            self.assertEqual(definitions["unknownStorage"]["properties"][key], {"type": "null"})
        self.assertEqual(definitions["shard"]["properties"]["artifact_sha256"], {"type": "null"})
        self.assertEqual(definitions["shard"]["properties"]["payload_read"], {"type": "boolean", "const": False})
        self.assertEqual(definitions["tensor"]["properties"]["tensor_sha256"], {"type": "null"})
        conditional = definitions["tensor"]["allOf"]
        self.assertEqual(len(conditional), 13)
        for clause in conditional[:-1]:
            type_id = clause["if"]["properties"]["ggml_type_id"]["const"]
            expected_name, elements, block_bytes = GEOMETRY[type_id]
            props = clause["then"]["properties"]
            self.assertEqual(props["ggml_type_name"], {"const": expected_name})
            self.assertEqual(props["storage"]["properties"]["block_elements"], {"const": elements})
            self.assertEqual(props["storage"]["properties"]["block_bytes"], {"const": block_bytes})
        self.assertEqual(set(conditional[-1]["if"]["properties"]["ggml_type_id"]["not"]["enum"]), set(GEOMETRY))
        self.assertEqual(definitions["uint63"]["maximum"], 9223372036854775807)
        self.assertEqual(definitions["tensor"]["properties"]["shape"]["maxItems"], 8)


if __name__ == "__main__":
    unittest.main()
