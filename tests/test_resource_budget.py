# Copyright (c) 2026 lvyufeng and contributors
# SPDX-License-Identifier: Apache-2.0
"""Synthetic arithmetic/adversarial cases, not model or device validation."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("resource_budget", ROOT / "scripts/resource_budget.py")
budget = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(budget)


def status(names, value="absent"):
    return {name: {"status": value, "reason": "Synthetic fixture explicitly accounts for this category."} for name in names}


def allocation(name="work", size=65, resource="ddr0", category="workspace", pinned=False, alignment=64):
    return {"id": name, "category": category, "bytes": size, "pinned": pinned, "tensor_ids": [],
            "placements": [{"resource": resource, "bytes": size, "alignment_bytes": alignment}]}


def operation(name, kind, covers, duration=1.0, deps=(), size=None, submissions=None):
    value = {"id": name, "kind": kind, "covers": list(covers), "depends_on": list(deps)}
    if kind in ("h2d", "d2h"):
        value.update(bytes=size, submissions=submissions)
    else:
        value["milliseconds"] = duration
    return value


def tensor(name="tiny", source_name=None):
    return {"id": name,
            "source": {"kind": "synthetic", "artifact": "tiny.synthetic", "revision": "fixture-v1",
                       "sha256": "0" * 64, "tensor": source_name or name},
            "packed": {"dtype": "synthetic-IQ2_XS-block", "layout": "row-blocked",
                       "shape": [256, 2], "block_values": 256, "block_bytes": 74,
                       "row_stride_bytes": 74, "data_bytes": 148, "external_scale_bytes": 0},
            "resident": {"status": "native", "dtype": "synthetic-IQ2_XS-block", "layout": "row-blocked",
                         "data_bytes": 148, "scale_bytes": 0, "overhead_bytes": 0, "source_retained": False}}


def fixture():
    """A complete but fictional two-layer window; no actual geometry/readiness claim."""
    memory_coverage = status(budget.CATEGORIES)
    memory_coverage["workspace"]["status"] = "complete"
    layer_coverage = status(budget.STAGE_CATEGORIES["layer"])
    for role in ("fixed", "shared_expert", "routing", "activation", "state", "cold_expert", "hot_expert", "transfers", "sync_merge"):
        layer_coverage[role]["status"] = "complete"
    outer_coverage = status(budget.STAGE_CATEGORIES["non_expert"])
    for role in ("embedding", "head", "sampling", "state"):
        outer_coverage[role]["status"] = "complete"
    global_coverage = status(budget.WALL_CATEGORIES)
    for role in ("non_expert", "transfers", "sync_merge", "sampling"):
        global_coverage[role]["status"] = "complete"
    return {
        "schema_version": 1, "id": "synthetic-fit", "input_kind": "synthetic",
        "model": {"layers": 2, "context_tokens": 4, "sessions": 1, "routed_top_k": 10,
                  "expert_skip": 0, "shared_expert": True},
        "memory": {
            "device": {"capacity_bytes": 2048, "banks": [
                {"id": "ddr0", "capacity_bytes": 1024, "reserve_bytes": 32, "fragmentation_bytes": 16},
                {"id": "ddr1", "capacity_bytes": 1024, "reserve_bytes": 0, "fragmentation_bytes": 0}]},
            "host": {"capacity_bytes": 4096, "reserve_bytes": 10, "fragmentation_bytes": 6,
                     "pinned_quota_bytes": 512, "numa_nodes": [
                         {"id": "numa0", "capacity_bytes": 2048, "reserve_bytes": 0, "fragmentation_bytes": 0, "pinned_quota_bytes": 256},
                         {"id": "numa1", "capacity_bytes": 2048, "reserve_bytes": 0, "fragmentation_bytes": 0, "pinned_quota_bytes": 256}]},
            "coverage": memory_coverage, "tensors": [], "allocations": [allocation()],
        },
        "wall": {
            "mode": "mtp_off", "count_basis": "accepted_output_tokens", "accepted_tokens": 1,
            "coverage": global_coverage,
            "numa": {"pcie_local_node": "numa0", "transfer_worker_node": "numa0", "cpu_worker_nodes": ["numa0", "numa1"], "page_placement_declared": True},
            "link": {"endpoint_capability": {"generation": 2, "lanes": 8}, "negotiated": {"generation": 2, "lanes": 4},
                     "h2d": {"payload_bytes_per_second": 1000000, "submission_milliseconds": 0.25},
                     "d2h": {"payload_bytes_per_second": 500000, "submission_milliseconds": 0.5}},
            "stages": [
                {"id": "layers", "scope": "layer", "repeat": 2, "coverage": layer_coverage,
                 "overlap": {"mode": "independent_experts", "supported": True,
                             "evidence": {"kind": "synthetic", "detail": "Independent fictional expert branches only."}},
                 "before": [operation("fixed", "host", ["fixed", "shared_expert", "routing", "activation", "state"], 1)],
                 "cpu": [operation("cold", "cpu", ["cold_expert"], 4)],
                 "fpga": [operation("upload", "h2d", ["transfers"], size=1000, submissions=2),
                          operation("hot", "fpga", ["hot_expert"], 2, deps=["upload"]),
                          operation("download", "d2h", ["transfers"], deps=["hot"], size=500, submissions=1)],
                 "after": [operation("join", "merge", ["sync_merge"], 0.5, deps=["cold", "download"])]},
                {"id": "outer", "scope": "non_expert", "repeat": 1, "coverage": outer_coverage,
                 "overlap": {"mode": "serial"}, "before": [], "cpu": [], "fpga": [],
                 "after": [operation("outer-all", "host", ["embedding", "head", "sampling", "state"], 1)]},
            ],
        },
    }


class ResourceBudgetTests(unittest.TestCase):
    def check_invalid(self, change, pattern=None):
        document = fixture()
        change(document)
        with self.assertRaisesRegex(budget.BudgetError, pattern or ".+"):
            budget.analyze(document)

    def resource(self, report, name):
        return next(resource for resource in report["memory"]["resources"] if resource["id"] == name)

    def test_synthetic_fit_never_certifies_readiness(self):
        report = budget.analyze(fixture())
        self.assertEqual(report["status"], "fit")
        self.assertTrue(report["complete"])
        self.assertTrue(report["strict_pass"])
        self.assertTrue(report["analytical_only"])
        self.assertFalse(report["input_provenance_verified"])
        self.assertFalse(report["device_ready"])
        self.assertFalse(report["inference_ready"])
        self.assertEqual(report["unknown_inputs"], [])

    def test_per_bank_aligned_reserve_fragmentation_accounting(self):
        report = budget.analyze(fixture())
        bank = self.resource(report, "ddr0")
        self.assertEqual(bank["used_bytes"], 176)
        self.assertEqual(bank["headroom_bytes"], 848)
        self.assertEqual(report["memory"]["allocations"][0]["aligned_bytes"], 128)
        self.assertEqual(report["memory"]["host_total"]["used_bytes"], 16)

    def test_unknown_reserve_cannot_be_fit_or_zero(self):
        document = fixture()
        document["memory"]["device"]["banks"][0]["reserve_bytes"] = None
        report = budget.analyze(document)
        bank = self.resource(report, "ddr0")
        self.assertEqual(bank["status"], "unknown")
        self.assertIsNone(bank["used_bytes"])
        self.assertIsNone(bank["headroom_bytes"])
        self.assertEqual(bank["known_used_lower_bound_bytes"], 144)
        self.assertFalse(report["strict_pass"])

    def test_unknown_fragmentation_and_capacity_remain_unknown(self):
        for key in ("fragmentation_bytes", "capacity_bytes"):
            with self.subTest(key=key):
                document = fixture()
                document["memory"]["device"]["banks"][0][key] = None
                self.assertEqual(self.resource(budget.analyze(document), "ddr0")["status"], "unknown")

    def test_overfull_bank_even_when_total_fits(self):
        document = fixture()
        document["memory"]["allocations"][0].update(bytes=1000)
        document["memory"]["allocations"][0]["placements"][0]["bytes"] = 1000
        report = budget.analyze(document)
        self.assertEqual(report["memory"]["device_total"]["status"], "fit")
        self.assertEqual(self.resource(report, "ddr0")["status"], "over_capacity")
        self.assertEqual(report["status"], "over_capacity")
        self.assertFalse(report["strict_pass"])

    def test_overfull_known_lower_bound_survives_unknown(self):
        document = fixture()
        bank = document["memory"]["device"]["banks"][0]
        bank.update(capacity_bytes=100, reserve_bytes=None)
        report = budget.analyze(document)
        self.assertEqual(self.resource(report, "ddr0")["status"], "over_capacity")
        self.assertIsNone(self.resource(report, "ddr0")["used_bytes"])
        self.assertFalse(report["complete"])

    def test_total_device_constraint_independent(self):
        document = fixture()
        document["memory"]["device"]["capacity_bytes"] = 100
        report = budget.analyze(document)
        self.assertEqual(self.resource(report, "ddr0")["status"], "fit")
        self.assertEqual(report["memory"]["device_total"]["status"], "over_capacity")

    def test_explicit_device_interleave_counts_each_bank(self):
        document = fixture()
        item = document["memory"]["allocations"][0]
        item["bytes"] = 130
        item["placements"] = [{"resource": name, "bytes": 65, "alignment_bytes": 64} for name in ("ddr0", "ddr1")]
        report = budget.analyze(document)
        self.assertEqual(self.resource(report, "ddr0")["used_bytes"], 176)
        self.assertEqual(self.resource(report, "ddr1")["used_bytes"], 128)

    def test_host_global_node_and_pinned_quotas_independent(self):
        document = fixture()
        document["memory"]["coverage"]["dma_staging"]["status"] = "complete"
        document["memory"]["allocations"].append(allocation("pin", 129, "numa0", "dma_staging", True))
        document["memory"]["host"]["pinned_quota_bytes"] = 128
        report = budget.analyze(document)
        self.assertEqual(self.resource(report, "numa0")["status"], "fit")
        self.assertEqual(self.resource(report, "numa0")["pinned"]["status"], "fit")
        self.assertEqual(report["memory"]["host_total"]["pinned"]["status"], "over_capacity")
        self.assertEqual(report["memory"]["host_total"]["used_bytes"], 208)
        document["memory"]["host"]["pinned_quota_bytes"] = 512
        document["memory"]["host"]["numa_nodes"][0]["pinned_quota_bytes"] = 128
        self.assertEqual(self.resource(budget.analyze(document), "numa0")["pinned"]["status"], "over_capacity")
        document["memory"]["host"]["numa_nodes"][0]["pinned_quota_bytes"] = 256
        document["memory"]["host"]["capacity_bytes"] = 200
        self.assertEqual(budget.analyze(document)["memory"]["host_total"]["status"], "over_capacity")
        document["memory"]["host"]["capacity_bytes"] = 4096
        document["memory"]["host"]["numa_nodes"][0]["capacity_bytes"] = 128
        self.assertEqual(self.resource(budget.analyze(document), "numa0")["status"], "over_capacity")

    def test_unknown_host_quota_not_inferred_from_ram(self):
        document = fixture()
        document["memory"]["host"]["pinned_quota_bytes"] = None
        report = budget.analyze(document)
        self.assertEqual(report["memory"]["host_total"]["pinned"]["status"], "unknown")
        self.assertFalse(report["strict_pass"])

    def test_missing_inventory_is_unknown(self):
        document = fixture()
        del document["memory"]["coverage"]["kv"]
        report = budget.analyze(document)
        self.assertEqual(report["memory"]["status"], "unknown")
        self.assertIn("memory.coverage.kv", report["unknown_inputs"])

    def test_complete_inventory_requires_allocation(self):
        self.check_invalid(lambda doc: doc["memory"]["coverage"]["kv"].update(status="complete"), "needs an allocation")

    def test_absent_inventory_cannot_have_allocation(self):
        self.check_invalid(lambda doc: doc["memory"]["coverage"]["workspace"].update(status="absent"), "absent category")

    def test_unknown_allocations_conservatively_taint_inventory(self):
        for key in ("bytes", "pinned"):
            with self.subTest(key=key):
                document = fixture()
                document["memory"]["allocations"][0][key] = None
                report = budget.analyze(document)
                self.assertEqual(report["memory"]["status"], "unknown")
                self.assertEqual(self.resource(report, "ddr1")["status"], "unknown")
        document = fixture()
        document["memory"]["allocations"][0]["placements"][0]["alignment_bytes"] = None
        bank = self.resource(budget.analyze(document), "ddr0")
        self.assertEqual(bank["known_used_lower_bound_bytes"], 113)
        self.assertEqual(bank["status"], "unknown")

    def test_allocations_must_be_explicitly_placed(self):
        for value in ([], [{"resource": "total_ddr", "bytes": 65, "alignment_bytes": 64}]):
            with self.subTest(value=value):
                self.check_invalid(lambda doc: doc["memory"]["allocations"][0].update(placements=value))

    def test_placement_bytes_must_equal_allocation_bytes(self):
        self.check_invalid(lambda doc: doc["memory"]["allocations"][0]["placements"][0].update(bytes=64), "placement byte")

    def test_no_mixed_host_device_or_cross_numa_allocations(self):
        for nodes in (("ddr0", "numa0"), ("numa0", "numa1")):
            with self.subTest(nodes=nodes):
                self.check_invalid(lambda doc: doc["memory"]["allocations"][0].update(
                    bytes=130, placements=[{"resource": node, "bytes": 65, "alignment_bytes": 64} for node in nodes]), "separate allocations")

    def test_duplicate_resources_allocations_and_placements_rejected(self):
        self.check_invalid(lambda doc: doc["memory"]["device"]["banks"].append(copy.deepcopy(doc["memory"]["device"]["banks"][0])), "duplicate resource")
        self.check_invalid(lambda doc: doc["memory"]["allocations"].append(copy.deepcopy(doc["memory"]["allocations"][0])), "duplicate allocation")
        self.check_invalid(lambda doc: doc["memory"]["allocations"][0]["placements"].append(copy.deepcopy(doc["memory"]["allocations"][0]["placements"][0])), "duplicate resource placement")

    def test_bytes_alignment_and_schema_bools_rejected(self):
        for value in (-1, True, False, 1.0, "64", budget.U64_MAX + 1):
            with self.subTest(value=value):
                self.check_invalid(lambda doc: doc["memory"]["allocations"][0].update(bytes=value), "u64")
        for value in (0, 3, True):
            with self.subTest(alignment=value):
                self.check_invalid(lambda doc: doc["memory"]["allocations"][0]["placements"][0].update(alignment_bytes=value))
        self.check_invalid(lambda doc: doc.update(schema_version=True), "integer 1")
        self.check_invalid(lambda doc: doc["model"].update(expert_skip=False))
        self.check_invalid(lambda doc: doc["memory"]["allocations"][0].update(pinned=1), "boolean")
        self.check_invalid(lambda doc: doc["memory"]["device"]["banks"][0].update(capacity_bytes=True), "u64")

    def test_u64_alignment_and_resource_sum_overflow(self):
        self.check_invalid(lambda doc: doc["memory"]["allocations"][0].update(
            bytes=budget.U64_MAX, placements=[{"resource": "ddr0", "bytes": budget.U64_MAX, "alignment_bytes": 64}]), "overflow")
        self.check_invalid(lambda doc: doc["memory"]["device"]["banks"][0].update(reserve_bytes=budget.U64_MAX), "overflow")
        self.check_invalid(lambda doc: doc["memory"]["device"]["banks"].extend([
            {"id": "huge0", "capacity_bytes": budget.U64_MAX, "reserve_bytes": budget.U64_MAX, "fragmentation_bytes": 0},
            {"id": "huge1", "capacity_bytes": budget.U64_MAX, "reserve_bytes": budget.U64_MAX, "fragmentation_bytes": 0}]), "overflow")

    def with_weight(self):
        document = fixture()
        document["memory"]["tensors"] = [tensor()]
        document["memory"]["coverage"]["hot_experts"]["status"] = "complete"
        item = allocation("weight", 148, "ddr1", "hot_experts")
        item.update(tensor_ids=["tiny"], tensor_storage="resident")
        document["memory"]["allocations"].append(item)
        return document

    def test_actual_packed_weight_metadata_accounted(self):
        document = self.with_weight()
        report = budget.analyze(document)
        self.assertTrue(report["strict_pass"])
        self.assertEqual(self.resource(report, "ddr1")["used_bytes"], 192)
        self.assertEqual(report["memory"]["tensors"][0]["resident_bytes"], 148)

    def test_weight_bytes_without_metadata_unknown_not_aggregate_division(self):
        document = fixture()
        document["memory"]["coverage"]["hot_experts"]["status"] = "complete"
        item = allocation("weight", 148, "ddr1", "hot_experts")
        item["tensor_storage"] = "resident"
        document["memory"]["allocations"].append(item)
        report = budget.analyze(document)
        self.assertEqual(report["memory"]["status"], "unknown")
        self.assertTrue(any("actual packed metadata required" in path for path in report["unknown_inputs"]))

    def test_duplicate_tensor_ids_source_metadata_and_refs_rejected(self):
        for mutate, match in (
            (lambda doc: doc["memory"]["tensors"].append(copy.deepcopy(doc["memory"]["tensors"][0])), "duplicate tensor id"),
            (lambda doc: doc["memory"]["tensors"].append(tensor("second", "tiny")), "duplicate source tensor"),
            (lambda doc: doc["memory"]["allocations"][1].update(tensor_ids=["tiny", "tiny"]), "duplicate tensor references"),
        ):
            with self.subTest(match=match):
                document = self.with_weight()
                mutate(document)
                with self.assertRaisesRegex(budget.BudgetError, match):
                    budget.analyze(document)

    def test_weight_shape_block_row_and_resident_metadata_rejected(self):
        for key, value in (("shape", [255, 2]), ("row_stride_bytes", 73), ("data_bytes", 147), ("block_values", True)):
            document = self.with_weight()
            document["memory"]["tensors"][0]["packed"][key] = value
            with self.subTest(key=key), self.assertRaises(budget.BudgetError):
                budget.analyze(document)
        document = self.with_weight()
        document["memory"]["tensors"][0]["resident"]["dtype"] = "FP4"
        with self.assertRaisesRegex(budget.BudgetError, "native resident dtype"):
            budget.analyze(document)

    def test_unknown_dtype_layout_scales_checksum_retention_are_not_fit(self):
        for section, key in (("packed", "dtype"), ("packed", "layout"), ("packed", "external_scale_bytes"),
                             ("resident", "scale_bytes"), ("resident", "overhead_bytes"), ("resident", "source_retained"), ("source", "sha256")):
            document = self.with_weight()
            document["memory"]["tensors"][0][section][key] = None
            with self.subTest(section=section, key=key):
                self.assertEqual(budget.analyze(document)["memory"]["status"], "unknown")

    def test_weight_allocation_cannot_hide_scales_or_conversion(self):
        document = self.with_weight()
        document["memory"]["tensors"][0]["packed"]["external_scale_bytes"] = 4
        document["memory"]["tensors"][0]["resident"]["scale_bytes"] = 4
        with self.assertRaisesRegex(budget.BudgetError, "weight allocation differs"):
            budget.analyze(document)
        document = self.with_weight()
        document["memory"]["tensors"][0]["resident"].update(status="converted", dtype="synthetic-FP4", layout="converted",
                                                               data_bytes=256, scale_bytes=16, overhead_bytes=8,
                                                               source_retained=True, evidence="Synthetic conversion overhead only.")
        document["memory"]["allocations"][1].update(bytes=280, placements=[{"resource": "ddr1", "bytes": 280, "alignment_bytes": 64}])
        with self.assertRaisesRegex(budget.BudgetError, "retained packed sources"):
            budget.analyze(document)
        original = allocation("original", 148, "numa0", "hot_experts")
        original.update(tensor_ids=["tiny"], tensor_storage="packed_source")
        document["memory"]["allocations"].append(original)
        with self.assertRaisesRegex(budget.BudgetError, "conversion_workspace"):
            budget.analyze(document)
        document["memory"]["coverage"]["conversion_workspace"]["status"] = "complete"
        document["memory"]["allocations"].append(allocation("convert", 32, "numa0", "conversion_workspace"))
        report = budget.analyze(document)
        self.assertTrue(report["strict_pass"])
        self.assertEqual(self.resource(report, "ddr1")["used_bytes"], 320)
        self.assertEqual(self.resource(report, "numa0")["used_bytes"], 256)

    def test_unused_or_undeclared_tensor_rejected(self):
        document = self.with_weight()
        document["memory"]["allocations"][1]["tensor_ids"] = ["none"]
        with self.assertRaisesRegex(budget.BudgetError, "undeclared tensor"):
            budget.analyze(document)
        document = self.with_weight()
        document["memory"]["tensors"].append(tensor("unused"))
        with self.assertRaisesRegex(budget.BudgetError, "unused metadata"):
            budget.analyze(document)

    def test_source_kind_and_sha_are_not_format_names(self):
        for key, value in (("kind", "aggregate_gguf_division"), ("sha256", "checksum")):
            document = self.with_weight()
            document["memory"]["tensors"][0]["source"][key] = value
            with self.subTest(key=key), self.assertRaises(budget.BudgetError):
                budget.analyze(document)

    def test_independent_wall_max_and_directional_submission_cost(self):
        report = budget.analyze(fixture())
        # FPGA: H2D 1 + 2*.25; hot 2; D2H 1 + 1*.5 = 5 ms.
        # Each layer: 1 + max(CPU 4, FPGA 5) + .5 = 6.5; 2 layers + outer 1 = 14 ms.
        self.assertEqual(report["wall"]["milliseconds"], 14)
        self.assertAlmostEqual(report["wall"]["estimated_accepted_tokens_per_second"], 1000 / 14)
        self.assertFalse(report["wall"]["estimate_is_measurement"])

    def test_serial_wall_sums_dependency_chains_not_rates(self):
        document = fixture()
        document["wall"]["stages"][0]["overlap"] = {"mode": "serial"}
        report = budget.analyze(document)
        self.assertEqual(report["wall"]["milliseconds"], 22)

    def test_unverified_overlap_never_estimates_rate(self):
        for overlap in (
            {"mode": "independent_experts", "supported": None, "evidence": None},
            {"mode": "independent_experts", "supported": False, "evidence": {"kind": "source", "detail": "No hardware evidence."}},
            {"mode": "independent_experts", "supported": True, "evidence": None},
        ):
            document = fixture()
            document["wall"]["stages"][0]["overlap"] = overlap
            with self.subTest(overlap=overlap):
                report = budget.analyze(document)
                self.assertEqual(report["wall"]["status"], "unknown")
                self.assertIsNone(report["wall"]["estimated_accepted_tokens_per_second"])

    def test_unsupported_overlap_and_empty_evidence_rejected(self):
        for value in ("add_rates", "all_work_parallel", "pipeline"):
            self.check_invalid(lambda doc: doc["wall"]["stages"][0]["overlap"].update(mode=value), "unsupported overlap")
        for evidence in ("yes", {"kind": "synthetic", "detail": ""}, {"kind": "guess", "detail": "yes"}):
            self.check_invalid(lambda doc: doc["wall"]["stages"][0]["overlap"].update(evidence=evidence))
        self.check_invalid(lambda doc: doc["wall"]["stages"][1].update(overlap={"mode": "independent_experts", "supported": True}), "only modeled")

    def test_missing_stage_coverage_or_nonexpert_path_is_unknown(self):
        document = fixture()
        del document["wall"]["stages"][0]["coverage"]["state"]
        report = budget.analyze(document)
        self.assertEqual(report["wall"]["status"], "unknown")
        self.assertIn("wall.stages[0].coverage.state", report["unknown_inputs"])
        self.assertIsNone(report["wall"]["stages"][0]["milliseconds"])
        document = fixture()
        document["wall"]["stages"].pop()
        document["wall"]["coverage"]["sampling"]["status"] = "unknown"
        self.assertIsNone(budget.analyze(document)["wall"]["estimated_accepted_tokens_per_second"])

    def test_stage_complete_requires_role_and_mandatory_work(self):
        self.check_invalid(lambda doc: doc["wall"]["stages"][0]["coverage"]["cache_misses"].update(status="complete"), "covering operation")
        self.check_invalid(lambda doc: doc["wall"]["stages"][0]["coverage"]["shared_expert"].update(status="absent"), "mandatory work")
        self.check_invalid(lambda doc: doc["wall"]["stages"][0]["before"][0].update(covers=[]), "stage categories")

    def test_global_complete_needs_stage_operations_and_absent_cannot_hide_them(self):
        self.check_invalid(lambda doc: doc["wall"]["coverage"]["fallback"].update(status="complete"), "mapped covering operation")
        for category in ("non_expert", "transfers", "sync_merge", "sampling"):
            with self.subTest(category=category):
                self.check_invalid(lambda doc: doc["wall"]["coverage"][category].update(status="absent"), "absent conflicts")
        self.check_invalid(lambda doc: doc["wall"]["stages"][0]["coverage"]["cache_misses"].update(status="unknown"), "absent conflicts")

    def test_expert_compute_cannot_be_mislabeled_as_transfer_or_join(self):
        for branch in ("cpu", "fpga"):
            with self.subTest(branch=branch):
                index = 0 if branch == "cpu" else 1
                self.check_invalid(lambda doc: doc["wall"]["stages"][0][branch][index].update(covers=["transfers"]), "expert compute must cover")
        for kind in ("sync", "merge"):
            self.check_invalid(lambda doc: doc["wall"]["stages"][0]["cpu"][0].update(kind=kind), "synchronization/merge")
        self.check_invalid(lambda doc: doc["wall"]["stages"][0]["fpga"][0].update(covers=["hot_expert", "transfers"]), "hot_expert needs fpga")

    def test_both_expert_tiers_cannot_be_omitted(self):
        document = fixture()
        stage = document["wall"]["stages"][0]
        stage["coverage"]["cold_expert"]["status"] = "absent"
        stage["coverage"]["hot_expert"]["status"] = "absent"
        stage["cpu"], stage["fpga"] = [], []
        with self.assertRaisesRegex(budget.BudgetError, "full top10"):
            budget.analyze(document)

    def test_tag_only_transfer_is_not_byte_accounting(self):
        document = fixture()
        stage = document["wall"]["stages"][0]
        stage["before"][0]["covers"].append("transfers")
        with self.assertRaisesRegex(budget.BudgetError, "byte/submission-charged"):
            budget.analyze(document)

    def test_residency_unknown_or_over_capacity_suppresses_wall_rate(self):
        for capacity in (None, 100):
            document = fixture()
            document["memory"]["device"]["capacity_bytes"] = capacity
            report = budget.analyze(document)
            self.assertIsNone(report["wall"]["milliseconds"])
            self.assertIsNone(report["wall"]["estimated_accepted_tokens_per_second"])
            self.assertEqual(report["wall"]["known_lower_bound_milliseconds"], 14)

    def test_full_routing_and_skip_invariants(self):
        self.check_invalid(lambda doc: doc["model"].update(routed_top_k=9))
        self.check_invalid(lambda doc: doc["model"].update(expert_skip=1))
        self.check_invalid(lambda doc: doc["model"].update(shared_expert=False))

    def test_fixed_shared_routing_work_cannot_hide_in_expert_branch(self):
        for role in ("fixed", "shared_expert", "routing", "activation", "state", "sync_merge"):
            with self.subTest(role=role):
                self.check_invalid(lambda doc: doc["wall"]["stages"][0]["cpu"][0].update(covers=[role]), "only cold/hot")
        self.check_invalid(lambda doc: doc["wall"]["stages"][0]["cpu"][0].update(kind="fpga"), "wrong independent branch")

    def test_cpu_producer_to_fpga_h2d_cross_chain_rejected(self):
        self.check_invalid(lambda doc: doc["wall"]["stages"][0]["fpga"][0].update(depends_on=["cold"]), "cross-expert-chain")
        self.check_invalid(lambda doc: doc["wall"]["stages"][0]["cpu"][0].update(depends_on=["upload"]), "earlier operation")

    def test_cpu_result_h2d_after_join_is_charged_serially(self):
        document = fixture()
        stage = document["wall"]["stages"][0]
        stage["after"].insert(0, operation("cold-result-upload", "h2d", ["transfers"], deps=["cold"], size=1000, submissions=2))
        report = budget.analyze(document)
        self.assertEqual(report["wall"]["milliseconds"], 17)

    def test_duplicate_operations_stages_and_dependencies_rejected(self):
        self.check_invalid(lambda doc: doc["wall"]["stages"].append(copy.deepcopy(doc["wall"]["stages"][0])), "duplicate stage")
        self.check_invalid(lambda doc: doc["wall"]["stages"][0]["cpu"].append(copy.deepcopy(doc["wall"]["stages"][0]["cpu"][0])), "duplicate operation")
        self.check_invalid(lambda doc: doc["wall"]["stages"][0]["fpga"][1].update(depends_on=["upload", "upload"]), "duplicate dependencies")
        self.check_invalid(lambda doc: doc["wall"]["stages"][0]["fpga"][1].update(depends_on=["missing"]), "earlier operation")

    def test_unknown_timings_bandwidth_submission_or_link_do_not_estimate(self):
        for area, key in (("h2d", "payload_bytes_per_second"), ("d2h", "payload_bytes_per_second"), ("h2d", "submission_milliseconds")):
            document = fixture()
            document["wall"]["link"][area][key] = None
            with self.subTest(area=area, key=key):
                self.assertIsNone(budget.analyze(document)["wall"]["estimated_accepted_tokens_per_second"])
        document = fixture()
        document["wall"]["stages"][0]["cpu"][0]["milliseconds"] = None
        self.assertEqual(budget.analyze(document)["wall"]["status"], "unknown")
        document = fixture()
        document["wall"]["link"]["negotiated"]["lanes"] = None
        self.assertEqual(budget.analyze(document)["wall"]["status"], "unknown")

    def test_capability_and_history_never_supply_payload_rate(self):
        document = fixture()
        document["wall"]["link"]["h2d"]["payload_bytes_per_second"] = None
        document["wall"]["link"]["historical_h2d_bytes_per_second"] = 1390000000
        self.assertIsNone(budget.analyze(document)["wall"]["estimated_accepted_tokens_per_second"])

    def test_numa_locality_page_placement_and_context_unknown(self):
        for key in ("pcie_local_node", "transfer_worker_node", "cpu_worker_nodes", "page_placement_declared"):
            document = fixture()
            document["wall"]["numa"][key] = None
            with self.subTest(key=key):
                self.assertEqual(budget.analyze(document)["wall"]["status"], "unknown")
        for key in ("layers", "context_tokens", "sessions"):
            document = fixture()
            document["model"][key] = None
            report = budget.analyze(document)
            self.assertFalse(report["strict_pass"])
            self.assertEqual(report["memory"]["status"], "unknown")
            self.assertIsNone(report["memory"]["device_total"]["used_bytes"])
            self.assertIsNone(report["wall"]["estimated_accepted_tokens_per_second"])
        self.check_invalid(lambda doc: doc["wall"]["numa"].update(pcie_local_node="third-node"), "undeclared NUMA")

    def test_layer_repeats_do_not_admit_partial_model_wall_rate(self):
        document = fixture()
        document["wall"]["stages"][0]["repeat"] = 1
        self.assertIsNone(budget.analyze(document)["wall"]["estimated_accepted_tokens_per_second"])

    def test_nonfinite_bool_negative_and_transfer_submission_rejected(self):
        for value in (True, -1, float("nan"), float("inf"), 10 ** 400):
            with self.subTest(value=value):
                self.check_invalid(lambda doc: doc["wall"]["stages"][0]["cpu"][0].update(milliseconds=value))
        self.check_invalid(lambda doc: doc["wall"]["link"]["h2d"].update(payload_bytes_per_second=0))
        self.check_invalid(lambda doc: doc["wall"]["stages"][0]["fpga"][0].update(submissions=0), "requires submissions")
        self.check_invalid(lambda doc: doc["wall"]["stages"][0].update(repeat=True))

    def test_wall_float_overflow_rejected(self):
        self.check_invalid(lambda doc: doc["wall"]["stages"][0]["cpu"][0].update(milliseconds=1e308), "overflow")
        self.check_invalid(lambda doc: doc["wall"]["link"]["h2d"].update(payload_bytes_per_second=5e-324), "overflow")

    def mtp_fixture(self):
        document = fixture()
        document["wall"].update(mode="mtp_on", accepted_tokens=2, accepted_accounting={
            "status": "declared", "evidence": {"kind": "synthetic", "detail": "Window includes all proposal/verification/commit costs, counts accepted output only."}})
        document["wall"]["coverage"]["mtp_verify_commit"]["status"] = "complete"
        covers = status(budget.STAGE_CATEGORIES["mtp"])
        for key in budget.MANDATORY_STAGE_WORK["mtp"]:
            covers[key]["status"] = "complete"
        document["wall"]["stages"].append({"id": "mtp", "scope": "mtp", "repeat": 1,
            "coverage": covers, "overlap": {"mode": "serial"}, "before": [], "cpu": [], "fpga": [],
            "after": [operation("mtp-work", "host", sorted(budget.MANDATORY_STAGE_WORK["mtp"]), 6)]})
        return document

    def test_mtp_on_counts_accepted_not_proposals(self):
        report = budget.analyze(self.mtp_fixture())
        self.assertEqual(report["wall"]["milliseconds"], 20)
        self.assertEqual(report["wall"]["estimated_accepted_tokens_per_second"], 100)
        document = self.mtp_fixture()
        document["wall"]["count_basis"] = "proposed_tokens"
        with self.assertRaisesRegex(budget.BudgetError, "accepted output"):
            budget.analyze(document)

    def test_mtp_unknown_zero_or_missing_accepted_frontier_never_estimates(self):
        for tokens in (None, 0):
            document = self.mtp_fixture()
            document["wall"]["accepted_tokens"] = tokens
            with self.subTest(tokens=tokens):
                self.assertIsNone(budget.analyze(document)["wall"]["estimated_accepted_tokens_per_second"])
        document = self.mtp_fixture()
        document["wall"]["accepted_accounting"]["status"] = "unknown"
        self.assertIsNone(budget.analyze(document)["wall"]["estimated_accepted_tokens_per_second"])
        document = self.mtp_fixture()
        document["wall"]["accepted_accounting"]["evidence"] = None
        self.assertIsNone(budget.analyze(document)["wall"]["estimated_accepted_tokens_per_second"])
        document = self.mtp_fixture()
        document["wall"]["stages"].pop()
        document["wall"]["coverage"]["mtp_verify_commit"]["status"] = "unknown"
        self.assertIsNone(budget.analyze(document)["wall"]["estimated_accepted_tokens_per_second"])

    def test_mtp_off_and_on_cannot_mix(self):
        self.check_invalid(lambda doc: doc["wall"]["coverage"]["mtp_verify_commit"].update(status="complete"), "MTP-off")
        document = self.mtp_fixture()
        document["wall"]["coverage"]["mtp_verify_commit"]["status"] = "absent"
        with self.assertRaisesRegex(budget.BudgetError, "MTP-on"):
            budget.analyze(document)

    def test_reference_deliberately_incomplete_no_model_artifacts_or_rates(self):
        document = json.loads(budget.DEFAULT_INPUT.read_text())
        report = budget.analyze(document)
        self.assertEqual(report["input_kind"], "reference")
        self.assertEqual(report["status"], "unknown")
        self.assertFalse(report["strict_pass"])
        self.assertEqual(document["memory"]["tensors"], [])
        self.assertEqual(document["memory"]["allocations"], [])
        self.assertIsNone(report["wall"]["milliseconds"])
        self.assertEqual(report["memory"]["device_total"]["capacity_bytes"], 4294967296)
        self.assertIsNone(document["memory"]["host"]["numa_nodes"][0]["capacity_bytes"])
        self.assertIsNone(document["wall"]["link"]["h2d"]["payload_bytes_per_second"])

    def cli(self, *arguments, cwd=None):
        return subprocess.run([sys.executable, str(ROOT / "scripts/resource_budget.py"), *map(str, arguments)],
                              cwd=cwd, text=True, capture_output=True, check=False)

    def test_cli_default_reference_from_unrelated_cwd_and_strict_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.cli(cwd=directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "unknown")
            result = self.cli("--strict", cwd=directory)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertFalse(json.loads(result.stdout)["inference_ready"])

    def test_cli_strict_fit_and_over_capacity_report_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            document = fixture()
            path.write_text(json.dumps(document))
            result = self.cli(path, "--strict")
            self.assertEqual(result.returncode, 0, result.stderr)
            document["memory"]["device"]["capacity_bytes"] = 100
            path.write_text(json.dumps(document))
            default = self.cli(path)
            strict = self.cli(path, "--strict")
            self.assertEqual(default.returncode, 0, default.stderr)
            self.assertEqual(strict.returncode, 1, strict.stderr)
            self.assertEqual(json.loads(default.stdout)["status"], "over_capacity")

    def test_cli_invalid_json_duplicate_keys_nonfinite_and_missing_file_exit2(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for text in ("{", '{"schema_version":1,"schema_version":1}', '{"x":NaN}', '{"x":Infinity}', '{"schema_version":true}'):
                path.write_text(text)
                with self.subTest(text=text):
                    result = self.cli(path)
                    self.assertEqual(result.returncode, 2)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("input error", result.stderr)
            result = self.cli(Path(directory) / "missing.json")
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")

    def test_cli_oversized_json_integer_returns_input_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized-integer.json"
            path.write_text('{"schema_version":' + "9" * 5000 + "}", encoding="utf-8")
            for arguments in ((path,), (path, "--strict")):
                with self.subTest(strict="--strict" in arguments):
                    result = self.cli(*arguments)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertTrue(result.stderr.startswith("resource budget input error: "), result.stderr)
                    self.assertNotIn("Traceback", result.stderr)

    def test_cli_malformed_enum_and_role_types_fail_closed(self):
        mutations = (
            lambda doc: doc.update(input_kind=[]),
            lambda doc: doc["memory"]["coverage"]["workspace"].update(status={}),
            lambda doc: doc["memory"]["allocations"][0].update(category=[]),
            lambda doc: doc["wall"].update(mode=[]),
            lambda doc: doc["wall"]["stages"][0].update(scope={}),
            lambda doc: doc["wall"]["stages"][0]["overlap"].update(mode=[]),
            lambda doc: doc["wall"]["stages"][0]["overlap"]["evidence"].update(kind={}),
            lambda doc: doc["wall"]["stages"][0]["cpu"][0].update(kind=[]),
            lambda doc: doc["wall"]["stages"][0]["cpu"][0].update(covers=[[]]),
            lambda doc: doc["wall"]["stages"][0]["cpu"][0].update(depends_on=[{}]),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.json"
            for index, mutate in enumerate(mutations):
                document = fixture()
                mutate(document)
                path.write_text(json.dumps(document))
                with self.subTest(case=index):
                    result = self.cli(path)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("input error", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)

    def test_input_annotations_paths_urls_commands_never_executed(self):
        document = fixture()
        document.update(path="/not/an/artifact", url="https://invalid.example", command="raise an error")
        document["wall"]["link"]["historical"] = {"instruction": "invent bandwidth"}
        self.assertTrue(budget.analyze(document)["strict_pass"])


if __name__ == "__main__":
    unittest.main()
