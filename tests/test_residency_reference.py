"""Offline checks for the reference-only 4 GiB residency calculator.

The calculator turns a validated mixed-GGUF header inventory receipt into
declared byte arithmetic. These tests use a synthetic receipt that reproduces
the pinned reference totals (three expert format bands, one oversize global
tensor, 48 layers) so that a silent change to the arithmetic, the declared
reference geometry or the fail-closed receipt handling appears as an explicit
reviewed diff. They never read model payload, never open a device and do not
authenticate any number as a measurement.
"""
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
SCRIPT = ROOT / "scripts/residency_reference.py"

SPEC = importlib.util.spec_from_file_location("residency_reference", SCRIPT)
residency = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(residency)

# Reference totals reproduced by the synthetic receipt.
PERSISTENT_FIXED = 3_071_375_360
EXPERT_TOTAL = 35_454_976_000
FITTABLE_GLOBALS = 688_578_560
OVERSIZE_GLOBAL = 28_800_138_240
DEVICE = 4_294_967_296
IQ2_S_LAYERS = 34
IQ2_XXS_LAYERS = 11
IQ1_M_LAYERS = 3
IQ2_S_SET = 773_324_800
IQ2_XXS_SET = 668_467_200
IQ1_M_SET = 602_931_200
FIXED_MAX_LAYER_BYTES = 127_660_416
FIXED_MIN_LAYER_BYTES = 58_294_272


def _layer_band(layer: int) -> int:
    if layer < IQ2_S_LAYERS:
        return IQ2_S_SET
    if layer < IQ2_S_LAYERS + IQ2_XXS_LAYERS:
        return IQ2_XXS_SET
    return IQ1_M_SET


def _fixed_layer(layer: int) -> int:
    if layer == 1:
        return FIXED_MAX_LAYER_BYTES
    if layer == 35:
        return FIXED_MIN_LAYER_BYTES
    others = 48 - 2
    quotient, remainder = divmod(
        PERSISTENT_FIXED - FIXED_MAX_LAYER_BYTES - FIXED_MIN_LAYER_BYTES, others
    )
    return quotient + (remainder if layer == 0 else 0)


def build_receipt() -> dict:
    tensors = []
    for layer in range(48):
        fixed = _fixed_layer(layer)
        tensors.append({
            "name": f"blk.{layer}.attn_norm.weight",
            "shape": [2560],
            "ggml_type_name": "F32",
            "storage": {"data_bytes": fixed},
        })
        expert_set = _layer_band(layer)
        gate = expert_set // 2
        up = expert_set // 4
        down = expert_set - gate - up
        fmt = "IQ2_S" if layer < IQ2_S_LAYERS else (
            "IQ2_XXS" if layer < IQ2_S_LAYERS + IQ2_XXS_LAYERS else "IQ1_M"
        )
        for role, size in (("gate", gate), ("up", up)):
            tensors.append({
                "name": f"blk.{layer}.ffn_{role}_exps.weight",
                "shape": [640, 2560, 512],
                "ggml_type_name": fmt,
                "storage": {"data_bytes": size},
            })
        tensors.append({
            "name": f"blk.{layer}.ffn_down_exps.weight",
            "shape": [2560, 640, 512],
            "ggml_type_name": "Q2_0",
            "storage": {"data_bytes": down},
        })
    globals_and_sizes = {
        "token_embd.weight": 337_715_200,
        "output.weight": 337_715_200,
        "output_hc_down.weight": 6_553_600,
        "output_hc_up.weight": 6_553_600,
        "output_hc_norm.weight": 40_960,
        "per_layer_token_embd.weight": OVERSIZE_GLOBAL,
    }
    for name, size in globals_and_sizes.items():
        tensors.append({
            "name": name,
            "shape": [160, 320_001_536],
            "ggml_type_name": "IQ4_NL",
            "storage": {"data_bytes": size},
        })
    total = sum(item["storage"]["data_bytes"] for item in tensors)
    return {
        "kind": "bounded-gguf-header-inventory",
        "input_kind": "local-unverified",
        "claims": {"header_only": True, "bytes_verified": False},
        "tensors": tensors,
        "totals": {"known_storage_bytes": total, "tensor_count": len(tensors)},
    }


class DeclaredGeometryTest(unittest.TestCase):
    def test_state_bytes_are_pinned(self) -> None:
        self.assertEqual(residency.LINEAR_ATTENTION_LAYERS, 36)
        self.assertEqual(residency.CONV_DIM, 10_240)
        self.assertEqual(residency.STATE_BYTES, 117_669_888)
        self.assertEqual(
            residency.STATE_BYTES,
            36 * 48 * 128 * 128 * 4 + 36 * 3 * 10_240 * 4,
        )

    def test_kv_bytes_per_token_are_pinned(self) -> None:
        self.assertEqual(residency.FULL_ATTENTION_KV_PER_TOKEN, 24_576)
        self.assertEqual(residency.INDEXER_KV_PER_TOKEN, 3_072)
        self.assertEqual(residency.KV_PER_TOKEN, 27_648)
        self.assertEqual(
            residency.DECLARED_MAX_CONTEXT_TOKENS * residency.KV_PER_TOKEN,
            7_247_757_312,
        )

    def test_device_envelope_is_two_equal_banks(self) -> None:
        self.assertEqual(residency.DEVICE_CAPACITY_BYTES, 4 * 1024**3)
        self.assertEqual(residency.DEVICE_BANK_BYTES * 2, residency.DEVICE_CAPACITY_BYTES)


class AnalyseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.receipt = build_receipt()
        cls.result = residency.analyze(cls.receipt, (4096, 32768), receipt_path="<synthetic>")

    def test_weights_grouping(self) -> None:
        weights = self.result["weights"]
        self.assertEqual(weights["persistent_fixed_bytes"], PERSISTENT_FIXED)
        self.assertEqual(weights["expert_bytes"], EXPERT_TOTAL)
        self.assertEqual(weights["total_known_bytes"],
                         PERSISTENT_FIXED + EXPERT_TOTAL + FITTABLE_GLOBALS + OVERSIZE_GLOBAL)
        self.assertEqual(
            weights["expert_gate_up_format_counts"],
            {"IQ1_M": 3, "IQ2_S": 34, "IQ2_XXS": 11},
        )
        self.assertEqual(weights["expert_down_format_counts"], {"Q2_0": 48})
        self.assertEqual(weights["per_expert_footprint_max_bytes"], 1_510_400)
        self.assertEqual(weights["per_expert_footprint_min_bytes"], 1_177_600)
        self.assertEqual(weights["fixed_layer_bounds"],
                         {"min": {"layer": 35, "bytes": 58_294_272},
                          "max": {"layer": 1, "bytes": 127_660_416}})

    def test_globals_split_and_oversize(self) -> None:
        weights = self.result["weights"]
        self.assertEqual(weights["globals_fittable_bytes"], FITTABLE_GLOBALS)
        self.assertEqual(weights["globals_oversize"],
                         [{"name": "per_layer_token_embd.weight", "bytes": OVERSIZE_GLOBAL}])
        self.assertEqual(weights["globals_bytes"]["per_layer_token_embd.weight"],
                         OVERSIZE_GLOBAL)
        self.assertEqual(sum(weights["globals_bytes"].values()),
                         FITTABLE_GLOBALS + OVERSIZE_GLOBAL)

    def test_declared_staging_is_double_buffered(self) -> None:
        staging = self.result["assumptions"]["staging"]
        self.assertEqual(staging["dma_double_buffer_bytes"], 2 * 1_510_400)
        self.assertEqual(staging["dense_layer_stream_double_buffer_bytes"],
                         2 * FIXED_MAX_LAYER_BYTES)

    def test_s0_all_fixed_resident_is_pinned(self) -> None:
        scenario = self._scenario("s0-all-fixed-resident")
        self.assertEqual(scenario["resident_declared_bytes"], 3_192_066_048)
        self.assertEqual(scenario["max_context_tokens_at_zero_hot"], 39_890)
        self.assertEqual(scenario["contexts"]["4096"]["hot_bytes"], 989_655_040)
        self.assertEqual(scenario["contexts"]["4096"]["hot_slots_worst"], 655)
        self.assertEqual(scenario["contexts"]["4096"]["per_layer_coverage_pct_worst"], 2.67)
        self.assertEqual(scenario["contexts"]["32768"]["hot_slots_worst"], 130)

    def test_s0g_globals_resident_does_not_fit_at_32k(self) -> None:
        scenario = self._scenario("s0g-all-fixed-and-fittable-globals-resident")
        self.assertEqual(scenario["resident_declared_bytes"], 3_880_644_608)
        self.assertEqual(scenario["max_context_tokens_at_zero_hot"], 14_985)
        self.assertEqual(scenario["contexts"]["4096"]["hot_slots_worst"], 199)
        context = scenario["contexts"]["32768"]
        self.assertFalse(context["feasible"])
        self.assertEqual(context["hot_bytes"], -491_646_976)
        self.assertEqual(context["hot_slots_worst"], 0)

    def test_s1_streamed_fixed_is_pinned(self) -> None:
        scenario = self._scenario("s1-all-fixed-streamed")
        self.assertEqual(scenario["resident_declared_bytes"], 376_011_520)
        self.assertEqual(scenario["max_context_tokens_at_zero_hot"], 141_744)
        self.assertEqual(scenario["contexts"]["4096"]["hot_bytes"], 3_805_709_568)
        self.assertEqual(scenario["contexts"]["4096"]["hot_slots_worst"], 2_519)
        self.assertEqual(scenario["contexts"]["4096"]["per_layer_coverage_pct_worst"], 10.25)
        self.assertEqual(scenario["contexts"]["32768"]["hot_slots_worst"], 1_994)

    def test_s1g_streamed_plus_globals_is_pinned(self) -> None:
        scenario = self._scenario("s1g-all-fixed-streamed-and-fittable-globals-resident")
        self.assertEqual(scenario["resident_declared_bytes"], 1_064_590_080)
        self.assertEqual(scenario["contexts"]["4096"]["hot_slots_worst"], 2_063)

    def test_s2_layer_pack_is_pinned(self) -> None:
        scenario = self._scenario("s2-layer-partitioned-hot")
        pack = scenario["cheapest_first_layer_pack"]
        self.assertEqual(pack["context"], 4096)
        self.assertEqual(pack["layers"], 5)
        self.assertEqual(pack["layer_coverage_pct"], 10.42)
        self.assertEqual(pack["packed_expert_bytes"], 3_145_728_000)

    def test_unknowns_are_explicit(self) -> None:
        self.assertGreaterEqual(len(self.result["unknowns"]), 8)
        self.assertEqual(self.result["unknowns"], list(residency.UNKNOWNS))
        self.assertTrue(all(isinstance(item, str) and item for item in self.result["unknowns"]))
        self.assertTrue(any("negotiated PCIe" in item for item in self.result["unknowns"]))
        for scenario in self.result["scenarios"]:
            for values in scenario["contexts"].values():
                self.assertNotIn("tokens_per_second", values)

    def _scenario(self, identifier: str) -> dict:
        for scenario in self.result["scenarios"]:
            if scenario["id"] == identifier:
                return scenario
        raise AssertionError(f"missing scenario {identifier}")


class FailClosedTest(unittest.TestCase):
    def test_unknown_storage_is_rejected(self) -> None:
        receipt = build_receipt()
        receipt["tensors"][0]["storage"] = None
        with self.assertRaises(residency.ResidencyError) as raised:
            residency.analyze(receipt)
        self.assertIn("unknown layout", str(raised.exception))

    def test_total_mismatch_is_rejected(self) -> None:
        receipt = build_receipt()
        receipt["totals"]["known_storage_bytes"] += 1
        with self.assertRaises(residency.ResidencyError) as raised:
            residency.analyze(receipt)
        self.assertIn("known_storage_bytes", str(raised.exception))

    def test_wrong_layer_count_is_rejected(self) -> None:
        receipt = build_receipt()
        receipt["tensors"] = [
            item for item in receipt["tensors"]
            if not item["name"].startswith("blk.47.")
        ]
        receipt["totals"]["known_storage_bytes"] = sum(
            item["storage"]["data_bytes"] for item in receipt["tensors"])
        with self.assertRaises(residency.ResidencyError) as raised:
            residency.analyze(receipt)
        self.assertIn("48-layer reference model", str(raised.exception))

    def test_expert_axis_mismatch_is_rejected(self) -> None:
        receipt = build_receipt()
        for item in receipt["tensors"]:
            if item["name"].endswith("ffn_gate_exps.weight"):
                item["shape"] = [640, 2560, 128]
                break
        with self.assertRaises(residency.ResidencyError) as raised:
            residency.analyze(receipt)
        self.assertIn("expert axis", str(raised.exception))

    def test_duplicate_tensor_name_is_rejected(self) -> None:
        receipt = build_receipt()
        receipt["tensors"].append(copy.deepcopy(receipt["tensors"][0]))
        with self.assertRaises(residency.ResidencyError) as raised:
            residency.analyze(receipt)
        self.assertIn("duplicate tensor name", str(raised.exception))


class CommandLineTest(unittest.TestCase):
    def _run(self, receipt: dict, *extra: str) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            return subprocess.run(
                [sys.executable, "-B", str(SCRIPT), "--receipt", str(path), *extra],
                capture_output=True, text=True, check=False,
            )

    def test_json_output_is_parseable(self) -> None:
        completed = self._run(build_receipt(), "--format", "json")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema_version"], "strata-reference-residency-arithmetic-v1")
        self.assertEqual(payload["input"]["tensor_count"], len(build_receipt()["tensors"]))
        self.assertEqual(payload["weights"]["persistent_fixed_bytes"], PERSISTENT_FIXED)

    def test_summary_output_has_no_rate_claim(self) -> None:
        completed = self._run(build_receipt(), "--contexts", "4096")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("s0-all-fixed-resident", completed.stdout)
        lowered = completed.stdout.lower()
        for forbidden in ("tok/s", "tokens/s", "tokens per second", "speedup"):
            self.assertNotIn(forbidden, lowered)

    def test_invalid_receipt_exits_2_with_empty_stdout(self) -> None:
        receipt = build_receipt()
        receipt["tensors"][5]["storage"] = None
        completed = self._run(receipt, "--format", "json")
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        self.assertIn("unknown layout", completed.stderr)

    def test_missing_receipt_exits_2(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--receipt", "/nonexistent/receipt.json"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        self.assertIn("unreadable", completed.stderr)


if __name__ == "__main__":
    unittest.main()