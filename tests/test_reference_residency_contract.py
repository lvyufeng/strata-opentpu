"""Governance checks for the reference-only 4 GiB residency arithmetic record.

These tests never authenticate a receipt, a model artifact or a measurement. They
pin the exact bytes of the reviewed calculator/design/test module as literals so
that a silent repoint must appear as an explicit reviewed diff here as well. The
documented arithmetic is cross-checked against the pinned literals used by the
unit test, so the design prose and the executed assertions cannot drift apart
without failing this module.
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/residency_reference.py"
DESIGN = ROOT / "docs/design/reference-residency-budget.md"
TEST_MODULE = ROOT / "tests/test_residency_reference.py"
README = ROOT / "README.md"
SCRIPTS_README = ROOT / "scripts/README.md"

CENTRAL_ISSUE = "https://github.com/lvyufeng/strata-opentpu/issues/6"
SCHEMA_V2 = "configs/contracts/resource-budget-v2.schema.json"

# Literals, not values derived from the files under test.
SCRIPT_SHA256 = "20d30b10b4b8ba1cb54e99fd3cc99ec9ca1af2d171595b116c5eca9ab7d8baa8"
DESIGN_SHA256 = "38d93dd39cd127cd059888d1a0f124c94a7ad66b00c741dbec84d851e9d9e5a1"
TEST_MODULE_SHA256 = "8acb015ac5bd2b60f1bc672aae1be48c39b8579543b87b58a004b4c0a591ccab"

# Curated declared arithmetic that must stay identical in the design prose and in
# the executed unit test. Values are digit strings with separators removed.
PINNED_ARITHMETIC = {
    "3071375360",     # persistent fixed weights over 48 layers
    "35454976000",    # all expert bytes over 48 layers
    "28800138240",    # oversize per-layer token embedding table
    "688578560",      # fittable global tensors
    "117669888",      # recurrent state total
    "27648",          # KV + indexer bytes per token
    "3192066048",     # S0 resident bytes
    "3880644608",     # S0G resident bytes
    "39890",          # S0 zero-hot max context
    "14985",          # S0G zero-hot max context
    "141744",         # S1/S2 zero-hot max context
    "2519",           # S1 ctx-4096 max-size hot slots
}

# Projection-only literals. They come from accepted v2 single-branch medians plus
# declared routing arithmetic, so they are documented but cannot be asserted by a
# byte calculator; they must still appear in the design prose.
DOC_ONLY_PROJECTION = {
    "750617730",      # serial full-routing token time, ns
    "75061773",       # 10-worker projected token time, ns
    "692480000",      # cold-expert bytes per token at full routing
}

REQUIRED_DESIGN_PHRASES = (
    "**Status: analytical planning only.**",
    CENTRAL_ISSUE,
    "resource-budget-v2.md",
    "../architecture.md",
    "no >10 tok/s claim",
    "Gen2 x8",
    "not a decode",
    "one expert branch",
    "Blocked",
    "read-only",
)

_SAFE_IMPORT_NAME = "residency_reference_contract"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digits(text: str) -> set[str]:
    """Digit runs with underscores and thousands separators removed."""
    return {run.replace(",", "").replace("_", "")
            for run in re.findall(r"\d[\d,_]*", text)}


class PinnedArtifactTest(unittest.TestCase):
    def test_script_bytes_are_pinned(self) -> None:
        self.assertEqual(_sha256(SCRIPT), SCRIPT_SHA256)

    def test_design_bytes_are_pinned(self) -> None:
        self.assertEqual(_sha256(DESIGN), DESIGN_SHA256)

    def test_unit_test_bytes_are_pinned(self) -> None:
        self.assertEqual(_sha256(TEST_MODULE), TEST_MODULE_SHA256)


class DocumentedArithmeticTest(unittest.TestCase):
    def setUp(self) -> None:
        self.design = DESIGN.read_text(encoding="utf-8")
        self.test_source = TEST_MODULE.read_text(encoding="utf-8")

    def test_design_and_test_share_every_declared_number(self) -> None:
        design_digits = _digits(self.design)
        test_digits = _digits(self.test_source)
        for value in sorted(PINNED_ARITHMETIC):
            with self.subTest(value=value):
                self.assertIn(value, design_digits,
                              f"{value} missing from design prose")
                self.assertIn(value, test_digits,
                              f"{value} missing from executed unit test")

    def test_projection_numbers_are_documented_but_marked_as_projection(self) -> None:
        design_digits = _digits(self.design)
        for value in sorted(DOC_ONLY_PROJECTION):
            with self.subTest(value=value):
                self.assertIn(value, design_digits,
                              f"{value} missing from design prose")
        self.assertIn("projection", self.design)
        self.assertIn("unmeasured", self.design)

    def test_required_scope_phrases_are_present(self) -> None:
        for phrase in REQUIRED_DESIGN_PHRASES:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.design)

    def test_design_links_the_v2_schema_and_inputs(self) -> None:
        for link in (
            "../../scripts/residency_reference.py",
            "../../benchmarks/cpu-cold-expert-protocol-v2.md",
            "../../configs/contracts/resource-budget-v2.schema.json",
        ):
            with self.subTest(link=link):
                self.assertIn(link, self.design)
        for local_input in (
            "`local/acquisition/iq2xs-inventory/inventory.v2.json`",
            "`local/acquisition/qwen38flasnext-config.json`",
            "`local/issue5-slices.json`",
        ):
            with self.subTest(local_input=local_input):
                self.assertIn(local_input, self.design)
        self.assertNotIn("../../local/", self.design,
                         "gitignored inputs must not be markdown links")

    def test_design_keeps_the_unknowns_and_limits_sections(self) -> None:
        self.assertIn("## 5. What is unknown", self.design)
        self.assertIn("## 8. Limits", self.design)
        self.assertIn("KV dtype (`bfloat16`)", self.design)
        self.assertIn("cannot be placed in 4 GiB", self.design)

    def test_design_measurement_plan_is_executable_and_read_only(self) -> None:
        for phase in range(6):
            with self.subTest(phase=phase):
                self.assertIn(f"### Phase {phase} ", self.design)
        self.assertIn("`lspci -vv -s <bdf>`", self.design)
        self.assertIn("current_link_speed", self.design)
        self.assertIn("1,510,400", self.design)
        self.assertIn("127,660,416", self.design)
        self.assertIn("write", self.design)
        self.assertIn("separate explicit approval", self.design)


class CalculatorContractTest(unittest.TestCase):
    def setUp(self) -> None:
        spec = importlib.util.spec_from_file_location(_SAFE_IMPORT_NAME, SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module

    def test_schema_version_and_envelope(self) -> None:
        self.assertEqual(self.module.SCHEMA_VERSION,
                         "strata-reference-residency-arithmetic-v1")
        self.assertEqual(self.module.DEVICE_CAPACITY_BYTES, 4 * 1024 ** 3)
        self.assertEqual(self.module.DEVICE_BANK_BYTES, 2 * 1024 ** 3)

    def test_unknowns_are_declared_and_include_the_gating_facts(self) -> None:
        unknowns = list(self.module.UNKNOWNS)
        self.assertEqual(len(unknowns), 10)
        self.assertEqual(len(set(unknowns)), len(unknowns))
        joined = " ".join(unknowns)
        for required in ("capacity", "bank", "pcie", "dma", "mtp"):
            with self.subTest(required=required):
                self.assertIn(required, joined.lower())

    def test_geometry_constants_match_the_documented_arithmetic(self) -> None:
        self.assertEqual(self.module.STATE_BYTES, 117_669_888)
        self.assertEqual(self.module.KV_PER_TOKEN, 27_648)
        self.assertEqual(self.module.FULL_ATTENTION_KV_PER_TOKEN, 24_576)
        self.assertEqual(self.module.INDEXER_KV_PER_TOKEN, 3_072)
        self.assertEqual(self.module.REFERENCE_LAYERS, 48)
        self.assertEqual(self.module.EXPERTS_PER_LAYER, 512)

    def test_summary_does_not_emit_a_rate_claim(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("tokens_per_second", source)
        self.assertNotIn('"tok/s"', source)
        self.assertIn("unknown", source.lower())


class PublicEntryPointTest(unittest.TestCase):
    def test_readmes_advertise_the_calculator_and_limits(self) -> None:
        readme = README.read_text(encoding="utf-8")
        scripts = SCRIPTS_README.read_text(encoding="utf-8")
        for text in (readme, scripts):
            with self.subTest(document=str(text is readme)):
                self.assertIn("scripts/residency_reference.py", text)
                self.assertIn("docs/design/reference-residency-budget.md", text)
        self.assertIn("not a fit, placement, cache hit rate, link speed", readme)
        self.assertIn("not a cache hit rate or a placement", scripts)
        self.assertIn("#6 stays Blocked", readme)
        self.assertIn("## Reference-only 4 GiB residency arithmetic", scripts)


if __name__ == "__main__":
    unittest.main()