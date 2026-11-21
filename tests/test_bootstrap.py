# Copyright (c) 2026 lvyufeng and contributors
# SPDX-License-Identifier: Apache-2.0
"""Hardware-free metadata checks, not inference tests."""
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BootstrapTests(unittest.TestCase):
    def test_upstream_references(self):
        data = json.loads((ROOT / "upstreams.json").read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["purpose"], "planned-integration-references")
        self.assertRegex(data["captured_on"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertEqual(set(data["projects"]), {"strata", "opentpu"})
        for entry in data["projects"].values():
            self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$")
            self.assertRegex(entry["repository"], r"^https://github\.com/[^/]+/[^/]+$")
            self.assertIn(entry["license"], {"MIT", "Apache-2.0"})
            self.assertEqual(entry["status"], "not-integrated")

    def test_documentation_links(self):
        for path in ROOT.rglob("*.md"):
            relative = path.relative_to(ROOT)
            if any(part in {".git", "third_party", ".venv", "venv", "local", "private", "results", "logs", "build", "dist", ".cache"} for part in relative.parts):
                continue
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                target = target.strip().removeprefix("<").removesuffix(">")
                if target.startswith(("https://", "http://", "#", "mailto:")):
                    continue
                dest = (path.parent / target.split("#", 1)[0]).resolve()
                with self.subTest(path=str(path.relative_to(ROOT)), target=target):
                    self.assertTrue(dest.is_relative_to(ROOT))
                    self.assertTrue(dest.is_file())

    def test_scope_is_explicit(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Status: project bootstrap only", text)
        self.assertIn("target, not a result", text)
        self.assertIn("not included or downloaded", text)

    def test_dependency_lock(self):
        lock = json.loads((ROOT / "dependencies.lock.json").read_text(encoding="utf-8"))
        refs = json.loads((ROOT / "upstreams.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["schema_version"], 1)
        self.assertEqual(lock["compatibility_status"], "not-validated-for-inference")
        self.assertEqual(set(lock["dependencies"]), {"strata", "opentpu"})
        for name, directory in (("strata", "Strata"), ("opentpu", "openTPU")):
            entry = lock["dependencies"][name]
            self.assertEqual(entry["repository"], f"https://github.com/lvyufeng/{directory}")
            self.assertEqual(entry["path"], f"third_party/{directory}")
            self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$")
            self.assertEqual(entry["upstream"], refs["projects"][name]["repository"])
            self.assertEqual(entry["license"], refs["projects"][name]["license"])

    def test_reference_profiles_fail_closed(self):
        profile = json.loads((ROOT / "configs/profiles/bootstrap.json").read_text(encoding="utf-8"))
        self.assertEqual(profile["schema_version"], 1)
        self.assertEqual(profile["status"], "bootstrap-only")
        self.assertFalse(profile["validated_combination"])
        self.assertEqual(profile["dependency_lock"], "dependencies.lock.json")
        self.assertEqual({key for key, enabled in profile["capabilities"].items() if enabled}, {"doctor", "bootstrap"})
        for item in ("board", "bitstream", "model"):
            self.assertIsNone(profile[item])
        board = json.loads((ROOT / "configs/boards/kintex7-reference.json").read_text(encoding="utf-8"))
        self.assertIsNone(board["device_identity"])
        self.assertIsNone(board["programming_transport"])
        model = json.loads((ROOT / "configs/models/qwen38-flash-next-iq2-xs-reference.json").read_text(encoding="utf-8"))
        self.assertIsNone(model["revision"])
        self.assertEqual(model["files"], [])
        self.assertEqual(model["routed_top_k"], 10)
        self.assertEqual(model["expert_skip"], 0)

    def test_three_repository_decision(self):
        old = (ROOT / "docs/adr/0001-independent-repository.md").read_text(encoding="utf-8")
        new = (ROOT / "docs/adr/0002-three-repository-integration.md").read_text(encoding="utf-8")
        self.assertIn("superseded", old.lower())
        self.assertIn("accepted", new.lower())
        self.assertIn("https://github.com/lvyufeng/Strata", new)
        self.assertIn("https://github.com/lvyufeng/openTPU", new)

    def test_development_tracking_contract(self):
        contract = (ROOT / "docs/development.md").read_text(encoding="utf-8")
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        project = "https://github.com/users/lvyufeng/projects/11"
        issues = "https://github.com/lvyufeng/strata-opentpu/issues"
        for text in (contract, agents, readme):
            self.assertIn(project, text)
            self.assertIn(issues, text)
        for state in ("Backlog", "Ready", "In Progress", "Blocked", "Awaiting approval", "Done"):
            self.assertIn(state, contract)
        self.assertIn("blocked_by", contract)
        self.assertIn("accepted output", contract)
        self.assertIn("Refs https://github.com/lvyufeng/strata-opentpu/issues/N", contract)
        self.assertIn("not another live", contract)
        self.assertIn("At every session end", agents)

    def test_development_templates(self):
        issue = (ROOT / ".github/ISSUE_TEMPLATE/development-task.md").read_text(encoding="utf-8")
        pr = (ROOT / ".github/pull_request_template.md").read_text(encoding="utf-8")
        for heading in ("Dependencies", "Acceptance", "permission", "Evidence"):
            self.assertIn(heading, issue)
        self.assertIn("type:task", issue)
        self.assertIn("projects/11", issue)
        self.assertIn("Refs https://github.com/lvyufeng/strata-opentpu/issues/N", pr)
        self.assertIn("central acceptance", pr)

    def test_license(self):
        text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Apache License", text)
        self.assertIn("Version 2.0", text)


if __name__ == "__main__":
    unittest.main()
