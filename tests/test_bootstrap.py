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
            if ".git" in path.relative_to(ROOT).parts:
                continue
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
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

    def test_license(self):
        text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Apache License", text)
        self.assertIn("Version 2.0", text)


if __name__ == "__main__":
    unittest.main()
