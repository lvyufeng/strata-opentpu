# Copyright (c) 2026 lvyufeng and contributors
# SPDX-License-Identifier: Apache-2.0
"""Offline citation-validator tests; synthetic git repositories, no inference."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("source_anchors", ROOT / "scripts/check_source_anchors.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class SourceAnchorTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "schema_version": 1,
            "sources": {"strata": {"repository": "https://github.com/example/Strata", "revision": "1" * 40}},
            "entries": [{"id": "example", "anchors": [{
                "repository": "strata", "path": "src/model.cpp", "start_line": 1, "end_line": 2,
                "excerpt": "example",
            }]}],
        }
        self.lock = {"dependencies": {"strata": {
            **self.manifest["sources"]["strata"], "path": "third_party/Strata",
        }}}

    def validate(self, check=False, root=ROOT):
        return CHECK.validate(self.manifest, self.lock, root, check)

    def test_metadata_mode_does_not_need_checkouts(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.validate(root=Path(directory))
        self.assertEqual(result["anchors"], 1)
        self.assertFalse(result["source_contents_checked"])
        self.assertFalse(result["inference_ready"])

    def test_reject_invalid_root_schema(self):
        for manifest in ([], None, {"schema_version": 2}):
            with self.subTest(manifest=manifest):
                with self.assertRaises(CHECK.AuditError):
                    CHECK.validate(manifest, self.lock, ROOT)
        with self.assertRaises(CHECK.AuditError):
            CHECK.validate(self.manifest, {"dependencies": []}, ROOT)
        self.lock["dependencies"]["strata"] = {}
        with self.assertRaises(CHECK.AuditError):
            self.validate()

    def test_source_mode_requires_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(CHECK.AuditError):
                self.validate(True, Path(directory))

    def test_revision_and_repository_must_match_lock(self):
        for key in ("revision", "repository"):
            with self.subTest(key=key):
                old = self.manifest["sources"]["strata"][key]
                self.manifest["sources"]["strata"][key] = "wrong"
                with self.assertRaises(CHECK.AuditError):
                    self.validate()
                self.manifest["sources"]["strata"][key] = old

    def test_reject_missing_and_duplicate_entries(self):
        for entries in ([], [{"id": "missing"}], self.manifest["entries"] * 2):
            with self.subTest(entries=entries):
                data = copy.deepcopy(self.manifest)
                data["entries"] = entries
                with self.assertRaises(CHECK.AuditError):
                    CHECK.validate(data, self.lock, ROOT)

    def test_reject_unsafe_paths_and_ranges(self):
        anchor = self.manifest["entries"][0]["anchors"][0]
        for path in ("../escape", "/absolute", "src/../outside", "src\\model.cpp", "src//model.cpp", "C:drive", ""):
            with self.subTest(path=path):
                anchor["path"] = path
                with self.assertRaises(CHECK.AuditError):
                    self.validate()
        anchor["path"] = "src/model.cpp"
        for start, end in ((0, 1), (2, 1), (True, 2), (1, 2.5)):
            with self.subTest(start=start, end=end):
                anchor.update(start_line=start, end_line=end)
                with self.assertRaises(CHECK.AuditError):
                    self.validate()

    def test_reject_undeclared_source_and_empty_excerpt(self):
        anchor = self.manifest["entries"][0]["anchors"][0]
        anchor["repository"] = "unknown"
        with self.assertRaises(CHECK.AuditError):
            self.validate()
        anchor["repository"] = "strata"
        anchor["excerpt"] = " "
        with self.assertRaises(CHECK.AuditError):
            self.validate()

    def test_nested_format_anchors_counted(self):
        self.manifest["format_matrix"] = [{"anchors": [copy.deepcopy(self.manifest["entries"][0]["anchors"][0])]}]
        self.assertEqual(self.validate()["anchors"], 2)

    def test_pinned_clean_sources_ranges_and_excerpts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / "third_party/Strata"
            checkout.mkdir(parents=True)
            def git(*args):
                return subprocess.run(["git", "-C", str(checkout), *args], check=True,
                                      capture_output=True, text=True).stdout.strip()
            git("init", "--quiet")
            git("config", "user.name", "Citation test")
            git("config", "user.email", "test@example.invalid")
            (checkout / "src").mkdir()
            source = checkout / "src/model.cpp"
            source.write_text("example\nline two\n", encoding="utf-8")
            git("add", ".")
            git("commit", "--quiet", "-m", "fixture")
            revision = git("rev-parse", "HEAD")
            self.manifest["sources"]["strata"]["revision"] = revision
            self.lock["dependencies"]["strata"]["revision"] = revision
            self.assertTrue(self.validate(True, root)["source_contents_checked"])
            anchor = self.manifest["entries"][0]["anchors"][0]
            anchor["end_line"] = 3
            with self.assertRaisesRegex(CHECK.AuditError, "file length"):
                self.validate(True, root)
            anchor["end_line"] = 2
            anchor["excerpt"] = "not in source"
            with self.assertRaisesRegex(CHECK.AuditError, "excerpt mismatch"):
                self.validate(True, root)
            anchor["excerpt"] = "example"
            source.write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(CHECK.AuditError, "dirty"):
                self.validate(True, root)
            source.write_text("example\nline two\n", encoding="utf-8")
            self.manifest["sources"]["strata"]["revision"] = "2" * 40
            self.lock["dependencies"]["strata"]["revision"] = "2" * 40
            with self.assertRaisesRegex(CHECK.AuditError, "HEAD"):
                self.validate(True, root)


if __name__ == "__main__":
    unittest.main()
