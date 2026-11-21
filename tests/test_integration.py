# Copyright (c) 2026 lvyufeng and contributors
# SPDX-License-Identifier: Apache-2.0
"""Hardware/network-free integration safety tests using temporary Git fixtures."""

import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sop", ROOT / "scripts" / "sop.py")
sop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sop)

LOCK = {
    "schema_version": 1,
    "dependencies": {
        "strata": {"repository": "https://github.com/lvyufeng/Strata",
                   "upstream": "https://github.com/Niko1221/Strata",
                   "revision": "3142b66b96e9c0e775e30f620b11b191a1427c2a",
                   "path": "third_party/Strata", "license": "MIT"},
        "opentpu": {"repository": "https://github.com/lvyufeng/openTPU",
                    "upstream": "https://github.com/FeSens/openTPU",
                    "revision": "d84db55712aba0d40811d7762cb052f05d55b7e5",
                    "path": "third_party/openTPU", "license": "Apache-2.0"},
    },
}
PROFILE = {"schema_version": 1, "id": "bootstrap", "status": "bootstrap-only",
           "capabilities": {name: name in ("doctor", "bootstrap") for name in sop.COMMANDS}}


class FixtureMixin:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sop-test-")
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name).resolve()
        self.lock = copy.deepcopy(LOCK)
        self.save_lock()
        (self.workspace / "profile.json").write_text(json.dumps(PROFILE), encoding="utf-8")
        profile_directory = self.workspace / "configs/profiles"
        profile_directory.mkdir(parents=True)
        (profile_directory / "bootstrap.json").write_text(json.dumps(PROFILE), encoding="utf-8")

    def save_lock(self):
        (self.workspace / "dependencies.lock.json").write_text(json.dumps(self.lock), encoding="utf-8")

    def cli(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = sop.main(["--workspace", str(self.workspace), *arguments])
        return result, stdout.getvalue(), stderr.getvalue()

class IntegrationTests(FixtureMixin, unittest.TestCase):
    def test_apply_without_git_does_not_create_dependency_directory(self):
        with mock.patch.object(sop.shutil, "which", return_value=None):
            result, _, error = self.cli("bootstrap", "--apply")
        self.assertEqual(result, 2)
        self.assertIn("Git is required", error)
        self.assertFalse((self.workspace / "third_party").exists())

    def test_existing_git_metadata_symlink_is_rejected(self):
        target = self.workspace / "third_party/Strata"
        target.mkdir(parents=True)
        (target / ".git").symlink_to(self.workspace / "outside", target_is_directory=True)
        with mock.patch.object(sop, "run_git") as git:
            with self.assertRaises(sop.IntegrationError):
                sop.bootstrap(self.workspace, "dependencies.lock.json", True)
            git.assert_not_called()

    def test_bootstrap_default_is_network_free_plan(self):
        with mock.patch.object(sop, "run_git") as git:
            result, output, error = self.cli("bootstrap")
        self.assertEqual(result, 0, error)
        self.assertFalse(json.loads(output)["network_enabled"])
        git.assert_not_called()
        self.assertFalse((self.workspace / "third_party").exists())

    def test_plan_records_both_exact_commits(self):
        report = sop.bootstrap(self.workspace, "dependencies.lock.json", False)
        self.assertEqual([entry["revision"] for entry in report["dependencies"]],
                         [entry["revision"] for entry in LOCK["dependencies"].values()])
        self.assertFalse(report["inference_ready"])

    def test_unsafe_dependency_paths(self):
        for path in ("../Strata", "/tmp/Strata", "third_party/../Strata", "third_party//Strata",
                     "third_party/Strata/", "third_party/Other", "third_party\\Strata"):
            with self.subTest(path=path):
                self.lock["dependencies"]["strata"]["path"] = path
                self.save_lock()
                with self.assertRaises(sop.IntegrationError):
                    sop.bootstrap(self.workspace, "dependencies.lock.json", True)
        self.assertFalse((self.workspace / "third_party").exists())

    def test_unsafe_repository_and_upstream_urls(self):
        for key in ("repository", "upstream"):
            for url in ("file:///tmp/repo", "git@github.com:lvyufeng/Strata", "https://evil.test/repo",
                        "https://github.com/x/y?x=1", "https://github.com/x/y#fragment",
                        "https://user:pass@github.com/x/y", "https://github.com/x/y/extra",
                        "https://github.com/x/y.git", "--upload-pack=evil", "https://github.com/x/y\n"):
                with self.subTest(key=key, url=url):
                    self.lock = copy.deepcopy(LOCK)
                    self.lock["dependencies"]["strata"][key] = url
                    self.save_lock()
                    with mock.patch.object(sop, "run_git") as git:
                        with self.assertRaises(sop.IntegrationError):
                            sop.bootstrap(self.workspace, "dependencies.lock.json", True)
                        git.assert_not_called()

    def test_invalid_credential_url_is_not_echoed(self):
        self.lock["dependencies"]["strata"]["repository"] = "https://user:secret@github.com/x/y"
        self.save_lock()
        result, _, error = self.cli("bootstrap")
        self.assertEqual(result, 2)
        self.assertNotIn("secret", error)
        self.assertNotIn("user:", error)

    def test_unsafe_revisions(self):
        for revision in ("main", "HEAD", "--detach", "a" * 39, "A" * 40, "a" * 40 + "\n", None, 123):
            with self.subTest(revision=revision):
                self.lock["dependencies"]["strata"]["revision"] = revision
                self.save_lock()
                with self.assertRaises(sop.IntegrationError):
                    sop.bootstrap(self.workspace, "dependencies.lock.json", True)

    def test_schema_and_dependency_names_fail_closed(self):
        for data in ({"schema_version": True, "dependencies": LOCK["dependencies"]},
                     {"schema_version": 2, "dependencies": LOCK["dependencies"]},
                     {"schema_version": 1, "dependencies": {"../bad": {}}},
                     {"schema_version": 1, "dependencies": []}):
            with self.subTest(data=data):
                (self.workspace / "dependencies.lock.json").write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(sop.IntegrationError):
                    sop.load_lock(self.workspace, "dependencies.lock.json")

    def test_malformed_json_diagnostic(self):
        (self.workspace / "dependencies.lock.json").write_text("{broken", encoding="utf-8")
        result, _, error = self.cli("bootstrap", "--apply")
        self.assertEqual(result, 2)
        self.assertIn("Cannot read JSON", error)

    def test_lock_file_and_parent_symlinks_are_rejected(self):
        outside = self.workspace / "outside"
        outside.mkdir()
        (outside / "lock.json").write_text(json.dumps(LOCK), encoding="utf-8")
        (self.workspace / "link").symlink_to(outside, target_is_directory=True)
        (self.workspace / "lock-link.json").symlink_to(outside / "lock.json")
        for path in ("link/lock.json", "lock-link.json", "../lock.json"):
            with self.subTest(path=path), self.assertRaises(sop.IntegrationError):
                sop.load_lock(self.workspace, path)

    def test_dependency_parent_symlink_is_rejected(self):
        outside = self.workspace / "outside"
        outside.mkdir()
        (self.workspace / "third_party").symlink_to(outside, target_is_directory=True)
        with mock.patch.object(sop, "run_git") as git:
            with self.assertRaises(sop.IntegrationError):
                sop.bootstrap(self.workspace, "dependencies.lock.json", True)
            git.assert_not_called()
        self.assertEqual(list(outside.iterdir()), [])

    def test_dependency_child_symlink_and_dangling_symlink_rejected(self):
        parent = self.workspace / "third_party"
        parent.mkdir()
        child = parent / "Strata"
        child.symlink_to(self.workspace / "does-not-exist", target_is_directory=True)
        with self.assertRaises(sop.IntegrationError):
            sop.bootstrap(self.workspace, "dependencies.lock.json", True)

    def test_existing_non_checkout_is_not_overwritten(self):
        parent = self.workspace / "third_party"
        parent.mkdir()
        target = parent / "Strata"
        target.mkdir()
        marker = target / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(sop.IntegrationError):
            sop.bootstrap(self.workspace, "dependencies.lock.json", True)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        self.assertFalse((parent / "openTPU").exists())

    def test_regular_file_parent_is_rejected_before_git(self):
        (self.workspace / "third_party").write_text("not a directory", encoding="utf-8")
        with mock.patch.object(sop, "run_git") as git:
            with self.assertRaises(sop.IntegrationError):
                sop.bootstrap(self.workspace, "dependencies.lock.json", True)
            git.assert_not_called()

    def test_unimplemented_apply_fails_without_actions(self):
        for command in sop.COMMANDS[2:]:
            with self.subTest(command=command), mock.patch.object(sop, "run_git") as git:
                result, output, error = self.cli(command, "--apply", "--profile", "profile.json")
                self.assertEqual(result, 2)
                self.assertEqual(output, "")
                self.assertIn("not implemented", error)
                git.assert_not_called()

    def test_unimplemented_plans_are_explicit(self):
        for command in sop.COMMANDS[2:]:
            with self.subTest(command=command):
                result, output, error = self.cli(command)
                self.assertEqual(result, 0, error)
                self.assertFalse(json.loads(output)["implemented"])
                self.assertFalse(json.loads(output)["inference_ready"])

    def test_default_profile_is_validated(self):
        default = self.workspace / "configs/profiles/bootstrap.json"
        profile = copy.deepcopy(PROFILE)
        profile["capabilities"]["program"] = True
        default.write_text(json.dumps(profile), encoding="utf-8")
        for command in ("doctor", "bootstrap", "program"):
            with self.subTest(command=command), mock.patch.object(sop, "run_git") as git:
                result, _, error = self.cli(command)
                self.assertEqual(result, 2)
                self.assertIn("cannot enable", error)
                git.assert_not_called()

    def test_profile_selects_lock_and_explicit_lock_overrides(self):
        profile = copy.deepcopy(PROFILE)
        profile["dependency_lock"] = "selected.lock.json"
        selected = copy.deepcopy(LOCK)
        selected["dependencies"]["strata"]["revision"] = "b" * 40
        (self.workspace / "selected.lock.json").write_text(json.dumps(selected), encoding="utf-8")
        (self.workspace / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
        result, output, error = self.cli("bootstrap", "--profile", "profile.json")
        self.assertEqual(result, 0, error)
        self.assertEqual(json.loads(output)["dependencies"][0]["revision"], "b" * 40)
        result, output, error = self.cli("bootstrap", "--profile", "profile.json",
                                         "--lock", "dependencies.lock.json")
        self.assertEqual(result, 0, error)
        self.assertEqual(json.loads(output)["dependencies"][0]["revision"],
                         LOCK["dependencies"]["strata"]["revision"])

    def test_profile_lock_path_is_validated(self):
        for path in (None, "../outside.lock.json", "/tmp/lock.json"):
            with self.subTest(path=path):
                profile = copy.deepcopy(PROFILE)
                profile["dependency_lock"] = path
                (self.workspace / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
                with mock.patch.object(sop, "run_git") as git:
                    result, _, _ = self.cli("bootstrap", "--profile", "profile.json")
                    self.assertEqual(result, 2)
                    git.assert_not_called()

    def test_profile_cannot_enable_unimplemented_features(self):
        profile = copy.deepcopy(PROFILE)
        profile["capabilities"]["program"] = True
        (self.workspace / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
        result, _, error = self.cli("program", "--apply", "--profile", "profile.json")
        self.assertEqual(result, 2)
        self.assertIn("cannot enable", error)

    def test_profile_commands_are_not_executed(self):
        profile = copy.deepcopy(PROFILE)
        profile["commands"] = {"bootstrap": "touch malicious", "program": ["sh", "-c", "bad"]}
        (self.workspace / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
        with mock.patch.object(sop, "run_git") as git:
            result, output, error = self.cli("bootstrap", "--profile", "profile.json")
        self.assertEqual(result, 0, error)
        self.assertEqual(json.loads(output)["mode"], "plan")
        git.assert_not_called()
        self.assertFalse((self.workspace / "malicious").exists())

    def test_doctor_truthfully_reports_unverified_hardware(self):
        with mock.patch.object(sop.shutil, "which", return_value=None), \
             mock.patch.object(sop.platform, "system", return_value="Darwin"):
            result, output, error = self.cli("doctor", "--profile", "profile.json")
        self.assertEqual(result, 0, error)
        report = json.loads(output)
        self.assertFalse(report["fpga_identity_verified"])
        self.assertFalse(report["inference_ready"])
        self.assertIsNone(report["git"])
        self.assertEqual(report["profile_status"], "bootstrap-only")

    def test_doctor_linux_inventory_is_best_effort(self):
        with mock.patch.object(sop.shutil, "which", return_value=None), \
             mock.patch.object(sop.platform, "system", return_value="Linux"), \
             mock.patch.object(sop, "read_optional", return_value=None), \
             mock.patch.object(Path, "glob", return_value=[]), \
             mock.patch.object(Path, "iterdir", side_effect=OSError("unavailable")):
            report = sop.doctor(self.workspace, None)
        self.assertEqual(report["linux_cpu"], {})
        self.assertIsNone(report["memory_total"])
        self.assertEqual(report["numa_nodes"], [])
        self.assertEqual(report["pci_devices"], [])
        self.assertFalse(report["fpga_identity_verified"])

    def test_git_runner_is_no_shell_noninteractive_and_disables_host_overrides(self):
        completed = subprocess.CompletedProcess([], 0, "git version fixture\n", "")
        with mock.patch.object(sop.shutil, "which", return_value="/usr/bin/git"), \
             mock.patch.dict(os.environ, {"GIT_DIR": "/wrong", "GIT_CONFIG_COUNT": "1"}), \
             mock.patch.object(sop.subprocess, "run", return_value=completed) as run:
            self.assertEqual(sop.run_git(["--version"], self.workspace), "git version fixture")
        arguments, kwargs = run.call_args
        self.assertIsInstance(arguments[0], list)
        self.assertNotIn("shell", kwargs)
        self.assertEqual(kwargs["env"]["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(kwargs["env"]["GIT_ALLOW_PROTOCOL"], "https")
        self.assertEqual(kwargs["env"]["GIT_OPTIONAL_LOCKS"], "0")
        self.assertNotIn("GIT_DIR", kwargs["env"])
        self.assertNotIn("GIT_CONFIG_COUNT", kwargs["env"])

    def test_git_error_and_timeout_have_diagnostics(self):
        for result in (subprocess.CompletedProcess([], 128, "", "fixture failure"),
                       subprocess.TimeoutExpired("git", 1)):
            with self.subTest(result=result), mock.patch.object(sop.shutil, "which", return_value="git"):
                kwargs = {"side_effect": result} if isinstance(result, Exception) else {"return_value": result}
                with mock.patch.object(sop.subprocess, "run", **kwargs):
                    with self.assertRaises(sop.IntegrationError):
                        sop.run_git(["--version"], self.workspace)


@unittest.skipUnless(shutil.which("git"), "local Git fixtures require Git")
class CheckoutTests(FixtureMixin, unittest.TestCase):
    """All clone operations are intercepted; only temporary local git init/commit runs."""

    def git_local(self, arguments, cwd):
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        env.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z"})
        result = subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                                 "-c", f"core.hooksPath={os.devnull}", *arguments], cwd=cwd, env=env,
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()

    def setUp(self):
        super().setUp()
        self.seed = self.workspace / "seed"
        self.seed.mkdir()
        self.git_local(["init", "-b", "main"], self.seed)
        (self.seed / "fixture.txt").write_text("fixture\n", encoding="utf-8")
        self.git_local(["add", "fixture.txt"], self.seed)
        self.git_local(["commit", "-m", "fixture"], self.seed)
        self.revision = self.git_local(["rev-parse", "HEAD"], self.seed)
        for entry in self.lock["dependencies"].values():
            entry["revision"] = self.revision
        self.save_lock()

    def existing(self, name):
        entry = self.lock["dependencies"][name]
        target = self.workspace / entry["path"]
        target.parent.mkdir(exist_ok=True)
        shutil.copytree(self.seed, target)
        self.git_local(["remote", "add", "origin", entry["repository"]], target)
        self.git_local(["checkout", "--detach", self.revision], target)
        return target

    def fake_clone(self, arguments, cwd):
        if arguments[0] == "clone":
            self.assertIn("--no-checkout", arguments)
            self.assertIn("--no-recurse-submodules", arguments)
            repository, destination = arguments[-2:]
            shutil.copytree(self.seed, destination)
            self.git_local(["remote", "add", "origin", repository], Path(destination))
            return ""
        return self.real_git(arguments, cwd)

    def test_apply_and_rerun_are_exact_and_idempotent(self):
        self.real_git = sop.run_git
        with mock.patch.object(sop, "run_git", side_effect=self.fake_clone) as git:
            first = sop.bootstrap(self.workspace, "dependencies.lock.json", True)
            self.assertEqual(sum(call.args[0][0] == "clone" for call in git.call_args_list), 2)
            git.reset_mock()
            second = sop.bootstrap(self.workspace, "dependencies.lock.json", True)
            self.assertFalse(any(call.args[0][0] in ("clone", "checkout") for call in git.call_args_list))
        self.assertEqual(first["mode"], "apply")
        self.assertTrue(all(entry["action"] == "already-verified" for entry in second["dependencies"]))
        for entry in self.lock["dependencies"].values():
            sop.verify_checkout(self.workspace, entry)
            self.assertEqual(self.git_local(["rev-parse", "HEAD"], self.workspace / entry["path"]), self.revision)

    def test_preflight_all_existing_before_any_clone(self):
        # The first dependency is missing, while the second is dirty.
        target = self.existing("opentpu")
        marker = target / "untracked.txt"
        marker.write_text("keep", encoding="utf-8")
        self.real_git = sop.run_git
        with mock.patch.object(sop, "run_git", wraps=self.real_git) as git:
            with self.assertRaisesRegex(sop.IntegrationError, "Dirty"):
                sop.bootstrap(self.workspace, "dependencies.lock.json", True)
            self.assertFalse(any(call.args[0][0] == "clone" for call in git.call_args_list))
        self.assertFalse((self.workspace / "third_party/Strata").exists())
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_tracked_modification_rejected_without_reset(self):
        target = self.existing("strata")
        (target / "fixture.txt").write_text("modified", encoding="utf-8")
        with self.assertRaisesRegex(sop.IntegrationError, "Dirty"):
            sop.bootstrap(self.workspace, "dependencies.lock.json", True)
        self.assertEqual((target / "fixture.txt").read_text(encoding="utf-8"), "modified")

    def test_wrong_origin_rejected_without_reset(self):
        target = self.existing("strata")
        self.git_local(["remote", "set-url", "origin", "https://github.com/other/Strata"], target)
        with self.assertRaisesRegex(sop.IntegrationError, "Wrong origin"):
            sop.bootstrap(self.workspace, "dependencies.lock.json", True)
        self.assertFalse((target.parent / "openTPU").exists())

    def test_wrong_head_rejected_without_reset(self):
        target = self.existing("strata")
        (target / "fixture.txt").write_text("new", encoding="utf-8")
        self.git_local(["add", "fixture.txt"], target)
        self.git_local(["commit", "-m", "another fixture"], target)
        unexpected = self.git_local(["rev-parse", "HEAD"], target)
        with self.assertRaisesRegex(sop.IntegrationError, "Wrong revision"):
            sop.bootstrap(self.workspace, "dependencies.lock.json", True)
        self.assertEqual(self.git_local(["rev-parse", "HEAD"], target), unexpected)

    def test_attached_existing_checkout_rejected_without_checkout(self):
        target = self.existing("strata")
        self.git_local(["checkout", "main"], target)
        with self.assertRaisesRegex(sop.IntegrationError, "not detached"):
            sop.bootstrap(self.workspace, "dependencies.lock.json", True)
        self.assertEqual(self.git_local(["symbolic-ref", "HEAD"], target), "refs/heads/main")
        self.assertFalse((self.workspace / "third_party/openTPU").exists())

    def test_plan_and_rerun_preserve_existing_metadata_bytes(self):
        targets = [self.existing(name) for name in self.lock["dependencies"]]
        metadata = [target / ".git" / name for target in targets
                    for name in ("HEAD", "index", "config")]
        before = {path: path.read_bytes() for path in metadata}
        plan = sop.bootstrap(self.workspace, "dependencies.lock.json", False)
        rerun = sop.bootstrap(self.workspace, "dependencies.lock.json", True)
        self.assertTrue(plan["source_ready"])
        self.assertTrue(rerun["source_ready"])
        self.assertEqual(before, {path: path.read_bytes() for path in metadata})

    def test_conventional_dot_git_origin_suffix_accepted(self):
        target = self.existing("strata")
        self.git_local(["remote", "set-url", "origin", self.lock["dependencies"]["strata"]["repository"] + ".git"], target)
        sop.verify_checkout(self.workspace, self.lock["dependencies"]["strata"])

    def test_failed_new_clone_remains_diagnostic(self):
        def fail_clone(arguments, cwd):
            if arguments[0] == "clone":
                target = Path(arguments[-1])
                target.mkdir()
                (target / "partial.txt").write_text("partial", encoding="utf-8")
                raise sop.IntegrationError("simulated clone failure")
            return self.real_git(arguments, cwd)
        self.real_git = sop.run_git
        with mock.patch.object(sop, "run_git", side_effect=fail_clone):
            with self.assertRaisesRegex(sop.IntegrationError, "simulated clone"):
                sop.bootstrap(self.workspace, "dependencies.lock.json", True)
        self.assertEqual((self.workspace / "third_party/Strata/partial.txt").read_text(encoding="utf-8"), "partial")
        self.assertFalse((self.workspace / "third_party/openTPU").exists())

    def test_checkout_failure_never_cleans_new_clone(self):
        self.real_git = sop.run_git
        def fail_checkout(arguments, cwd):
            if arguments[0] == "checkout":
                raise sop.IntegrationError("simulated checkout failure")
            return self.fake_clone(arguments, cwd)
        with mock.patch.object(sop, "run_git", side_effect=fail_checkout):
            with self.assertRaisesRegex(sop.IntegrationError, "simulated checkout"):
                sop.bootstrap(self.workspace, "dependencies.lock.json", True)
        self.assertTrue((self.workspace / "third_party/Strata/.git").is_dir())
        self.assertFalse((self.workspace / "third_party/openTPU").exists())


if __name__ == "__main__":
    unittest.main()
