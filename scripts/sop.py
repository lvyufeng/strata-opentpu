#!/usr/bin/env python3
# Copyright (c) 2026 lvyufeng and contributors
# SPDX-License-Identifier: Apache-2.0
"""Safe integration entry points. Only doctor and dependency bootstrap work today."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any

COMMANDS = ("doctor", "bootstrap", "build", "program", "smoke-test", "launch", "benchmark")
DEPENDENCIES = {
    "strata": ("third_party/Strata", "MIT"),
    "opentpu": ("third_party/openTPU", "Apache-2.0"),
}
REPOSITORY = re.compile(r"https://github\.com/([A-Za-z0-9][A-Za-z0-9_.-]*)/([A-Za-z0-9][A-Za-z0-9_.-]*)")
REVISION = re.compile(r"[0-9a-f]{40}")


class IntegrationError(Exception):
    """An operation was refused or could not finish safely."""


def workspace_path(workspace: Path, relative: str) -> Path:
    """Validate a workspace-relative path without following child symlinks."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise IntegrationError(f"Invalid relative path: {relative!r}")
    parts = relative.split("/")
    if any(part in ("", ".", "..") for part in parts) or Path(relative).is_absolute():
        raise IntegrationError(f"Unsafe workspace path: {relative!r}")
    candidate = workspace
    for part in parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise IntegrationError(f"Refusing symlink workspace child: {candidate}")
        if candidate.exists() and candidate != workspace / relative and not candidate.is_dir():
            raise IntegrationError(f"Workspace parent is not a directory: {candidate}")
    if not candidate.resolve().is_relative_to(workspace):
        raise IntegrationError(f"Path escapes workspace: {relative!r}")
    return candidate


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as source:
            result = json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IntegrationError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise IntegrationError(f"Expected a JSON object in {path}")
    return result


def validate_repository(value: Any) -> None:
    if not isinstance(value, str) or REPOSITORY.fullmatch(value) is None:
        raise IntegrationError("Only plain GitHub HTTPS repository URLs are allowed (invalid value omitted)")

    if value.endswith(".git"):
        raise IntegrationError("Lock repository URLs must omit the .git suffix")


def load_lock(workspace: Path, relative: str) -> dict[str, dict[str, Any]]:
    data = load_json(workspace_path(workspace, relative))
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise IntegrationError("Unsupported dependency lock schema_version")
    dependencies = data.get("dependencies")
    if not isinstance(dependencies, dict) or set(dependencies) != set(DEPENDENCIES):
        raise IntegrationError("Lock must contain exactly strata and opentpu dependencies")
    for name, entry in dependencies.items():
        if not isinstance(entry, dict):
            raise IntegrationError(f"Invalid dependency entry: {name}")
        expected_path, expected_license = DEPENDENCIES[name]
        if entry.get("path") != expected_path:
            raise IntegrationError(f"{name} must use workspace path {expected_path}")
        workspace_path(workspace, entry["path"])
        validate_repository(entry.get("repository"))
        validate_repository(entry.get("upstream"))
        revision = entry.get("revision")
        if not isinstance(revision, str) or REVISION.fullmatch(revision) is None:
            raise IntegrationError(f"{name} revision must be an exact lowercase 40-character commit")
        if entry.get("license") != expected_license:
            raise IntegrationError(f"Unexpected license metadata for {name}")
    return dependencies


def load_profile(workspace: Path, relative: str | None) -> dict[str, Any] | None:
    if relative is None:
        return None
    data = load_json(workspace_path(workspace, relative))
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise IntegrationError("Unsupported profile schema_version")
    if data.get("id") != "bootstrap" or data.get("status") != "bootstrap-only":
        raise IntegrationError("Only the bootstrap / bootstrap-only profile is supported")
    capabilities = data.get("capabilities")
    expected = {command: command in ("doctor", "bootstrap") for command in COMMANDS}
    if not isinstance(capabilities, dict) or set(capabilities) != set(expected):
        raise IntegrationError("Profile must explicitly declare every CLI capability")
    if any(type(capabilities[key]) is not bool or capabilities[key] != value
           for key, value in expected.items()):
        raise IntegrationError("Profile cannot enable unimplemented capabilities")
    lock_path = data.get("dependency_lock", "dependencies.lock.json")
    workspace_path(workspace, lock_path)
    return data


def run_git(arguments: list[str], cwd: Path) -> str:
    """No shell, submodules, prompts, or repository hooks."""
    if shutil.which("git") is None:
        raise IntegrationError("Git is required; install it separately with authorization")
    # Do not inherit Git directory overrides, injected configuration, credential
    # helpers, URL rewrites, global filters, or transport settings from the host.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update({"GIT_TERMINAL_PROMPT": "0", "GIT_ALLOW_PROTOCOL": "https",
                        "GIT_OPTIONAL_LOCKS": "0",
                        "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
                        "GIT_CONFIG_NOSYSTEM": "1"})
    command = ["git", "-c", f"core.hooksPath={os.devnull}", "-c", "submodule.recurse=false",
               "-c", "core.fsmonitor=false", "-c", "credential.helper=",
               "-c", "protocol.file.allow=never", "-c", "protocol.ext.allow=never", *arguments]
    try:
        result = subprocess.run(command, cwd=cwd, env=environment, text=True,
                                capture_output=True, stdin=subprocess.DEVNULL, timeout=1800,
                                check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise IntegrationError(f"Git operation failed: {exc}") from exc
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise IntegrationError(f"Git exited {result.returncode}: {detail}")
    return result.stdout.strip()


def verify_checkout(workspace: Path, entry: dict[str, Any]) -> None:
    target = workspace_path(workspace, entry["path"])
    if not target.is_dir() or not (target / ".git").is_dir() or (target / ".git").is_symlink():
        raise IntegrationError(f"Existing path is not a standalone Git checkout: {target}")
    toplevel = run_git(["rev-parse", "--show-toplevel"], target)
    if Path(toplevel).resolve() != target.resolve():
        raise IntegrationError(f"Unexpected Git checkout root: {target}")
    origin = run_git(["remote", "get-url", "origin"], target)
    # Accept the conventional .git suffix, but not any different host/path/protocol.
    if origin.removesuffix(".git") != entry["repository"]:
        raise IntegrationError(f"Wrong origin at {target}; expected {entry['repository']}")
    head = run_git(["rev-parse", "HEAD"], target)
    if head != entry["revision"]:
        raise IntegrationError(f"Wrong revision at {target}; expected {entry['revision']}")
    head_reference = run_git(["rev-parse", "--symbolic-full-name", "HEAD"], target)
    if head_reference != "HEAD":
        raise IntegrationError(f"Existing checkout is not detached at {target}; no checkout performed")
    dirty = run_git(["status", "--porcelain=v1", "--untracked-files=all"], target)
    if dirty:
        raise IntegrationError(f"Dirty tracked or untracked files at {target}; no reset performed")


def bootstrap(workspace: Path, lock: str, apply: bool) -> dict[str, Any]:
    dependencies = load_lock(workspace, lock)
    # Preflight the whole set before mkdir, clone, or checkout can mutate anything.
    existing: dict[str, bool] = {}
    for name, entry in dependencies.items():
        target = workspace_path(workspace, entry["path"])
        existing[name] = target.exists()
        if existing[name]:
            verify_checkout(workspace, entry)
    if apply and shutil.which("git") is None:
        raise IntegrationError("Git is required; install it separately with authorization")
    actions = []
    for name, entry in dependencies.items():
        target = workspace_path(workspace, entry["path"])
        action = "already-verified" if existing[name] else "clone-locked-revision"
        actions.append({"dependency": name, "path": entry["path"],
                        "repository": entry["repository"], "revision": entry["revision"],
                        "action": action})
        if apply and not existing[name]:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Recheck immediately before mutation; no overwrite or automatic cleanup.
            target = workspace_path(workspace, entry["path"])
            if target.exists():
                raise IntegrationError(f"Dependency appeared after preflight: {target}")
            run_git(["clone", "--no-checkout", "--no-recurse-submodules", "--",
                     entry["repository"], str(target)], workspace)
            run_git(["checkout", "--detach", entry["revision"]], target)
            verify_checkout(workspace, entry)
    return {"command": "bootstrap", "mode": "apply" if apply else "plan",
            "network_enabled": apply, "source_ready": apply or all(existing.values()),
            "inference_ready": False, "dependencies": actions,
            "note": "Dependencies only; no submodules, builds, installs, models or hardware operations."}


def read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def doctor(workspace: Path, profile: dict[str, Any] | None) -> dict[str, Any]:
    git = shutil.which("git")
    git_version = run_git(["--version"], workspace) if git else None
    report: dict[str, Any] = {
        "command": "doctor", "python": platform.python_version(), "git": git_version,
        "platform": platform.system(), "logical_cpu_count": os.cpu_count(),
        "profile": profile["id"] if profile else None,
        "profile_status": profile["status"] if profile else "not-selected",
        "inference_ready": False, "fpga_identity_verified": False,
        "note": "Read-only host inventory; PCI identifiers do not verify an FPGA design or bitstream.",
    }
    if platform.system() == "Linux":
        cpuinfo = read_optional(Path("/proc/cpuinfo"))
        cpu: dict[str, Any] = {}
        if cpuinfo:
            for line in cpuinfo.splitlines():
                key, separator, value = line.partition(":")
                if separator and key.strip() in ("model name", "flags"):
                    cpu.setdefault(key.strip(), value.strip())
        report["linux_cpu"] = cpu
        meminfo = read_optional(Path("/proc/meminfo"))
        report["memory_total"] = next((line for line in (meminfo or "").splitlines()
                                       if line.startswith("MemTotal:")), None)
        nodes = Path("/sys/devices/system/node")
        try:
            report["numa_nodes"] = [
                {"node": path.name, "cpulist": read_optional(path / "cpulist")}
                for path in sorted(nodes.glob("node[0-9]*")) if path.is_dir()]
        except OSError:
            report["numa_nodes"] = []
        devices = Path("/sys/bus/pci/devices")
        try:
            report["pci_devices"] = [
                {"address": path.name, **{field: read_optional(path / field)
                                         for field in ("vendor", "device", "class", "numa_node")}}
                for path in sorted(devices.iterdir())]
        except OSError:
            report["pci_devices"] = []
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Integration workspace (default: parent of scripts/)")
    commands = result.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        command = commands.add_parser(name)
        command.add_argument("--profile", default="configs/profiles/bootstrap.json",
                             help="Workspace-relative profile JSON (default: configs/profiles/bootstrap.json)")

        if name != "doctor":
            command.add_argument("--apply", action="store_true",
                                 help="Explicit opt-in; currently implemented only for dependency bootstrap")
        if name == "bootstrap":
            command.add_argument("--lock",
                                 help="Workspace-relative lock override (default: profile dependency_lock or dependencies.lock.json)")
    return result


def main(arguments: list[str] | None = None) -> int:
    options = parser().parse_args(arguments)
    try:
        if sys.version_info < (3, 11):
            raise IntegrationError("Python 3.11+ is required")
        workspace = options.workspace.resolve()
        if not workspace.is_dir():
            raise IntegrationError(f"Workspace is not an existing directory: {workspace}")
        profile = load_profile(workspace, options.profile)
        if options.command == "doctor":
            output = doctor(workspace, profile)
        elif options.command == "bootstrap":
            lock = options.lock or (profile or {}).get("dependency_lock", "dependencies.lock.json")
            output = bootstrap(workspace, lock, options.apply)
        else:
            if options.apply:
                raise IntegrationError(f"{options.command} is not implemented; --apply refused")
            output = {"command": options.command, "mode": "plan", "implemented": False,
                      "inference_ready": False,
                      "note": "Placeholder only; no command, model, toolchain or hardware action executed."}
        print(json.dumps(output, indent=2))
        return 0
    except (IntegrationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
