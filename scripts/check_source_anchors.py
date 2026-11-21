# Copyright (c) 2026 lvyufeng and contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate static audit citations; no fetching, building or device operations.

Default mode checks the manifest against the integration lock only. Opt-in
--check-sources also verifies clean, exact-HEAD local checkouts and citation
ranges/excerpts. A valid citation is not proof that its interpretation is true.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]


class AuditError(ValueError):
    """The static audit or its local source inputs cannot be verified."""


def git(directory: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(directory), *arguments],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise AuditError(f"git verification failed: {' '.join(arguments)}")
    return result.stdout.strip()


def collect_anchors(value):
    """Find citation objects, including operator and quantization matrix entries."""
    if isinstance(value, dict):
        if {"repository", "path", "start_line", "end_line"} <= value.keys():
            yield value
        for child in value.values():
            yield from collect_anchors(child)
    elif isinstance(value, list):
        for child in value:
            yield from collect_anchors(child)


def validate(manifest: dict, lock: dict, root: Path, check_sources: bool = False) -> dict:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise AuditError("unsupported audit schema")
    if not isinstance(lock, dict) or not isinstance(lock.get("dependencies"), dict):
        raise AuditError("invalid dependency lock")
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise AuditError("audit must declare exact sources")
    dependencies = lock["dependencies"]
    for name, source in sources.items():
        if name not in dependencies or not isinstance(source, dict):
            raise AuditError("unknown source")
        expected = dependencies[name]
        if not isinstance(expected, dict) or not {"revision", "repository", "path"} <= expected.keys():
            raise AuditError("invalid locked source entry")
        if source.get("revision") != expected["revision"]:
            raise AuditError(f"{name}: audit revision differs from lock")
        if source.get("repository") != expected["repository"]:
            raise AuditError(f"{name}: audit repository differs from lock")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise AuditError("audit must contain entries")
    ids = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"]:
            raise AuditError("each audit entry needs an id")
        if entry["id"] in ids:
            raise AuditError("duplicate audit entry id")
        ids.add(entry["id"])
        if not list(collect_anchors(entry)):
            raise AuditError(f"{entry['id']}: no source anchor")
    anchors = list(collect_anchors(manifest))
    if not anchors:
        raise AuditError("no source anchors")
    checked_sources = {}
    for anchor in anchors:
        name = anchor["repository"]
        if not isinstance(name, str) or name not in sources:
            raise AuditError("anchor references undeclared source")
        path = anchor["path"]
        if not isinstance(path, str) or not path:
            raise AuditError("anchor path must be repo-relative")
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts or "\\" in path or ":" in path or str(relative) != path:
            raise AuditError("unsafe or noncanonical anchor path")
        start, end = anchor["start_line"], anchor["end_line"]
        if type(start) is not int or type(end) is not int or not 1 <= start <= end:
            raise AuditError("invalid anchor line range")
        excerpt = anchor.get("excerpt")
        if excerpt is not None and (not isinstance(excerpt, str) or not excerpt.strip()):
            raise AuditError("excerpt must be a nonempty string")
        if not check_sources:
            continue
        checkout = (root / dependencies[name]["path"]).resolve()
        if not checkout.is_relative_to(root.resolve()) or not checkout.is_dir():
            raise AuditError(f"{name}: locked source directory missing or unsafe")
        if name not in checked_sources:
            if git(checkout, "rev-parse", "--show-toplevel") != str(checkout):
                raise AuditError(f"{name}: not an independent checkout")
            if git(checkout, "rev-parse", "HEAD") != sources[name]["revision"]:
                raise AuditError(f"{name}: local HEAD differs from audit")
            if git(checkout, "status", "--porcelain", "--untracked-files=all"):
                raise AuditError(f"{name}: local checkout is dirty")
            checked_sources[name] = checkout
        target = (checkout / path).resolve()
        if not target.is_relative_to(checkout) or not target.is_file():
            raise AuditError(f"{name}: citation file missing or unsafe: {path}")
        lines = target.read_text(encoding="utf-8").splitlines()
        if end > len(lines):
            raise AuditError(f"{name}/{path}: citation exceeds file length")
        if excerpt is not None and excerpt not in "\n".join(lines[start - 1:end]):
            raise AuditError(f"{name}/{path}: citation excerpt mismatch")
    return {
        "schema_version": 1,
        "entries": len(entries),
        "anchors": len(anchors),
        "source_contents_checked": check_sources,
        "inference_ready": False,
        "note": "Citation consistency only; semantic review and model validation remain separate.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-sources", action="store_true", help="verify existing locked source files; never fetch")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads((ROOT / "data/strata-execution-audit.json").read_text(encoding="utf-8"))
        lock = json.loads((ROOT / "dependencies.lock.json").read_text(encoding="utf-8"))
        result = validate(manifest, lock, ROOT, args.check_sources)
    except (AuditError, OSError, UnicodeError, json.JSONDecodeError) as error:
        parser.exit(1, f"audit verification failed: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
