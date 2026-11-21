# Copyright (c) 2026 lvyufeng and contributors
# SPDX-License-Identifier: Apache-2.0
"""Check the packed-oracle receipt; never fetch, build, or access a device.

Default mode reads integration metadata only. --fixture checks a supplied JSON
fixture's identity/integrity, not its arithmetic. --oracle-root explicitly runs
one fixed stdlib CLI from an existing clean checkout at the recorded exact commit.
This is trusted source execution, not a hardware or operating-system sandbox.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs/contracts/opentpu-packed-oracle.json"
ARITHMETIC_ID = "strata-generic-f32-rne-separate-v1"
FIXTURE_PATH = "tools/packed_oracle/data/q2_q8_synthetic.json"
WORKFLOW_PATH = ".github/workflows/packed-oracle.yml"
WORKFLOW_NAME = "Packed-byte oracle checks"
MODULE = "tools.packed_oracle"
FORMATS = {
    "Q2_0": {"ggml_type": 42, "block_elements": 64, "block_bytes": 18,
             "activation_format": "Q8_0", "byte_order": "little"},
    "Q8_0": {"ggml_type": 8, "block_elements": 32, "block_bytes": 34,
             "activation_format": "Q8_0", "byte_order": "little"},
}
LIMITS = {"max_elements": 65536, "max_fixture_bytes": 8388608, "max_cases": 256}
ANCHORS = [
    {"path": "engine/src/quant/dequant.cpp", "start_line": 25, "end_line": 43},
    {"path": "engine/src/quant/dequant.cpp", "start_line": 139, "end_line": 153},
    {"path": "engine/src/quant/dot_generic.cpp", "start_line": 22, "end_line": 63},
    {"path": "engine/src/quant/dot_generic.cpp", "start_line": 203, "end_line": 226},
    {"path": "engine/include/strata/quant/strata-quant-common.h", "start_line": 195, "end_line": 201},
    {"path": "engine/include/strata/quant/strata-quant-common.h", "start_line": 266, "end_line": 272},
]
PROVENANCE = {
    "strata_anchors": ANCHORS,
    "ggml_origin_revision": "3cf03257f219afbe7334045ff7c6a06ac68c627d",
    "generator": {"id": "fraction-integer-reference", "version": 1},
    "expected_derivation": "independent exact rational binary32 RNE; no DUT results",
    "scope_exclusions": ["real-model coverage", "native CPU or AVX parity",
                         "RTL or FPGA parity", "DMA or device operations",
                         "quantization or model loading", "performance or whole decode"],
}
LICENSES = [
    {"id": "strata", "license": "MIT", "path": "tools/packed_oracle/licenses/STRATA-MIT.txt",
     "source_path": "LICENSE", "sha256": "949b9b0aada7476006469e58ee4ffb0e0f68c2d9137e90a356757b61a09d37f0"},
    {"id": "ggml", "license": "MIT", "path": "tools/packed_oracle/licenses/GGML-MIT.txt",
     "source_path": "engine/third_party/strata-ggml/LICENSE",
     "sha256": "94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d"},
]
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_FP32 = re.compile(r"0x[0-9a-f]{8}\Z")
_HEX = re.compile(r"(?:[0-9a-f]{2})+\Z")
_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")
_MAX_JSON_DEPTH = 64
_MAX_NUMBER_CHARS = 64


class ContractError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _require(condition: bool, message: str, code: str = "invalid_contract") -> None:
    if not condition:
        raise ContractError(code, message)


def _object(value, keys: set[str], label: str) -> None:
    _require(type(value) is dict and set(value) == keys, f"{label}: unexpected schema fields")


def _same(value, expected) -> bool:
    """JSON equality that does not confuse True with 1."""
    if type(value) is not type(expected):
        return False
    if type(value) is dict:
        return value.keys() == expected.keys() and all(_same(value[k], expected[k]) for k in value)
    if type(value) is list:
        return len(value) == len(expected) and all(_same(a, b) for a, b in zip(value, expected))
    return value == expected


def _fixed(value, expected, label: str) -> None:
    _require(_same(value, expected), f"{label}: does not match the frozen contract")


def _hash(value, length: int, label: str, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    pattern = _HEX40 if length == 40 else _HEX64
    _require(type(value) is str and pattern.fullmatch(value) is not None, f"{label}: invalid digest/revision")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("invalid_json", "duplicate JSON object key")
        result[key] = value
    return result


def _nonfinite(_value):
    raise ContractError("invalid_json", "non-finite JSON number")


def _bounded_int(text: str) -> int:
    _require(len(text) <= _MAX_NUMBER_CHARS, "JSON integer exceeds the literal limit", "invalid_json")
    return int(text)


def _bounded_float(text: str) -> float:
    _require(len(text) <= _MAX_NUMBER_CHARS, "JSON number exceeds the literal limit", "invalid_json")
    value = float(text)
    _require(math.isfinite(value), "non-finite JSON number", "invalid_json")
    return value


def _check_json_depth(text: str) -> None:
    # Bound nesting before the decoder; quoted/escaped brackets are not syntax.
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            _require(depth <= _MAX_JSON_DEPTH, "JSON nesting exceeds the depth limit", "invalid_json")
        elif char in "]}":
            depth -= 1


def parse_json(raw: bytes):
    _require(type(raw) is bytes and len(raw) <= LIMITS["max_fixture_bytes"], "invalid JSON byte input or byte limit", "invalid_json")
    try:
        text = raw.decode("utf-8")
        _check_json_depth(text)
        return json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=_nonfinite,
                          parse_int=_bounded_int, parse_float=_bounded_float)
    except (UnicodeError, ValueError, RecursionError) as error:
        if isinstance(error, ContractError):
            raise
        raise ContractError("invalid_json", "invalid UTF-8 JSON input") from error


def read_json(path: Path, limit: int = LIMITS["max_fixture_bytes"]) -> tuple[dict, str]:
    try:
        _require(path.is_file(), "JSON input must be an existing regular file", "input_io")
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        _require(len(raw) <= limit, "JSON input exceeds the byte limit", "input_io")
    except OSError as error:
        raise ContractError("input_io", "cannot read JSON input") from error
    return parse_json(raw), hashlib.sha256(raw).hexdigest()


def canonical_digest(document: dict) -> str:
    try:
        payload = {key: value for key, value in document.items() if key != "payload_sha256"}
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, RecursionError) as error:
        raise ContractError("invalid_fixture", "fixture is not canonical JSON data") from error
    return hashlib.sha256(raw).hexdigest()


def validate(contract: dict, lock: dict) -> dict:
    """Validate a recorded receipt, not the truth of CI or review assertions."""
    _object(contract, {"schema_version", "contract_id", "evidence_scope", "central_issue", "arithmetic_id",
                       "sources", "formats", "limits", "fixture", "oracle", "licenses", "provenance", "acceptance"}, "contract")
    for name, expected in {
        "schema_version": 1, "contract_id": "opentpu-q2-q8-packed-synthetic-v1",
        "evidence_scope": "supplied-packed-synthetic",
        "central_issue": "https://github.com/lvyufeng/strata-opentpu/issues/17",
        "arithmetic_id": ARITHMETIC_ID, "formats": FORMATS, "limits": LIMITS,
        "licenses": LICENSES, "provenance": PROVENANCE,
    }.items():
        _fixed(contract[name], expected, name)
    _require(type(lock) is dict and type(lock.get("dependencies")) is dict, "invalid bootstrap source lock")
    _fixed(lock.get("schema_version"), 1, "lock schema")
    _fixed(lock.get("purpose"), "bootstrap-source-lock", "lock purpose")
    _fixed(lock.get("compatibility_status"), "not-validated-for-inference", "lock status")
    _object(contract["sources"], {"strata", "opentpu_base"}, "sources")
    for name, lock_name in (("strata", "strata"), ("opentpu_base", "opentpu")):
        source = contract["sources"][name]
        _object(source, {"repository", "revision"}, name)
        entry = lock["dependencies"].get(lock_name)
        _require(type(entry) is dict, "source is missing from the bootstrap lock")
        expected_repo = f"https://github.com/lvyufeng/{'Strata' if name == 'strata' else 'openTPU'}"
        _fixed(source["repository"], expected_repo, f"{name} repository")
        _hash(source["revision"], 40, f"{name} revision")
        _fixed(source, {"repository": entry.get("repository"), "revision": entry.get("revision")}, f"{name} lock identity")
    fixture = contract["fixture"]
    _object(fixture, {"path", "fixture_id", "payload_sha256", "file_sha256", "case_count", "decode_cases", "dot_cases"}, "fixture receipt")
    _fixed(fixture["path"], FIXTURE_PATH, "fixture path")
    _fixed(fixture["fixture_id"], "q2-q8-supplied-packed-v1", "fixture ID")
    for key in ("payload_sha256", "file_sha256"):
        _hash(fixture[key], 64, key, nullable=True)
    for key in ("case_count", "decode_cases", "dot_cases"):
        count = fixture[key]
        _require(count is None or (type(count) is int and 1 <= count <= LIMITS["max_cases"]), f"{key}: invalid count")
    counts = [fixture[key] for key in ("case_count", "decode_cases", "dot_cases")]
    if all(count is not None for count in counts):
        _require(counts[0] == counts[1] + counts[2], "case counts do not add up")
    oracle = contract["oracle"]
    _object(oracle, {"repository", "revision", "module", "workflow_path", "workflow_name"}, "oracle receipt")
    _fixed(oracle["repository"], contract["sources"]["opentpu_base"]["repository"], "oracle repository")
    _hash(oracle["revision"], 40, "oracle revision", nullable=True)
    _require(oracle["revision"] is None or oracle["revision"] != contract["sources"]["opentpu_base"]["revision"], "oracle revision is still the unmodified base")
    _fixed(oracle["module"], MODULE, "oracle module")
    _fixed(oracle["workflow_path"], WORKFLOW_PATH, "oracle workflow path")
    _fixed(oracle["workflow_name"], WORKFLOW_NAME, "oracle workflow name")
    acceptance = contract["acceptance"]
    _object(acceptance, {"kind", "local_python_versions", "independent_math_review", "independent_scope_license_review", "ci"}, "acceptance")
    _fixed(acceptance["kind"], "reviewed-synthetic-source-preparation", "acceptance kind")
    _fixed(acceptance["local_python_versions"], ["3.11", "3.14"], "local Python matrix")
    for key in ("independent_math_review", "independent_scope_license_review"):
        _require(type(acceptance[key]) is bool, f"{key}: must be an explicit boolean")
    ci = acceptance["ci"]
    _object(ci, {"url", "head_sha", "workflow", "event", "conclusion", "python_versions"}, "CI receipt")
    _hash(ci["head_sha"], 40, "CI head", nullable=True)
    _fixed(ci["head_sha"], oracle["revision"], "CI head must identify the oracle commit")
    _fixed(ci["workflow"], WORKFLOW_NAME, "CI workflow")
    _fixed(ci["event"], "push", "CI event")
    _fixed(ci["python_versions"], ["3.11", "3.14"], "CI Python matrix")
    _require(ci["url"] is None or (type(ci["url"]) is str and re.fullmatch(r"https://github\.com/lvyufeng/openTPU/actions/runs/[1-9][0-9]*", ci["url"]) is not None), "CI URL does not identify the public fork run")
    _require(ci["conclusion"] in (None, "success"), "recorded oracle CI did not pass")
    missing = [f"fixture.{key}" for key in ("payload_sha256", "file_sha256", "case_count", "decode_cases", "dot_cases") if fixture[key] is None]
    if oracle["revision"] is None:
        missing.append("oracle.revision")
    missing.extend(f"acceptance.{key}" for key in ("independent_math_review", "independent_scope_license_review") if not acceptance[key])
    missing.extend(f"acceptance.ci.{key}" for key in ("url", "head_sha", "conclusion") if ci[key] is None)
    return {
        "schema_version": 1, "contract_id": contract["contract_id"], "evidence_scope": contract["evidence_scope"],
        "arithmetic_id": ARITHMETIC_ID, "receipt_complete": not missing, "missing": missing,
        "oracle_revision": oracle["revision"], "fixture_payload_sha256": fixture["payload_sha256"],
        "fixture_metadata_checked": False, "source_identity_checked": False, "oracle_executed": False,
        "synthetic_vectors_verified": False, "device_ready": False, "inference_ready": False,
        "note": "Recorded receipt consistency only; CI/review truth, native/RTL parity and real-model performance are not certified.",
    }


def _payload(value, blocks: int, block_bytes: int, label: str) -> None:
    _require(type(value) is str and len(value) == blocks * block_bytes * 2 and _HEX.fullmatch(value) is not None,
             f"{label}: invalid canonical packed hex length", "invalid_fixture")


def _bits(value, label: str) -> None:
    _require(type(value) is str and _FP32.fullmatch(value) is not None, f"{label}: invalid FP32 bits", "invalid_fixture")


def validate_fixture(document: dict, contract: dict, file_sha256: str | None = None) -> dict:
    """Validate only fixture schema, counts, provenance and digests; no dot/decode."""
    _object(document, {"schema_version", "fixture_id", "synthetic", "arithmetic_id", "formats", "provenance", "cases", "payload_sha256"}, "fixture")
    for name, expected in {"schema_version": 1, "fixture_id": contract["fixture"]["fixture_id"],
                           "synthetic": True, "arithmetic_id": ARITHMETIC_ID, "formats": FORMATS}.items():
        _fixed(document[name], expected, f"fixture {name}")
    expected_provenance = {
        "strata": {**contract["sources"]["strata"], "anchors": ANCHORS},
        "opentpu_base": contract["sources"]["opentpu_base"],
        **{key: value for key, value in PROVENANCE.items() if key != "strata_anchors"},
    }
    _fixed(document["provenance"], expected_provenance, "fixture provenance")
    _hash(document["payload_sha256"], 64, "fixture payload digest")
    _fixed(document["payload_sha256"], canonical_digest(document), "fixture canonical digest")
    if contract["fixture"]["payload_sha256"] is not None:
        _fixed(document["payload_sha256"], contract["fixture"]["payload_sha256"], "pinned fixture payload digest")
    if file_sha256 is not None and contract["fixture"]["file_sha256"] is not None:
        _fixed(file_sha256, contract["fixture"]["file_sha256"], "pinned fixture file digest")
    cases = document["cases"]
    _require(type(cases) is list and 2 <= len(cases) <= LIMITS["max_cases"], "fixture: invalid case count", "invalid_fixture")
    seen, counts = set(), {"decode": 0, "dot": 0}
    for case in cases:
        _require(type(case) is dict, "fixture case must be an object", "invalid_fixture")
        operation = case.get("operation")
        _require(type(operation) is str and operation in counts, "fixture: unsupported operation", "invalid_fixture")
        keys = {"id", "operation", "format", "elements", "payload_hex", "expected_fp32_bits"} if operation == "decode" else {"id", "operation", "weight_format", "activation_format", "elements", "weight_hex", "activation_hex", "expected_fp32_bits", "expected_integer_subdots"}
        _object(case, keys, "fixture case")
        case_id = case["id"]
        _require(type(case_id) is str and _ID.fullmatch(case_id) is not None and case_id not in seen, "fixture: invalid/duplicate case ID", "invalid_fixture")
        seen.add(case_id)
        name = case["format" if operation == "decode" else "weight_format"]
        _require(type(name) is str and name in FORMATS, "fixture: unsupported format", "invalid_fixture")
        elements, spec = case["elements"], FORMATS[name]
        _require(type(elements) is int and 0 < elements <= LIMITS["max_elements"] and elements % spec["block_elements"] == 0, "fixture: invalid element count", "invalid_fixture")
        blocks = elements // spec["block_elements"]
        if operation == "decode":
            _payload(case["payload_hex"], blocks, spec["block_bytes"], "decode payload")
            bits = case["expected_fp32_bits"]
            _require(type(bits) is list and len(bits) == elements, "decode expected length differs from elements", "invalid_fixture")
            for value in bits:
                _bits(value, "decode expected")
        else:
            _fixed(case["activation_format"], "Q8_0", "fixture activation format")
            _payload(case["weight_hex"], blocks, spec["block_bytes"], "weight payload")
            _payload(case["activation_hex"], elements // 32, 34, "activation payload")
            _bits(case["expected_fp32_bits"], "dot expected")
            subdots = case["expected_integer_subdots"]
            _require(type(subdots) is list and len(subdots) == blocks, "integer subdot block count mismatch", "invalid_fixture")
            width, low, high = (2, -8192, 8128) if name == "Q2_0" else (1, -520192, 524288)
            for block in subdots:
                _require(type(block) is list and len(block) == width and all(type(value) is int and low <= value <= high for value in block), "invalid integer subdots", "invalid_fixture")
        counts[operation] += 1
    _require(all(counts.values()), "fixture must contain both decode and dot", "invalid_fixture")
    actual_counts = {"case_count": len(cases), "decode_cases": counts["decode"], "dot_cases": counts["dot"]}
    for key, count in actual_counts.items():
        if contract["fixture"][key] is not None:
            _fixed(count, contract["fixture"][key], f"fixture {key}")
    return {"fixture_metadata_checked": True, "fixture_payload_sha256": document["payload_sha256"], **actual_counts}


def _safe_file(checkout: Path, relative: str) -> Path:
    target = checkout / relative
    _require(target.is_relative_to(checkout), "source path escapes the checkout", "unsafe_source")
    for part in [target, *target.parents]:
        if part == checkout.parent:
            break
        _require(not part.is_symlink(), "source paths must not use symlinks", "unsafe_source")
    _require(target.is_file(), "required oracle source file is missing", "unsafe_source")
    return target


def _clean_environment() -> dict[str, str]:
    # Ignore inherited Git replacement/worktree/config settings and Python import
    # hooks. -E -S additionally disables Python environment and site processing.
    env = {key: value for key, value in os.environ.items() if not key.startswith(("GIT_", "PYTHON", "LD_", "DYLD_"))}
    env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_NO_REPLACE_OBJECTS": "1", "GIT_NO_LAZY_FETCH": "1", "GIT_OPTIONAL_LOCKS": "0"})
    return env


def _run(command: list[str], cwd: Path, *, timeout: int = 15):
    try:
        result = subprocess.run(command, cwd=cwd, env=_clean_environment(), capture_output=True,
                                text=False, check=False, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ContractError("source_execution_failed", "fixed source verification command failed or timed out") from error
    _require(result.returncode == 0, "fixed source verification command returned nonzero", "source_execution_failed")
    _require(len(result.stdout) <= LIMITS["max_fixture_bytes"], "source verification output exceeds the byte limit", "source_execution_failed")
    return result.stdout


def _git(checkout: Path, *arguments: str) -> str:
    raw = _run(["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=" + os.devnull,
                "-c", "core.untrackedCache=false", "-C", str(checkout), *arguments], checkout)
    try:
        return raw.decode("utf-8").strip()
    except UnicodeError as error:
        raise ContractError("unsafe_source", "Git verification output is not UTF-8") from error


def _check_no_promisor(checkout: Path) -> None:
    """Refuse partial clones before object lookup, even on older Git versions."""
    configuration = _git(checkout, "config", "--null", "--list")
    for record in configuration.split("\0"):
        key = record.partition("\n")[0].lower()
        _require(key != "extensions.partialclone" and re.fullmatch(r"remote\..+\.(?:promisor|partialclonefilter)", key) is None,
                 "partial-clone/promisor source checkouts are not supported", "unsafe_source")


def _tracked_records(text: str, *, index: bool) -> dict[str, tuple[str, str]]:
    """Parse NUL-delimited Git metadata; reject index flags that hide changes."""
    records = text.split("\0")
    _require(len(records) > 1 and records[-1] == "", "invalid or empty tracked source listing", "unsafe_source")
    result = {}
    for record in records[:-1]:
        metadata, separator, path = record.partition("\t")
        fields = metadata.split(" ")
        expected = 4 if index else 3
        _require(separator == "\t" and len(fields) == expected and path not in result,
                 "invalid tracked source entry", "unsafe_source")
        if index:
            flag, mode, digest, stage = fields
            _require(flag == "H" and stage == "0", "oracle index has hidden, sparse or unmerged entries", "unsafe_source")
        else:
            mode, kind, digest = fields
            _require(kind == "blob", "oracle tree contains non-blob entries", "unsafe_source")
        _require(mode in {"100644", "100755"} and _HEX40.fullmatch(digest) is not None,
                 "oracle tree requires ordinary tracked files", "unsafe_source")
        _require(path and not path.startswith("/") and all(part not in {"", ".", ".."} for part in path.split("/")),
                 "invalid tracked source path", "unsafe_source")
        result[path] = mode, digest
    return result


def _check_tracked_bytes(checkout: Path, revision: str) -> None:
    """Compare index and actual bytes to commit blobs, without invoking filters.

    Git status alone can hide assume-unchanged/skip-worktree edits or consult a
    local clean filter. Hash every ordinary tracked file, including Python parent
    packages and potential import shadows, not just the named oracle entry point.
    This is an integrity check, not race-proof execution confinement.
    """
    _check_no_promisor(checkout)
    tree = _tracked_records(_git(checkout, "ls-tree", "-r", "--full-tree", "-z", revision), index=False)
    index = _tracked_records(_git(checkout, "ls-files", "--stage", "-v", "-z"), index=True)
    _require(index == tree, "oracle index differs from the recorded commit", "unsafe_source")
    _require(not _git(checkout, "ls-files", "--others", "-z"),
             "oracle checkout has extra files, including ignored files", "unsafe_source")
    for relative, (mode, expected_digest) in tree.items():
        source = _safe_file(checkout, relative)
        try:
            stat = source.stat()
            _require(bool(stat.st_mode & 0o111) == (mode == "100755"), "oracle tracked file mode changed", "unsafe_source")
            # SHA-1 here is Git's object identity, not a new cryptographic trust claim.
            digest = hashlib.sha1(b"blob " + str(stat.st_size).encode("ascii") + b"\0", usedforsecurity=False)
            count = 0
            with source.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    count += len(chunk)
                    digest.update(chunk)
            _require(count == stat.st_size and digest.hexdigest() == expected_digest,
                     "oracle tracked bytes differ from the recorded commit", "unsafe_source")
        except OSError as error:
            raise ContractError("unsafe_source", "cannot read ordinary tracked source bytes") from error


def verify_oracle(contract: dict, lock: dict, oracle_root: Path) -> dict:
    """Execute the reviewed standalone CLI only after identity/integrity checks."""
    receipt = validate(contract, lock)
    _require(receipt["receipt_complete"], "a complete recorded receipt is required before executing source", "incomplete_receipt")
    checkout = oracle_root.absolute()
    _require(checkout.is_dir(), "oracle checkout is missing", "unsafe_source")
    _require(not any(part.is_symlink() for part in [checkout, *checkout.parents]), "oracle checkout must not use symlinks", "unsafe_source")
    _fixed(_git(checkout, "rev-parse", "--show-toplevel"), str(checkout), "independent oracle checkout")
    _fixed(_git(checkout, "rev-parse", "HEAD"), contract["oracle"]["revision"], "oracle HEAD")
    _fixed(_git(checkout, "config", "--local", "--get-all", "remote.origin.url"), contract["oracle"]["repository"], "oracle origin")
    _check_tracked_bytes(checkout, contract["oracle"]["revision"])
    _git(checkout, "merge-base", "--is-ancestor", contract["sources"]["opentpu_base"]["revision"], contract["oracle"]["revision"])
    for notice in LICENSES:
        file = _safe_file(checkout, notice["path"])
        _fixed(hashlib.sha256(file.read_bytes()).hexdigest(), notice["sha256"], "full MIT notice digest")
    for path in ("tools/packed_oracle/__init__.py", "tools/packed_oracle/__main__.py", "tools/packed_oracle/formats.py", "tools/packed_oracle/vectors.py", WORKFLOW_PATH):
        _safe_file(checkout, path)
    fixture, file_digest = read_json(_safe_file(checkout, FIXTURE_PATH))
    metadata = validate_fixture(fixture, contract, file_digest)
    # Both module and argument are code constants, never commands/URLs from JSON.
    raw = _run([sys.executable, "-B", "-E", "-S", "-m", MODULE, "verify", FIXTURE_PATH], checkout, timeout=60)
    report = parse_json(raw)
    expected = {
        "schema_version": 1, "fixture_id": contract["fixture"]["fixture_id"], "arithmetic_id": ARITHMETIC_ID,
        "payload_sha256": fixture["payload_sha256"], "case_count": metadata["case_count"],
        "decode_cases": metadata["decode_cases"], "dot_cases": metadata["dot_cases"],
        "synthetic": True, "verified": True, "mismatches": [], "device_ready": False, "inference_ready": False,
    }
    _fixed(report, expected, "fixed oracle verification report")
    # No checkout mutations are expected, including on a branch worktree.
    _check_tracked_bytes(checkout, contract["oracle"]["revision"])
    _fixed(_git(checkout, "rev-parse", "HEAD"), contract["oracle"]["revision"], "oracle HEAD after execution")
    return {**receipt, **metadata, "source_identity_checked": True, "oracle_executed": True,
            "synthetic_vectors_verified": True,
            "note": "Exact reviewed source passed supplied synthetic packed vectors only; not native/RTL/model/device or throughput evidence."}


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ContractError("invalid_arguments", "invalid command-line arguments")


def main(argv=None) -> int:
    parser = _Parser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--fixture", type=Path, help="check fixture metadata/integrity without running its arithmetic")
    parser.add_argument("--strict-receipt", action="store_true", help="require a structurally complete recorded fork acceptance")
    parser.add_argument("--oracle-root", type=Path, help="explicitly execute the fixed CLI from a clean exact existing checkout; no fetch")
    try:
        args = parser.parse_args(argv)
        contract, _ = read_json(args.contract)
        lock, _ = read_json(ROOT / "dependencies.lock.json")
        report = validate(contract, lock)
        if args.fixture:
            fixture, digest = read_json(args.fixture)
            report.update(validate_fixture(fixture, contract, digest))
        if args.oracle_root:
            report = verify_oracle(contract, lock, args.oracle_root)
    except ContractError as error:
        print(json.dumps({"code": error.code, "message": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 1 if args.strict_receipt and not report["receipt_complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
