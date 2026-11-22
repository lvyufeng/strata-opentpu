# Copyright (c) 2026 lvyufeng and contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate an inert Strata GGUF header receipt, never the referenced artifacts.

The accompanying draft-2020-12 schemas describe the fixed JSON envelopes. This
stdlib checker enforces those schemas' exact fields/types and the cross-field
semantics which JSON Schema cannot express. It contains no GGUF reader, imports
no Strata code, executes no supplied commands, and never authenticates hashes or
input_kind. Two frozen profiles exist: a schema_version 1 or 2 receipt is checked
against its own pinned producer, limits and confirmed block geometry, and every
other version is rejected without coercion. Complete declarations are not model,
backend or inference acceptance.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT = ROOT / "tests/fixtures/gguf-header-inventory.synthetic.json"
DEFAULT_RECEIPT_V2 = ROOT / "tests/fixtures/gguf-header-inventory-v2.synthetic.json"
SCHEMA_PATH = ROOT / "configs/contracts/gguf-header-inventory.schema.json"
SCHEMA_PATH_V2 = ROOT / "configs/contracts/gguf-header-inventory-v2.schema.json"
MAX_INTEGER = (1 << 63) - 1
MAX_JSON_BYTES = 64 << 20
MAX_JSON_DEPTH = 32
MAX_NUMBER_CHARS = 64
MAX_PATH_BYTES = 4096
LIMITS = {
    "max_shards": 64,
    "max_header_bytes_per_file": 16777216,
    "max_header_bytes_total": 33554432,
    "max_string_bytes": 1048576,
    "max_string_bytes_total": 16777216,
    "max_metadata_entries_per_file": 4096,
    "max_metadata_entries_total": 8192,
    "max_tensors_total": 100000,
    "max_array_items_per_file": 500000,
    "max_array_items_total": 1000000,
    "max_rank": 8,
    "max_glob_entries": 4096,
    "max_path_arguments": 64,
}
# v2 raised ONLY the two array-item ceilings after measuring the real shard 1
# header (744317 array items in one file). Every other ceiling is unchanged.
LIMITS_V2 = {
    "max_shards": 64,
    "max_header_bytes_per_file": 16777216,
    "max_header_bytes_total": 33554432,
    "max_string_bytes": 1048576,
    "max_string_bytes_total": 16777216,
    "max_metadata_entries_per_file": 4096,
    "max_metadata_entries_total": 8192,
    "max_tensors_total": 100000,
    "max_array_items_per_file": 2000000,
    "max_array_items_total": 4000000,
    "max_rank": 8,
    "max_glob_entries": 4096,
    "max_path_arguments": 64,
}
PRODUCER = {
    "tool": "Strata/terminal/tools/quant_formats.py",
    "mode": "bounded-local-v1",
    "base_revision": "3142b66b96e9c0e775e30f620b11b191a1427c2a",
    "parser_sha256": "c84a0f88c6c75a588cb7de74abd23b2e6387c92c58e088012aedef5bd3c6c431",
}
PRODUCER_V2 = {
    "tool": "Strata/terminal/tools/quant_formats.py",
    "mode": "bounded-local-v2",
    "base_revision": "3142b66b96e9c0e775e30f620b11b191a1427c2a",
    "parser_sha256": "c84a0f88c6c75a588cb7de74abd23b2e6387c92c58e088012aedef5bd3c6c431",
}
CLAIMS = {
    "header_only": True,
    "input_provenance_verified": False,
    "artifact_integrity_verified": False,
    "model_completeness_verified": False,
    "backend_compatibility_verified": False,
    "device_ready": False,
    "inference_ready": False,
}
# Reviewed receipt geometry ONLY: the confirmed ID subset from the pinned
# terminal/tools/gguf_reader.py GGML_TYPES / BLOCK_GEOMETRY (lines 26-63).
# This is not a second general Strata type table or support/quantization code.
CONFIRMED_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34),
    17: ("IQ2_XS", 256, 74), 21: ("IQ3_S", 256, 110),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4),
    27: ("I64", 1, 8), 28: ("F64", 1, 8), 30: ("BF16", 1, 2),
    42: ("Q2_0", 64, 18),
}
# v2 additionally confirms the six measured types present in the real shard 1
# header. IQ1_S (19), MXFP4 (39), NVFP4 (40) and 41 stay unknown.
CONFIRMED_TYPES_V2 = {
    **CONFIRMED_TYPES,
    14: ("Q6_K", 256, 210), 16: ("IQ2_XXS", 256, 66),
    20: ("IQ4_NL", 32, 18), 22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136), 29: ("IQ1_M", 256, 56),
}
PROFILES = {
    1: {"producer": PRODUCER, "limits": LIMITS, "claims": CLAIMS,
        "confirmed_types": CONFIRMED_TYPES},
    2: {"producer": PRODUCER_V2, "limits": LIMITS_V2, "claims": CLAIMS,
        "confirmed_types": CONFIRMED_TYPES_V2},
}
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_STORAGE_KEYS = {"layout", "block_elements", "block_bytes", "row_stride_bytes", "data_bytes"}
_SPLIT_KEYS = {"index", "count", "total_tensors"}


class InventoryError(ValueError):
    """Invalid input or unsafe receipt access, with a stable, sanitized code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _require(condition, message, code="invalid_receipt"):
    if not condition:
        raise InventoryError(code, message)


def _object(value, keys, label):
    _require(type(value) is dict and all(type(key) is str for key in value)
             and set(value) == set(keys), f"{label}: unexpected schema fields")


def _same(value, expected):
    """Unlike Python equality, JSON integers and booleans are not aliases."""
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return set(value) == set(expected) and all(_same(value[key], child)
                                                   for key, child in expected.items())
    if type(expected) is list:
        return len(value) == len(expected) and all(_same(a, b) for a, b in zip(value, expected))
    return value == expected


def _fixed(value, expected, label):
    _require(_same(value, expected), f"{label}: fixed declaration differs")


def _integer(value, label, minimum=0, maximum=MAX_INTEGER, nullable=False):
    if nullable and value is None:
        return
    _require(type(value) is int and minimum <= value <= maximum,
             f"{label}: integer outside permitted bounds")


def _text(value, label, limits=LIMITS):
    _require(type(value) is str and bool(value), f"{label}: expected nonempty text")
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeError:
        raise InventoryError("invalid_receipt", f"{label}: invalid Unicode") from None
    _require(size <= limits["max_string_bytes"], f"{label}: string exceeds byte limit")
    _require(not any(unicodedata.category(char) in ("Cc", "Cf", "Cs") for char in value),
             f"{label}: unsafe control characters")
    return size


def _add(a, b, label):
    _require(a <= MAX_INTEGER - b, f"{label}: integer sum overflow")
    return a + b


def _product(values, label):
    product = 1
    for value in values:
        _require(product <= MAX_INTEGER // value, f"{label}: integer product overflow")
        product *= value
    return product


def _derive_split(shards, actual_tensors):
    declarations = [shard["split"] for shard in shards]
    if not any(value is not None for item in declarations for value in item.values()):
        return {"status": "unknown", "declared_shards": None, "declared_tensors": None,
                "present_indices": [], "missing_indices": [],
                "diagnostics": ["split-not-declared"]}
    counts = {item["count"] for item in declarations if item["count"] is not None}
    totals = {item["total_tensors"] for item in declarations if item["total_tensors"] is not None}
    _require(len(counts) <= 1 and len(totals) <= 1, "split: inconsistent declarations")
    count = next(iter(counts), None)
    total = next(iter(totals), None)
    present = sorted(item["index"] for item in declarations if item["index"] is not None)
    _require(len(present) == len(set(present)), "split: duplicate shard indices")
    if count is not None:
        _require(len(shards) <= count and all(index < count for index in present),
                 "split: inputs or indices exceed declared shard count")
    if total is not None:
        _require(actual_tensors <= total, "split: actual tensors exceed declared total")
    missing = [] if count is None else [index for index in range(count) if index not in present]
    partial = any(value is None for item in declarations for value in item.values())
    diagnostics = ["split-declarations-partial"] if partial else []
    if missing:
        diagnostics.append("split-shards-missing")
    status = "incomplete" if partial or missing else "complete"
    if status == "complete":
        _require(len(shards) == count and actual_tensors == total,
                 "split: complete declarations disagree with actual totals")
    return {"status": status, "declared_shards": count, "declared_tensors": total,
            "present_indices": present, "missing_indices": missing, "diagnostics": diagnostics}


def _select_profile(document):
    """Pick the one frozen profile named by schema_version; never coerce versions."""
    version = document.get("schema_version") if type(document) is dict else None
    profile = PROFILES.get(version) if type(version) is int else None
    _require(profile is not None, "schema_version: unsupported receipt schema version")
    return version, profile


def validate(document):
    """Check fixed receipt shape and arithmetic; return an unverified summary.

    Observable name/header/count bounds are checked against the profile that
    schema_version selects. Metadata/array/string data not exported in the receipt
    cannot be independently inspected: their limits entries remain fixed producer
    declarations, not authenticated execution facts.
    No source lock, model file, hash target, path or URL in the receipt is opened.
    """
    version, profile = _select_profile(document)
    limits = profile["limits"]
    _object(document, {"schema_version", "kind", "input_kind", "producer", "limits",
                       "shards", "tensors", "split", "totals", "claims"}, "receipt")
    _fixed(document["schema_version"], version, "schema_version")
    _fixed(document["kind"], "gguf-header-inventory", "kind")
    _require(type(document["input_kind"]) is str
             and document["input_kind"] in ("synthetic", "local-unverified"), "input_kind: invalid declaration")
    _fixed(document["producer"], profile["producer"], "producer")
    _fixed(document["limits"], limits, "limits")
    _fixed(document["claims"], profile["claims"], "claims")
    shards = document["shards"]
    tensors = document["tensors"]
    _require(type(shards) is list and 1 <= len(shards) <= limits["max_shards"], "shards: invalid count")
    _require(type(tensors) is list and len(tensors) <= limits["max_tensors_total"], "tensors: invalid count")
    shard_map = {}
    header_bytes = 0
    declared_tensor_count = 0
    for ordinal, shard in enumerate(shards):
        _object(shard, {"label", "file_size_bytes", "gguf_version", "alignment_bytes", "header_end_bytes",
                        "data_start_bytes", "header_sha256", "artifact_sha256", "payload_read",
                        "tensor_count", "split"}, "shard")
        _fixed(shard["label"], f"shard-{ordinal:04d}", "shard label/order")
        _fixed(shard["gguf_version"], 3, "gguf_version")
        _fixed(shard["artifact_sha256"], None, "artifact_sha256")
        _fixed(shard["payload_read"], False, "payload_read")
        _require(type(shard["header_sha256"]) is str
                 and _HEX64.fullmatch(shard["header_sha256"]) is not None, "header_sha256: invalid declaration")
        _integer(shard["file_size_bytes"], "file_size_bytes", 24)
        _integer(shard["header_end_bytes"], "header_end_bytes", 24, limits["max_header_bytes_per_file"])
        _integer(shard["data_start_bytes"], "data_start_bytes", 24)
        _integer(shard["alignment_bytes"], "alignment_bytes", 8, 1048576)
        alignment = shard["alignment_bytes"]
        _require(alignment & (alignment - 1) == 0, "alignment_bytes: not a power of two")
        _require(shard["header_end_bytes"] <= shard["file_size_bytes"], "header: exceeds declared file size")
        expected_start = _add(shard["header_end_bytes"], alignment - 1, "header alignment") // alignment * alignment
        _require(shard["data_start_bytes"] == expected_start, "data_start_bytes: incorrect header alignment")
        header_bytes += shard["header_end_bytes"]
        _require(header_bytes <= limits["max_header_bytes_total"], "headers: total byte limit exceeded")
        _integer(shard["tensor_count"], "shard tensor_count", 0, limits["max_tensors_total"])
        declared_tensor_count += shard["tensor_count"]
        _require(declared_tensor_count <= limits["max_tensors_total"], "shards: total tensor limit exceeded")
        if shard["tensor_count"]:
            _require(shard["data_start_bytes"] <= shard["file_size_bytes"], "data region: exceeds declared file size")
        split = shard["split"]
        _object(split, _SPLIT_KEYS, "shard split")
        _integer(split["index"], "split index", 0, limits["max_shards"] - 1, nullable=True)
        _integer(split["count"], "split count", 1, limits["max_shards"], nullable=True)
        _integer(split["total_tensors"], "split total_tensors", 0, limits["max_tensors_total"], nullable=True)
        if split["index"] is not None and split["count"] is not None:
            _require(split["index"] < split["count"], "split: shard index outside count")
        shard_map[shard["label"]] = {"declaration": shard, "actual_count": 0, "known": [], "unknown": []}
    _require(declared_tensor_count == len(tensors), "shards: tensor counts disagree with tensor list")
    seen_names = set()
    previous_key = None
    known_bytes = unknown_count = name_bytes = 0
    for tensor in tensors:
        _object(tensor, {"name", "shard", "shape", "ggml_type_id", "ggml_type_name", "relative_offset_bytes",
                         "absolute_offset_bytes", "storage", "tensor_sha256"}, "tensor")
        name_bytes += _text(tensor["name"], "tensor name", limits)
        _require(name_bytes <= limits["max_string_bytes_total"], "tensor names: total byte limit exceeded")
        _require(tensor["name"] not in seen_names, "tensor names: global duplicate")
        seen_names.add(tensor["name"])
        _require(type(tensor["shard"]) is str and tensor["shard"] in shard_map, "tensor shard: unknown label")
        key = (tensor["shard"], tensor["name"])
        _require(previous_key is None or previous_key < key, "tensors: noncanonical ordering")
        previous_key = key
        entry = shard_map[tensor["shard"]]
        entry["actual_count"] += 1
        shard = entry["declaration"]
        shape = tensor["shape"]
        _require(type(shape) is list and 1 <= len(shape) <= limits["max_rank"], "shape: invalid raw GGUF rank")
        for dimension in shape:
            _integer(dimension, "shape dimension", 1)
        _product(shape, "shape")
        _integer(tensor["ggml_type_id"], "ggml_type_id", 0, (1 << 32) - 1)
        _text(tensor["ggml_type_name"], "ggml_type_name", limits)
        _integer(tensor["relative_offset_bytes"], "relative_offset_bytes")
        _integer(tensor["absolute_offset_bytes"], "absolute_offset_bytes", 1)
        relative = tensor["relative_offset_bytes"]
        absolute = tensor["absolute_offset_bytes"]
        _require(relative % shard["alignment_bytes"] == 0, "tensor offset: not alignment-multiple")
        _require(absolute == _add(shard["data_start_bytes"], relative, "tensor absolute offset"),
                 "tensor absolute offset: disagrees with data start")
        _require(shard["data_start_bytes"] <= absolute < shard["file_size_bytes"],
                 "tensor offset: outside declared data region")
        _fixed(tensor["tensor_sha256"], None, "tensor_sha256")
        storage = tensor["storage"]
        _object(storage, _STORAGE_KEYS, "storage")
        geometry = profile["confirmed_types"].get(tensor["ggml_type_id"])
        if geometry is None:
            _fixed(storage, {"layout": "unknown", "block_elements": None, "block_bytes": None,
                             "row_stride_bytes": None, "data_bytes": None}, "unknown storage")
            unknown_count += 1
            entry["unknown"].append(absolute)
            continue
        name, block_elements, block_bytes = geometry
        _fixed(tensor["ggml_type_name"], name, "confirmed ggml_type_name")
        _require(shape[0] % block_elements == 0, "shape: fastest dimension not divisible by block elements")
        row_stride = _product([shape[0] // block_elements, block_bytes], "row stride")
        data_bytes = _product([row_stride, *shape[1:]], "storage bytes")
        _fixed(storage, {"layout": "gguf-row-blocks", "block_elements": block_elements,
                         "block_bytes": block_bytes, "row_stride_bytes": row_stride,
                         "data_bytes": data_bytes}, "known storage/raw row geometry")
        end = _add(absolute, data_bytes, "tensor extent")
        _require(end <= shard["file_size_bytes"], "tensor extent: exceeds declared file size")
        entry["known"].append((absolute, end))
        known_bytes = _add(known_bytes, data_bytes, "known storage subtotal")
    for entry in shard_map.values():
        _require(entry["actual_count"] == entry["declaration"]["tensor_count"], "shard: actual tensor count differs")
        intervals = sorted(entry["known"])
        unknown_offsets = sorted(entry["unknown"])
        previous_end = None
        for start, end in intervals:
            _require(previous_end is None or start >= previous_end, "tensor extents: known intervals overlap")
            previous_end = end
            index = bisect_right(unknown_offsets, start)
            _require(index == len(unknown_offsets) or unknown_offsets[index] >= end,
                     "tensor extents: unknown offset strictly inside known interval")
    totals = {"tensor_count": len(tensors), "known_storage_bytes": known_bytes,
              "unknown_storage_tensors": unknown_count}
    _fixed(document["totals"], totals, "totals")
    _object(document["split"], {"status", "declared_shards", "declared_tensors", "present_indices",
                                "missing_indices", "diagnostics"}, "split summary")
    split_summary = _derive_split(shards, len(tensors))
    _fixed(document["split"], split_summary, "split summary/facts/order/diagnostics")
    report = {"valid": True, "kind": "gguf-header-inventory",
              "input_kind": document["input_kind"],
              "split_status": split_summary["status"], "split": split_summary, "totals": totals,
              "header_declarations_complete": document["split"]["status"] == "complete" and unknown_count == 0,
              "source_identity_verified": False, **profile["claims"],
              "note": "Receipt consistency only. input_kind, producer identity, header hashes and inspection limits "
                      "are unverified declarations. Known storage is a subtotal excluding unknown tensors, padding, "
                      "metadata and runtime conversions. No artifact, model, backend, device or inference acceptance."}
    # The schema-version-1 summary surface is the one accepted with #18; it stays
    # byte-identical, so the selected profile is reported only for version 2.
    if version == 2:
        report["schema_version"] = version
        report["profile"] = "bounded-local-v2"
    return report


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "JSON: duplicate object key", "invalid_json")
        result[key] = value
    return result


def _json_int(token):
    _require(len(token) <= MAX_NUMBER_CHARS, "JSON: number literal exceeds limit", "invalid_json")
    return int(token)


def _json_float(token):
    _require(len(token) <= MAX_NUMBER_CHARS, "JSON: number literal exceeds limit", "invalid_json")
    value = float(token)
    _require(math.isfinite(value), "JSON: nonfinite number", "invalid_json")
    return value


def _json_constant(_token):
    raise InventoryError("invalid_json", "JSON: nonfinite constant")


def _depth_check(data):
    depth = 0
    in_string = escaped = False
    for byte in data:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                in_string = False
        elif byte == 34:
            in_string = True
        elif byte in (91, 123):
            depth += 1
            _require(depth <= MAX_JSON_DEPTH, "JSON: nesting exceeds limit", "invalid_json")
        elif byte in (93, 125):
            depth -= 1
            _require(depth >= 0, "JSON: malformed nesting", "invalid_json")


def _unicode_tree(value):
    if type(value) is str:
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeError:
            raise InventoryError("invalid_json", "JSON: invalid Unicode") from None
    elif type(value) is dict:
        for key, child in value.items():
            _unicode_tree(key)
            _unicode_tree(child)
    elif type(value) is list:
        for child in value:
            _unicode_tree(child)


def loads_receipt(data):
    """Strict bounded JSON decoding; semantic validation is a separate operation."""
    _require(type(data) is bytes, "JSON: expected UTF-8 bytes", "invalid_json")
    _require(len(data) <= MAX_JSON_BYTES, "JSON: receipt exceeds byte limit", "receipt_too_large")
    _depth_check(data)
    try:
        document = json.loads(data.decode("utf-8", errors="strict"), object_pairs_hook=_json_pairs,
                              parse_int=_json_int, parse_float=_json_float, parse_constant=_json_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError, OverflowError, ValueError) as error:
        if isinstance(error, InventoryError):
            raise
        raise InventoryError("invalid_json", "JSON: malformed UTF-8 document") from None
    _unicode_tree(document)
    return document


def _fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def load_receipt(path):
    """Read only one explicitly supplied regular JSON file, using one raw FD.

    O_NOFOLLOW covers the final component, not ancestor directories. Metadata
    comparisons detect some changes, not all races; this is not a sandbox or an
    authenticity guarantee. No path/URL found inside the document is followed.
    """
    _require(bool(getattr(os, "O_NOFOLLOW", 0)) and bool(getattr(os, "O_NONBLOCK", 0)),
             "Receipt: safe open flags unavailable", "unsupported_file_safety")
    descriptor = None
    try:
        _require(len(os.fsencode(path)) <= MAX_PATH_BYTES, "Receipt: path exceeds limit", "unsafe_receipt")
        before_path = os.lstat(path)
        _require(stat.S_ISREG(before_path.st_mode), "Receipt: input must be a regular nonsymlink file", "unsafe_receipt")
        _require(before_path.st_size <= MAX_JSON_BYTES, "Receipt: byte limit exceeded", "receipt_too_large")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        before_fd = os.fstat(descriptor)
        _require(stat.S_ISREG(before_fd.st_mode) and _fingerprint(before_path) == _fingerprint(before_fd),
                 "Receipt: input changed before read", "receipt_changed")
        stream = os.fdopen(descriptor, "rb", buffering=0)
        descriptor = None  # Ownership transferred to stream; every exit closes it.
        with stream:
            chunks = []
            size = 0
            while True:
                chunk = stream.read(min(1 << 20, MAX_JSON_BYTES + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                _require(size <= MAX_JSON_BYTES, "Receipt: byte limit exceeded", "receipt_too_large")
            after_fd = os.fstat(stream.fileno())
            _require(_fingerprint(before_fd) == _fingerprint(after_fd) and size == before_fd.st_size,
                     "Receipt: input changed during read", "receipt_changed")
        return loads_receipt(b"".join(chunks))
    except InventoryError:
        raise
    except (OSError, TypeError, ValueError):
        raise InventoryError("receipt_access", "Receipt: could not safely read input") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, _message):
        raise InventoryError("invalid_arguments", "Unsupported or missing checker arguments")


def main(argv=None):
    parser = _ArgumentParser(prog="check_gguf_inventory.py", description=__doc__)
    parser.add_argument("--receipt", type=Path, default=None,
                        help="Explicit receipt JSON validated against the version it declares; no artifact paths are followed")
    parser.add_argument("--receipt-v2", type=Path, nargs="?", const=DEFAULT_RECEIPT_V2, default=None, metavar="PATH",
                        help="Explicit receipt JSON that must declare schema_version 2 (default: the frozen v2 synthetic fixture)")
    parser.add_argument("--strict-complete", action="store_true", help="Require complete split declarations and no unknown storage (not inference acceptance)")
    try:
        args = parser.parse_args(argv)
        _require(args.receipt is None or args.receipt_v2 is None, "Receipt: choose one of --receipt or --receipt-v2")
        if args.receipt is not None:
            report = validate(load_receipt(args.receipt))
        elif args.receipt_v2 is not None:
            document = load_receipt(args.receipt_v2)
            declared = document.get("schema_version") if type(document) is dict else None
            _require(type(declared) is int and declared == 2, "Receipt: --receipt-v2 requires schema_version 2")
            report = validate(document)
        else:
            report = validate(load_receipt(DEFAULT_RECEIPT))
    except InventoryError as error:
        print(f"error: {error.code}: {error}", file=sys.stderr)
        return 2
    report["strict_complete_requested"] = args.strict_complete
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if report["split_status"] == "incomplete" or (args.strict_complete and not report["header_declarations_complete"]):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
