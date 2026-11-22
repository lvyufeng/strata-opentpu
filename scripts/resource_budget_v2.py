# Copyright (c) 2026 lvyufeng and contributors
# SPDX-License-Identifier: Apache-2.0
"""Offline, declaration-only partial-tensor expert residency budget checks.

Schema v2 extends the frozen schema-v1 budget input with explicit expert-axis
selections over one packed tensor. It uses the Python standard library and the
frozen v1 helpers read-only. It reads one JSON input, not paths, URLs or
commands contained in that input. It does not probe a host/device, read a model
artifact, run an engine, validate tensor contents or produce performance
measurements. Every selection is a declaration: its item bounds, packed bytes
and category charges are checked arithmetically, never authenticated.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from resource_budget import (  # noqa: E402 - path bootstrap must precede import
    BudgetError,
    CATEGORIES,
    Inputs,
    U64_MAX,
    WEIGHT_CATEGORIES,
    add,
    aligned,
    constraint,
    declared_evidence,
    identifier,
    list_at,
    multiply,
    object_at,
    reject_constant,
    unique_object,
    wall_budget,
)

SCHEMA_VERSION = 2
CONVENTION = "last_dimension"
ROLES = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
# A gate/up sub-slice may be resident on either side; the down projection is
# only charged as CPU cold-expert storage in this profile.
ROLE_CATEGORIES = {
    "ffn_gate_exps": ("hot_experts", "cold_experts"),
    "ffn_up_exps": ("hot_experts", "cold_experts"),
    "ffn_down_exps": ("cold_experts",),
}
SELECTION_MODES = ("range", "indices", "unknown")
MAX_SELECTION_ITEMS = 4096
MAX_SELECTIONS = 4096
MAX_SELECTIONS_PER_TENSOR = 64
MAX_TENSOR_RECORDS = 4096
MAX_TOTAL_SELECTED_ITEMS = 1000000
MAX_EXPERT_COUNT = 1000000
TOP_KEYS = frozenset({"schema_version", "id", "input_kind", "selection_contract",
                      "model", "memory", "wall"})
CONTRACT_KEYS = frozenset({"convention", "expert_dimension", "expert_count", "roles"})
MODEL_KEYS = frozenset({"layers", "context_tokens", "sessions", "routed_top_k",
                        "expert_skip", "shared_expert"})
MEMORY_KEYS = frozenset({"coverage", "device", "host", "tensors", "allocations"})
DEVICE_KEYS = frozenset({"capacity_bytes", "banks"})
HOST_KEYS = frozenset({"capacity_bytes", "reserve_bytes", "fragmentation_bytes",
                       "pinned_quota_bytes", "numa_nodes"})
BANK_KEYS = frozenset({"id", "capacity_bytes", "reserve_bytes", "fragmentation_bytes"})
NODE_KEYS = frozenset({"id", "capacity_bytes", "reserve_bytes", "fragmentation_bytes",
                       "pinned_quota_bytes"})
TENSOR_KEYS = frozenset({"id", "source", "layer", "role", "packed", "selection", "resident"})
SOURCE_KEYS = frozenset({"kind", "artifact", "revision", "sha256", "tensor"})
PACKED_KEYS = frozenset({"dtype", "layout", "shape", "block_values", "block_bytes",
                         "row_stride_bytes", "data_bytes", "external_scale_bytes"})
SELECTION_KEYS = frozenset({"id", "mode", "dimension", "axis_size", "start", "count",
                            "indices", "reason"})
RESIDENT_KEYS = frozenset({"status", "dtype", "layout", "data_bytes", "scale_bytes",
                           "overhead_bytes", "source_retained", "evidence"})
ALLOCATION_KEYS = frozenset({"id", "category", "bytes", "pinned", "tensor_ids",
                             "tensor_storage", "placements"})
PLACEMENT_KEYS = frozenset({"resource", "bytes", "alignment_bytes"})
COVERAGE_KEYS = frozenset({"status", "reason"})
WALL_KEYS = frozenset({"mode", "coverage", "numa", "link", "stages", "accepted_tokens",
                       "count_basis", "accepted_accounting"})
NUMA_KEYS = frozenset({"pcie_local_node", "transfer_worker_node", "cpu_worker_nodes",
                       "page_placement_declared"})
LINK_KEYS = frozenset({"endpoint_capability", "negotiated", "h2d", "d2h"})
STAGE_KEYS = frozenset({"id", "scope", "coverage", "repeat", "overlap", "before",
                        "cpu", "fpga", "after"})
OPERATION_KEYS = frozenset({"id", "depends_on", "covers", "kind", "milliseconds",
                            "bytes", "submissions"})
OVERLAP_KEYS = frozenset({"mode", "supported", "evidence"})


def expect_keys(value, path, allowed):
    value = object_at(value, path)
    extra = sorted(set(value) - set(allowed))
    if extra:
        raise BudgetError(f"{path}: unknown fields {extra}")
    return value


def optional_keys(value, path, allowed):
    if value is None:
        return
    expect_keys(value, path, allowed)


def selection_contract(document):
    """Read the reviewed expert-axis declaration; no record may redefine it."""
    data = expect_keys(document.get("selection_contract"),
                       "budget.selection_contract", CONTRACT_KEYS)
    if data.get("convention") != CONVENTION:
        raise BudgetError(f"budget.selection_contract.convention: expected {CONVENTION}")
    dimension = data.get("expert_dimension")
    if type(dimension) is not int or dimension < 1:
        raise BudgetError("budget.selection_contract.expert_dimension: "
                          "expected positive integer")
    count = data.get("expert_count")
    if type(count) is not int or not 1 <= count <= MAX_EXPERT_COUNT:
        raise BudgetError("budget.selection_contract.expert_count: expected positive "
                          f"integer bounded by {MAX_EXPERT_COUNT}")
    roles = expect_keys(data.get("roles"), "budget.selection_contract.roles", set(ROLES))
    shapes = {}
    for role in ROLES:
        path = f"budget.selection_contract.roles.{role}"
        if role not in roles:
            raise BudgetError(f"{path}: every declared expert role needs a shape")
        shape = list_at(roles[role], path)
        if len(shape) != dimension:
            raise BudgetError(f"{path}: expected {dimension} dimensions before the expert axis")
        parsed = []
        for index, size in enumerate(shape):
            if type(size) is not int or not 1 <= size <= U64_MAX:
                raise BudgetError(f"{path}[{index}]: expected positive integer")
            parsed.append(size)
        shapes[role] = parsed
    return {"convention": CONVENTION, "expert_dimension": dimension,
            "expert_count": count, "roles": shapes}


def selection_items(selection, path, axis_size):
    """Parse one bounded selection; it never identifies expert semantics."""
    selection = expect_keys(selection, path, SELECTION_KEYS)
    identity = identifier(selection.get("id"), f"{path}.id")
    mode = selection.get("mode")
    if mode not in SELECTION_MODES:
        raise BudgetError(f"{path}.mode: expected range, indices or unknown")
    dimension = selection.get("dimension")
    if type(dimension) is not int or dimension < 1:
        raise BudgetError(f"{path}.dimension: expected positive integer")
    declared_axis = selection.get("axis_size")
    if type(declared_axis) is not int or declared_axis < 1:
        raise BudgetError(f"{path}.axis_size: expected positive integer")
    result = {"id": identity, "mode": mode, "dimension": dimension,
              "axis_size": declared_axis, "start": None, "count": None,
              "indices": None, "reason": None, "items": None}
    if mode == "unknown":
        for key in ("start", "count", "indices"):
            if key in selection:
                raise BudgetError(f"{path}: unknown selection cannot declare {key}")
        result["reason"] = identifier(selection.get("reason"), f"{path}.reason")
        return result
    count = selection.get("count")
    if type(count) is not int or not 1 <= count <= MAX_SELECTION_ITEMS:
        raise BudgetError(f"{path}.count: expected positive integer bounded by "
                          f"{MAX_SELECTION_ITEMS}")
    result["count"] = count
    if mode == "range":
        if "indices" in selection:
            raise BudgetError(f"{path}: contiguous selection cannot declare indices")
        start = selection.get("start")
        if type(start) is not int or not 0 <= start <= U64_MAX:
            raise BudgetError(f"{path}.start: expected nonnegative integer")
        if start >= axis_size or count > axis_size - start:
            raise BudgetError(f"{path}: contiguous range exceeds the declared "
                              f"expert axis of {axis_size}")
        result["start"] = start
        result["items"] = range(start, start + count)
        return result
    if "start" in selection:
        raise BudgetError(f"{path}: indexed selection cannot declare start")
    raw = list_at(selection.get("indices"), f"{path}.indices")
    if not raw:
        raise BudgetError(f"{path}.indices: cannot be empty")
    if len(raw) > MAX_SELECTION_ITEMS:
        raise BudgetError(f"{path}.indices: exceeds the identifying bound of "
                          f"{MAX_SELECTION_ITEMS}")
    indices = []
    for index, value in enumerate(raw):
        if type(value) is not int or not 0 <= value <= U64_MAX:
            raise BudgetError(f"{path}.indices[{index}]: expected nonnegative integer")
        if value >= axis_size:
            raise BudgetError(f"{path}.indices[{index}]: {value} is outside the "
                              f"declared expert axis of {axis_size}")
        indices.append(value)
    if any(left >= right for left, right in zip(indices, indices[1:])):
        raise BudgetError(f"{path}.indices: expected strictly increasing indices")
    if count != len(indices):
        raise BudgetError(f"{path}.count: differs from the identifying indices")
    result["indices"] = indices
    result["items"] = indices
    return result


def selected_row_bytes(shape, dimension, count, row_stride):
    """Packed bytes of `count` expert rows: rows per item times row stride."""
    rows = 1
    for size in shape[1:dimension]:
        rows = multiply(rows, size)
    return multiply(multiply(rows, count), row_stride)


def required_integer(inputs, value, path, positive=False):
    if value is None:
        raise BudgetError(f"{path}: required for a schema-v2 selection record")
    parsed = inputs.integer(value, path, positive=positive)
    if parsed is None:
        raise BudgetError(f"{path}: required for a schema-v2 selection record")
    return parsed


def required_text(inputs, value, path):
    if value is None:
        raise BudgetError(f"{path}: required for a schema-v2 selection record")
    parsed = inputs.text(value, path)
    if parsed is None:
        raise BudgetError(f"{path}: required for a schema-v2 selection record")
    return parsed


def tensor_metadata(records, inputs, contract):
    """Check supplied selection arithmetic, not artifact contents or kernels."""
    dimension = contract["expert_dimension"]
    axis_size = contract["expert_count"]
    roles = contract["roles"]
    tensors, groups, order, selection_ids = {}, {}, [], set()
    records = list_at(records, "memory.tensors")
    if len(records) > MAX_TENSOR_RECORDS:
        raise BudgetError(f"memory.tensors: exceeds the record bound of {MAX_TENSOR_RECORDS}")
    total_items = 0
    for index, record in enumerate(records):
        path = f"memory.tensors[{index}]"
        record = expect_keys(record, path, TENSOR_KEYS)
        name = identifier(record.get("id"), f"{path}.id")
        if name in tensors:
            raise BudgetError(f"{path}: duplicate tensor id")
        layer = required_integer(inputs, record.get("layer"), f"{path}.layer")
        role = record.get("role")
        if role not in ROLES:
            raise BudgetError(f"{path}.role: expected one of {list(ROLES)}")
        source = expect_keys(record.get("source"), f"{path}.source", SOURCE_KEYS)
        if source.get("kind") not in ("tensor_metadata", "synthetic"):
            raise BudgetError(f"{path}.source.kind: expected tensor_metadata or synthetic")
        provenance = [required_text(inputs, source.get(key), f"{path}.source.{key}")
                      for key in ("artifact", "revision", "sha256", "tensor")]
        if not re.fullmatch(r"[0-9a-fA-F]{64}", provenance[2]):
            raise BudgetError(f"{path}.source.sha256: expected 64 hexadecimal characters")
        identity = tuple(provenance)
        packed = expect_keys(record.get("packed"), f"{path}.packed", PACKED_KEYS)
        dtype = required_text(inputs, packed.get("dtype"), f"{path}.packed.dtype")
        layout = required_text(inputs, packed.get("layout"), f"{path}.packed.layout")
        shape = list_at(packed.get("shape"), f"{path}.packed.shape")
        if len(shape) != dimension + 1:
            raise BudgetError(f"{path}.packed.shape: expected rank {dimension + 1} for "
                              f"expert axis {dimension} ({CONVENTION})")
        shape = [required_integer(inputs, size, f"{path}.packed.shape[{axis}]",
                                  positive=True)
                 for axis, size in enumerate(shape)]
        if shape[dimension] != axis_size:
            raise BudgetError(f"{path}.packed.shape[{dimension}]: does not match "
                              f"selection_contract.expert_count {axis_size}")
        if shape[:dimension] != roles[role]:
            raise BudgetError(f"{path}.packed.shape: prefix does not match "
                              f"selection_contract.roles.{role}")
        values = {key: required_integer(inputs, packed.get(key), f"{path}.packed.{key}",
                                        positive=key in ("block_values", "block_bytes",
                                                         "row_stride_bytes"))
                  for key in ("block_values", "block_bytes", "row_stride_bytes",
                              "data_bytes", "external_scale_bytes")}
        if shape[0] % values["block_values"]:
            raise BudgetError(f"{path}: row shape is not an integral number of "
                              "declared blocks")
        row_min = multiply(shape[0] // values["block_values"], values["block_bytes"])
        if values["row_stride_bytes"] < row_min:
            raise BudgetError(f"{path}: row stride omits packed blocks/scales")
        rows = 1
        for size in shape[1:]:
            rows = multiply(rows, size)
        if multiply(rows, values["row_stride_bytes"]) != values["data_bytes"]:
            raise BudgetError(f"{path}: packed data bytes differ from row-stride/shape "
                              "metadata")
        if values["external_scale_bytes"] != 0:
            raise BudgetError(f"{path}.packed.external_scale_bytes: embedded scales are "
                              "required for a schema-v2 partial selection; external "
                              "scale bytes must be 0")
        selection = selection_items(record.get("selection"), f"{path}.selection", axis_size)
        selection_id = selection["id"]
        if selection_id in selection_ids:
            raise BudgetError(f"{path}.selection.id: duplicate selection id")
        selection_ids.add(selection_id)
        if selection["dimension"] != dimension:
            raise BudgetError(f"{path}.selection.dimension: expected the declared "
                              f"expert axis {dimension} ({CONVENTION})")
        if selection["dimension"] != len(shape) - 1:
            raise BudgetError(f"{path}.selection.dimension: expected the last declared "
                              "tensor dimension")
        if selection["axis_size"] != shape[dimension]:
            raise BudgetError(f"{path}.selection.axis_size: does not match packed "
                              f"shape[{dimension}]")
        if selection["count"] is not None:
            total_items = add(total_items, selection["count"])
            if total_items > MAX_TOTAL_SELECTED_ITEMS:
                raise BudgetError(f"memory.tensors: total selected items exceed the "
                                  f"bound of {MAX_TOTAL_SELECTED_ITEMS}")
        selected_bytes = None
        if selection["count"] is not None:
            selected_bytes = selected_row_bytes(shape, dimension, selection["count"],
                                                values["row_stride_bytes"])
        resident = expect_keys(record.get("resident"), f"{path}.resident", RESIDENT_KEYS)
        status = resident.get("status")
        if status not in ("native", "converted", "unknown"):
            raise BudgetError(f"{path}.resident.status: expected native, converted or "
                              "unknown")
        if status == "unknown":
            inputs.missing("memory", f"{path}.resident.status")
        resident_dtype = resident_layout = None
        stored = {key: None for key in ("data_bytes", "scale_bytes", "overhead_bytes")}
        retained = None
        if status in ("native", "converted"):
            resident_dtype = required_text(inputs, resident.get("dtype"),
                                           f"{path}.resident.dtype")
            resident_layout = required_text(inputs, resident.get("layout"),
                                            f"{path}.resident.layout")
            stored = {key: required_integer(inputs, resident.get(key),
                                            f"{path}.resident.{key}")
                      for key in ("data_bytes", "scale_bytes", "overhead_bytes")}
            if resident.get("source_retained") is None:
                raise BudgetError(f"{path}.resident.source_retained: required for a "
                                  "declared resident status")
            retained = inputs.flag(resident.get("source_retained"),
                                   f"{path}.resident.source_retained")
            if status == "converted":
                evidence = declared_evidence(resident.get("evidence"),
                                             f"{path}.resident.evidence", inputs,
                                             "memory")
                if evidence is None:
                    raise BudgetError(f"{path}.resident.evidence: converted resident "
                                      "needs declared conversion evidence")
        else:
            retained = inputs.flag(resident.get("source_retained"),
                                   f"{path}.resident.source_retained")
        if status == "native":
            if resident_dtype != dtype or resident_layout != layout:
                raise BudgetError(f"{path}: native resident dtype/layout differs from "
                                  "packed metadata")
            if selected_bytes is not None and stored["data_bytes"] != selected_bytes:
                raise BudgetError(f"{path}.resident.data_bytes: native sub-slice must "
                                  "equal selected packed bytes")
            if stored["scale_bytes"] != 0:
                raise BudgetError(f"{path}.resident.scale_bytes: native resident "
                                  "cannot declare external scale bytes for a selection")
            if retained is True:
                raise BudgetError(f"{path}: native resident cannot retain the packed "
                                  "source; separate copies need allocations")
        if status == "unknown" and selection["count"] is None:
            inputs.missing("memory", f"{path}.resident.data_bytes: unknown selection "
                           "extent is not charged")
        storage = add(*stored.values()) if None not in stored.values() else None
        packed_whole = add(values["data_bytes"], values["external_scale_bytes"])
        tensors[name] = {"id": name, "layer": layer, "role": role,
                         "resident_bytes": storage, "packed_bytes": packed_whole,
                         "resident_status": status, "source_retained": retained}
        if identity in groups:
            group = groups[identity]
            if group["layer"] != layer or group["role"] != role:
                raise BudgetError(f"{path}: conflicting layer/role mapping for source "
                                  f"tensor {provenance[3]}")
            for key, expected in (("dtype", dtype), ("layout", layout),
                                  ("shape", shape),
                                  ("block_values", values["block_values"]),
                                  ("block_bytes", values["block_bytes"]),
                                  ("row_stride_bytes", values["row_stride_bytes"]),
                                  ("data_bytes", values["data_bytes"]),
                                  ("external_scale_bytes", values["external_scale_bytes"])):
                if group["packed"][key] != expected:
                    raise BudgetError(f"{path}: conflicting packed {key} for source "
                                      f"tensor {provenance[3]}")
            if len(group["selections"]) >= MAX_SELECTIONS_PER_TENSOR:
                raise BudgetError(f"{path}: selection records per packed tensor exceed "
                                  f"the bound of {MAX_SELECTIONS_PER_TENSOR}")
        else:
            group = {"id": name, "source_tensor": provenance[3], "layer": layer,
                     "role": role, "first_index": index,
                     "packed": {"dtype": dtype, "layout": layout, "shape": list(shape),
                                "block_values": values["block_values"],
                                "block_bytes": values["block_bytes"],
                                "row_stride_bytes": values["row_stride_bytes"],
                                "data_bytes": values["data_bytes"],
                                "external_scale_bytes": values["external_scale_bytes"]},
                     "selections": []}
            groups[identity] = group
            order.append(identity)
        group["selections"].append({"id": selection_id, "mode": selection["mode"],
                                    "count": selection["count"],
                                    "items": selection["items"]})
        if len(selection_ids) > MAX_SELECTIONS:
            raise BudgetError(f"memory.tensors: selection records exceed the bound of "
                              f"{MAX_SELECTIONS}")
    entries = []
    accounted_total, unaccounted_total, unknown_total = 0, 0, False
    for identity in order:
        group = groups[identity]
        path = f"memory.tensors[{group['first_index']}].selection"
        if not group["selections"]:
            raise BudgetError(f"{path}: packed tensor needs at least one selection")
        covered, unknown = set(), False
        for selection in group["selections"]:
            if selection["items"] is None:
                unknown = True
                continue
            overlap = covered & set(selection["items"])
            if overlap:
                raise BudgetError(f"{path}: overlapping expert selections "
                                  f"{sorted(overlap)[:8]} for source tensor "
                                  f"{group['source_tensor']}")
            covered |= set(selection["items"])
        if unknown:
            coverage_status = "unknown"
            accounted = None
            packed_selected = None
        else:
            accounted = len(covered)
            coverage_status = "complete" if accounted == axis_size else "partial"
            packed_selected = add(*(selected_row_bytes(
                group["packed"]["shape"], dimension, selection["count"],
                group["packed"]["row_stride_bytes"])
                for selection in group["selections"])) if group["selections"] else 0
            if accounted == 0:
                raise BudgetError(f"{path}: covered no expert items")
        if coverage_status == "partial":
            inputs.missing("memory", f"{path}: expert-axis coverage is partial "
                           f"({accounted} of {axis_size} items)")
        if coverage_status == "unknown":
            inputs.missing("memory", f"{path}: expert-axis coverage is unknown; a "
                           "mode 'unknown' selection is declared")
        whole = group["packed"]["data_bytes"] + group["packed"]["external_scale_bytes"]
        if packed_selected is not None and whole is not None and packed_selected > whole:
            raise BudgetError(f"{path}: selected packed bytes exceed the whole packed "
                              "tensor extent")
        if coverage_status == "complete" and packed_selected is not None:
            if whole is None:
                inputs.missing("memory", f"{path}: whole packed extent is undeclared")
            elif packed_selected != whole:
                raise BudgetError(f"{path}: complete coverage selected packed bytes "
                                  "differ from the whole packed tensor extent")
        entries.append({"tensor_id": group["id"],
                        "source_tensor": group["source_tensor"],
                        "layer": group["layer"], "role": group["role"],
                        "expert_axis_size": axis_size,
                        "accounted_items": accounted,
                        "unaccounted_items": None if accounted is None else axis_size - accounted,
                        "coverage": coverage_status,
                        "selections": [{"id": selection["id"],
                                        "mode": selection["mode"],
                                        "count": selection["count"]}
                                       for selection in group["selections"]],
                        "packed_selected_bytes": packed_selected,
                        "packed_total_bytes": whole})
        if accounted is None:
            unknown_total = True
        else:
            accounted_total = add(accounted_total, accounted)
            unaccounted_total = add(unaccounted_total, axis_size - accounted)
    statuses = {entry["coverage"] for entry in entries}
    if "unknown" in statuses:
        status = "unknown"
    elif "partial" in statuses:
        status = "partial"
    else:
        status = "complete"
    report = {"status": status, "tensor_count": len(entries),
              "expert_axis_size": axis_size, "tensors": entries,
              "declared_items": multiply(axis_size, len(entries)),
              "accounted_items": None if unknown_total else accounted_total,
              "unaccounted_items": None if unknown_total else unaccounted_total,
              "full_coverage": status == "complete",
              "fit_scope": "complete" if status == "complete" else "partial_coverage"}
    return tensors, report


def memory_budget(data, inputs, contract):
    data = expect_keys(data, "memory", MEMORY_KEYS)
    statuses = coverage_memory(data.get("coverage"), inputs)
    tensors, selection_report = tensor_metadata(data.get("tensors"), inputs, contract)
    device = expect_keys(data.get("device"), "memory.device", DEVICE_KEYS)
    host = expect_keys(data.get("host"), "memory.host", HOST_KEYS)
    resources = {}
    for kind, items in (("ddr_bank", device.get("banks")),
                        ("host_numa", host.get("numa_nodes"))):
        items = list_at(items, f"memory.{kind}")
        if not items:
            raise BudgetError(f"memory.{kind}: must explicitly list resources")
        for index, item in enumerate(items):
            path = f"memory.{kind}[{index}]"
            item = expect_keys(item, path, BANK_KEYS if kind == "ddr_bank" else NODE_KEYS)
            name = identifier(item.get("id"), f"{path}.id")
            if name in resources:
                raise BudgetError(f"{path}: duplicate resource id")
            start = len(inputs.unknown["memory"])
            capacity = inputs.integer(item.get("capacity_bytes"),
                                      f"{path}.capacity_bytes", positive=True)
            reserve = inputs.integer(item.get("reserve_bytes"), f"{path}.reserve_bytes")
            fragmentation = inputs.integer(item.get("fragmentation_bytes"),
                                           f"{path}.fragmentation_bytes")
            quota = inputs.integer(item.get("pinned_quota_bytes"),
                                   f"{path}.pinned_quota_bytes") if kind == "host_numa" else None
            resources[name] = {
                "id": name, "kind": kind, "capacity": capacity, "pinned_quota": quota,
                "known": add(*(value for value in (reserve, fragmentation)
                               if value is not None)),
                "pinned_known": 0,
                "unknown": len(inputs.unknown["memory"]) != start,
                "pinned_unknown": kind == "host_numa" and quota is None,
            }
    inventory_unknown = bool(inputs.unknown["context"]) or any(
        not path.startswith(("memory.ddr_bank[", "memory.host_numa["))
        for path in inputs.unknown["memory"])
    allocation_ids, used_tensors, source_copies = set(), set(), set()
    categories_seen, allocations = set(), []
    for index, allocation in enumerate(list_at(data.get("allocations"),
                                               "memory.allocations")):
        path = f"memory.allocations[{index}]"
        start = len(inputs.unknown["memory"])
        allocation = expect_keys(allocation, path, ALLOCATION_KEYS)
        name = identifier(allocation.get("id"), f"{path}.id")
        if name in allocation_ids:
            raise BudgetError(f"{path}: duplicate allocation id")
        allocation_ids.add(name)
        category = allocation.get("category")
        if not isinstance(category, str) or category not in CATEGORIES:
            raise BudgetError(f"{path}.category: unknown category")
        if statuses[category] == "absent":
            raise BudgetError(f"{path}: allocation in a declared absent category")
        categories_seen.add(category)
        raw = inputs.integer(allocation.get("bytes"), f"{path}.bytes")
        pinned = inputs.flag(allocation.get("pinned"), f"{path}.pinned")
        refs = list_at(allocation.get("tensor_ids"), f"{path}.tensor_ids")
        if any(not isinstance(ref, str) for ref in refs) or len(set(refs)) != len(refs):
            raise BudgetError(f"{path}: invalid or duplicate tensor references")
        if any(ref not in tensors for ref in refs):
            raise BudgetError(f"{path}: undeclared tensor reference")
        for ref in refs:
            allowed = ROLE_CATEGORIES[tensors[ref]["role"]]
            if category not in allowed:
                raise BudgetError(f"{path}: {tensors[ref]['role']} selections must be "
                                  f"charged to {' or '.join(allowed)}")
        tensor_storage = allocation.get("tensor_storage")
        if category in WEIGHT_CATEGORIES:
            if tensor_storage not in ("resident", "packed_source"):
                raise BudgetError(f"{path}.tensor_storage: expected resident or "
                                  "packed_source")
            if not refs:
                inputs.missing("memory", f"{path}.tensor_ids: actual packed metadata "
                               "required")
                inventory_unknown = True
            sizes = [tensors[ref]["resident_bytes" if tensor_storage == "resident"
                                 else "packed_bytes"] for ref in refs]
            if refs and None not in sizes and raw is not None and add(*sizes) != raw:
                raise BudgetError(f"{path}: weight allocation differs from selected "
                                  "tensor storage bytes")
            if tensor_storage == "packed_source":
                if any(tensors[ref]["resident_status"] != "converted"
                       or tensors[ref]["source_retained"] is not True for ref in refs):
                    raise BudgetError(f"{path}: packed_source copy requires converted "
                                      "tensors with source_retained=true")
                source_copies.update(refs)
            else:
                used_tensors.update(refs)
        elif refs or tensor_storage is not None:
            raise BudgetError(f"{path}: tensor references/storage are reserved for "
                              "weight allocations")
        placements = list_at(allocation.get("placements"), f"{path}.placements")
        if not placements:
            raise BudgetError(f"{path}: allocations cannot be unassigned to banks/NUMA "
                              "nodes")
        placed_raw, placed_aligned, kinds, seen = [], [], set(), set()
        for pindex, placement in enumerate(placements):
            ppath = f"{path}.placements[{pindex}]"
            placement = expect_keys(placement, ppath, PLACEMENT_KEYS)
            resource_id = identifier(placement.get("resource"), f"{ppath}.resource")
            if resource_id not in resources or resource_id in seen:
                raise BudgetError(f"{ppath}: unknown or duplicate resource placement")
            seen.add(resource_id)
            resource = resources[resource_id]
            kinds.add(resource["kind"])
            size = inputs.integer(placement.get("bytes"), f"{ppath}.bytes")
            alignment = inputs.integer(placement.get("alignment_bytes"),
                                       f"{ppath}.alignment_bytes", positive=True)
            if alignment is not None and alignment & (alignment - 1):
                raise BudgetError(f"{ppath}.alignment_bytes: must be a power of two")
            padded = aligned(size, alignment) if size is not None and alignment is not None \
                else size
            known = padded if padded is not None else 0
            resource["known"] = add(resource["known"], known)
            resource["unknown"] |= size is None or alignment is None or raw is None \
                or pinned is None
            if resource["kind"] == "host_numa":
                if pinned is True:
                    resource["pinned_known"] = add(resource["pinned_known"], known)
                    resource["pinned_unknown"] |= size is None or alignment is None \
                        or raw is None
                elif pinned is None:
                    resource["pinned_unknown"] = True
            elif pinned is True:
                raise BudgetError(f"{path}: pinned quotas apply to host allocations only")
            placed_raw.append(size)
            placed_aligned.append(padded if size is not None and alignment is not None
                                  else None)
        if len(kinds) != 1 or ("host_numa" in kinds and len(placements) != 1):
            raise BudgetError(f"{path}: use separate allocations for host/device or "
                              "cross-NUMA physical copies")
        if raw is not None:
            known_raw = add(*(value for value in placed_raw if value is not None))
            if known_raw > raw or (None not in placed_raw and known_raw != raw):
                raise BudgetError(f"{path}: explicit placement byte counts do not match "
                                  "allocation")
        inventory_unknown |= len(inputs.unknown["memory"]) != start
        allocations.append({"id": name, "category": category, "bytes": raw,
                            "aligned_bytes": add(*placed_aligned)
                            if None not in placed_aligned else None,
                            "resources": sorted(seen), "pinned": pinned,
                            "tensor_storage": tensor_storage})
    if set(tensors) != used_tensors:
        raise BudgetError("memory.tensors: unused metadata; include only physically "
                          "allocated resident tensors")
    retained_tensors = {name for name, tensor in tensors.items()
                        if tensor["source_retained"] is True}
    if retained_tensors != source_copies:
        raise BudgetError("memory.tensors: retained packed sources need separately "
                          "placed and charged allocations")
    if any(tensor["resident_status"] == "converted" for tensor in tensors.values()) \
            and "conversion_workspace" not in categories_seen:
        raise BudgetError("memory: conversions need explicit conversion_workspace "
                          "allocation, including zero if evidenced")
    for category, status in statuses.items():
        if status == "complete" and category not in categories_seen:
            raise BudgetError(f"memory.coverage.{category}: complete needs an "
                              "allocation; use explicit absent with reason")
    reports = []
    for resource in resources.values():
        report = {"id": resource["id"], "kind": resource["kind"], **constraint(
            resource["capacity"], resource["known"],
            inventory_unknown or resource["unknown"])}
        if resource["kind"] == "host_numa":
            report["pinned"] = constraint(resource["pinned_quota"],
                                          resource["pinned_known"],
                                          inventory_unknown or resource["pinned_unknown"])
        reports.append(report)
    banks = [resource for resource in resources.values()
             if resource["kind"] == "ddr_bank"]
    nodes = [resource for resource in resources.values()
             if resource["kind"] == "host_numa"]
    device_capacity = inputs.integer(device.get("capacity_bytes"),
                                     "memory.device.capacity_bytes", positive=True)
    device_report = constraint(device_capacity, add(*(bank["known"] for bank in banks)),
                               inventory_unknown or any(bank["unknown"] for bank in banks))
    host_capacity = inputs.integer(host.get("capacity_bytes"),
                                   "memory.host.capacity_bytes", positive=True)
    reserve = inputs.integer(host.get("reserve_bytes"), "memory.host.reserve_bytes")
    fragmentation = inputs.integer(host.get("fragmentation_bytes"),
                                   "memory.host.fragmentation_bytes")
    quota = inputs.integer(host.get("pinned_quota_bytes"),
                           "memory.host.pinned_quota_bytes")
    host_known = add(*(node["known"] for node in nodes),
                     *(value for value in (reserve, fragmentation)
                       if value is not None))
    host_report = constraint(host_capacity, host_known,
                             inventory_unknown or reserve is None or fragmentation is None
                             or any(node["unknown"] for node in nodes))
    host_report["pinned"] = constraint(quota, add(*(node["pinned_known"] for node in nodes)),
                                       inventory_unknown or any(node["pinned_unknown"]
                                                                for node in nodes))
    gates = [device_report, host_report, host_report["pinned"]]
    for report in reports:
        gates.append(report)
        if "pinned" in report:
            gates.append(report["pinned"])
    status = "over_capacity" if any(gate["status"] == "over_capacity" for gate in gates) \
        else ("unknown" if any(gate["status"] == "unknown" for gate in gates) else "fit")
    return {"status": status, "resources": reports, "device_total": device_report,
            "host_total": host_report, "allocations": allocations,
            "tensors": list(tensors.values()),
            "selection_coverage": selection_report}, set(resources), \
        set(node["id"] for node in nodes)


def coverage_memory(data, inputs):
    """Validate memory coverage entries and reject undeclared fields."""
    data = object_at(data, "memory.coverage")
    statuses = {}
    for category in CATEGORIES:
        path = f"memory.coverage.{category}"
        if category not in data:
            inputs.missing("memory", path)
            statuses[category] = "unknown"
            continue
        item = expect_keys(data[category], path, COVERAGE_KEYS)
        status = item.get("status")
        if status not in ("complete", "absent", "unknown"):
            raise BudgetError(f"{path}.status: expected complete, absent or unknown")
        identifier(item.get("reason"), f"{path}.reason")
        if status == "unknown":
            inputs.missing("memory", path)
        statuses[category] = status
    extra = sorted(set(data) - set(CATEGORIES))
    if extra:
        raise BudgetError(f"memory.coverage: unknown categories {extra}")
    return statuses


def reject_unknown_wall(data):
    data = expect_keys(data, "wall", WALL_KEYS)
    coverage = data.get("coverage")
    if isinstance(coverage, dict):
        for category, item in coverage.items():
            if isinstance(item, dict):
                expect_keys(item, f"wall.coverage.{category}", COVERAGE_KEYS)
    numa = data.get("numa")
    if isinstance(numa, dict):
        expect_keys(numa, "wall.numa", NUMA_KEYS)
    link = data.get("link")
    if isinstance(link, dict):
        expect_keys(link, "wall.link", LINK_KEYS)
        for field in ("endpoint_capability", "negotiated"):
            pair = link.get(field)
            if isinstance(pair, dict):
                expect_keys(pair, f"wall.link.{field}", frozenset({"generation", "lanes"}))
        for direction in ("h2d", "d2h"):
            item = link.get(direction)
            if isinstance(item, dict):
                expect_keys(item, f"wall.link.{direction}",
                            frozenset({"payload_bytes_per_second",
                                       "submission_milliseconds"}))
    stages = data.get("stages")
    if isinstance(stages, list):
        from resource_budget import STAGE_CATEGORIES
        for index, stage in enumerate(stages):
            if not isinstance(stage, dict):
                continue
            path = f"wall.stages[{index}]"
            expect_keys(stage, path, STAGE_KEYS)
            scope = stage.get("scope")
            stage_coverage = stage.get("coverage")
            if isinstance(stage_coverage, dict):
                for role, item in stage_coverage.items():
                    if isinstance(item, dict):
                        if scope in STAGE_CATEGORIES \
                                and role not in STAGE_CATEGORIES[scope]:
                            raise BudgetError(f"{path}.coverage: unknown category {role}")
                        expect_keys(item, f"{path}.coverage.{role}", COVERAGE_KEYS)
            overlap = stage.get("overlap")
            if isinstance(overlap, dict):
                expect_keys(overlap, f"{path}.overlap", OVERLAP_KEYS)
                evidence = overlap.get("evidence")
                if isinstance(evidence, dict):
                    expect_keys(evidence, f"{path}.overlap.evidence",
                                frozenset({"kind", "detail"}))
            for branch in ("before", "cpu", "fpga", "after"):
                operations = stage.get(branch)
                if not isinstance(operations, list):
                    continue
                for oindex, operation in enumerate(operations):
                    if isinstance(operation, dict):
                        expect_keys(operation, f"{path}.{branch}[{oindex}]",
                                    OPERATION_KEYS)
    accounting = data.get("accepted_accounting")
    if isinstance(accounting, dict):
        expect_keys(accounting, "wall.accepted_accounting",
                    frozenset({"status", "evidence"}))
        evidence = accounting.get("evidence")
        if isinstance(evidence, dict):
            expect_keys(evidence, "wall.accepted_accounting.evidence",
                        frozenset({"kind", "detail"}))


def accepted_window(data):
    """Require a positive accepted-output window; an empty window is an input error."""
    data = object_at(data, "wall")
    accepted = data.get("accepted_tokens")
    if type(accepted) is not int or not 1 <= accepted <= U64_MAX:
        raise BudgetError("wall.accepted_tokens: expected a positive accepted-output "
                          "token count; an empty window is not undefined, it is invalid")
    if data.get("count_basis") != "accepted_output_tokens":
        raise BudgetError("wall.count_basis: expected accepted_output_tokens")
    return accepted


def analyze(document):
    """Return an analytical report; valid declarations cannot establish readiness."""
    document = expect_keys(document, "budget", TOP_KEYS)
    if type(document.get("schema_version")) is not int \
            or document["schema_version"] != SCHEMA_VERSION:
        raise BudgetError("budget.schema_version: expected integer 2")
    name = identifier(document.get("id"), "budget.id")
    if document.get("input_kind") not in ("reference", "synthetic", "declared"):
        raise BudgetError("budget.input_kind: expected reference, synthetic or declared")
    contract = selection_contract(document)
    inputs = Inputs()
    model = expect_keys(document.get("model"), "model", MODEL_KEYS)
    for key, expected in (("routed_top_k", 10), ("expert_skip", 0)):
        if type(model.get(key)) is not int or model[key] != expected:
            raise BudgetError(f"model.{key}: full top10 and expert-skip=0 are required")
    if model.get("shared_expert") is not True:
        raise BudgetError("model.shared_expert: shared expert must be retained")
    layers = inputs.integer(model.get("layers"), "model.layers", "context", positive=True)
    context = inputs.integer(model.get("context_tokens"), "model.context_tokens",
                             "context", positive=True)
    sessions = inputs.integer(model.get("sessions"), "model.sessions", "context",
                              positive=True)
    memory, _, nodes = memory_budget(document.get("memory"), inputs, contract)
    reject_unknown_wall(document.get("wall"))
    accepted_window(document.get("wall"))
    wall = wall_budget(document.get("wall"), inputs, layers, nodes, memory["status"])
    unknown = sorted({path for paths in inputs.unknown.values() for path in paths})
    complete = not unknown
    over = memory["status"] == "over_capacity"
    selection = memory["selection_coverage"]
    return {
        "schema_version": SCHEMA_VERSION, "id": name,
        "input_kind": document["input_kind"],
        "analytical_only": True, "input_provenance_verified": False,
        "device_ready": False, "inference_ready": False,
        "status": "over_capacity" if over else "unknown" if not complete else "fit",
        "complete": complete, "strict_pass": complete and not over,
        "fit_scope": selection["fit_scope"],
        "selection_contract": {"convention": contract["convention"],
                               "expert_dimension": contract["expert_dimension"],
                               "expert_count": contract["expert_count"],
                               "roles": contract["roles"]},
        "model": {"layers": layers, "context_tokens": context, "sessions": sessions,
                  "routed_top_k": 10, "expert_skip": 0, "shared_expert": True},
        "selection_coverage": selection, "memory": memory, "wall": wall,
        "unknown_inputs": unknown,
        "notes": [
            "Declarations and checked arithmetic only; no packed tensor, artifact "
            "hash, expert mapping or measurement was authenticated.",
            "Each selection is charged as rows-per-item * selected items * row stride; "
            "overlap, out-of-range and whole-extent conflicts are rejected, not merged.",
            "Partial expert-axis coverage is never reported as a fit; it keeps every "
            "affected resource unknown until the selected items are declared.",
            "All allocations are conservatively simultaneous; converted resident bytes, "
            "row padding, scales, conversion overhead, alignment, reserves and "
            "fragmentation must be explicit.",
            "No CPU/FPGA rates are added. Strata remains owner of hot/cold routing, "
            "cache and MTP accepted-state semantics.",
            "Estimated accepted-token rates are not full-wall measurements or proof of "
            "the >10 tok/s objective.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON schema-v2 budget input")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 for incomplete inputs or any capacity/quota excess")
    args = parser.parse_args(argv)
    try:
        document = json.loads(args.input.read_text(encoding="utf-8"),
                              parse_constant=reject_constant,
                              object_pairs_hook=unique_object)
        report = analyze(document)
    # Schema, JSON and Unicode errors are ValueError subclasses; the JSON
    # decoder's protective integer-digit limit can raise a plain ValueError too.
    except (ValueError, OSError, RecursionError) as error:
        parser.exit(2, f"resource budget v2 input error: {error}\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 1 if args.strict and not report["strict_pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())