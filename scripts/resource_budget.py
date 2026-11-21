# Copyright (c) 2026 lvyufeng and contributors
# SPDX-License-Identifier: Apache-2.0
"""Offline, declaration-only memory and dependency-aware wall-budget checks.

This module uses the Python standard library. It reads one JSON input, not paths,
URLs or commands contained in that input. It does not probe a host/device, run an
engine, validate tensor contents or produce performance measurements.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "configs/budgets/kintex7-qwen38-reference.json"
U64_MAX = (1 << 64) - 1
CATEGORIES = (
    "fixed_weights", "hot_experts", "cold_experts", "shared_experts",
    "ple_ngram_weights", "draft_weights", "fixed_buffers", "kv", "qsa_indexer",
    "deltanet_recurrent", "deltanet_conv", "ple_ngram_state", "residual_draft",
    "sampler_history", "cache_metadata", "workspace", "dma_staging", "conversion_workspace",
)
WEIGHT_CATEGORIES = frozenset(CATEGORIES[:6])
WALL_CATEGORIES = (
    "non_expert", "cache_misses", "transfers", "numa_traffic", "sync_merge",
    "sampling", "fallback", "mtp_verify_commit",
)
STAGE_CATEGORIES = {
    "layer": ("fixed", "shared_expert", "routing", "activation", "state",
              "cold_expert", "hot_expert", "transfers", "cache_misses",
              "numa_traffic", "sync_merge"),
    "non_expert": ("embedding", "head", "sampling", "state", "numa_traffic", "fallback"),
    "mtp": ("draft", "verification", "commit_rollback", "state", "transfers",
            "numa_traffic", "sync_merge"),
}
MANDATORY_STAGE_WORK = {
    "layer": {"fixed", "shared_expert", "routing", "activation", "state", "sync_merge"},
    "non_expert": {"embedding", "head", "sampling", "state"},
    "mtp": {"draft", "verification", "commit_rollback", "state", "sync_merge"},
}
GLOBAL_STAGE_ROLES = {
    "non_expert": {"layer": {"fixed", "shared_expert", "routing", "activation", "state"},
                   "non_expert": {"embedding", "head", "state"}, "mtp": {"state"}},
    "cache_misses": {"layer": {"cache_misses"}},
    "transfers": {"layer": {"transfers"}, "mtp": {"transfers"}},
    "numa_traffic": {scope: {"numa_traffic"} for scope in STAGE_CATEGORIES},
    "sync_merge": {"layer": {"sync_merge"}, "mtp": {"sync_merge"}},
    "sampling": {"non_expert": {"sampling"}},
    "fallback": {"non_expert": {"fallback"}},
    "mtp_verify_commit": {"mtp": {"draft", "verification", "commit_rollback", "state"}},
}


class BudgetError(ValueError):
    """Invalid or arithmetically unsafe input; never a successful fit report."""


class Inputs:
    def __init__(self):
        self.unknown = {"memory": [], "wall": [], "context": []}

    def missing(self, domain, path):
        if path not in self.unknown[domain]:
            self.unknown[domain].append(path)

    def integer(self, value, path, domain="memory", positive=False):
        if value is None:
            self.missing(domain, path)
            return None
        if type(value) is not int or not (1 if positive else 0) <= value <= U64_MAX:
            raise BudgetError(f"{path}: expected {'positive' if positive else 'nonnegative'} u64 or null")
        return value

    def number(self, value, path, positive=False):
        if value is None:
            self.missing("wall", path)
            return None
        if type(value) not in (int, float):
            raise BudgetError(f"{path}: expected finite number or null")
        try:
            number = float(value)
        except OverflowError as error:
            raise BudgetError(f"{path}: numeric overflow") from error
        if not math.isfinite(number) or number < 0 or (positive and number == 0):
            raise BudgetError(f"{path}: expected {'positive' if positive else 'nonnegative'} finite number")
        return number

    def text(self, value, path, domain="memory"):
        if value is None:
            self.missing(domain, path)
            return None
        if not isinstance(value, str) or not value.strip():
            raise BudgetError(f"{path}: expected nonempty string or null")
        return value

    def flag(self, value, path, domain="memory"):
        if value is None:
            self.missing(domain, path)
            return None
        if type(value) is not bool:
            raise BudgetError(f"{path}: expected boolean or null")
        return value


def object_at(value, path):
    if not isinstance(value, dict):
        raise BudgetError(f"{path}: expected object")
    return value


def list_at(value, path):
    if not isinstance(value, list):
        raise BudgetError(f"{path}: expected list")
    return value


def identifier(value, path):
    if not isinstance(value, str) or not value.strip():
        raise BudgetError(f"{path}: expected nonempty identifier")
    return value


def add(*values):
    result = sum(values)
    if result > U64_MAX:
        raise BudgetError("u64 byte/count accumulation overflow")
    return result


def multiply(left, right):
    result = left * right
    if result > U64_MAX:
        raise BudgetError("u64 byte/count multiplication overflow")
    return result


def aligned(value, alignment):
    return multiply(add(value, alignment - 1) // alignment, alignment)


def finite_sum(values):
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as error:
        raise BudgetError("wall-time accumulation overflow") from error
    if not math.isfinite(result):
        raise BudgetError("wall-time accumulation overflow")
    return result


def finite_product(left, right):
    result = left * right
    if not math.isfinite(result):
        raise BudgetError("wall-time multiplication overflow")
    return result


def coverage(data, categories, inputs, domain, prefix=None):
    prefix = prefix or f"{domain}.coverage"
    data = object_at(data, prefix)
    extra = set(data) - set(categories)
    if extra:
        raise BudgetError(f"{prefix}: unknown categories {sorted(extra)}")
    statuses = {}
    for category in categories:
        path = f"{prefix}.{category}"
        if category not in data:
            inputs.missing(domain, path)
            statuses[category] = "unknown"
            continue
        item = object_at(data[category], path)
        status = item.get("status")
        if status not in ("complete", "absent", "unknown"):
            raise BudgetError(f"{path}.status: expected complete, absent or unknown")
        identifier(item.get("reason"), f"{path}.reason")
        if status == "unknown":
            inputs.missing(domain, path)
        statuses[category] = status
    return statuses


def declared_evidence(value, path, inputs, domain="wall"):
    if value is None:
        inputs.missing(domain, path)
        return None
    value = object_at(value, path)
    if value.get("kind") not in ("synthetic", "analytical", "measured", "source"):
        raise BudgetError(f"{path}.kind: expected synthetic, analytical, measured or source")
    identifier(value.get("detail"), f"{path}.detail")
    return {"kind": value["kind"], "detail": value["detail"]}


def tensor_metadata(records, inputs):
    """Check supplied packed metadata arithmetic, not artifact contents or kernels."""
    tensors, identities = {}, set()
    for index, record in enumerate(list_at(records, "memory.tensors")):
        path = f"memory.tensors[{index}]"
        record = object_at(record, path)
        name = identifier(record.get("id"), f"{path}.id")
        if name in tensors:
            raise BudgetError(f"{path}: duplicate tensor id")
        source = object_at(record.get("source"), f"{path}.source")
        if source.get("kind") not in ("tensor_metadata", "synthetic"):
            raise BudgetError(f"{path}.source.kind: expected tensor_metadata or synthetic")
        provenance = [inputs.text(source.get(key), f"{path}.source.{key}")
                      for key in ("artifact", "revision", "sha256", "tensor")]
        checksum = provenance[2]
        if checksum is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", checksum):
            raise BudgetError(f"{path}.source.sha256: expected 64 hexadecimal characters")
        selection = source.get("selection")
        if selection is not None:
            identifier(selection, f"{path}.source.selection")
        identity = (*provenance, selection)
        if None not in provenance:
            if identity in identities:
                raise BudgetError(f"{path}: duplicate source tensor/selection metadata")
            identities.add(identity)
        packed = object_at(record.get("packed"), f"{path}.packed")
        dtype = inputs.text(packed.get("dtype"), f"{path}.packed.dtype")
        layout = inputs.text(packed.get("layout"), f"{path}.packed.layout")
        shape = packed.get("shape")
        if shape is None:
            inputs.missing("memory", f"{path}.packed.shape")
        else:
            shape = list_at(shape, f"{path}.packed.shape")
            if not shape:
                raise BudgetError(f"{path}.packed.shape: cannot be empty")
            shape = [inputs.integer(size, f"{path}.packed.shape[{i}]", positive=True)
                     for i, size in enumerate(shape)]
        values = {key: inputs.integer(packed.get(key), f"{path}.packed.{key}",
                                     positive=key in ("block_values", "block_bytes", "row_stride_bytes"))
                  for key in ("block_values", "block_bytes", "row_stride_bytes", "data_bytes", "external_scale_bytes")}
        if shape is not None and None not in shape and values["block_values"] is not None:
            if shape[0] % values["block_values"]:
                raise BudgetError(f"{path}: row shape is not an integral number of declared blocks")
            if values["block_bytes"] is not None and values["row_stride_bytes"] is not None:
                row_min = multiply(shape[0] // values["block_values"], values["block_bytes"])
                if values["row_stride_bytes"] < row_min:
                    raise BudgetError(f"{path}: row stride omits packed blocks/scales")
            if values["row_stride_bytes"] is not None:
                rows = 1
                for size in shape[1:]:
                    rows = multiply(rows, size)
                expected = multiply(rows, values["row_stride_bytes"])
                if values["data_bytes"] is not None and expected != values["data_bytes"]:
                    raise BudgetError(f"{path}: packed data bytes differ from row-stride/shape metadata")
        resident = object_at(record.get("resident"), f"{path}.resident")
        status = resident.get("status")
        if status not in ("native", "converted", "unknown"):
            raise BudgetError(f"{path}.resident.status: expected native, converted or unknown")
        if status == "unknown":
            inputs.missing("memory", f"{path}.resident.status")
        resident_dtype = inputs.text(resident.get("dtype"), f"{path}.resident.dtype")
        resident_layout = inputs.text(resident.get("layout"), f"{path}.resident.layout")
        stored = {key: inputs.integer(resident.get(key), f"{path}.resident.{key}")
                  for key in ("data_bytes", "scale_bytes", "overhead_bytes")}
        retained = inputs.flag(resident.get("source_retained"), f"{path}.resident.source_retained")
        if status == "converted":
            inputs.text(resident.get("evidence"), f"{path}.resident.evidence")
        if status == "native":
            for source_value, target_value, field in (
                (dtype, resident_dtype, "dtype"), (layout, resident_layout, "layout"),
                (values["data_bytes"], stored["data_bytes"], "data_bytes"),
                (values["external_scale_bytes"], stored["scale_bytes"], "scale_bytes"),
            ):
                if source_value is not None and target_value is not None and source_value != target_value:
                    raise BudgetError(f"{path}: native resident {field} differs from packed metadata")
            if retained is True:
                raise BudgetError(f"{path}: native source is the resident copy; separate copies need allocations")
        storage = add(*stored.values()) if None not in stored.values() else None
        packed_storage = add(values["data_bytes"], values["external_scale_bytes"]) if values["data_bytes"] is not None and values["external_scale_bytes"] is not None else None
        tensors[name] = {"id": name, "source_kind": source["kind"], "resident_bytes": storage,
                         "packed_bytes": packed_storage, "resident_status": status,
                         "source_retained": retained}
    return tensors


def constraint(capacity, lower_bound, unknown):
    over = capacity is not None and lower_bound > capacity
    complete = not unknown and capacity is not None
    return {
        "status": "over_capacity" if over else "fit" if complete else "unknown",
        "capacity_bytes": capacity,
        "known_used_lower_bound_bytes": lower_bound,
        "used_bytes": lower_bound if not unknown else None,
        "headroom_bytes": capacity - lower_bound if complete else None,
        "headroom_upper_bound_bytes": capacity - lower_bound if capacity is not None else None,
    }


def memory_budget(data, inputs):
    data = object_at(data, "memory")
    statuses = coverage(data.get("coverage"), CATEGORIES, inputs, "memory")
    tensors = tensor_metadata(data.get("tensors"), inputs)
    device = object_at(data.get("device"), "memory.device")
    host = object_at(data.get("host"), "memory.host")
    resources = {}
    for kind, items in (("ddr_bank", device.get("banks")), ("host_numa", host.get("numa_nodes"))):
        items = list_at(items, f"memory.{kind}")
        if not items:
            raise BudgetError(f"memory.{kind}: must explicitly list resources")
        for index, item in enumerate(items):
            path = f"memory.{kind}[{index}]"
            item = object_at(item, path)
            name = identifier(item.get("id"), f"{path}.id")
            if name in resources:
                raise BudgetError(f"{path}: duplicate resource id")
            start = len(inputs.unknown["memory"])
            capacity = inputs.integer(item.get("capacity_bytes"), f"{path}.capacity_bytes", positive=True)
            reserve = inputs.integer(item.get("reserve_bytes"), f"{path}.reserve_bytes")
            fragmentation = inputs.integer(item.get("fragmentation_bytes"), f"{path}.fragmentation_bytes")
            quota = inputs.integer(item.get("pinned_quota_bytes"), f"{path}.pinned_quota_bytes") if kind == "host_numa" else None
            resources[name] = {
                "id": name, "kind": kind, "capacity": capacity, "pinned_quota": quota,
                "known": add(*(value for value in (reserve, fragmentation) if value is not None)),
                "pinned_known": 0, "unknown": len(inputs.unknown["memory"]) != start,
                "pinned_unknown": kind == "host_numa" and quota is None,
            }
    # Coverage/metadata incompleteness can hide allocations on any resource.
    inventory_unknown = bool(inputs.unknown["context"]) or any(
        not path.startswith(("memory.ddr_bank[", "memory.host_numa["))
        for path in inputs.unknown["memory"]
    )
    allocation_ids, used_tensors, source_copies, categories_seen, allocations = set(), set(), set(), set(), []
    for index, allocation in enumerate(list_at(data.get("allocations"), "memory.allocations")):
        path = f"memory.allocations[{index}]"
        start = len(inputs.unknown["memory"])
        allocation = object_at(allocation, path)
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
        tensor_storage = allocation.get("tensor_storage")
        if category in WEIGHT_CATEGORIES:
            if tensor_storage not in ("resident", "packed_source"):
                raise BudgetError(f"{path}.tensor_storage: expected resident or packed_source")
            if not refs:
                inputs.missing("memory", f"{path}.tensor_ids: actual packed metadata required")
                inventory_unknown = True
            sizes = [tensors[ref]["resident_bytes" if tensor_storage == "resident" else "packed_bytes"] for ref in refs]
            if refs and None not in sizes and raw is not None and add(*sizes) != raw:
                raise BudgetError(f"{path}: weight allocation differs from tensor storage bytes")
            if tensor_storage == "packed_source":
                if any(tensors[ref]["resident_status"] != "converted" or tensors[ref]["source_retained"] is not True for ref in refs):
                    raise BudgetError(f"{path}: packed_source copy requires converted tensors with source_retained=true")
                source_copies.update(refs)
            else:
                used_tensors.update(refs)
        elif refs or tensor_storage is not None:
            raise BudgetError(f"{path}: tensor references/storage are reserved for weight allocations")
        placements = list_at(allocation.get("placements"), f"{path}.placements")
        if not placements:
            raise BudgetError(f"{path}: allocations cannot be unassigned to banks/NUMA nodes")
        placed_raw, placed_aligned, kinds, seen = [], [], set(), set()
        for pindex, placement in enumerate(placements):
            ppath = f"{path}.placements[{pindex}]"
            placement = object_at(placement, ppath)
            resource_id = identifier(placement.get("resource"), f"{ppath}.resource")
            if resource_id not in resources or resource_id in seen:
                raise BudgetError(f"{ppath}: unknown or duplicate resource placement")
            seen.add(resource_id)
            resource = resources[resource_id]
            kinds.add(resource["kind"])
            size = inputs.integer(placement.get("bytes"), f"{ppath}.bytes")
            alignment = inputs.integer(placement.get("alignment_bytes"), f"{ppath}.alignment_bytes", positive=True)
            if alignment is not None and alignment & (alignment - 1):
                raise BudgetError(f"{ppath}.alignment_bytes: must be a power of two")
            padded = aligned(size, alignment) if size is not None and alignment is not None else size
            known = padded if padded is not None else 0
            resource["known"] = add(resource["known"], known)
            resource["unknown"] |= size is None or alignment is None or raw is None or pinned is None
            if resource["kind"] == "host_numa":
                if pinned is True:
                    resource["pinned_known"] = add(resource["pinned_known"], known)
                    resource["pinned_unknown"] |= size is None or alignment is None or raw is None
                elif pinned is None:
                    resource["pinned_unknown"] = True
            elif pinned is True:
                raise BudgetError(f"{path}: pinned quotas apply to host allocations only")
            placed_raw.append(size)
            placed_aligned.append(padded if size is not None and alignment is not None else None)
        if len(kinds) != 1 or ("host_numa" in kinds and len(placements) != 1):
            raise BudgetError(f"{path}: use separate allocations for host/device or cross-NUMA physical copies")
        if raw is not None:
            known_raw = add(*(value for value in placed_raw if value is not None))
            if known_raw > raw or (None not in placed_raw and known_raw != raw):
                raise BudgetError(f"{path}: explicit placement byte counts do not match allocation")
        inventory_unknown |= len(inputs.unknown["memory"]) != start
        allocations.append({"id": name, "category": category, "bytes": raw,
                            "aligned_bytes": add(*placed_aligned) if None not in placed_aligned else None,
                            "resources": sorted(seen), "pinned": pinned})
    if set(tensors) != used_tensors:
        raise BudgetError("memory.tensors: unused metadata; include only physically allocated resident tensors")
    retained_tensors = {name for name, tensor in tensors.items() if tensor["source_retained"] is True}
    if retained_tensors != source_copies:
        raise BudgetError("memory.tensors: retained packed sources need separately placed and charged allocations")
    if any(tensor["resident_status"] == "converted" for tensor in tensors.values()) and "conversion_workspace" not in categories_seen:
        raise BudgetError("memory: conversions need explicit conversion_workspace allocation, including zero if evidenced")
    for category, status in statuses.items():
        if status == "complete" and category not in categories_seen:
            raise BudgetError(f"memory.coverage.{category}: complete needs an allocation; use explicit absent with reason")
    reports = []
    for resource in resources.values():
        report = {"id": resource["id"], "kind": resource["kind"], **constraint(
            resource["capacity"], resource["known"], inventory_unknown or resource["unknown"])}
        if resource["kind"] == "host_numa":
            report["pinned"] = constraint(resource["pinned_quota"], resource["pinned_known"],
                                           inventory_unknown or resource["pinned_unknown"])
        reports.append(report)
    banks = [resource for resource in resources.values() if resource["kind"] == "ddr_bank"]
    nodes = [resource for resource in resources.values() if resource["kind"] == "host_numa"]
    device_capacity = inputs.integer(device.get("capacity_bytes"), "memory.device.capacity_bytes", positive=True)
    device_report = constraint(device_capacity, add(*(bank["known"] for bank in banks)),
                               inventory_unknown or any(bank["unknown"] for bank in banks))
    host_capacity = inputs.integer(host.get("capacity_bytes"), "memory.host.capacity_bytes", positive=True)
    reserve = inputs.integer(host.get("reserve_bytes"), "memory.host.reserve_bytes")
    fragmentation = inputs.integer(host.get("fragmentation_bytes"), "memory.host.fragmentation_bytes")
    quota = inputs.integer(host.get("pinned_quota_bytes"), "memory.host.pinned_quota_bytes")
    host_known = add(*(node["known"] for node in nodes),
                     *(value for value in (reserve, fragmentation) if value is not None))
    host_report = constraint(host_capacity, host_known, inventory_unknown or reserve is None or
                             fragmentation is None or any(node["unknown"] for node in nodes))
    host_report["pinned"] = constraint(quota, add(*(node["pinned_known"] for node in nodes)),
                                       inventory_unknown or any(node["pinned_unknown"] for node in nodes))
    gates = [device_report, host_report, host_report["pinned"]]
    for report in reports:
        gates.append(report)
        if "pinned" in report:
            gates.append(report["pinned"])
    status = "over_capacity" if any(gate["status"] == "over_capacity" for gate in gates) else (
        "unknown" if any(gate["status"] == "unknown" for gate in gates) else "fit")
    return {"status": status, "resources": reports, "device_total": device_report,
            "host_total": host_report, "allocations": allocations, "tensors": list(tensors.values())}, set(resources), set(node["id"] for node in nodes)


def wall_budget(data, inputs, layers, node_ids, residency_status):
    data = object_at(data, "wall")
    mode = data.get("mode")
    if mode not in ("mtp_off", "mtp_on"):
        raise BudgetError("wall.mode: expected mtp_off or mtp_on")
    statuses = coverage(data.get("coverage"), WALL_CATEGORIES, inputs, "wall")
    if mode == "mtp_off" and statuses["mtp_verify_commit"] != "absent":
        raise BudgetError("wall.coverage.mtp_verify_commit: MTP-off must explicitly declare absent")
    if mode == "mtp_on" and statuses["mtp_verify_commit"] == "absent":
        raise BudgetError("wall.coverage.mtp_verify_commit: MTP-on cannot omit verification/commit")
    topology = object_at(data.get("numa"), "wall.numa")
    for key in ("pcie_local_node", "transfer_worker_node"):
        node = inputs.text(topology.get(key), f"wall.numa.{key}", "wall")
        if node is not None and node not in node_ids:
            raise BudgetError(f"wall.numa.{key}: undeclared NUMA node")
    workers = topology.get("cpu_worker_nodes")
    if workers is None:
        inputs.missing("wall", "wall.numa.cpu_worker_nodes")
    else:
        workers = list_at(workers, "wall.numa.cpu_worker_nodes")
        if not workers or any(not isinstance(node, str) or node not in node_ids for node in workers) or len(set(workers)) != len(workers):
            raise BudgetError("wall.numa.cpu_worker_nodes: expected distinct declared nodes")
    placement = inputs.flag(topology.get("page_placement_declared"), "wall.numa.page_placement_declared", "wall")
    if placement is not True:
        inputs.missing("wall", "wall.numa.page_placement_declared")
    link = object_at(data.get("link"), "wall.link")
    # Endpoint capability is informational only; never substitute it for payload rates.
    for field in ("endpoint_capability", "negotiated"):
        pair = object_at(link.get(field), f"wall.link.{field}")
        for key in ("generation", "lanes"):
            inputs.integer(pair.get(key), f"wall.link.{field}.{key}", "wall", positive=True)
    rates = {}
    for direction in ("h2d", "d2h"):
        item = object_at(link.get(direction), f"wall.link.{direction}")
        rates[direction] = (
            inputs.number(item.get("payload_bytes_per_second"), f"wall.link.{direction}.payload_bytes_per_second", positive=True),
            inputs.number(item.get("submission_milliseconds"), f"wall.link.{direction}.submission_milliseconds"),
        )
    operation_ids, stage_ids, reports, layer_repeats, mtp_seen = set(), set(), [], 0, False
    global_seen = set()

    def operations(items, path, branch, scope, local, stage_statuses):
        known_times, exact = [], True
        for index, operation in enumerate(list_at(items, path)):
            opath = f"{path}[{index}]"
            operation = object_at(operation, opath)
            name = identifier(operation.get("id"), f"{opath}.id")
            if name in operation_ids:
                raise BudgetError(f"{opath}: duplicate operation id")
            operation_ids.add(name)
            local[name] = branch
            depends = list_at(operation.get("depends_on"), f"{opath}.depends_on")
            if any(not isinstance(dep, str) for dep in depends) or len(set(depends)) != len(depends):
                raise BudgetError(f"{opath}: invalid/duplicate dependencies")
            for dep in depends:
                if dep not in local or dep == name:
                    raise BudgetError(f"{opath}: dependency is not an earlier operation in this stage")
                if branch in ("cpu", "fpga") and local[dep] in ("cpu", "fpga") and branch != local[dep]:
                    raise BudgetError(f"{opath}: cross-expert-chain dependency is unsupported; move dependent work to after")
            roles = list_at(operation.get("covers"), f"{opath}.covers")
            if not roles or any(not isinstance(role, str) or role not in stage_statuses for role in roles) or len(set(roles)) != len(roles):
                raise BudgetError(f"{opath}.covers: expected distinct stage categories")
            if any(stage_statuses[role] == "absent" for role in roles):
                raise BudgetError(f"{opath}: operation in absent timing category")
            if branch in ("cpu", "fpga"):
                allowed = {("cold_expert" if branch == "cpu" else "hot_expert"),
                           "transfers", "cache_misses", "numa_traffic"}
                if scope != "layer" or any(role not in allowed for role in roles):
                    raise BudgetError(f"{opath}: only cold/hot expert work and its transfers may be in expert chains")
            kind = operation.get("kind")
            if kind not in ("cpu", "fpga", "host", "sync", "merge", "h2d", "d2h"):
                raise BudgetError(f"{opath}.kind: unsupported operation")
            if (branch == "cpu" and kind == "fpga") or (branch == "fpga" and kind == "cpu"):
                raise BudgetError(f"{opath}: compute belongs to the wrong independent branch")
            if branch in ("cpu", "fpga") and kind in ("sync", "merge"):
                raise BudgetError(f"{opath}: synchronization/merge belongs in serial before/after work")
            expert_role = "cold_expert" if branch == "cpu" else "hot_expert"
            if branch in ("cpu", "fpga") and kind == branch and expert_role not in roles:
                raise BudgetError(f"{opath}: expert compute must cover {expert_role}")
            for role, required_kind in (("cold_expert", "cpu"), ("hot_expert", "fpga")):
                if role in roles and kind != required_kind:
                    raise BudgetError(f"{opath}: {role} needs {required_kind} compute, not a transfer or tag")
            if "transfers" in roles and kind not in rates:
                raise BudgetError(f"{opath}: transfers coverage requires byte/submission-charged h2d or d2h")
            if kind in rates:
                if "transfers" not in roles:
                    raise BudgetError(f"{opath}: transfer operation needs transfers coverage")
                size = inputs.integer(operation.get("bytes"), f"{opath}.bytes", "wall")
                count = inputs.integer(operation.get("submissions"), f"{opath}.submissions", "wall")
                if size is not None and size > 0 and count == 0:
                    raise BudgetError(f"{opath}: nonzero transfer requires submissions")
                rate, overhead = rates[kind]
                byte_time = None if size is None or (size > 0 and rate is None) else (0.0 if size == 0 else size / rate * 1000.0)
                submit_time = None if count is None or (count > 0 and overhead is None) else (0.0 if count == 0 else finite_product(count, overhead))
                parts = (byte_time, submit_time)
                if any(value is not None and not math.isfinite(value) for value in parts):
                    raise BudgetError(f"{opath}: transfer-time overflow")
                exact &= None not in parts
                known_times.extend(value for value in parts if value is not None)
            else:
                duration = inputs.number(operation.get("milliseconds"), f"{opath}.milliseconds")
                exact &= duration is not None
                if duration is not None:
                    known_times.append(duration)
        return finite_sum(known_times), exact

    for index, stage in enumerate(list_at(data.get("stages"), "wall.stages")):
        path = f"wall.stages[{index}]"
        stage = object_at(stage, path)
        name = identifier(stage.get("id"), f"{path}.id")
        if name in stage_ids:
            raise BudgetError(f"{path}: duplicate stage id")
        stage_ids.add(name)
        scope = stage.get("scope")
        if scope not in ("layer", "non_expert", "mtp"):
            raise BudgetError(f"{path}.scope: expected layer, non_expert or mtp")
        timing_start = len(inputs.unknown["wall"])
        stage_statuses = coverage(stage.get("coverage"), STAGE_CATEGORIES[scope], inputs, "wall", f"{path}.coverage")
        for role in MANDATORY_STAGE_WORK[scope]:
            if stage_statuses[role] == "absent":
                raise BudgetError(f"{path}.coverage.{role}: mandatory work cannot be absent")
        if scope == "layer" and all(stage_statuses[role] == "absent" for role in ("cold_expert", "hot_expert")):
            raise BudgetError(f"{path}.coverage: full top10 cannot omit both cold and hot expert computation")
        repeat = inputs.integer(stage.get("repeat"), f"{path}.repeat", "wall", positive=True)
        if scope == "layer" and repeat is not None:
            layer_repeats = add(layer_repeats, repeat)
        if scope == "mtp":
            mtp_seen = True
            if mode == "mtp_off":
                raise BudgetError(f"{path}: MTP stage in MTP-off experiment")
        overlap = object_at(stage.get("overlap"), f"{path}.overlap")
        overlap_mode = overlap.get("mode")
        if overlap_mode not in ("serial", "independent_experts"):
            raise BudgetError(f"{path}.overlap.mode: unsupported overlap model")
        supported, evidence = True, None
        if overlap_mode == "independent_experts":
            if scope != "layer":
                raise BudgetError(f"{path}: overlap is only modeled for layer expert branches")
            supported = inputs.flag(overlap.get("supported"), f"{path}.overlap.supported", "wall") is True
            evidence = declared_evidence(overlap.get("evidence"), f"{path}.overlap.evidence", inputs)
            supported &= evidence is not None
            if not supported:
                inputs.missing("wall", f"{path}.overlap: unverified independence")
        local = {}
        chains = {branch: operations(stage.get(branch), f"{path}.{branch}", branch,
                                      scope, local, stage_statuses)
                  for branch in ("before", "cpu", "fpga", "after")}
        seen_roles = {role for branch in ("before", "cpu", "fpga", "after")
                      for operation in stage[branch] for role in operation["covers"]}
        for role, status in stage_statuses.items():
            if status == "complete" and role not in seen_roles:
                raise BudgetError(f"{path}.coverage.{role}: complete needs a covering operation")
        for category, by_scope in GLOBAL_STAGE_ROLES.items():
            roles = by_scope.get(scope, set())
            if statuses[category] == "absent" and any(stage_statuses[role] != "absent" for role in roles):
                raise BudgetError(f"wall.coverage.{category}: absent conflicts with {path} roles/operations")
            if roles & seen_roles:
                global_seen.add(category)
        branch_time = finite_sum((chains["cpu"][0], chains["fpga"][0])) if overlap_mode == "serial" else max(chains["cpu"][0], chains["fpga"][0])
        lower = finite_sum((chains["before"][0], branch_time, chains["after"][0]))
        complete = supported and repeat is not None and all(item[1] for item in chains.values()) and len(inputs.unknown["wall"]) == timing_start
        reports.append({"id": name, "scope": scope, "repeat": repeat, "overlap": overlap_mode,
                        "independence_evidence": evidence,
                        "known_lower_bound_milliseconds": finite_product(lower, repeat if repeat is not None else 1),
                        "milliseconds": finite_product(lower, repeat) if complete else None})
    if not reports:
        inputs.missing("wall", "wall.stages: complete decode path required")
    for category, status in statuses.items():
        if status == "complete" and category not in global_seen:
            raise BudgetError(f"wall.coverage.{category}: complete needs a mapped covering operation")
    if layers is not None and layer_repeats != layers:
        # A repeated stage accounts for the selected layer count, not a device-only subset.
        inputs.missing("wall", "wall.stages: layer repeats differ from model.layers")
    if not any(report["scope"] == "non_expert" for report in reports):
        inputs.missing("wall", "wall.stages: non-expert/head/sampling path required")
    if mode == "mtp_on" and not mtp_seen:
        inputs.missing("wall", "wall.stages: MTP verification/commit path required")
    accepted = inputs.integer(data.get("accepted_tokens"), "wall.accepted_tokens", "wall")
    if accepted == 0:
        inputs.missing("wall", "wall.accepted_tokens: no accepted output in this window")
    if data.get("count_basis") != "accepted_output_tokens":
        raise BudgetError("wall.count_basis: only accepted output tokens may be counted")
    if mode == "mtp_on":
        accounting = object_at(data.get("accepted_accounting"), "wall.accepted_accounting")
        if accounting.get("status") not in ("declared", "unknown"):
            raise BudgetError("wall.accepted_accounting.status: expected declared or unknown")
        if accounting["status"] == "unknown":
            inputs.missing("wall", "wall.accepted_accounting.status")
        declared_evidence(accounting.get("evidence"), "wall.accepted_accounting.evidence", inputs)
    if residency_status != "fit":
        inputs.missing("wall", "wall.residency: complete fitting allocation inventory required for a wall estimate")
    lower = finite_sum(report["known_lower_bound_milliseconds"] for report in reports)
    complete = not inputs.unknown["wall"] and not inputs.unknown["context"] and all(report["milliseconds"] is not None for report in reports)
    elapsed = lower if complete else None
    if elapsed == 0:
        raise BudgetError("wall: a complete decode window must have positive duration")
    rate = accepted / elapsed * 1000.0 if complete and accepted is not None else None
    if rate is not None and not math.isfinite(rate):
        raise BudgetError("wall: accepted-token rate overflow")
    return {"status": "estimated" if complete else "unknown", "mode": mode,
            "milliseconds": elapsed, "known_lower_bound_milliseconds": lower,
            "accepted_tokens": accepted, "estimated_accepted_tokens_per_second": rate,
            "estimate_is_measurement": False, "stages": reports}


def analyze(document):
    """Return an analytical report; valid declarations cannot establish readiness."""
    document = object_at(document, "budget")
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise BudgetError("budget.schema_version: expected integer 1")
    name = identifier(document.get("id"), "budget.id")
    if document.get("input_kind") not in ("reference", "synthetic", "declared"):
        raise BudgetError("budget.input_kind: expected reference, synthetic or declared")
    inputs = Inputs()
    model = object_at(document.get("model"), "model")
    for key, expected in (("routed_top_k", 10), ("expert_skip", 0)):
        if type(model.get(key)) is not int or model[key] != expected:
            raise BudgetError(f"model.{key}: full top10 and expert-skip=0 are required")
    if model.get("shared_expert") is not True:
        raise BudgetError("model.shared_expert: shared expert must be retained")
    layers = inputs.integer(model.get("layers"), "model.layers", "context", positive=True)
    context = inputs.integer(model.get("context_tokens"), "model.context_tokens", "context", positive=True)
    sessions = inputs.integer(model.get("sessions"), "model.sessions", "context", positive=True)
    memory, _, nodes = memory_budget(document.get("memory"), inputs)
    wall = wall_budget(document.get("wall"), inputs, layers, nodes, memory["status"])
    unknown = sorted({path for paths in inputs.unknown.values() for path in paths})
    complete = not unknown
    over = memory["status"] == "over_capacity"
    return {
        "schema_version": 1, "id": name, "input_kind": document["input_kind"],
        "analytical_only": True, "input_provenance_verified": False,
        "device_ready": False, "inference_ready": False,
        "status": "over_capacity" if over else "unknown" if not complete else "fit",
        "complete": complete, "strict_pass": complete and not over,
        "model": {"layers": layers, "context_tokens": context, "sessions": sessions,
                  "routed_top_k": 10, "expert_skip": 0, "shared_expert": True},
        "memory": memory, "wall": wall, "unknown_inputs": unknown,
        "notes": [
            "Declarations and checked arithmetic only; no input artifact, measurement or overlap evidence was authenticated.",
            "All allocations are conservatively simultaneous; physical copies, row padding, scales, conversion overhead, alignment, reserves and fragmentation must be explicit.",
            "Endpoint capability is not the negotiated PCIe link or directional payload bandwidth.",
            "No CPU/FPGA rates are added. Strata remains owner of hot/cold routing, cache and MTP accepted-state semantics.",
            "Estimated accepted-token rates are not full-wall measurements or proof of the >10 tok/s objective.",
        ],
    }


def reject_constant(value):
    raise BudgetError(f"JSON nonfinite constant {value} is forbidden")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BudgetError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT,
                        help="JSON budget (default: repository conservative reference)")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 for incomplete inputs or any capacity/quota excess")
    args = parser.parse_args(argv)
    try:
        document = json.loads(args.input.read_text(encoding="utf-8"),
                              parse_constant=reject_constant, object_pairs_hook=unique_object)
        report = analyze(document)
    # Schema, JSON and Unicode errors are ValueError subclasses; the JSON
    # decoder's protective integer-digit limit can raise a plain ValueError too.
    except (ValueError, OSError, RecursionError) as error:
        parser.exit(2, f"resource budget input error: {error}\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 1 if args.strict and not report["strict_pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
