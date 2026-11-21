# Copyright (c) 2026 lvyufeng and contributors
# SPDX-License-Identifier: Apache-2.0
"""Check inert CPU evidence JSON offline; never build, fetch or execute its data.

Default: check the partial template and report missing observations with exit 0.
--strict-complete: require complete, declared measured matrix evidence, including
observed Reject/error outcomes. --strict-accept: additionally require every
per-element abs+rel check and every observed placement/ISA gate to pass.
Neither mode authenticates measurements, authorizes experiments or proves
inference readiness. Only the input JSON and the integration source lock are read.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "strata-cpu-cold-expert-v1"
STRATA_REVISION = "3142b66b96e9c0e775e30f620b11b191a1427c2a"
STRATA_REPOSITORY = "https://github.com/lvyufeng/Strata"
FORMATS = ("IQ2_XS", "Q2_0", "IQ3_S")
ROLES = ("gate", "up", "down")
POLICIES = ("local", "remote", "interleave", "contention")
CACHE_STATES = ("warm", "cold")
STAGES = ("x_quant", "gate", "up", "silu_multiply", "h_quant", "down", "synchronize")
SCENARIOS = ("zero", "signed", "scale-edges", "grid-sign-edges")
FEATURES = ("osxsave", "avx", "fma", "f16c", "avx2")
# Pinned types.cpp / strata-quant-common.h metadata, not quantization code.
# Q8_0 is an explicitly audited auxiliary mixed-role format, not a claimed model
# choice. Other auxiliary types require a reviewed protocol/schema extension.
TYPE_INFO = {
    "IQ2_XS": (17, 256, 74, "Q8_K"),
    "Q2_0": (42, 64, 18, "Q8_0"),
    "IQ3_S": (21, 256, 110, "Q8_K"),
    "Q8_0": (8, 32, 34, "Q8_0"),
}
LAYOUT = "native-row-packed-gate-up-down-v1"
KINDS = ("partial", "synthetic", "measured")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")


class EvidenceError(ValueError):
    """Malformed or internally contradictory evidence (not merely unknown)."""


def _fail(path, message):
    raise EvidenceError(f"{path}: {message}")


def _object(value, path, keys):
    if not isinstance(value, dict):
        _fail(path, "expected an object")
    missing, extra = set(keys) - value.keys(), value.keys() - set(keys)
    if missing or extra:
        _fail(path, f"schema fields missing={sorted(missing)}, unknown={sorted(extra)}")
    return value


def _finite_tree(value, path="$"):
    if type(value) in (int, float) and not _number(value):
        _fail(path, "nonfinite or overflowing number")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                _fail(path, "object keys must be strings")
            _finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, f"{path}[{index}]")


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def vector_sha256(values):
    """Hash canonical finite float64-JSON values, not original binary tensors.

    This is a reproducible envelope checksum. It neither dequantizes weights nor
    certifies that values came from a CPU. Tensor/binary hashes are separate.
    """
    if not isinstance(values, list) or not all(_number(item) for item in values):
        raise EvidenceError("vector hash requires a finite numeric list")
    normalized = [float(item) for item in values]
    encoded = json.dumps(normalized, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


class Checker:
    def __init__(self):
        self.missing = set()

    def need(self, value, path, test, description):
        if value is None:
            self.missing.add(path)
            return None
        if not test(value):
            _fail(path, description)
        return value

    def text(self, value, path):
        return self.need(value, path, lambda x: isinstance(x, str) and bool(x.strip()), "expected nonempty text")

    def integer(self, value, path, minimum=0):
        return self.need(value, path, lambda x: type(x) is int and x >= minimum, f"expected integer >= {minimum}")

    def numeric(self, value, path, minimum=0):
        return self.need(value, path, lambda x: _number(x) and x >= minimum, f"expected finite number >= {minimum}")

    def boolean(self, value, path):
        return self.need(value, path, lambda x: type(x) is bool, "expected boolean")

    def enum(self, value, path, choices):
        return self.need(value, path, lambda x: isinstance(x, str) and x in choices, f"expected one of {tuple(choices)}")

    def digest(self, value, path, revision=False):
        regex = HEX40 if revision else HEX64
        return self.need(value, path, lambda x: isinstance(x, str) and bool(regex.fullmatch(x)), "expected lowercase hexadecimal digest")

    def sequence(self, value, path, nonempty=True):
        return self.need(value, path, lambda x: isinstance(x, list) and (not nonempty or bool(x)), "expected a nonempty list" if nonempty else "expected a list")

    def integers(self, value, path, minimum=0, nonempty=True, distinct=True):
        values = self.sequence(value, path, nonempty)
        if values is None:
            return None
        for i, item in enumerate(values):
            self.integer(item, f"{path}[{i}]", minimum)
        if any(item is None for item in values):
            _fail(path, "list items cannot be unknown; make the whole list null")
        if distinct and len(set(values)) != len(values):
            _fail(path, "duplicate list items")
        return values

    def strings(self, value, path, nonempty=True):
        values = self.sequence(value, path, nonempty)
        if values is None:
            return None
        for i, item in enumerate(values):
            self.text(item, f"{path}[{i}]")
        if any(item is None for item in values):
            _fail(path, "list items cannot be null")
        return values

    def exact_set(self, value, path, expected):
        values = self.strings(value, path)
        if values is not None and (len(values) != len(set(values)) or set(values) != set(expected)):
            _fail(path, f"must contain exactly {tuple(expected)}")
        return values


def _equal_known(value, expected, path):
    if value is not None and expected is not None and value != expected:
        _fail(path, f"does not match derived value {expected!r}")


def _validate_source(check, source, lock):
    _object(source, "source", ("repository", "revision"))
    check.text(source["repository"], "source.repository")
    check.digest(source["revision"], "source.revision", True)
    dependencies = lock.get("dependencies") if isinstance(lock, dict) else None
    if not isinstance(dependencies, dict) or not isinstance(dependencies.get("strata"), dict):
        _fail("lock", "missing locked Strata source")
    pinned = dependencies["strata"]
    if pinned.get("repository") != STRATA_REPOSITORY or pinned.get("revision") != STRATA_REVISION:
        _fail("lock", "this protocol requires its exact audited Strata source revision")
    _equal_known(source["repository"], STRATA_REPOSITORY, "source.repository")
    _equal_known(source["revision"], STRATA_REVISION, "source.revision")


def _validate_build(check, build):
    _object(build, "build", ("compiler", "compiler_version", "flags", "avx2_translation_unit_flags", "quant_backend", "ggml_revision", "compiled_isa_floor", "binary_sha256", "build_log_sha256", "environment"))
    for name in ("compiler", "compiler_version"):
        check.text(build[name], f"build.{name}")
    check.strings(build["flags"], "build.flags")
    flags = check.strings(build["avx2_translation_unit_flags"], "build.avx2_translation_unit_flags")
    if flags is not None and not ({"-mavx2", "-mfma", "-mf16c"} <= set(flags) or "/arch:AVX2" in flags):
        _fail("build.avx2_translation_unit_flags", "must record the audited AVX2/FMA/F16C flags")
    backend = check.enum(build["quant_backend"], "build.quant_backend", ("strata-own", "ggml-pinned"))
    if backend == "ggml-pinned" or build["ggml_revision"] is not None:
        check.digest(build["ggml_revision"], "build.ggml_revision", True)
    # This schema certifies no AVX512 runtime guard and does not support a higher
    # compiled ISA floor. ISA caps cannot make a higher-floor binary safe.
    check.enum(build["compiled_isa_floor"], "build.compiled_isa_floor", ("none", "avx", "avx2"))
    for name in ("binary_sha256", "build_log_sha256"):
        check.digest(build[name], f"build.{name}")
    environment = check.need(build["environment"], "build.environment", lambda x: isinstance(x, dict), "expected environment object")
    if environment is not None:
        for name, value in environment.items():
            if not isinstance(name, str) or not name or not isinstance(value, str):
                _fail("build.environment", "environment names and values must be strings; values are inert")


def _validate_host(check, host):
    _object(host, "host", ("label", "architecture", "os", "sockets", "numa_nodes", "topology", "features", "cpuid_1_ecx", "cpuid_7_ebx", "xgetbv_executed", "xcr0", "avx2_usable", "feature_probe_sha256", "topology_sha256"))
    for name in ("label", "os"):
        check.text(host[name], f"host.{name}")
    check.enum(host["architecture"], "host.architecture", ("x86_64",))
    sockets = check.integer(host["sockets"], "host.sockets", 1)
    nodes = check.integers(host["numa_nodes"], "host.numa_nodes")
    topology = check.sequence(host["topology"], "host.topology")
    cpus = {}
    if topology is not None:
        for i, item in enumerate(topology):
            path = f"host.topology[{i}]"
            _object(item, path, ("cpu", "core", "socket", "node"))
            for key in item:
                check.integer(item[key], f"{path}.{key}")
            if any(value is None for value in item.values()):
                continue
            if item["cpu"] in cpus:
                _fail(path, "duplicate topology CPU")
            if nodes is not None and item["node"] not in nodes:
                _fail(path, "topology node not declared")
            if sockets is not None and item["socket"] >= sockets:
                _fail(path, "socket index exceeds socket count")
            cpus[item["cpu"]] = item
        if nodes is not None and topology and all(all(v is not None for v in item.values()) for item in topology):
            if set(nodes) != {item["node"] for item in topology}:
                _fail("host.topology", "each declared NUMA node needs observed CPU topology")
    _object(host["features"], "host.features", FEATURES)
    for name in FEATURES:
        check.boolean(host["features"][name], f"host.features.{name}")
    cpuid1 = check.integer(host["cpuid_1_ecx"], "host.cpuid_1_ecx")
    cpuid7 = check.integer(host["cpuid_7_ebx"], "host.cpuid_7_ebx")
    for value, path in ((cpuid1, "host.cpuid_1_ecx"), (cpuid7, "host.cpuid_7_ebx")):
        if value is not None and value >= 2**32:
            _fail(path, "CPUID register exceeds uint32")
    for name, bit in (("fma", 12), ("osxsave", 27), ("avx", 28), ("f16c", 29)):
        if cpuid1 is not None:
            _equal_known(host["features"][name], bool(cpuid1 & (1 << bit)), f"host.features.{name}")
    if cpuid7 is not None:
        _equal_known(host["features"]["avx2"], bool(cpuid7 & (1 << 5)), "host.features.avx2")
    executed = check.boolean(host["xgetbv_executed"], "host.xgetbv_executed")
    xcr0 = check.integer(host["xcr0"], "host.xcr0") if executed is not False else None
    if xcr0 is not None and xcr0 >= 2**64:
        _fail("host.xcr0", "XGETBV register exceeds uint64")
    if executed is False and host["xcr0"] is not None:
        _fail("host.xcr0", "unexecuted XGETBV must remain null, not a fabricated zero")
    if executed is True and host["features"]["osxsave"] is False:
        _fail("host.xgetbv_executed", "XGETBV cannot execute without prerequisite OSXSAVE")
    declared = check.boolean(host["avx2_usable"], "host.avx2_usable")
    usable = None
    if all(host["features"][name] is not None for name in FEATURES):
        if not all(host["features"].values()):
            usable = False
        elif executed is True and xcr0 is not None:
            usable = (xcr0 & 0x6) == 0x6
        elif executed is False:
            _fail("host.xgetbv_executed", "all AVX prerequisites present but OS YMM state was not probed")
        _equal_known(declared, usable, "host.avx2_usable")
    for name in ("feature_probe_sha256", "topology_sha256"):
        check.digest(host[name], f"host.{name}")
    return cpus, nodes, usable


def _validate_matrix(check, matrix, host_nodes):
    _object(matrix, "matrix", ("formats", "roles", "batch_sizes", "thread_counts", "numa_policies", "cache_states", "worker_nodes", "repeat_count", "warmup_count"))
    for name, expected in (("formats", FORMATS), ("roles", ROLES), ("numa_policies", POLICIES), ("cache_states", CACHE_STATES)):
        check.exact_set(matrix[name], f"matrix.{name}", expected)
    batches = check.integers(matrix["batch_sizes"], "matrix.batch_sizes", 1)
    if batches is not None and (1 not in batches or not any(2 <= item <= 8 for item in batches) or any(item > 8 for item in batches)):
        _fail("matrix.batch_sizes", "requires single-token and small batch (2..8), never a decode token count")
    threads = check.integers(matrix["thread_counts"], "matrix.thread_counts", 1)
    if threads is not None and (1 not in threads or not any(item >= 2 for item in threads)):
        _fail("matrix.thread_counts", "requires single-thread and multi-thread scaling")
    nodes = check.integers(matrix["worker_nodes"], "matrix.worker_nodes")
    if nodes is not None:
        if len(nodes) < 2:
            _fail("matrix.worker_nodes", "dual-NUMA evidence requires at least two selected nodes")
        if host_nodes is not None and not set(nodes) <= set(host_nodes):
            _fail("matrix.worker_nodes", "selected worker node absent from observed topology")
    check.integer(matrix["repeat_count"], "matrix.repeat_count", 5)
    check.integer(matrix["warmup_count"], "matrix.warmup_count", 1)
    return nodes


def _validate_profiles(check, profiles, kind):
    values = check.sequence(profiles, "profiles")
    if values is None:
        return {}, {}
    by_id, by_target = {}, {}
    for i, profile in enumerate(values):
        path = f"profiles[{i}]"
        _object(profile, path, ("id", "target_format", "data_origin", "model_id", "model_sha256", "tensor_metadata_sha256", "layer", "expert", "n_embd", "n_ff", "layout", "blob_sha256", "blob_bytes", "roles"))
        identifier = check.text(profile["id"], f"{path}.id")
        target = check.enum(profile["target_format"], f"{path}.target_format", FORMATS)
        origin = check.enum(profile["data_origin"], f"{path}.data_origin", ("model-derived", "synthetic-fixture"))
        if kind == "measured" and origin == "synthetic-fixture":
            _fail(path, "synthetic model fixture cannot be declared model-derived measured evidence")
        if identifier is not None:
            if identifier in by_id:
                _fail(path, "duplicate profile id")
            by_id[identifier] = profile
        if target is not None:
            if target in by_target:
                _fail(path, "exactly one selected mixed-role profile per target format")
            by_target[target] = profile
        check.text(profile["model_id"], f"{path}.model_id")
        for name in ("model_sha256", "tensor_metadata_sha256", "blob_sha256"):
            check.digest(profile[name], f"{path}.{name}")
        for name in ("layer", "expert"):
            check.integer(profile[name], f"{path}.{name}")
        n_embd = check.integer(profile["n_embd"], f"{path}.n_embd", 1)
        n_ff = check.integer(profile["n_ff"], f"{path}.n_ff", 1)
        check.enum(profile["layout"], f"{path}.layout", (LAYOUT,))
        check.integer(profile["blob_bytes"], f"{path}.blob_bytes", 1)
        _object(profile["roles"], f"{path}.roles", ROLES)
        derived_rows = {}
        for role in ROLES:
            row, role_path = profile["roles"][role], f"{path}.roles.{role}"
            _object(row, role_path, ("tensor_name", "format", "type_id", "shape", "block_values", "block_bytes", "row_bytes", "offset_bytes", "tensor_sha256", "native_vec_dot_activation"))
            check.text(row["tensor_name"], f"{role_path}.tensor_name")
            fmt = check.enum(row["format"], f"{role_path}.format", TYPE_INFO)
            shape = check.integers(row["shape"], f"{role_path}.shape", 1, distinct=False)
            if shape is not None and len(shape) != 2:
                _fail(f"{role_path}.shape", "one expert projection needs [rows, columns]")
            for name in ("type_id", "block_values", "block_bytes", "row_bytes"):
                check.integer(row[name], f"{role_path}.{name}", 1)
            check.integer(row["offset_bytes"], f"{role_path}.offset_bytes")
            check.digest(row["tensor_sha256"], f"{role_path}.tensor_sha256")
            check.enum(row["native_vec_dot_activation"], f"{role_path}.native_vec_dot_activation", ("Q8_K", "Q8_0"))
            if fmt is not None:
                type_id, block_values, block_bytes, activation = TYPE_INFO[fmt]
                for name, expected in (("type_id", type_id), ("block_values", block_values), ("block_bytes", block_bytes), ("native_vec_dot_activation", activation)):
                    _equal_known(row[name], expected, f"{role_path}.{name}")
                columns = n_ff if role == "down" else n_embd
                rows = n_embd if role == "down" else n_ff
                if columns is not None:
                    if columns % block_values:
                        _fail(role_path, "projection columns are not whole packed weight blocks")
                    act_block = 256 if activation == "Q8_K" else 32
                    if columns % act_block:
                        _fail(role_path, "projection columns are not whole native activation blocks")
                    derived_rows[role] = (columns // block_values) * block_bytes
                    _equal_known(row["row_bytes"], derived_rows[role], f"{role_path}.row_bytes")
                if rows is not None and columns is not None:
                    _equal_known(shape, [rows, columns], f"{role_path}.shape")
        gate, up = profile["roles"]["gate"], profile["roles"]["up"]
        _equal_known(gate["format"], target, f"{path}.roles.gate.format")
        _equal_known(up["format"], gate["format"], f"{path}.roles.up.format")
        _equal_known(gate["offset_bytes"], 0, f"{path}.roles.gate.offset_bytes")
        if n_ff is not None and n_embd is not None and set(derived_rows) == set(ROLES):
            up_offset = derived_rows["gate"] * n_ff
            down_offset = 2 * up_offset
            _equal_known(up["offset_bytes"], up_offset, f"{path}.roles.up.offset_bytes")
            _equal_known(profile["roles"]["down"]["offset_bytes"], down_offset, f"{path}.roles.down.offset_bytes")
            _equal_known(profile["blob_bytes"], down_offset + derived_rows["down"] * n_embd, f"{path}.blob_bytes")
    for target in FORMATS:
        if target not in by_target:
            check.missing.add(f"profiles.target_format={target}")
    return by_id, by_target


def _placement(check, value, path, threads, worker_node, policy, cpus, host_nodes):
    _object(value, path, ("requested", "observed"))
    requested, observed = value["requested"], value["observed"]
    _object(requested, f"{path}.requested", ("cpu_ids", "memory_nodes", "memory_policy", "first_touch"))
    check.integers(requested["cpu_ids"], f"{path}.requested.cpu_ids")
    check.integers(requested["memory_nodes"], f"{path}.requested.memory_nodes")
    check.enum(requested["memory_policy"], f"{path}.requested.memory_policy", POLICIES)
    check.text(requested["first_touch"], f"{path}.requested.first_touch")
    if requested["memory_policy"] is not None and policy is not None and requested["memory_policy"] != policy:
        _fail(f"{path}.requested.memory_policy", "requested strategy differs from the matrix case")
    requested_cpus, requested_nodes = requested["cpu_ids"], requested["memory_nodes"]
    if requested_cpus is not None and cpus:
        if not set(requested_cpus) <= cpus.keys():
            _fail(f"{path}.requested.cpu_ids", "requested CPU absent from selected host topology")
        if worker_node is not None and any(cpus[cpu]["node"] != worker_node for cpu in requested_cpus):
            _fail(f"{path}.requested.cpu_ids", "requested worker affinity contradicts the selected worker node")
    if requested_nodes is not None:
        if host_nodes is not None and not set(requested_nodes) <= set(host_nodes):
            _fail(f"{path}.requested.memory_nodes", "requested node absent from selected host topology")
        if worker_node is not None:
            valid_requested = (
                policy in ("local", "contention") and set(requested_nodes) == {worker_node}
                or policy == "remote" and worker_node not in requested_nodes
                or policy == "interleave" and len(requested_nodes) >= 2
                or policy is None
            )
            if not valid_requested:
                _fail(f"{path}.requested.memory_nodes", "requested placement contradicts the matrix strategy")
    _object(observed, f"{path}.observed", ("workers", "memory_policy", "page_counts", "total_pages", "first_touch", "contention", "affinity_sha256", "memory_sha256"))
    workers = check.sequence(observed["workers"], f"{path}.observed.workers")
    conform = True
    if workers is not None:
        if threads is not None and len(workers) != threads:
            _fail(f"{path}.observed.workers", "one effective-affinity observation required per worker thread")
        worker_ids = set()
        for i, worker in enumerate(workers):
            worker_path = f"{path}.observed.workers[{i}]"
            _object(worker, worker_path, ("thread", "effective_cpu_ids", "sampled_cpu_ids"))
            thread = check.integer(worker["thread"], f"{worker_path}.thread")
            if thread is not None:
                if thread in worker_ids or (threads is not None and thread >= threads):
                    _fail(worker_path, "duplicate or out-of-range worker thread")
                worker_ids.add(thread)
            effective = check.integers(worker["effective_cpu_ids"], f"{worker_path}.effective_cpu_ids")
            sampled = check.integers(worker["sampled_cpu_ids"], f"{worker_path}.sampled_cpu_ids")
            if effective is not None and sampled is not None and not set(sampled) <= set(effective):
                _fail(worker_path, "sampled CPU outside the reported effective affinity")
            for ids in (effective, sampled):
                if ids is None:
                    conform = False
                    continue
                if cpus and not set(ids) <= cpus.keys():
                    _fail(worker_path, "observed CPU absent from host topology")
                if not cpus or worker_node is None or any(cpus[cpu]["node"] != worker_node for cpu in ids if cpu in cpus):
                    conform = False
    else:
        conform = False
    observed_policy = check.enum(observed["memory_policy"], f"{path}.observed.memory_policy", POLICIES)
    conform &= observed_policy is not None and observed_policy == policy
    pages = check.sequence(observed["page_counts"], f"{path}.observed.page_counts")
    total = check.integer(observed["total_pages"], f"{path}.observed.total_pages", 1)
    page_nodes, page_total = set(), 0
    if pages is not None:
        for i, page in enumerate(pages):
            page_path = f"{path}.observed.page_counts[{i}]"
            _object(page, page_path, ("node", "pages"))
            node = check.integer(page["node"], f"{page_path}.node")
            count = check.integer(page["pages"], f"{page_path}.pages", 1)
            if node is not None:
                if node in page_nodes or (host_nodes is not None and node not in host_nodes):
                    _fail(page_path, "duplicate or undeclared page-residency node")
                page_nodes.add(node)
            if count is not None:
                page_total += count
        if all(item["pages"] is not None for item in pages):
            _equal_known(total, page_total, f"{path}.observed.total_pages")
    touch_path, touch = f"{path}.observed.first_touch", observed["first_touch"]
    _object(touch, touch_path, ("cpu_ids", "node_ids", "pages_touched", "evidence_sha256"))
    touch_cpus = check.integers(touch["cpu_ids"], f"{touch_path}.cpu_ids")
    touch_nodes = check.integers(touch["node_ids"], f"{touch_path}.node_ids")
    touched = check.integer(touch["pages_touched"], f"{touch_path}.pages_touched", 1)
    check.digest(touch["evidence_sha256"], f"{touch_path}.evidence_sha256")
    if touch_cpus is not None and cpus:
        if not set(touch_cpus) <= cpus.keys():
            _fail(touch_path, "first-touch CPU absent from host topology")
        _equal_known(set(touch_nodes) if touch_nodes is not None else None, {cpus[cpu]["node"] for cpu in touch_cpus}, f"{touch_path}.node_ids")
    if total is not None and touched is not None:
        if touched > total:
            _fail(f"{touch_path}.pages_touched", "unique first-touch pages exceed the observed allocation")
        if touched < total:
            conform = False
    # The field is observed policy, not the requested case label. A coherent
    # mismatch may be Reject; internally contradictory observed metadata may not.
    if pages is not None and all(item["node"] is not None for item in pages):
        if observed_policy in ("local", "contention") and worker_node is not None and page_nodes != {worker_node}:
            _fail(f"{path}.observed.memory_policy", "local observed policy contradicts page residency")
        if observed_policy == "remote" and worker_node is not None and worker_node in page_nodes:
            _fail(f"{path}.observed.memory_policy", "remote observed policy contradicts page residency")
        if observed_policy == "interleave" and len(page_nodes) < 2:
            _fail(f"{path}.observed.memory_policy", "interleaved observed policy needs multiple resident nodes")
    if policy in ("local", "contention"):
        conform &= page_nodes == {worker_node} and touch_nodes is not None and set(touch_nodes) == {worker_node}
    elif policy == "remote":
        conform &= bool(page_nodes) and worker_node not in page_nodes and touch_nodes is not None and set(touch_nodes) == page_nodes
    elif policy == "interleave":
        conform &= len(page_nodes) >= 2 and touch_nodes is not None and set(touch_nodes) == page_nodes
    contention_path, contention = f"{path}.observed.contention", observed["contention"]
    _object(contention, contention_path, ("active", "cpu_ids", "memory_nodes", "description", "evidence_sha256"))
    active = check.boolean(contention["active"], f"{contention_path}.active")
    if active is True:
        competing_cpus = check.integers(contention["cpu_ids"], f"{contention_path}.cpu_ids")
        competing_nodes = check.integers(contention["memory_nodes"], f"{contention_path}.memory_nodes")
        check.text(contention["description"], f"{contention_path}.description")
        check.digest(contention["evidence_sha256"], f"{contention_path}.evidence_sha256")
        if cpus and competing_cpus is not None and not set(competing_cpus) <= cpus.keys():
            _fail(contention_path, "contention CPU absent from topology")
        if host_nodes is not None and competing_nodes is not None and not set(competing_nodes) <= set(host_nodes):
            _fail(contention_path, "contention memory node absent from topology")
        if policy == "contention":
            conform &= competing_nodes is not None and worker_node in competing_nodes
            conform &= competing_cpus is not None and bool(cpus) and all(cpus[cpu]["node"] == worker_node for cpu in competing_cpus if cpu in cpus)
    elif active is False:
        if contention["cpu_ids"] != [] or contention["memory_nodes"] != [] or contention["description"] is not None or contention["evidence_sha256"] is not None:
            _fail(contention_path, "inactive contention must have empty IDs and null details")
    conform &= active is not None and active == (policy == "contention")
    conform &= requested["memory_policy"] == policy
    if workers is not None and requested_cpus is not None:
        conform &= all(worker["effective_cpu_ids"] is not None and set(worker["effective_cpu_ids"]) <= set(requested_cpus) for worker in workers)
    if requested_nodes is not None:
        conform &= set(requested_nodes) == page_nodes
    for name in ("affinity_sha256", "memory_sha256"):
        check.digest(observed[name], f"{path}.observed.{name}")
    return bool(conform)


def _checks(check, checks, path, expected_shapes, source_revision, ggml_revision):
    values = check.sequence(checks, path)
    if values is None:
        return {}, None
    stages, passes = {}, []
    for i, item in enumerate(values):
        item_path = f"{path}[{i}]"
        _object(item, item_path, ("stage", "criterion", "atol", "rtol", "shape", "actual", "reference", "actual_sha256", "reference_sha256", "reference_identity", "failure_count", "passed"))
        stage = check.enum(item["stage"], f"{item_path}.stage", expected_shapes)
        if stage is not None:
            if stage in stages:
                _fail(item_path, "duplicate numerical-check stage")
            stages[stage] = item
        check.enum(item["criterion"], f"{item_path}.criterion", ("elementwise-abs-plus-rel",))
        atol = check.numeric(item["atol"], f"{item_path}.atol")
        rtol = check.numeric(item["rtol"], f"{item_path}.rtol")
        shape = check.integers(item["shape"], f"{item_path}.shape", 1, distinct=False)
        # Shapes can have equal dimensions: unlike matrix axes they are not sets.
        if stage is not None:
            _equal_known(shape, expected_shapes[stage], f"{item_path}.shape")
        actual = check.sequence(item["actual"], f"{item_path}.actual")
        reference = check.sequence(item["reference"], f"{item_path}.reference")
        for name, array in (("actual", actual), ("reference", reference)):
            if array is not None:
                for j, value in enumerate(array):
                    if not _number(value):
                        _fail(f"{item_path}.{name}[{j}]", "expected finite number, not bool/null")
                if shape is not None and len(array) != math.prod(shape):
                    _fail(f"{item_path}.{name}", "vector length differs from shape")
                check.digest(item[f"{name}_sha256"], f"{item_path}.{name}_sha256")
                _equal_known(item[f"{name}_sha256"], vector_sha256(array), f"{item_path}.{name}_sha256")
            else:
                check.digest(item[f"{name}_sha256"], f"{item_path}.{name}_sha256")
        identity_path, identity = f"{item_path}.reference_identity", item["reference_identity"]
        _object(identity, identity_path, ("kind", "revision", "artifact_sha256"))
        reference_kind = check.enum(identity["kind"], f"{identity_path}.kind", ("strata-scalar-reference", "ggml-pinned-reference"))
        check.digest(identity["revision"], f"{identity_path}.revision", True)
        if reference_kind == "strata-scalar-reference":
            _equal_known(identity["revision"], source_revision, f"{identity_path}.revision")
        elif reference_kind == "ggml-pinned-reference":
            if ggml_revision is None:
                check.missing.add("build.ggml_revision")
            _equal_known(identity["revision"], ggml_revision, f"{identity_path}.revision")
        check.digest(identity["artifact_sha256"], f"{identity_path}.artifact_sha256")
        declared_failures = check.integer(item["failure_count"], f"{item_path}.failure_count")
        declared_pass = check.boolean(item["passed"], f"{item_path}.passed")
        if actual is not None and reference is not None:
            if len(actual) != len(reference):
                _fail(item_path, "actual/reference lengths differ")
            if atol is not None and rtol is not None:
                failures = 0
                for a, r in zip(actual, reference):
                    try:
                        difference = abs(a - r)
                        bound = atol + rtol * abs(r)
                    except (OverflowError, ValueError):
                        _fail(item_path, "overflow in numerical acceptance criterion")
                    if not _number(difference) or not _number(bound):
                        _fail(item_path, "overflow in numerical acceptance criterion")
                    failures += difference > bound
                _equal_known(declared_failures, failures, f"{item_path}.failure_count")
                _equal_known(declared_pass, failures == 0, f"{item_path}.passed")
                passes.append(failures == 0)
            else:
                passes.append(False)
        else:
            passes.append(False)
    for stage in expected_shapes:
        if stage not in stages:
            check.missing.add(f"{path}.stage={stage}")
    return stages, bool(passes) and all(passes) and set(stages) == set(expected_shapes)


def _timing(check, timing, path, unit, matrix, cache_state):
    _object(timing, path, ("clock", "scope", "stages", "warmup_wall_ns", "raw_start_ns", "raw_end_ns", "raw_wall_ns", "median_ns", "p95_ns", "variance_ns2", "cache_preparation", "cache_evidence_sha256"))
    check.enum(timing["clock"], f"{path}.clock", ("monotonic-wall",))
    expected_scope = "full-branch-with-quantization-and-synchronization" if unit == "full" else unit
    check.enum(timing["scope"], f"{path}.scope", (expected_scope,))
    check.exact_set(timing["stages"], f"{path}.stages", STAGES if unit == "full" else (unit,))
    if timing["stages"] is not None and unit == "full" and timing["stages"] != list(STAGES):
        _fail(f"{path}.stages", "full branch stages must be in dependency order")
    arrays = {}
    for name in ("warmup_wall_ns", "raw_start_ns", "raw_end_ns", "raw_wall_ns"):
        array = check.sequence(timing[name], f"{path}.{name}")
        arrays[name] = array
        if array is not None:
            for i, value in enumerate(array):
                check.integer(value, f"{path}.{name}[{i}]", 0 if name in ("raw_start_ns", "raw_end_ns") else 1)
                if value is None:
                    _fail(f"{path}.{name}", "raw timing items must not be null")
    raw, starts, ends, warmups = (arrays[name] for name in ("raw_wall_ns", "raw_start_ns", "raw_end_ns", "warmup_wall_ns"))
    if warmups is not None:
        _equal_known(len(warmups), matrix["warmup_count"], f"{path}.warmup_wall_ns")
    if raw is not None:
        _equal_known(len(raw), matrix["repeat_count"], f"{path}.raw_wall_ns")
    if raw is not None and starts is not None and ends is not None:
        if len(raw) != len(starts) or len(raw) != len(ends):
            _fail(path, "raw repeat/start/end lengths differ")
        for i, (duration, start, end) in enumerate(zip(raw, starts, ends)):
            if end <= start or end - start != duration or (i and start < ends[i - 1]):
                _fail(path, "wall timestamps must be ordered, nonoverlapping and match durations")
    for name in ("median_ns", "p95_ns", "variance_ns2"):
        check.numeric(timing[name], f"{path}.{name}")
    if raw is not None:
        sorted_raw = sorted(raw)
        try:
            derived = {
                "median_ns": statistics.median(raw),
                "p95_ns": sorted_raw[(95 * len(raw) + 99) // 100 - 1],
                "variance_ns2": statistics.pvariance(raw),
            }
        except (OverflowError, ValueError, statistics.StatisticsError):
            _fail(path, "overflow in recomputed raw-wall statistics")
        for name, expected in derived.items():
            if not _number(expected):
                _fail(f"{path}.{name}", "overflow in recomputed raw-wall statistic")
            supplied = timing[name]
            if supplied is not None and not math.isclose(supplied, expected, rel_tol=1e-12, abs_tol=1e-9):
                _fail(f"{path}.{name}", "does not match recomputed raw-wall statistic")
    check.text(timing["cache_preparation"], f"{path}.cache_preparation")
    check.digest(timing["cache_evidence_sha256"], f"{path}.cache_evidence_sha256")
    # Descriptions are evidence metadata, not commands. A cold label alone never
    # replaces the recorded preparation/residency artifact. Actual efficacy still
    # needs independent experimental review.
    if cache_state == "cold" and timing["cache_preparation"] == "none":
        _fail(f"{path}.cache_preparation", "cold measurement needs explicit preparation")


def _trace(check, trace, path, checks, input_sha256):
    _object(trace, path, ("stages", "x_input_sha256", "x_quantized_sha256", "gate_input_sha256", "up_input_sha256", "gate_output_sha256", "up_output_sha256", "silu_gate_input_sha256", "silu_up_input_sha256", "hidden_output_sha256", "down_quantization_input_sha256", "h_quantized_sha256", "down_input_sha256", "down_output_sha256", "synchronized"))
    check.exact_set(trace["stages"], f"{path}.stages", STAGES)
    if trace["stages"] is not None and trace["stages"] != list(STAGES):
        _fail(f"{path}.stages", "must record full dependency order")
    for name in trace:
        if name.endswith("sha256"):
            check.digest(trace[name], f"{path}.{name}")
    _equal_known(trace["x_input_sha256"], input_sha256, f"{path}.x_input_sha256")
    for name, producer in (("gate_input_sha256", "x_quantized_sha256"), ("up_input_sha256", "x_quantized_sha256"), ("silu_gate_input_sha256", "gate_output_sha256"), ("silu_up_input_sha256", "up_output_sha256"), ("down_input_sha256", "h_quantized_sha256")):
        _equal_known(trace[name], trace[producer], f"{path}.{name}")
    for stage, name in (("gate", "gate_output_sha256"), ("up", "up_output_sha256"), ("hidden", "hidden_output_sha256"), ("down", "down_output_sha256")):
        if stage in checks:
            _equal_known(trace[name], checks[stage]["actual_sha256"], f"{path}.{name}")
    _equal_known(trace["down_quantization_input_sha256"], trace["hidden_output_sha256"], f"{path}.down_quantization_input_sha256")
    check.boolean(trace["synchronized"], f"{path}.synchronized")
    if trace["synchronized"] is False:
        _fail(f"{path}.synchronized", "complete branch timing must include synchronization")


def _case_key(case):
    return tuple(case[name] for name in ("unit", "target_format", "role", "batch_size", "threads", "cache_state", "numa_policy", "worker_node"))


def _expected_keys(matrix):
    if any(matrix[key] is None for key in ("batch_sizes", "thread_counts", "worker_nodes")):
        return None
    first_node = min(matrix["worker_nodes"])
    expected = set()
    for fmt, role in itertools.product(FORMATS, ROLES):
        expected.add(("block", fmt, role, 1, 1, "warm", "local", first_node))
        for batch, thread in itertools.product(matrix["batch_sizes"], matrix["thread_counts"]):
            expected.add(("dot", fmt, role, batch, thread, "warm", "local", first_node))
    for fmt, batch, thread, cache, policy, node in itertools.product(FORMATS, matrix["batch_sizes"], matrix["thread_counts"], CACHE_STATES, POLICIES, matrix["worker_nodes"]):
        expected.add(("full", fmt, "all", batch, thread, cache, policy, node))
    return expected


def _remember_dispatch_choice(choices, name, value, path):
    """Known choices must agree across this one fresh-process report."""
    if value is None:
        return
    previous = choices.get(name)
    if previous is not None and value != previous[0]:
        _fail(path, f"report-wide {name} conflicts with {previous[0]!r} at {previous[1]}; use separate fresh-process reports")
    choices.setdefault(name, (value, path))


def _validate_case(check, case, index, kind, profiles, matrix, cpus, nodes, avx2_usable, build, dispatch_choices):
    path = f"cases[{index}]"
    _object(case, path, ("id", "evidence_kind", "unit", "target_format", "role", "batch_size", "threads", "cache_state", "numa_policy", "worker_node", "profile_id", "seed", "input_sha256", "packed_sha256", "control", "dispatch", "placement", "attempt", "outcome", "reason", "checks", "trace", "timing"))
    check.text(case["id"], f"{path}.id")
    case_kind = check.enum(case["evidence_kind"], f"{path}.evidence_kind", KINDS)
    _equal_known(case_kind, kind, f"{path}.evidence_kind")
    unit = check.enum(case["unit"], f"{path}.unit", ("block", "dot", "full"))
    fmt = check.enum(case["target_format"], f"{path}.target_format", FORMATS)
    role = check.enum(case["role"], f"{path}.role", (*ROLES, "all"))
    if unit is not None and role is not None and ((unit == "full") != (role == "all")):
        _fail(f"{path}.role", "full=all; block/dot=one projection role")
    batch = check.integer(case["batch_size"], f"{path}.batch_size", 1)
    threads = check.integer(case["threads"], f"{path}.threads", 1)
    cache = check.enum(case["cache_state"], f"{path}.cache_state", CACHE_STATES)
    policy = check.enum(case["numa_policy"], f"{path}.numa_policy", POLICIES)
    worker_node = check.integer(case["worker_node"], f"{path}.worker_node")
    check.integer(case["seed"], f"{path}.seed")
    for name in ("input_sha256", "packed_sha256"):
        check.digest(case[name], f"{path}.{name}")
    expected_shapes = {}
    profile = None
    if unit == "full":
        profile_id = check.text(case["profile_id"], f"{path}.profile_id")
        if profile_id is not None:
            if profile_id not in profiles:
                _fail(f"{path}.profile_id", "unknown selected mixed-role profile")
            profile = profiles[profile_id]
            _equal_known(profile["target_format"], fmt, f"{path}.target_format")
            _equal_known(case["packed_sha256"], profile["blob_sha256"], f"{path}.packed_sha256")
            expected_shapes = {stage: [batch, profile["n_ff"]] for stage in ("gate", "up", "hidden")}
            expected_shapes["down"] = [batch, profile["n_embd"]]
            if any(item is None for shape in expected_shapes.values() for item in shape):
                expected_shapes = {stage: None for stage in expected_shapes}
        else:
            expected_shapes = {stage: None for stage in ("gate", "up", "hidden", "down")}
        if case["control"] is not None:
            _fail(f"{path}.control", "full model branch must not use generated block/dot geometry")
    elif unit in ("block", "dot"):
        if case["profile_id"] is not None:
            _fail(f"{path}.profile_id", "controlled block/dot cases are not model profiles")
        control = case["control"]
        _object(control, f"{path}.control", ("data_origin", "scenarios", "rows", "columns", "row_bytes", "packed_bytes"))
        check.enum(control["data_origin"], f"{path}.control.data_origin", ("generated-controls",))
        check.exact_set(control["scenarios"], f"{path}.control.scenarios", SCENARIOS)
        for name in ("rows", "columns", "row_bytes", "packed_bytes"):
            check.integer(control[name], f"{path}.control.{name}", 1)
        if fmt is not None:
            _, block_values, block_bytes, _ = TYPE_INFO[fmt]
            columns, rows = control["columns"], control["rows"]
            if columns is not None:
                if columns % block_values:
                    _fail(f"{path}.control.columns", "not whole packed blocks")
                _equal_known(control["row_bytes"], columns // block_values * block_bytes, f"{path}.control.row_bytes")
            if unit == "block":
                _equal_known(columns, block_values, f"{path}.control.columns")
                _equal_known(rows, len(SCENARIOS), f"{path}.control.rows")
                expected_shapes = {"block": [rows, columns] if rows is not None and columns is not None else None}
            else:
                expected_shapes = {"dot": [batch, rows] if batch is not None and rows is not None else None}
            if rows is not None and control["row_bytes"] is not None:
                _equal_known(control["packed_bytes"], rows * control["row_bytes"], f"{path}.control.packed_bytes")
    dispatch_path, dispatch = f"{path}.dispatch", case["dispatch"]
    _object(dispatch, dispatch_path, ("arithmetic", "isa", "runtime_usable", "gate_up_activation", "down_activation", "implementation", "environment"))
    arithmetic = check.enum(dispatch["arithmetic"], f"{dispatch_path}.arithmetic", ("default", "window-invariant"))
    _remember_dispatch_choice(dispatch_choices, "arithmetic", arithmetic, f"{dispatch_path}.arithmetic")
    check.enum(dispatch["isa"], f"{dispatch_path}.isa", ("avx2",))
    usable = check.boolean(dispatch["runtime_usable"], f"{dispatch_path}.runtime_usable")
    _equal_known(usable, avx2_usable, f"{dispatch_path}.runtime_usable")
    check.text(dispatch["implementation"], f"{dispatch_path}.implementation")
    environment = check.need(dispatch["environment"], f"{dispatch_path}.environment", lambda x: isinstance(x, dict), "expected environment object")
    if environment is not None:
        for name, value in environment.items():
            if not isinstance(name, str) or not name or not isinstance(value, str):
                _fail(f"{dispatch_path}.environment", "inert environment entries must be strings")
        if build["environment"] is not None and environment != build["environment"]:
            _fail(f"{dispatch_path}.environment", "fresh process environment must match this selected build/run configuration")
    for name in ("gate_up_activation", "down_activation"):
        check.enum(dispatch[name], f"{dispatch_path}.{name}", ("Q8_K", "Q8_0", "ActQ-legacy", "ActQ-oracle-Q8_0"))
    def activation_matches(format_name, activation, activation_path):
        if format_name is None or activation is None:
            return
        choices = ("ActQ-legacy", "ActQ-oracle-Q8_0") if format_name == "Q2_0" else (TYPE_INFO[format_name][3],)
        if activation not in choices:
            _fail(activation_path, "does not match the actual native/pool activation contract")
        _remember_dispatch_choice(dispatch_choices, f"{format_name} activation", activation, activation_path)
    if profile is not None:
        activation_matches(profile["roles"]["gate"]["format"], dispatch["gate_up_activation"], f"{dispatch_path}.gate_up_activation")
        activation_matches(profile["roles"]["down"]["format"], dispatch["down_activation"], f"{dispatch_path}.down_activation")
    elif unit in ("block", "dot") and fmt is not None:
        activation_matches(fmt, dispatch["gate_up_activation"], f"{dispatch_path}.gate_up_activation")
        activation_matches(fmt, dispatch["down_activation"], f"{dispatch_path}.down_activation")
    placement_pass = _placement(check, case["placement"], f"{path}.placement", threads, worker_node, policy, cpus, nodes)
    attempt_path, attempt = f"{path}.attempt", case["attempt"]
    _object(attempt, attempt_path, ("started", "exit_code", "elapsed_wall_ns", "observation_sha256"))
    started = check.boolean(attempt["started"], f"{attempt_path}.started")
    exit_code = check.integer(attempt["exit_code"], f"{attempt_path}.exit_code")
    check.integer(attempt["elapsed_wall_ns"], f"{attempt_path}.elapsed_wall_ns", 1)
    check.digest(attempt["observation_sha256"], f"{attempt_path}.observation_sha256")
    if started is False:
        _fail(f"{attempt_path}.started", "planned/unattempted matrix entries are not experimental evidence")
    outcome = check.enum(case["outcome"], f"{path}.outcome", ("pass", "reject", "error"))
    if outcome in ("reject", "error"):
        check.text(case["reason"], f"{path}.reason")
    elif case["reason"] is not None:
        _fail(f"{path}.reason", "reason is reserved for observed reject/error")
    checks_pass = None
    if outcome == "error":
        if exit_code == 0:
            _fail(attempt_path, "error requires a nonzero observed exit code")
        if case["checks"] != [] or case["trace"] is not None or case["timing"] is not None:
            _fail(path, "unexecuted error must have no claimed numerical/full-branch timing pass")
        # No waiver of source/build/model/affinity/feature/attempt observations.
        # Only numerical checks and full execution timing are inapplicable.
    else:
        if exit_code is not None and exit_code != 0:
            _fail(attempt_path, "completed pass/reject requires exit code zero")
        stages, checks_pass = _checks(check, case["checks"], f"{path}.checks", expected_shapes, STRATA_REVISION, build["ggml_revision"])
        if case["timing"] is None:
            check.missing.add(f"{path}.timing")
        elif unit is not None:
            _timing(check, case["timing"], f"{path}.timing", unit, matrix, cache)
        if unit == "full":
            if case["trace"] is None:
                check.missing.add(f"{path}.trace")
            else:
                _trace(check, case["trace"], f"{path}.trace", stages, case["input_sha256"])
        elif case["trace"] is not None:
            _fail(f"{path}.trace", "block/dot cannot claim a complete branch trace")
        if outcome == "pass" and (usable is False or not placement_pass or checks_pass is False):
            _fail(path, "pass contradicts observed ISA/placement/per-element numerical checks")
        if outcome == "reject" and checks_pass is True and placement_pass and usable is True:
            _fail(path, "reject requires an observed numerical/placement/ISA failure; not an invented claim")
    return outcome, checks_pass is True and placement_pass and usable is True


def validate(evidence, lock, mode="partial"):
    """Return consistency/completeness metadata; no evidence paths are followed."""
    if mode not in ("partial", "strict-complete", "strict-accept"):
        raise EvidenceError("unknown validation mode")
    _finite_tree(evidence)
    _object(evidence, "$", ("schema_version", "protocol", "evidence_kind", "source", "provenance", "build", "host", "routing", "matrix", "profiles", "cases", "notes"))
    if type(evidence["schema_version"]) is not int or evidence["schema_version"] != 1 or evidence["protocol"] != PROTOCOL:
        _fail("$", "unsupported CPU evidence protocol/schema")
    check = Checker()
    kind = check.enum(evidence["evidence_kind"], "evidence_kind", KINDS)
    _validate_source(check, evidence["source"], lock)
    provenance = evidence["provenance"]
    _object(provenance, "provenance", ("experiment_id", "observed_at", "operator_label", "producer", "raw_artifact_sha256"))
    for name in ("experiment_id", "observed_at", "operator_label", "producer"):
        check.text(provenance[name], f"provenance.{name}")
    check.digest(provenance["raw_artifact_sha256"], "provenance.raw_artifact_sha256")
    _validate_build(check, evidence["build"])
    cpus, host_nodes, usable = _validate_host(check, evidence["host"])
    routing = evidence["routing"]
    _object(routing, "routing", ("top_k", "expert_skip", "mtp"))
    check.integer(routing["top_k"], "routing.top_k", 1)
    check.integer(routing["expert_skip"], "routing.expert_skip")
    check.enum(routing["mtp"], "routing.mtp", ("off",))
    _equal_known(routing["top_k"], 10, "routing.top_k")
    _equal_known(routing["expert_skip"], 0, "routing.expert_skip")
    _validate_matrix(check, evidence["matrix"], host_nodes)
    profiles, _ = _validate_profiles(check, evidence["profiles"], kind)
    cases = check.sequence(evidence["cases"], "cases", False)
    ids, keys, outcomes, accepted_cases = set(), set(), {}, []
    dispatch_choices = {}
    if cases is not None:
        for i, case in enumerate(cases):
            outcome, acceptable = _validate_case(check, case, i, kind, profiles, evidence["matrix"], cpus, host_nodes, usable, evidence["build"], dispatch_choices)
            identifier = case["id"]
            if identifier is not None:
                if identifier in ids:
                    _fail(f"cases[{i}].id", "duplicate case id")
                ids.add(identifier)
            key = _case_key(case)
            if all(item is not None for item in key):
                if key in keys:
                    _fail(f"cases[{i}]", "duplicate matrix case")
                keys.add(key)
            if outcome is not None:
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
            accepted_cases.append(outcome == "pass" and acceptable)
    expected = _expected_keys(evidence["matrix"])
    missing_cases = []
    if expected is None:
        check.missing.add("coverage.derived_matrix")
    else:
        extra = keys - expected
        if extra:
            _fail("cases", f"cases outside the mandatory derived matrix: {sorted(extra)!r}")
        missing_cases = sorted(expected - keys)
        if missing_cases:
            check.missing.add("coverage.case_observations")
    check.strings(evidence["notes"], "notes", False)
    complete = not check.missing and expected is not None and keys == expected
    measured_complete = complete and kind == "measured"
    declared_pass = measured_complete and bool(accepted_cases) and all(accepted_cases)
    satisfied = mode == "partial" or (mode == "strict-complete" and measured_complete) or (mode == "strict-accept" and declared_pass)
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "validation_mode": mode,
        "mode_satisfied": satisfied,
        "status": "schema-complete" if complete else "incomplete",
        "evidence_kind": kind,
        "schema_complete": complete,
        "measured_evidence_complete": measured_complete,
        "declared_cpu_correctness_pass": declared_pass,
        "structural_consistency_only": True,
        "measurements_independently_verified": False,
        "inference_ready": False,
        "cases_present": len(keys),
        "cases_required": len(expected) if expected is not None else None,
        "outcomes": outcomes,
        "missing_fields": sorted(check.missing),
        "missing_cases": [list(key) for key in missing_cases],
        "note": "Checks declared JSON consistency only; no experiment authorization, authenticity, model quality or decode-throughput claim.",
    }


def load_json(path):
    """Reject duplicate JSON keys and nonstandard NaN/Infinity constants."""
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result
    def invalid_constant(value):
        raise EvidenceError(f"nonfinite JSON constant: {value}")
    def finite_integer(value):
        try:
            number = int(value)
        except (ValueError, OverflowError):
            raise EvidenceError("JSON integer exceeds supported numeric range") from None
        if not _number(number):
            raise EvidenceError("JSON integer exceeds supported finite numeric range")
        return number
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_pairs, parse_constant=invalid_constant, parse_int=finite_integer)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", default=str(ROOT / "benchmarks/templates/cpu-cold-expert.json"), help="inert JSON input; default is the partial template; paths/commands/URLs inside it are never read or executed")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--strict-complete", action="store_true", help="require complete declared measured coverage; observed Reject/error is allowed, synthetic/partial is not")
    group.add_argument("--strict-accept", action="store_true", help="also require every elementwise abs+rel and effective placement/ISA check to pass; never an inference certificate")
    args = parser.parse_args(argv)
    mode = "strict-accept" if args.strict_accept else "strict-complete" if args.strict_complete else "partial"
    try:
        result = validate(load_json(args.report), load_json(ROOT / "dependencies.lock.json"), mode)
    except (EvidenceError, OSError, UnicodeError, json.JSONDecodeError, RecursionError, OverflowError) as error:
        parser.exit(1, f"CPU evidence validation failed: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["mode_satisfied"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
