#!/usr/bin/env python3
"""Reference-only 4 GiB residency arithmetic for the pinned mixed-GGUF target.

This is a descriptive calculator, not a budget contract, a measurement or a
device capability check. It reads a validated mixed-GGUF header inventory
receipt (v1 or v2), groups native per-tensor storage bytes into per-layer
persistent (non-expert) weights, per-layer expert sets and unplaced global
tensors, then charges the published reference 4 GiB device envelope (two 2 GiB
DDR3 banks) against declared linear-attention state, declared
full-attention/indexer KV bytes per token, a double-buffered one-expert DMA
staging area and a double-buffered per-layer dense staging area.

Only byte arithmetic over already-declared header facts is performed. Every
rate, usable capacity, reserve, fragmentation, allocator overhead, activation
workspace, cache state and routing trace is unknown here and listed as such in
the output. The command never reads model payload, never opens a device and
never claims a residency fit, a speed or a tokens/s number.

Usage:
    python3 scripts/residency_reference.py --receipt local/inventory.v2.json
    python3 scripts/residency_reference.py --receipt local/inventory.v2.json \\
        --format json --contexts 4096,32768

Exit codes: 0 on success, 2 on any malformed/ambiguous receipt, unknown
storage or unsupported model geometry (stdout empty, diagnostic on stderr).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

SCHEMA_VERSION = "strata-reference-residency-arithmetic-v1"

# Reference device envelope: Inspur YPCB-00338-1P1 / Kintex-7 XC7K480T with two
# 2 GiB DDR3 banks (docs/architecture.md). Capacity is declared, not measured;
# usable capacity, bank interleaving and link/DMA facts remain unknown.
DEVICE_CAPACITY_BYTES = 4 * 1024**3
DEVICE_BANK_BYTES = 2 * 1024**3

# Pinned model geometry (local/acquisition/qwen38flasnext-config.json): 48
# decoder layers, 512 experts per MoE layer, 10 routed experts per token, 36
# linear-attention layers and 12 full-attention layers at indices 3,7,...,47.
REFERENCE_LAYERS = 48
EXPERTS_PER_LAYER = 512
LINEAR_ATTENTION_LAYERS = 36
FULL_ATTENTION_LAYERS = 12

# Declared state dtype: text_config.mamba_ssm_dtype = "float32". DeltaNet value
# state per linear layer = 48 value heads * 128 * 128 (fp32). Conv state per
# linear layer = (linear_conv_kernel_dim 4 - 1) * conv_dim * 4, where
# conv_dim = 2 * 16 key heads * 128 + 48 value heads * 128 = 10240 (float32).
STATE_DTYPE_BYTES = 4
LINEAR_VALUE_HEADS = 48
LINEAR_VALUE_HEAD_DIM = 128
CONV_STATE_STEPS = 3
CONV_DIM = 2 * 16 * 128 + 48 * 128

# Declared KV dtype: text_config.dtype = "bfloat16" (unverified assumption).
# Full-attention KV per layer per token = 2 KV heads * 256 head_dim * (K+V)
# elements. Indexer cache per full-attention layer per token = 1 KV head * 128
# head_dim. Declared max_position_embeddings = 262144.
KV_DTYPE_BYTES = 2
KV_HEADS = 2
KV_HEAD_DIM = 256
INDEXER_KV_HEADS = 1
INDEXER_HEAD_DIM = 128
DECLARED_MAX_CONTEXT_TOKENS = 262144

UNKNOWNS = (
    "device usable capacity after reserve/ECC and allocator overhead",
    "largest contiguous allocation and bank interleaving on the reference board",
    "device-side DDR3 read bandwidth and access latency",
    "negotiated PCIe link width/speed (capability is not the negotiated link)",
    "DMA transfer directions, sizes, repeat counts and measured rates",
    "activation/workspace, DMA ring, metadata and sampling buffer bytes",
    "fragmentation and per-allocation padding",
    "routing trace / expert hit distribution and cache state across requests",
    "device compute availability and per-layer rate for dense and expert work",
    "MTP layer placement and its extra state/weight cost",
)

STATE_BYTES = (
    LINEAR_ATTENTION_LAYERS * LINEAR_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM
    * LINEAR_VALUE_HEAD_DIM * STATE_DTYPE_BYTES
) + (
    LINEAR_ATTENTION_LAYERS * CONV_STATE_STEPS * CONV_DIM * STATE_DTYPE_BYTES
)
FULL_ATTENTION_KV_PER_TOKEN = (
    FULL_ATTENTION_LAYERS * KV_HEADS * KV_HEAD_DIM * 2 * KV_DTYPE_BYTES
)
INDEXER_KV_PER_TOKEN = (
    FULL_ATTENTION_LAYERS * INDEXER_KV_HEADS * INDEXER_HEAD_DIM * KV_DTYPE_BYTES
)
KV_PER_TOKEN = FULL_ATTENTION_KV_PER_TOKEN + INDEXER_KV_PER_TOKEN


class ResidencyError(ValueError):
    """A receipt or geometry problem that makes byte arithmetic unsafe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ResidencyError(f"{label}: expected a positive integer")
    return value


def _tensor_bytes(tensor: Any, index: int) -> tuple[str, int, list[int], str]:
    if not isinstance(tensor, dict):
        raise ResidencyError(f"tensors[{index}]: expected an object")
    name = tensor.get("name")
    if not isinstance(name, str) or not name:
        raise ResidencyError(f"tensors[{index}].name: expected a nonempty string")
    shape = tensor.get("shape")
    if not isinstance(shape, list) or not shape:
        raise ResidencyError(f"{name}.shape: expected a nonempty list")
    for axis, size in enumerate(shape):
        _positive_int(size, f"{name}.shape[{axis}]")
    storage = tensor.get("storage")
    if not isinstance(storage, dict):
        raise ResidencyError(
            f"{name}.storage: unknown layout cannot be byte-charged; "
            "publish a receipt with known storage before budgeting"
        )
    data_bytes = _positive_int(storage.get("data_bytes"), f"{name}.storage.data_bytes")
    fmt = tensor.get("ggml_type_name")
    fmt = fmt if isinstance(fmt, str) and fmt else f"type-id-{tensor.get('ggml_type_id')}"
    return name, data_bytes, shape, fmt


def _layer_index(name: str) -> int | None:
    if not name.startswith("blk."):
        return None
    head, dot, _ = name[4:].partition(".")
    if not dot:
        raise ResidencyError(f"{name}: malformed layer prefix")
    try:
        return int(head)
    except ValueError as exc:
        raise ResidencyError(f"{name}: non-integer layer index") from exc


def classify(receipt: dict[str, Any]) -> dict[str, Any]:
    """Group receipt bytes into per-layer weights, expert sets and globals."""
    tensors = receipt.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ResidencyError("receipt.tensors: expected a nonempty list")
    fixed: dict[int, int] = {}
    expert: dict[int, int] = {}
    global_bytes: dict[str, int] = {}
    gate_up_formats: dict[int, set[str]] = {}
    down_formats: dict[int, set[str]] = {}
    seen: set[str] = set()
    total = 0
    for index, tensor in enumerate(tensors):
        name, data_bytes, shape, fmt = _tensor_bytes(tensor, index)
        if name in seen:
            raise ResidencyError(f"duplicate tensor name: {name}")
        seen.add(name)
        total += data_bytes
        layer = _layer_index(name)
        if layer is None:
            global_bytes[name] = data_bytes
            continue
        if name.endswith("_exps.weight"):
            if shape[-1] != EXPERTS_PER_LAYER:
                raise ResidencyError(
                    f"{name}: expert axis {shape[-1]} != {EXPERTS_PER_LAYER}"
                )
            expert[layer] = expert.get(layer, 0) + data_bytes
            target = down_formats if "down" in name else gate_up_formats
            target.setdefault(layer, set()).add(fmt)
        else:
            fixed[layer] = fixed.get(layer, 0) + data_bytes
    declared_total = receipt.get("totals", {})
    if isinstance(declared_total, dict) and "known_storage_bytes" in declared_total:
        declared = _positive_int(declared_total["known_storage_bytes"], "totals.known_storage_bytes")
        if declared != total:
            raise ResidencyError(
                f"totals.known_storage_bytes {declared} != summed tensor bytes {total}"
            )
    if sorted(fixed) != list(range(REFERENCE_LAYERS)) or sorted(expert) != list(range(REFERENCE_LAYERS)):
        raise ResidencyError(
            "receipt layer set does not match the pinned 48-layer reference model; "
            "this calculator only describes that declared geometry"
        )
    if any(expert[layer] % EXPERTS_PER_LAYER for layer in expert):
        raise ResidencyError("expert tensor bytes are not divisible by 512")
    per_expert = {layer: expert[layer] // EXPERTS_PER_LAYER for layer in expert}
    format_counts: dict[str, int] = {}
    down_counts: dict[str, int] = {}
    layer_formats: dict[int, str] = {}
    for layer, names in gate_up_formats.items():
        if len(names) != 1:
            raise ResidencyError(
                f"layer {layer}: mixed gate/up expert formats {sorted(names)}"
            )
        name = next(iter(names))
        format_counts[name] = format_counts.get(name, 0) + 1
        layer_formats[layer] = name
    for layer, names in down_formats.items():
        for name in names:
            down_counts[name] = down_counts.get(name, 0) + 1
    return {
        "fixed": fixed,
        "expert": expert,
        "per_expert": per_expert,
        "formats": layer_formats,
        "format_counts": dict(sorted(format_counts.items())),
        "down_format_counts": dict(sorted(down_counts.items())),
        "globals": dict(sorted(global_bytes.items())),
        "total_known_bytes": total,
    }


def analyze(
    receipt: dict[str, Any],
    contexts: tuple[int, ...] = (4096, 32768),
    receipt_path: str = "<in-memory>",
    receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Return the reference-only residency arithmetic as a JSON-serialisable dict."""
    if not isinstance(receipt, dict):
        raise ResidencyError("receipt: expected a JSON object")
    for context in contexts:
        _positive_int(context, "context")
    grouped = classify(receipt)
    fixed = grouped["fixed"]
    expert = grouped["expert"]
    per_expert = grouped["per_expert"]
    globals_all = grouped["globals"]
    globals_fittable = {
        name: size for name, size in globals_all.items() if size <= DEVICE_CAPACITY_BYTES
    }
    globals_oversize = [
        {"name": name, "bytes": size}
        for name, size in globals_all.items()
        if size > DEVICE_CAPACITY_BYTES
    ]
    fixed_sum = sum(fixed.values())
    expert_sum = sum(expert.values())
    fixed_max_layer = max(fixed, key=lambda layer: (fixed[layer], -layer))
    fixed_min_layer = min(fixed, key=lambda layer: (fixed[layer], layer))
    max_per_expert = max(per_expert.values())
    min_per_expert = min(per_expert.values())
    dma_staging = 2 * max_per_expert
    dense_stream = 2 * fixed[fixed_max_layer]
    globals_fittable_sum = sum(globals_fittable.values())

    def scenario(identifier: str, resident: int, charge_layer_pack: bool) -> dict[str, Any]:
        per_context = {}
        for context in contexts:
            kv = context * KV_PER_TOKEN
            hot = DEVICE_CAPACITY_BYTES - resident - kv
            feasible = hot > 0
            slots = hot // max_per_expert if feasible else 0
            slots_best = hot // min_per_expert if feasible else 0
            per_context[str(context)] = {
                "kv_bytes": kv,
                "hot_bytes": hot,
                "feasible": feasible,
                "hot_slots_worst": slots,
                "hot_slots_best": slots_best,
                "hot_slots_per_layer_worst": round(slots / REFERENCE_LAYERS, 2),
                "per_layer_coverage_pct_worst": round(
                    100.0 * slots / REFERENCE_LAYERS / EXPERTS_PER_LAYER, 2
                ),
            }
        remaining = DEVICE_CAPACITY_BYTES - resident
        max_context = remaining // KV_PER_TOKEN if remaining > 0 else 0
        result: dict[str, Any] = {
            "id": identifier,
            "resident_declared_bytes": resident,
            "free_without_kv_bytes": remaining,
            "max_context_tokens_at_zero_hot": max(0, max_context),
            "contexts": per_context,
        }
        if charge_layer_pack:
            budget = DEVICE_CAPACITY_BYTES - resident - contexts[0] * KV_PER_TOKEN
            packed = 0
            layers = 0
            for layer in sorted(expert, key=lambda item: (expert[item], item)):
                if packed + expert[layer] > budget:
                    break
                packed += expert[layer]
                layers += 1
            result["cheapest_first_layer_pack"] = {
                "context": contexts[0],
                "budget_bytes": budget,
                "layers": layers,
                "layer_coverage_pct": round(100.0 * layers / REFERENCE_LAYERS, 2),
                "packed_expert_bytes": packed,
                "layer_formats": {str(layer): grouped["formats"][layer]
                                  for layer in sorted(expert, key=lambda item: (expert[item], item))[:layers]},
            }
        return result

    scenarios = [
        scenario("s0-all-fixed-resident", fixed_sum + STATE_BYTES + dma_staging, False),
        scenario(
            "s0g-all-fixed-and-fittable-globals-resident",
            fixed_sum + globals_fittable_sum + STATE_BYTES + dma_staging,
            False,
        ),
        scenario("s1-all-fixed-streamed", STATE_BYTES + dense_stream + dma_staging, False),
        scenario(
            "s1g-all-fixed-streamed-and-fittable-globals-resident",
            STATE_BYTES + dense_stream + globals_fittable_sum + dma_staging,
            False,
        ),
        scenario("s2-layer-partitioned-hot", STATE_BYTES + dense_stream + dma_staging, True),
    ]
    fixed_bits = {
        "min": {"layer": fixed_min_layer, "bytes": fixed[fixed_min_layer]},
        "max": {"layer": fixed_max_layer, "bytes": fixed[fixed_max_layer]},
    }
    return {
        "kind": "strata-reference-residency-arithmetic",
        "schema_version": SCHEMA_VERSION,
        "input": {
            "path": receipt_path,
            "sha256": receipt_sha256,
            "tensor_count": len(receipt.get("tensors", [])),
            "known_storage_bytes": grouped["total_known_bytes"],
            "input_kind": receipt.get("input_kind"),
            "claims": receipt.get("claims"),
        },
        "assumptions": {
            "device": {
                "capacity_bytes": DEVICE_CAPACITY_BYTES,
                "bank_bytes": DEVICE_BANK_BYTES,
                "banks": 2,
            },
            "state": {
                "dtype_bytes": STATE_DTYPE_BYTES,
                "linear_attention_layers": LINEAR_ATTENTION_LAYERS,
                "value_heads": LINEAR_VALUE_HEADS,
                "value_head_dim": LINEAR_VALUE_HEAD_DIM,
                "conv_dim": CONV_DIM,
                "conv_state_steps": CONV_STATE_STEPS,
            },
            "kv": {
                "dtype_bytes": KV_DTYPE_BYTES,
                "full_attention_layers": FULL_ATTENTION_LAYERS,
                "kv_heads": KV_HEADS,
                "head_dim": KV_HEAD_DIM,
                "indexer_kv_heads": INDEXER_KV_HEADS,
                "indexer_head_dim": INDEXER_HEAD_DIM,
                "declared_max_context_tokens": DECLARED_MAX_CONTEXT_TOKENS,
            },
            "staging": {
                "dma_double_buffer_bytes": dma_staging,
                "dense_layer_stream_double_buffer_bytes": dense_stream,
                "dense_stream_layer": fixed_max_layer,
            },
            "not_charged": list(UNKNOWNS),
        },
        "weights": {
            "layer_count": REFERENCE_LAYERS,
            "experts_per_layer": EXPERTS_PER_LAYER,
            "persistent_fixed_bytes": fixed_sum,
            "fixed_layer_bounds": fixed_bits,
            "expert_bytes": expert_sum,
            "per_expert_footprint_min_bytes": min_per_expert,
            "per_expert_footprint_max_bytes": max_per_expert,
            "per_layer_expert_bytes": {str(layer): expert[layer] for layer in sorted(expert)},
            "per_layer_per_expert_bytes": {str(layer): per_expert[layer] for layer in sorted(per_expert)},
            "expert_gate_up_format_counts": grouped["format_counts"],
            "expert_down_format_counts": grouped["down_format_counts"],
            "globals_bytes": globals_all,
            "globals_fittable_bytes": globals_fittable_sum,
            "globals_oversize": globals_oversize,
            "total_known_bytes": grouped["total_known_bytes"],
        },
        "state": {
            "value_state_bytes": LINEAR_ATTENTION_LAYERS * LINEAR_VALUE_HEADS
            * LINEAR_VALUE_HEAD_DIM * LINEAR_VALUE_HEAD_DIM * STATE_DTYPE_BYTES,
            "conv_state_bytes": LINEAR_ATTENTION_LAYERS * CONV_STATE_STEPS
            * CONV_DIM * STATE_DTYPE_BYTES,
            "total_bytes": STATE_BYTES,
        },
        "kv": {
            "full_attention_bytes_per_token": FULL_ATTENTION_KV_PER_TOKEN,
            "indexer_bytes_per_token": INDEXER_KV_PER_TOKEN,
            "bytes_per_token": KV_PER_TOKEN,
            "declared_max_context_tokens": DECLARED_MAX_CONTEXT_TOKENS,
        },
        "scenarios": scenarios,
        "unknowns": list(UNKNOWNS),
    }


def render_summary(result: dict[str, Any]) -> str:
    """Render a compact human-readable summary; not an evidence artifact."""
    weights = result["weights"]
    lines = [
        f"input: {result['input']['path']} tensors={result['input']['tensor_count']} "
        f"known_storage_bytes={weights['total_known_bytes']}",
        f"device (declared): {result['assumptions']['device']['capacity_bytes']} B "
        f"in {result['assumptions']['device']['banks']} banks",
        "",
        "layer weights (native GGUF header bytes):",
        f"  persistent fixed: {weights['persistent_fixed_bytes']} B "
        f"(min {weights['fixed_layer_bounds']['min']['bytes']} B layer "
        f"{weights['fixed_layer_bounds']['min']['layer']}, max "
        f"{weights['fixed_layer_bounds']['max']['bytes']} B layer "
        f"{weights['fixed_layer_bounds']['max']['layer']})",
        f"  all experts: {weights['expert_bytes']} B; per expert footprint "
        f"{weights['per_expert_footprint_min_bytes']}..{weights['per_expert_footprint_max_bytes']} B",
        f"  expert gate/up formats: {weights['expert_gate_up_format_counts']}; "
        f"down formats: {weights['expert_down_format_counts']}",
        f"  fittable globals: {weights['globals_fittable_bytes']} B; "
        f"oversize globals: {[item['name'] for item in weights['globals_oversize']]}",
        "",
        f"state (declared fp32): {result['state']['total_bytes']} B",
        f"KV+indexer (declared bf16): {result['kv']['bytes_per_token']} B/token",
        "",
    ]
    for scenario in result["scenarios"]:
        lines.append(f"{scenario['id']}: resident {scenario['resident_declared_bytes']} B, "
                     f"max ctx at zero hot {scenario['max_context_tokens_at_zero_hot']}")
        for context, values in scenario["contexts"].items():
            lines.append(
                f"  ctx {context}: hot {values['hot_bytes']} B "
                f"slots {values['hot_slots_worst']}..{values['hot_slots_best']} "
                f"per-layer {values['hot_slots_per_layer_worst']} "
                f"coverage {values['per_layer_coverage_pct_worst']}%"
            )
        pack = scenario.get("cheapest_first_layer_pack")
        if pack:
            lines.append(f"  cheapest-first layer pack at ctx {pack['context']}: "
                         f"{pack['layers']} layers ({pack['layer_coverage_pct']}%) "
                         f"bytes {pack['packed_expert_bytes']}")
    return "\n".join(lines) + "\n"


def load_receipt(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            receipt = json.load(handle)
    except OSError as exc:
        raise ResidencyError(f"receipt unreadable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ResidencyError(f"receipt is not valid JSON: {exc}") from exc
    if not isinstance(receipt, dict):
        raise ResidencyError("receipt: expected a JSON object")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reference-only residency byte arithmetic for the pinned "
                    "mixed-GGUF target. Reads one inventory receipt; no device I/O."
    )
    parser.add_argument("--receipt", required=True, help="path to an inventory receipt JSON")
    parser.add_argument("--format", choices=("summary", "json"), default="summary")
    parser.add_argument(
        "--contexts",
        default="4096,32768",
        help="comma-separated context lengths to charge KV against (default 4096,32768)",
    )
    args = parser.parse_args(argv)
    try:
        contexts = tuple(int(item, 10) for item in args.contexts.split(",") if item)
        if not contexts:
            raise ResidencyError("--contexts: expected at least one value")
        path = Path(args.receipt)
        receipt = load_receipt(path)
        digest = sha256_file(path)
        result = analyze(receipt, contexts, receipt_path=str(path), receipt_sha256=digest)
    except ResidencyError as exc:
        print(f"residency_reference: {exc}", file=sys.stderr)
        return 2
    if args.format == "json":
        json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())