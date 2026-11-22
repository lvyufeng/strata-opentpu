# Copyright (c) 2026 lvyufeng and contributors
# SPDX-License-Identifier: Apache-2.0
"""Check inert CPU cold-expert evidence JSON for protocol v2 offline.

Protocol v2 is the real audited routed-expert format set. The frozen protocol-v1
matrix (IQ2_XS / Q2_0 / IQ3_S targets) is replaced by IQ2_S / IQ2_XXS / IQ1_M
gate/up targets with the actual Q2_0 down tensors. Every other rule -- schema
version, field sets, role geometry and blob arithmetic, placement/NUMA, gating,
timing, outcomes and exit codes -- is intentionally identical to v1.

To keep "identical" auditable, this module does not fork the v1 implementation:
it loads the byte-frozen v1 checker ``scripts/check_cpu_evidence.py``
(sha256 ``d727806ba677563cd4bfa267bebec9d04c67c4b1b048da04f1405678c318826a``,
936 lines) through ``importlib`` as a **private** module instance and rebinds
only ``PROTOCOL``, ``FORMATS`` and ``TYPE_INFO`` on that private instance. The
v1 CLI, the on-disk v1 script and any other v1 import are unaffected. If the v1
checker or its pinned type table no longer matches the reviewed pin, importing
this module fails loudly instead of silently validating v2 evidence with drifted
v1 semantics.

New in v2 (all sourced, none fabricated):

- ``FORMATS = ("IQ2_S", "IQ2_XXS", "IQ1_M")``: the actual routed expert gate/up
  tensor types of the downloaded IQ2_XS-quantised model header.
- ``IQ2_S`` = type 22, 256 values / 82 bytes, Q8_K native activation.
- ``IQ2_XXS`` = type 16, 256 values / 66 bytes, Q8_K native activation.
- ``IQ1_M`` = type 29, 256 values / 56 bytes, Q8_K native activation.
- v1's four audited type entries (IQ2_XS, Q2_0, IQ3_S, Q8_0) stay in
  ``TYPE_INFO`` so a real Q2_0 down role and reviewed auxiliary/Q8_0 metadata
  remain expressible; they are no longer target bodies.

v1 evidence is rejected by v2: the protocol string differs and the v1 format
matrix is not the v2 matrix. Neither strict mode authenticates measurements,
authorizes experiments or proves inference readiness. Only the explicit input
JSON and the fixed integration source lock are read; nothing here builds,
fetches, imports dependency code or executes the data.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1_CHECKER_PATH = ROOT / "scripts" / "check_cpu_evidence.py"
# Reviewed revision of the frozen v1 core. protocol-v1.md and this pin must be
# updated together, through review; editing v1 in place is not permitted.
V1_CHECKER_SHA256 = "d727806ba677563cd4bfa267bebec9d04c67c4b1b048da04f1405678c318826a"
V1_PROTOCOL = "strata-cpu-cold-expert-v1"
# The exact v1 type table this v2 module reuses. Reviewed, not inferred.
V1_FORMATS = ("IQ2_XS", "Q2_0", "IQ3_S")
V1_TYPE_INFO = {
    "IQ2_XS": (17, 256, 74, "Q8_K"),
    "Q2_0": (42, 64, 18, "Q8_0"),
    "IQ3_S": (21, 256, 110, "Q8_K"),
    "Q8_0": (8, 32, 34, "Q8_0"),
}

PROTOCOL = "strata-cpu-cold-expert-v2"
# Real routed expert gate/up target formats from the audited GGUF header
# inventory. Q2_0 expert down tensors are recorded per profile, not as targets.
FORMATS = ("IQ2_S", "IQ2_XXS", "IQ1_M")
V2_TYPE_INFO = {
    "IQ2_S": (22, 256, 82, "Q8_K"),
    "IQ2_XXS": (16, 256, 66, "Q8_K"),
    "IQ1_M": (29, 256, 56, "Q8_K"),
}
DEFAULT_REPORT = ROOT / "benchmarks" / "templates" / "cpu-cold-expert-v2.json"
LAYOUT = "native-row-packed-gate-up-down-v1"


def _load_private_v1():
    """Load the frozen v1 checker as a module instance only this module owns."""
    try:
        payload = V1_CHECKER_PATH.read_bytes()
    except OSError as error:
        raise RuntimeError(f"protocol v2 requires the frozen v1 checker: {error}") from None
    digest = hashlib.sha256(payload).hexdigest()
    if digest != V1_CHECKER_SHA256:
        raise RuntimeError(
            "protocol v1 checker changed: expected sha256 "
            f"{V1_CHECKER_SHA256}, observed {digest}; review CPU evidence protocol v2 "
            "against the new v1 before validating v2 evidence"
        )
    spec = importlib.util.spec_from_file_location(
        "_strata_cpu_evidence_v1_core_for_v2", V1_CHECKER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen protocol v1 checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (
        module.PROTOCOL != V1_PROTOCOL
        or module.FORMATS != V1_FORMATS
        or module.TYPE_INFO != V1_TYPE_INFO
        or module.LAYOUT != LAYOUT
    ):
        raise RuntimeError(
            "protocol v1 core drifted from its reviewed semantics; review "
            "CPU evidence protocol v2 instead of patching v1"
        )
    return module


_impl = _load_private_v1()
# Rebind only this private copy. Nothing outside this module sees these values.
_impl.PROTOCOL = PROTOCOL
_impl.FORMATS = FORMATS
_impl.TYPE_INFO = {**V1_TYPE_INFO, **V2_TYPE_INFO}

# Re-export the reused v1 surface with v2 semantics in effect for this copy.
ROLES = _impl.ROLES
POLICIES = _impl.POLICIES
CACHE_STATES = _impl.CACHE_STATES
STAGES = _impl.STAGES
SCENARIOS = _impl.SCENARIOS
FEATURES = _impl.FEATURES
KINDS = _impl.KINDS
TYPE_INFO = _impl.TYPE_INFO
EvidenceError = _impl.EvidenceError
Checker = _impl.Checker
_expected_keys = _impl._expected_keys
_validate_matrix = _impl._validate_matrix
_validate_profiles = _impl._validate_profiles
_validate_case = _impl._validate_case
_finite_tree = _impl._finite_tree
_number = _impl._number
vector_sha256 = _impl.vector_sha256
load_json = _impl.load_json


def validate(evidence, lock, mode="partial"):
    """Validate v2 evidence with v1's reviewed mechanics; explicit v1 signature."""
    return _impl.validate(evidence, lock, mode)


def main(argv=None):
    """Offline v2 CLI: default is the v2 partial template; exit 0/1/2 as in v1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        nargs="?",
        default=str(DEFAULT_REPORT),
        help="inert JSON input; default is the protocol v2 partial template; paths/commands/URLs inside it are never read or executed",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--strict-complete",
        action="store_true",
        help="require complete declared measured coverage; observed Reject/error is allowed, synthetic/partial is not",
    )
    group.add_argument(
        "--strict-accept",
        action="store_true",
        help="also require every elementwise abs+rel and effective placement/ISA check to pass; never an inference certificate",
    )
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
