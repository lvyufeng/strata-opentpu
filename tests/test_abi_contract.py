# Copyright (c) 2026 lvyufeng and contributors
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests of a hardware-free specification, NOT inference tests."""
from dataclasses import replace
import unittest

from runtime.abi_contract import (
    Binding, Capabilities, ConformanceModel, ContractError, Fence, Format,
    Identity, STATE_COMPONENTS, Version,
)

F32 = Format("f32", "contiguous-row-major/v1", 1, 4, "none", "ieee754-f32-ftz")
FP4 = Format("opentpu-fp4", "synthetic-test/v1", 128, 68, "test-only", "test-only")
IDENTITY = Identity("test-target-not-a-board", "test-config", "a" * 64)


class AbiContractTests(unittest.TestCase):
    def setUp(self):
        self.caps = Capabilities(IDENTITY, frozenset({F32}), state_components=STATE_COMPONENTS)
        self.model = ConformanceModel(self.caps, IDENTITY)
        self.model.register_session("s", "base", 10)

    def binding(self, domain="device", access="readwrite", writable=True):
        return Binding(self.model.allocate(512, domain, writable), F32, (1, 128), access)

    def candidate(self, rows=3):
        tx = self.model.begin("s", rows)
        b = self.binding()
        return tx, b, self.model.submit("compute", [b], tx)

    def assert_rejected(self, fn, *args, **kwargs):
        with self.assertRaises(ContractError):
            fn(*args, **kwargs)

    def test_identity_and_abi_fail_closed(self):
        for bad in (replace(IDENTITY, target="other"), replace(IDENTITY, config_hash="other"),
                    replace(IDENTITY, bitstream_sha256="b" * 64),
                    replace(IDENTITY, abi=Version(2, 0)), replace(IDENTITY, abi=Version(1, -1))):
            with self.subTest(identity=bad):
                self.assert_rejected(ConformanceModel, replace(self.caps, identity=bad), IDENTITY)
        self.assert_rejected(ConformanceModel, self.caps, replace(IDENTITY, abi=Version(1, 1)))
        for version in (Version(1, 1), Version(2, 0)):
            future = replace(IDENTITY, abi=version)
            self.assert_rejected(ConformanceModel, replace(self.caps, identity=future), future)

    def test_new_minor_can_negotiate_old_semantics(self):
        newer = replace(self.caps, identity=replace(IDENTITY, abi=Version(1, 2)))
        self.assertEqual(ConformanceModel(newer, IDENTITY).negotiated_abi, Version(1, 0))

    def test_resource_limit_validation(self):
        for kwargs in ({"alignment": 3}, {"max_buffer_bytes": 0}, {"max_inflight": False}):
            self.assert_rejected(ConformanceModel, replace(self.caps, **kwargs), IDENTITY)
        for size in (0, -1, True, (1 << 32) + 1):
            self.assert_rejected(self.model.allocate, size)

    def test_iq2_xs_never_aliases_fp4(self):
        model = ConformanceModel(replace(self.caps, formats=frozenset({FP4})), IDENTITY)
        iq2 = replace(FP4, name="gguf-iq2_xs")
        b = Binding(model.allocate(512), iq2, (1, 128), "read")
        self.assert_rejected(model.submit, "compute", [b])

    def test_exact_layout_scale_and_numerics_required(self):
        b = self.binding()
        for kwargs in ({"layout": "transposed"}, {"scale_encoding": "fp16"},
                       {"numerics": "different"}, {"byte_order": "big"}):
            self.assert_rejected(self.model.submit, "compute", [replace(b, format=replace(F32, **kwargs))])

    def test_shape_alignment_and_bounds(self):
        b = self.binding()
        for bad in (replace(b, shape=(0, 128)), replace(b, shape=(1, True)),
                    replace(b, shape=(1, 129)), replace(b, offset=1), replace(b, offset=128)):
            self.assert_rejected(self.model.submit, "compute", [bad])
        self.assert_rejected(FP4.size, (1, 127))

    def test_access_and_alias_rejected(self):
        b = self.binding(writable=False)
        self.assert_rejected(self.model.submit, "compute", [b])
        self.assert_rejected(self.model.submit, "compute", [replace(b, access="execute")])
        self.assert_rejected(self.model.submit, "compute", [replace(b, access="read")] * 2)
        self.assert_rejected(self.model.submit, "unsupported", [b])
        self.assert_rejected(self.model.submit, "compute", [])

    def test_h2d_and_d2h_pin_both_ends_until_visible_completion(self):
        for op, domains in (("h2d", ("host", "device")), ("d2h", ("device", "host"))):
            with self.subTest(operation=op):
                bindings = [self.binding(domains[0], "read"), self.binding(domains[1], "write")]
                ticket = self.model.submit(op, bindings)
                for b in bindings:
                    self.assert_rejected(self.model.release, b.buffer)
                host = next(b.buffer for b in bindings if b.buffer.domain == "host")
                self.assert_rejected(self.model.host_access, host)
                self.assert_rejected(self.model.complete, ticket, visible=False)
                self.model.complete(ticket)
                self.model.host_access(host)
                for b in bindings:
                    self.model.release(b.buffer)

    def test_transfer_is_not_conversion_or_implicit_memory_alias(self):
        h, d = self.binding("host", "read"), self.binding("device", "write")
        self.assert_rejected(self.model.submit, "h2d", [d, h])
        self.assert_rejected(self.model.submit, "h2d", [h, replace(d, shape=(1, 64))])
        self.assert_rejected(self.model.submit, "compute", [h])

    def test_inflight_reuse_release_and_queue_limit(self):
        b = self.binding()
        t = self.model.submit("compute", [b])
        self.assert_rejected(self.model.submit, "compute", [b])
        self.assert_rejected(self.model.release, b.buffer)
        self.model.complete(t)
        self.model.release(b.buffer)
        self.assert_rejected(self.model.release, b.buffer)
        model = ConformanceModel(replace(self.caps, max_inflight=1), IDENTITY)
        b = Binding(model.allocate(512), F32, (128,), "read")
        model.submit("compute", [b])
        self.assert_rejected(model.submit, "compute", [Binding(model.allocate(512), F32, (128,), "read")])

    def test_foreign_handle_and_completion_rejected(self):
        other = ConformanceModel(self.caps, IDENTITY)
        b = self.binding()
        self.assert_rejected(other.submit, "compute", [b])
        t = self.model.submit("compute", [b])
        self.assert_rejected(other.complete, t)
        fence = self.model.complete(t)
        self.assert_rejected(self.model.complete, t)
        self.assert_rejected(self.model.check_fence, replace(fence, outcome="error"))

    def test_dependencies_require_successful_visible_fence(self):
        b = self.binding()
        t = self.model.submit("compute", [b])
        self.assert_rejected(self.model.check_fence, Fence(t, "ok"))
        bad = self.model.complete(t, "error")
        self.assert_rejected(self.model.submit, "compute", [b], depends_on=[bad])
        t2 = self.model.submit("compute", [b])
        good = self.model.complete(t2)
        self.model.submit("compute", [b], depends_on=[good])

    def test_completion_does_not_commit_and_prefix_commit_is_atomic(self):
        tx, b, t = self.candidate()
        self.assert_rejected(self.model.commit, tx, 2, "accepted")
        self.model.complete(t)
        self.assertEqual(self.model.snapshot("s").frontier, 10)
        self.assert_rejected(self.model.cpu_fallback_checkpoint, "s")
        for count in (-1, 0, 4, True):
            self.assert_rejected(self.model.commit, tx, count, "accepted")
        result = self.model.commit(tx, 2, "accepted")
        self.assertEqual((result.version, result.frontier, result.checkpoint), (1, 12, "accepted"))
        self.assert_rejected(self.model.commit, tx, 2, "again")
        self.assert_rejected(self.model.submit, "compute", [b], tx)

    def test_transaction_requires_inventory_and_candidate_computation(self):
        for inventory in (frozenset({"kv"}), STATE_COMPONENTS - {"sampler_rng_history"}):
            model = ConformanceModel(replace(self.caps, state_components=inventory), IDENTITY)
            model.register_session("s", "base")
            self.assert_rejected(model.begin, "s", 1)
        tx = self.model.begin("s", 1)
        self.assert_rejected(self.model.begin, "s", 1)
        self.assert_rejected(self.model.commit, tx, 1, "invented")
        self.model.rollback(tx)

    def test_all_transaction_transfers_must_drain_before_commit(self):
        tx, _, t = self.candidate()
        self.model.complete(t)
        transfer = self.model.submit("d2h", [self.binding("device", "read"),
                                               self.binding("host", "write")], tx)
        self.assert_rejected(self.model.commit, tx, 1, "checkpoint")
        self.model.complete(transfer)
        self.model.commit(tx, 1, "checkpoint")

    def test_timeout_and_cancel_are_not_completion(self):
        for action in ("timeout", "request_cancel"):
            with self.subTest(action=action):
                tx, b, t = self.candidate()
                getattr(self.model, action)(t)
                self.assert_rejected(self.model.release, b.buffer)
                self.assert_rejected(self.model.rollback, tx)
                self.assert_rejected(self.model.cpu_fallback_checkpoint, "s")
                self.model.complete(t)  # late success still cannot accept poisoned state
                self.assert_rejected(self.model.commit, tx, 1, "unsafe")
                self.model.rollback(tx)
                self.assertEqual(self.model.cpu_fallback_checkpoint("s").frontier, 10)

    def test_error_and_cancelled_completion_require_rollback(self):
        for outcome in ("error", "cancelled"):
            tx, _, t = self.candidate()
            self.model.complete(t, outcome)
            self.assert_rejected(self.model.commit, tx, 1, "unsafe")
            self.model.rollback(tx)

    def test_epoch_advance_requires_drain_proof_and_restore(self):
        tx, b, t = self.candidate()
        self.model.timeout(t)
        self.assert_rejected(self.model.reset)
        self.model.reset(proven_quiescent=True)
        self.assert_rejected(self.model.complete, t)
        self.assert_rejected(self.model.release, b.buffer)
        self.assert_rejected(self.model.rollback, tx)
        self.assert_rejected(self.model.cpu_fallback_checkpoint, "s")
        snapshot = self.model.restore_checkpoint("s", "restored-base", 10)
        self.assertEqual((snapshot.version, snapshot.frontier), (1, 10))
        self.assertEqual(self.model.cpu_fallback_checkpoint("s"), snapshot)

    def test_global_uncertain_dma_blocks_even_idle_session_fallback(self):
        t = self.model.submit("h2d", [self.binding("host", "read"), self.binding("device", "write")])
        self.model.timeout(t)
        self.assert_rejected(self.model.cpu_fallback_checkpoint, "s")
        self.model.complete(t, "cancelled")
        self.model.cpu_fallback_checkpoint("s")


if __name__ == "__main__":
    unittest.main()
