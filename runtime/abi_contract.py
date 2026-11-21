# Copyright (c) 2026 lvyufeng and contributors
# SPDX-License-Identifier: Apache-2.0
"""Hardware-free ABI conformance MODEL, not a device runtime or model engine.

No device, DMA, tensor arithmetic, checkpoint bytes or real clock is accessed.
Completion/reset/checkpoint evidence is injected by tests, not verified here.
See docs/design/device-abi.md for the proposed fork implementation boundary.
"""
from dataclasses import dataclass, field
from math import prod
from uuid import uuid4


class ContractError(ValueError):
    """A proposed ABI precondition was violated; no device action occurred."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def natural(value, positive=False):
    return type(value) is int and value >= (1 if positive else 0)


@dataclass(frozen=True)
class Version:
    major: int = 1
    minor: int = 0


@dataclass(frozen=True)
class Identity:
    target: str
    config_hash: str
    bitstream_sha256: str
    abi: Version = Version()


@dataclass(frozen=True)
class Format:
    """Exact packed layout signature, not a GGUF pack name or conversion rule."""
    name: str
    layout: str
    block_elements: int
    block_bytes: int
    scale_encoding: str
    numerics: str
    byte_order: str = "little"

    def size(self, shape):
        require(bool(shape) and all(natural(d, True) for d in shape), "invalid shape")
        require(natural(self.block_elements, True) and natural(self.block_bytes, True), "invalid block")
        require(all((self.name, self.layout, self.scale_encoding, self.numerics)), "incomplete format")
        require(self.byte_order == "little", "unsupported byte order")
        require(shape[-1] % self.block_elements == 0, "row is not block aligned")
        return prod(shape) // self.block_elements * self.block_bytes


STATE_COMPONENTS = frozenset({
    "kv", "qsa_indexer", "deltanet_recurrent", "deltanet_conv",
    "ngram_ple_history", "residual_draft", "sampler_rng_history",
})


@dataclass(frozen=True)
class Capabilities:
    identity: Identity
    formats: frozenset[Format]
    operations: frozenset[str] = frozenset({"h2d", "d2h", "compute"})
    state_components: frozenset[str] = frozenset()
    alignment: int = 128
    max_buffer_bytes: int = 1 << 32
    max_inflight: int = 16


@dataclass(frozen=True)
class Buffer:
    owner: str
    epoch: int
    serial: int
    size: int
    domain: str
    writable: bool


@dataclass(frozen=True)
class Binding:
    buffer: Buffer
    format: Format
    shape: tuple[int, ...]
    access: str
    offset: int = 0


@dataclass(frozen=True)
class Ticket:
    owner: str
    epoch: int
    serial: int


@dataclass(frozen=True)
class Fence:
    ticket: Ticket
    outcome: str


@dataclass(frozen=True)
class Transaction:
    ticket: Ticket
    session: str
    base_version: int
    base_frontier: int
    candidate_rows: int


@dataclass(frozen=True)
class Snapshot:
    version: int
    frontier: int
    checkpoint: str


@dataclass
class _Session:
    snapshot: Snapshot
    active: Transaction | None = None
    valid: bool = True
    poisoned: bool = False
    submissions: list[Ticket] = field(default_factory=list)


@dataclass
class _Pending:
    buffers: tuple[Buffer, ...]
    operation: str
    transaction: Transaction | None
    uncertain: bool = False
    fence: Fence | None = None


class ConformanceModel:
    """Conservative single-process, exclusive-buffer ownership specification.

    Serializes even read/read sharing. Real runtimes can permit more concurrency
    only with equivalent lifetime/visibility guarantees. Public methods model
    atomic events; this is deliberately not a thread-safe runtime implementation.
    """

    def __init__(self, capabilities, expected):
        have = capabilities.identity
        require(have.target == expected.target and have.config_hash == expected.config_hash,
                "wrong target/configuration")
        require(bool(have.target) and bool(have.config_hash), "missing target/configuration")
        require(have.bitstream_sha256 == expected.bitstream_sha256 and
                len(have.bitstream_sha256) == 64 and
                all(c in "0123456789abcdef" for c in have.bitstream_sha256), "wrong bitstream digest")
        require(all(natural(v) for v in (have.abi.major, have.abi.minor,
                                        expected.abi.major, expected.abi.minor)), "invalid ABI version")
        require(expected.abi == Version(1, 0), "unsupported host ABI semantics")
        require(have.abi.major == expected.abi.major and have.abi.minor >= expected.abi.minor,
                "incompatible ABI")
        require(natural(capabilities.alignment, True) and
                capabilities.alignment & (capabilities.alignment - 1) == 0, "invalid alignment")
        require(natural(capabilities.max_buffer_bytes, True) and
                natural(capabilities.max_inflight, True), "invalid resource limits")
        self.capabilities = capabilities
        self.negotiated_abi = expected.abi
        self._owner, self.epoch, self._serial = uuid4().hex, 0, 0
        self._buffers, self._pending, self._sessions = {}, {}, {}

    def _ticket(self):
        self._serial += 1
        return Ticket(self._owner, self.epoch, self._serial)

    def _buffer(self, ref):
        require(ref.owner == self._owner and ref.epoch == self.epoch and
                self._buffers.get(ref.serial) == ref, "stale/foreign/released buffer")
        return ref

    def _in_use(self, ref):
        return any(ref in p.buffers and p.fence is None for p in self._pending.values())

    def allocate(self, size, domain="device", writable=True):
        require(natural(size, True) and size <= self.capabilities.max_buffer_bytes, "invalid buffer size")
        require(domain in {"host", "device"} and type(writable) is bool, "invalid buffer domain/access")
        t = self._ticket()
        ref = Buffer(t.owner, t.epoch, t.serial, size, domain, writable)
        self._buffers[ref.serial] = ref
        return ref

    def host_access(self, ref, write=False):
        self._buffer(ref)
        require(ref.domain == "host" and not self._in_use(ref), "host access while unavailable")
        require(not write or ref.writable, "read-only buffer")

    def release(self, ref):
        self._buffer(ref)
        require(not self._in_use(ref), "release while in flight")
        del self._buffers[ref.serial]

    def _transaction(self, tx):
        require(tx.ticket.owner == self._owner and tx.ticket.epoch == self.epoch, "stale/foreign transaction")
        state = self._sessions.get(tx.session)
        require(state is not None and state.valid and state.active == tx and
                state.snapshot.version == tx.base_version, "inactive or stale transaction")
        return state

    def _command(self, ticket):
        require(ticket.owner == self._owner and ticket.epoch == self.epoch and
                ticket in self._pending, "stale/foreign/unknown completion")
        return self._pending[ticket]

    def check_fence(self, fence):
        p = self._command(fence.ticket)
        require(p.fence == fence and fence.outcome == "ok", "fence not successful/visible")

    def submit(self, operation, bindings, transaction=None, depends_on=()):
        require(operation in self.capabilities.operations and operation in {"h2d", "d2h", "compute"},
                "unsupported operation")
        require(sum(p.fence is None for p in self._pending.values()) < self.capabilities.max_inflight,
                "queue full")
        state = self._transaction(transaction) if transaction is not None else None
        require(state is None or not state.poisoned, "transaction requires rollback")
        for fence in depends_on:
            self.check_fence(fence)
        bindings = tuple(bindings)
        require(bool(bindings), "empty submission")
        refs = []
        for b in bindings:
            ref = self._buffer(b.buffer)
            require(ref not in refs and not self._in_use(ref), "buffer alias or in-flight use")
            require(b.format in self.capabilities.formats, "unsupported exact format/layout")
            size = b.format.size(b.shape)
            require(natural(b.offset) and b.offset % self.capabilities.alignment == 0 and
                    b.offset + size <= ref.size, "unaligned/out-of-bounds tensor")
            require(b.access in {"read", "write", "readwrite"} and
                    (b.access == "read" or ref.writable), "invalid buffer access")
            refs.append(ref)
        if operation == "compute":
            require(all(r.domain == "device" for r in refs), "compute requires device buffers")
        else:
            domains = ("host", "device") if operation == "h2d" else ("device", "host")
            require(len(bindings) == 2 and tuple(r.domain for r in refs) == domains and
                    tuple(b.access for b in bindings) == ("read", "write"), "invalid transfer ownership")
            require(bindings[0].format == bindings[1].format and
                    bindings[0].shape == bindings[1].shape, "DMA is not a format conversion")
        ticket = self._ticket()
        self._pending[ticket] = _Pending(tuple(refs), operation, transaction)
        if state is not None:
            state.submissions.append(ticket)
        return ticket

    def complete(self, ticket, outcome="ok", visible=True):
        p = self._command(ticket)
        require(p.fence is None, "duplicate completion")
        require(outcome in {"ok", "error", "cancelled"} and visible is True,
                "terminal completion must prove all accesses drained and visible")
        p.fence = Fence(ticket, outcome)
        if p.transaction is not None and outcome != "ok":
            self._transaction(p.transaction).poisoned = True
        return p.fence

    def _uncertain(self, ticket):
        p = self._command(ticket)
        require(p.fence is None, "already completed")
        p.uncertain = True
        if p.transaction is not None:
            self._transaction(p.transaction).poisoned = True

    def timeout(self, ticket):
        """A wait expired, NOT proof that DMA/device writes stopped."""
        self._uncertain(ticket)

    def request_cancel(self, ticket):
        """A request only; pins survive until terminal completion/reset proof."""
        self._uncertain(ticket)

    def register_session(self, name, checkpoint, frontier=0):
        require(bool(name) and name not in self._sessions, "duplicate/empty session")
        require(bool(checkpoint) and natural(frontier), "invalid checkpoint/frontier")
        self._sessions[name] = _Session(Snapshot(0, frontier, checkpoint))

    def snapshot(self, name):
        require(name in self._sessions and self._sessions[name].valid, "session requires checkpoint restore")
        return self._sessions[name].snapshot

    def begin(self, name, candidate_rows):
        require(STATE_COMPONENTS <= self.capabilities.state_components, "incomplete state capability")
        base = self.snapshot(name)
        state = self._sessions[name]
        require(state.active is None and natural(candidate_rows, True), "busy/invalid transaction")
        tx = Transaction(self._ticket(), name, base.version, base.frontier, candidate_rows)
        state.active, state.poisoned, state.submissions = tx, False, []
        return tx

    def _drained(self, state):
        require(all(self._pending[t].fence is not None for t in state.submissions),
                "transaction still has in-flight accesses")

    def commit(self, tx, accepted_rows, checkpoint):
        """Injected accepted-prefix checkpoint proof, including commit-graph fence."""
        state = self._transaction(tx)
        self._drained(state)
        require(not state.poisoned and bool(checkpoint), "poisoned transaction/invalid checkpoint")
        require(natural(accepted_rows, True) and accepted_rows <= tx.candidate_rows,
                "invalid accepted prefix")
        require(any(self._pending[t].operation == "compute" for t in state.submissions),
                "no candidate computation")
        state.snapshot = Snapshot(tx.base_version + 1, tx.base_frontier + accepted_rows, checkpoint)
        state.active = None
        return state.snapshot

    def rollback(self, tx):
        state = self._transaction(tx)
        self._drained(state)
        # Logical checkpoint restore is injected, not a device memory copy.
        state.active, state.poisoned = None, False
        return state.snapshot

    def cpu_fallback_checkpoint(self, name):
        snapshot = self.snapshot(name)
        require(self._sessions[name].active is None and
                all(p.fence is not None for p in self._pending.values()), "unsafe CPU fallback")
        return snapshot

    def reset(self, proven_quiescent=False):
        """Injected global stop+DMA-drain proof; invalidates ALL device state."""
        require(proven_quiescent is True, "reset needs externally proven quiescence")
        self.epoch += 1
        self._buffers.clear()
        self._pending.clear()
        for state in self._sessions.values():
            state.valid, state.active, state.poisoned = False, None, False
            state.submissions = []

    def restore_checkpoint(self, name, checkpoint, frontier):
        require(name in self._sessions and not self._sessions[name].valid, "restore only after reset")
        require(bool(checkpoint) and natural(frontier) and
                all(p.fence is not None for p in self._pending.values()), "invalid/unsafe restore")
        state = self._sessions[name]
        state.snapshot = Snapshot(state.snapshot.version + 1, frontier, checkpoint)
        state.valid = True
        return state.snapshot
