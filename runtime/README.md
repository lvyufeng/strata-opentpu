# Runtime integration boundary (planned)

Model execution, routing, expert scheduling, synchronization and MTP changes
belong in the [Strata fork](https://github.com/lvyufeng/Strata), not a second
engine here. This directory is reserved for integration contracts and launch
configuration. No device runtime/API is implemented in this repository.

The [proposed device ABI](../docs/design/device-abi.md) and its
[host-only conformance model](abi_contract.py) describe and test a future
cross-fork boundary. The model does not open devices, transfer tensors, execute
operators or make the bootstrap profile inference-ready. Production device
transport belongs in openTPU; scheduling and model state integration belong in
Strata. See the [source execution audit](../docs/audits/strata-execution.md),
[architecture](../docs/architecture.md) and [roadmap](../docs/roadmap.md).
