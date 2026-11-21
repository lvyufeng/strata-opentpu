# Ideas and hypotheses

This directory is for design proposals and falsifiable hypotheses, not claims of
implemented features. See the [architecture](../architecture.md) and
[roadmap](../roadmap.md) for accepted boundaries and current capability status.

Strata already provides resident hot GPU experts, cold-expert CPU computation,
adaptive caching, and MTP. A proposal should identify what changes for an FPGA
backend rather than presenting those existing mechanisms as new inventions.

## Tracking proposals

Record actionable proposals in a [central Issue](https://github.com/lvyufeng/strata-opentpu/issues)
and add them to the [development Project](https://github.com/users/lvyufeng/projects/11).
Link that Issue from any versioned design note and link the note back from the
Issue. The Issue owns live decisions, executor and next steps; this directory is
not a separate backlog. Follow the [development contract](../development.md).

Useful initial questions:

- Which actual expert quantization format covers enough bytes/calls to justify a
  native compressed-block FPGA decode/dot path?
- How much board DDR remains for hot experts after fixed weights, state, and
  buffers, and does measured routing reuse justify residency?
- Can full expert branches stay on-device without costly intermediate transfers?
- How should cold work be distributed across NUMA nodes while minimizing the
  completion tail of a full top-10 branch?
- Can Strata MTP verification reuse weights efficiently on this backend without
  compromising state commit/rollback or increasing total wall latency?

For each idea, record: hypothesis, upstream mechanism being reused, proposed
fork changes, required formats/operators, capacity/transfer budget, smallest
correctness experiment, baseline, success/rejection criteria, and licensing or
safety dependencies. Separate analytical estimates, simulations, and real device
measurements. Neither a proposal nor a cached-route simulation proves full-model
throughput. Reference sanitized evidence in [experiments](../../experiments/README.md).
