# CPU integration boundary (planned)

CPU kernels and NUMA/expert scheduling changes belong in the
[Strata fork](https://github.com/lvyufeng/Strata). Keep packed expert computation
and dispatch by actual tensor format; audit numerical correctness and
single-token/small-batch behavior. This repository coordinates configurations
and reproducible measurements, not another CPU implementation.
No CPU binding is implemented here. See [policy](../../docs/upstreams.md).

The [cold-expert evidence protocol](../../benchmarks/cpu-cold-expert-protocol.md)
and [offline evidence validator](../../scripts/check_cpu_evidence.py) are
integration-owned preparation only. They require actual role-level packed formats,
safe CPU/OS ISA dispatch, numerical parity and repeated full-branch wall timing
with verified thread/NUMA placement before an experiment can be accepted. A
microbenchmark or expert library is not whole-model CPU decode. No host experiment,
upstream build or model acquisition is performed by these offline tools.
