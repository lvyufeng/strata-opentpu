# CPU integration boundary (planned)

CPU kernels and NUMA/expert scheduling changes belong in the
[Strata fork](https://github.com/lvyufeng/Strata). Keep packed expert computation
and dispatch by actual tensor format; audit numerical correctness and
single-token/small-batch behavior. This repository coordinates configurations
and reproducible measurements, not another CPU implementation.
No CPU binding is implemented here. See [policy](../../docs/upstreams.md).
