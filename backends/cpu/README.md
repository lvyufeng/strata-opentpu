# CPU backend (planned)

Audit/adapt Strata AVX2 packed expert kernels through a small native interface.
Decode blocks during dot products; dispatch by real GGUF type, quantize
activations, expose capabilities, benchmark threads/NUMA. No bindings or imported
kernels yet. See [policy](../../docs/upstreams.md).
