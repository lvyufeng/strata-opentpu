# FPGA integration boundary (planned)

RTL/ISA, quantized hardware computation, DMA and device runtime changes belong
in the [openTPU fork](https://github.com/lvyufeng/openTPU). Strata's FPGA backend
and model-side scheduling belong in the [Strata fork](https://github.com/lvyufeng/Strata).
This repository locks their ABI-compatible combinations and coordinates future
build/program/test operations. No integrated FPGA backend or bitstream is provided.

Native support for actual IQ tensor formats needs implementation and measurement;
existing FP4 execution does not imply IQ2_XS support. Verify buffer lifetime,
submission/completion and errors. Device programming is never an automatic test
step, and volatile loading is distinct from persistent flash writes.
