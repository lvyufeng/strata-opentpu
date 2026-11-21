# Contributor / agent instructions

Bootstrap project: do not claim implemented inference or measured performance.
Keep CPU kernels, execution, and FPGA transport in separate boundaries.
Read architecture and upstream policy before implementing adapters.
Do not import whole trees without scope/license review. Pin commits and preserve
copyright, licenses, relevant NOTICE files and modification records.
Never commit weights, bitstreams, vendor IP, keys, private host addresses, SSH
configuration, credentials or unsanitized logs.
Downloads, installs, programming, flash writes and network changes require
explicit user authorization for the operation.
Validate correctness before optimization. Retain complete routing. Report
prefill, TTFT and wall decode separately, and MTP off/on separately.
Place FPGA work based on measurements, not a requirement to use it everywhere.
Run `python3 -m unittest discover -s tests -v` and review staged diffs before
committing; bootstrap checks are not inference tests.
