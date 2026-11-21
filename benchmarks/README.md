# Benchmark protocol (planned)

No inference measurements here yet.
The [CPU cold-expert evidence protocol](cpu-cold-expert-protocol.md) and
[partial JSON template](templates/cpu-cold-expert.json) prepare reproducible block,
dot and full gate/up/activation/down branch experiments. Their
[offline validator](../scripts/check_cpu_evidence.py) checks declared evidence
consistency, not its scientific authenticity; it does not build or run CPU kernels.
An accepted branch experiment is still not full-model CPU decode or a wall-token
rate. Actual experiments need specific host/input/build authorization.

Record software/checkpoint revisions, checksums, actual formats, CPU features,
threads/NUMA, bitstream/config, negotiated link/measured DMA, context/output count,
batch/concurrency, sampling/seed, warm/cold state, partition and memory footprint.
Use full top-10, expert skip=0. Report MTP off/on separately with acceptance stats.

Measure activation preparation, block decoding/dot, single/multi-token separately,
bytes read and numerical error. Trace serial work, CPU experts, FPGA, transfers,
sync/merge and errors. Account for true overlap, not sums of standalone rates.

Separate prefill/TTFT from decode. Warm up then generate >=128 tokens (prefer 256),
record timing boundaries. Wall decode counts committed tokens, includes host,
transfers/sync, excludes unaccepted speculative drafts. Document early stops.
Repeat comparable runs and report variation. Correctness before speed; compare
quality across quantization variants.

Target: >10 single-user wall decode tokens/s at stated conditions. Upstream GPU
and analytical card rates are not local results. Keep raw identifying logs out
of Git; publish reviewed sanitized summaries with reproducible metadata.
