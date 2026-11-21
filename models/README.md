# Model scope (planned)

[Qwen3.8-Flash-Next GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF):
IQ2_XS first, Q2_0/IQ3_S comparisons. Names are mixed-precision labels: read types
from metadata. Two main shards include 28.8 GB n-gram lookup. Vision/MTP separate,
not implemented. Top-10 is per layer, not for the entire forward pass.
Configurations/operator mappings only; never weights. Downloads need explicit
authorization and model revision pins; software pins do not pin checkpoints.
