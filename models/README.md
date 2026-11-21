# Model references (not an executable profile)

[Qwen3.8-Flash-Next GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF):
IQ2_XS first, Q2_0/IQ3_S comparisons. Names are mixed-precision labels: read types
from metadata. Two main shards include 28.8 GB n-gram lookup. Vision/MTP integration
is not validated; reuse Strata's existing model/MTP implementation rather than
create one here. Top-10 is per layer, not for the entire forward pass.
Configurations/operator mappings only; never weights. Downloads need explicit
authorization and model revision pins; software pins do not pin checkpoints.
