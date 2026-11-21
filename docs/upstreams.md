# Upstream reuse and licensing

[upstreams.json](../upstreams.json) records bootstrap references, not a build
lockfile or compatibility claim. No upstream implementation downloaded, vendored
or linked yet.

| Project | Planned role | License |
| --- | --- | --- |
| [Strata](https://github.com/Niko1221/Strata) | Audited CPU experts/quantization | MIT |
| [openTPU](https://github.com/FeSens/openTPU) | FPGA execution/transport | Apache-2.0 |

Strata currently may use its own CPU quant layer; do not assume every path wraps
ggml-cpu. Audit full dependency closure and embedded third-party code.
Before import/distribution, record commits/files/patches; check each component's
license. Preserve Strata copyright/MIT, openTPU Apache-2.0 and relevant NOTICE if
present; mark modifications. Review binary distribution obligations. Original
Apache-2.0 does not relicense upstream code, models or vendor IP. Model licensing
cannot be inferred solely from quantized-repository YAML.
Prefer narrow adapters and pins over full trees; evaluate scoped vendoring,
submodules or a dependency fork at M1. This project is unofficial, not endorsed.
