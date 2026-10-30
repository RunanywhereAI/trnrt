# Third-party notices

This repository is licensed under the Apache License, Version 2.0 (LICENSE); NOTICE carries the attribution the license
requires. The files below include source from other projects, each under its own license.

## Source included in this repository

| Path | Origin | License | Changes |
|---|---|---|---|

## Designs followed, no source copied

These are implemented here from their published descriptions or reference code, without copying source; they are listed
for credit, not because their licenses require it.

| Component | Reference | License of the reference |
|---|---|---|
| KDA chunked prefill (the per-chunk quantities of the delta-rule recurrence) | flash-linear-attention, `fla/ops/kda/chunk_fwd.py` (`chunk_kda_fwd`), https://github.com/fla-org/flash-linear-attention | MIT |
| Radix prefix cache | SGLang's RadixAttention, https://github.com/sgl-project/sglang | Apache-2.0 |

## Models and services this code works with

GLM-5.3-Flash (`zai-org/GLM-5.3-Flash` on Hugging Face) is released under the MIT license; its weights are downloaded at run
time and are not part of this repository. The AWS Neuron SDK (neuronx-cc, NKI, the Neuron runtime) is installed separately
under AWS's terms.
