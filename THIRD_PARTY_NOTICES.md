# Third-party notices

This repository is licensed under the Apache License, Version 2.0 (LICENSE); NOTICE carries the attribution the license
requires. The files below include source from other projects, each under its own license.

## Source included in this repository

| Path | Origin | License | Changes |
|---|---|---|---|
| `trnrt/serve/parsers.py` | vLLM, https://github.com/vllm-project/vllm, Copyright contributors to the vLLM project. Portions adapted from v0.30.0 (commit ced6857afa0ea7b2e3f0846a62e1394e90f15607): `vllm/parser/glm47_moe.py`, `vllm/parser/engine/parser_engine.py`, `vllm/tool_parsers/utils.py`. Rules also followed: `tool_choice` checks (`vllm/entrypoints/openai/chat_completion/protocol.py`), the finish reason (`vllm/entrypoints/openai/chat_completion/serving.py`), and normalization of assistant tool-call arguments (`vllm/entrypoints/chat_utils.py`, commit 36fa72d2d0). | Apache-2.0 | Adapted: the GLM tool-call regexes, the argument typing by tool schema, and the rule that streamed arguments only extend what was sent follow vLLM. Reasoning and tool calls are parsed by one state machine over the detokenized text, shared by streaming and non-streaming. Whether output starts inside a reasoning block is read from the rendered prompt. A tool block that yields no call is returned as content. A call cut off by the token limit keeps its partial last argument, and its finish reason stays "length". `tool_choice` "required" and named tools are refused: vLLM enforces them with guided decoding, which this server does not run. |

## Data included in this repository

| Path | Origin | License | Changes |
|---|---|---|---|
| `tests/fixtures/GLM-5.3-Flash-config.json` | `config.json` of zai-org/GLM-5.3-Flash at revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a, https://huggingface.co/zai-org/GLM-5.3-Flash | MIT | none (byte-identical) |
| `tests/fixtures/GLM-5.3-Flash-chat_template.jinja` | `chat_template.jinja` of zai-org/GLM-5.3-Flash at revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a, https://huggingface.co/zai-org/GLM-5.3-Flash | MIT | none (byte-identical) |

## Designs followed, no source copied

These are implemented here from their published descriptions or reference code, without copying source; they are listed
for credit, not because their licenses require it.

| Component | Reference | License of the reference |
|---|---|---|
| KDA chunked prefill (the per-chunk quantities of the delta-rule recurrence) | flash-linear-attention, `fla/ops/kda/chunk_fwd.py` (`chunk_kda_fwd`), https://github.com/fla-org/flash-linear-attention | MIT |
| Radix prefix cache | SGLang's RadixAttention, https://github.com/sgl-project/sglang | Apache-2.0 |
| GLM tool-call and reasoning parsing behaviour (cross-checked against vLLM's, above) | SGLang v0.5.21 (commit e00930c5489053f26d86b179cee0d087f846acbb): `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/parser/reasoning_parser.py` | Apache-2.0 |

## Models and services this code works with

GLM-5.3-Flash (`zai-org/GLM-5.3-Flash` on Hugging Face) is released under the MIT license; its weights are downloaded at run
time and are not part of this repository. The AWS Neuron SDK (neuronx-cc, NKI, the Neuron runtime) is installed separately
under AWS's terms.
