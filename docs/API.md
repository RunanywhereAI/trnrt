# API

The router on `--port` (default 8000) speaks the OpenAI API: `/v1/chat/completions`, `/v1/completions`, `/v1/models` and `/health`, plus `/tokenize` and `/detokenize` ([below](#tokenize-and-detokenize)). Any OpenAI client works with `base_url="http://<host>:8000/v1"`. It also speaks SGLang's native API: [`/generate`](#generate) and the [info and admin endpoints](#info-and-admin-endpoints).

## Example

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm",
  "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
  "session_id": "my-agent-session-1",
  "stream": true
}'
```

- `session_id` pins a conversation to one engine, so each turn resumes from the prefix cache. Without it, the router
  routes by the longest prefix it has sent to each engine.
- Reasoning comes back as `reasoning_content`, and tool calls as OpenAI `tool_calls`.
- GLM-5.3-Flash always reasons: its chat template has no switch to turn thinking off, so
  `"chat_template_kwargs": {"enable_thinking": false}` is refused. `"reasoning_effort"` sets how hard it reasons:
  `"low"`, `"high"` or `"max"`, the default. Other values are refused, since the template would answer them at max.
- `max_tokens` counts reasoning too. Without it, a chat request may generate up to the rest of the context
  (`--max-model-len` minus the prompt), as vLLM does; a completion stops at 16 tokens, the OpenAI API's default.
- Sampling defaults come from the checkpoint's `generation_config.json`, as in vLLM: a request that sets no
  `temperature`, `top_p`, `top_k` or `min_p` gets the checkpoint's value (GLM-5.3-Flash: temperature 1.0, top_p 0.95).
  Without one, temperature and top_p are 1.0. Sampling draws from the 64 most likely tokens at most
  (`trnrt/engine/sampler.py`), so top_p 1.0 is top-64.
- A response cut off by `max_tokens` has `finish_reason` `"length"`, even when it ends in a tool call: that call may
  be incomplete.
- A client that disconnects aborts its request, streamed or not.
- Endpoints on `--port` (the router): `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/tokenize`,
  `/detokenize`, and SGLang's `/generate`, `/model_info`, `/server_info`, `/get_load`, `/health_generate`,
  `/abort_request` and `/flush_cache`. Each
  engine also serves Prometheus metrics at `/metrics` on its own port (8001 and 8002 with the default `--port`), on
  127.0.0.1 only, among them the sparse-attention selection's counters (`trnrt:dsa_local_k_*`,
  [KNOWN_ISSUES.md](KNOWN_ISSUES.md)).
- The router listens on every interface (`--host 0.0.0.0`) and has no authentication: put it behind your own proxy, or
  pass `--host 127.0.0.1`.
- The router keeps an idle client connection open for 75 s (`--timeout-keep-alive`). A proxy or client connection pool
  in front of it should drop idle connections sooner than that; otherwise a request can go out on a connection the
  router is closing.
- Not supported, answered with 400: repetition, frequency and presence penalties, guided decoding (`response_format`
  other than text, `tool_choice` `"required"` or a named tool), prompt logprobs and thinking budgets. Every graph
  samples on the device, and these need host inputs no served graph takes.
- Status codes: 400 for a request that cannot be served as sent (a field of the wrong type or out of range, a prompt
  longer than the context, a chat template variable the template never reads); 500 for an error inside the engine;
  503 once the engine has failed, with the reason on `/health`.

## Tokenize and detokenize

`/tokenize` and `/detokenize` (also under `/v1`) answer as SGLang and vLLM do. Neither runs the model, and the router
sends them to any live engine.

| Endpoint | Request | Response |
|---|---|---|
| `/tokenize` | exactly one of `prompt` (a string, or a list of strings for a batch) or `messages` (with the chat fields `/v1/chat/completions` takes: `tools`, `tool_choice`, `chat_template_kwargs`, `reasoning_effort`); `add_special_tokens` (prompt only, default `true`); `return_token_strs` | `tokens`, `count`, `max_model_len`, and `token_strs` when asked |
| `/detokenize` | `tokens`: a list of token ids, or a list of such lists for a batch; `skip_special_tokens` (default `true`) | the text as `text` (SGLang's name) and as `prompt` (vLLM's) |

A chat request is counted exactly as `/v1/chat/completions` would send it to the engine, rendered by the model's chat
template, so `count` is the prompt length that request would use against `--max-model-len`. Fields the chat endpoint
refuses that would change the rendering (`add_generation_prompt` false, `continue_final_message` true) are refused here
too.

```bash
curl -s localhost:8000/tokenize -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "What is the weather in Paris?"}]}'
```

## Generate

`/generate` (POST or PUT) is SGLang's native generation endpoint, with SGLang's request and response shapes, so SGLang
clients work unchanged. It runs through the same engine path as `/v1/completions`: the same checks, stop strings, and a
client that disconnects aborts its request. The router routes it like a completion, by its prompt or its `session_id`.

| Request field | Meaning |
|---|---|
| `text` or `input_ids` | One prompt (a string, or a list of token ids), or a batch (a list of either). Exactly one of the two. |
| `sampling_params` | SGLang's names: `max_new_tokens` (default 128), `temperature`, `top_p`, `top_k`, `min_p`, `stop`, `stop_token_ids`, `ignore_eos`, `sampling_seed`. A batch may send one object per prompt. Defaults come from the checkpoint's `generation_config.json`, as on the OpenAI routes. |
| `stream` | Server-sent events, then `data: [DONE]`. Each event carries the output so far (SGLang's default): `text`, `output_ids` and `meta_info`. A batch's events also carry their prompt's `index`. |
| `return_logprob`, `top_logprobs_num` (0 to 20), `return_text_in_logprobs` | Output-token logprobs in `meta_info`. |
| `logprob_start_len` | `-1` (or absent) only, when `return_logprob` is set. |
| `rid` | The request's id, which `/abort_request` names. A batch's ids are `rid_0`, `rid_1`, ..., or a list of one per prompt. A rid already running is refused. |
| `session_id` | As on the OpenAI routes: the router pins the session to one engine. |
| `return_prompt_token_ids` | Adds `prompt_token_ids` to the response. |
| `log_metrics` | `false` leaves the request out of `/metrics`. |

The response is one object for one prompt and a list for a batch:

| Field | Value |
|---|---|
| `text` | The output, cut before a stop string. |
| `output_ids` | The output token ids. |
| `meta_info.id` | The rid. |
| `meta_info.finish_reason` | `{"type": "stop", "matched": <stop string or token id>}`, `{"type": "length", "length": n}`, `{"type": "abort", "message": "Aborted"}`, or `null` while a stream runs. |
| `meta_info.prompt_tokens`, `completion_tokens`, `cached_tokens` | Token counts; `cached_tokens` is the prompt tokens resumed from the prefix cache. |
| `meta_info.num_retractions`, `e2e_latency` | Times the request was preempted; seconds from arrival to finish. |
| `meta_info.output_token_logprobs` | With `return_logprob`: `[logprob, token_id, text or null]` per output token. |
| `meta_info.output_top_logprobs` | With `top_logprobs_num` > 0: a list of `[logprob, token_id, text or null]` per output token. |
| `meta_info.input_token_logprobs`, `input_top_logprobs` | Always `[]`: prompt logprobs are not computed. |

Answered with 400, naming the field: prompt logprobs (`logprob_start_len` >= 0 with `return_logprob`), `n` > 1,
guided decoding (`json_schema`, `regex`, `ebnf`, `structural_tag`), `min_new_tokens`, `no_stop_trim`, `logit_bias`,
`skip_special_tokens` false, penalties, multimodal input, LoRA, hidden states, `priority`, `session_params` and the
other GenerateReqInput fields this server does not serve. An unknown `sampling_params` key is refused too, as SGLang
does. Each is accepted at its neutral value.

```bash
curl -s localhost:8000/generate -H 'Content-Type: application/json' \
  -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 16, "temperature": 0}}'
```

## Info and admin endpoints

SGLang's names (the `get_` forms are SGLang's older ones, served too). The router answers each from one engine or from
all of them:

| Endpoint | Engine answer | Router |
|---|---|---|
| `GET /model_info`, `/get_model_info` | `model_path` (as passed to `--model`), `served_model_name`, `is_generation`, `preferred_sampling_params` (the checkpoint's defaults), `reasoning_parser` and `tool_call_parser` (SGLang's names for the equivalent parsers: `glm45`, `glm47`), `model_type`, `architectures` | The first live engine |
| `GET /server_info`, `/get_server_info` | The engine's configuration (every `EngineConfig` field), `tp_size`, `dp_size`, `context_length`, `preset`, `version`, `startup_seconds`, and `internal_states` (its `/get_load` entry) | The first answer, with `dp_size` the number of engines and `internal_states` one entry per engine |
| `GET /get_load` | `[{dp_rank, num_reqs, num_running_reqs, num_waiting_reqs, num_tokens, num_pending_tokens, token_usage, ts_tic}]`: `num_tokens` is the KV tokens in use plus the waiting requests' tokens | Every engine's entry, `dp_rank` its index |
| `GET /health_generate` | 200 once a step ends after a one-token probe is sent (the probe itself on an idle engine); 503 if none ends within 20 s, or once the engine failed. The probe is not counted in `/metrics` | Every engine: 200 `"ok"` or `"degraded"` while any generated, 503 `"down"` when none did, naming the engines down |
| `POST /abort_request` | `{"rid": ...}` aborts that request (an exact match); `{"abort_all": true}` aborts every request. The aborted request answers with its output so far and finish reason `abort`. 200 whether or not the rid runs | Every engine: only the one holding the rid has it. 200 once any engine took it |
| `GET` or `POST /flush_cache` | Drops the prefix cache (device pages, state checkpoints and the host-memory tier), only while no request runs or waits. `?timeout=<s>` (default 0) waits up to that long for the engine to be idle. 200 when flushed, 400 with the reason when not | Every engine; 200 only when all flushed. The router also forgets a flushed engine's prefixes |

None of these returns a secret or a host path the operator did not pass.
