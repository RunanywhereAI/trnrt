# API

The router on `--port` (default 8000) speaks the OpenAI API: `/v1/chat/completions`, `/v1/completions`, `/v1/models` and `/health`, plus `/tokenize` and `/detokenize` ([below](#tokenize-and-detokenize)). Any OpenAI client works with `base_url="http://<host>:8000/v1"`.

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
- Endpoints on `--port` (the router): `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/tokenize`
  and `/detokenize`. Each
  engine also serves Prometheus metrics at `/metrics` on its own port (8001 and 8002 with the default `--port`), on
  127.0.0.1 only, among them the sparse-attention selection's counters (`trnrt:dsa_local_k_*`,
  [KNOWN_ISSUES.md](KNOWN_ISSUES.md)).
- The router listens on every interface (`--host 0.0.0.0`) and has no authentication: put it behind your own proxy, or
  pass `--host 127.0.0.1`.
- The router keeps an idle client connection open for 75 s (`--timeout-keep-alive`). A proxy or client connection pool
  in front of it should drop idle connections sooner than that; otherwise a request can go out on a connection the
  router is closing.
- Structured outputs, penalties, `min_tokens` and thinking budgets need an engine started with `--structured-outputs` or
  `--penalties` ([below](#structured-outputs-penalties-and-thinking-budgets)); otherwise they are answered with 400,
  naming the flag.
- Not supported, answered with 400: `tool_choice` `"required"` or a named tool, and prompt logprobs (`prompt_logprobs`,
  `echo` with `logprobs`).
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

## Structured outputs, penalties and thinking budgets

Each needs graphs that take inputs from the host. An engine serves them only when started with the flag
([CONFIGURATION.md](CONFIGURATION.md#structured-outputs-penalties-and-thinking-budgets)).

| Field | Flag | Behaviour |
|---|---|---|
| `response_format` `{"type": "json_object"}` | `--structured-outputs` | a JSON object (the schema `{"type": "object"}`, as SGLang) |
| `response_format` `{"type": "json_schema", "json_schema": {"schema": ...}}` | `--structured-outputs` | JSON matching the schema |
| vLLM `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`; `structured_outputs` `{json, regex, choice, grammar, json_object}` | `--structured-outputs` | a schema, a regular expression, one of the strings, an EBNF grammar |
| SGLang `json_schema` (a string), `regex`, `ebnf` | `--structured-outputs` | the same |
| `min_tokens` | `--structured-outputs` | no end-of-sequence or `stop_token_ids` id, and no stop string, among the first `min_tokens` tokens |
| `thinking_token_budget` | `--structured-outputs` | at most this many reasoning tokens after `<think>`; then `</think>` is forced |
| `repetition_penalty`, `frequency_penalty`, `presence_penalty` | `--penalties` | vLLM's formulas: repetition over prompt and output tokens, frequency and presence over the output |

- One constraint per request. Two (say `response_format` and `regex`) are refused with 400, as SGLang refuses them.
- Guided decoding uses [xgrammar](https://github.com/mlc-ai/xgrammar), an optional dependency:
  `pip install 'trnrt[grammar]'`. Without it, guided requests are answered with 400.
- GLM-5.3-Flash reasons first. As in vLLM, the constraint applies to the answer after `</think>`, not to the reasoning.
- Logprobs stay the raw model's, before the mask and the penalties, as vLLM reports them by default.
- A request with penalties, a grammar, or a thinking budget near its end waits for its newest token before each step,
  so it gives up the overlap of host and device work; the requests beside it keep it. Under the dynamic `mtp-*` presets
  the engine stops speculating while a request that uses any of these runs.
- If a grammar can continue only with text no token spells, the request ends there with `finish_reason` `"stop"`, and a
  warning is logged.
