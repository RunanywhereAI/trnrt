# API

The router on `--port` (default 8000) speaks the OpenAI API: `/v1/chat/completions`, `/v1/completions`, `/v1/models` and `/health`, plus `/tokenize` and `/detokenize` ([below](#tokenize-and-detokenize)). Any OpenAI client works with `base_url="http://<host>:8000/v1"`. It also serves the OpenAI Responses API ([below](#responses-api)) and the Anthropic Messages API ([below](#anthropic-messages-api)).

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
- Endpoints on `--port` (the router): `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`,
  `/v1/messages/count_tokens`, `/v1/models`, `/health`, `/tokenize` and `/detokenize`. Each
  engine also serves Prometheus metrics at `/metrics` on its own port (8001 and 8002 with the default `--port`), on
  127.0.0.1 only, among them the sparse-attention selection's counters (`trnrt:dsa_local_k_*`,
  [KNOWN_ISSUES.md](KNOWN_ISSUES.md)).
- The router listens on every interface (`--host 0.0.0.0`) and has no authentication: put it behind your own proxy, or
  pass `--host 127.0.0.1`.
- The router keeps an idle client connection open for 75 s (`--timeout-keep-alive`). A proxy or client connection pool
  in front of it should drop idle connections sooner than that; otherwise a request can go out on a connection the
  router is closing.
- Not supported, answered with 400: repetition, frequency and presence penalties, guided decoding (`response_format`
  other than text, `tool_choice` `"required"` or a named tool), prompt logprobs, thinking budgets, and a minimum length
  (`min_tokens`, SGLang's `min_new_tokens`). Every graph samples on the device, and these need host inputs no served
  graph takes. A minimum length needs EOS masked out of the distribution a token is drawn from; ignoring EOS on the
  host instead would put it in the output and let the model write past it.
- Status codes: 400 for a request that cannot be served as sent (a field of the wrong type or out of range, a prompt
  longer than the context, a chat template variable the template never reads); 500 for an error inside the engine;
  503 once the engine has failed, with the reason on `/health`.

## Sampling extras

These fields work on `/v1/chat/completions` and `/v1/completions`, as vLLM and SGLang take them.

| Field | What it does |
|---|---|
| `n` | Up to 128 choices of one prompt, each its own engine request. The prefix cache computes the shared prompt once: the scheduler holds the other choices until they can resume from the first one's prompt checkpoint. Each choice has its `index`, streamed chunks included. `usage` counts the prompt once and every choice's output; `cached_tokens` is the fewest any choice read from the cache. A client that disconnects aborts every choice. |
| `best_of` | Accepted only equal to `n` (ranking extra candidates needs their logprobs, which no request returns). |
| `seed` with `n` | Choice i samples with `seed + i`, as vLLM does, so the choices differ and each is reproducible. |
| `stop_regex` | SGLang's: a pattern or a list of up to 16, matched on the output text. The output ends at the end of the first match, the matched text included (as in SGLang), with `finish_reason` `"stop"`. |
| `include_stop_str_in_output`, `no_stop_trim` | vLLM's and SGLang's names for one switch: keep the stop string, or the stop token's text, in the output. |
| `skip_special_tokens` | `false` keeps special tokens in the output text. |
| `stop_token_ids` | Tokens that end the request. As in vLLM and SGLang, the stop token (or EOS) is not part of the text. |

Every finished choice carries SGLang's `matched_stop`: the stop string or `stop_regex` pattern that ended it, the token id
that did (EOS or a stop token), or `null` (a length).

## Responses API

`POST /v1/responses` serves the OpenAI Responses API, stateless. It is translated into a chat request, so the same
prompt, sampling and GLM parsers apply, and its output comes back as Responses items.

| Request field | Served as |
|---|---|
| `input` (a string, or message, `reasoning`, `function_call` and `function_call_output` items), `instructions` | the chat messages; an assistant turn's items join into one assistant message |
| `tools` (function tools), `tool_choice` | chat tools; `"auto"` and `"none"` (`"required"` and a named function are refused, as in chat) |
| `max_output_tokens`, `temperature`, `top_p`, `stream`, `parallel_tool_calls`, `session_id`, the chat sampling extras | the same chat fields |
| `reasoning.effort` | `reasoning_effort`: `"low"`, `"high"` or `"max"`, as in chat |
| `reasoning.summary` | a `summary_text` that mirrors the reasoning (there is no summarizer), as SGLang does |
| `include: ["reasoning.encrypted_content"]` | the reasoning text in the reasoning item's `encrypted_content`, not encrypted, so a client can send it back |

| Output | Item |
|---|---|
| reasoning | a `reasoning` item with `reasoning_text` content |
| the answer | a `message` item with one `output_text` part |
| each tool call | a `function_call` item; send its result back as `function_call_output` with its `call_id` |

A response cut off by `max_output_tokens` has status `"incomplete"`. `usage.output_tokens_details.reasoning_tokens`
counts the output tokens before `</think>`. Streamed, the events are `response.created`, `response.in_progress`, each
item's `response.output_item.added`, deltas and `.done` events, then `response.completed` (`response.incomplete`, or
`response.failed` on an engine error).

Nothing is stored. Refused with 400: `store: true`, `previous_response_id`, `background`, `truncation: "auto"`,
`max_tool_calls`, `top_logprobs`, `text.format` other than text, other `include` values, item references, non-text
input parts and tools other than functions. `GET`, `DELETE` and `cancel` of a response answer 404.

## Anthropic Messages API

`POST /v1/messages` serves the Anthropic Messages API, and `POST /v1/messages/count_tokens` counts a request's input
tokens exactly as `/v1/messages` would render it (the same count `/tokenize` gives for the chat messages). Point an
Anthropic client at `base_url="http://<host>:8000"`.

| Request field | Served as |
|---|---|
| `system` (a string or text blocks), `messages` with `text`, `tool_use`, `tool_result` and `thinking` blocks | the chat messages: a user's tool results become tool messages in their place, an assistant's thinking its `reasoning_content` |
| `tools` (custom tools: `name`, `description`, `input_schema`), `tool_choice` `auto` / `none`, `disable_parallel_tool_use` | chat tools, `tool_choice`, `parallel_tool_calls: false` |
| `max_tokens` (required), `temperature`, `top_p`, `top_k`, `stop_sequences`, `stream`, `session_id` | the same chat fields; `stop` |
| `thinking` `{"type": "adaptive"}` or absent | GLM-5.3-Flash reasons as always |
| `output_config.effort` | `reasoning_effort`: `"low"`, `"high"` or `"max"` |

The reply's content is a `thinking` block (with an empty `signature`), a `text` block and `tool_use` blocks.
GLM-5.3-Flash always reasons, so the `thinking` block comes back whether or not the request set `thinking`.
`stop_reason` is `end_turn`, `max_tokens`, `stop_sequence` (with `stop_sequence`) or `tool_use`. `usage.input_tokens`
counts the prompt tokens not read from the prefix cache and `cache_read_input_tokens` those that were, as Anthropic
splits them. Streamed, the events are Anthropic's: `message_start`, each block's `content_block_start`, deltas
(`thinking_delta`, `text_delta`, `input_json_delta`) and `content_block_stop`, then `message_delta` and `message_stop`.

Errors use Anthropic's shape, `{"type": "error", "error": {"type": "invalid_request_error", "message": ...}}`. Refused
with 400: `thinking` `{"type": "disabled"}` (the template cannot turn reasoning off), `{"type": "enabled"}` with
`budget_tokens` (a thinking budget, which no served graph enforces), `tool_choice` `any` and `tool` (guided decoding),
image and document blocks, `redacted_thinking`, server tools, `mcp_servers`, `container` and `context_management`.

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
