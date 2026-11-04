# API

The router on `--port` (default 8000) speaks the OpenAI API: `/v1/chat/completions`, `/v1/completions`, `/v1/models` and `/health`. Any OpenAI client works with `base_url="http://<host>:8000/v1"`.

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
- A response cut off by `max_tokens` has `finish_reason` `"length"`, even when it ends in a tool call: that call may
  be incomplete.
- A client that disconnects aborts its request, streamed or not.
- Endpoints on `--port` (the router): `/v1/chat/completions`, `/v1/completions`, `/v1/models` and `/health`. Each
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
