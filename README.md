# trnRT for GLM-5.3-Flash

Serve GLM-5.3-Flash on one AWS trn2.48xlarge: two engines of 32 tensor-parallel ranks each behind a session-affine
router, an OpenAI-compatible API with streaming and GLM tool calls, 1M-token context, and a prefix cache that keeps
evicted KV pages and linear-attention state checkpoints in host memory.

**Status: work in progress.** This repository has not yet served on a device. A graph-equivalence gate
(`tools/graph_equiv.py`) holds it to compiling the same graphs as the engine it was extracted from; that gate's full run
and an on-device check are pending. The numbers in [docs/RESULTS.md](docs/RESULTS.md) were measured with that engine, in
the configuration this repository's defaults reproduce.

## What you need

| | |
|---|---|
| Serving host | one trn2.48xlarge: 16 Trainium2 chips (64 logical NeuronCores at LNC 2), 192 vCPUs, 2 TiB of host memory [unverified] |
| Compile host | any x86_64 host with the Neuron compiler (the same container image); no NeuronCore needed. Its core count sets the compile time, and its memory need for a 32-rank capture is [unverified] |
| Software | Neuron SDK 2.32 (neuronx-cc 2.27), Python 3.13: the container in `docker/` pins the base image by digest |
| Model | `zai-org/GLM-5.3-Flash` (FP8 weights, as released), at one fixed revision for both compiling and serving |
| Cost | one trn2.48xlarge for as long as it serves; see [docs/RESULTS.md](docs/RESULTS.md) for a cost per request at a public price [unverified] |

## Quickstart

### 1. Build the image

```bash
docker build -f docker/Dockerfile \
  --build-arg TRNRT_VERSION=$(python3 -c 'import trnrt; print(trnrt.__version__)') \
  --build-arg TRNRT_REVISION=$(git rev-parse HEAD) -t trnrt .
```

The build checks every Neuron piece trnRT imports and its version, and that the image's transformers has `glm5_next`
(`docker/check_image.py`).

### 2. Build the compile cache (CPU host)

A graph depends on the weights' shapes and dtypes, never their values. So the cache is built from the checkpoint's
config and safetensors headers (a few MB, read without downloading the weights) on a host without NeuronCores:

```bash
OUT=/data/trnrt            # everything goes under one directory
REV=<GLM-5.3-Flash commit>
mkdir -p $OUT/nki          # the NKI kernel binaries, shared by every step
# each step: `python -m trnrt` in a new container of the image, with $OUT at the same path and $OUT/nki mounted
trnrt() { docker run --rm -v $OUT:$OUT -v $OUT/nki:/var/tmp/nki-intermediate-cache trnrt "$@"; }
trnrt compile shapes zai-org/GLM-5.3-Flash --revision $REV --out $OUT
trnrt compile decisions --model zai-org/GLM-5.3-Flash --revision $REV --out $OUT   # a few range reads per MoE layer
trnrt compile capture --out $OUT -- --model $OUT/shapes                            # every graph, all 32 ranks
trnrt compile compile --out $OUT                                                    # neuronx-cc on each graph
trnrt compile strip --out $OUT                                                      # drop the NEFFs' debug info
```

Each step is its own `docker run --rm`, so whatever a step leaves inside its container is gone by the next. A graph
that calls an NKI kernel names the kernel's compiled binary by absolute path under `/var/tmp/nki-intermediate-cache`.
The capture writes those binaries, and neuronx-cc adds its own kernels there while compiling. So `capture` and
`compile` must see the same directory at that path: the host's `$OUT/nki`, mounted into every step above. Without it,
or on another host without a copy of it, `compile` fails every such graph ("NKI kernel binaries missing"). To split the
steps across hosts, copy `$OUT/nki` with the cache. A serving engine traces its graphs again and rewrites the
directory; whether it needs the compile host's copy on a cache hit is [unverified], so `serve` below mounts it too.

`compile` runs as many neuronx-cc processes at once as host memory admits. One can take tens of GB, so each
reserves `--mem-per-compile-gb` (default 30) of MemAvailable, and one the OOM killer takes runs once more, alone.

`strip` is not optional at 1M-token buckets: the runtime holds each loaded NEFF in host memory several times over, and
two engines' unstripped NEFFs do not fit.

The arguments after `capture --` must be the engine arguments the server runs with, spelled out in full. Capture refuses
any other; the server's own, such as `--engines` and `--port`, do not change the graphs. The defaults are the served
configuration, so `--model` alone is enough.

### 3. Serve (trn2.48xlarge)

Copy `$OUT/cache` and `$OUT/nki` to the serving host, download the checkpoint at the same revision, and start the
server:

```bash
docker run --rm --privileged --network host \
  -v /data/trnrt/cache:/cache -v /data/trnrt/nki:/var/tmp/nki-intermediate-cache -v /models/GLM-5.3-Flash:/model \
  -e NEURON_LIBTORCH_CACHE_ROOT=/cache \
  trnrt serve --model /model --port 8000
```

`serve` starts two engines, on logical NeuronCores 0-31 and 32-63, on ports 8001 and 8002 of the host, one at a time:
each loads its weights and warms every compiled graph from the cache, and the next starts once it answers `/health`.
Then the router answers on `--port`.

- **A graph missing from the cache fails start-up.** The engines run with `NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`: instead
  of compiling on the serving host (neuronx-cc, minutes to hours per graph), the engine stops and names the missing
  key, and `serve` stops every engine and exits nonzero. The cache was built for other arguments, another checkpoint
  revision, another trnRT version or another Neuron SDK, or `NEURON_LIBTORCH_CACHE_ROOT` does not point at it.
  `--allow-compile` compiles the missing graphs on the spot instead; give it a `--startup-timeout` that allows for that
  (0: none).
- **`--startup-timeout`** (default 3600 s): how long each engine has, from its start, to answer `/health`. Past it,
  `serve` stops every engine and exits nonzero.
- The router's `/health` answers 200 while both engines are up, 200 with `"status": "degraded"` while one is (its
  requests and sessions go to the other), and 503 while neither is. The body names the engines that are down.
- SIGTERM (`docker stop`) stops the router, then each engine, and `serve` exits with status 143.

### 4. Use it

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
- Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics` (Prometheus).
- Not supported, answered with 400: repetition, frequency and presence penalties, guided decoding (`response_format`
  other than text, `tool_choice` `"required"` or a named tool), prompt logprobs and thinking budgets. Every graph
  samples on the device, and these need host inputs no served graph takes.
- Status codes: 400 for a request that cannot be served as sent (a field of the wrong type or out of range, a prompt
  longer than the context, a chat template variable the template never reads); 500 for an error inside the engine;
  503 once the engine has failed, with the reason on `/health`.

## Configuration

Every `serve` default is the served configuration:

| Argument | Default | |
|---|---|---|
| `--engines`, `--tp` | 2, 32 | engines per host, tensor-parallel ranks per engine |
| `--max-model-len` | 1048576 | context per request |
| `--max-num-seqs`, `--decode-buckets` | 8; 1,2,4,6,8 | requests decoding at once per engine, compiled batch sizes |
| `--max-prefill-tokens`, `--prefill-buckets` | 4096; 1024,2048,4096 | prefill chunk per step, compiled chunk sizes |
| `--page-size`, `--page-buckets`, `--prefill-page-buckets` | 256; 512,1024,4096; 1024,4096 | KV page, compiled context lengths in pages |
| `--kv-cache-gb`, `--kv-cache-dtype` | 4.0, fp8 | KV pool per rank |
| `--state-checkpoints` | 16 | linear-attention state rows kept for the prefix cache |
| `--host-cache-gb` | 4.0 | host memory per rank under the prefix cache (half for state checkpoints) |

The bucket ladders, the pool sizes and the page size are graph shapes: changing any of them needs a new capture.
`python -m trnrt engine` runs one engine without the router.

## Tests

```bash
pip install -e '.[test]'
pytest
```

The CPU tests run without a NeuronCore. Engine tests use a tiny random model of GLM-5.3-Flash's architecture and need a
transformers release with `glm5_next`.

## License

Apache License 2.0 (LICENSE). NOTICE and THIRD_PARTY_NOTICES.md list the code this repository derives from.
