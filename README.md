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

The build checks every Neuron piece trnRT imports (`docker/check_image.py`).

### 2. Build the compile cache (CPU host)

A graph depends on the weights' shapes and dtypes, never their values. So the cache is built from the checkpoint's
config and safetensors headers (a few MB, read without downloading the weights) on a host without NeuronCores:

```bash
OUT=/data/trnrt            # everything goes under one directory
REV=<GLM-5.3-Flash commit>
python -m trnrt compile shapes zai-org/GLM-5.3-Flash --revision $REV --out $OUT
python -m trnrt compile decisions --model zai-org/GLM-5.3-Flash --revision $REV --out $OUT   # a few range reads per MoE layer
python -m trnrt compile capture --out $OUT -- --model $OUT/shapes                            # every graph, all 32 ranks
python -m trnrt compile compile --out $OUT                                                    # neuronx-cc on each graph
python -m trnrt compile strip --out $OUT                                                      # drop the NEFFs' debug info
```

Run `capture` and `compile` on the same host. A graph that calls an NKI kernel names the kernel's compiled binary by
absolute path under `/var/tmp/nki-intermediate-cache`. The capture writes those binaries, and neuronx-cc adds its own
kernels there while compiling. A compile on another host fails every such graph ("NKI kernel binaries missing") unless
that directory comes along to the same path. To split the steps across hosts, copy `/var/tmp/nki-intermediate-cache`
with the cache. A serving engine traces its graphs again and rewrites the directory; whether it needs the compile
host's copy on a cache hit is [unverified].

`compile` runs as many neuronx-cc processes at once as host memory admits. One can take tens of GB, so each
reserves `--mem-per-compile-gb` (default 30) of MemAvailable, and one the OOM killer takes runs once more, alone.

`strip` is not optional at 1M-token buckets: the runtime holds each loaded NEFF in host memory several times over, and
two engines' unstripped NEFFs do not fit.

The arguments after `capture --` must be the ones the server runs with. The defaults are the served configuration, so
`--model` alone is enough.

### 3. Serve (trn2.48xlarge)

Copy `$OUT/cache` to the serving host, download the checkpoint at the same revision, and start the server:

```bash
docker run --rm --privileged --network host \
  -v /data/trnrt/cache:/cache -v /models/GLM-5.3-Flash:/model \
  -e NEURON_LIBTORCH_CACHE_ROOT=/cache \
  trnrt serve --model /model --port 8000
```

`serve` starts two engines, on logical NeuronCores 0-31 and 32-63, each on port 8001 and 8002 of the host. Each loads its
weights and warms every compiled graph from the cache before it answers. Then the router answers on `--port`. A
graph missing from the cache is compiled on the spot, which takes minutes per graph: if start-up compiles, the cache was
built for other arguments or another revision.

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
- Reasoning comes back as `reasoning_content`, and tool calls as OpenAI `tool_calls`. Pass
  `"chat_template_kwargs": {"enable_thinking": false}` to turn reasoning off.
- Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics` (Prometheus).
- Not supported, answered with 400: repetition, frequency and presence penalties, guided decoding, prompt logprobs and
  thinking budgets. Every graph samples on the device, and these need host inputs no served graph takes.

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
