# trnRT for GLM-5.3-Flash

Serve GLM-5.3-Flash on one AWS trn2.48xlarge: two engines of 32 tensor-parallel ranks each behind a session-affine
router, an OpenAI-compatible API with streaming and GLM tool calls, 1M-token context, and a prefix cache that keeps
evicted KV pages and linear-attention state checkpoints in host memory. Decode runs on NKI megakernels, about one
kernel per layer half.

## Status

This repository was extracted from the engine the published numbers were measured with, and a graph-equivalence check
(`tools/graph_equiv.py`) holds it to compiling the same graphs. The numbers below were measured with that engine, in
the configurations this repository reproduces: the default (batch 12) and the batch-8 and batch-4 alternatives
([Configuration](#configuration)). Verification of this repository on a device is in progress, as is a full run of the
graph-equivalence check over all 32 ranks.

More: [how the decode kernels work](docs/MEGAKERNELS.md), [all results and methodology](docs/RESULTS.md),
[known issues](docs/KNOWN_ISSUES.md).

## Performance

SemiAnalysis's AgentX benchmark replays the request patterns of real Claude Code sessions. At a 1M-token context and 32
concurrent agents, one hour per point, one trn2.48xlarge serves:

| Decode batch per engine | Requests / min | Tokens / min (box) | Decode tokens/s per user (p50 / p90) | First token, cached prompt (p50 / p90) | $ per M tokens at $35.76/h / $8.60/h |
|---|---|---|---|---|---|
| 4 | 19.3 | 2.36M | 46.6 / 34.7 | 76.73 s / 141.47 s | $0.253 / $0.061 |
| 8 | 25.2 | 3.02M | 29.4 / 20.0 | 28.82 s / 77.29 s | $0.197 / $0.047 |
| 12 (the default) | 28.9 | 3.57M | 22.3 / 14.5 | 6.53 s / 37.35 s | $0.167 / $0.040 |

Tokens count input, cached input included, and output. Prices: AWS's public capacity-block price for trn2.48xlarge,
$35.76 per hour, and its public spot price in us-east-2 on 3 Nov 2026, $8.60 per hour (spot prices vary by region and
hour). A single 1,039,912-token prompt reaches its first token in 190.8 s on one engine, and a follow-up over that
cached context answers in 1.84 s. One user alone decodes at 73.7 tokens/s. [docs/RESULTS.md](docs/RESULTS.md) has the
full tables, including first-token times for uncached prompts and lower loads.

## What you need

| | |
|---|---|
| Serving host | one trn2.48xlarge: 16 Trainium2 chips with 96 GiB of HBM each (64 logical NeuronCores at LNC 2), 192 vCPUs, 2 TB of memory (AWS's published specification) |
| Compile host | any x86_64 host with Docker; no NeuronCore needed. `compile` runs as many neuronx-cc processes at once as host memory admits, 30 GB of available memory each by default (`--mem-per-compile-gb`), so its memory and cores set the compile time |
| Software | Docker, and the image built from `docker/`: Neuron SDK 2.32 (neuronx-cc 2.27), Python 3.13, its base image pinned by digest. The base image is about an 8 GB download, and the built image takes about 27 GB of disk |
| Model | `zai-org/GLM-5.3-Flash` at revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`, FP8 weights as released, about 330 GB: the same revision for compiling and serving |
| Disk | the image on both hosts; the compile output under `$OUT` on the compile host; the checkpoint and the compile cache on the serving host |
| Time | building the compile cache takes hours. Start-up loads the weights and warms every compiled graph, one engine at a time, and each engine has `--startup-timeout` (default 3600 s) to come up |
| Cost | one trn2.48xlarge for as long as it serves; [docs/RESULTS.md](docs/RESULTS.md) gives the cost per million tokens and per request at the two prices above |

## Quickstart

### 1. Get the code

```bash
git clone https://github.com/RunanywhereAI/trnrt-glm.git
cd trnrt-glm
```

Run the commands below from the repository's root.

### 2. Build the image

```bash
docker build -f docker/Dockerfile \
  --build-arg TRNRT_VERSION=$(python3 -c 'import trnrt; print(trnrt.__version__)') \
  --build-arg TRNRT_REVISION=$(git rev-parse HEAD) -t trnrt .
```

The build checks every Neuron piece trnRT imports and its version, and that the image's transformers has `glm5_next`
(`docker/check_image.py`). Build the image on the compile host and on the serving host.

### 3. Build the compile cache (CPU host)

A graph depends on the weights' shapes and dtypes, never their values. So the cache is built from the checkpoint's
config and safetensors headers (a few MB, read without downloading the weights) on a host without NeuronCores:

```bash
OUT=/data/trnrt                               # everything goes under one directory
REV=eb9eb208eb0d988989d07a6a12d0fdeb5f52574a  # the checkpoint revision (step 4)
mkdir -p $OUT/nki $OUT/work                   # the NKI kernel binaries, shared by every step; neuronx-cc's scratch
# each step: `python -m trnrt` in a new container of the image, with $OUT at the same path and $OUT/nki mounted
trnrt() { docker run --rm -e HF_TOKEN -v $OUT:$OUT -v $OUT/nki:/var/tmp/nki-intermediate-cache trnrt "$@"; }
trnrt compile shapes zai-org/GLM-5.3-Flash --revision $REV --out $OUT
trnrt compile decisions --model zai-org/GLM-5.3-Flash --revision $REV --out $OUT   # a few range reads per MoE layer
trnrt compile capture --out $OUT -- --model $OUT/shapes                            # every graph, all 32 ranks
trnrt compile compile --out $OUT --work-root $OUT/work                              # neuronx-cc on each graph
trnrt compile strip --out $OUT                                                      # drop the NEFFs' debug info
```

Each step is its own `docker run --rm`, so whatever a step leaves inside its container is gone by the next. A graph
that calls an NKI kernel names the kernel's compiled binary by absolute path under `/var/tmp/nki-intermediate-cache`.
The capture writes those binaries, and neuronx-cc adds its own kernels there while compiling. So `capture` and
`compile` must see the same directory at that path: the host's `$OUT/nki`, mounted into every step above. Without it,
or on another host without a copy of it, `compile` fails every such graph ("NKI kernel binaries missing"). To split the
steps across hosts, copy `$OUT/nki` with the cache. A serving engine traces its graphs again and writes to that
directory too, so `serve` below mounts the compile host's copy.

- Hugging Face requests carry no token unless `HF_TOKEN` is set on the host (`-e HF_TOKEN` passes it in). Without one,
  the Hub rate-limits requests by IP.
- `compile` runs as many neuronx-cc processes at once as host memory admits. One can take tens of GB, so each reserves
  `--mem-per-compile-gb` (default 30) of MemAvailable, and one the OOM killer takes runs once more, alone.
  `--work-root $OUT/work` keeps neuronx-cc's scratch trees on the host: a failed compile's tree stays there with its
  logs, where the container's own temporary directory would be gone when the step ends.
- `strip` is not optional at 1M-token buckets: the runtime holds each loaded NEFF in host memory several times over,
  and two engines' unstripped NEFFs do not fit.
- The containers run as root, so everything under `$OUT` belongs to root; `sudo chown -R "$(id -u):$(id -g)" $OUT`
  hands it back once the steps are done.

The arguments after `capture --` must be the engine arguments the server runs with, spelled out in full. Capture refuses
any other; the server's own, such as `--engines` and `--port`, do not change the graphs. The defaults are the default
configuration (batch 12), so `--model` alone is enough. For batch 8 ([Batch 8](#batch-8)), pass its arguments to both
`capture` and `serve`.

### 4. Download the checkpoint and serve (trn2.48xlarge)

Copy `$OUT/cache` and `$OUT/nki` to the serving host, and download the checkpoint at the revision the cache was built
from (the `hf` command comes with `pip install -U huggingface_hub`):

```bash
hf download zai-org/GLM-5.3-Flash --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a --local-dir /models/GLM-5.3-Flash
```

That is the revision the published numbers were measured on; later revisions change the chat template. Give `serve`
the downloaded directory: `--model` also takes a Hugging Face repo id, but then downloads the repo's default branch,
not a pinned revision. Then start the server:

```bash
docker run --rm --privileged --network host --ipc=host \
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
- Start-up needs free contiguous host memory, which a full page cache can withhold: drop the page cache before starting
  (`sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`).
- `--ipc=host` gives the engines' processes the host's shared memory instead of Docker's 64 MB default (or pass
  `--shm-size=1g`).
- The router's `/health` answers 200 while both engines are up, 200 with `"status": "degraded"` while one is (its
  requests and sessions go to the other), and 503 while neither is. The body names the engines that are down.
- SIGTERM (`docker stop`) stops the router, then each engine, and `serve` exits with status 143.

### 5. Use it

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
  127.0.0.1 only.
- The router listens on every interface (`--host 0.0.0.0`) and has no authentication: put it behind your own proxy, or
  pass `--host 127.0.0.1`.
- Not supported, answered with 400: repetition, frequency and presence penalties, guided decoding (`response_format`
  other than text, `tool_choice` `"required"` or a named tool), prompt logprobs and thinking budgets. Every graph
  samples on the device, and these need host inputs no served graph takes.
- Status codes: 400 for a request that cannot be served as sent (a field of the wrong type or out of range, a prompt
  longer than the context, a chat template variable the template never reads); 500 for an error inside the engine;
  503 once the engine has failed, with the reason on `/health`.

## Configuration

Every `serve` default is the batch-12 configuration, the one the headline numbers use: decode batches of up to 12
requests per engine.

| Argument | Default | |
|---|---|---|
| `--engines`, `--tp` | 2, 32 | engines per host, tensor-parallel ranks per engine |
| `--host`, `--port` | 0.0.0.0, 8000 | the router's address; engine i listens on 127.0.0.1, at `--port` + 1 + i |
| `--max-model-len` | 1048576 | context per request |
| `--max-num-seqs`, `--decode-buckets` | 16; 1,2,4,6,8,12 | request seats per engine (the state pool's rows), compiled decode batch sizes |
| `--max-running` | 12 | requests running at once per engine; host only |
| `--max-prefill-tokens`, `--prefill-buckets` | 4096; 512,1024,2048,4096 | prefill chunk per step, compiled chunk sizes |
| `--page-size`, `--page-buckets`, `--prefill-page-buckets` | 256; 512,1024,4096; 1024,4096 | KV page, compiled context lengths in pages (decode; prefill) |
| `--prefill-page-min-rows` | 2048 | prefill calls of at least this many rows take every prefill page bucket, smaller ones the largest |
| `--kv-cache-gb`, `--kv-cache-dtype` | 4.0, fp8 | KV pool per rank |
| `--state-checkpoints` | 16 | linear-attention state rows kept for the prefix cache |
| `--host-cache-gb` | 4.0 | host memory per rank under the prefix cache (half for state checkpoints) |
| `--kda-shell-max-rows` | 12 | KDA layers' decode attention as one fused kernel up to this many rows |
| `--dsa-layer-max-rows` | 16 | DSA layers' decode attention and FFN as one fused kernel up to this many rows |
| `--ffn-half-below` | 17 | decode FFN blocks as one fused kernel below this many rows |
| `--dsa-index-rows` | 8 | the DSA indexer's decode scores in kernel calls of at most this many rows (0: one call) |

The seats, the bucket ladders, the pool sizes, the page size and the decode kernel routes are graph shapes: changing
any of them needs a new capture. `--preset batch12` names this configuration explicitly and sets the same values.
`python -m trnrt engine` runs one engine without the router.

### Batch 8

Decode batches of up to 8 requests per engine: the configuration of the batch-8 numbers, and the one the batch-4
numbers ran on. It differs from the default in these engine arguments; everything else stays as above:

| Argument | Default (batch 12) | Batch 8 | |
|---|---|---|---|
| `--max-num-seqs` | 16 | 8 | request seats per engine (the state pool's rows) |
| `--decode-buckets` | 1,2,4,6,8,12 | 1,2,4,6,8 | compiled decode batch sizes |
| `--max-running` | 12 | 8 | requests running at once per engine |
| `--kda-shell-max-rows` | 12 | 8 | KDA layers' decode attention as one fused kernel up to this many rows |
| `--dsa-layer-max-rows` | 16 | 8 | DSA layers' decode attention and FFN as one fused kernel up to this many rows |
| `--ffn-half-below` | 17 | 16 | decode FFN blocks as one fused kernel below this many rows |
| `--dsa-index-rows` | 8 | 0 | the DSA indexer's decode scores in calls of at most this many rows (0: one call) |

Batch 8 has its own graphs: 8 seats change every graph that carries the linear-attention state, prefill graphs
included. Capture and serve it with the same arguments (step 3's shape directory and `trnrt` function; its entries sit
beside the default configuration's in the cache, under their own keys, and `$OUT/capture` then holds this capture's
records):

```bash
BATCH8=(--max-num-seqs 8 --decode-buckets 1,2,4,6,8 --max-running 8 --kda-shell-max-rows 8 --dsa-layer-max-rows 8
        --ffn-half-below 16 --dsa-index-rows 0)
trnrt compile capture --out $OUT -- --model $OUT/shapes "${BATCH8[@]}"
trnrt compile compile --out $OUT --work-root $OUT/work
trnrt compile strip --out $OUT
docker run ... trnrt serve --model /model --port 8000 "${BATCH8[@]}"   # the mounts and variables of step 4
```

An argument given explicitly wins over a default; the last of a repeated argument wins.

### Batch 4: `--max-running 4`

`--max-running N` admits at most N requests at once per engine (never more than `--max-num-seqs`); the rest wait in
arrival order. It changes no graph, so it runs on the compile cache of the configuration it is added to. Batch 4 is the
batch-8 configuration with `--max-running 4`, on batch 8's compile cache:

```bash
docker run ... trnrt serve --model /model --port 8000 "${BATCH8[@]}" --max-running 4   # the mounts and variables of step 4
```

### Supported scope

Supported scope is these configurations: the default (batch 12), batch 8, and either of them with a lower
`--max-running` (batch 4: batch 8 with `--max-running 4`). Larger decode batches, and values of `--kda-shell-max-rows`,
`--dsa-layer-max-rows`, `--ffn-half-below` and `--dsa-index-rows` other than those of these two configurations, are not
supported.

## Tests

The tests run on CPU, without a NeuronCore. They are run on x86_64 Linux with Python 3.13, torch 2.11, transformers
5.17.0 and tokenizers 0.23.3, the image's versions. In the image:

```bash
docker run --rm --entrypoint python trnrt -m pytest
```

Or in a Python 3.13 environment, where the package's dependencies bring torch:

```bash
pip install -e '.[test]'
pytest
```

Engine tests use a tiny random model of GLM-5.3-Flash's architecture and need a transformers release with `glm5_next`
(5.17 or later). Some tests run only when an environment variable provides what they need:

- `TRNRT_TEST_MODEL`: a GLM-5.3-Flash checkpoint directory, for the API tests over the real model;
- `TRNRT_TEST_GLM_CKPT`: a local GLM-5.3-Flash checkpoint directory, for the KDA shell's test on the checkpoint's own
  hyper-connection weights;
- `TRNRT_TEST_NETWORK=1`: lets tests fetch the model's small files (config, tokenizer, chat template) from the Hugging
  Face Hub; without it they skip unless those files are cached;
- `HLO_PB2_PATH`: a directory holding `hlo_pb2.py`, for the graph-equivalence tests where `libtorch_neuronx_lite` is not
  installed (they also need zstd).

## License

Apache License 2.0 (`LICENSE`). `NOTICE` and `THIRD_PARTY_NOTICES.md` list the code and data this repository derives
from.
