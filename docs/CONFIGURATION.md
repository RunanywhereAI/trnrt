# Configuration

TrnRT serves GLM-5.3-Flash in one of three builds. Pick the build from the concurrent users you expect per engine, or
let `--preset auto --users N` pick it:

| Build | Users per engine | Decode | Running (seats) | Page buckets | KV per rank | Host cache per rank |
|---|---|---|---|---|---|---|
| `--preset latency` | 1-8 | MTP speculative decoding at every step: buckets 1,2,4,6,8, each a verify of 3 rows a request | 8 (16) | 1024,2048,4096 | 4 GiB | 4 GiB, one copy per rank |
| `--preset balanced` (the default) | 9-16 | plain decode, buckets 8,16,24 | 24 (32) | 1024,3072,4096 | 4 GiB | 16 GiB, one copy per context-parallel index |
| `--preset throughput` | 17 and up | plain decode, buckets 16,24 | 24 (32) | 1024,3072,4096 | 5.25 GiB | 16 GiB, one copy per context-parallel index |

`--users N` counts users per engine, not in total. With `E` engines behind the router (`serve --engines`, 2 by default),
`N` concurrent users are about `ceil(N / E)` per engine: for 16 users on two engines pass `--users 8`, which runs the
latency build (`--users 16` would run balanced). `--users N` alone means `--preset auto --users N`.

Every build has a 1M-token context, 256-token FP8 KV pages, 4,096-token prefill chunks (buckets 512,1024,2048,4096;
prefill page buckets 1024,4096 from 2,048 rows), and the fused decode kernels up to 24 rows. Each build also fixes the
kernel forms of its DSA and KDA layers, which have no command-line argument (`BUILD_FIELDS` in `trnrt/__main__.py`).

A build's graphs are its own. Capture and serve with the same build ([DEPLOY.md](DEPLOY.md), step 3's shape directory
and `trnrt` function; each build's entries sit beside the others' in the cache, under their own keys):

```bash
trnrt compile capture --out $OUT -- --model $OUT/shapes --preset throughput
trnrt compile compile --out $OUT --work-root $OUT/work
trnrt compile strip --out $OUT
docker run ... trnrt serve --model /model --port 8000 --preset throughput   # the mounts and variables of DEPLOY.md, step 4
```

`--preset auto --users N` captures and serves the build its band picks, so give capture the same `--users`.

## Arguments

The defaults are the balanced build's. A build sets the arguments below it differs in; an argument given explicitly
still wins over the build.

| Argument | Default (balanced) | latency | throughput | |
|---|---|---|---|---|
| `--engines`, `--tp` | 2, 32 | | | engines per host, tensor-parallel ranks per engine |
| `--host`, `--port` | 0.0.0.0, 8000 | | | the router's address; engine i listens on 127.0.0.1, at `--port` + 1 + i |
| `--max-model-len` | 1048576 | | | context per request |
| `--max-num-seqs` | 32 | 16 | | request seats per engine (the token board's slots) |
| `--state-seats` | 24 | 8 | | seats of the linear-attention state pool |
| `--max-running` | 24 | 8 | | requests running at once per engine; host only |
| `--decode-buckets` | 8,16,24 | 1,2,4,6,8 | 16,24 | compiled decode batch sizes |
| `--max-prefill-tokens`, `--prefill-buckets` | 4096; 512,1024,2048,4096 | | | prefill chunk per step, compiled chunk sizes |
| `--page-size`, `--page-buckets` | 256; 1024,3072,4096 | 1024,2048,4096 | | KV page, compiled decode context lengths in pages |
| `--prefill-page-buckets`, `--prefill-page-min-rows` | 1024,4096; 2048 | | | prefill calls of at least this many rows take every prefill page bucket, smaller ones the largest |
| `--kv-cache-gb`, `--kv-cache-dtype` | 4.0, fp8 | | 5.25 | KV pool per rank |
| `--state-checkpoints` | 24 | 8 | | linear-attention state rows kept for the prefix cache |
| `--kda-shell-max-rows` | 12 | 24 | | KDA layers' decode attention as one fused kernel up to this many rows |
| `--dsa-layer-max-rows`, `--ffn-half-below` | 24, 25 | | | DSA layers' decode attention and FFN as one fused kernel up to this many rows; decode FFN blocks as one fused kernel below this many |
| `--dsa-index-rows` | 8 | | | the DSA indexer's decode scores in kernel calls of at most this many rows (0: one call) |
| `--dense-half-skip-rows` | 2,18,24 | | | the decode row counts at which a dense MLP layer runs as the XLA graph instead of the dense half's kernel |
| `--sp-gather`, `--sp-gather-tp-experts`, `--dsa-sel2g` | nki, on, on | | | the sequence-parallel all-gather of a decode block's output as the NKI kernel over the experts' tensor-parallel group, and the DSA layer's two-stage selection in one kernel (the 3072-page rung needs it) |
| `--revision` | the repo's default branch | | | with a Hugging Face repo id as `--model`, the commit (or branch, tag) to download; refused with a local directory; host only |
| `--moe-kernel` | nki on Neuron, xla with `--device cpu` | | | the routed experts: the NKI kernels (the served path) or XLA's gather |

The seats, the bucket ladders, the pool sizes, the page size and the kernel routes are graph shapes: changing any of
them needs a new capture. `python -m trnrt engine` runs one engine without the router.

## The host cache

Host memory under the prefix cache holds the KV pages and linear-attention state checkpoints the device evicts, and a
later request whose prefix continues into them gets them copied back instead of prefilled again. Its arguments are host
only: no graph changes, so they run on the compile cache of the build they are added to.

| Argument | Default (balanced, throughput) | latency | |
|---|---|---|---|
| `--host-cache-gb` | 16.0 | 4.0 | host memory per rank (0: off) |
| `--host-cache-ckpt-frac` | 0.6 | 0.5 | the share of it for state checkpoints, the rest for KV pages |
| `--host-cache-batch` | on | off | a restore's and an eviction's pages in one call each, every cache copied once per run of consecutive pages, instead of a call and a copy per page |
| `--host-cache-save-once` | on | off | under context-parallel attention the ranks of one context-parallel index hold the same bytes of every page: keep one shared copy (in /dev/shm) instead of one per rank, so the same memory holds as many times the pages as there are row groups (4 for GLM-5.3-Flash on 32 ranks). Needs `--host-cache-batch` |
| `--pool-ascending` | on | off | hand out the lowest free KV pages in ascending order, so a request's pages form few runs for the batched copies |

The save-once copies live in shared memory: the container needs the host's /dev/shm (`--ipc=host`, as in
[DEPLOY.md](DEPLOY.md)), and an engine refuses to start when /dev/shm cannot hold its share at its fullest.

## Speculative decoding: the latency build

The model's own multi-token-prediction (MTP) layer drafts 2 tokens a step and the target model verifies them in the next
step, with the accepted count, the newest token, the positions and the drafts kept on the device, so a step is scheduled
before the previous one is read back. [AGENTX.md](AGENTX.md) has the commands. The latency build speculates at every
step: its running cap (8) is its speculation limit, so it has no plain decode graph at all. Its speculation arguments:

| Argument | latency | |
|---|---|---|
| `--spec-method`, `--spec-k` | mtp, 2 | speculative decoding with the model's MTP layer, drafts per step |
| `--spec-max-batch`, `--spec-hysteresis` | 8, 1 | speculate while at most this many requests decode; once plain, speculate again at this many fewer |
| `--spec-bucket-max`, `--plain-bucket-min` | 8, 0 | the verify family's graphs only at decode buckets up to this, the plain decode graphs only from this |
| `--no-plain-decode` | set | no plain decode graph: allowed only when `--spec-max-batch` covers the largest decode bucket and the running cap |
| `--spec-page-buckets` | 1024,2048,4096 | the page buckets of every graph the MTP layer or a verify runs in |
| `--mtp-chunk-q`, `--mtp-fill-rows` | 64, 1024 | the MTP pass over a prefill chunk takes the prefill form from 64 rows, in calls of at most 1024 |
| `--mtp-kv-only`, `--dyn-mtp-kv` | on, on | KV-only windows for the MTP pass over a chunk, and the MTP layer's KV written by plain decode steps |
| `--state-spec-seats` | unset | of the state seats, how many hold the 3 rows a verify writes (unset: all) |
| `--spec-synth-al` | unset | the benchmark mode of [AGENTX.md](AGENTX.md#benchmark-mode) |

These are graph shapes, except `--spec-hysteresis`, which is host only. A speculative engine serves temperature and top-p
sampling as the plain engine does (verification is the rejection sampler's, so the output distribution is unchanged),
and refuses the requests the plain engine refuses.

## Earlier preset names

The presets of earlier releases are still accepted: each runs the build nearest its decode batch, and logs a
deprecation warning. Their graphs are the build's, so a cache captured under an old name is stale: capture again.

| Earlier name | Runs |
|---|---|
| `batch8`, `mtp-batch8`, `mtp-always` | `latency` |
| `batch12`, `mtp-batch12` | `balanced` |
| `mtp-batch24` | `throughput` |

### Batch 8

`--preset batch8` is a deprecated alias for the latency build: it serves up to 8 requests per engine with MTP speculation
at every step, where release 0.1.0's `batch8` decoded them without speculation. Batch 4 is
`--preset latency --max-running 4` ([Fewer requests](#fewer-requests---max-running)).

### Speculative decoding: the `mtp-*` presets

`mtp-batch8` and `mtp-always` now run the latency build, `mtp-batch12` the balanced build and `mtp-batch24` the
throughput build. Only the latency build speculates ([Speculative decoding](#speculative-decoding-the-latency-build)):
from 9 users per engine the builds decode without drafts.

## Fewer requests: `--max-running`

`--max-running N` admits at most N requests at once per engine (never more than the state seats); the rest wait in
arrival order. It changes no graph, so it runs on the compile cache of the build it is added to:

```bash
docker run ... trnrt serve --model /model --port 8000 --preset latency --max-running 4   # the mounts and variables of DEPLOY.md, step 4
```

## Supported scope

The supported configurations are the three builds (`--preset auto` picks one of them), each with a lower
`--max-running` and any host cache arguments. Other values of the graph-shape arguments are not supported.

## Environment variables

Set them on the server's container (`docker run -e NAME=value ...`); its engines inherit them.

| Variable | Default | |
|---|---|---|
| `TRNRT_TP_SHM` | on | each engine's rank 0 hands every graph call to its other ranks through a shared-memory segment; `0` sends them over gloo (TCP) instead, as happens anyway, with a warning, when /dev/shm is unusable |
| `TRNRT_TP_SHM_BYTES` | 67108864 (64 MiB) | that segment's size, one segment per engine in /dev/shm; a larger call goes over gloo |
| `TRNRT_SPEC_SYNTH_AL` | unset | the benchmark mode's mean acceptance length when `--spec-synth-al` is not given ([AGENTX.md](AGENTX.md#benchmark-mode)); needs the latency build |
| `TRNRT_DMA_HW` | `1` under every build | `1`: the kernels' static DMA helpers issue their DMAs in hardware DGE. The kernels read it when they are imported (some bind the helper by name, and every kernel folds its source into its key then), so it is an environment variable, not a flag. `trnrt engine`, `trnrt serve` and `trnrt compile capture` set it before they import the kernels, and a value already in the environment wins. Any other entry point has to set it itself. It changes the compiled graphs, so capture with the value you serve with |
| `TRNRT_NUMA_PIN_OPTIONAL` | unset | every rank runs on its NeuronCore's NUMA node, and an engine refuses to start when the device listing fails or its launch CPU set holds no CPU of some device's node; `1`: start anyway, the ranks it cannot pin running as launched, with a warning |
| `TRNRT_SKIP_DEVICE_CHECK` | unset | `1`: start without checking that the engine's Neuron devices are free |
| `TRNRT_SKIP_HOST_MEM_CHECK` | unset | `1`: start without checking the host's free contiguous memory |
| `TRNRT_LOG_LEVEL` | `INFO` | the level of the server's, engines' and router's log messages, on stderr (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `TRNRT_ROUTER_LOG_SESSIONS` | unset | `1`, for debugging: the router's log names session ids: a line per new pin and per side request, and the id in its failover and return-home lines; unset, it logs no session id |
| `NEURON_LIBTORCH_CACHE_ROOT` | `~/.cache/neuron_libtorch` | the compile cache's root ([DEPLOY.md](DEPLOY.md), step 4) |
