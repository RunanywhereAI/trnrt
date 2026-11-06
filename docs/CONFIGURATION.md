# Configuration

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
| `--revision` | the repo's default branch | with a Hugging Face repo id as `--model`, the commit (or branch, tag) to download; refused with a local directory; host only |
| `--moe-kernel` | nki on Neuron, xla with `--device cpu` | the routed experts: the NKI kernels (the served path) or XLA's gather |
| `--kda-shell-max-rows` | 12 | KDA layers' decode attention as one fused kernel up to this many rows |
| `--dsa-layer-max-rows` | 16 | DSA layers' decode attention and FFN as one fused kernel up to this many rows |
| `--ffn-half-below` | 17 | decode FFN blocks as one fused kernel below this many rows |
| `--dsa-index-rows` | 8 | the DSA indexer's decode scores in kernel calls of at most this many rows (0: one call) |
| `--dense-half-skip-rows` | 2 | the decode row counts at which a dense MLP layer runs as the XLA graph instead of the dense half's kernel |
| `--sp-gather`, `--sp-gather-tp-experts`, `--dsa-sel2g` | xla, off, off | the sequence-parallel all-gather of a decode block's output as the NKI kernel, over the experts' tensor-parallel group, and the DSA layer's two-stage selection in one kernel (the page ladder with a 3072-page rung needs it) |

The seats, the bucket ladders, the pool sizes, the page size and the decode kernel routes are graph shapes: changing
any of them needs a new capture. `--preset batch12` names this configuration: it is accepted, and changes nothing.
`python -m trnrt engine` runs one engine without the router.

## Batch 8

`--preset batch8`: decode batches of up to 8 requests per engine, the configuration of the batch-8 numbers and the one
the batch-4 numbers ran on. The preset sets these engine arguments; everything else stays at the defaults above:

| Argument | Default (batch 12) | `--preset batch8` | |
|---|---|---|---|
| `--max-num-seqs` | 16 | 8 | request seats per engine (the state pool's rows) |
| `--decode-buckets` | 1,2,4,6,8,12 | 1,2,4,6,8 | compiled decode batch sizes |
| `--max-running` | 12 | unset: all 8 seats run | requests running at once per engine |
| `--kda-shell-max-rows` | 12 | 8 | KDA layers' decode attention as one fused kernel up to this many rows |
| `--dsa-layer-max-rows` | 16 | 8 | DSA layers' decode attention and FFN as one fused kernel up to this many rows |
| `--ffn-half-below` | 17 | 16 | decode FFN blocks as one fused kernel below this many rows |
| `--dsa-index-rows` | 8 | 0 | the DSA indexer's decode scores in calls of at most this many rows (0: one call) |

The preset has its own graphs: 8 seats change every graph that carries the linear-attention state, prefill graphs
included. Capture and serve with the same preset ([DEPLOY.md](DEPLOY.md), step 3's shape directory and `trnrt` function; its entries sit beside
the default configuration's in the cache, under their own keys, and `$OUT/capture` then holds this capture's records):

```bash
trnrt compile capture --out $OUT -- --model $OUT/shapes --preset batch8
trnrt compile compile --out $OUT --work-root $OUT/work
trnrt compile strip --out $OUT
docker run ... trnrt serve --model /model --port 8000 --preset batch8   # the mounts and variables of DEPLOY.md, step 4
```

An argument given explicitly overrides the preset's value. `--max-running` is host only; any other override changes
the graphs and needs its own capture.

## Batch 4: `--max-running 4`

`--max-running N` admits at most N requests at once per engine (never more than `--max-num-seqs`); the rest wait in
arrival order. It changes no graph, so it runs on the compile cache of the configuration it is added to. Batch 4 is
`--preset batch8` with `--max-running 4`, on batch 8's compile cache:

```bash
docker run ... trnrt serve --model /model --port 8000 --preset batch8 --max-running 4   # the mounts and variables of DEPLOY.md, step 4
```

## Speculative decoding: the `mtp-*` presets

The model's own multi-token-prediction (MTP) layer drafts 2 tokens a step and the target model verifies them in the next
step, with the accepted count, the newest token, the positions and the drafts kept on the device, so a step is scheduled
before the previous one is read back. [AGENTX.md](AGENTX.md) has the commands. Four presets name the configurations
that were measured:

| Preset | Decode batch | Speculation | Page buckets (decode; verify and MTP) | Host cache |
|---|---|---|---|---|
| `mtp-always` | up to 8 (buckets 1,2,4,6,8) | at every batch size | 1024,4096 | 4 GB |
| `mtp-batch8` | up to 8 (1,2,4,6,8) | while at most 4 requests decode | 1024,3072,4096; 1024,4096 | 4 GB |
| `mtp-batch12` | up to 12 (1,2,4,8,12) | while at most 4 decode | 1024,3072,4096; 1024,4096 | 8 GB |
| `mtp-batch24` | up to 24, 32 seats (1,2,4,16,24) | while at most 4 decode | 1024,3072,4096; 1024,4096 | 12 GB |

Above its limit a dynamic preset steps every request with the plain decode graph (the idea of vLLM's
`num_speculative_tokens_per_batch_size`: no drafts above a batch size, here with hysteresis), and its plain steps still write the MTP layer's KV, so a return to
speculation finds it complete. `--host-cache-gb` is host only: `mtp-batch12 --host-cache-gb 12` is the same graphs with a
larger prefix-cache tier. The presets' speculation arguments:

| Argument | Preset value | |
|---|---|---|
| `--spec-method`, `--spec-k` | mtp, 2 | speculative decoding with the model's MTP layer, drafts per step |
| `--spec-max-batch`, `--spec-hysteresis` | 4, 1 (`mtp-always`: 0) | speculate while at most this many requests decode; once plain, speculate again at this many fewer |
| `--spec-bucket-max`, `--plain-bucket-min` | 4, 4 (`mtp-batch24`: 4, 16) | the verify family's graphs only at decode buckets up to this, the plain decode graphs only from this |
| `--spec-page-buckets` | 1024,4096 | the page buckets of every graph the MTP layer or a verify runs in |
| `--mtp-chunk-q`, `--mtp-fill-rows` | 64, 1024 | the MTP pass over a prefill chunk takes the prefill form from 64 rows, in calls of at most 1024 |
| `--mtp-kv-only`, `--dyn-mtp-kv` | on, on (`mtp-always`: off, off) | KV-only windows for the MTP pass over a chunk, and the MTP layer's KV written by plain decode steps |
| `--state-seats`, `--state-spec-seats` | 8, unset (`mtp-batch12`: 12, 4; `mtp-batch24`: 24, 4) | recurrent-state seats, and how many of them hold the 3 rows a verify writes |
| `--spec-synth-al` | unset | the benchmark mode of [AGENTX.md](AGENTX.md#benchmark-mode) |

Like every row of the first table, these are graph shapes: an argument other than a preset's changes the graphs and
needs its own capture. A speculative engine serves temperature and top-p sampling as the plain engine does (verification
is the rejection sampler's, so the output distribution is unchanged), and refuses the requests the plain engine refuses.

## Supported scope

The supported configurations are the default (batch 12), `--preset batch8`, the four `mtp-*` presets, and any of them
with a lower `--max-running` (batch 4: `--preset batch8 --max-running 4`). Larger decode batches, and values of
`--kda-shell-max-rows`, `--dsa-layer-max-rows`, `--ffn-half-below`, `--dsa-index-rows` and the speculation arguments
other than those of these configurations, are not supported.

## Environment variables

Set them on the server's container (`docker run -e NAME=value ...`); its engines inherit them.

| Variable | Default | |
|---|---|---|
| `TRNRT_TP_SHM` | on | each engine's rank 0 hands every graph call to its other ranks through a shared-memory segment; `0` sends them over gloo (TCP) instead, as happens anyway, with a warning, when /dev/shm is unusable |
| `TRNRT_TP_SHM_BYTES` | 67108864 (64 MiB) | that segment's size, one segment per engine in /dev/shm; a larger call goes over gloo |
| `TRNRT_SPEC_SYNTH_AL` | unset | the benchmark mode's mean acceptance length when `--spec-synth-al` is not given ([AGENTX.md](AGENTX.md#benchmark-mode)); needs a speculative preset |
| `TRNRT_SKIP_DEVICE_CHECK` | unset | `1`: start without checking that the engine's Neuron devices are free |
| `TRNRT_SKIP_HOST_MEM_CHECK` | unset | `1`: start without checking the host's free contiguous memory |
| `TRNRT_LOG_LEVEL` | `INFO` | the level of the server's, engines' and router's log messages, on stderr (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `TRNRT_ROUTER_LOG_SESSIONS` | unset | `1`, for debugging: the router's log names session ids: a line per new pin and per side request, and the id in its failover and return-home lines; unset, it logs no session id |
| `NEURON_LIBTORCH_CACHE_ROOT` | `~/.cache/neuron_libtorch` | the compile cache's root ([DEPLOY.md](DEPLOY.md), step 4) |
