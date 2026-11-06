# Reproducing the InferenceX AgentX results

> **Work in progress.** This page tracks the port of multi-token prediction (MTP) speculative decoding into TrnRT, with
> the serving configurations behind our InferenceX AgentX measurements of GLM-5.3-Flash on one trn2.48xlarge. The code and
> the commands below are in place; the on-device checks are listed under [Status](#status) and the measured results are not
> published here yet.

## What is added

- **MTP speculative decoding** with GLM-5.3-Flash's own MTP layer: 2 draft tokens per step, verified by the target model
  in the next step. The accepted count, the newest token, the positions, the state row and the drafts stay on the device,
  so the next step is scheduled before the previous one is read back (`trnrt/engine/spec_async.py`).
- **MTP only while few users decode.** MTP runs while at most 4 requests decode in a step and plain decode runs above that,
  because past that point a verify step costs more per token than it saves. Plain steps still write the MTP layer's KV
  cache, so a return to MTP finds it complete.
- **Larger decode batches** for higher concurrency (12 and 24 requests per engine), with a state checkpoint row per
  decoding seat. Only the seats that speculate hold the 3 state rows a verify writes.
- **Host DRAM prompt cache sizes** of 8 and 12 GB per NeuronCore for concurrency 12 to 32.
- **A benchmark mode** that reproduces InferenceX's synthetic acceptance. It is off by default;
  see [Benchmark mode](#benchmark-mode).

## Configurations

One engine is 8 trn2 chips (tensor parallel 32 at logical NeuronCore 2); `trnrt serve` runs two behind the router.
Each row is a preset ([CONFIGURATION.md](CONFIGURATION.md#speculative-decoding-the-mtp--presets)) and the
`trnrt serve` arguments that select it:

| Concurrency | Arguments | Speculation | Decode batch |
|---|---|---|---|
| 1, 4, 8 | `--preset mtp-batch8` | while at most 4 users decode | up to 8 |
| 2 | `--preset mtp-batch2` | always | up to 8 |
| 12, 16 | `--preset mtp-batch12` | while at most 4 users decode | up to 12, 8 GB host cache per core |
| 24 | `--preset mtp-batch12 --host-cache-gb 12` | while at most 4 users decode | up to 12, 12 GB host cache per core |
| 32 | `--preset mtp-batch24` | while at most 4 users decode | up to 24, 12 GB host cache per core |

The host cache size is host only: it changes no graph, so `mtp-batch12` serves concurrency 12 to 24 on one compile cache.
Every other row has its own graphs and its own capture.

## Build the compile cache

Follow [DEPLOY.md](DEPLOY.md) for the image, the model and the shape directory; steps 3 and 4 take the preset. For each
preset you serve, capture, compile and strip with the arguments you will serve with:

```bash
PRESET=mtp-batch8      # or mtp-batch2, mtp-batch12, mtp-batch24
trnrt compile capture --out $OUT -- --model $OUT/shapes --preset $PRESET
trnrt compile compile --out $OUT --work-root $OUT/work
trnrt compile strip --out $OUT
```

A speculative preset adds graphs to the plain engine's: the verify call, the MTP passes, and a handful of small graphs
around them that read and write the board. The cache for one preset is therefore not the batch-12 cache plus a few
entries; capture and compile each preset you serve.

## Serve

```bash
docker run ... trnrt serve --model /model --port 8000 --preset mtp-batch8   # the mounts and variables of DEPLOY.md, step 4
```

`GET /v1/stats` carries a `spec` block (drafts proposed and accepted, the mean acceptance length, the per-position counts,
and in a dynamic preset the mode flips), and `GET /metrics` the same counters as `trnrt:spec_decode_*`, under vLLM's
metric names.

## Benchmark mode

InferenceX's AgentX scoring fixes the acceptance of speculative decoding at a golden length instead of measuring the
draft head's. `--spec-synth-al 2.518506` (or the environment variable `TRNRT_SPEC_SYNTH_AL`) reproduces that: the accepted
draft count of every verify step is drawn from a minimum-variance schedule whose mean acceptance length (one plus the
accepted drafts a step) is the given value, after the design of vLLM's synthetic rejection sampler. 2.518506 is the golden
length for 2 draft tokens.

Drafting and verification still run for real, and the step costs what it costs; only which positions count as accepted is
drawn. An accepted position takes its draft whether or not the target agrees, so the **text of a benchmark-mode run is not
the model's**: use it only to compare serving throughput with InferenceX's published runs, never to judge output quality.
The mode adds one input to one small graph (the draw's flags), so it needs its own capture: run `compile capture` with the
same `--spec-synth-al`.

## Status

Checked on the CPU suite: the scheduler, the board graphs, the engine's speculative step against a fake device that checks
pages, state rows and token chains, dynamic speculation and its seat moves, the presets' values and graph families.

Not yet checked here, and listed in the pull request: that each preset's graphs are the graphs the measured configurations
ran (`tools/graph_equiv.py`), compile from source, a served run, GSM8K accuracy with MTP, and an AgentX point.

## Workload

InferenceX's canonical AgentX workload (agentx-harness 1.0.12), one hour per point, scored with InferenceX's definitions.
