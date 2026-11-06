# Reproducing the InferenceX AgentX results

> **Work in progress.** This page tracks the port of multi-token prediction (MTP) speculative decoding into TrnRT, with
> the serving configurations behind our InferenceX AgentX measurements of GLM-5.3-Flash on one trn2.48xlarge. The serving
> side and the commands below are in place; the model's verify and draft passes land in the same pull request. The
> remaining checks are listed under [Status](#status), and the measured numbers in [RESULTS.md](RESULTS.md#mtp-speculative-decoding-one-engine)
> are marked unverified on this tree until those checks reproduce them.

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
| 2 | `--preset mtp-always` | always | up to 8 |
| 12, 16 | `--preset mtp-batch12` | while at most 4 users decode | up to 12, 8 GB host cache per core |
| 24, 32 | `--preset mtp-batch24` | while at most 4 users decode | up to 24, 12 GB host cache per core |

A preset names its decode batch; `mtp-always` is the one with speculation at every batch size. The host cache size is host
only: it changes no graph, so `mtp-batch12 --host-cache-gb 12` is a larger tier on the same compile cache, though none of
the measured points uses it. Every preset has its own graphs and its own capture.

## Build the compile cache

Follow [DEPLOY.md](DEPLOY.md) for the image, the model and the shape directory; steps 3 and 4 take the preset. For each
preset you serve, capture, compile and strip with the arguments you will serve with:

```bash
PRESET=mtp-batch8      # or mtp-always, mtp-batch12, mtp-batch24
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
pages, state rows and token chains (dynamic speculation, its seat moves, preemption, and a prefix restored from the host
tier then speculated), the runner on a stub model with the real board graphs, and the presets' values and graph families.

Not yet checked here, and listed in the pull request: that each preset's graphs are the graphs the measured configurations
ran (`tools/graph_equiv.py`), compile from source, a served run, GSM8K accuracy with MTP, and an AgentX point.

## Run and score a point

The load client is SemiAnalysis's AgentX harness, version 1.0.12 (built on AIPerf, Apache-2.0), on a separate CPU
machine: the harness is CPU-heavy and the requests a minute it reaches depend on its client's cores, so state the
client's core count next to any number (ours pinned the harness to 48 logical CPUs). Serve a preset from the table
above, then run one point per concurrency:

```bash
pip install agentx-harness==1.0.12
agentx profile \
  --url http://<server>:<port> \
  --endpoint /v1/chat/completions \
  --model <the served model name> \
  --tokenizer <a directory holding the model's tokenizer files> --tokenizer-trust-remote-code \
  --max-context-length 1048576 \
  --concurrency <C> \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --output-artifact-dir <dir>
```

- **Tokenizer files:** take them from the revision the engine serves, `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` of
  `zai-org/GLM-5.3-Flash` ([DEPLOY.md](DEPLOY.md)). `tokenizer.json` and `tokenizer_config.json` of that revision are
  byte-identical to the repository's current head (sha256 `19e77364…` and `98b12715…`); its chat template is not, so pin
  the revision rather than the name.
- **What the `agentx` command applies:** its built-in `agentx` scenario. The dataset has 393 entries (random seed 42,
  idle gaps capped at 300 s); one profiling phase of 3,600 s at the given concurrency with a 30 s grace period and a warmup
  of 10 requests per lane; chat completions, streamed, with the token counts taken from the server's usage (so the server
  must return usage on streamed responses); `ignore_eos` required; no input truncation; the first turn's prefix cache
  busted. The dataset downloads from the Hugging Face Hub on first use.
- **Score it with InferenceX's own code,** `inferencex-e2e/infx/results/agentic` in
  [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX), not a restatement: requests a minute, total
  tokens per second per chip (input, cached input included, plus output, over the profiled span, per chip: 8 for one
  engine), and the time to first token, end-to-end and inter-token latencies, with interactivity the inverse of the
  inter-token latency statistic. Records from the warmup phase and records with an error are dropped; each of our points
  dropped 164 to 222 records as invalid inference results, about 11 warmup requests per unit of concurrency among them.
- **Dollars per token** use a price per chip-hour you state. The numbers in [RESULTS.md](RESULTS.md#mtp-speculative-decoding-one-engine)
  use SemiAnalysis's $0.58 per trn2 chip-hour.
