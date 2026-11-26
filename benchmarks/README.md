# Benchmarks

The recipes behind every trn2 number we publish for GLM-5.3-Flash: what was served, how the load was generated, and how a
run is scored and checked for accuracy. One row of [points.csv](points.csv) is one measurement; its `published` column
dates the run behind the point's published number, and is empty for the AgentX points at 2, 6, 7 and 12 users, which
this repository has not measured. Each point is served by a build from [builds.csv](builds.csv), and each build's engine
arguments are the same for the compile cache and for `serve`.

| Benchmark | Load | Scored with | Points |
|---|---|---|---|
| AgentX | SemiAnalysis's AgentX harness 1.0.12, its public coding-agent traces, one hour per point | [agentx/score.py](agentx/score.py): InferenceX's AgentX definitions | `agentx-c*` |
| 8K/1K fixed length | InferenceX's fixed_seq recipe at commit 47cdc9d0: 8,192 input / 1,024 output tokens, random prompts | [fixed_seq/score.py](fixed_seq/score.py) | `fixed-off-c*` (no speculation), `fixed-on-c*` (speculation, real acceptance), `fixed-golden-c*` (speculation at the golden length, as InferenceX's GPU runs with MTP on) |
| GSM8K | InferenceX's own GSM8K task and lm-eval hooks | lm-eval | per build |
| Needle | Three facts at depths 0.1, 0.5 and 0.9 of a long document, asked cold and after other documents have gone through the cache | [accuracy/needle.py](accuracy/needle.py) | per build |

## The unit: one engine, 8 chips

Every point is one engine of 8 Trainium2 chips (tensor parallel 32 at logical NeuronCore 2), half a trn2.48xlarge, with
the other half idle. [serve.sh](serve.sh) starts exactly that (`trnrt serve --engines 1`). Throughput is per chip, and the
cost uses a price per chip-hour, so a point compares directly with an 8-GPU node. `trnrt serve`'s default of two engines
is the deployment form for a whole box. Its numbers are per 16 chips.

InferenceX's rule is one server deployment per concurrency, so start a fresh server for every point. The published AgentX
points each ran on a fresh server. **The published 8K/1K points did not follow that rule**: each build's points ran in
ascending concurrency on one server, the form of our earlier 8K/1K runs. That was `fixed-off-c1` to `fixed-off-c16` on
one `balanced` server, `fixed-off-c32` and `fixed-off-c64` on one `throughput` server, and `fixed-on-*` and
`fixed-golden-*` on one server each. Their prompts are random, so no point reuses another's prefix cache, but they have
not been rerun with a server each. To reproduce them as they ran, start a build's server once and run its points in that
order; to follow the rule, start one per point.

## Builds

| Build | Engine arguments | Speculation |
|---|---|---|
| `latency-golden` | `--preset latency --spec-synth-al 2.518506` | every step, **synthetic acceptance** at InferenceX's golden length for 2 draft tokens |
| `latency` | `--preset latency` | every step, real acceptance |
| `balanced` | `--preset balanced` | none |
| `throughput` | `--preset throughput` | none |

`latency-golden` is the benchmark mode ([AGENTX.md](../docs/AGENTX.md#benchmark-mode)). InferenceX scores speculative
decoding at a fixed acceptance length, so its AgentX points use that length instead of the draft head's. Drafting and
verification run for real, but an accepted position emits its draft whether or not the target agrees, so **the text such
a server returns is not the model's**. The server says so in its log, in `GET /health` (`benchmark_mode`) and in the
`x-trnrt-synthetic-acceptance` header on every response. The accuracy tools refuse it, and the clients check that a
point's server runs the point's mode. Accuracy is always measured with real acceptance, on every build that serves with
it: `latency`, `balanced` and `throughput` ([Accuracy](#accuracy)).

The mode adds one input to one small graph, so `latency-golden` has a compile cache of its own.

## Running a point

On a compile host, once per build ([DEPLOY.md](../docs/DEPLOY.md) steps 1-3 inside one script):

```bash
OUT=/data/trnrt benchmarks/compile.sh latency-golden latency balanced throughput
```

On the trn2.48xlarge, for each point:

```bash
CACHE=/data/trnrt/cache NKI=/data/trnrt/nki MODEL=/models/GLM-5.3-Flash benchmarks/serve.sh agentx-c16
```

The published points kept the engine off the CPUs the 8K/1K client and the host used:
`DOCKER_ARGS=--cpuset-cpus=0-39,48-87,96-135,144-183`.

On the load client, a separate CPU machine for AgentX (state its core count next to any number; the harness is
CPU-bound). The published points ran it on a c7i.48xlarge, pinned with `CPUS` to 48 logical CPUs (24 cores):

```bash
pip install agentx-harness==1.0.12 -c benchmarks/agentx/constraints.txt   # the exact versions the published points ran
CPUS=48-71,144-167 TOKENIZER=/models/GLM-5.3-Flash benchmarks/agentx/client.sh agentx-c16 http://<server>:8000 runs/agentx-c16
```

For the 8K/1K points, check out InferenceX at commit `47cdc9d02d2f50018efbbc207c0c3ce9e425dae4` and install what its
client imports, constrained to the versions the published points ran:

```bash
pip install aiohttp numpy transformers tokenizers tqdm huggingface_hub jinja2 -c benchmarks/fixed_seq/constraints.txt
```

That gives aiohttp 3.14.5, numpy 2.5.4 and transformers 5.22.1 (InferenceX's own pins), tokenizers 0.23.3 (it sets how the client
counts and builds the prompts' tokens), tqdm 4.70.1 and huggingface_hub 2.4.0. jinja2 3.1.6 is needed by the speculative points
(`fixed-on-*`, `fixed-golden-*`) for the chat template.

The published points ran this client on the serving host, on CPUs the engine did not use:

```bash
CPUS=40-47 INFX=/src/InferenceX TOKENIZER=/models/GLM-5.3-Flash benchmarks/fixed_seq/client.sh fixed-off-c16 http://127.0.0.1:8000 runs/fixed-off-c16
```

`serve.sh`, `compile.sh` and the two `client.sh` scripts take `--dry-run`, which prints the exact command without running
it. `python3 -I benchmarks/points.py list` lists the points.

## Accuracy

On each build served with real acceptance (`latency`, `balanced`, `throughput`):

```bash
benchmarks/accuracy/install_lm_eval.sh                       # once: lm-eval at InferenceX's pinned commit
benchmarks/accuracy/gsm8k.sh http://127.0.0.1:8000 runs/gsm8k-latency 8     # concurrency: 8 latency, 16 balanced, 24 throughput
python3 -I benchmarks/accuracy/needle.py --server http://127.0.0.1:8000 --verdict runs/needle-latency.txt
```

GSM8K is 5-shot strict exact match over the 1,319 test problems. The bar for this repository's builds is 94%; H100 with MTP
scores 97.42% on the same eval.

## What a result states

Per [CONTRIBUTING.md](../CONTRIBUTING.md), every number says:
- the instance and the Neuron SDK;
- the build (and the repository commit);
- the workload and the client's core count;
- the precision (FP8 weights as released, FP8 KV cache);
- the baseline it is compared with;
- how correctness was checked.

[record.py](record.py) writes one result file per run. It holds the point, its build and engine arguments, the score,
the versions from [versions.sh](versions.sh) (the image, the Neuron SDK inside it, the host's driver), the client's core
count and the commit. It is scrubbed of hosts, addresses and paths:

```bash
IMAGE=trnrt benchmarks/versions.sh > versions.json                       # on the serving host
python3 -I benchmarks/record.py --point agentx-c16 --score runs/agentx-c16/score.json --versions versions.json \
  --client-cpus 48 --out results/agentx-c16.json
```

The scorers print one JSON object per run, with tokens per dollar at SemiAnalysis's trn2 rate ($0.58 per chip-hour) and at
AWS's EC2 Capacity Block list price for trn2.48xlarge ($35.76 an hour, $2.235 per chip-hour). `--price NAME=USD` adds
another.
