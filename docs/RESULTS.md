# Results

GLM-5.3-Flash on one AWS trn2.48xlarge, measured with an earlier build of trnrt, in the configurations this repository
reproduces ([CONFIGURATION.md](CONFIGURATION.md)). The graph-equivalence check (`tools/graph_equiv.py`) holds this
repository to compiling the same graphs as that build. On 2 Nov 2026 it read
EQUAL for 144 graphs on 4 of 32 ranks, before a later selection-kernel fix, and its positive control (one constant
changed in one prefill kernel) read DIFFER on exactly the graphs that kernel builds. A full 32-rank run and an
on-device check are in progress.

## Serving layout

| Setting | Value |
|---|---|
| Engines per box | 2, each tensor-parallel over 32 logical NeuronCores (8 Trainium2 chips), LNC = 2, behind a session-affine router |
| Decode batch | at most 4, 8 or 12 requests per engine, as each table states; 12 is the default configuration |
| Prefill | chunks of 512 / 1,024 / 2,048 / 4,096 tokens |
| Prefix cache | on; KV cache in FP8, pages of 256 tokens; a host-memory tier of 4 GiB per rank |
| Context | 1,048,576 tokens |
| Weights | FP8, as released |

## AgentX

We measured the box with SemiAnalysis's AgentX benchmark (agentx-harness), which replays the request patterns of real
Claude Code sessions: long, growing prompts with heavy prefix reuse and subagent fan-out. The standard scenario and
public trace set ran unmodified apart from the endpoint's details, at a 1M-token context, one hour per concurrency
point. A point counts only if the harness marks it a valid submission.

### Batch size

Each serving instance decodes at most a fixed number of requests at once: its batch size. Requests beyond it wait for
a slot. We ran AgentX at batch sizes of 4, 8 and 12 per serving instance, and show each at its peak load, the most
agents it serves before throughput stops rising. A batch size counts as saturated when, on each instance, it runs at
least 95% full on average with requests queued at least 90% of the time. If it isn't, we add agents (48, then 64) until
requests a minute rise by less than 5% between steps. Each row below is one batch size at its peak load, 32 concurrent
agents.

| Batch size | Requests / min | Tokens/s per chip (per box) | Peak TPM (per box) | Decode tokens/s per user (p50 / p90) | First token, cold (p50 / p90) | First token, cached (p50 / p90) | $ per M tokens, capacity block, $35.76/h | $ per M tokens, spot, $8.60/h (us-east-2, 3 Nov 2026) |
|---|---|---|---|---|---|---|---|---|
| 4 | 19.3 | 2,454 (39,266) | 2.36M | 46.6 / 34.7 | 82.37 s / 154.81 s | 76.73 s / 141.47 s | $0.253 | $0.061 |
| 8 | 25.2 | 3,147 (50,360) | 3.02M | 29.4 / 20.0 | 31.28 s / 96.69 s | 28.82 s / 77.29 s | $0.197 | $0.047 |
| 12 | 28.9 | 3,720 (59,516) | 3.57M | 22.3 / 14.5 | 15.18 s / 50.27 s | 6.53 s / 37.35 s | $0.167 | $0.040 |

*First token, cold: requests with less than 50% of their prompt read from the prompt cache (74, 91 and 108 requests at
batch sizes 4, 8 and 12, respectively); cached: the rest. Decode tokens/s per user: each request's output tokens after
the first, divided by its time after the first token; p50 is the median request, and p90 the speed that 90% of
requests reach or beat. Spot prices vary by region and hour.*

Batch 4 is the batch-8 configuration with `--max-running 4`. With 32 agents, each instance held about 16 sessions. At
batch sizes 4 and 8 the wait to start was most of a request's time: the median request waited 77 seconds for its first
token, of 95 seconds end to end, at batch size 4, and 29 of 47 seconds at batch size 8. At batch size 12 it waited
7.1 seconds. By AgentX's own per-user measure, which divides each request's whole time, the wait for its first token
included, by the tokens it returns, the P90 user gets 1.0, 2.4 and 4.0 tokens per second at batch sizes 4, 8 and 12.

Beyond its peak, batch size 12 runs into the host-side prompt cache ([KNOWN_ISSUES.md](KNOWN_ISSUES.md)).

### Batch 8 at 4, 8, 16 and 32 agents

At a batch size of 8 per serving instance, we also ran AgentX at 4, 8 and 16 concurrent agents. The 32-agent row is the
batch-size-8 run from the table above.

| Concurrent agents | Requests / min | Tokens/s per chip (per box) | TPM (per box) | Decode tokens/s per user (p50 / p90) | First token, cold (p50 / p90) | First token, cached (p50 / p90) | Prompt-cache hit | $ per request, capacity block, $35.76/h | $ per request, spot, $8.60/h (us-east-2, 3 Nov 2026) |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 5.5 | 913 (14,603) | 0.88M | 63.0 / 43.9 | 4.87 s / 12.39 s | 1.04 s / 2.74 s | 97% | $0.1084 | $0.0261 |
| 8 | 14.4 | 1,754 (28,065) | 1.68M | 48.5 / 33.3 | 7.35 s / 12.25 s | 0.94 s / 2.55 s | 96% | $0.0413 | $0.0099 |
| 16 | 19.1 | 2,580 (41,281) | 2.48M | 35.9 / 24.8 | 6.86 s / 18.83 s | 1.24 s / 5.08 s | 96% | $0.0312 | $0.0075 |
| 32 | 25.2 | 3,147 (50,360) | 3.02M | 29.4 / 20.0 | 31.28 s / 96.69 s | 28.82 s / 77.29 s | 95% | $0.0236 | $0.0057 |

*First token, cold and cached, and decode tokens/s per user: as in the batch-size table; 17, 31, 57 and 91 requests
were cold at 4, 8, 16 and 32 agents, respectively.*

By AgentX's per-user measure the P90 user gets 25.3, 25.3, 16.0 and 2.4 tokens per second at 4, 8, 16 and 32 agents.
It is no higher at 4 agents than at 8 because the slowest tenth of requests, which set the P90, are short answers: at 4
agents their median is 100 output tokens, and the first token takes 2.7 seconds, 60% of the request's time, a wait no
shorter at 4 agents than at 8. Counting only the time after the first token, the P90 user sees 44 tokens per second at
4 agents and 33 at 8. At 16 agents, the capacity-block price comes to $0.241 per million tokens, or $0.0312 per agent
request.

## Long prompts

Unique prompts with one output token, so the cost is all prompt processing:

- a 1,039,912-token prompt reaches its first token in 190.8 s cold on one engine (half the box), 5,451 prompt tokens per
  second;
- with that context cached, a new question over it (1,040,435 tokens, 1,039,872 cached) answers in 1.84 s (p50 of
  three, 1.93 s max).

A 4,096-token prefill chunk takes 716.9 ms end to end (median of 24 calls, timed on the host).

## Where a decode step's time goes

A device profile of one batch-8 decode step at 122,880 tokens of context, on the critical core, taken on an earlier
build than the AgentX runs above:

| Part | ms |
|---|---|
| Whole step | 24.22 |
| Memory floor (the step's 2,038 MB of HBM reads at full bandwidth) | 6.07 |
| Engines busy beyond the floor | 9.99 |
| Waits inside kernels (DMA, semaphores, the other core) | 4.60 |
| Exposed collectives (nothing computes while a cross-chip sum runs) | 2.84 |
| Idle | 0.72 |

## Other workloads

Both of these ran at batch size 8.

### Coding agents

OpenCode agents working through 16 fixed coding tasks with a 1M-token client context window, at 1 to 32 concurrent
sessions (C), 300 seconds per level. A level counts only if every agent seat sent a request in it. In this run one
agent sent no successful request for about 10 minutes and changed nothing in its worktree for 32, until the load
generator restarted it, so only C = 1 and C = 4 are valid, and only those are shown. Decode p10 is the speed that 90%
of requests reach or beat, which the AgentX tables call p90.

| C | RPM | TPM in / out | Decode p50 / p10 (tok/s/user) | TTFT p50 / p90 (s) | Prefix hit |
|---|---|---|---|---|---|
| 1 | 2.8 | 0.08M / 3,920 | 73.7 / 73.3 | 0.51 / 1.42 | 88.5% |
| 4 | 30.0 | 2.10M / 10,610 | 63.4 / 42.3 | 0.62 / 1.87 | 96.2% |

A second run of the same shape, on a fresh server start, ended the same way: one agent on a different seat and task sent
no successful request for about 10 minutes and changed nothing in its worktree for 25, so again only C = 1 and C = 4
are valid (C = 4: 28.0 requests per minute, decode p50 63.8 tokens/s per user, TTFT p50 0.62 s). Both stalls are on
the client side: the server kept answering the other seats.

### A different workload: synthetic sessions

Not AgentX: seeded synthetic sessions shaped like coding-agent traffic. One shared system prompt with 12 tools; first
prompts of 80K-150K tokens; each turn extends the previous prompt until 233K tokens, then the seat starts a new
session; every seventh seat runs long sessions (260K-400K tokens at the start, up to 520K); replies p50 258 / p90 834
tokens; think time p50 1.3 s. The client replays it closed-loop and sends each reply the model generated, with its
reasoning, back in the next turn. Load levels C = 12, 24 and 48 concurrent sessions, 900 seconds each, three runs with
different seeds per level. A level counts only if every seat sent a request in it, no request failed and the client
never limited the load; all nine level-runs are valid.

| C | RPM, median run (range) | Decode p50 (tok/s/user), median run (range) | TTFT p50 / p90 (s), median run | Prefix hit (range) |
|---|---|---|---|---|
| 12 | 29.6 (29.3-33.5) | 51.8 (48.4-51.8) | 0.80 / 1.79 | 97.2-97.5% |
| 24 | 51.5 (48.5-53.7) | 34.1 (34.1-35.3) | 1.07 / 3.60 | 98.2-98.4% |
| 48 | 48.2 (47.0-48.9) | 29.1 (28.2-29.1) | 17.41 / 40.33 | 96.7-96.9% |

Box totals over both engines. At C = 24 the median run moved 9.0M tokens a minute, input and output, with 98.3% of
input tokens from the prefix cache; a request averaged 174.5K input tokens, mostly cached, and 384 output. Throughput
peaks near C = 24: the decode batch is capped at 8 per engine, so at C = 48 further sessions wait in the queue, which
is where the TTFT rise comes from. The three runs shared one server start: seed 1 started on empty caches, seeds 2 and
3 on caches holding the earlier runs' sessions. The load client ran on a separate machine and never limited the load
(its run-queue wait share stayed under 0.4%). No device errors occurred and nothing else ran on the serving box. The
mix is synthetic: real agent traffic differs in content and pacing.

## Methodology

- **Tokens** per second or per minute count input, cached input included, and output, summed over both engines. AgentX
  counts them over the measured requests' span; in the coding-agent and synthetic runs a request counts at a level if
  it starts inside that level's window.
- **Decode speed per user** is the per-request rate of output tokens after the first ÷ (end-to-end time − time to first
  token); p50 is the median request.
- **AgentX's per-user measure** is the P90 of end-to-end time per output token, inverted: it includes the wait for the
  first token.
- **Cost:** $ per million tokens = price per hour ÷ (tokens per second × 3,600) × 10⁶; $ per request = price per hour ÷
  (requests per minute × 60). Prices: AWS's public capacity-block price for trn2.48xlarge, $35.76 per hour, and its
  public spot price in us-east-2 on 3 Nov 2026, $8.60 per hour (spot prices vary by region and hour).
- **Accuracy:** changes that should not move any bits are compared byte for byte. Everything else is checked against a
  high-precision reference on captured intermediates, not just final outputs. The benchmark numbers above ran with the
  sparse-attention selection that [KNOWN_ISSUES.md](KNOWN_ISSUES.md) describes.

## Reproducing

1. Serve the configuration under test ([DEPLOY.md](DEPLOY.md); `--preset batch8` and `--max-running 4` for batch 8 and
   4, [CONFIGURATION.md](CONFIGURATION.md)).
2. Run [AgentX](https://github.com/SemiAnalysisAI/agentx-harness) with its standard scenario and public trace set
   against the router's `/v1` endpoint. Use a 1M-token context, one hour per concurrency point, and a load client on
   a separate machine.
3. Count a point only if the harness marks it a valid submission.

A number added to these docs states the instance, Neuron SDK, configuration, workload and shapes, precision, the
baseline it is compared with, and how correctness was checked ([CONTRIBUTING.md](../CONTRIBUTING.md)).

## Known issues

[KNOWN_ISSUES.md](KNOWN_ISSUES.md) covers the sparse-attention selection that is inexact for a small share of prompt
rows, the host-side prompt cache at batch size 12 beyond 32 agents, and one CPU test marked as a strict expected
failure.
