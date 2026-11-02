# Results

GLM-5.3-Flash on one AWS trn2.48xlarge, measured on 2026-11-02 (UTC) with the engine this repository was extracted from, in
the serving configuration this repository hard-wires. The repository's own graph-equivalence gate
(`tools/graph_equiv.py`) holds it to building the same 144 compiled graphs as that engine: on 2026-11-02 it read EQUAL
against the measured configuration (144 distinct graphs over four ranks, no differences), and its positive control (one
constant changed in one prefill kernel) still read DIFFER on exactly the graphs that kernel builds. Numbers carry a tag:
**[M]** measured, **[D]** arithmetic on measured numbers, **[E]** an estimate, **[P]** published by a third party.

## Headline

A synthetic agentic mix (described below) at 24 concurrent sessions, the highest load at which every run kept a median
output speed of at least 30 tokens/s per user. Three runs with different seeds; the middle run is shown, the range in brackets.

| | |
|---|---|
| Requests per minute per box | **51.5** [M] (48.5-53.7 over three runs) |
| Output speed per user | **34.1 tokens/s** (p50) [M] (34.1-35.3) |
| Time to first token | **1.07 s** (p50), 3.60 s (p90) [M] |
| Tokens per minute | **9.0M** input + output [M]; 98.3% of input tokens come from the prefix cache |
| Cost per request | **$0.0116** [D] at the trn2 capacity-block price ($35.7608/h); **$0.0334** at the AWS on-demand price ($103.157/h). A request averages 174.5K input tokens (mostly cached) and 384 output |
| Answer over a cached 1M-token context | **1.84 s** [M]; a 1.04M-token prompt takes 190.8 s cold |

## Serving layout

| Setting | Value |
|---|---|
| Engines per box | 2, each tensor-parallel over 32 logical NeuronCores (8 Trainium2 chips), LNC = 2, behind a session-affine router |
| Decode batch cap | 8 requests per engine (16 per box) |
| Prefill | chunks of 512 / 1,024 / 2,048 / 4,096 tokens |
| Prefix cache | on; KV cache in FP8, pages of 256 tokens; a host-memory tier of 4 GiB per rank |
| Context | 1,048,576 tokens |
| Weights | FP8, as released |
| Compiled graphs | 144 per box layout |

## The synthetic agentic mix (the headline workload)

Seeded synthetic sessions shaped like production coding-agent traffic: one shared system prompt with 12 tools; first prompts
of 80K-150K tokens; each turn extends the previous prompt until 233K tokens, then the seat starts a new session; every seventh
seat runs long sessions (260K-400K tokens at the start, up to 520K); replies p50 258 / p90 834 tokens; think time p50 1.3 s.
The client replays it closed-loop and sends each reply the model generated, with its reasoning, back in the next turn. Load
levels C = 12, 24 and 48 concurrent sessions, 900 seconds each. A level counts only if every seat sent a request in it, no
request failed and the client never limited the load; all nine level-runs are valid.

| C | RPM, median run (range) | Decode p50 (tok/s/user), median run (range) | TTFT p50 / p90 (s), median run | Prefix hit (range) |
|---|---|---|---|---|
| 12 | 29.6 (29.3-33.5) | 51.8 (48.4-51.8) | 0.80 / 1.79 | 97.2-97.5% |
| **24** | **51.5 (48.5-53.7)** | **34.1** (34.1-35.3) | **1.07 / 3.60** | 98.2-98.4% |
| 48 | 48.2 (47.0-48.9) | 29.1 (28.2-29.1) | 17.41 / 40.33 | 96.7-96.9% |

Box totals over both engines [M]. Throughput peaks near C = 24: the decode batch is capped at 8 per engine, so at C = 48
further sessions wait in the queue, which is where the TTFT rise comes from. The mix is synthetic: real agent traffic replays
differ in content and pacing.

## Coding agents

OpenCode agents working through 16 fixed coding tasks with a 1M-token client context window, at 1 to 32 concurrent sessions
(C), 300 seconds per level, on the same configuration [M]. A level counts only if every agent seat sent a request in it. In
this run one agent made no change to its task for 32 minutes and sent no request for 10 until the load generator restarted
it, so only C = 1 and C = 4 are valid; the other levels are shown as measured.

| C | Validity | RPM | TPM in / out | Decode p50 / p10 (tok/s/user) | TTFT p50 / p90 (s) | Prefix hit |
|---|---|---|---|---|---|---|
| 1 | valid | 2.8 | 0.08M / 3,920 | 73.7 / 73.3 | 0.51 / 1.42 | 88.5% |
| 4 | valid | 30.0 | 2.10M / 10,610 | 63.4 / 42.3 | 0.62 / 1.87 | 96.2% |
| 8 | no verdict | 49.4 | 5.64M / 14,561 | 47.9 / 31.0 | 0.90 / 2.23 | 97.7% |
| 12 | no verdict | 35.0 | 3.20M / 27,835 | 38.8 / 25.0 | 0.77 / 2.65 | 97.4% |
| 16 | no verdict | 45.6 | 4.90M / 17,160 | 30.4 / 20.4 | 1.01 / 3.51 | 96.7% |
| 24 | no verdict | 54.0 | 5.28M / 25,365 | 30.1 / 23.7 | 7.70 / 15.46 | 98.0% |
| 32 | no verdict | 56.6 | 5.08M / 19,495 | 26.8 / 20.0 | 17.28 / 24.12 | 97.2% |

A second run of the same shape, on a fresh server start, ended the same way: a different agent stalled on a different task
(no change for 25 minutes), so again only C = 1 and C = 4 are valid (C = 4: 28.0 requests per minute, decode p50 63.8
tokens/s per user, TTFT p50 0.62 s) [M].

## Long prompts, one output token

Unique prompts with one output token, so the cost is all prefill [M]:

- a 1,039,912-token prompt answers in 190.8 s cold, 5,451 prompt tokens per second on one engine;
- with that context cached, a new question over it (1,040,435 tokens, 1,039,872 cached) answers in 1.84 s (p50 of three,
  1.93 s max).

## Cost at public prices

Dollars per million total tokens (input including cached, plus output): price per hour ÷ (tokens per second × 3,600) × 10⁶.
trn2.48xlarge at the headline point (C = 24, 34.1 tokens/s per user, 150.0K tokens/s per box) [D]:

| Price | Per hour | Per million tokens | Per request | Per million output tokens |
|---|---|---|---|---|
| Capacity block | $35.7608 | $0.066 | $0.0116 | $30.17 |
| 3-year EC2 Instance Savings Plan | $37.14 | $0.069 | $0.0120 | $31.33 |
| On-demand | $103.157 | $0.191 | $0.0334 | $87.03 |

Prices are AWS's public price list: the capacity block and the 3-year plan read on 2026-10-06, the on-demand price confirmed
with the AWS Pricing API on 2026-11-02. Per-million-token costs depend on the request mix: this mix's requests carry 174.5K
input tokens each, 98.3% of them cached, so cached input dominates the token count.

## Where a decode step's time goes

One batch-8 decode step at 122,880 tokens of context, profiled on the critical core on an earlier configuration (2026-10-25)
[M]; the measured configuration's own decode profile is not shown here:

| Part | ms |
|---|---|
| Whole step | 24.22 |
| Memory floor (the step's 2,038 MB of HBM reads at full bandwidth) | 6.07 |
| Engines busy beyond the floor | 9.99 |
| Waits inside kernels (DMA, semaphores, the other core) | 4.60 |
| Exposed collectives (nothing computes while a cross-chip sum runs) | 2.84 |
| Idle | 0.72 |

On the measured configuration a 4,096-token prefill call takes 716.9 ms end to end (median of 24, host-timed) [M], against
748.0 ms on the configuration before it.

## Known issues

One CPU test is open: `tests/test_dsa_long.py::test_cp_engine_matches_replicated`, whose logprobs differ from the
replicated path by more than its tolerance while every greedy token agrees. It failed the same way before the port to the
measured configuration. [KNOWN_ISSUES.md](KNOWN_ISSUES.md) gives what is known and what is not.

## Methodology

- **Interactivity** is the per-request median of output tokens ÷ (end-to-end time − time to first token). **RPM** and **TPM**
  are box totals; input tokens include the cached prefix. A request counts at a level if it starts inside that level's
  window.
- **Cost per request** = price per hour ÷ (RPM × 60).
- **Runs:** the synthetic mix ran three times (seeds 1-3) on one server start; seed 1 started on empty caches, seeds 2 and 3 on
  caches holding the earlier runs' sessions. The load client ran on a separate machine and never limited the load (its
  run-queue wait share stayed under 0.4%). No device errors occurred and nothing else ran on the serving box.
- **Accuracy:** speed changes are adopted only after accuracy gates: byte-identical outputs where a change should not move any
  bits, and registered accuracy comparisons where it should.
