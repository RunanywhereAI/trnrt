# Results

GLM-5.3-Flash on one AWS trn2.48xlarge under a coding-agent load, measured on 2026-10-26 (UTC) with the engine this
repository was extracted from, in the serving configuration this repository hard-wires. A graph-equivalence gate
(`tools/graph_equiv.py`) holds this repository to building the same compiled graphs as that engine; the gate's full run and
an on-device confirmation are pending. Numbers carry a tag: **[M]** measured, **[D]** arithmetic on measured numbers,
**[E]** an estimate, **[P]** published by a third party.

## Headline

| | |
|---|---|
| Requests per minute per box | **48.6** [M], at 12 concurrent agent sessions (the highest valid load level) |
| Output speed per user | **34.0 tokens/s** (p50) [M] |
| Time to first token | **0.96 s** (p50), 2.90 s (p90) [M] |
| Tokens per minute | **5.27M** input + output [M]; 97.8% of input tokens come from the prefix cache |
| Cost per agent request | **$0.0123** [D] at the public trn2 capacity-block price, $35.76/h; a request averages 108.1K input tokens (mostly cached) and 447 output |
| Answer over a cached 1M-token context | **1.84 s** [M]; a 1.04M-token prompt takes 210.8 s cold |

## Serving layout

| Setting | Value |
|---|---|
| Engines per box | 2, each tensor-parallel over 32 logical NeuronCores (8 Trainium2 chips), LNC = 2, behind a session-affine router |
| Decode batch cap | 8 requests per engine (16 per box), compiled batch sizes 1, 2, 4, 6, 8 |
| Prefill | chunks of 1,024 / 2,048 / 4,096 tokens |
| Prefix cache | on; KV cache in FP8 |
| Queue | unbounded (no request or token limit); admission mode `reserve` |
| Context | 1,048,576 tokens |
| Weights | FP8, as released |
| Compiled graphs | 130 per box layout, debug sections stripped |

## Agent load

OpenCode agents working through 16 fixed coding tasks with a 1M-token client context window, at 1 to 32 concurrent sessions
(C), 300 seconds per level. A level counts only if every agent seat ran for the whole window, no seat restarted and no
request stalled. C = 1, 4, 8 and 12 are valid; at C = 16, 24 and 32 one fewer seat ran than the level called for (and C = 24
restarted a seat), so those levels have no verdict and are shown as measured, not headlined.

| C | Validity | RPM | TPM in / out | Decode p50 / p10 (tok/s/user) | TTFT p50 / p90 (s) | Prefix hit | Running mean / max | Waiting mean / max | $/request at $35.76/h |
|---|---|---|---|---|---|---|---|---|---|
| 1 | valid | 2.0 | 0.08M / 3,499 | 73.4 / 73.4 | 0.82 / 3.54 | 80.6% | 0.9 / 2 | 0.0 / 0 | $0.2980 |
| 4 | valid | 24.6 | 1.17M / 10,652 | 62.6 / 43.7 | 0.56 / 1.42 | 95.5% | 3.2 / 7 | 0.0 / 0 | $0.0242 |
| 8 | valid | 43.0 | 3.59M / 17,018 | 45.3 / 29.9 | 0.81 / 2.12 | 96.7% | 7.0 / 11 | 0.0 / 0 | $0.0139 |
| **12** | **valid** | **48.6** | 5.25M / 21,713 | 34.0 / 19.4 | 0.96 / 2.90 | 97.8% | 10.6 / 14 | 0.1 / 2 | **$0.0123** |
| 16 | no verdict | 44.0 | 4.99M / 24,058 | 31.9 / 19.6 | 1.23 / 4.00 | 96.6% | 13.8 / 16 | 0.2 / 1 | $0.0135 |
| 24 | no verdict | 50.6 | 4.80M / 25,561 | 30.5 / 22.8 | 7.12 / 14.25 | 98.1% | 15.9 / 16 | 5.2 / 10 | $0.0118 |
| 32 | no verdict | 47.2 | 4.21M / 21,608 | 30.0 / 24.2 | 17.78 / 33.96 | 98.0% | 15.9 / 16 | 14.3 / 24 | $0.0126 |

Box totals over both engines [M]; each level's $/request [D] uses that level's own request mix. Throughput stops rising at
about C = 12 because the decode batch is capped at 8 per engine: running requests average 15.9 of 16 seats from C = 24 on,
and further sessions wait in the queue, which is where the TTFT rise comes from.

## Long prompts, one output token

Unique prompts with one output token, so the cost is all prefill [M]:

- cold time to first token is linear in prompt length, at 4,849 to 5,171 prompt tokens per second per engine, from 1.68 s at
  8,133 tokens to 210.82 s at 1,039,940 tokens;
- with the long prefix already cached, a 1,040,461-token request answers in 1.84 s and a 262,606-token request in 0.66 s;
- both engines together process 10,125 prompt tokens per second on 64K-token prompts, which is $0.98 per million uncached
  input tokens at the capacity-block price [D].

## Cost at public prices

Dollars per million total tokens (input including cached, plus output), SemiAnalysis's InferenceX formula: price per hour ÷
(tokens per second × 3,600) × 10⁶.

| Platform, load point | Per-user speed | AWS list | 3-year plan | SemiAnalysis owner TCO |
|---|---|---|---|---|
| trn2.48xlarge, C = 12 [D] | 34.0 tok/s | $0.113 | $0.117 | not published |
| 8×H200, estimated [E] | 29.6 tok/s | $0.264 | $0.114 | $0.041 |
| 8×B200, measured at saturation [M/D] | 110 tok/s | $0.137 | $0.059 | $0.017 |

AWS list means the trn2 capacity block ($35.76/h; AWS publishes no on-demand trn2 price) and GPU on-demand. Per request at
a similar per-user speed, trn2 at the capacity block costs 0.99× an 8×H200 box on a 3-year plan and 0.43× H200 on-demand [E].
8×B200 is 3.2× faster per user at saturation and, on a 3-year plan, cheaper per token.

Caveats:
- **H200 is an estimate.** No one has run GLM-5.3-Flash on H200. The point estimate scales vLLM's published run of the larger
  GLM-5.3 (753B parameters; 8,340 tokens/s per GPU including cached input, at 29.6 tokens/s per user, 16 GPUs split into
  prefill and decode) to an 8-GPU box on this run's request mix [P/E].
- **Different workloads.** The B200 numbers come from a different agent fleet (contexts to about 198K, 67.6K input tokens per
  request at saturation) with NVFP4 weights and speculative decoding; ours are FP8 weights, no speculation, contexts to
  268,977 tokens. No two of these runs replayed the same requests.
- **SemiAnalysis owner TCO is not a rental price,** and none is published for Trainium2.
- **Prices** are AWS's public price list and Capacity Blocks page read on 2026-10-06 (trn2.48xlarge capacity block $35.76/h,
  3-year EC2 Instance Savings Plan $37.14/h; p5en.48xlarge $63.30/h on-demand and $27.34/h on a 3-year plan; p6-b200.48xlarge
  $113.93/h and $49.22/h) and SemiAnalysis's TCO tiers read on 2026-10-26 ($1.22 per H200-hour, $1.73 per B200-hour, owner).

## Where a decode step's time goes

One batch-8 decode step at 122,880 tokens of context, profiled on the critical core [M] (the configuration served one day
before the release, without its last two decode changes; the released step is 23.72 ms):

| Part | ms |
|---|---|
| Whole step | 24.22 |
| Memory floor (the step's 2,038 MB of HBM reads at full bandwidth) | 6.07 |
| Engines busy beyond the floor | 9.99 |
| Waits inside kernels (DMA, semaphores, the other core) | 4.60 |
| Exposed collectives (nothing computes while a cross-chip sum runs) | 2.84 |
| Idle | 0.72 |

The step grows by about 1.4 ms per added row, from 13.6 ms at batch 1 to 23.7 ms at batch 8 [M], so a user's speed is capped
near 42 tokens/s at batch 8. One 4,096-token prefill chunk takes 795.95 ms on the device, 3.3% of its BF16 matrix-engine floor
(26.6 ms) [M/D]: prefill has the most headroom.

## Methodology

- **Interactivity** is the per-request median of output tokens ÷ (end-to-end time − time to first token). **RPM** and **TPM**
  are box totals; input tokens include the cached prefix. A request counts at a level if it starts inside that level's
  window.
- **Cost per request** at a trn2 level = price per hour ÷ (RPM × 60). GPU per-request costs use this run's C = 12 request mix,
  converted through total-token throughput.
- **Accuracy:** speed changes are adopted only after accuracy gates: byte-identical outputs where a change should not move any
  bits, and registered accuracy comparisons where it should.
