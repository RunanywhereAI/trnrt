# Known issues

## Verification status

The published numbers were measured with an earlier build of TrnRT. This repository passes these checks against that
build ([RESULTS.md](RESULTS.md)):

| Check | Result |
|---|---|
| Graph equivalence (`tools/graph_equiv.py`), all 32 ranks, default batch-12 configuration | EQUAL for all 157 graphs that build compiles |
| Outputs on the device | Byte-identical to that build's on 384 of 384 requests |
| The README's two-engine serve | Passes end to end |

## Prompt processing: a small share of sparse-attention selections are inexact

In the sparse-attention (DSA) layers each prompt token selects 512 blocks of its earlier context to attend to. The
context is split across the eight ranks of a context-parallel group, so the selection merges each rank's shortlist (its
top 120 blocks, `CP_LOCAL_K` in `trnrt/models/mla.py`) and then checks a certificate: that the merge provably holds the
true top set. Rows that fail it, up to 128 per chunk and row group (`CP_LOCAL_F`), are selected again exactly. Rows
past that keep the merged shortlist: it always contains the true top 120 blocks, but up to 392 of the 512 may differ
from the exact selection.

- **How often.** In a run with real coding agents, none of 13 million selections failed the certificate. On AgentX's
  synthetic prompt content (the runs at 4, 8 and 16 agents), 2.8% of selections failed it and 1.2% kept the merged
  shortlist; in the batch-size runs, 2.8%, 2.9% and 2.5% failed and 1.2%, 1.3% and 1.1% kept it at batch sizes 4, 8
  and 12.
- **Effect on outputs.** AgentX prompts of 48K, 180K and 420K tokens were run through the engine twice: with this
  selection, and with every failed selection redone exactly. Scored on the same text, the merged shortlist moved the
  model's next-token log-probabilities by 0.012 to 0.021 nats on average, and changed the most likely next token at 3
  of 448 positions. Generating 64 tokens greedily, 2 of 6 prompts took a different continuation, from the 6th and 18th
  token on; a greedy continuation can diverge from a single near-tie.
- **Effect on the published times.** The benchmark results in [RESULTS.md](RESULTS.md) ran with this selection. On the
  1.2% of selections that kept the merged shortlist, an exact re-selection would have added an estimated 0.4 to 0.7%
  to total prompt-processing time, and up to 29 to 50% on the worst-affected prompt chunks. Per-request impact was not
  measured, so on requests built from such chunks the first-token and end-to-end times may be optimistic.
- **Watching it.** Each engine counts these rows on rank 0's row group, warmup included: `/metrics` reports
  `trnrt:dsa_local_k_rows_total`, `trnrt:dsa_local_k_failed_rows_total` (rows that failed the certificate) and
  `trnrt:dsa_local_k_inexact_rows_total` (rows past the limit, which kept the merged shortlist), and the engine logs
  the same counts when it closes (stderr), EXACT only if no row went past the limit: at INFO when EXACT, at WARNING
  when NOT EXACT, so `TRNRT_LOG_LEVEL=WARNING` still shows an inexact run.

## Batch 12 beyond 32 agents: the host-side prompt cache fills

Past its peak load, batch size 12 runs into the host-side tier of the prompt cache. In a run at 48 agents, one
instance's host tier was full before the measured hour began (7,339 of 7,340 pages) and had dropped 93,548 pages, so
more of each prompt was processed again: the share read from cache fell from 95% to 88%, and throughput from 3,720 to
2,642 tokens per second per chip. That run was cut at 44 of its 60 minutes, so it is not a valid AgentX submission and
is not in the results. Past that point, cache capacity, not the batch size, sets the limit. The host tier's size is
`--host-cache-gb` (4 GiB per rank by default); a larger tier has not been measured.

## `tests/test_dsa_long.py::test_cp_engine_matches_replicated`: logprobs outside 1e-4

On the CPU test model, context parallelism at tp 2 and 4 is compared with the replicated long path at tp 1. Every greedy
token agrees. The logprobs differ by up to 2.2e-3 at tp 2 and 1.3e-3 at tp 4, against the test's absolute tolerance of
1e-4, on x86 CPUs.

What is known:

- The failure is not new: an earlier version of this code fails the same tests with the same values.
- The difference is spread over nearly every logprob (9 of a request's 10), not concentrated in one token.
- It does not move with the CPU math library's threading or its reproducible mode: 2.17e-3 to 2.19e-3 at tp 2.
- The comparison runs with the FP8 KV cache, which context parallelism requires. Under FP8, a change in summation order
  that moves a cached value across a rounding boundary changes it by a whole FP8 step. The replicated path alone, with only
  its prefill chunk size changed (16 tokens to 12), already moves logprobs by up to 5.7e-4, above the 1e-4 tolerance.

Not shown: that context parallelism adds nothing beyond that ordering noise. Its difference is about 4 times the
chunk-size floor, so the tolerance is not loosened. The test is marked as an expected failure, strict: if a change makes it
pass, the suite fails until this entry and the mark are revisited.

This CPU test does not measure served accuracy. The served engine's accuracy was checked on the device, against a
high-precision reference, with the build the published numbers were measured on. On the device, this repository's
outputs were byte-identical to that build's on 384 of 384 requests ([Verification status](#verification-status)).
