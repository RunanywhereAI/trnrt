# Reproducing the InferenceX AgentX results

> **Work in progress.** This page tracks the port of multi-token prediction (MTP) speculative decoding into trnRT, together
> with the serving configurations behind our InferenceX AgentX measurements of GLM-5.3-Flash on one trn2.48xlarge. The
> commands below are filled in as each piece lands and passes its checks.

## What is being added

- **MTP speculative decoding** with GLM-5.3-Flash's own MTP layer: 2 draft tokens per step, verified by the target model.
- **MTP only while few users decode.** MTP runs while at most 4 requests decode in a step and plain decode runs above that,
  because past that point a verify step costs more per token than it saves. Plain steps still write the MTP layer's KV
  cache, so a return to MTP finds it complete.
- **Larger decode batches** for higher concurrency (12 requests per engine), with a state checkpoint row per decoding seat.
- **Host DRAM prompt cache sizes** of 8 and 12 GB per NeuronCore for concurrency 12 to 32.
- **A benchmark mode** that reproduces InferenceX's synthetic acceptance: drafting and verification run for real, and
  accepted positions follow InferenceX's golden acceptance length (2.519 tokens per step for 2 draft tokens). It is off
  by default, and only meant for comparing with InferenceX's published GPU runs.

## Configurations

| Concurrency | Configuration |
|---|---|
| 1, 4, 8 | MTP while ≤ 4 users decode, decode batch 8 |
| 2 | MTP always on, decode batch 8 |
| 12, 16 | MTP while ≤ 4 users decode, decode batch 12, 8 GB host cache per core |
| 24, 32 | MTP while ≤ 4 users decode, decode batch 12, 12 GB host cache per core |

Launch commands, compile steps and the harness invocation will follow here.

## Workload

InferenceX's canonical AgentX workload (agentx-harness 1.0.12), one hour per point, scored with InferenceX's definitions.
One engine is 8 trn2 chips (tensor parallel 32 at logical NeuronCore 2).
