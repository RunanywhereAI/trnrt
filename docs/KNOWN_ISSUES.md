# Known issues

## `tests/test_dsa_long.py::test_cp_engine_matches_replicated`: logprobs outside 1e-4

On the CPU test model, context parallelism at tp 2 and 4 is compared with the replicated long path at tp 1. Every greedy
token agrees. The logprobs differ by up to 2.2e-3 at tp 2 and 1.3e-3 at tp 4, against the test's absolute tolerance of
1e-4, on x86 CPUs.

What is known:

- The failure predates the port to the frozen configuration: the same tests fail with the same values before and after it.
- The difference is spread over nearly every logprob (9 of a request's 10), not concentrated in one token.
- It does not move with the CPU math library's threading or its reproducible mode: 2.17e-3 to 2.19e-3 at tp 2.
- The comparison runs with the FP8 KV cache, which context parallelism requires. Under FP8, a change in summation order
  that moves a cached value across a rounding boundary changes it by a whole FP8 step. The replicated path alone, with only
  its prefill chunk size changed (16 tokens to 12), already moves logprobs by up to 5.7e-4, above the 1e-4 tolerance.

Not shown: that context parallelism adds nothing beyond that ordering noise. Its difference is about 4 times the
chunk-size floor, so the tolerance is not loosened. The test is marked as an expected failure, strict: if a change makes it
pass, the suite fails until this entry and the mark are revisited.

This CPU test does not gate served accuracy: the served configuration's accuracy is checked on the device, against the
reference model, by its own gates.
