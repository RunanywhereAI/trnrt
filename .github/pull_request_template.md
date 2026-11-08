<!-- Target `development`. The title reads "<area>: <what changes>". AI reviewers start when the PR is ready for review. -->

## What and why

<!-- The problem, and how this change solves it. Link the issue. -->

## Graphs

- [ ] Host-side only: no change under `trnrt/kernels/` or `trnrt/models/`, and no graph shape changes
- [ ] Changes compiled graphs: which ones, and the `tools/graph_equiv.py` or device evidence

## Accuracy

<!-- Byte-identical outputs where nothing should move; an accuracy comparison where something should. -->

## Performance

<!-- Only if this claims a speed-up: instance, Neuron SDK, configuration, workload, baseline, before / after. -->

## Checklist

- [ ] `make check` passes (lint, type-check and the CPU suite)
- [ ] Tests cover the change
- [ ] Docs updated (README, `docs/`, `CHANGELOG.md`)

<!-- label smoke test: closed unmerged -->
