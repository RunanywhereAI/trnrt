# Contributing to TrnRT

Thanks for helping. Bug reports, performance work, kernel fixes, docs and new model families are all welcome.

## Quick start

1. **Fork** the repository and clone your fork.
2. **Set up and check:** `make dev && make check`. You need Python 3.13 and no Trainium device.
3. **Open a pull request against `development`.**

That's all. The rest of this page says what happens next and what a change needs.

## Where to go

| You want to | Go to |
|---|---|
| Report a bug | [a bug report](https://github.com/RunanywhereAI/trnrt/issues/new?template=10-bug-report.yml) |
| Report something slower than before | [a performance regression](https://github.com/RunanywhereAI/trnrt/issues/new?template=40-performance-regression.yml) |
| Ask for a feature or a model | [a feature request](https://github.com/RunanywhereAI/trnrt/issues/new?template=20-feature-request.yml) or [a model request](https://github.com/RunanywhereAI/trnrt/issues/new?template=30-model-request.yml) |
| Fix or improve the docs | [a docs issue](https://github.com/RunanywhereAI/trnrt/issues/new?template=50-docs.yml), or just open the PR |
| Ask a question | [Discussions](https://github.com/RunanywhereAI/trnrt/discussions) |
| Report a security issue | [SECURITY.md](SECURITY.md), never a public issue |
| Find something to work on | issues labelled [`good first issue`](https://github.com/RunanywhereAI/trnrt/labels/good%20first%20issue) or [`help wanted`](https://github.com/RunanywhereAI/trnrt/labels/help%20wanted) |

For a larger change (a new model family, a new kernel, an API change), open an issue first so we can agree on the
approach before you write it.

## What happens to your pull request

| Step | What it is | Blocks the merge |
|---|---|---|
| CI | lint and format (pre-commit, including the workflow linters), mypy, the CPU test suite, package build, Dockerfile lint. `make check` runs the first three locally | yes |
| PR checks | the title format, and a note when the PR touches traced code | no, advice |
| AI review | CodeRabbit, cubic and Greptile review the PR when it is ready for review (not while it is a draft) and after each push | their status never blocks; their comments do, like any other (next row) |
| Maintainer review | one approval from a maintainer ([CODEOWNERS](.github/CODEOWNERS)) | yes |
| Open comments | every review comment, a bot's included, is fixed or answered, and its conversation resolved | yes |

A first-time contributor's CI runs once a maintainer approves it. PRs into `development` are squash-merged, so the PR
title becomes the commit subject. If a comment is wrong, reply with why and resolve it; a fix gets a reply naming its
commit.

## Rules every change follows

1. **Traced code changes graphs.** Any edit under `trnrt/kernels/` or `trnrt/models/`, or in
   `trnrt/engine/model_runner.py`, `trnrt/compile/capture.py`, `trnrt/config.py` or `trnrt/platform.py`, and any
   change to a graph shape, can change the compiled graphs. That includes a comment that moves a line. A change there
   needs `tools/graph_equiv.py` evidence, and a device run for new kernel forms. Say in the PR what it changes and how
   you checked it. The rest of the host code (`trnrt/serve/`, the rest of `trnrt/engine/` and `trnrt/compile/`) has
   no such constraint.
2. **A performance claim carries its context.** State the instance, Neuron SDK, configuration (`--preset` and flags),
   workload and shapes, precision, the baseline it is compared with, and how correctness was checked. A number
   without these does not go in the docs.
3. **Accuracy before speed.** A change that should not move any bits must show byte-identical outputs. A change that
   should move them must show an accuracy comparison against a reference.
4. **Tests come with the change.** Test the behaviour, not the implementation. Host-side code is tested on CPU, and
   kernels against their `emulate()` reference in the simulator.
5. **Keep the OpenAI API compatible.** Refuse what is unsupported with a 400 that says why. Never silently ignore a
   field.

The CPU suite needs no NeuronCore. Kernel tests that need the NKI simulator, and tests that need a checkpoint, skip
cleanly without them. [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) lists every test tier and how to run it.

## Commits and pull requests

- PR title and commit subject: `<area>: <what changes>`, written as a plain sentence. Examples: `router: answer a
  malformed body with 400` or `kda_half: one program's measured weight-stream rate`. The area is a module or doc name.
- One logical change per PR. Fill in the PR template.
- Pull requests target `development`. `main` takes releases only
  ([docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#branches-and-releases)).

## License

By contributing you agree that your contributions are licensed under the [Apache License 2.0](LICENSE).
