# Contributing to TrnRT

Thanks for helping. Bug reports, performance work, kernel fixes, docs and new model families are all welcome.

## Before you start

- **Bugs:** open an issue with the bug form. It asks for the instance type, the Neuron SDK version, `neuron-ls` output,
  the exact command and the logs.
- **Larger changes** (a new model family, a new kernel, an API change): open an issue first, so we can agree on the
  approach before you write it.
- **Security issues:** do not open a public issue. See [SECURITY.md](SECURITY.md).

## Set up

```bash
git clone https://github.com/RunanywhereAI/trnrt-glm.git && cd trnrt-glm
make dev      # Python 3.13 venv: editable install, test and lint tools, the pre-commit hook
make check    # lint, type-check, then the CPU test suite
```

The CPU suite needs no NeuronCore. Kernel tests that need the NKI simulator, and tests that need a checkpoint, skip
cleanly without them. [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) lists every test tier and how to run it.

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

## Commits and pull requests

- Commit subject: `<area>: <what changes>`, written as a plain sentence. Examples: `router: answer a malformed body
  with 400` or `kda_half: one program's measured weight-stream rate`. The area is a module or doc name.
- One logical change per commit. Keep each PR focused, and fill in the PR template.
- CI runs `pre-commit run --all-files` (ruff lint and format), mypy and the CPU suite. All must pass: `make check`
  runs the same locally.

## License

By contributing you agree that your contributions are licensed under the [Apache License 2.0](LICENSE).
