# Development

## Set up

```bash
make dev     # .venv with Python 3.13: CPU torch, editable install with test and lint tools, the pre-commit hook
make check   # lint, type-check, then the CPU test suite
```

`make help` lists every target. If `python3.13` on your PATH is a shim (uv, pyenv) that cannot create a venv, pass the
interpreter's real path: `make dev PYTHON=/path/to/python3.13`. Without `make`, install CPU torch first
(`pip install torch --index-url https://download.pytorch.org/whl/cpu`), then run
`pip install -e '.[dev]' transformers==5.17.0 tokenizers==0.23.3`.

## Tests

The suite runs on CPU, without a NeuronCore, on x86_64 Linux and on macOS. The image's versions are Python 3.13,
transformers 5.17.0 and tokenizers 0.23.3. In the image:

```bash
docker run --rm --entrypoint python trnrt -m pytest
```

| Tier | What runs | Needs |
|---|---|---|
| CPU (default, CI) | Host code, the engine end to end on a tiny random model of GLM-5.3-Flash's architecture, the API and router over real HTTP, every kernel's `emulate()` reference | Nothing beyond `make dev` |
| NKI simulator | Kernels against their reference in the NKI simulator | The Neuron venv's `nki` package; skipped otherwise |
| Checkpoint | API tests over the real model; the KDA shell on the checkpoint's own weights | `TRNRT_TEST_MODEL` / `TRNRT_TEST_GLM_CKPT` = a local GLM-5.3-Flash directory |
| Hub | Tests that read the model's small files (config, tokenizer, chat template, safetensors headers) from the Hub | `TRNRT_TEST_NETWORK=1` |
| Graph equivalence | `tools/graph_equiv.py` over captured graphs | `libtorch_neuronx_lite`, or `HLO_PB2_PATH` = a directory holding `hlo_pb2.py`, plus `zstd` |
| Device | `docker/check_image.py --device`, `tools/check_mixed.py`, serving | A Trainium2 host |

Two tests document known limits rather than bugs. `tests/test_dsa_long.py::test_cp_engine_matches_replicated` is a
strict expected failure ([KNOWN_ISSUES.md](KNOWN_ISSUES.md)).
`tests/test_router.py::test_router_does_not_cap_streams_in_flight` opens 130 connections at once, more than macOS's
default listen backlog of 128, so it runs on Linux only (CI).

## Lint

| Tool | What it enforces | Run |
|---|---|---|
| ruff (lint) | pycodestyle, pyflakes, isort, pyupgrade, bugbear, bandit (security), pytest style, simplify, comprehensions, logging, refurb and more; every exception is listed with its reason in `pyproject.toml` | `make lint` |
| ruff (format) | one code style, line length 120 | `make lint` |
| mypy | types over `trnrt/`, `tools/` and `docker/` | `make typecheck` |
| pre-commit | the above plus file hygiene (whitespace, EOF, YAML/TOML, merge markers, private keys, large files) and `tools/check_docs.py` | `pre-commit install` once; then on every commit |

**Traced code is the one exception.** `trnrt/kernels/`, `trnrt/models/`, `trnrt/engine/model_runner.py`,
`trnrt/compile/capture.py`, `trnrt/config.py` and `trnrt/platform.py` are never reformatted or auto-fixed, and are
linted for correctness (pyflakes) only. A moved line in a kernel can change its compiled binary, and with it every
compile cache. A change there is made by hand, on purpose, with `tools/graph_equiv.py` evidence.

Server and engine messages go through `logging` (`trnrt/log.py`): library code logs, and each entry point configures
output once. `TRNRT_LOG_LEVEL` sets the level (default `INFO`). A program that embeds TrnRT (builds an engine or an
app itself, under its own uvicorn for example) calls `trnrt.log.configure()` once, or configures the root logger
itself; `configure()` then adds no second handler. Hang reports reach stderr either way.

## CI

| Workflow | Runs | On |
|---|---|---|
| `ci.yml` | pre-commit (ruff lint and format, file checks, doc links); mypy; the CPU suite; package build and `twine check`; Dockerfile lint | every push to `main` and every PR |
| `codeql.yml` | CodeQL security analysis of the Python (once the repository is public) | push and PR to `main`, weekly |
| `release.yml` | build, check and attach the sdist and wheel to a GitHub release; PyPI when enabled | a `vX.Y.Z` tag that matches `trnrt.__version__` |

Dependabot keeps the pinned actions current. Every third-party action is pinned to a commit SHA.

## Releasing

1. Bump `__version__` in `trnrt/__init__.py` and move `CHANGELOG.md`'s Unreleased entries under the new version.
   Rebuild the compile cache for every release: a release can change graphs, and a mismatched cache fails start-up.
2. Merge to `main`, then tag: `git tag v0.2.0 && git push origin v0.2.0`.
3. To publish to PyPI as well, configure a trusted publisher for the `pypi` environment, and set the repository
   variable `PUBLISH_PYPI` to `true`.
