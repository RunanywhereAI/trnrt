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
| Private trace scan | `tests/test_no_trace.py`'s scan for the maintainers' private patterns, and its positive control | `TRNRT_TRACE_PATTERNS` = the maintainers' patterns file, kept outside the repository |
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

| Workflow | Runs | On | Blocks a merge |
|---|---|---|---|
| `ci.yml` | pre-commit (ruff lint and format, file checks, doc links, actionlint and zizmor on the workflows); mypy; the CPU suite; package build and `twine check`; Dockerfile lint | every PR, every push to `development` and `main` | yes, all four jobs |
| `promotion.yml` | "Promotion source": the PR comes from this repository's `development` | every PR into `main` | yes, on `main` |
| `pr-checks.yml` | the PR title reads `<area>: <what changes>`; a warning and checklist when traced code changes | every PR | no |
| `dependency-review.yml` | new dependencies' known vulnerabilities and licences | every PR | no |
| CodeQL (GitHub's default setup, extended query suite) | security analysis of the Python and the workflows | pushes to `development` and `main`, PRs from branches of this repository, weekly | no |
| `scorecard.yml` | OpenSSF Scorecard | pushes to `development`, weekly | no |
| `links.yml` | external links in the Markdown docs | weekly | no |
| `release.yml` | build and check the sdist, wheel and container image; on a release tag, the GitHub release, PyPI and the image when enabled | a `vX.Y.Z` tag on `main`; a dry run on PRs that change it, `docker/` or `pyproject.toml`, or by hand | — |

CodeRabbit, cubic and Greptile review every ready-for-review PR into `development` (`.coderabbit.yaml`, `cubic.yaml`,
`greptile.json`). None is a required check. Dependabot keeps the pinned actions and Python dependencies current, a
week after each release. Every third-party action is pinned to a commit SHA, and every workflow runs on a read-only
token unless a job needs more.

## Branches and releases

| Branch | Takes | Protection |
|---|---|---|
| `development` (default) | every pull request, squash or merge | 1 maintainer approval, the 4 CI jobs, no force-push or deletion |
| `main` | promotions of `development`, for releases only | 1 maintainer approval, the 4 CI jobs and "Promotion source", merge commits only, only admins merge, no force-push or deletion |

`v*` tags are created by admins only, and nobody can move or delete one.

To cut a release:

1. **Prepare on `development`.** One PR bumps `__version__` in `trnrt/__init__.py` and moves `CHANGELOG.md`'s
   Unreleased entries under the new version. Rebuild the compile cache for every release: a release can change graphs,
   and a mismatched cache fails start-up.
2. **Check on a device.** On a `trn2.48xlarge`, `docker/check_image.py --device` passes on the image built from that
   commit, and the default configuration (batch 12) compiles and serves on all 32 ranks. `tools/graph_equiv.py`
   against the previous release's capture shows which graphs changed, if any. CI has no Trainium runner, so the
   commands, configuration and results go in the release PR.
3. **Promote.** A PR from `development` into `main`, titled `release: vX.Y.Z`, merged with a merge commit.
4. **Tag** `main`'s merge commit: `git tag vX.Y.Z <commit> && git push origin vX.Y.Z`. `release.yml` refuses a tag
   that is not on `main` or does not match `__version__`.
5. **Publish.** `release.yml` creates the GitHub release with the sdist, the wheel and their `SHA256SUMS`. PyPI runs when the repository
   variable `PUBLISH_PYPI` is `true` and a trusted publisher is configured for the `pypi` environment. The image is
   pushed to `ghcr.io/runanywhereai/trnrt` when `PUBLISH_IMAGE` is `true`.

An urgent fix takes the same path: a PR into `development`, then a promotion and a patch release.
