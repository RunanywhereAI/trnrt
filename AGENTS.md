# AGENTS.md

Guidance for coding agents and people working in this repository. `AGENTS.md` is the canonical file. `CLAUDE.md` is a
symlink to it, which `tools/check_docs.py` enforces, so edit this file and never the link.

trnrt serves LLMs on AWS Trainium2. The first model is GLM-5.3-Flash on one `trn2.48xlarge`: two engines of 32
tensor-parallel ranks behind a session-affine router, with NKI decode megakernels. The map of the code is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Commands

```bash
make dev                 # .venv (Python 3.13): CPU torch, editable install, test + lint tools, pre-commit hook
make lint                # pre-commit run --all-files: ruff, file checks, tools/check_docs.py
make test                # pytest -q: the CPU suite, no NeuronCore needed
make check               # both; CI runs the same (.github/workflows/ci.yml)
pytest tests/test_router.py -q    # one file
```

Nothing here needs a Trainium device. Device-only work (`docker/check_image.py --device`, `tools/check_mixed.py`,
`python -m trnrt serve`) needs a Trainium2 host and the image from `docker/Dockerfile`.

## Invariants

1. **Traced code is frozen against casual edits.** The compile cache's keys hash the traced graph code. Any edit under
   `trnrt/kernels/` or `trnrt/models/` (even a comment that shifts a line) can change every compiled graph and
   invalidate every compile cache. The same holds for anything that changes a graph shape: the seats, the bucket
   ladders, the pool sizes, the page size or a decode kernel route (`trnrt/config.py`, `trnrt/__main__.py` defaults).
   Make such a change only on purpose, prove it with `tools/graph_equiv.py`, and say so in the PR. Never run a
   formatter over these paths. The repository has no formatter configured for this reason.
2. **`trnrt.__version__` is part of every cache key.** Bump it only for a release ([docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#releasing)).
3. **Every argparse parser passes `allow_abbrev=False`** (`tests/test_no_abbrev.py`): one parser's argv is handed to
   another, and an abbreviation silently became a different flag once.
4. **The published tree names nothing it must not.** `tests/test_no_trace.py` scans every tracked file, path and
   commit message against encoded patterns. Its docstring says what it looks for. The upstream project this code
   derives from is credited in `NOTICE` and nowhere else. Run it before pushing; commit messages count.
5. **Copied source is listed in `THIRD_PARTY_NOTICES.md`** (`tests/test_third_party_notices.py` checks it).
6. **The engine thread owns engine state.** API handlers hand work to it through its inbox (`trnrt/serve/api.py`);
   they never mutate the scheduler or pools directly.
7. **Unsupported API fields are refused with a 400 that says why**, never silently ignored.
8. **Numbers in the docs carry their context**: instance, Neuron SDK, configuration, workload, precision, baseline,
   and how correctness was checked. Where a doc tags its numbers ([M] measured, [D] derived, [E] estimate), keep the
   tags.

## Tests

- Host code is tested on CPU. Kernels are tested against their `emulate()` reference; simulator tests are gated on the
  `nki` package with a `needs_nki` skip mark, and a new simulator test must carry it.
- Engine tests build a tiny random GLM-5.3-Flash and need transformers 5.17.0 (`glm5_next`).
- Env-gated tiers (`TRNRT_TEST_MODEL`, `TRNRT_TEST_GLM_CKPT`, `TRNRT_TEST_NETWORK=1`, `HLO_PB2_PATH`) are listed in
  [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#tests).
- `tests/test_dsa_long.py::test_cp_engine_matches_replicated` is a strict expected failure
  ([docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)). Do not loosen its tolerance to make it pass.

## Style

- Commit subjects are `<area>: <what changes>`, written as a sentence (`router: answer a malformed body with 400`).
- Docstrings and comments say why, with the measurement or the reference behind a number.
- Docs and the README use plain, short sentences, with tables over paragraphs. Use relative links; `make lint` checks
  them.
- Brand assets live in `docs/assets/`, in dark and light variants, selected with `<picture>`.

## Definition of done

`make check` passes, new behaviour has a test, docs and `CHANGELOG.md` are updated, and a change to traced code
carries graph-equivalence or device evidence.
