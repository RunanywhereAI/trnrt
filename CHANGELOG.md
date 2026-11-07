# Changelog

All notable changes to TrnRT are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/),
and versions follow [Semantic Versioning](https://semver.org/). Rebuild the compile cache for each release.

## [Unreleased]

### Added

- Public release: README, docs, contributing and security policies, issue and pull request templates.
- CI: pre-commit (ruff lint and format), mypy, the CPU test suite, a package build check, Dockerfile lint and CodeQL.
  Tag-triggered release workflow.
- `tools/check_docs.py`: every relative Markdown link and image resolves.

### Changed

- The display name is TrnRT. The package, the `trnrt` command, environment variables and metric names are unchanged.
- Server, engine and router messages go through Python `logging`, on stderr. Their text is unchanged, and
  `TRNRT_LOG_LEVEL` sets the level. They used to be printed, some on stdout. A hang report still reaches stderr when
  nothing has configured logging. The CLIs' own start-up status lines (`engine up in`, `bucket warmup`,
  `engine N ready after`) are still printed, as is one status line in the frozen `model_runner.py`.
- Code style: the ruff formatter and a strict ruff rule set over all host code, tests and tools, plus mypy with zero
  errors. All enforced in pre-commit and CI. Traced code (kernels, models) is excluded, so no compiled graph changes.

### Fixed

- An unknown request field is refused with 400 naming it, as AGENTS.md invariant 7 requires. It was ignored, and the
  request answered 200 as if it had been applied (`prediction`, `modalities`, `audio`, a misspelled field).
- The router answers a request body that is not a JSON object with 400, as each engine does, instead of 500.

## [0.1.0]

- GLM-5.3-Flash on one trn2.48xlarge: two engines of 32 tensor-parallel ranks behind a session-affine router, an
  OpenAI-compatible API, 1M-token context, a prefix cache with a host-memory tier, and NKI decode megakernels.
