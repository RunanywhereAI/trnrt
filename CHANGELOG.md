# Changelog

All notable changes to trnrt are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/),
and versions follow [Semantic Versioning](https://semver.org/). A version change also changes the compile cache's
keys, so a cache is rebuilt for each release.

## [Unreleased]

### Added

- Public release: README, docs, contributing and security policies, issue and pull request templates.
- CI: pre-commit lint, the CPU test suite, a package build check and CodeQL. Tag-triggered release workflow.
- `tools/check_docs.py`: every relative Markdown link and image resolves.

### Fixed

- The router answers a request body that is not a JSON object with 400, as each engine does, instead of 500.

## [0.1.0]

- GLM-5.3-Flash on one trn2.48xlarge: two engines of 32 tensor-parallel ranks behind a session-affine router, an
  OpenAI-compatible API, 1M-token context, a prefix cache with a host-memory tier, and NKI decode megakernels.
