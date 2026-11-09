# Changelog

All notable changes to TrnRT are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/),
and versions follow [Semantic Versioning](https://semver.org/). Rebuild the compile cache for each release.

## [Unreleased]

### Added

- `/tokenize` and `/detokenize` (also under `/v1`), as SGLang and vLLM serve them. A chat request is counted exactly as
  `/v1/chat/completions` would send it to the engine. The router sends them to any live engine without touching its
  routing state ([docs/API.md](docs/API.md)).
- SGLang's native `/generate` (also PUT): `text` or `input_ids`, one prompt or a batch, SGLang's `sampling_params`,
  streaming, output logprobs and request ids. The router routes it by its prompt or session
  ([docs/API.md](docs/API.md#generate)).
- SGLang's `/model_info`, `/server_info`, `/get_load`, `/health_generate`, `/abort_request` and `/flush_cache` (and
  their `get_` names), on each engine and on the router, which asks one engine or all of them
  ([docs/API.md](docs/API.md#info-and-admin-endpoints)).
- `n` choices on `/v1/chat/completions` and `/v1/completions`, sharing the prompt through the prefix cache; a seed gives
  choice i `seed + i`, as in vLLM ([docs/API.md](docs/API.md#sampling-extras)).
- SGLang's `stop_regex`, and keeping the stop text with `include_stop_str_in_output` (vLLM) or `no_stop_trim` (SGLang).
- `skip_special_tokens: false`, and SGLang's `matched_stop` on every finished choice.
- The OpenAI Responses API, `POST /v1/responses`, stateless: reasoning, message and function-call items, and the
  Responses stream events ([docs/API.md](docs/API.md#responses-api)).
- The Anthropic Messages API, `POST /v1/messages` and `/v1/messages/count_tokens`: thinking, text and tool-use blocks,
  Anthropic's stream events and error shape ([docs/API.md](docs/API.md#anthropic-messages-api)).
- The router routes `/v1/responses` and `/v1/messages` by session and prefix as it routes chat, and sends
  `/v1/messages/count_tokens` to any live engine.

### Changed

- The stop token that ends a request (EOS or a `stop_token_ids` entry) is no longer part of its text when the tokenizer
  does not mark it special, as in vLLM and SGLang.
- `min_new_tokens` (SGLang's name for `min_tokens`) and `spaces_between_special_tokens: false` are refused with 400;
  they were ignored.

## [0.1.0] - 2026-11-08

The first public release.

- GLM-5.3-Flash on one trn2.48xlarge: two engines of 32 tensor-parallel ranks behind a session-affine router, an
  OpenAI-compatible API, 1M-token context, a prefix cache with a host-memory tier, and NKI decode megakernels.
  These describe the configuration, not a measurement; [docs/RESULTS.md](docs/RESULTS.md) has the measured numbers with
  their context, and [docs/DEPLOY.md](docs/DEPLOY.md) the deployment.

### Added

- MTP speculative decoding with GLM-5.3-Flash's own MTP layer: 2 draft tokens a step, verified by the target in the next
  step, scheduled before the previous step is read back. Presets `mtp-always`, `mtp-batch8`, `mtp-batch12` and
  `mtp-batch24` ([docs/CONFIGURATION.md](docs/CONFIGURATION.md)), a dynamic mode that speculates only while few requests
  decode, and `trnrt:spec_decode_*` metrics. A benchmark mode (`--spec-synth-al`) reproduces InferenceX's synthetic
  acceptance; it is off by default ([docs/AGENTX.md](docs/AGENTX.md)).
- `TRNRT_DMA_HW`: the kernels' static DMA helpers in hardware DGE. The `mtp-*` presets set it, as in the builds behind
  the published MTP numbers; the other configurations leave it unset.
- Public release: README, docs, contributing and security policies, issue and pull request templates.
- CI: pre-commit (ruff lint and format), mypy, the CPU test suite, a package build check, Dockerfile lint and CodeQL.
  Release workflow.
- `tools/check_docs.py`: every relative Markdown link and image resolves.
- Branches: pull requests target `development` and are squash-merged; `main` takes promotions of `development` for
  releases only, as merge commits. Both need a maintainer's approval (`.github/CODEOWNERS`) and every review
  conversation resolved. Merging a promotion that bumps the version into `main` is the release: it tags `vX.Y.Z` on the
  merge commit and publishes, once per version (`tools/release_plan.py`).
- Contributor CI: actionlint and zizmor on the workflows (pre-commit); advisory PR checks (title format, a warning
  when traced code changes); dependency review; OpenSSF Scorecard; a weekly external-link check; a release dry run
  that builds the package and the container image.
- Area labels on every PR from the paths it changes (`.github/labeler.yml`), forks included.
- AI review on every ready-for-review PR into `development` (CodeRabbit, cubic, Greptile), never a required check.
- Contributing: a three-step quick start, issue forms for performance regressions and docs, Discussions for questions,
  `SUPPORT.md`.

### Changed

- A request without `temperature`, `top_p`, `top_k` or `min_p` takes the checkpoint's `generation_config.json`
  values, as vLLM does by default. For GLM-5.3-Flash that is top_p 0.95; it was 1.0.
- The display name is TrnRT. The package, the `trnrt` command, environment variables and metric names are unchanged.
- Server, engine and router messages go through Python `logging`, on stderr. Their text is unchanged, and
  `TRNRT_LOG_LEVEL` sets the level. They used to be printed, some on stdout. A hang report still reaches stderr when
  nothing has configured logging. The CLIs' own start-up status lines (`engine up in`, `bucket warmup`,
  `engine N ready after`) are still printed, as is one status line in the frozen `model_runner.py`.
- Code style: the ruff formatter and a strict ruff rule set over all host code, tests and tools, plus mypy with zero
  errors. All enforced in pre-commit and CI. Traced code (kernels, models) is excluded, so no compiled graph changes.

### Fixed

- The router matches and inserts a sessionless request's text in its prefix trees in a worker thread, not on its event
  loop, which every relayed stream shares: a 1.1 MB (~250K-token) request stalled the loop for ~300 ms; now ~12 ms.
  The routing decisions are unchanged.
- A non-streaming client that disconnects from the router now aborts its request on the engine. The router kept its
  own connection to the engine open, so the engine generated to `max_tokens` and held a decode seat.
- The router answers a request body that is not a JSON object with 400, as each engine does, instead of 500.
