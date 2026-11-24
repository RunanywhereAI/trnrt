# Changelog

All notable changes to TrnRT are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/),
and versions follow [Semantic Versioning](https://semver.org/). Rebuild the compile cache for each release.

## [Unreleased]

### Added

- `/tokenize` and `/detokenize` (also under `/v1`). A chat request is counted exactly as
  `/v1/chat/completions` would send it to the engine. The router sends them to any live engine without touching its
  routing state ([docs/API.md](docs/API.md)).
- The native `/generate` (also PUT): `text` or `input_ids`, one prompt or a batch, `sampling_params`,
  streaming, output logprobs and request ids. The router routes it by its prompt or session
  ([docs/API.md](docs/API.md#generate)).
- `/model_info`, `/server_info`, `/get_load`, `/health_generate`, `/abort_request` and `/flush_cache` (and
  their `get_` names), on each engine and on the router, which asks one engine or all of them
  ([docs/API.md](docs/API.md#info-and-admin-endpoints)).
- `n` choices on `/v1/chat/completions` and `/v1/completions`, sharing the prompt through the prefix cache; a seed gives
  choice i `seed + i` ([docs/API.md](docs/API.md#sampling-extras)).
- `stop_regex`, and keeping the stop text with `include_stop_str_in_output` or `no_stop_trim`.
- `skip_special_tokens: false`, and `matched_stop` on every finished choice.
- `stop_regex` refuses a pattern whose search can take more than about quadratic time in the text's length. Each search
  reaches back about 2048 characters before the text already searched, so a match must start there or later.
- `no_logs` on `/generate`: the request's record stays out of the production log.
- The OpenAI Responses API, `POST /v1/responses`, stateless: reasoning, message and function-call items, and the
  Responses stream events ([docs/API.md](docs/API.md#responses-api)).
- The Anthropic Messages API, `POST /v1/messages` and `/v1/messages/count_tokens`: thinking, text and tool-use blocks,
  Anthropic's stream events and error shape ([docs/API.md](docs/API.md#anthropic-messages-api)).
- The router routes `/v1/responses` and `/v1/messages` by session and prefix as it routes chat, and sends
  `/v1/messages/count_tokens` to any live engine.
- Scheduling flags, host only and per engine: `--schedule-policy` (`fcfs`, `lpm`, `lof`, `spf`),
  priority scheduling with the request field `priority` and priority preemption, `--max-queued-requests` and
  `--max-queued-tokens` (a full queue answers 503), and `--radix-eviction-policy` (`lru`, `lfu`, `slru`, `priority`,
  `tlru`). The defaults schedule as before ([docs/CONFIGURATION.md](docs/CONFIGURATION.md#scheduling)).
- `GET /v1/stats` adds the engine's live state and request rates, TTFT, decode-speed and end-to-end percentiles over
  recent windows, and `GET /dashboard` draws them; the router serves both for each engine
  ([docs/API.md](docs/API.md#statistics-and-the-dashboard)). The fields the router reads are unchanged.
- `TRNRT_PROD_DIR`: an opt-in on-disk log of request records (counts and times only) and gauges.
- `TRNRT_PROMPT_CACHE_MB`: an opt-in prompt token cache that builds a chat turn's prompt ids from the previous turn's,
  with the same ids as a full encode or none.
- Structured outputs through xgrammar, an optional dependency: `response_format` `json_object` and `json_schema`,
  `guided_*` and `structured_outputs`, and `/generate`'s `json_schema`, `regex` and `ebnf`. Also `min_tokens` and
  `thinking_token_budget`. They need `--structured-outputs`, which adds graphs that take a token mask
  ([docs/API.md](docs/API.md#structured-outputs-penalties-and-thinking-budgets)). Tested on CPU; not yet run on
  Trainium2, and the check that the default graphs are unchanged is pending.
- Repetition, frequency and presence penalties, with `--penalties` and `--penalty-buckets`, which add graphs that take
  penalty inputs ([docs/CONFIGURATION.md](docs/CONFIGURATION.md#structured-outputs-penalties-and-thinking-budgets)).
  Tested on CPU; not yet run on Trainium2.
- Host cache: batched moves (`--host-cache-batch`: a restore's and an eviction's pages in one call, every cache copied
  once per run of consecutive pages), save-once (`--host-cache-save-once`: the ranks of one context-parallel index keep
  one shared copy of a page in /dev/shm instead of one each), ascending page allocation (`--pool-ascending`) and the
  checkpoints' share of the host cache (`--host-cache-ckpt-frac`). Host only; on in the balanced and throughput builds.
  An engine with save-once refuses to start when /dev/shm cannot hold its arenas beside those of engines already running.
- `--no-plain-decode`: no plain decode graph, for a build whose speculation never turns off.
- The runner's side of the DSA selection's row classes: a plain decode batch in class-slot order, and a batch whose rows
  do not fit their classes sent to the next page bucket.
- Closing an engine bounds the tensor-parallel stop (120 s): a rank that never answers is named in the log and the
  workers are terminated, instead of holding the process.

### Changed

- The stop token that ends a request (EOS or a `stop_token_ids` entry) is no longer part of its text when the tokenizer
  does not mark it special.
- `min_new_tokens` on the OpenAI endpoints and `spaces_between_special_tokens: false` are refused with 400;
  they were ignored.
- `service_tier` other than `"auto"` or `"default"` is refused with 400 on `/v1/chat/completions`, `/v1/completions`
  and `/v1/responses`.
- Three builds replace the presets, one per band of expected users per engine: `--preset latency` (1-8 users, MTP
  speculative decoding at every step, no plain decode graph), `--preset balanced` (9-16, decode batches of 8, 16 and 24;
  the default) and `--preset throughput` (17 and up, decode batches of 16 and 24). `--preset auto --users N` (or
  `--users N` alone) picks the build for N users per engine ([docs/CONFIGURATION.md](docs/CONFIGURATION.md)). The
  earlier names still work and run the nearest build, with a deprecation warning: `batch8`, `mtp-batch8` and
  `mtp-always` run `latency`, `batch12` and `mtp-batch12` run `balanced`, `mtp-batch24` runs `throughput`. Every build
  has its own graphs: rebuild the compile cache.
- `TRNRT_DMA_HW=1` under every build.
- An engine whose device listing fails, or whose launch CPU set cannot pin every rank to its NeuronCore's NUMA node,
  refuses to start; `TRNRT_NUMA_PIN_OPTIONAL=1` starts it anyway, with a warning. Rank 0 logs how many ranks pinned.
- Every engine warms every slot of its IO pool, so no slot's buffers are built on a serving call.

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
  values. For GLM-5.3-Flash that is top_p 0.95; it was 1.0.
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
