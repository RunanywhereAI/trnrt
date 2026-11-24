#!/usr/bin/env bash
# Run one fixed-length point (benchmarks/points.csv, fixed-*) with InferenceX's fixed_seq recipe, verbatim
# (inferencex-e2e/infx/bench/fixed_seq.py's client arguments: infx.bench_serving.benchmark_serving), then score it.
# 8,192 input / 1,024 output tokens with --random-range-ratio 0.8 (inputs uniform in [6,553, 8,192], outputs in [819,
# 1,024]), 10 x C prompts at concurrency C, 2 x C warmups, request rate unlimited, EOS ignored, /v1/completions; the
# chat template only on a speculative build, as InferenceX does for speculative runs. No prefix cache reuse: the prompts
# are random.
#
#   INFX=/src/InferenceX TOKENIZER=/models/GLM-5.3-Flash benchmarks/fixed_seq/client.sh [--dry-run] POINT URL OUT_DIR
#
# Environment: INFX (a checkout of SemiAnalysisAI/InferenceX at commit 47cdc9d02d2f50018efbbc207c0c3ce9e425dae4, the
# one the published points used); PY (default python3: a Python with InferenceX's client requirements); TOKENIZER (the
# model's tokenizer files at revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a); CPUS (optional taskset -c CPUS).
# URL: the router of benchmarks/serve.sh. The published points ran the client on the serving host over localhost, on
# CPUs the engine did not use.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
dry=0
if [ "${1:-}" = "--dry-run" ]; then dry=1; shift; fi
[ $# -eq 3 ] || { echo "usage: INFX=DIR TOKENIZER=DIR $0 [--dry-run] POINT URL OUT_DIR" >&2; exit 2; }
point=$1 url=${2%/} out=$3
P() { python3 -I "$here/../points.py" get "$point" "$1"; }
[ "$(P benchmark)" = fixed_seq ] || { echo "$point is not a fixed_seq point" >&2; exit 2; }
c=$(P concurrency)
build=$(P build)
: "${INFX:?set INFX to the InferenceX checkout}" "${TOKENIZER:?set TOKENIZER to the tokenizer directory}"
PY=${PY:-python3}
WANT_COMMIT=47cdc9d02d2f50018efbbc207c0c3ce9e425dae4
chat=()
case $build in latency | latency-golden) chat=(--use-chat-template) ;; esac

if [ "$dry" = 0 ]; then
  [ ! -e "$out" ] || { echo "$out exists: give each run a new directory" >&2; exit 2; }
  got=$(git -C "$INFX" rev-parse HEAD 2>/dev/null || echo none)
  [ "$got" = "$WANT_COMMIT" ] || { echo "InferenceX at $got; the published points used $WANT_COMMIT" >&2; exit 2; }
  python3 -I "$here/../accuracy/acceptance_mode.py" --expect real --url "$url" || exit 2
  mkdir -p "$out"
  out=$(cd "$out" && pwd)  # absolute: the client runs from the InferenceX checkout
fi

pin=()
[ -z "${CPUS:-}" ] || pin=(taskset -c "$CPUS")
cmd=(${pin[@]+"${pin[@]}"} "$PY" -m infx.bench_serving.benchmark_serving --model glm --backend vllm --base-url "$url"
  --dataset-name random --random-input-len 8192 --random-output-len 1024 --random-range-ratio 0.8
  --num-prompts $((10 * c)) --max-concurrency "$c" --request-rate inf --ignore-eos --save-result
  --num-warmups $((2 * c)) --percentile-metrics "ttft,tpot,itl,e2el"
  --result-dir "$out" --result-filename "c$c.json" --tokenizer "$TOKENIZER" ${chat[@]+"${chat[@]}"})
echo "point $point: 8K/1K at concurrency $c on build $build"
echo "+ (cd $INFX/inferencex-e2e && ${cmd[*]})"
[ "$dry" = 1 ] && exit 0
(cd "$INFX/inferencex-e2e" && "${cmd[@]}") 2>&1 | tee "$out/client.log"
python3 -I "$here/score.py" "$out/c$c.json" --chips 8 --label "$point" | tee "$out/score.json"
