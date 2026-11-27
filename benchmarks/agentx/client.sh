#!/usr/bin/env bash
# Run one AgentX point (benchmarks/points.csv) against a server started by benchmarks/serve.sh, then score it.
# Run it on a separate CPU machine: the harness is CPU-bound, and the requests a minute it reaches depend on the client's
# cores, so report the client's core count next to the numbers (ours: 48 logical CPUs, pinned with taskset).
#
#   TOKENIZER=/models/GLM-5.3-Flash benchmarks/agentx/client.sh [--dry-run] POINT URL OUT_DIR
#
# POINT: an agentx-* point of points.csv (its concurrency is taken from there). URL: the router, http://HOST:8000.
# OUT_DIR: a new directory; the harness's artifacts go to OUT_DIR/artifacts, the score to OUT_DIR/score.json.
# Environment: TOKENIZER (a directory with the model's tokenizer files at revision
# eb9eb208eb0d988989d07a6a12d0fdeb5f52574a; the chat template changed after it); AGENTX (default `agentx`: the
# agentx-harness 1.0.12 command); AGENTX_PYTHON (the Python that agentx-harness is installed in, default python3: the
# version check); CPUS (optional: taskset -c CPUS for the harness); PREFLIGHT_WAIT (default 3600: seconds the preflight
# waits for a server still loading and warming its graphs, as long as `trnrt serve`'s --startup-timeout).
#
# Before the run it checks the server against the point (benchmarks/preflight.py): one engine (the score is per its 8
# chips), of the point's build, and in its acceptance mode on every engine (the benchmark mode at the golden length for
# a latency-golden point, real acceptance for any other); a server still starting is waited for.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
dry=0
if [ "${1:-}" = "--dry-run" ]; then dry=1; shift; fi
[ $# -eq 3 ] || { echo "usage: TOKENIZER=DIR $0 [--dry-run] POINT URL OUT_DIR" >&2; exit 2; }
point=$1 url=${2%/} out=$3
P() { python3 -I "$here/../points.py" get "$point" "$1"; }
[ "$(P benchmark)" = agentx ] || { echo "$point is not an AgentX point" >&2; exit 2; }
conc=$(P concurrency)
build=$(P build)
: "${TOKENIZER:?set TOKENIZER to the tokenizer directory}"
AGENTX=${AGENTX:-agentx}
AGENTX_PYTHON=${AGENTX_PYTHON:-python3}
WANT_VERSION=1.0.12

WAIT=${PREFLIGHT_WAIT:-3600}
pre=(python3 -I "$here/../preflight.py" "$point" "$url" --wait "$WAIT")
if [ "$dry" = 0 ]; then
  [ ! -e "$out" ] || { echo "$out exists: give each run a new directory" >&2; exit 2; }
  v=$("$AGENTX_PYTHON" -c 'import importlib.metadata as m; print(m.version("agentx-harness"))' 2>/dev/null || echo none)
  [ "$v" = "$WANT_VERSION" ] || { echo "agentx-harness $v installed; the published points used $WANT_VERSION" >&2; exit 2; }
  "${pre[@]}" || exit 2
fi

pin=()
[ -z "${CPUS:-}" ] || pin=(taskset -c "$CPUS")
cmd=(${pin[@]+"${pin[@]}"} "$AGENTX" profile --url "$url" --endpoint /v1/chat/completions --model glm
  --tokenizer "$TOKENIZER" --tokenizer-trust-remote-code --max-context-length 1048576 --concurrency "$conc"
  --public-dataset semianalysis_cc_traces_weka_062126 --output-artifact-dir "$out/artifacts")
[ "$dry" = 0 ] || echo "+ ${pre[*]}"
echo "point $point: AgentX at concurrency $conc on build $build"
echo "+ ${cmd[*]}"
[ "$dry" = 1 ] && exit 0
mkdir -p "$out"
"${cmd[@]}" 2>&1 | tee "$out/harness.log"
python3 -I "$here/score.py" "$out/artifacts" --chips 8 --label "$point" | tee "$out/score.json"
