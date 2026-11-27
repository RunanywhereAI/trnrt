#!/usr/bin/env bash
# Serve one benchmark point (benchmarks/points.csv) the way it was measured: ONE engine of 8 chips (`--engines 1`, logical
# NeuronCores 0-31, half a trn2.48xlarge) with the router in front, the point's build, the other half idle. InferenceX's
# rule is one server deployment per concurrency, so start a fresh server for every point.
#
#   CACHE=/data/trnrt/cache NKI=/data/trnrt/nki MODEL=/models/GLM-5.3-Flash benchmarks/serve.sh [--dry-run] POINT
#
# Environment: CACHE, NKI (the compile output of benchmarks/compile.sh, copied here); MODEL (the checkpoint at revision
# eb9eb208eb0d988989d07a6a12d0fdeb5f52574a, docs/DEPLOY.md step 4); IMAGE (default trnrt); PORT (default 8000: the
# router; the engine is on PORT + 1, on 127.0.0.1); DOCKER_ARGS (optional, extra `docker run` arguments); REV (default
# the checkpoint revision the published numbers were measured on). The other arguments are docs/DEPLOY.md step 4's.
# The compile cache does not depend on the weights, so a checkpoint of another revision would serve without an error:
# a checkpoint downloaded by `hf download --local-dir` (DEPLOY.md step 4) records the revision it is, and is refused if
# that is not REV; one without that record is served with a warning that its revision could not be checked. The box layout of
# two engines (the default of `trnrt serve`) is the deployment form; the published points are per 8 chips.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
dry=0
if [ "${1:-}" = "--dry-run" ]; then dry=1; shift; fi
[ $# -eq 1 ] || { echo "usage: $0 [--dry-run] POINT   (python3 -I benchmarks/points.py list)" >&2; exit 2; }
point=$1
args=$(python3 -I "$here/points.py" get "$point" engine_args) || exit 2
: "${CACHE:?set CACHE to the compile cache directory}" "${NKI:?set NKI to the NKI kernel directory}" "${MODEL:?set MODEL to the checkpoint directory}"
IMAGE=${IMAGE:-trnrt}
PORT=${PORT:-8000}
REV=${REV:-eb9eb208eb0d988989d07a6a12d0fdeb5f52574a}
meta="$MODEL/.cache/huggingface/download/config.json.metadata"  # huggingface_hub's record: the commit on its first line
if [ -f "$meta" ]; then
  got=$(head -n 1 "$meta")
  [ "$got" = "$REV" ] || { echo "the checkpoint at $MODEL is revision $got; the published numbers used $REV" >&2; exit 2; }
else
  echo "warning: $MODEL has no Hugging Face download record, so its revision cannot be checked against $REV" >&2
fi

# shellcheck disable=SC2206  # the engine arguments and DOCKER_ARGS are words by design
extra=(${DOCKER_ARGS:-})
# shellcheck disable=SC2206
cmd=(docker run --rm --privileged --network host --ipc=host ${extra[@]+"${extra[@]}"}
  -v "$CACHE:/cache" -v "$NKI:/var/tmp/nki-intermediate-cache" -v "$MODEL:/model"
  -e NEURON_LIBTORCH_CACHE_ROOT=/cache
  "$IMAGE" serve --model /model --port "$PORT" --engines 1 $args)
echo "point $point: $(python3 -I "$here/points.py" get "$point" benchmark) at concurrency $(python3 -I "$here/points.py" get "$point" concurrency), build $(python3 -I "$here/points.py" get "$point" build)"
echo "+ ${cmd[*]}"
[ "$dry" = 1 ] || exec "${cmd[@]}"
