#!/usr/bin/env bash
# Build the compile cache for one or more builds of benchmarks/builds.csv, by docs/DEPLOY.md step 3 (any x86_64 host
# with Docker; no NeuronCore). Every build's graphs go into one cache under $OUT: each capture adds its build's entries,
# then one `compile` compiles every entry not yet compiled and one `strip` strips them.
#
#   OUT=/data/trnrt benchmarks/compile.sh [--dry-run] BUILD [BUILD ...]      e.g. latency-golden balanced throughput
#
# Environment: OUT (required: the compile output, then copied to the serving host); IMAGE (default trnrt, the image of
# docs/DEPLOY.md step 2); REV (default the checkpoint revision the published numbers were measured on); HF_TOKEN
# (optional, passed into the containers). --dry-run prints the commands without running them.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
dry=0
if [ "${1:-}" = "--dry-run" ]; then dry=1; shift; fi
[ $# -ge 1 ] || { echo "usage: OUT=DIR $0 [--dry-run] BUILD [BUILD ...]" >&2; exit 2; }
: "${OUT:?set OUT to the compile output directory}"
IMAGE=${IMAGE:-trnrt}
REV=${REV:-eb9eb208eb0d988989d07a6a12d0fdeb5f52574a}

run() {  # one step: `python -m trnrt` in a new container, $OUT at the same path, $OUT/nki at the kernels' cache path
  local cmd=(docker run --rm -e HF_TOKEN -v "$OUT:$OUT" -v "$OUT/nki:/var/tmp/nki-intermediate-cache" "$IMAGE" "$@")
  echo "+ ${cmd[*]}"
  [ "$dry" = 1 ] || "${cmd[@]}"
}

args=()
for b in "$@"; do  # every build's engine arguments, checked before anything runs
  a=$(python3 -I "$here/points.py" build-args "$b") || exit 2
  args+=("$a")
done

[ "$dry" = 1 ] || mkdir -p "$OUT/nki" "$OUT/work"
if [ ! -d "$OUT/shapes" ] || [ "$dry" = 1 ]; then
  run compile shapes zai-org/GLM-5.3-Flash --revision "$REV" --out "$OUT"
  run compile decisions --model zai-org/GLM-5.3-Flash --revision "$REV" --out "$OUT"
fi
for i in "${!args[@]}"; do
  echo "# build $(( i + 1 )) of ${#args[@]}: ${*:i+1:1}"
  # shellcheck disable=SC2086  # the engine arguments are words by design
  run compile capture --out "$OUT" -- --model "$OUT/shapes" ${args[$i]}
done
run compile compile --out "$OUT" --work-root "$OUT/work"
run compile strip --out "$OUT"
echo "done: copy $OUT/cache and $OUT/nki to the serving host (docs/DEPLOY.md step 4)"
