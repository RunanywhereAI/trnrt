#!/usr/bin/env bash
# Build the TrnRT image on a GitHub-hosted runner (release.yml): IMAGE=<name> docker/ci_build_image.sh
#
# Measured on ubuntu-24.04 (PR #20's dry run): 86 GB free before, an 18.2 GB image, about 5 minutes from a cold cache.
# Disk use is printed before and after, so each run records the margin. Writes VERSION to $GITHUB_ENV.
set -euo pipefail
: "${IMAGE:?set IMAGE}" "${GITHUB_SHA:?}" "${GITHUB_ENV:?}"

df -h /
v=$(python3 -c "import ast;print(next(n.value.value for n in ast.parse(open('trnrt/__init__.py').read()).body if isinstance(n, ast.Assign) and n.targets[0].id == '__version__'))")
echo "VERSION=$v" >> "$GITHUB_ENV"
docker build -f docker/Dockerfile \
  --build-arg "TRNRT_VERSION=$v" --build-arg "TRNRT_REVISION=$GITHUB_SHA" \
  -t "$IMAGE:$v" .
docker image ls "$IMAGE"
df -h /
