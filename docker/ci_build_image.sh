#!/usr/bin/env bash
# Build the TrnRT image on a GitHub-hosted runner (release.yml): IMAGE=<name> docker/ci_build_image.sh
#
# Measured on ubuntu-24.04 (PR #20's dry run): 86 GB free before, an 18.2 GB image that used 18 GB of disk, about
# 5 minutes from a cold cache. GitHub documents only 14 GB of guaranteed storage, so the build stops up front, with a
# clear message, when less than 30 GB is free. Disk use is printed before and after. Writes VERSION to $GITHUB_ENV.
set -euo pipefail
: "${IMAGE:?set IMAGE}" "${GITHUB_SHA:?}" "${GITHUB_ENV:?}"

df -h /
free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$free_gb" -lt 30 ]; then
  echo "::error::only ${free_gb} GB free on /; the image build needs about 18 GB plus headroom (30 GB). Use a larger runner."
  exit 1
fi
v=$(python3 -c "import ast;print(next(n.value.value for n in ast.parse(open('trnrt/__init__.py').read()).body if isinstance(n, ast.Assign) and n.targets[0].id == '__version__'))")
echo "VERSION=$v" >> "$GITHUB_ENV"
docker build -f docker/Dockerfile \
  --build-arg "TRNRT_VERSION=$v" --build-arg "TRNRT_REVISION=$GITHUB_SHA" \
  -t "$IMAGE:$v" .
docker image ls "$IMAGE"
df -h /
