#!/usr/bin/env bash
# Build the TrnRT image on a GitHub-hosted runner (release.yml): IMAGE=<name> docker/ci_build_image.sh
#
# The built image is about 27 GB (docs/DEPLOY.md), more than a standard runner's root disk has free. Docker's data
# root moves to the runner's larger temporary disk (/mnt), and preinstalled toolchains TrnRT doesn't use are removed.
# Disk use is printed before and after, so each run records whether the build fits. Writes VERSION to $GITHUB_ENV.
set -euo pipefail
: "${IMAGE:?set IMAGE}" "${GITHUB_SHA:?}" "${GITHUB_ENV:?}"

df -h / /mnt
sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /usr/local/.ghcup /opt/hostedtoolcache/CodeQL
sudo systemctl stop docker
sudo mkdir -p /mnt/docker
echo '{"data-root": "/mnt/docker"}' | sudo tee /etc/docker/daemon.json
sudo systemctl start docker
df -h / /mnt

v=$(python3 -c "import ast;print(next(n.value.value for n in ast.parse(open('trnrt/__init__.py').read()).body if isinstance(n, ast.Assign) and n.targets[0].id == '__version__'))")
echo "VERSION=$v" >> "$GITHUB_ENV"
docker build -f docker/Dockerfile \
  --build-arg "TRNRT_VERSION=$v" --build-arg "TRNRT_REVISION=$GITHUB_SHA" \
  -t "$IMAGE:$v" .
docker image ls "$IMAGE"
df -h / /mnt
sudo du -sh /mnt/docker
