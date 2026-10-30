# trnRT on the AWS Neuron vLLM inference image: Neuron SDK 2.32 (neuronx-cc 2.27), Python 3.13, Ubuntu 24.04.
#
# The base already holds everything trnRT imports (torch, libtorch-neuronx-lite, nki, neuronx-cc, transformers,
# tokenizers, safetensors, huggingface_hub, fastapi, uvicorn, pydantic, pytest), so this layer installs trnRT alone,
# without dependencies, and checks every Neuron piece before the build may finish (docker/check_image.py).
#
# Build from the root of this repository:
#   docker build -f docker/Dockerfile \
#     --build-arg TRNRT_VERSION=<trnrt/__init__.py __version__> --build-arg TRNRT_REVISION=<git commit> \
#     -t trnrt:<version> .
#
# The base is pinned by digest. Its tag is from the Neuron image gallery
# (gallery.ecr.aws/neuron/pytorch-inference-vllm-neuronx), its digest from the registry for that tag.
ARG BASE_IMAGE=public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.24.0.1.1.0-neuronx-py313-sdk2.32.0-ubuntu24.04
ARG BASE_DIGEST=sha256:be11c204f419a63e2487b2124005156dad091fb9edbfcadf42d81b745e284c12
FROM ${BASE_IMAGE}@${BASE_DIGEST}

ARG BASE_IMAGE
ARG BASE_DIGEST
ARG TRNRT_VERSION
ARG TRNRT_REVISION

LABEL org.opencontainers.image.title="trnRT" \
      org.opencontainers.image.description="GLM-5.3-Flash inference on AWS Trainium2" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${TRNRT_VERSION}" \
      org.opencontainers.image.revision="${TRNRT_REVISION}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      org.opencontainers.image.base.digest="${BASE_DIGEST}"

COPY . /opt/trnrt

# --no-deps: nothing trnRT imports is missing from the base; --no-build-isolation: build with the base's own
# setuptools instead of fetching one; --no-cache-dir: no pip cache in the layer.
RUN test -n "${TRNRT_VERSION}" && test -n "${TRNRT_REVISION}" \
    || { echo "build with --build-arg TRNRT_VERSION=... --build-arg TRNRT_REVISION=..." >&2; exit 1; } \
 && pip install --no-cache-dir --no-deps --no-build-isolation /opt/trnrt \
 && rm -rf /opt/trnrt/build /opt/trnrt/trnrt.egg-info \
 && find /opt/trnrt -name __pycache__ -type d -prune -exec rm -rf {} + \
 && TRNRT_VERSION="${TRNRT_VERSION}" python /opt/trnrt/docker/check_image.py

ENV TRNRT_VERSION=${TRNRT_VERSION}
WORKDIR /opt/trnrt
EXPOSE 8000
# The base's health check probes its own server on 8080; trnRT's router listens on --port (8000 by default) and
# answers GET /health there.
HEALTHCHECK NONE
ENTRYPOINT ["python", "-m", "trnrt"]
CMD ["--help"]
