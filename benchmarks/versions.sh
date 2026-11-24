#!/usr/bin/env bash
# The versions a result depends on, as one JSON object: the image (its id and the TrnRT revision it was built from), the
# Neuron SDK inside it (Python distributions and the runtime's Debian packages), and the host's Neuron driver. Run it on
# the serving host; benchmarks/record.py attaches its output to a result.
#
#   [IMAGE=trnrt] benchmarks/versions.sh > versions.json
set -euo pipefail
IMAGE=${IMAGE:-trnrt}
image_id=$(docker image inspect "$IMAGE" --format '{{.Id}}')
revision=$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' 2>/dev/null || true)
inside=$(docker run --rm --entrypoint python3 "$IMAGE" - <<'PY'
import importlib.metadata as m, json, subprocess
names = ("trnrt", "neuronx-cc", "libtorch-neuronx-lite", "nki", "torch", "transformers")
py = {n: (m.version(n) if any(d.metadata["Name"] == n for d in m.distributions()) else None) for n in names}
try:
    out = subprocess.run(["dpkg-query", "-W", "-f", "${Package} ${Version}\n", "aws-neuronx-*"], capture_output=True, text=True).stdout
    debs = dict(line.split(" ", 1) for line in out.splitlines() if " " in line)
except OSError:
    debs = {}
print(json.dumps({"python": py, "deb": debs}))
PY
)
driver=$(cat /sys/module/neuron/version 2>/dev/null || modinfo -F version neuron 2>/dev/null || echo unknown)
python3 -I - "$image_id" "$revision" "$inside" "$driver" <<'PY'
import json, sys
image_id, revision, inside, driver = sys.argv[1:5]
print(json.dumps({"image": {"id": image_id, "trnrt_revision": revision or None}, "image_contents": json.loads(inside),
                  "host": {"neuron_driver": driver}}, indent=1))
PY
