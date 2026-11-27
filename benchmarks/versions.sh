#!/usr/bin/env bash
# The versions a result depends on, as one JSON object: the image (its id and the TrnRT revision it was built from), the
# Neuron SDK inside it (Python distributions and the runtime's Debian packages, benchmarks/image_versions.py), and the
# host's Neuron driver. Run it on the serving host; benchmarks/record.py attaches its output to a result.
#
#   [IMAGE=trnrt] benchmarks/versions.sh > versions.json
set -euo pipefail
IMAGE=${IMAGE:-trnrt}
image_id=$(docker image inspect "$IMAGE" --format '{{.Id}}')
revision=$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' 2>/dev/null || true)
here=$(cd "$(dirname "$0")" && pwd)
# the SDK inside the image (benchmarks/image_versions.py): exits nonzero, and so this script, on a missing distribution
# or no aws-neuronx-* package, so a result never cites an empty inventory
inside=$(docker run --rm -i --entrypoint python3 "$IMAGE" - < "$here/image_versions.py")
driver=$(cat /sys/module/neuron/version 2>/dev/null || modinfo -F version neuron 2>/dev/null || echo unknown)
python3 -I - "$image_id" "$revision" "$inside" "$driver" <<'PY'
import json, sys
image_id, revision, inside, driver = sys.argv[1:5]
print(json.dumps({"image": {"id": image_id, "trnrt_revision": revision or None}, "image_contents": json.loads(inside),
                  "host": {"neuron_driver": driver}}, indent=1))
PY
