"""The Neuron SDK inside an image, as one JSON object: its Python distributions and the runtime's Debian packages.

benchmarks/versions.sh runs this inside the container (`docker run -i ... python3 - < image_versions.py`), so it is
stdlib only. It exits nonzero when a distribution is not installed or no aws-neuronx-* package is found, so a result
never cites an empty or partial SDK inventory.
"""

from __future__ import annotations

import importlib.metadata as m
import json
import subprocess
import sys

NAMES = ("trnrt", "neuronx-cc", "libtorch-neuronx-lite", "nki", "torch", "transformers")


def python_versions(version=m.version) -> dict[str, str | None]:
    """Each distribution's version, None where it is not installed (looked up by its name, as pip normalizes it)."""
    out: dict[str, str | None] = {}
    for n in NAMES:
        try:
            out[n] = version(n)
        except m.PackageNotFoundError:
            out[n] = None
    return out


def deb_versions(run=subprocess.run) -> dict[str, str]:
    """The installed aws-neuronx-* Debian packages and their versions; SystemExit if there are none."""
    try:
        r = run(
            ["dpkg-query", "-W", "-f", "${db:Status-Abbrev}\t${Package}\t${Version}\n", "aws-neuronx-*"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as e:
        raise SystemExit(f"image_versions: dpkg-query did not run: {e}") from None
    debs = {}
    for line in r.stdout.splitlines():
        status, name, version = ([*line.split("\t"), "", ""])[:3]
        # "ii": installed. dpkg also lists packages removed with their configuration kept ("rc", with the old version)
        # and ones it only knows of ("un", no version): neither is in the image
        if status.startswith("ii") and name and version.strip():
            debs[name] = version.strip()
    if r.returncode != 0 or not debs:
        why = (r.stderr or "").strip()[:200]
        raise SystemExit(
            f"image_versions: no aws-neuronx-* package installed (dpkg-query status {r.returncode}: {why})"
        )
    return debs


def main() -> int:
    py = python_versions()
    missing = [n for n, v in py.items() if v is None]
    if missing:
        raise SystemExit(f"image_versions: not installed in the image: {', '.join(missing)}")
    print(json.dumps({"python": py, "deb": deb_versions()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
