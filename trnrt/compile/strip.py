"""Strip the compiler's debug info from NEFFs, in place in a compile cache.

Why: a rank's host memory under the Neuron runtime grows with every NEFF it loads, at many times the
NEFF's bytes, and a large share of those bytes is the compiler's backend debug info (the
debug_info_*.dbg members), which serving never reads. With two engines of 32 ranks each on one host and graphs built
for 1M-token buckets, unstripped NEFFs do not fit in host RAM; stripped ones do.

A NEFF is a 1,024-byte header followed by a gzip-compressed tar. strip() rewrites the tar without
the members matching `drop` (default *.dbg) and patches the header. The header's layout is not
documented ("a 1024-byte header ... records the tool version, header_size, data_size and the NEFF
major/minor version"), so its fields are found by VALUE, never by offset:

- the data size (the bytes after the header) as a little-endian integer, 8 or 4 bytes: it must
  appear exactly once, or the NEFF is refused and nothing is written;
- any md5 / sha1 / sha256 of the data, raw or hex: every occurrence is rewritten. neuronx-cc stores
  the data's md5 more than once on trn2; a full digest of the data found at several offsets is that
  digest every time. The runtime checks it only with NEURON_RT_VALIDATE_HASH set, but it is kept
  correct anyway.

strip_cache() strips every compiled entry of a cache directory in place and keeps it an LNL cache
hit throughout: the stripped NEFF replaces the original atomically (written beside it, then
renamed), and the completion marker is rewritten after it with the new neff_size. A NEFF that holds
no debug members is left as it is, so running it twice changes nothing.
"""

from __future__ import annotations

import fnmatch
import hashlib
import io
import os
import struct
import tarfile
from typing import Any

from . import cache

HEADER = 1024
DROP = "*.dbg"


def _find_all(buf: bytes, needle: bytes) -> list[int]:
    out, i = [], buf.find(needle)
    while i >= 0:
        out.append(i)
        i = buf.find(needle, i + 1)
    return out


def strip(neff: bytes, drop: str = DROP) -> tuple[bytes, dict]:
    """(the new NEFF, a report) for one NEFF's bytes. A NEFF with no member matching `drop` comes
    back unchanged. Raises ValueError when the data size is not in the header exactly once, and
    tarfile.TarError when the data is not a gzip-compressed tar."""
    head, data = bytearray(neff[:HEADER]), neff[HEADER:]
    out = io.BytesIO()
    removed = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as src, tarfile.open(fileobj=out, mode="w:gz") as dst:
        for m in src.getmembers():
            if m.isfile() and fnmatch.fnmatch(os.path.basename(m.name), drop):
                removed.append((m.name, m.size))
                continue
            dst.addfile(m, src.extractfile(m) if m.isfile() else None)
    if not removed:
        return neff, {"removed": [], "removed_bytes": 0, "fields": {}, "bytes": (len(neff), len(neff))}
    new = out.getvalue()
    fields: dict[str, Any] = {}
    for fmt in ("<Q", "<I"):  # the data size, 8 or 4 bytes
        if len(data) >= 2 ** (8 * struct.calcsize(fmt)):
            continue
        at = _find_all(bytes(head), struct.pack(fmt, len(data)))
        if at:
            if len(at) != 1:
                raise ValueError(f"the data size {len(data)} appears {len(at)} times in the header as {fmt}")
            head[at[0] : at[0] + struct.calcsize(fmt)] = struct.pack(fmt, len(new))
            fields["data_size"] = (at[0], fmt, len(data), len(new))
            break
    if "data_size" not in fields:
        raise ValueError(
            f"the data size {len(data)} is not in the header (as <Q or <I): its layout is not the one "
            "assumed; nothing written"
        )
    for name in ("md5", "sha1", "sha256"):
        old, rep = hashlib.new(name, data), hashlib.new(name, new)
        for a, b in ((old.digest(), rep.digest()), (old.hexdigest().encode(), rep.hexdigest().encode())):
            at = _find_all(bytes(head), a)
            for i in at:
                head[i : i + len(a)] = b
            if at:
                fields[name] = at if len(at) > 1 else at[0]
    return bytes(head) + new, {
        "removed": removed,
        "removed_bytes": sum(s for _, s in removed),
        "fields": fields,
        "bytes": (len(neff), HEADER + len(new)),
    }


def strip_entry(entry: str, drop: str = DROP) -> dict:
    """Strip one compiled cache entry's NEFF in place. status: "stripped", "already" (no debug
    members; the marker's neff_size is corrected if a run was interrupted between the NEFF and the
    marker), "refused" (strip raised: the entry is untouched) or "skipped" (not a complete entry)."""
    key = cache.key_of(entry)
    if not cache.is_complete(entry):
        return {"key": key, "status": "skipped", "reason": "not a complete cache entry"}
    path = cache.neff_path(entry)
    with open(path, "rb") as f:
        neff = f.read()
    try:
        new, rep = strip(neff, drop)
    except (ValueError, tarfile.TarError, EOFError, OSError) as err:
        return {"key": key, "status": "refused", "error": f"{type(err).__name__}: {err}"}
    if not rep["removed"]:
        if cache.read_marker(entry).get("neff_size") != str(len(neff)):
            cache.write_marker(entry, len(neff))
        return {"key": key, "status": "already", "bytes": len(neff)}
    tmp = f"{path}.strip-{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(new)
    os.replace(tmp, path)
    cache.write_marker(entry, len(new))  # after the NEFF: the marker describes the NEFF in place
    return {
        "key": key,
        "status": "stripped",
        "bytes": rep["bytes"],
        "removed": len(rep["removed"]),
        "removed_mb": round(rep["removed_bytes"] / 1e6, 1),
        "fields": rep["fields"],
    }


def strip_cache(root: str, keys=None, workers: int = 4, drop: str = DROP, on_entry=None) -> dict:
    """strip_entry over every compiled entry under `root` (a compile_cache directory; cache.local_root),
    or only those named in `keys`, `workers` at once (zlib releases the GIL, so threads run in
    parallel). Returns counts by status and the NEFF bytes before and after."""
    from concurrent.futures import ThreadPoolExecutor

    todo = [
        d
        for d in cache.entries(root)
        if os.path.exists(os.path.join(d, cache.MARKER)) and (keys is None or cache.key_of(d) in keys)
    ]
    counts = {
        "entries": len(todo),
        "stripped": 0,
        "already": 0,
        "refused": 0,
        "skipped": 0,
        "bytes_before": 0,
        "bytes_after": 0,
    }
    with ThreadPoolExecutor(max(1, workers)) as pool:
        for rep in pool.map(lambda d: strip_entry(d, drop), todo):
            counts[rep["status"]] += 1
            if rep["status"] == "stripped":
                counts["bytes_before"] += rep["bytes"][0]
                counts["bytes_after"] += rep["bytes"][1]
            elif rep["status"] == "already":
                counts["bytes_before"] += rep["bytes"]
                counts["bytes_after"] += rep["bytes"]
            if on_entry is not None:
                on_entry(rep)
    return counts
