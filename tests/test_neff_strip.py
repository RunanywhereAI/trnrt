"""trnrt/compile/strip.py on synthetic NEFFs: a 1,024-byte header holding the data size and digests of the data
(the fields strip() finds by value; the md5 twice, as neuronx-cc writes it on trn2), then a gzip-compressed tar
with program members and a debug_info_*.dbg member. No real NEFF is needed."""

from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import tarfile

import pytest

from trnrt.compile import cache, strip

MEMBERS = ("neff.json", "sg00/def.json", "sg00/program.bin", "sg00/debug_info_0.dbg", "debug_info_1.dbg")


def _tar(seed: int, members=MEMBERS) -> bytes:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as tf:
        for name in members:
            data = b"".join(f"{name} {seed} {i} {i * i % 977}\n".encode() for i in range(2000))
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return out.getvalue()


def neff(seed: int = 1, size_fmt: str = "<Q", size_twice: bool = False, members=MEMBERS) -> bytes:
    data = _tar(seed, members)
    head = bytearray(b"NEFF" + bytes(range(1, 60)))  # some header bytes that are no field
    head += struct.pack(size_fmt, len(data))
    if size_twice:
        head += struct.pack(size_fmt, len(data))
    md5 = hashlib.md5(data).digest()
    head += md5 + b"\x00" * 7 + md5  # the data's md5, twice
    head += hashlib.sha256(data).hexdigest().encode()  # and its sha256 as hex
    return bytes(head.ljust(strip.HEADER, b"\x00")) + data


def _members(blob: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(blob[strip.HEADER:]), mode="r:gz") as tf:
        return {m.name: tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}


def test_strip_drops_the_debug_members_and_rewrites_the_header_fields():
    old = neff()
    new, rep = strip.strip(old)
    assert sorted(_members(new)) == ["neff.json", "sg00/def.json", "sg00/program.bin"]
    assert all(_members(new)[k] == v for k, v in _members(old).items() if not k.endswith(".dbg"))
    assert [n for n, _ in rep["removed"]] == ["sg00/debug_info_0.dbg", "debug_info_1.dbg"]
    data = new[strip.HEADER:]
    head = new[:strip.HEADER]
    at = rep["fields"]["data_size"][0]
    assert struct.unpack_from("<Q", head, at)[0] == len(data) and rep["bytes"] == (len(old), len(new))
    md5 = hashlib.md5(data).digest()
    assert head.count(md5) == 2 and rep["fields"]["md5"] == [at + 8, at + 8 + 16 + 7]
    assert hashlib.sha256(data).hexdigest().encode() in head
    assert hashlib.md5(old[strip.HEADER:]).digest() not in head
    # every other header byte as it was
    changed = [i for i in range(strip.HEADER) if head[i] != old[i]]
    fields = set(range(at, at + 8)) | set(range(at + 8, at + 24)) | set(range(at + 31, at + 47)) \
        | set(range(at + 47, at + 47 + 64))
    assert set(changed) <= fields


def test_strip_finds_a_4_byte_data_size():
    new, rep = strip.strip(neff(size_fmt="<I"))
    at = rep["fields"]["data_size"]
    assert at[1] == "<I" and struct.unpack_from("<I", new, at[0])[0] == len(new) - strip.HEADER


def test_strip_refuses_a_header_it_cannot_read():
    with pytest.raises(ValueError, match="appears 2 times"):
        strip.strip(neff(size_twice=True))
    b = bytearray(neff())
    b[63:71] = b"\xff" * 8  # the data size gone
    with pytest.raises(ValueError, match="not in the header"):
        strip.strip(bytes(b))
    with pytest.raises(tarfile.TarError):
        strip.strip(b"\x00" * strip.HEADER + b"not a tar.gz")


def test_a_neff_without_debug_members_comes_back_unchanged():
    once, _ = strip.strip(neff())
    twice, rep = strip.strip(once)
    assert twice == once and rep["removed"] == []
    plain = neff(members=("neff.json", "sg00/program.bin"))
    assert strip.strip(plain)[0] == plain


def entry(root, key: str, blob: bytes, marker: bool = True):
    d = root / key
    d.mkdir(parents=True)
    (d / cache.neff_name(key)).write_bytes(blob)
    (d / "graph.hlo").write_bytes(b"hlo " + key.encode())
    (d / "command.txt").write_text("neuronx-cc compile graph.hlo --target trn2")
    (d / ".artifact_metadata_v0.json").write_text(json.dumps({"cache_key": key}))
    if marker:
        (d / cache.MARKER).write_text(cache.marker_text(len(blob), 1.0))
    return d


def test_strip_cache_strips_in_place_and_keeps_every_entry_a_cache_hit(tmp_path):
    root = tmp_path / "cache" / "neuron" / "compile_cache"
    good = [entry(root, c * 32, neff(i)) for i, c in enumerate("ab")]
    bad_blob = neff(3, size_twice=True)
    bad = entry(root, "c" * 32, bad_blob)
    raw = entry(root, "d" * 32, neff(4), marker=False)  # captured, not compiled: not touched
    before = {d.name: (d / cache.neff_name(d.name)).read_bytes() for d in (*good, bad, raw)}
    seen = []
    s = strip.strip_cache(str(root), workers=2, on_entry=seen.append)
    assert (s["entries"], s["stripped"], s["refused"], s["already"]) == (3, 2, 1, 0)
    assert s["bytes_after"] < s["bytes_before"]
    for d in good:
        blob = (d / cache.neff_name(d.name)).read_bytes()
        assert not [m for m in _members(blob) if m.endswith(".dbg")]
        assert cache.is_complete(str(d))
        assert cache.read_marker(str(d))["neff_size"] == str(len(blob))
        assert struct.unpack_from("<Q", blob, 63)[0] == len(blob) - strip.HEADER
        assert blob[:strip.HEADER].count(hashlib.md5(blob[strip.HEADER:]).digest()) == 2
    assert (bad / cache.neff_name(bad.name)).read_bytes() == bad_blob  # refused: untouched
    assert cache.read_marker(str(bad))["neff_size"] == str(len(bad_blob))
    assert (raw / cache.neff_name(raw.name)).read_bytes() == before[raw.name]
    assert not (raw / cache.MARKER).exists()
    assert {r["key"]: r["status"] for r in seen}["c" * 32] == "refused"
    assert not [p for p in root.rglob("*.strip-*")]  # no temporary file left

    again = strip.strip_cache(str(root))
    assert (again["stripped"], again["already"], again["refused"]) == (0, 2, 1)


def test_an_interrupted_strip_leaves_a_marker_that_the_next_run_corrects(tmp_path):
    root = tmp_path / "compile_cache"
    d = entry(root, "e" * 32, neff(5))
    stripped, _ = strip.strip(neff(5))
    (d / cache.neff_name(d.name)).write_bytes(stripped)  # the NEFF replaced, the marker not yet rewritten
    rep = strip.strip_entry(str(d))
    assert rep["status"] == "already"
    assert cache.read_marker(str(d))["neff_size"] == str(len(stripped))


def test_strip_command_line(tmp_path, capsys):
    from trnrt.compile import farm

    root = tmp_path / "out" / "cache" / "neuron" / "compile_cache"
    for i, c in enumerate("fg"):
        entry(root, c * 32, neff(10 + i))
    assert farm.main(["strip", "--out", str(tmp_path / "out"), "--keys", "f" * 32]) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["stripped"] == 1 and summary["entries"] == 1
    assert farm.main(["strip", "--out", str(tmp_path / "out")]) == 0
    assert all(not [m for m in _members((root / k / cache.neff_name(k)).read_bytes()) if m.endswith(".dbg")]
               for k in os.listdir(root))
