"""Graph equivalence between two AOT captures of the same serving layout.

Two capture roots, each holding `capture/rank<R>.json` (the keys each rank's warmup visited, in visit order) and a
compile cache `cache/neuron/compile_cache/<key>/` (graph.hlo, example_inputs.txt, command.txt, metadata), are paired graph
by graph and compared after normalization. Cache keys differ between two code bases even for identical graphs (a key hashes
the traced FX code, which carries module names), so graphs are paired by position: rank R's i-th visited graph on one side
with rank R's i-th on the other, and each pair's input signature must match.

What is compared, and what is normalized away:
  - HLO: instruction and computation names and ids (ids are renumbered by position), op metadata, the module's stack
    frames, its name and id, and parameter names. Each NKI custom call's backend config is decoded: the kernel's function
    name loses the package prefixes given on the command line, its binary path is dropped, everything else stays. The
    rest of the module must be byte-identical.
  - Kernels: each NKI binary a custom call names (.colz: a header and zstd frames holding the kernel's module JSON, its
    instruction records and binary tables) is decompressed. The module JSON and the instruction records are compared
    with their `debug` entries (source file and line, op names) dropped and package prefixes removed. The binary tables
    are compared for information only: some are byte offsets into the JSON, which move with string lengths.
  - Inputs (example_inputs.txt), compiler arguments (command.txt, absolute paths reduced to file names) and the artifact
    metadata (without its cache key).

    python tools/graph_equiv.py REF_ROOT NEW_ROOT --ref-package NAME [--new-package trnrt] [--json OUT]
                               [--name-map MAP.json] [--hlo-proto-path DIR] [--kernels / --no-kernels]

Exit status 0 iff every pair is equal. --json writes one record per pair (rank, index, both keys, the parts that differ)
and the summary. A name map ({"kernels.old_mod.old_kernel": "kernels.new_mod.new_kernel"}) covers kernels the new tree
renamed beyond the package prefix.
"""

import argparse
import ast
import base64
import binascii
import glob
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys

NKI_TARGET = "AwsNeuronCustomNativeKernel"
_B64 = re.compile(rb"eyJ[A-Za-z0-9+/=]+")
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def hlo_pb2(extra_path=None):
    if extra_path:
        sys.path.insert(0, extra_path)
    try:
        from libtorch_neuronx_lite.pyhlo import hlo_pb2 as m
    except ImportError:
        import hlo_pb2 as m  # a plain directory holding hlo_pb2.py / xla_data_pb2.py
    return m


class Names:
    """Package prefixes to drop from kernel names, plus explicit renames."""

    def __init__(self, packages, name_map=None):
        self.packages = sorted({p for p in packages if p}, key=len, reverse=True)
        self.name_map = name_map or {}
        alt = "|".join(re.escape(p) for p in self.packages)
        self.rx = re.compile(rf"\b(?:{alt})[._]") if alt else None
        self.rxb = re.compile(rf"\b(?:{alt})[._]".encode()) if alt else None

    def name(self, s):
        out = self.rx.sub("", s) if self.rx else s
        return self.name_map.get(out, out)

    def blob(self, b):
        return self.rxb.sub(b"", b) if self.rxb else b


# ---- captures ----

def load_capture(root, cache_subdir):
    ranks = {}
    for p in glob.glob(os.path.join(root, "capture", "rank*.json")):
        with open(p) as f:
            rec = json.load(f)
        ranks[int(rec.get("rank", re.findall(r"\d+", os.path.basename(p))[0]))] = list(rec["keys"])
    if not ranks:
        raise SystemExit(f"{root}: no capture/rank*.json")
    return ranks, os.path.join(root, cache_subdir)


def pair(ref_ranks, new_ranks):
    """[(rank, index, ref_key, new_key)], plus problems (rank sets, lengths, one key paired two ways)."""
    pairs, problems, seen_r, seen_n = [], [], {}, {}
    if set(ref_ranks) != set(new_ranks):
        problems.append(f"rank sets differ: {sorted(ref_ranks)} vs {sorted(new_ranks)}")
    for r in sorted(set(ref_ranks) & set(new_ranks)):
        a, b = ref_ranks[r], new_ranks[r]
        if len(a) != len(b):
            problems.append(f"rank {r}: {len(a)} graphs vs {len(b)}")
        for i, (ka, kb) in enumerate(zip(a, b)):
            if seen_r.setdefault(ka, kb) != kb or seen_n.setdefault(kb, ka) != ka:
                problems.append(f"rank {r} graph {i}: {ka} / {kb} pair differently on another rank")
            pairs.append((r, i, ka, kb))
    return pairs, problems


# ---- HLO ----

def _decode_bc(bc):
    for m in _B64.finditer(bc):
        blob = m.group(0)
        try:
            cfg = json.loads(base64.b64decode(blob + b"=" * (-len(blob) % 4)))
        except (binascii.Error, ValueError):
            continue
        if isinstance(cfg, dict):
            return cfg
    return None


def _norm_kernel_config(cfg, names):
    """The backend config with the kernel's name normalized and its binary path taken out; returns (config, path)."""
    cfg = dict(cfg)
    if "func_name" in cfg:
        cfg["func_name"] = names.name(cfg["func_name"])
    path = None
    klir = cfg.get("klir_binary")
    if isinstance(klir, str):
        try:
            klir = ast.literal_eval(klir)
        except (ValueError, SyntaxError):
            klir = None
    if isinstance(klir, dict):
        klir = dict(klir)
        path = klir.pop("binary", None)
        for k in [k for k in klir if "rev" in k.lower()]:
            klir.pop(k)
        cfg["klir_binary"] = klir
    for k in [k for k in cfg if "rev" == k.lower()]:
        cfg.pop(k)
    return cfg, path


def normalize_hlo(path, names, pb2):
    """(normalized module bytes, [(kernel name, binary path)] in instruction order)."""
    m = pb2.HloModuleProto()
    with open(path, "rb") as f:
        m.ParseFromString(f.read())
    cid = {c.id: i for i, c in enumerate(m.computations)}
    iid, n = {}, 0
    for c in m.computations:
        for ins in c.instructions:
            iid[ins.id] = n
            n += 1
    kernels = []
    entry = m.entry_computation_id
    for fld in ("name", "id", "entry_computation_name", "stack_frame_index", "schedule", "profile_info"):
        try:
            m.ClearField(fld)
        except ValueError:  # a field this proto version does not have
            pass
    m.entry_computation_id = cid.get(entry, -1)
    for i, c in enumerate(m.computations):
        c.name = f"c{i}"
        c.id = i
        c.root_id = iid.get(c.root_id, -1)
        if c.HasField("program_shape"):
            del c.program_shape.parameter_names[:]
        for ins in c.instructions:
            ins.name = ""
            ins.id = iid[ins.id]
            ins.operand_ids[:] = [iid[x] for x in ins.operand_ids]
            ins.control_predecessor_ids[:] = [iid[x] for x in ins.control_predecessor_ids]
            ins.called_computation_ids[:] = [cid[x] for x in ins.called_computation_ids]
            ins.ClearField("metadata")
            for k in list(ins.frontend_attributes.map.keys()):
                ins.frontend_attributes.map[k] = names.name(ins.frontend_attributes.map[k])
            if ins.opcode == "custom-call" and ins.custom_call_target == NKI_TARGET:
                cfg = _decode_bc(ins.backend_config if isinstance(ins.backend_config, bytes) else ins.backend_config.encode())
                if cfg is not None:
                    cfg, kpath = _norm_kernel_config(cfg, names)
                    kernels.append((cfg.get("func_name", "?"), kpath))
                    ins.backend_config = json.dumps(cfg, sort_keys=True).encode()
    if m.HasField("host_program_shape"):
        del m.host_program_shape.parameter_names[:]
    return m.SerializeToString(deterministic=True), kernels


def hlo_diff(a, b, pb2, limit=12):
    from google.protobuf import text_format

    ma, mb = pb2.HloModuleProto(), pb2.HloModuleProto()
    ma.ParseFromString(a)
    mb.ParseFromString(b)
    la, lb = text_format.MessageToString(ma).splitlines(), text_format.MessageToString(mb).splitlines()
    import difflib

    return [l for l in difflib.unified_diff(la, lb, "ref", "new", n=1, lineterm="")][:limit]


# ---- kernels (.colz) ----

def _frame_decoder():
    """A one-frame zstd decoder factory: stdlib compression.zstd (3.14+), else the zstandard module, else None."""
    try:
        from compression import zstd

        return lambda: zstd.ZstdDecompressor()
    except ImportError:
        pass
    try:
        import zstandard

        return lambda: zstandard.ZstdDecompressor().decompressobj()
    except ImportError:
        return None


def _zstd_frames(payload):
    """The decompressed frames of a COLZ payload. The container puts a few bytes of its own between frames, so each frame
    is decoded alone and the next one is looked for after the bytes the decoder consumed."""
    new = _frame_decoder()
    out, i = [], payload.find(_ZSTD_MAGIC)
    if new is not None:
        while i != -1:
            d = new()
            out.append(d.decompress(payload[i:]))
            consumed = len(payload) - i - len(d.unused_data)
            i = payload.find(_ZSTD_MAGIC, i + consumed)
        return out
    # zstd CLI fallback: decode each span between magic numbers; the CLI writes the frame, then rejects the container's
    # trailing bytes ("unsupported format"), which is expected here. A magic number inside compressed data is not handled.
    starts = [m.start() for m in re.finditer(re.escape(_ZSTD_MAGIC), payload)] + [len(payload)]
    for s, e in zip(starts, starts[1:]):
        p = subprocess.run(["zstd", "-d", "-c", "-q"], input=payload[s:e], capture_output=True)
        if p.returncode != 0 and b"unsupported format" not in p.stderr:
            raise ValueError(f"zstd: {p.stderr[:200]!r}")
        out.append(p.stdout)
    return out


def _strip_debug(x):
    if isinstance(x, dict):
        return {k: _strip_debug(v) for k, v in x.items() if k != "debug"}
    if isinstance(x, list):
        return [_strip_debug(v) for v in x]
    return x


def _json_records(data):
    dec, i, n = json.JSONDecoder(), 0, len(data)
    while i < n:
        while i < n and data[i] in " \n\r\t,":
            i += 1
        if i >= n:
            break
        obj, i = dec.raw_decode(data, i)
        yield obj


_KCACHE = {}


def kernel_digest(path, names):
    """{'json': sha of the normalized JSON frames, 'records': count, 'tables': [sha of each binary frame]}."""
    key = (path, tuple(names.packages))
    if key in _KCACHE:
        return _KCACHE[key]
    with open(path, "rb") as f:
        b = f.read()
    if b[:4] != b"COLZ":
        raise ValueError(f"{path}: not a COLZ kernel binary")
    h, recs, tables = hashlib.sha256(), 0, []
    for fr in _zstd_frames(b[16:]):
        if fr[:1] == b"{":
            text = names.blob(fr).decode("utf-8")
            for obj in _json_records(text):
                h.update(json.dumps(_strip_debug(obj), sort_keys=True, separators=(",", ":")).encode())
                recs += 1
        else:
            tables.append(hashlib.sha256(fr).hexdigest()[:16])
    out = {"json": h.hexdigest(), "records": recs, "tables": tables}
    _KCACHE[key] = out
    return out


# ---- the rest ----

def _read(p):
    try:
        with open(p, "rb") as f:
            return f.read()
    except OSError:
        return None


def norm_command(txt):
    if txt is None:
        return None
    try:
        args = shlex.split(txt.decode())
    except ValueError:
        return txt
    def one(a):
        if "/" not in a or a.startswith("--internal"):
            return a
        flag, eq, val = a.rpartition("=") if a.startswith("-") and "=" in a else ("", "", a)
        return f"{flag}{eq}{os.path.basename(val)}"

    return [one(a) for a in args]


def norm_meta(txt):
    if txt is None:
        return None
    try:
        d = json.loads(txt)
    except ValueError:
        return txt
    return {k: v for k, v in d.items() if k != "cache_key"} if isinstance(d, dict) else d


def compare_pair(ref_dir, new_dir, names_ref, names_new, pb2, kernels=True):
    diffs, notes = [], []
    ia, ib = _read(os.path.join(ref_dir, "example_inputs.txt")), _read(os.path.join(new_dir, "example_inputs.txt"))
    if ia != ib:
        diffs.append("inputs")
    ha, ka = normalize_hlo(os.path.join(ref_dir, "graph.hlo"), names_ref, pb2)
    hb, kb = normalize_hlo(os.path.join(new_dir, "graph.hlo"), names_new, pb2)
    if ha != hb:
        diffs.append("hlo")
        notes += hlo_diff(ha, hb, pb2)
    if norm_command(_read(os.path.join(ref_dir, "command.txt"))) != norm_command(_read(os.path.join(new_dir, "command.txt"))):
        diffs.append("command")
    ma = norm_meta(_read(os.path.join(ref_dir, ".artifact_metadata_v0.json")))
    mb = norm_meta(_read(os.path.join(new_dir, ".artifact_metadata_v0.json")))
    if ma != mb:
        diffs.append("metadata")
    kernel_diffs = []
    if kernels and len(ka) == len(kb):
        for (na, pa), (nb, pb) in zip(ka, kb):
            if not pa or not pb:
                kernel_diffs.append(f"{na}: binary path missing")
                continue
            try:
                da, db = kernel_digest(pa, names_ref), kernel_digest(pb, names_new)
            except (OSError, ValueError) as e:
                kernel_diffs.append(f"{na}: {e}")
                continue
            if da["json"] != db["json"] or da["records"] != db["records"]:
                kernel_diffs.append(f"{na}: kernel body differs ({da['records']} vs {db['records']} records)")
            elif da["tables"] != db["tables"]:
                notes.append(f"{na}: binary tables differ (offsets; informational)")
    if kernel_diffs:
        diffs.append("kernels")
        notes += kernel_diffs
    return diffs, notes, [n for n, _ in ka]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ref_root")
    ap.add_argument("new_root")
    ap.add_argument("--ref-package", required=True, help="the reference tree's package name, dropped from kernel names")
    ap.add_argument("--new-package", default="trnrt")
    ap.add_argument("--name-map", help="JSON {normalized reference kernel name: normalized new name}")
    ap.add_argument("--cache-subdir", default="cache/neuron/compile_cache")
    ap.add_argument("--hlo-proto-path", help="a directory holding hlo_pb2 when libtorch_neuronx_lite is not installed")
    ap.add_argument("--no-kernels", dest="kernels", action="store_false", help="skip the kernel-body comparison")
    ap.add_argument("--json", help="write per-pair records and the summary here")
    a = ap.parse_args(argv)
    pb2 = hlo_pb2(a.hlo_proto_path)
    nmap = json.load(open(a.name_map)) if a.name_map else {}
    names_ref, names_new = Names([a.ref_package], nmap), Names([a.new_package])
    ref_ranks, ref_cache = load_capture(a.ref_root, a.cache_subdir)
    new_ranks, new_cache = load_capture(a.new_root, a.cache_subdir)
    pairs, problems = pair(ref_ranks, new_ranks)
    done, records, bad = {}, [], 0
    for r, i, ka, kb in pairs:
        if (ka, kb) not in done:
            done[(ka, kb)] = compare_pair(os.path.join(ref_cache, ka), os.path.join(new_cache, kb), names_ref, names_new,
                                          pb2, a.kernels)
        diffs, notes, kernels = done[(ka, kb)]
        records.append({"rank": r, "index": i, "ref": ka, "new": kb, "diffs": diffs, "notes": notes, "kernels": kernels})
    distinct = len(done)
    differ = sorted({(ka, kb) for (ka, kb), (d, _, _) in done.items() if d})
    for ka, kb in differ:
        d, notes, _ = done[(ka, kb)]
        print(f"DIFFER {ka} / {kb}: {', '.join(d)}")
        for l in notes[:8]:
            print(f"    {l}")
    for p in problems:
        print(f"PROBLEM {p}")
    ok = not differ and not problems
    print(f"GRAPH_EQUIV pairs={len(pairs)} distinct={distinct} differ={len(differ)} problems={len(problems)} "
          f"verdict={'EQUAL' if ok else 'DIFFER'}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"pairs": records, "problems": problems, "distinct": distinct, "differ": [list(x) for x in differ],
                       "verdict": "EQUAL" if ok else "DIFFER"}, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
