"""tools/graph_equiv.py on synthetic captures: a renamed tree compares equal, and a one-constant kernel change makes exactly
the graph that uses the kernel differ (the harness's positive control), as does a changed HLO shape.

Needs the XLA HLO protos (libtorch_neuronx_lite.pyhlo, or a directory holding hlo_pb2.py in HLO_PB2_PATH) and zstd (the
`zstandard` module or the zstd CLI); skipped without them."""

import base64
import json
import os
import shutil
import struct
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import graph_equiv as G  # noqa: E402

try:
    PB2 = G.hlo_pb2(os.environ.get("HLO_PB2_PATH"))
except ImportError:
    PB2 = None
try:
    import zstandard
except ImportError:
    zstandard = None
HAVE_ZSTD = zstandard is not None or shutil.which("zstd") is not None
pytestmark = pytest.mark.skipif(PB2 is None or not HAVE_ZSTD, reason="needs the XLA HLO protos and zstd")
REF = "pkga"  # the reference tree's package name in these tests


def _zstd(b):
    if zstandard is not None:
        return zstandard.ZstdCompressor().compress(b)
    return subprocess.run(["zstd", "-q", "-c"], input=b, capture_output=True, check=True).stdout


def write_colz(path, pkg, eps=1e-6, src="/src/a.py", kname="k1"):
    """A kernel binary shaped like the real ones: module JSON, a binary table, instruction records with debug info."""
    module = {
        "arch": "gen3",
        "functions": [
            {
                "name": f"{pkg}.kernels.{kname}.{pkg}_{kname}_kernel",
                "origin": "NKI",
                "allocations": [{"name": "x_set", "debug": {"tensor_name": "x"}}],
            }
        ],
    }
    recs = [
        {"name": "I-0", "opcode": "TensorScalar", "imm": eps, "debug": {"filename": f"{src}", "lineno": 12}},
        {"name": "I-1", "opcode": "Activation", "func": "exp", "debug": {"filename": f"{src}", "lineno": 30}},
    ]
    frames = [
        json.dumps(module).encode(),
        struct.pack("<4I", 0, len(src), 7, 9),
        b"".join(json.dumps(r).encode() for r in recs),
    ]
    with open(path, "wb") as f:
        f.write(b"COLZ" + struct.pack("<3I", 1, 0, 0) + b"".join(_zstd(fr) for fr in frames))


def write_hlo(path, pkg, kernel_path, dim=8, id_base=100, src="/src/model.py", kname="k1"):
    m = PB2.HloModuleProto(name=f"{pkg}_module.{id_base}", id=id_base, entry_computation_name="main")
    c = m.computations.add(name=f"main.{id_base}", id=id_base + 1)

    def ins(name, opcode, dims, operands=(), **kw):
        i = c.instructions.add(name=f"{name}.{id_base}", opcode=opcode, id=id_base + 10 + len(c.instructions), **kw)
        i.shape.element_type = 11  # F32
        i.shape.dimensions.extend(dims)
        i.operand_ids.extend(operands)
        i.metadata.op_name = f"{pkg}/{name}"
        i.metadata.source_file = src
        i.metadata.source_line = 40 + len(c.instructions)
        return i

    p = ins("p0", "parameter", [dim, 4], parameter_number=0)
    cfg = {
        "func_name": f"{pkg}.kernels.{kname}.{pkg}_{kname}_kernel",
        "platform_target": 3,
        "kernel_format": "colz",
        "klir_binary": str({"binary": kernel_path, "input_names": ["x"], "output_names": ["o"]}),
        "grid": 2,
    }
    k = ins(
        "nki",
        "custom-call",
        [dim, 4],
        [p.id],
        custom_call_target=G.NKI_TARGET,
        backend_config=base64.b64encode(json.dumps(cfg).encode()),
    )
    r = ins("add", "add", [dim, 4], [k.id, p.id])
    c.root_id = r.id
    m.entry_computation_id = c.id
    with open(path, "wb") as f:
        f.write(m.SerializeToString())


def make_capture(root, pkg, keys, eps=1e-6, dim=8, id_base=100, src="/src", kname="k1"):
    """Two graphs per capture: g0 calls the kernel, g1 is plain HLO (it references no kernel binary)."""
    cache = os.path.join(root, "cache", "neuron", "compile_cache")
    kdir = os.path.join(root, "nki")
    os.makedirs(kdir, exist_ok=True)
    kpath = os.path.join(kdir, f"{pkg}.kernels.{kname}.{pkg}_{kname}_kernel_lnc0_c{id_base:x}.colz")
    write_colz(kpath, pkg, eps=eps, src=f"{src}/{pkg}/kernels/{kname}.py", kname=kname)
    for j, key in enumerate(keys):
        d = os.path.join(cache, key)
        os.makedirs(d, exist_ok=True)
        if j == 0:
            write_hlo(
                os.path.join(d, "graph.hlo"),
                pkg,
                kpath,
                dim=dim,
                id_base=id_base,
                src=f"{src}/{pkg}/model.py",
                kname=kname,
            )
        else:
            m = PB2.HloModuleProto(name=f"plain.{id_base}", id=id_base)
            c = m.computations.add(name="main", id=id_base + 1)
            i = c.instructions.add(name=f"p.{id_base}", opcode="parameter", id=id_base + 2, parameter_number=0)
            i.shape.element_type = 11
            i.shape.dimensions.extend([4])
            c.root_id = i.id
            m.entry_computation_id = c.id
            with open(os.path.join(d, "graph.hlo"), "wb") as f:
                f.write(m.SerializeToString())
        with open(os.path.join(d, "example_inputs.txt"), "w") as f:
            f.write(f"Shape: ({dim}, 4) dtype float32\n" if j == 0 else "Shape: (4,) dtype float32\n")
        with open(os.path.join(d, "command.txt"), "w") as f:
            f.write(f"neuronx-cc compile {d}/graph.hlo --target trn2 --output {d}/graph.neff -O2\n")
    os.makedirs(os.path.join(root, "capture"), exist_ok=True)
    with open(os.path.join(root, "capture", "rank0.json"), "w") as f:
        json.dump({"rank": 0, "keys": dict.fromkeys(keys, "captured")}, f)


def run(tmp_path, new_kw, args=()):
    ref, new = str(tmp_path / "ref"), str(tmp_path / "new")
    make_capture(ref, REF, ["a" * 32, "b" * 32])
    make_capture(new, "trnrt", ["c" * 32, "d" * 32], id_base=500, src="/other/checkout", **new_kw)
    out = str(tmp_path / "eq.json")
    rc = G.main(
        [ref, new, "--ref-package", REF, "--json", out, *args]
        + (["--hlo-proto-path", os.environ["HLO_PB2_PATH"]] if os.environ.get("HLO_PB2_PATH") else [])
    )
    return rc, json.load(open(out))


def test_renamed_tree_is_equal(tmp_path):
    rc, rep = run(tmp_path, {})
    assert rc == 0 and rep["verdict"] == "EQUAL", rep
    assert rep["pairs"][0]["kernels"] == ["kernels.k1.k1_kernel"]


def test_a_renamed_kernel_is_equal_under_the_name_map_and_differs_without_it(tmp_path):
    """The new tree renamed kernels.k1.k1_kernel to kernels.k2.k2_kernel: its name is in the HLO's func_name and in the
    kernel binary's module JSON (functions[].name); --name-map must reach both."""
    rc, rep = run(tmp_path, {"kname": "k2"})
    assert rc == 1 and rep["pairs"][0]["diffs"] == ["hlo", "kernels"], rep["pairs"][0]
    nmap = tmp_path / "map.json"
    nmap.write_text(json.dumps({"kernels.k1.k1_kernel": "kernels.k2.k2_kernel"}))
    rc, rep = run(tmp_path / "mapped", {"kname": "k2"}, ["--name-map", str(nmap)])
    assert rc == 0 and rep["verdict"] == "EQUAL", rep


def test_a_kernel_digest_is_cached_per_name_map(tmp_path, monkeypatch):
    """One kernel binary digested under two name maps, in either order: each digest is its own map's (the cache key
    holds the map), and the renamed one is the new tree's digest of the renamed kernel."""
    monkeypatch.setattr(G, "_KCACHE", {})
    path, new = str(tmp_path / "k1.colz"), str(tmp_path / "k2.colz")
    write_colz(path, REF)
    write_colz(new, "trnrt", kname="k2")
    rename = {"kernels.k1.k1_kernel": "kernels.k2.k2_kernel"}
    plain = G.kernel_digest(path, G.Names([REF]))
    mapped = G.kernel_digest(path, G.Names([REF], rename))
    assert mapped["json"] != plain["json"] and mapped == G.kernel_digest(new, G.Names(["trnrt"]))
    monkeypatch.setattr(G, "_KCACHE", {})
    assert G.kernel_digest(path, G.Names([REF], rename)) == mapped  # the other order
    assert G.kernel_digest(path, G.Names([REF])) == plain


def test_one_kernel_constant_differs_exactly_its_graph(tmp_path):
    rc, rep = run(tmp_path, {"eps": 1e-5})
    assert rc == 1
    differ = {(p["ref"], p["new"]) for p in rep["pairs"] if p["diffs"]}
    assert differ == {("a" * 32, "c" * 32)}, rep
    assert rep["pairs"][0]["diffs"] == ["kernels"], rep["pairs"][0]


def test_changed_hlo_shape_differs(tmp_path):
    rc, rep = run(tmp_path, {"dim": 16})
    assert rc == 1
    assert "hlo" in rep["pairs"][0]["diffs"] and "inputs" in rep["pairs"][0]["diffs"], rep["pairs"][0]
    assert not rep["pairs"][1]["diffs"]


def test_pairing_flags_a_length_mismatch(tmp_path):
    ranks_a = {0: ["k1", "k2"], 8: ["k3"]}
    ranks_b = {0: ["n1"], 8: ["n3"]}
    pairs, problems = G.pair(ranks_a, ranks_b)
    assert pairs == [(0, 0, "k1", "n1"), (8, 0, "k3", "n3")]
    assert problems == ["rank 0: 2 graphs vs 1"]


def test_pairing_by_signature_survives_a_reordered_warmup():
    sig = {"k1": b"A", "k2": b"B", "n1": b"B", "n2": b"A"}.get
    pairs, problems = G.pair_by_signature({0: ["k1", "k2"]}, {0: ["n1", "n2"]}, sig, sig)
    assert pairs == [(0, 0, "k1", "n2"), (0, 1, "k2", "n1")] and not problems
    pairs, problems = G.pair_by_signature({0: ["k1"]}, {0: ["n1", "n2"]}, sig, sig)
    assert pairs == [(0, 0, "k1", "n2")] and problems == ["rank 0: 1 graphs only in the new capture"], (pairs, problems)
    pairs, problems = G.pair_by_signature({0: ["k1", "k2"]}, {0: ["n2"]}, sig, sig)
    assert pairs == [(0, 0, "k1", "n2")] and problems == ["rank 0 graph 1 (k2): no graph with its input signature"]
