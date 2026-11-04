"""shell.HOST_SIM, the simulator harness's seam (the served decode loop with its kernels in nki.simulate): a module
attribute, False unless a harness sets it. These tests pin that nothing else can turn it on (no environment variable
reads it; no source outside tests/ assigns it; serving's import path leaves it False), that the guards it opens follow
it (mhc_pre and mhc_mix take their device path, the launch included, for host tensors only while it is set), and that
every host / device guard on the decode path is classified."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import types

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHELL = os.path.join("trnrt", "kernels", "shell.py")
# serving's import path: python -m trnrt, the server, the engine and its runner, the model code and every kernel module
SERVING = (
    "trnrt.__main__",
    "trnrt.serve.api",
    "trnrt.engine.engine",
    "trnrt.engine.model_runner",
    "trnrt.engine.tp",
    "trnrt.models.loader",
    "trnrt.models.decoder",
    "trnrt.models.hybrid",
    "trnrt.models.glm5_next",
)
REQUIRED = ("trnrt.models.decoder", "trnrt.models.hybrid", "trnrt.engine.engine")


def sets_host_sim(src: str, path: str) -> list[int]:
    """The lines of src that assign HOST_SIM (x.HOST_SIM = / +=, a bare HOST_SIM = outside shell.py, or
    setattr(x, "HOST_SIM", ...))."""
    out = []
    for n in ast.walk(ast.parse(src)):
        targets = (
            n.targets
            if isinstance(n, ast.Assign)
            else [n.target]
            if isinstance(n, (ast.AugAssign, ast.AnnAssign))
            else []
        )
        for t in targets:
            if (isinstance(t, ast.Attribute) and t.attr == "HOST_SIM") or (
                isinstance(t, ast.Name) and t.id == "HOST_SIM" and path != SHELL
            ):
                out.append(n.lineno)
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "setattr"
            and len(n.args) >= 2
            and isinstance(n.args[1], ast.Constant)
            and n.args[1].value == "HOST_SIM"
        ):
            out.append(n.lineno)
    return out


def test_shell_defines_it_once_false():
    tree = ast.parse(open(os.path.join(ROOT, SHELL)).read())
    defs = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "HOST_SIM" for t in n.targets)
    ]
    assert len(defs) == 1 and isinstance(defs[0].value, ast.Constant) and defs[0].value.value is False


def test_no_source_outside_tests_sets_it():
    found = []
    for top in ("trnrt", "tools", "docker"):
        for d, _, files in os.walk(os.path.join(ROOT, top)):
            for f in files:
                if f.endswith(".py"):
                    p = os.path.join(d, f)
                    rel = os.path.relpath(p, ROOT)
                    try:
                        src = open(p).read()
                        lines = sets_host_sim(src, rel)
                    except (SyntaxError, UnicodeDecodeError):
                        continue
                    found += [f"{rel}:{ln}" for ln in lines]
    assert found == [], found


def test_the_scan_finds_a_planted_setter():
    """The positive path: each form a harness (or a slip) would use is found."""
    for src in (
        "from trnrt.kernels import shell as SH\nSH.HOST_SIM = True\n",
        "import x\nx.y.HOST_SIM |= 1\n",
        "setattr(SH, 'HOST_SIM', True)\n",
        "HOST_SIM = True\n",
    ):
        assert sets_host_sim(src, "trnrt/models/x.py"), src


def test_serving_import_path_leaves_it_off():
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRNRT_")}
    env.update(PYTHONPATH=ROOT, TRNRT_HOST_SIM="1", HOST_SIM="1", NEURON_PLATFORM_TARGET_OVERRIDE="trn2")
    code = (
        "import importlib, json, os, sys\n"
        "ok, skipped = [], {}\n"
        f"for m in {SERVING!r} + tuple('trnrt.kernels.' + f[:-3] for f in sorted(os.listdir('trnrt/kernels'))\n"
        "                                if f.endswith('.py') and f != '__init__.py'):\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "        ok.append(m)\n"
        "    except ModuleNotFoundError as e:\n"
        "        if (e.name or '').startswith('trnrt'):\n"
        "            raise\n"
        "        skipped[m] = e.name\n"
        "from trnrt.kernels import shell\n"
        "print(json.dumps({'host_sim': shell.HOST_SIM, 'ok': ok, 'skipped': skipped}))\n"
    )
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    import json

    r = json.loads(p.stdout.strip().splitlines()[-1])
    assert r["host_sim"] is False, r
    assert all(m in r["ok"] for m in REQUIRED), r


@pytest.fixture
def fake_launch(monkeypatch):
    """A stand-in libtorch_neuronx_lite.nki.nki_hop whose wrap_nki records each launch (kernel, grid, static ints)."""
    calls = []

    class Launch:
        def __init__(self, k):
            self.k = k

        def __getitem__(self, grid):
            def run(**kw):
                calls.append((self.k, grid, {n: v for n, v in kw.items() if isinstance(v, int)}))
                T, hc = kw["S"].shape[0], kw["hc"]
                if "fn" in kw:  # mhc_pre's (x, post, comb)
                    return torch.zeros(T, kw["S"].shape[1] // hc), torch.zeros(T, hc), torch.zeros(T, hc * hc)
                return torch.zeros(T, kw["S"].shape[1])  # mhc_mix's streams

            return run

    mod = types.ModuleType("libtorch_neuronx_lite.nki.nki_hop")
    mod.wrap_nki = Launch
    for name in ("libtorch_neuronx_lite", "libtorch_neuronx_lite.nki"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "libtorch_neuronx_lite.nki.nki_hop", mod)
    from trnrt import platform

    monkeypatch.setattr(platform, "nki_grid", lambda: 2)
    return calls


def test_mhc_mix_follows_it(monkeypatch, fake_launch):
    from trnrt.kernels import mhc_mix
    from trnrt.kernels import shell as SH

    monkeypatch.setattr(mhc_mix, "mhc_mix_kernel", "the mhc_mix kernel")
    T, hc, H = 2, 4, 256
    g = torch.Generator().manual_seed(0)
    post, comb = torch.rand(T, hc, generator=g), torch.rand(T, hc, hc, generator=g)
    y, S = torch.randn(T, H, generator=g).bfloat16(), torch.randn(T, hc, H, generator=g).bfloat16()
    assert not mhc_mix.takes(S.device, H, T)
    assert torch.equal(mhc_mix.mix(post, comb, y, S), mhc_mix.emulate(post, comb, y, S)) and fake_launch == []
    monkeypatch.setattr(SH, "HOST_SIM", True)
    assert mhc_mix.takes(S.device, H, T)
    mhc_mix.mix(post, comb, y, S)
    assert [(k, g, kw["rev"]) for k, g, kw in fake_launch] == [("the mhc_mix kernel", 2, mhc_mix.REV)], fake_launch


def test_mhc_pre_follows_it(monkeypatch, fake_launch):
    from trnrt.kernels import mhc_pre
    from trnrt.kernels import shell as SH

    monkeypatch.setattr(mhc_pre, "mhc_pre_kernel", "the mhc_pre kernel")
    T, hc, H = 2, 4, 256
    O = 2 * hc + hc * hc
    g = torch.Generator().manual_seed(1)
    S, fn = torch.randn(T, hc, H, generator=g).bfloat16(), (0.02 * torch.randn(O, hc * H, generator=g)).bfloat16()
    base, scale = 0.1 * torch.randn(O, generator=g), 1 + 0.1 * torch.randn(3, generator=g)
    assert not mhc_pre.takes(S.device, T, H, hc, fn.dtype)
    x = mhc_pre.pre(S, fn, base, scale, 20, 1e-6, 1e-5)
    assert len(x) == 3 and fake_launch == []  # the emulation
    monkeypatch.setattr(SH, "HOST_SIM", True)
    assert mhc_pre.takes(S.device, T, H, hc, fn.dtype)
    assert not mhc_pre.takes(S.device, T, H, hc, torch.float32)  # its other conditions still decide
    x, post, comb = mhc_pre.pre(S, fn, base, scale, 20, 1e-6, 1e-5)
    assert x.shape == (T, H) and comb.shape == (T, hc, hc)  # the stand-in launch's outputs, not the emulation's
    assert [(k, g, kw["rev"], kw["skf"]) for k, g, kw in fake_launch] == [
        ("the mhc_pre kernel", 2, mhc_pre.REV, mhc_pre.SK)
    ], fake_launch


# -- every host / device guard on the decode path's files, classified ----------------------------------------------
# (file, function) -> one class per `.device.type ==/!= "cpu"` site in it, in source order:
#   flip: a kernel dispatch the harness must reach under HOST_SIM (the line reads SH.HOST_SIM);
#   host: collective emulation that stays on the host (funcol needs a process group; the identity all-reduce stands in);
#   noop: decides nothing on the served path (a permissive host OR);
#   off:  off the served decode path (prefill chunks, the host's own forms).
SITES = {
    ("trnrt/models/hybrid.py", "_moe_clamped"): ["off"],  # decode's MoE runs in the FFN half
    ("trnrt/models/hybrid.py", "kda_shell_takes"): ["flip"],
    ("trnrt/models/hybrid.py", "ffn_half_takes"): ["flip"],
    ("trnrt/kernels/mhc_pre.py", "pre"): ["flip"],
    ("trnrt/kernels/mhc_mix.py", "mix"): ["flip"],
    ("trnrt/kernels/mhc_front.py", "front"): ["flip"],
    ("trnrt/models/mla.py", "_cp_reduce_scatter"): ["host"],
    ("trnrt/models/mla.py", "_shell_sel_takes"): ["noop"],
    ("trnrt/models/mla.py", "_shell_merge_takes"): ["noop"],
    ("trnrt/models/mla.py", "attention_cp"): ["flip", "flip", "flip"],
    ("trnrt/models/mla.py", "attention_cp_rows"): ["off"],
    ("trnrt/kernels/dsa_index.py", "scores"): ["flip"],
    ("trnrt/kernels/dsa_long_pipe.py", "select"): ["off"],
    ("trnrt/kernels/dsa_long_select.py", "select"): ["off"],
    ("trnrt/kernels/dsa_long_select_t.py", "select"): ["off"],
    ("trnrt/kernels/dsa_shell.py", "attend"): ["flip"],
    ("trnrt/kernels/dsa_shell.py", "select2"): ["flip"],  # sel2 is the served decode selection
    ("trnrt/kernels/dsa_slots_c.py", "attend"): ["off"],
    ("trnrt/kernels/dsa_slots_n.py", "attend"): ["flip"],
    ("trnrt/kernels/dsa_topk.py", "select"): ["flip"],
}
GUARD = re.compile(r'\.device\.type\s*(==|!=)\s*"cpu"')


def guard_sites(src: str) -> dict:
    """{function: [line text, ...]}: every `.device.type ==/!= "cpu"` line, by its innermost enclosing function."""
    tree = ast.parse(src)
    funcs = sorted((n.lineno, n.end_lineno, n.name) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    out = {}
    for i, ln in enumerate(src.splitlines(), 1):
        if GUARD.search(ln):
            fn = [name for a, b, name in funcs if a <= i <= b]
            out.setdefault(fn[-1] if fn else "<module>", []).append(ln)
    return out


def classify_errors(rel: str, src: str, table: dict) -> list[str]:
    bad = []
    got = guard_sites(src)
    for fn, lines in got.items():
        want = table.get((rel, fn))
        if want is None or len(want) != len(lines):
            bad.append(f"{rel}:{fn}: {len(lines)} guard(s), table has {0 if want is None else len(want)} (classify it)")
            continue
        for cls, ln in zip(want, lines):
            if (cls == "flip") != ("HOST_SIM" in ln):
                bad.append(
                    f"{rel}:{fn}: a {cls} site {'without' if cls == 'flip' else 'reading'} HOST_SIM: {ln.strip()}"
                )
    for r, fn in table:
        if r == rel and fn not in got:
            bad.append(f"{rel}:{fn}: listed, but has no guard any more (update the table)")
    return bad


def test_every_guard_is_classified():
    import glob

    files = sorted({r for r, _ in SITES} | set(glob.glob("trnrt/kernels/dsa_*.py", root_dir=ROOT)))
    bad = []
    for rel in files:
        bad += classify_errors(rel, open(os.path.join(ROOT, rel)).read(), SITES)
    assert bad == [], bad


def test_the_classifier_catches_planted_sites():
    """The positive path: an unlisted guard, a flip without HOST_SIM, a host site reading it, a removed guard."""
    t = {("x.py", "f"): ["flip"], ("x.py", "g"): ["host"]}
    ok = 'def f(a):\n    return a.device.type == "cpu" and not SH.HOST_SIM\n\ndef g(a):\n    return a.device.type == "cpu"\n'
    assert classify_errors("x.py", ok, t) == []
    assert classify_errors("x.py", ok + 'def h(a):\n    return a.device.type != "cpu"\n', t)  # unlisted
    assert classify_errors("x.py", ok.replace(" and not SH.HOST_SIM", ""), t)  # a flip lost its HOST_SIM
    assert classify_errors("x.py", ok.replace('a.device.type == "cpu"\n', 'a.device.type == "cpu" or SH.HOST_SIM\n'), t)
    assert classify_errors("x.py", ok.split("\ndef g")[0] + "\n", t)  # g's guard removed: the table is stale
