"""kernels/mk_kda_layer.py, the fused KDA + MoE decode layer in its served split form (the KDA shell's call, then one
call of the FFN half's body with the glue after it, _b5), on the host, by its source:
- the second half's kernel takes the FFN half kernel's parameters, in order, plus frev, and traces the FFN half's body
  (ffn_half_pb's, not a copy) with exactly those arguments (dbg 0, rev passed as frev), then _b5 on its partials;
- decode_layer_split builds the FFN half's arguments with the very keywords ffn_half.decode passes its run(), and calls
  the KDA shell as hybrid.layer does (the previous layer's pend in the shell's slot form);
- hybrid hands decode_layer_split the previous layer's raw (post, comb, y);
- layer_takes: the rows split evenly over the programs and H / 128 chunks fit 128 partitions (a program's rows past one
  chunk run the chunked glue, tests/test_mk_b5c.py);
- its REV is one CRC of its source, ffn_half_pb's REV and shell's.
Each pin is shown to fail on a planted source."""

import ast
import importlib
import pathlib
import types
import zlib

from trnrt.kernels import ffn_half_pb as FB
from trnrt.kernels import mk_kda_layer as MK
from trnrt.kernels import shell as SH

KDIR = pathlib.Path(MK.__file__).parent
MSRC = (KDIR / "mk_kda_layer.py").read_text()
FBSRC = (KDIR / "ffn_half_pb.py").read_text()
FHSRC = (KDIR / "ffn_half.py").read_text()
HSRC = (KDIR.parent / "models" / "hybrid.py").read_text()


def _fn(src: str, name: str) -> ast.FunctionDef:
    for f in ast.walk(ast.parse(src)):
        if isinstance(f, ast.FunctionDef) and f.name == name:
            return f
    raise AssertionError(f"no {name}")


def _params(f: ast.FunctionDef) -> list[str]:
    return [a.arg for a in f.args.args]


def split_kernel_errors(fbsrc: str, msrc: str) -> list[str]:
    f = _params(_fn(fbsrc, "ffn_half_kernel"))
    k = _fn(msrc, "mk_ffn_b5_kernel")
    bad = [] if _params(k) == [*f, "frev"] else [f"parameters {_params(k)}, want the FFN half kernel's + frev"]
    calls = [c for c in ast.walk(k) if isinstance(c, ast.Call) and ast.unparse(c.func) == "_ffn_body"]
    if len(calls) != 1:
        return [*bad, f"{len(calls)} _ffn_body calls, want 1"]
    got = [ast.unparse(a) for a in calls[0].args]
    want = [{"dbg": "0", "rev": "frev"}.get(p, p) for p in f]
    if got != want or calls[0].keywords:
        bad.append(f"_ffn_body({', '.join(got)}), want ({', '.join(want)})")
    if "return (fo[0], fo[1], fo[2], _b5(fo[3], fo[4]))" not in ast.unparse(k):
        bad.append("the kernel does not return the body's streams and _b5 of its partials")
    return bad


def test_the_split_kernel_traces_the_ffn_halfs_body_with_its_own_arguments():
    assert split_kernel_errors(FBSRC, MSRC) == []
    assert "_ffn_body = FB.ffn_half_kernel.func" in MSRC  # the served body, not a copy


def _run_keywords(src: str, fn: str) -> dict:
    f = _fn(src, fn)
    c = next(c for c in ast.walk(f) if isinstance(c, ast.Call) and ast.unparse(c.func) in ("run", "FH.run"))
    return {k.arg: ast.unparse(k.value) for k in c.keywords if k.arg is not None}


def decode_args_errors(fhsrc: str, msrc: str) -> list[str]:
    """decode_layer_split's FH.run keywords against ffn_half.decode's: decode_args' `kw = dict(...)` plus decode's
    own (grid), module globals as FH.<name>."""
    d = next(
        s.value
        for s in ast.walk(_fn(fhsrc, "decode_args"))
        if isinstance(s, ast.Assign) and [ast.unparse(t) for t in s.targets] == ["kw"]
    )
    served = {k.arg: ast.unparse(k.value) for k in d.keywords}
    served.update(_run_keywords(fhsrc, "_decode_moe" if "def _decode_moe" in fhsrc else "decode"))
    glob = {
        t.id for s in ast.parse(fhsrc).body if isinstance(s, ast.Assign) for t in s.targets if isinstance(t, ast.Name)
    }
    served = {k: (f"FH.{v}" if v in glob else v) for k, v in served.items()}
    got = _run_keywords(msrc, "decode_layer_split")
    got.pop("build_only", None)
    return [] if got == served else [f"decode_layer_split's FH.run keywords {got}, the served decode's {served}"]


def test_decode_layer_split_passes_the_served_decodes_keywords():
    assert decode_args_errors(FHSRC, MSRC) == []
    src = ast.unparse(_fn(MSRC, "decode_layer_split"))
    assert "pend=None if pend is None else _shell_pend(layer, pend)" in src  # the shell's call, as hybrid.layer's


def test_hybrid_hands_the_split_layer_its_raw_pend():
    """hybrid passes the previous layer's (post, comb, y) to decode_layer_split as is: decode_layer_split puts it into
    the shell's slot form itself."""
    c = next(
        c
        for c in ast.walk(ast.parse(HSRC))
        if isinstance(c, ast.Call) and ast.unparse(c.func) == "MK.decode_layer_split"
    )
    assert {k.arg: ast.unparse(k.value) for k in c.keywords} == {"pend": "pend"}


def test_layer_takes_needs_whole_rows_per_program_and_fits_the_glue():
    """The programs come from the layer's packing (ks_dims[3], decode_layer_split's npg), not the runtime: no runtime
    is configured here. Two packings with different answers, so the rule can't be reading a constant. Rows past one
    chunk of the glue (T > 8 at grid 2) take the split layer too, through the chunked glue; a hidden size of more than
    128 chunks fits no glue."""
    model = types.SimpleNamespace(cfg=types.SimpleNamespace(hidden_size=4096))
    Ts = (1, 2, 4, 6, 8, 10, 12, 16)
    two = types.SimpleNamespace(ks_dims=(4096, 64, 4, 2, 4))
    one = types.SimpleNamespace(ks_dims=(4096, 64, 4, 1, 4))
    assert [MK.layer_takes(model, two, T) for T in Ts] == [False, True, True, True, True, True, True, True]
    assert [MK.layer_takes(model, one, T) for T in Ts] == [True] * len(Ts)
    wide = types.SimpleNamespace(cfg=types.SimpleNamespace(hidden_size=129 * 128))
    assert [MK.layer_takes(wide, two, T) for T in (2, 8)] == [False, False]


def test_rev_is_one_crc_of_its_parts():
    own = SH.kernel_rev(MK.__file__, "# -- the kernel source", "    mk_ffn_b5_kernel = None")
    assert zlib.crc32(repr((own, FB.REV, SH.REV)).encode()) == MK.REV
    assert zlib.crc32(repr((own, FB.REV ^ 1, SH.REV)).encode()) != MK.REV  # ffn_half_pb's edit re-keys it


def test_the_pins_fail_on_planted_sources():
    assert split_kernel_errors(FBSRC, MSRC.replace("o_rb, 0, fin, frev)", "o_rb, 0, fin, rev)", 1))  # outer rev
    assert split_kernel_errors(
        FBSRC, MSRC.replace("hc, iters, eps, rms_eps, sig, skf,\n", "hc, iters, rms_eps, eps, sig, skf,\n", 1)
    )
    assert split_kernel_errors(FBSRC.replace("fin: int, rev: int):", "fin: int, rev: int, x: int):", 1), MSRC)
    assert decode_args_errors(FHSRC, MSRC.replace("ring=FH.RING,", "ring=3,", 1))
    assert decode_args_errors(FHSRC, MSRC.replace("rsp=MH.RSPLIT,", "rsp=0,", 1))
    importlib.import_module("trnrt.kernels.mk_kda_layer")  # still importable (the plants are text only)
