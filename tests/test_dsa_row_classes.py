"""Static row classes and form W (kernels/dsa_sel2gc.py, kernels/dsa_satwfcga.py, kernels/dsa_satwf.py's dispatch;
EngineConfig.dsa_row_classes / dsa_row_classes_cw / dsa_rowsplit_layout).

On the host: both generated kernels are their generators' output for the current sources; the layout specs parse and
refuse what the kernels can't take; the kernel's module ints round-trip; each batch size's revision term is its own;
classed_ppg puts slot s's page p NG_s + g at [p, s, g]; fits / takes / select2gc guard the classed call; form W's row
groups cover the batch once and fit the full layout; the fused call's dispatch sends exactly the classed calls to form W
with the rank's row group; EngineConfig refuses inconsistent fields; set_decode_routes sets every layout.
In the NKI simulator: the classed selection is sel2g's bit for bit (with and without CW, grids 1 / 2, stacked and not), a
planted partial chunk and an unclassed ppg differ; form W's S2 consumers give the row group sum's gv / gc / rows bit for
bit, its send canonicalizes -0.0, and swapped runs or a raw gather differ."""

from __future__ import annotations

import importlib.util
import os
import pathlib

import numpy as np
import pytest
import torch

from tests.test_dsa_shell import _bits, case
from trnrt.config import EngineConfig
from trnrt.kernels import dsa_satwfcga as W
from trnrt.kernels import dsa_sel2gc as C
from trnrt.kernels import dsa_shell as dsh

pytestmark = pytest.mark.usefixtures("decode_routes")  # set_decode_routes's writes are undone

needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
ROOT = pathlib.Path(__file__).resolve().parents[1]
PPP = dsh.PPP
CLASSES = {"9-16": "24:6x24+6x8+12x4,16:4x24+12x8", "17+": "24:2x24+2x16+4x12+16x6"}
FORM_W = {"9-16": "16:2x24+2x8,24:2x24+4x8", "17+": "24:2x24+4x6"}


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def layouts(monkeypatch):
    """Whatever a test sets in the two modules' layouts (and the guard, CW) is undone after it."""
    for mod, names in ((C, ("LAYOUTS", "_LAYOUT_CRC", "RUNNER", "CW", *C._RC_NAMES)), (W, ("W_LAYOUTS", "_W_CRC"))):
        for n in names:
            monkeypatch.setattr(mod, n, getattr(mod, n))


def _use_layouts(monkeypatch, lay):
    """LAYOUTS = lay for the host functions and the kernel's module ints, without parse()'s order rule (a test layout)."""
    monkeypatch.setattr(C, "LAYOUTS", dict(lay))
    for n, v in zip(C._RC_NAMES, C._rc_ints(lay), strict=True):
        monkeypatch.setattr(C, n, v)


# -- the generated files ------------------------------------------------------------------------------------------------


def test_dsa_sel2gc_is_the_transform_of_dsa_sel2g():
    m = _tool("make_sel2gc")
    with open(m.SRC) as f:
        want = m.transform(f.read())
    with open(m.DST) as f:
        assert f.read() == want, "trnrt/kernels/dsa_sel2gc.py is stale: run python tools/make_sel2gc.py"


def test_dsa_satwfcga_is_the_transform_of_dsa_satwfc():
    m = _tool("make_satwfcga")
    with open(m.SRC) as f:
        want = m.generate(f.read())
    with open(m.DST) as f:
        assert f.read() == want, "trnrt/kernels/dsa_satwfcga.py is stale: run python tools/make_satwfcga.py"


def test_the_generators_refuse_a_source_they_dont_match():
    """An edit that moves an anchor fails the generator instead of producing a copy that drifted."""
    m = _tool("make_satwfcga")
    with open(m.SRC) as f:
        src = f.read()
    with pytest.raises(ValueError, match="occurrences"):
        m.generate(
            src.replace("_s2_payload(out, rank, B, keep, A, s2src, pq)", "_s2_payload(out, rank, B, keep, A, s2src)")
        )
    g = _tool("make_sel2gc")
    with open(g.SRC) as f:
        src = f.read()
    with pytest.raises(ValueError, match="expected once"):
        g.transform(src.replace("offset=bb[r] * NG))", "offset=bb[r] * NG ))", 1))


# -- the layouts ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "want"),
    [
        ("", {}),
        ("4:2x8+2x4", {4: (8, 8, 4, 4)}),
        (CLASSES["9-16"], {24: (24,) * 6 + (8,) * 6 + (4,) * 12, 16: (24,) * 4 + (8,) * 12}),
        (CLASSES["17+"], {24: (24,) * 2 + (16,) * 2 + (12,) * 4 + (6,) * 16}),
        (" 4:2x8+2x4 , ", {4: (8, 8, 4, 4)}),
    ],
)
def test_parse(spec, want):
    assert C.parse(spec) == want


@pytest.mark.parametrize("spec", ["4:3x8+1x4", "4:2x8+2x0", "4:2x8+4x4", "6:2x8+2x4", "24:6x24+6x8+11x4", "4:2x4+2x8"])
def test_parse_refuses(spec):
    with pytest.raises(ValueError, match="row classes"):
        C.parse(spec)


@pytest.mark.parametrize("spec", ["4:2x8+2x4,4:4x4", "4:2x8+2x4, 4:2x8+2x4", "4:4x4,16:4x24+12x8,04:2x8+2x4"])
def test_parse_refuses_a_batch_size_given_twice(spec):
    """A repeated batch size would replace the first layout without a word (the last one won)."""
    with pytest.raises(ValueError, match="batch size 4 has a layout already"):
        C.parse(spec)


def test_parse_takes_a_class_up_to_the_kernels_score_tile():
    """The kernel holds a score tile of at most 512 columns (NTM = NG PPP <= 512): NG 64 at PPP 8 parses, 65 does not."""
    widest = 512 // PPP
    assert widest * PPP == 512
    assert C.parse(f"4:2x{widest}+2x4") == {4: (widest, widest, 4, 4)}
    for spec in (f"4:2x{widest + 1}+2x4", f"4:2x{widest}+2x{widest + 1}", f"16:16x{widest + 1}"):
        with pytest.raises(ValueError, match="above the kernel's 512"):
            C.parse(spec)


def test_engine_config_refuses_a_class_wider_than_the_score_tile():
    """The case that passed every other check: a 65-group class at a page bucket of its own, below the largest."""
    with pytest.raises(ValueError, match="above the kernel's 512"):
        EngineConfig(
            model_path="m",
            dsa_row_classes="16:16x65",
            dsa_rowsplit_layout="16:2x65+2x65",
            decode_batch_buckets=(16,),
            page_buckets=(8320, 8448),
        )


def _decode_ints(ints):
    """The test's own reading of _rc_ints' tuple (as _rc_layout reads it): {B: [NG per slot]}."""
    out = {}
    for k in range(2):
        b, runs = ints[9 * k], ints[9 * k + 1 : 9 * k + 9]
        if b:
            out[b] = [g for j in range(4) for g in [runs[2 * j + 1]] * runs[2 * j]]
    return out


@pytest.mark.parametrize("band", sorted(CLASSES))
def test_rc_ints_round_trip(band):
    lay = C.parse(CLASSES[band])
    assert _decode_ints(C._rc_ints(lay)) == {b: list(v) for b, v in lay.items()}


@pytest.mark.parametrize("lay", [{4: (8, 8, 4, 4), 6: (8,) * 6, 8: (8,) * 8}, {10: (10, 10, 8, 8, 6, 6, 4, 4, 2, 2)}])
def test_rc_ints_refuses_what_the_kernel_cant_read(lay):
    with pytest.raises(ValueError, match="at most 2 batch sizes"):
        C._rc_ints(lay)


def test_set_layouts_sets_the_ints_and_each_batch_sizes_own_term(layouts):
    C.set_layouts(CLASSES["9-16"])
    assert C.parse(CLASSES["9-16"]) == C.LAYOUTS
    assert _decode_ints(tuple(getattr(C, n) for n in C._RC_NAMES)) == {b: list(v) for b, v in C.LAYOUTS.items()}
    t24 = C.layout_crc(24)
    C.set_layouts("24:6x24+6x8+12x4")  # B 16's layout gone: B 24's term (its keys) unchanged
    assert C.layout_crc(24) == t24 and C.layout_crc(16) == 0
    C.set_layouts("24:6x24+6x8+10x4+2x2")
    assert C.layout_crc(24) != t24
    assert C.revgc(24) == C.REVGC_BASE ^ C.layout_crc(24)
    C.RUNNER = "test"
    C.set_layouts("")
    assert C.LAYOUTS == {} and C.RCB0 == 0 and C.layout_crc(24) == 0
    assert C.RUNNER is None  # new layouts: the runner built for them installs its guard again


@pytest.mark.parametrize("ngs", [(8, 8, 4, 4), (4, 4, 8, 8), (8, 8, 2, 2, 4, 4), (24,) * 6 + (8,) * 6 + (4,) * 12])
def test_classed_ppg(ngs):
    from trnrt.kernels import dsa_index

    B, NGM = len(ngs), max(ngs)
    t = torch.randint(0, 1 << 20, (B, 128 * NGM), dtype=torch.int32)
    got, base = C.classed_ppg(t, ngs), dsa_index.page_groups(t)
    assert got.shape == (128, B, NGM) and got.dtype == torch.int32 and got.is_contiguous()
    p = torch.arange(128).view(128, 1)
    for s, g in enumerate(ngs):
        gi = torch.arange(g).view(1, g)
        assert torch.equal(got[:, s, :g], t[s][(p * g + gi).reshape(-1)].view(128, g)), s  # page p NG_s + g
        assert torch.equal(got[:, s, g:], base[:, s, g:]), s  # unread: page_groups'


def test_pool_index_is_layout_free():
    """A slot's tile entry (p, c), c = g PPP + j, is pool (p NG_s + g) PPP + j = p NT_s + c at every NG_s: the
    selection's tie order and the compaction's pool ids are the row's own, whatever the class."""
    for NG in (4, 6, 8, 12, 16, 24, 32):
        NT = NG * PPP
        for p in (0, 1, 77, 127):
            for c in (0, 5, NT - 1):
                g, j = divmod(c, PPP)
                assert (p * NG + g) * PPP + j == p * NT + c


def test_fits():
    ngs = (8, 8, 4, 4)
    cap = [128 * g * PPP for g in ngs]
    assert C.fits(torch.tensor(cap), ngs) and C.fits(torch.tensor([0, 1, 0, 0]), ngs)
    for s in range(4):
        n = list(cap)
        n[s] += 1
        assert not C.fits(torch.tensor(n), ngs), s


def test_takes_needs_the_runner_and_the_layouts_bucket(layouts):
    C.set_layouts(CLASSES["9-16"])
    C.RUNNER = None
    assert not C.takes(24, torch.zeros(24, 3072))
    C.RUNNER = "test"
    shapes = ((24, 3072), (16, 3072), (24, 4096), (24, 1024), (12, 3072), (16, 2048))
    assert [C.takes(r, torch.zeros(r, n)) for r, n in shapes] == [True, True, False, False, False, False]


def test_select2gc_refuses(monkeypatch, layouts):
    _use_layouts(monkeypatch, {4: (8, 8, 4, 4)})
    q, w, pkc, table, nl_ = case(4, 1024, 8, [8192, 100, 4096, 4097], seed=7)
    r = torch.tensor([3])
    with pytest.raises(ValueError, match="pass its slot"):
        C.select2gc(q, w, pkc, table, nl_, 128, 8, r)  # slot 3: 4,097 pools in a 4,096-pool class
    with pytest.raises(ValueError, match="no class layout"):
        C.select2gc(q, w, pkc, table[:, :512], nl_.clamp(max=4096), 128, 8, r)  # not the layout's bucket


def test_select2gc_on_the_host_is_the_classless_selection(monkeypatch, layouts):
    """Rows that fit their slots: the classes change no bit (the host runs the emulation either way)."""
    _use_layouts(monkeypatch, {4: (8, 8, 4, 4)})
    q, w, pkc, table, nl_ = case(4, 1024, 8, [8189, 6000, 4096, 90], seed=9)
    got = C.select2gc(q, w, pkc, table, nl_, 128, 8, torch.tensor([3]))
    want = dsh.emulate(q, w, pkc, table, nl_, 128, 8, 3)[:3]
    assert all(torch.equal(x, y) for x, y in zip(got, want, strict=True))


def test_the_built_constant_is_the_masks_bits():
    """CW: (pool < nloc) - 1 in {+0.0, -1.0} times 1e30 is the mask's +0.0 / NEG_INF bit for bit."""
    neg = np.float32(1e30) * np.float32(-1.0)
    assert neg.view(np.uint32) == np.float32(-1e30).view(np.uint32) == np.uint32(0xF149F2CA)
    assert (np.float32(0.0) * np.float32(1e30)).view(np.uint32) == 0  # +0.0, as the mask's (not -0.0)
    assert dsh.NEG_INF == -1e30


# -- form W's layouts -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("band", sorted(FORM_W))
def test_form_w_rows_cover_the_batch_once_and_fit(band):
    full, w = C.parse(CLASSES[band]), W.parse_w(FORM_W[band])
    W.check_layouts(w, full)
    for T in w:
        seen = []
        for g in range(W.RG):
            ra, na, rb, nb = W.w_slots(T, g, w)
            seen += list(range(ra, ra + na)) + list(range(rb, rb + nb))
        assert sorted(seen) == list(range(T)), (band, T)


@pytest.mark.parametrize(
    ("full", "w", "why"),
    [
        ("24:6x24+6x8+12x4", "16:2x24+2x8", "no row-class layout"),
        ("24:6x24+6x8+12x4", "24:2x16+4x8", "widest class"),
        ("24:6x24+6x8+12x4", "24:2x24+4x4", "don't fit"),  # rows 8 .. 11 are NG 8 in the full layout
        ("24:6x24+6x8+12x4", "24:2x24+2x8", "groups of"),
    ],
)
def test_form_w_refuses_a_layout_that_does_not_fit(full, w, why):
    with pytest.raises(ValueError, match=why):
        W.check_layouts(W.parse_w(w), C.parse(full))


@pytest.mark.parametrize("spec", ["24:2x24", "24:1x24+5x6", "24:2x24+4x6+2x2"])
def test_parse_w_refuses(spec):
    with pytest.raises(ValueError, match="two runs"):
        W.parse_w(spec)


def test_w_args_take_the_row_groups_rows(layouts):
    C.set_layouts(CLASSES["17+"])
    W.set_layouts(FORM_W["17+"])
    T, Hi, P = 24, 4, 3072 * PPP
    sargs = {
        "qT": torch.arange(dsh.D * T * Hi, dtype=torch.float32).view(dsh.D, T * Hi),
        "wb": torch.arange(T, dtype=torch.float32).view(T, 1, 1).expand(T, 128, 4 * Hi),
        "cand": torch.arange(T, dtype=torch.float32).view(T, 1).expand(T, P),
        "table": torch.arange(T * 3072, dtype=torch.int32).view(T, 3072),
    }
    q = torch.zeros(T, Hi, dsh.D)
    for g in range(W.RG):
        a = W._w_args(sargs, q, T, g)
        ra, na, rb, nb = W.w_slots(T, g)
        rows = list(range(ra, ra + na)) + list(range(rb, rb + nb))
        assert a["wb"][:, 0, 0].tolist() == rows and a["cand"][:, 0].tolist() == rows
        cols = [r * Hi + h for r in rows for h in range(Hi)]
        assert torch.equal(a["qT"], sargs["qT"][:, cols])
        assert torch.equal(a["ppg"], C.classed_ppg(sargs["table"][rows], (24,) * na + (6,) * nb))
        assert (a["ra"], a["na"], a["rb"], a["nb"], a["qa"], a["qb"]) == (ra, na, rb, nb, 24, 6)


def test_set_layouts_needs_the_row_classes(layouts):
    C.set_layouts("")
    with pytest.raises(ValueError, match="no row-class layout"):
        W.set_layouts(FORM_W["17+"])


# -- the fused call's dispatch ----------------------------------------------------------------------------------------------


def _args(T: int, pages: int):
    """dsa_satwf.satwf's positional arguments as models/mla.py passes them: q, w, pkc, table, nloc, positions, q_all,
    kc, w_uv_p, o_p, ident, keep, A (12), rank, world (14), scale."""
    return (torch.zeros(T, 1), 1, 2, torch.zeros(T, pages), 4, 5, 6, 7, 8, 9, 10, 512, 8, 13, 32, 0.1)


@pytest.fixture
def routes(monkeypatch, layouts):
    from trnrt.kernels import dsa_satwf, mhc_front

    seen = []
    monkeypatch.setattr(W, "satwf", lambda *a, **k: seen.append(("w", k.get("rowgroup"))) or "w")
    monkeypatch.setattr(
        dsa_satwf, "_satwf_pre_rowsplit", lambda *a, **k: seen.append(("pre", "rowgroup" in k)) or "pre"
    )
    C.set_layouts(CLASSES["9-16"])
    W.set_layouts(FORM_W["9-16"])
    C.RUNNER = "test"
    return dsa_satwf, mhc_front.PendingComb(torch.zeros(1)), seen


def test_dispatch_sends_the_classed_calls_to_form_w(routes):
    dsa_satwf, pend, seen = routes
    assert dsa_satwf.satwf(*_args(24, 3072), comb_a=pend, rowgroup=3) == "w"
    assert dsa_satwf.satwf(*_args(16, 3072), comb_a=pend, rowgroup=1) == "w"
    assert dsa_satwf.satwf(*_args(24, 4096), comb_a=pend, rowgroup=3) == "pre"  # not the classes' bucket
    assert dsa_satwf.satwf(*_args(8, 3072), comb_a=pend, rowgroup=0) == "pre"  # no layout at B 8
    assert seen == [("w", 3), ("w", 1), ("pre", False), ("pre", False)]  # the row group taken off the others


def test_dispatch_refuses_a_classed_call_form_w_does_not_take(routes):
    dsa_satwf, pend, _ = routes
    with pytest.raises(NotImplementedError, match="form W does not take"):
        dsa_satwf.satwf(*_args(24, 3072), comb_a=torch.zeros(1), rowgroup=3)  # not a PendingComb
    with pytest.raises(NotImplementedError, match="form W does not take"):
        dsa_satwf.satwf(*_args(24, 3072), comb_a=pend)  # no row group


def test_dispatch_without_classes_is_the_function_below(routes):
    dsa_satwf, pend, seen = routes
    C.set_layouts("")
    W.set_layouts("")
    assert dsa_satwf.satwf(*_args(24, 3072), comb_a=pend, rowgroup=3) == "pre"
    assert dsa_satwf.satwf(comb_a=pend) == "pre"  # keyword-only calls pass through
    assert seen == [("pre", False), ("pre", False)]


def test_the_fused_call_passes_the_row_group():
    src = (ROOT / "trnrt" / "models" / "mla.py").read_text()
    a = src.index("r_ = dsa_satwf.satwf(")
    assert "rowgroup=model.attn_rank // A" in src[a : src.index("\n\n", a)]


# -- the engine's fields --------------------------------------------------------------------------------------------------

BAND = {
    "9-16": {"decode_batch_buckets": (8, 16, 24), "page_buckets": (1024, 3072, 4096)},
    "17+": {"decode_batch_buckets": (16, 24), "page_buckets": (1024, 3072, 4096)},
}


@pytest.mark.parametrize("band", sorted(BAND))
def test_engine_config_takes_the_bands_layouts(band):
    cfg = EngineConfig(
        model_path="m",
        dsa_row_classes=CLASSES[band],
        dsa_row_classes_cw=band == "17+",
        dsa_rowsplit_layout=FORM_W[band],
        **BAND[band],
    )
    assert cfg.dsa_row_classes == CLASSES[band]


@pytest.mark.parametrize(
    ("kw", "why"),
    [
        ({"dsa_row_classes_cw": True}, "need dsa_row_classes"),
        ({"dsa_rowsplit_layout": FORM_W["17+"]}, "need dsa_row_classes"),
        ({"dsa_row_classes": CLASSES["9-16"], "dsa_rowsplit_layout": "24:2x24+4x8"}, "must cover"),
        (
            {
                "dsa_row_classes": "24:2x24+2x16+4x12+16x6",
                "dsa_rowsplit_layout": "24:2x24+4x6",
                "decode_batch_buckets": (8, 16),
            },
            "not a decode bucket",
        ),
        (
            {
                "dsa_row_classes": "24:2x24+2x16+4x12+16x6",
                "dsa_rowsplit_layout": "24:2x24+4x6",
                "page_buckets": (1024, 3072),
            },
            "below the largest",
        ),
        (
            {"dsa_row_classes": "24:2x24+2x16+4x12+16x6", "dsa_rowsplit_layout": "24:2x24+4x6", "spec_method": "mtp"},
            "speculative",
        ),
    ],
)
def test_engine_config_refuses_inconsistent_layouts(kw, why):
    base = {"decode_batch_buckets": (16, 24), "page_buckets": (1024, 3072, 4096)}
    with pytest.raises(ValueError, match=why):
        EngineConfig(model_path="m", **{**base, **kw})


def test_set_decode_routes_sets_every_layout(layouts, monkeypatch):
    from trnrt.engine import engine as E
    from trnrt.kernels import dsa_slots_n_p, dsa_topk, sp_gather
    from trnrt.models import mla

    for mod, name in (
        (dsa_topk, "ROWSPLIT_BIG"),
        (mla, "CP_FAIL_TRIP"),
        (mla, "CP_FAIL_ATTN_TRIP"),
        (dsa_slots_n_p, "PIPE_ROWS"),
        (sp_gather, "DIRECT"),
        (mla, "ODEC_DROP"),
    ):
        monkeypatch.setattr(mod, name, getattr(mod, name))
    E.set_decode_routes(
        EngineConfig(
            model_path="m",
            dsa_row_classes=CLASSES["17+"],
            dsa_row_classes_cw=True,
            dsa_rowsplit_layout=FORM_W["17+"],
            **BAND["17+"],
        )
    )
    assert C.parse(CLASSES["17+"]) == C.LAYOUTS and W.parse_w(FORM_W["17+"]) == W.W_LAYOUTS and C.CW is True
    E.set_decode_routes(EngineConfig(model_path="m"))  # a later engine inherits nothing
    assert C.LAYOUTS == {} and C.CW is False and W.W_LAYOUTS == {}


# -- the NKI simulator ----------------------------------------------------------------------------------------------------


def _pair(monkeypatch, ngs, q, w, pkc, table, nl_, keep, rank, grid, cs=0, pf=2, nr=None, dbg=0, cw=False):
    """(sel2g's outputs on `table`, sel2gc's with the classes `ngs`)."""
    from trnrt.kernels import dsa_sel2g as G

    _use_layouts(monkeypatch, {len(ngs): tuple(ngs)})
    monkeypatch.setattr(C, "CW", cw)
    r = torch.tensor([rank])
    a = G.select2g(q, w, pkc, table, nl_, keep, 8, r, simulate=True, grid=grid, cs=cs, pf=pf, nr=nr)
    b = C.select2gc(q, w, pkc, table, nl_, keep, 8, r, simulate=True, grid=grid, cs=cs, pf=pf, nr=nr, dbg=dbg)
    return a, b


SIM_CASES = [  # (ngs, NPg, Hi, nloc, keep, ties)
    ((8, 8, 4, 4), 1024, 8, [8189, 6000, 4096, 90], 128, True),  # a row at its class's edge, a short row
    ((4, 4, 8, 8), 1024, 8, [4000, 0, 8192, 3000], 128, False),  # not longest first; an empty row
    ((16, 16, 8, 8, 4, 4), 2048, 8, [16383, 9000, 8192, 300, 4095, 2000], 256, True),  # three classes
    ((24, 24, 4, 4), 3072, 8, [24573, 13000, 4096, 1500], 128, True),  # NT 192 (the partial chunk) beside NT 32
    ((24, 24, 12, 12, 6, 6), 3072, 8, [24573, 13000, 12000, 700, 6100, 6144], 128, True),  # NT 96 / 48: the 17+ widths
]


@needs_nki
@pytest.mark.parametrize("grid", [1, 2])
@pytest.mark.parametrize("cs", [0, 1])
@pytest.mark.parametrize("cw", [False, True])
@pytest.mark.parametrize(("ngs", "NPg", "Hi", "nloc", "keep", "ties"), SIM_CASES)
def test_sel2gc_is_sel2g(monkeypatch, layouts, grid, cs, cw, ngs, NPg, Hi, nloc, keep, ties):
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    q, w, pkc, table, nl_ = case(len(ngs), NPg, Hi, nloc, seed=2100 + NPg + len(ngs), repeat=ties, pages=NPg * 2 + 64)
    a, b = _pair(monkeypatch, ngs, q, w, pkc, table, nl_, keep, 3, grid, cs=cs, cw=cw)
    for name, x, y in zip(("lval", "cpool", "rows"), a, b, strict=True):
        assert _bits(x, y), (name, ngs, grid, cs, cw, int((x != y).sum()))


@needs_nki
@pytest.mark.skipif(os.environ.get("TRNRT_TEST_RC_SERVED") != "1", reason="the served shapes: TRNRT_TEST_RC_SERVED=1")
@pytest.mark.parametrize(("band", "B"), [("9-16", 24), ("9-16", 16), ("17+", 24)])
def test_sel2gc_is_sel2g_at_the_served_layouts(monkeypatch, layouts, band, B):
    """The bands' layouts at P3072: Hi 32, keep 512, pf 5 / nr 6, cs 1, grid 2, each slot near its class's edge."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    ngs = C.parse(CLASSES[band])[B]
    nloc = [(128 * g * PPP) - (37 * s) % 900 for s, g in enumerate(ngs)]
    nloc[len(ngs) - 1] = 400  # a row shorter than keep
    q, w, pkc, table, nl_ = case(B, 3072, 32, nloc, seed=2300 + B, repeat=True, pages=3072 * 2 + 64)
    a, b = _pair(monkeypatch, ngs, q, w, pkc, table, nl_, 512, 5, 2, cs=1, pf=5, nr=6, cw=band == "17+")
    for name, x, y in zip(("lval", "cpool", "rows"), a, b, strict=True):
        assert _bits(x, y), (name, band, B, int((x != y).sum()))


@needs_nki
@pytest.mark.parametrize("grid", [1, 2])
def test_planted_partial_chunk_differs(monkeypatch, layouts, grid):
    """dbg 1 (sel2g's plant: the NG 24 class's columns 128-191 never counted) differs: the comparison can fail."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    q, w, pkc, table, nl_ = case(4, 3072, 8, [24573, 20000, 4096, 1500], seed=2401, pages=3072 * 2 + 64)
    a, b = _pair(monkeypatch, (24, 24, 4, 4), q, w, pkc, table, nl_, 128, 1, grid, dbg=1)
    assert not all(_bits(x, y) for x, y in zip(a, b, strict=True)), "the planted fault went unseen"


@needs_nki
@pytest.mark.parametrize("grid", [1, 2])
def test_planted_unclassed_ppg_differs(monkeypatch, layouts, grid):
    """The kernel fed page_groups(table) (no classes in the ppg): the narrow slots read pages p NGM + g."""
    from trnrt.kernels import dsa_index

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    monkeypatch.setattr(C, "classed_ppg", lambda table, ngs: dsa_index.page_groups(table))
    q, w, pkc, table, nl_ = case(4, 1024, 8, [8189, 6000, 4096, 3000], seed=2402, pages=1024 * 2 + 64)
    a, b = _pair(monkeypatch, (8, 8, 4, 4), q, w, pkc, table, nl_, 128, 1, grid)
    assert not all(_bits(x, y) for x, y in zip(a, b, strict=True)), "the unclassed ppg went unseen"


# form W's S2 consumers: the gathered rows -> gv / gc / the rank's rows, against the row group sum's (numpy)
A, NA, NB, KEEP = 8, 2, 4, 128
BG, B_, WORLD = NA + NB, W.RG * (NA + NB), W.RG * A

if importlib.util.find_spec("nki") is not None:
    import nki
    import nki.language as nl

    @nki.jit
    def _ag_consumers(Gr, rank, keep: int, A_: int, B_: int, Bg: int, na: int, nb: int):
        """_ag_block x 3 (gv, gc, the rows block) over the gathered rows Gr [world Bg, 3 keep], then _rows_g."""
        gv = nl.ndarray((A_, B_, keep), dtype=nl.float32, buffer=nl.shared_hbm)
        gc = nl.ndarray((A_, B_, keep), dtype=nl.float32, buffer=nl.shared_hbm)
        s2g = nl.ndarray((A_ * B_, keep), dtype=nl.float32, buffer=nl.shared_hbm)
        RS = nl.ndarray((B_, keep), dtype=nl.int32, buffer=nl.shared_hbm)
        W._ag_block(Gr, gv, 0, A_, B_, Bg, keep, na, nb)
        W._ag_block(Gr, gc, 1, A_, B_, Bg, keep, na, nb)
        W._ag_block(Gr, s2g, 2, A_, B_, Bg, keep, na, nb)
        W._rows_g(s2g, rank, B_, keep, A_, RS)
        return gv, gc, RS

    @nki.jit
    def _send(out, keep: int, Bg: int):
        xs = nl.ndarray((Bg, 3 * keep), dtype=nl.float32, buffer=nl.shared_hbm)
        W._s2_send_ag(out, xs, Bg, keep)
        return xs


def _outs(seed: int):
    """Every rank's out [3, Bg, keep]: lval normal with planted -0.0 / +0.0, cpool / rows non-negative integers."""
    g = np.random.default_rng(seed)
    o = np.zeros((WORLD, 3, BG, KEEP), np.float32)
    o[:, 0] = g.standard_normal((WORLD, BG, KEEP)).astype(np.float32)
    o[:, 0, :, ::17] = np.float32(-0.0)
    o[:, 0, :, 5::19] = np.float32(0.0)
    o[:, 1] = g.integers(0, 4096, (WORLD, BG, KEEP)).astype(np.float32)
    o[:, 2] = g.integers(0, 1 << 20, (WORLD, BG, KEEP)).astype(np.float32)
    return o


def _rows_of(grp: int):
    ra, rb = NA * grp, NA * W.RG + NB * grp
    return list(range(ra, ra + NA)) + list(range(rb, rb + NB))


def _group_sum(o, rank: int):
    """The classless form's S2 consumers: gv / gc [A, B, keep] (slot a's lists at every batch row, row b from its row
    group's rank of slot a; a sum of one nonzero term and +0.0s, so -0.0 arrives as +0.0) and the rank's pool rows."""
    gv = np.zeros((A, B_, KEEP), np.float32)
    gc = np.zeros((A, B_, KEEP), np.float32)
    rs = np.zeros((B_, KEEP), np.int32)
    for r in range(WORLD):
        grp, a = divmod(r, A)
        for j, b in enumerate(_rows_of(grp)):
            gv[a, b] = o[r, 0, j] + np.float32(0.0)
            gc[a, b] = o[r, 1, j] + np.float32(0.0)
            if a == rank:
                rs[b] = o[r, 2, j].astype(np.int32)
    return gv, gc, rs


def _gather(o, canon: bool = True):
    x = np.concatenate([o[:, p] for p in range(3)], axis=2).reshape(WORLD * BG, 3 * KEEP)
    return (x + np.float32(0.0)).astype(np.float32) if canon else x.copy()


def _same(x, y) -> bool:
    x, y = np.asarray(x), np.asarray(y)
    return x.shape == y.shape and x.tobytes() == y.tobytes()


@needs_nki
@pytest.mark.parametrize("rank", [0, 3, 5, 7])
def test_ag_consumers_are_the_group_sums(monkeypatch, rank):
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    o = _outs(40 + rank)
    rk = np.array([[rank]], np.float32)  # sat_args: rank.to(torch.float32).reshape(1, 1)
    got = nki.simulate(_ag_consumers)(_gather(o), rk, keep=KEEP, A_=A, B_=B_, Bg=BG, na=NA, nb=NB)
    for want, x in zip(_group_sum(o, rank), got, strict=True):
        assert _same(want, x)


@needs_nki
def test_send_is_rows_plus_zero(monkeypatch):
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    o = _outs(7)
    for r in (0, 22):
        xs = nki.simulate(_send)(o[r], keep=KEEP, Bg=BG)
        raw = np.concatenate([o[r, p] for p in range(3)], axis=1)
        want = (raw + np.float32(0.0)).astype(np.float32)
        assert _same(xs, want)
        assert not _same(raw, want)  # the case has a -0.0 the + 0.0 turns


@needs_nki
def test_planted_controls_differ(monkeypatch):
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    o = _outs(99)
    want = _group_sum(o, 3)
    rk = np.array([[3]], np.float32)
    swapped = nki.simulate(_ag_consumers)(_gather(o), rk, keep=KEEP, A_=A, B_=B_, Bg=BG, na=NB, nb=NA)
    assert not _same(want[0], swapped[0]), "the runs swapped left gv unchanged"
    raw = nki.simulate(_ag_consumers)(_gather(o, canon=False), rk, keep=KEEP, A_=A, B_=B_, Bg=BG, na=NA, nb=NB)
    assert not _same(want[0], raw[0]), "a gather without the + 0.0 matched the group sum on planted -0.0"
