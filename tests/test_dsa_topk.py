"""kernels/dsa_topk.py: the exact DSA top-k selection kernel, and models/dsa_select.py's device-safe top-k.

The kernel's arithmetic (emulate) against the tie rule spelled out by a stable sort (dsa_select.reference_mask) on
the host, the kernel itself in the NKI CPU simulator against emulate (Neuron venv only), and GLM-5.3-Flash's pooled
selection (glm5_next.block_mask) through it.

The tie rule under test: per row the `keep` largest scores; every selected score >= every unselected one; among the
scores equal to the keep-th largest, the lowest indices; with vis_only, nothing at or below dsa_topk.VISIBLE (an
invisible candidate, NEG_INF + score) is selected, so a row with fewer visible scores than `keep` selects exactly its
visible ones.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from trnrt import platform
from trnrt.kernels import dsa_topk as dk
from trnrt.models import dsa_select as ds

NEG_INF = -1e30
needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")


def _scores(kind: str, rows: int, n: int, g: torch.Generator) -> torch.Tensor:
    if kind == "randn":
        return torch.randn(rows, n, generator=g)
    if kind == "integers":  # thousands tied at the threshold
        return torch.randint(-3, 4, (rows, n), generator=g).float()
    if kind == "zeros":  # a relu-sum indexer with dead keys: exact 0s (both signs) beside tiny values
        s = torch.relu(torch.randn(rows, n, generator=g)) * (torch.rand(rows, n, generator=g) - 0.5) * 1e-6
        s[:, ::3] = 0.0
        s[:, 1::5] = -0.0
        return s
    if kind == "wide":  # magnitudes from underflow to overflow, both signs, infinities
        return torch.sign(torch.randn(rows, n, generator=g)) * torch.exp(torch.randn(rows, n, generator=g) * 30)
    if kind == "invisible":  # a DSA row: visible scores, the rest at NEG_INF + score
        s = torch.randn(rows, n, generator=g) * 10
        s[:, n // 3 :] += -1e30
        return s
    if kind == "ulps":  # distinct scores one ulp apart at the threshold
        s = torch.randn(rows, n, generator=g)
        s[:, : n // 2] = 1.0 + torch.randint(0, 4, (rows, n // 2), generator=g).float() * 2**-23
        return s
    if kind == "equal":
        return torch.full((rows, n), 0.37)
    raise ValueError(kind)


KINDS = ["randn", "integers", "zeros", "wide", "invisible", "ulps", "equal"]


def want(s: torch.Tensor, keep: int, vis_only: bool) -> torch.Tensor:
    sel = ds.reference_mask(s, keep)
    if vis_only:
        sel = sel & (s > dk.VISIBLE)
    return sel


def pooled(rows: int, n: int, g: torch.Generator, visible: int | None = None, zeros: bool = False) -> torch.Tensor:
    """Scores as glm5_next.pooled_selection forms them: index = sum_h w_h relu(q_h . k_p) (w of both
    signs, so negative and exactly-zero scores occur) plus 0 / NEG_INF for invisible pools."""
    q = torch.randn(rows, 8, 16, generator=g)
    k = torch.randn(n, 16, generator=g)
    w = torch.randn(rows, 8, generator=g)
    rel = torch.relu(torch.einsum("rhd,nd->rhn", q, k))
    if zeros:  # dead keys: every head's relu zero
        rel[..., ::5] = 0.0
    index = torch.einsum("rh,rhn->rn", w, rel)
    if visible is not None:
        index = index + torch.where(torch.arange(n) < visible, 0.0, NEG_INF)
    return index


# -- the reference rule and the device-safe top-k ----------------------------------------------------------------


def test_reference_mask_is_torch_topk_without_ties():
    g = torch.Generator().manual_seed(7)
    s = torch.randn(64, 8192, generator=g)
    want_ = torch.zeros_like(s, dtype=torch.bool).scatter(-1, torch.topk(s, 2048, dim=-1).indices, True)
    assert torch.equal(ds.reference_mask(s, 2048), want_)
    assert ds.reference_mask(s[:3, :10], 50).all()  # fewer scores than keep: all of them


def test_topk_ids_are_in_range_on_nan_and_inf_rows():
    """The router's top-k (dsa_select.topk_ids): a NaN or all -inf row ranks every entry as finfo.min, so it takes k
    distinct in-range ids; a finite row's ids are torch.topk's."""
    g = torch.Generator().manual_seed(1)
    s = torch.randn(6, 288, generator=g)
    s[1] = float("nan")
    s[2] = float("-inf")
    s[3, ::7] = float("nan")
    ids = ds.topk_ids(s, 8)
    assert ((ids >= 0) & (ids < 288)).all()
    for r in (1, 2):
        assert len(set(ids[r].tolist())) == 8
    for r in (0, 4, 5):
        assert torch.equal(ids[r], torch.topk(s[r], 8).indices)
    assert not torch.isnan(s[3].gather(0, ids[3])).any()  # NaN entries rank below every finite one


# -- the kernel's arithmetic -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("vis_only", [False, True])
@pytest.mark.parametrize("kind", KINDS)
def test_emulation_follows_the_tie_rule(kind, vis_only):
    g = torch.Generator().manual_seed(200 + KINDS.index(kind))
    for _ in range(10):
        n = int(torch.randint(16, 3000, (1,), generator=g))
        keep = int(torch.randint(1, n, (1,), generator=g))
        s = _scores(kind, 4, n, g)
        got = dk.emulate(s, keep, vis_only) == 0
        assert torch.equal(got, want(s, keep, vis_only)), (kind, n, keep)
        if not vis_only:
            assert (got.sum(-1) == keep).all()


@pytest.mark.parametrize("visible", [None, 3000, 1500, 400, 3])
def test_emulation_on_pooled_scores(visible):
    """GLM-5.3-Flash's decode shape (8 rows, 2112 pools, keep 512) and a prefill chunk's (256 rows),
    with all pools visible or only the first `visible` (fewer than keep: all of them)."""
    g = torch.Generator().manual_seed(7)
    for rows in (8, 256):
        for zeros in (False, True):
            s = pooled(rows, 2112, g, visible, zeros)
            got = dk.emulate(s, 512) == 0
            assert torch.equal(got, want(s, 512, True))
            if visible is not None and visible < 512:
                assert torch.equal(got, (torch.arange(2112) < visible).expand(rows, -1))


def test_pieces():
    assert dk.pieces(8, 2112) == 16 and dk.pieces(256, 2112) == 1 and dk.pieces(3, 100) == 4
    assert dk.pieces(1, 2112) == 64 and dk.pieces(8, 2112, group=False) == 1 and dk.pieces(128, 4096) == 1


@pytest.mark.parametrize("visible", [None, 6000, 1000])
def test_block_mask_is_the_tie_rule(visible):
    """glm5_next.block_mask through the kernel's selection, tail and expansion included, decode and prefill forms: the
    pools it keeps are the tie rule's among the candidates (pools whose last token is visible), expanded to their
    tokens, plus (tail) every token of a non-candidate pool (the query's own incomplete pool; the attention's causal
    mask removes the invisible ones)."""
    from trnrt.models.glm5_next import block_mask

    g = torch.Generator().manual_seed(11)
    kp, L, keep = 4, 8448, 512
    for B, Q in ((8, 1), (1, 64)):
        P = L // kp
        index = pooled(B * Q, P, g).view(B, Q, P)
        n = L if visible is None else visible
        # Visible prefix per query: decode rows each see n tokens; a chunk's queries see n - Q + 1 + q.
        nv = torch.full((B, Q), n) if Q == 1 else (n - Q + 1 + torch.arange(Q)).view(1, Q)
        vis = torch.where(torch.arange(L).view(1, 1, L) < nv.unsqueeze(-1), 0.0, NEG_INF)
        b = block_mask(index, vis, kp, keep, True) == 0
        cand = vis.reshape(B, Q, P, kp)[..., kp - 1]
        sc = index + cand
        pools = want(sc.reshape(-1, P), keep, True).view(B, Q, P)
        tail = cand < 0  # non-candidate pools
        assert torch.equal(b, (pools | tail).repeat_interleave(kp, -1))
        assert torch.equal(pools.sum(-1), (cand == 0).sum(-1).clamp(max=keep))  # min(keep, candidates) pools


def test_token_output_is_the_expanded_pool_selection():
    """kp > 1 repeats each pool's value on its tokens; tail adds every non-candidate pool."""
    g = torch.Generator().manual_seed(5)
    s = pooled(4, 600, g, visible=450)
    pool = dk.emulate(s, 100) == 0
    tok = dk.emulate(s, 100, kp=4) == 0
    assert torch.equal(tok, pool.repeat_interleave(4, -1))
    tok = dk.emulate(s, 100, kp=4, tail=True) == 0
    assert torch.equal(tok, (pool | (torch.arange(600) >= 450)).repeat_interleave(4, -1))


def test_pooled_selection_is_exact_on_wide_range_scores():
    """On scores spread over many orders of magnitude a bisection of [min, max] keeps nearly every visible pool; the
    kernel's radix search keeps exactly `keep`, the tie rule's."""
    from trnrt.models.glm5_next import block_mask

    g = torch.Generator().manual_seed(9)
    B, kp, P, keep = 8, 4, 2112, 512
    index = torch.sign(torch.randn(B, 1, P, generator=g)) * torch.exp(torch.randn(B, 1, P, generator=g) * 20)
    index = index.clamp(-1e20, 1e20)  # finite, as pooled scores are
    vis = torch.zeros(B, 1, P * kp)
    got = block_mask(index, vis, kp, keep, False) == 0
    want_ = ds.reference_mask(index.view(B, P), keep).repeat_interleave(kp, -1).view(B, 1, P * kp)
    assert torch.equal(got, want_)


def _same_args(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        torch.equal(a[k], b[k]) if isinstance(a[k], torch.Tensor) else a[k] == b[k] for k in a
    )


def test_rowsplit_cuts_a_one_tile_call_into_two_tiles(monkeypatch):
    """The row split: a call of R >= 2 rows that fits one 128-partition tile (every decode call: B rows x g pieces) is
    cut for ceil(R / 2) rows instead, so it is two tiles with tiles = 1 (tile t on program t), over the same scores;
    one row, a call of several tiles and a call of more than ROWSPLIT_ROWS rows (a prefill chunk's merge, one tile at
    128 rows) keep their arguments. On a grid-2 kernel the split is the default; on grid 1 there is nothing to split."""
    g = torch.Generator().manual_seed(11)
    # (rows, local pools, pieces, partitions): 8192 / 32768 local pools are the 1024 / 4096 rungs at CP 8
    for R, P, gp, N in (
        (2, 8192, 128, 256),
        (4, 8192, 64, 256),
        (6, 8192, 32, 192),
        (8, 8192, 32, 256),
        (4, 32768, 64, 256),
        (8, 4096, 32, 256),
    ):
        s = torch.randn(R, P, generator=g)
        off = dk.kernel_inputs(s, 512, True, rowsplit=False)
        on = dk.kernel_inputs(s, 512, True, rowsplit=True)
        assert off["g"] == dk.pieces(R, P) and off["tiles"] == dk.TILES and off["sc"].shape[0] <= 128, (R, P)
        assert (on["g"], on["lg"], on["sc"].shape[0], on["tiles"]) == (gp, gp.bit_length() - 1, N, 1), (R, P)
        assert torch.equal(on["sc"].reshape(R, P), s), (R, P)  # the same scores, cut finer
        assert {k: v for k, v in on.items() if k not in ("sc", "g", "lg", "tiles")} == {
            k: v for k, v in off.items() if k not in ("sc", "g", "lg", "tiles")
        }
        monkeypatch.setattr(platform, "nki_grid", lambda: 2)
        assert _same_args(dk.kernel_inputs(s, 512, True), on), (R, P)
        monkeypatch.setattr(platform, "nki_grid", lambda: 1)  # grid 1: one program, nothing to split
        assert _same_args(dk.kernel_inputs(s, 512, True), off), (R, P)
    for R, P in ((1, 8192), (512, 600), (16, 4096), (128, 4096)):  # one row; several tiles; a prefill merge's rows
        s = torch.randn(R, P, generator=g)
        assert _same_args(dk.kernel_inputs(s, 100, True, rowsplit=True), dk.kernel_inputs(s, 100, True, rowsplit=False))


# -- the kernel in the simulator -------------------------------------------------------------------------------------


def _simulate(grid: int, monkeypatch):
    """The kernel in the simulator on `grid` programs, with platform.nki_grid pinned to it: kernel_inputs reads the grid
    (the row split is on at grid 2), and nothing in a test process sets the runtime's LNC. Grid 2 is the served LNC 2
    launch (a decode call's row split: tile t on program t), grid 1 one program over every tile."""
    import nki

    monkeypatch.setattr(platform, "nki_grid", lambda: grid)
    return nki.simulate(dk.kernel()[grid] if grid > 1 else dk.kernel())


@needs_nki
@pytest.mark.parametrize("grid", [1, 2], ids=["grid1", "grid2"])
@pytest.mark.parametrize(
    "rows,n,keep,kind",
    [
        (8, 2112, 512, "pooled"),
        (8, 2112, 512, "integers"),
        (8, 2112, 512, "zeros"),
        (8, 2112, 512, "invisible"),
        (256, 2112, 512, "pooled"),
        (3, 100, 7, "wide"),
        (5, 96, 40, "equal"),
        (8, 2112, 512, "ulps"),
        (1, 2112, 512, "pooled"),
        (2, 640, 120, "equal"),
    ],
)
def test_nki_simulator_matches_emulation(rows, n, keep, kind, grid, monkeypatch):
    """The kernel (the served RBITS rounds) selects exactly what the emulation (the 1-bit search's arithmetic)
    selects, on the adversarial score kinds and every output form, on one program and on the served two."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    sim = _simulate(grid, monkeypatch)
    g = torch.Generator().manual_seed(rows * 31 + n)
    s = (
        pooled(rows, n, g, visible=n // 3 if kind == "pooled" and rows == 256 else None)
        if kind == "pooled"
        else _scores(kind, rows, n, g)
    )
    if kind == "wide":  # no subnormals (the device may flush them; the simulator need not)
        s = torch.where(s.abs() < 1.2e-38, torch.zeros_like(s), s)
    for vis_only, kp, tail in ((True, 1, False), (False, 1, False), (True, 4, True), (True, 4, False)):
        args = dk.kernel_inputs(s, keep, vis_only, kp=kp, tail=tail)
        out = torch.as_tensor(sim(**args)).reshape(rows, n * kp)
        assert torch.equal(out, dk.emulate(s, keep, vis_only, kp, tail)), (kind, vis_only, kp, tail)


@needs_nki
@pytest.mark.parametrize("grid", [1, 2], ids=["grid1", "grid2"])
@pytest.mark.parametrize("rbits", [1, 3])
def test_other_round_widths_are_the_same_selection(rbits, grid, monkeypatch):
    """w bits a round in the radix and tie searches (the served RBITS and its neighbours) select the same: the w-bit
    search is exact for any w, so a kernel that only matched at the served w would be wrong."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    sim = _simulate(grid, monkeypatch)
    g = torch.Generator().manual_seed(37 + rbits)
    for rows, n, keep, kind in ((8, 2112, 512, "integers"), (8, 2112, 512, "ulps"), (5, 96, 40, "equal")):
        s = _scores(kind, rows, n, g)
        args = dict(dk.kernel_inputs(s, keep, True), rbits=rbits)
        out = torch.as_tensor(sim(**args)).reshape(rows, n)
        assert torch.equal(out, dk.emulate(s, keep, True)), (kind, rbits)


@needs_nki
def test_wbit_rounds_reach_the_kernel(monkeypatch):
    """The positive control for the round width: with g > 1 every round is one count matmul, so the trace holds
    1 + ceil(31 / w) + 1 + ceil(nbits / w) of them (sign, radix, above, ties), for both frontends. Were the width not
    to reach the kernel (shadowed, dropped), the equalities above would compare the 1-bit search with itself."""
    import numpy as np
    from nki.compiler.frontend import FrontendKind, source_to_mlir

    # grid 1: the count is per row tile. Grid 2's row split cuts this call into two tiles, and a trace without a grid
    # walks both, so it would hold twice the matmuls.
    monkeypatch.setattr(platform, "nki_grid", lambda: 1)
    g = torch.Generator().manual_seed(3)
    args = dk.kernel_inputs(pooled(8, 2112, g), 512, True)
    assert args["g"] > 1 and args["rbits"] == dk.RBITS
    nb = args["nbits"]
    for fe in (FrontendKind.PARSER, FrontendKind.TRACER):
        for w in (1, 2, 3):
            a = dict(args, sc=np.asarray(args["sc"]), rbits=w)
            m = str(source_to_mlir(dk.kernel(), **a, target="trn2", frontend=fe)).count("nisa.matmul")
            assert m == 2 + -(-31 // w) + -(-nb // w), (fe, w, m)
