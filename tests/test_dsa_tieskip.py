"""The merge's stacked selection with its tie search as a 0 / 1-trip device loop (kernels/dsa_shell.py _select_st,
_tie_need_st, _tie_agree).

On the host: the quantized case the planted control runs on does need the search (more scores tied at a visible
threshold than room), and the skip's visibility gate is dsa_long's.
In the NKI simulator: the attend with the skip is the attend with the search forced on every call, bit for bit, on
continuous scores (0 trips), quantized scores (1 trip) and short rows (an invisible threshold: 0 trips), at rbits 4 and
2, grids 1 and 2; the split cases where only one LNC program's rows need the search (the agreed trip count); and the
planted fault (the trip count forced to 0) must change the quantized case's selection."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from tests.test_dsa_shell import _bits, attend_case
from trnrt.kernels import dsa_shell as dsh
from trnrt.models import dsa_long

needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")

MERGE = {  # name: (T, npool, levels)
    "continuous": (3, [2000, 900, 1500], None),
    "quantized": (3, [2000, 1800, 1500], 16),
    "short": (3, [40, 2000, 70], 16),
}


def _case(name: str):
    T, npool, levels = MERGE[name]
    return attend_case(T, 128, 32, npool, seed=700 + T, levels=levels)


def _row_needs(ev, keep: int = 128) -> list[bool]:
    """Each row's need: more scores tied at a visible threshold than room (torch, the merge's rule)."""
    out = []
    for t in range(ev.shape[1]):
        c = ev[:, t, :].reshape(-1)
        th = c.sort(descending=True).values[keep - 1]
        out.append(bool(int((c == th).sum()) > keep - int((c > th).sum()) and th > dsh.VISIBLE_T))
    return out


def test_quantized_merge_needs_the_search():
    """The planted control's case: rows 1 and 2 have more scores tied at a visible threshold than room; the continuous
    case has no such row (its search would run 0 trips)."""
    need = _row_needs(_case("quantized")[0])
    assert need[1] and need[2], need
    assert not any(_row_needs(_case("continuous")[0]))


def test_dsa_long_visible_is_the_kernels():
    """The skip's visibility gate reads dsa_shell.VISIBLE_T: dsa_long's VISIBLE (what the selection's mask means)."""
    assert dsh.VISIBLE_T == dsa_long.VISIBLE


def test_the_lim_bound_holds_at_the_largest_bucket():
    """Without the search LIM = 2^nbits - 1 must stay below every untied entry (BIG_T + pool): nbits of the largest
    merge (4096 pages x 8 pools a page row x 8 ranks) fits."""
    nbits = (4096 * dsh.PPP * 8 - 1).bit_length()
    assert (1 << nbits) <= dsh.BIG_T


# -- the NKI simulator ----------------------------------------------------------------------------------------------------


def _always(tied, th, room, Gc, CD, n, nt):
    """The search runs on every call (trip count 1): the selection before the skip, through the loop body."""
    z = dsh._sb((1, 1), dsh.I32)
    dsh.nisa.memset(dst=z, value=1)
    return z


def _never(tied, th, room, Gc, CD, n, nt):
    """The planted fault: the search never runs (trip count 0)."""
    z = dsh._sb((1, 1), dsh.I32)
    dsh.nisa.memset(dst=z, value=0)
    return z


def _boom(*a, **k):
    raise RuntimeError("patched helper traced")


def _rekey(m, salt: int):
    """A distinct static rev per patched variant: the simulator caches a trace by the static arguments, so a patched
    helper under the served rev would replay the served trace."""
    m.setattr(dsh, "REVA", dsh.REVA ^ salt)


def _attend(args, rank: int, grid: int, rbits: int):
    ev, ec, rows_sel, table, pos, q, kc = args
    return dsh.attend(
        ev,
        ec,
        rows_sel,
        table,
        pos,
        q,
        kc,
        128,
        8,
        torch.tensor([rank]),
        0.05,
        ns=256,
        rbits=rbits,
        il=1,
        sb=1,
        st=1,
        ss=1,
        tl=1,
        fo=1,
        simulate=True,
        grid=grid,
    )


def _same5(a, b):
    return all(_bits(x.float(), y.float()) if x.dtype != torch.int32 else torch.equal(x, y) for x, y in zip(a, b))


@needs_nki
@pytest.mark.parametrize("grid", [1, 2])
@pytest.mark.parametrize("name", list(MERGE))
def test_skip_is_the_search_bits(monkeypatch, grid, name):
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    args = _case(name)
    for rank in (0, 5):
        for rbits in (4, 2):
            got = _attend(args, rank, grid, rbits)
            with monkeypatch.context() as m:
                m.setattr(dsh, "_tie_need_st", _always)
                _rekey(m, 0x1 + rbits)
                want = _attend(args, rank, grid, rbits)
            assert _same5(want, got), (name, grid, rank, rbits)


def split_case(which: str):
    """T 3 at grid 2: program 0 has rows 0 and 1, program 1 row 2 (dsa_sa._attend's split). One quantized row placed in
    program 0's rows ("p0": row 1) or program 1's ("p1": row 2), the others continuous: the programs' own flags
    disagree, the case the agreed trip count exists for."""
    T, npool = 3, [2000, 1800, 1500]
    cont = attend_case(T, 128, 32, npool, seed=703, levels=None)
    quant = attend_case(T, 128, 32, npool, seed=703, levels=16)
    row = {"p0": 1, "p1": 2}[which]
    ev, ec = cont[0].clone(), cont[1].clone()
    ev[:, row], ec[:, row] = quant[0][:, row], quant[1][:, row]
    return (ev, ec, *cont[2:]), row


@needs_nki
@pytest.mark.parametrize("grid", [1, 2])
@pytest.mark.parametrize("which", ["p0", "p1"])
def test_agreed_split_is_the_search_bits(monkeypatch, grid, which):
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    args, row = split_case(which)
    needs = _row_needs(args[0])
    assert needs[row] and sum(needs) == 1, needs
    for rank in (0, 5):
        got = _attend(args, rank, grid, 4)
        with monkeypatch.context() as m:
            m.setattr(dsh, "_tie_need_st", _always)
            _rekey(m, 0x10)
            assert _same5(_attend(args, rank, grid, 4), got), (which, grid, rank)


@needs_nki
def test_agree_is_traced_at_grid_2_only(monkeypatch):
    """The exchange runs where the programs split rows (grid 2, T >= 2) and not at grid 1 (a raise control)."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    args, _ = split_case("p1")
    monkeypatch.setattr(dsh, "_tie_agree", _boom)
    _rekey(monkeypatch, 0x20)
    with pytest.raises(Exception, match="patched helper traced"):
        _attend(args, 0, 2, 4)
    _rekey(monkeypatch, 0x40)
    _attend(args, 0, 1, 4)  # grid 1: one program, no exchange, no raise


@needs_nki
@pytest.mark.parametrize("grid", [1, 2])
def test_planted_skip_fails(monkeypatch, grid):
    """The search forced off changes the quantized case's selection bias (sbias) on some rank: every tied score
    selected, not room of them. A raise control first: the patched helper is traced at all."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    args = _case("quantized")
    wants = {r: _attend(args, r, grid, 4) for r in (0, 5)}
    with monkeypatch.context() as m:
        m.setattr(dsh, "_tie_need_st", _boom)
        _rekey(m, 0x80)
        with pytest.raises(Exception, match="patched helper traced"):
            _attend(args, 0, grid, 4)
    monkeypatch.setattr(dsh, "_tie_need_st", _never)
    _rekey(monkeypatch, 0x100)
    moved = [not _bits(wants[r][3], _attend(args, r, grid, 4)[3]) for r in (0, 5)]
    assert any(moved), "the planted skip left every rank's selection bias unchanged"
