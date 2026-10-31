"""The state pools' zero row (kernels/shell.py start_row): the pools' extra last row, zeros and never written, that a
row starting its sequence reads in range instead of row R past the pool (which the KDA kernels' hardware-DGE reads skip,
oob_mode skip: a serve warmup hung on such reads at four decode rows after a 4,096-row prefill).
Host: every slot builder reads the start row. Simulator (Neuron venv): with the zero row, every served form's outputs
equal the skip's bit for bit at warmup / real / pad rows and the zero row stays zero; a positive control (the zero row
planted nonzero) must change the starting rows' outputs, so the equality cannot hold because the reads missed the row.
(The pool side, the zero row in no free list and never written, is the engine's: tests/test_state_pool.py.)"""

from __future__ import annotations

import importlib.util
import inspect

import pytest
import torch

from trnrt.kernels import kda_shell as KS
from trnrt.kernels import shell as SH
from tests import test_kda_shell as TKS

NKI = importlib.util.find_spec("nki") is not None


def test_every_slot_builder_reads_the_start_row():
    assert SH.start_row(17) == 16
    assert "SH.slots(state_slot, positions, SH.start_row(R))" in inspect.getsource(KS.decode)


def _bits(t):
    return t.contiguous().view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)


def _slots(R, regime, T, seed):
    """[2, T] slots of a regime (warmup: every row at position 0, write row 0; real: in-range rows past position 0;
    pad: row 0 starting, the last row's read -1), the start row R, as the knob-off pools' builder makes them."""
    g = torch.Generator().manual_seed(seed)
    slot = torch.randperm(R - 1, generator=g)[:T] + 1
    if regime == "warmup":
        return SH.slots(torch.zeros(T, dtype=torch.long), torch.zeros(T, dtype=torch.long), R)
    if regime == "real":
        return SH.slots(slot, torch.arange(T) + 5, R)
    positions = torch.arange(T) + 5
    positions[0] = 0
    s = SH.slots(slot, positions, R)
    s[0, -1] = -1
    return s


def _pair(T, recb, cbat, regime, monkeypatch, plant=False, grid=2):
    """(skip's outputs: pools of R rows, a starting row reading R out of range; the zero row's: the same pools plus a
    zero last row R, the same slots, R now in range), and the zero row's pools as returned."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    c = TKS.case(T, 200 + 3 * T + recb + 2 * cbat, grid)
    L = c["L"]
    t = KS.arrange(c["wcat"], L.conv_w, L.f_b, L.g_b, L.dt_bias, L.out, c["fn"], L.in_norm, L.A_log, L.o_norm,
                   c["base"], c["scale"], TKS.NV, grid, TKS.HC)
    blobs = KS.pack(t, TKS.H, TKS.NV, TKS.K, grid, TKS.HC)
    R = L.rec_state.shape[0]
    slots = _slots(R, regime, T, 9 + T)
    conv_z = torch.cat([L.conv_state, torch.zeros_like(L.conv_state[:1])])
    rec_z = torch.cat([L.rec_state, torch.zeros_like(L.rec_state[:1])])
    if plant:
        conv_z[R] = 3.0
        rec_z[R] = 0.5

    def run(conv, rec):
        return KS.run(c["S"], blobs, conv.clone(), rec.clone(), slots, H=TKS.H, nv=TKS.NV, hc=TKS.HC, iters=TKS.ITERS,
                      hc_eps=TKS.HC_EPS, rms_eps=TKS.EPS, lb=TKS.LB, simulate=True, grid=grid, recb=recb, cbat=cbat)
    return run(L.conv_state, L.rec_state), run(conv_z, rec_z), R


FORMS = [(1, 0, 0), (2, 0, 0), (4, 1, 1), (6, 1, 1), (8, 1, 1), (4, 0, 1), (4, 1, 0)]


@pytest.mark.skipif(not NKI, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("regime", ["warmup", "real", "pad"])
@pytest.mark.parametrize("T,recb,cbat", FORMS)
def test_the_zero_row_is_the_skip_bit_for_bit(T, recb, cbat, regime, monkeypatch):
    """post, comb, P and the pools' first R rows equal the skip's bit for bit, and the zero row is still zero."""
    off, on, R = _pair(T, recb, cbat, regime, monkeypatch)
    for name, a, b in zip(("S", "post", "comb", "P"), off[:4], on[:4]):
        if a is None:
            continue
        assert torch.equal(_bits(a), _bits(b)), (name, regime)
    for name, i in (("conv", 4), ("rec", 5)):
        assert torch.equal(_bits(off[i]), _bits(on[i][:R])), (name, regime)
        assert not on[i][R].float().any(), (name, "the zero row was written")


@pytest.mark.skipif(not NKI, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("T,recb,cbat", [(2, 0, 0), (4, 1, 1), (4, 0, 1), (4, 1, 0)])
def test_a_planted_zero_row_shows(T, recb, cbat, monkeypatch):
    """The positive control: the zero row planted nonzero, warmup's rows (every one reads it) change: P and both written
    pools differ from the skip's. Had the reads missed row R, they would not."""
    off, on, R = _pair(T, recb, cbat, "warmup", monkeypatch, plant=True)
    assert not torch.equal(_bits(off[3]), _bits(on[3])), "P: the zero row was not read"
    assert not torch.equal(_bits(off[5]), _bits(on[5][:R])), "rec: the zero row was not read"
    assert not torch.equal(_bits(off[4]), _bits(on[4][:R])), "conv: the zero row was not read"
