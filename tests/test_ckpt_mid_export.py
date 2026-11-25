"""The in-chunk state checkpoint from the main scan (models/linear_attn.py MID_EXPORT, _scan_mid_export;
kernels/delta_rule.py chunk(states=True)), against the masked second scan.

Host: the kernel's arguments with states (the static flag and the row offsets of mrow, only then); split_rows'
layout; _mix_rows' chunk form takes the export only with MID_EXPORT, an in-chunk checkpoint and the NKI prefill, and
writes the end state and the checkpoint to their rows as the second scan's form does (the route:
tests/test_kda_routes.py).
Simulator (Neuron venv): the states flag leaves o and the end state alone and its rows are the chunk ends; the row
output is the operands' rows at a runtime offset; the export equals the second scan bit for bit at aligned and
unaligned checkpoints, with a planted wrong k caught; padding rows stay masked. The device facts it rests on (a fully
masked chunk leaves the state unchanged: exp(0) == 1.0, an exact identity product with the state moving) are the
device's, not the simulator's."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from tests.test_kda import inputs
from trnrt.kernels import delta_rule as dr
from trnrt.models import linear_attn as la

needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
T, H = 384, 2


def test_kernel_inputs_carry_the_states_flag_and_the_row_offsets_only_with_states():
    q, k, v, g, beta, S0 = inputs(T, H, seed=5)
    plain = dr.kernel_inputs(q, k, v, g, beta, S0, device=torch.device("cpu"), ug=6)
    assert "states" not in plain and "moff" not in plain
    a = dr.kernel_inputs(q, k, v, g, beta, S0, device=torch.device("cpu"), ug=6, states=True, mrow=torch.tensor([128]))
    assert (
        a["states"] == 1
        and a["moff"].dtype == torch.int32
        and a["moff"].tolist() == [[128 * H * 128, 128 * H * 128, 128 * H * 128, 128 * H]]
    )
    assert {n: a[n] for n in plain if not isinstance(a[n], torch.Tensor)} == {
        n: plain[n] for n in plain if not isinstance(plain[n], torch.Tensor)
    }


def test_split_rows_is_the_kernels_column_layout():
    Hk, Hv, D = 2, 2, 128
    widths = (Hk * D, Hk * D, Hv * D, Hv * D, Hv)
    R = torch.cat([torch.full((128, w), float(i)) for i, w in enumerate(widths)], dim=1)
    rows = dr.split_rows(R, Hk, Hv, True)
    assert [tuple(x.shape) for x in rows] == [(128, Hk, D), (128, Hk, D), (128, Hv, D), (128, Hv, D), (128, Hv)]
    assert [float(x.unique()) for x in rows] == [0.0, 1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize("on", [True, False])
def test_the_chunk_form_takes_the_export_only_with_the_switch(monkeypatch, on):
    """_mix_rows' chunk form with an in-chunk checkpoint on device tensors ("meta": shapes only), the kernels stood
    in: the export (one call) with MID_EXPORT, else the scan and the masked second scan; either way one write of the
    end state and the checkpoint to their two rows."""
    from types import SimpleNamespace

    from tests.test_kda_shell import stand_in
    from trnrt.kernels import gated_norm, short_conv

    _m, L0, _g = stand_in(1, R=6)
    meta = torch.device("meta")
    L = SimpleNamespace(**{n: (t.to(meta) if isinstance(t, torch.Tensor) else t) for n, t in vars(L0).items()})
    m = SimpleNamespace(cfg=SimpleNamespace(rms_norm_eps=1e-5), dtype=torch.bfloat16, page_size=256, _w=getattr)
    C = 16
    x = torch.empty(C, 4096, dtype=torch.bfloat16, device=meta)
    positions = torch.empty(C, dtype=torch.long, device=meta)
    slot_mapping = torch.empty(C, dtype=torch.long, device=meta)
    state_slot = torch.empty(3, dtype=torch.long, device=meta)  # the row, the checkpoint's row, k
    calls, writes = [], []

    def out(n):
        return torch.empty(n, 2, 128, device=meta), torch.empty(2, 128, 128, device=meta)

    def fake_export(q, k, v, g_, beta, S0, mid_k, vf, keep_pad, ug):
        calls.append(("export", keep_pad, ug))
        return (*out(C), torch.empty(2, 128, 128, device=meta))

    def fake_chunk(q, k, v, g_, beta, S, keep_pad=False, ug=None):
        calls.append(("scan", keep_pad, ug))
        return out(C)

    monkeypatch.setattr(la, "MID_EXPORT", on)
    monkeypatch.setattr(la, "kernel_takes", lambda spec: True)
    monkeypatch.setattr(la, "_scan_mid_export", fake_export)
    monkeypatch.setattr(la, "_nki_chunk", fake_chunk)
    monkeypatch.setattr(
        la, "_write_rows", lambda pool, rows, values: writes.append((pool is L.rec_state, rows.shape[0]))
    )
    monkeypatch.setattr(gated_norm, "takes2", lambda *a: False)
    monkeypatch.setattr(short_conv, "takes", lambda *a: False)
    with torch.no_grad():
        la._mix_rows(m, L, x, positions, slot_mapping, state_slot)
    assert calls == ([("export", False, 6)] if on else [("scan", False, 6), ("scan", False, 6)])
    assert writes[-1] == (True, 2) and [w for w in writes if w[0]] == [(True, 2)]


# -- the simulator ----------------------------------------------------------------------------------------------------


def _sim_chunk(q, k, v, g, beta, S0, keep_pad=False, ug=None, states=False, mrow=None):
    """delta_rule.chunk's contract through nki.simulate (grid 1). The simulator's target is set for the call only, and
    the variable restored after it: the module-scoped fixtures call this too, where no test's monkeypatch reaches."""
    import nki

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
        a = dr.kernel_inputs(q, k, v, g, beta, S0, device=torch.device("cpu"), ug=ug, states=states, mrow=mrow)
        out = tuple(torch.as_tensor(x) for x in nki.simulate(dr.kernel())(**a))
    if states:
        return (*out[:3], dr.split_rows(out[3], k.shape[1], v.shape[1], g.dim() == 3))
    return out


def test_the_simulator_target_is_set_for_the_call_only(monkeypatch):
    """_sim_chunk leaves the process environment as it found it, so a later test does not compile for its target. A
    stand-in nki.simulate records the variable during the call."""
    import os
    import sys
    import types

    seen = []

    def simulate(kernel):
        def run(**a):
            seen.append(os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE"))
            return (torch.zeros(1), torch.zeros(1))

        return run

    monkeypatch.setitem(sys.modules, "nki", types.SimpleNamespace(simulate=simulate))
    monkeypatch.setattr(dr, "kernel", lambda: None)
    q, k, v, g, beta, S0 = inputs(T, H, seed=5)
    for before in (None, "trn1"):
        if before is None:
            monkeypatch.delenv("NEURON_PLATFORM_TARGET_OVERRIDE", raising=False)
        else:
            monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", before)
        _sim_chunk(q, k, v, g, beta, S0, ug=6)
        assert seen[-1] == "trn2" and os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE") == before


def _bits(t):
    return t.contiguous().float().view(torch.int32)


def _eq(a, b):
    return torch.equal(_bits(a), _bits(b))


@pytest.fixture(scope="module")
def data():
    return inputs(T, H, seed=5, correlated=True)


@pytest.fixture(scope="module")
def states(data):
    return _sim_chunk(*data, ug=6, states=True, mrow=torch.tensor([128]))


@needs_nki
def test_states_flag_leaves_o_and_s_alone_and_rows_are_chunk_ends(data, states):
    o1, S1, S_all, _ = states
    o0, S0f = _sim_chunk(*data, ug=6)
    assert _eq(o1, o0) and _eq(S1, S0f)
    assert S_all.shape == (T // dr.L, H, 128, 128)
    assert _eq(S_all[-1], S1)
    q, k, v, g, beta, S0 = data
    for c in range(T // dr.L - 1):
        n = (c + 1) * dr.L
        _, Sc = _sim_chunk(q[:n], k[:n], v[:n], g[:n], beta[:n], S0, ug=6)
        assert _eq(S_all[c], Sc), f"chunk {c}"


@needs_nki
@pytest.mark.parametrize("r0", [0, 128, T - dr.L])
def test_rows_are_the_operands_rows(data, r0):
    """The kernel's row output is rows r0 .. r0 + 127 of its fp32 operands, bitwise, at a runtime offset (plant: r0 +
    128's rows differ from r0's)."""
    q, k, v, g, beta, _S0 = data
    rows = _sim_chunk(*data, ug=6, states=True, mrow=torch.tensor([r0]))[3]
    for name, got, x in zip(["q", "k", "v", "g", "beta"], rows, (q, k, v, g, beta), strict=True):
        assert _eq(got, x[r0 : r0 + dr.L].float()), f"{name} at r0 {r0}"
        if r0 + dr.L < T:
            assert not _eq(got, x[r0 + dr.L : r0 + 2 * dr.L].float()), f"{name}: the plant does not bite"


def _two_scan(data, mid_k):
    """The masked second scan over the whole chunk (linear_attn's chunk form without the export)."""
    q, k, v, g, beta, S0 = data
    mf = (torch.arange(T) < mid_k).view(T, 1)
    _, S_m = _sim_chunk(
        q, k, v, torch.where(mf.view(T, 1, 1), g, torch.zeros_like(g)), torch.where(mf, beta, torch.zeros_like(beta)),
        S0, ug=6,
    )  # fmt: skip
    return S_m


@needs_nki
@pytest.mark.parametrize("mid_k", [256, 200, 1, 383, T])
def test_export_equals_the_second_scan(data, states, monkeypatch, mid_k):
    calls = []

    def fake_chunk(q, k, v, g, beta, S0, keep_pad=False, ug=None, states=False, mrow=None):
        calls.append((q.shape[0], states))
        return _sim_chunk(q, k, v, g, beta, S0, keep_pad=keep_pad, ug=ug, states=states, mrow=mrow)

    monkeypatch.setattr(dr, "chunk", fake_chunk)
    q, k, v, g, beta, S0 = data
    vf = torch.ones(T, 1, dtype=torch.bool)
    _o, S, S_m = la._scan_mid_export(q, k, v, g, beta, S0, torch.tensor(mid_k), vf, keep_pad=False, ug=6)
    assert calls[0] == (T, True) and calls[1] == (dr.L, False)  # one full pass, one 128-row call
    want = _two_scan(data, mid_k)
    assert _eq(S_m, want), f"mid_k {mid_k}: {int((_bits(S_m) != _bits(want)).sum())} differ"
    assert _eq(S, states[1])


@needs_nki
def test_plant_a_wrong_k_is_seen(data, states, monkeypatch):
    """The comparison bites: the export at k 200 against the second scan at k 201 must differ."""
    monkeypatch.setattr(
        dr, "chunk",
        lambda q, k, v, g, beta, S0, keep_pad=False, ug=None, states=False, mrow=None: _sim_chunk(
            q, k, v, g, beta, S0, ug=ug, states=states, mrow=mrow
        ),
    )  # fmt: skip
    q, k, v, g, beta, S0 = data
    _, _, S_m = la._scan_mid_export(
        q, k, v, g, beta, S0, torch.tensor(200), torch.ones(T, 1, dtype=torch.bool), keep_pad=False, ug=6
    )
    assert not _eq(S_m, _two_scan(data, 201))


@needs_nki
def test_padding_rows_stay_masked(data, states, monkeypatch):
    """Rows past the chunk's valid length are padding (vf False): the export masks them as the second scan does."""
    monkeypatch.setattr(
        dr, "chunk",
        lambda q, k, v, g, beta, S0, keep_pad=False, ug=None, states=False, mrow=None: _sim_chunk(
            q, k, v, g, beta, S0, ug=ug, states=states, mrow=mrow
        ),
    )  # fmt: skip
    q, k, v, g, beta, S0 = data
    n_valid = 300
    vf = (torch.arange(T) < n_valid).view(T, 1)
    beta_v = torch.where(vf, beta, torch.zeros_like(beta))
    g_v = torch.where(vf.view(T, 1, 1), g, torch.zeros_like(g))
    _, _, S_m = la._scan_mid_export(q, k, v, g_v, beta_v, S0, torch.tensor(T), vf, keep_pad=False, ug=6)
    _, S_full = _sim_chunk(q, k, v, g_v, beta_v, S0, ug=6)
    assert _eq(S_m, S_full)  # k = T (no checkpoint in the chunk): the scratch row gets the chunk's end
