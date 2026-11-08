"""kernels/kda_decode.py: the KDA decode step kernel's emulation against models/linear_attn.py's decode form, its
route (the decode form takes it on the device; the KDA shell's row cap decides which steps reach the decode form),
and the kernel in the NKI simulator against its emulation, at grid 1 and split over the two programs of an LNC 2 core
(rows that start from a zero state, a shared padding write row)."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from trnrt.kernels import kda_decode as kd
from trnrt.models import hybrid
from trnrt.models import linear_attn as LA

needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")


def inputs(B=6, R=9, H=4, D=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    pool = torch.randn(R, H, D, D, generator=g) * 0.05
    slot = torch.randperm(R - 1, generator=g)[:B]
    keep = torch.ones(B, dtype=torch.bool)
    slot[-2:], keep[-2:] = R - 1, False  # two padding rows: the scratch row, from zero
    q = torch.nn.functional.normalize(torch.randn(B, H, D, generator=g), dim=-1) * D**-0.5
    k = torch.nn.functional.normalize(torch.randn(B, H, D, generator=g), dim=-1)
    v = torch.randn(B, H, D, generator=g)
    gl = -5.0 * torch.sigmoid(torch.randn(B, H, D, generator=g))
    beta = torch.sigmoid(torch.randn(B, H, generator=g))
    return pool, slot, keep, q, k, v, gl, beta


def _slots(pool, slot, keep):
    R = pool.shape[0]
    return torch.stack([torch.where(keep, slot, torch.full_like(slot, R)), slot]).to(torch.int32)


def test_emulation_is_the_decode_form():
    pool, slot, keep, q, k, v, g, beta = inputs()
    R = pool.shape[0]
    want_pool = pool.clone()
    S = want_pool[slot]
    S = torch.where(keep.view(-1, 1, 1, 1), S, torch.zeros_like(S))
    o_want, S = LA.recurrent_step(q, k, v, g, beta, S)
    LA._write_rows(want_pool, slot, S)
    o, got_pool = kd.emulate(pool, _slots(pool, slot, keep), q, k, v, g, beta)
    real = keep.nonzero().flatten()
    assert torch.equal(o[real], o_want[real])
    assert torch.equal(got_pool[slot[real]], want_pool[slot[real]])
    other = torch.ones(R, dtype=torch.bool)
    other[slot] = False
    assert torch.equal(got_pool[other], pool[other])


def test_takes_a_kda_step_of_128_wide_heads_on_the_device_only():
    dev, cpu = torch.empty(2, 8, device="meta"), torch.empty(2, 8)
    assert kd.takes(dev, 128, 128, True)
    assert not kd.takes(cpu, 128, 128, True)  # the CPU keeps the einsums
    assert not kd.takes(dev, 128, 128, False)  # a scalar (GDN) gate
    assert not kd.takes(dev, 64, 128, True) and not kd.takes(dev, 128, 64, True)


def test_the_shell_cap_sends_16_and_24_row_steps_to_the_decode_form(monkeypatch):
    """At the mtp-* presets' cap of 12 rows, every step up to 12 rows takes the KDA shell and never reaches the decode
    form, so the kernel cannot move one of their graphs; mtp-batch24's 16- and 24-row steps take the decode form."""
    monkeypatch.setattr(hybrid, "KDA_SHELL_MAXB", 12)
    assert all(hybrid.kda_shell_rows(b) for b in (1, 2, 4, 6, 8, 12))
    assert not hybrid.kda_shell_rows(16) and not hybrid.kda_shell_rows(24)


@pytest.fixture(scope="module")
def flash(tmp_path_factory):
    from tests import test_glm5_next as G

    path = str(tmp_path_factory.mktemp("kda_decode_route"))
    G.build(path, seed=7)
    return path


def test_the_decode_form_runs_the_kernel_where_it_takes(flash, monkeypatch):
    """_mix_rows' decode form on a KDA layer of the tiny engine: with takes() true (the device) it calls
    decode_step with the form's pool, state rows, keep mask and q, k, v, g, beta, and returns its output; the stand-in
    computes the kernel's emulation, so output and rec_state equal the einsum form's. Dropping the route fails here."""
    from tests.test_glm5_next import engine

    eng = engine(flash, max_num_seqs=8, max_prefill_tokens=64)
    try:
        m = eng.model
        L = next(x for x in m.layers if hasattr(x, "conv_dim"))
        B = 4
        g = torch.Generator().manual_seed(B)
        S = L.conv_state.shape[0]
        conv0 = torch.randn(L.conv_state.shape, generator=g).to(L.conv_state.dtype)
        rec0 = torch.randn(L.rec_state.shape, generator=g).to(L.rec_state.dtype) * 0.1
        x = torch.randn(B, m.cfg.hidden_size, generator=g).to(m.dtype)
        positions = torch.arange(B) + 5
        positions[0] = 0
        slot = torch.randperm(S - 1, generator=g)[:B] + 1
        calls = []

        def stand_in(pool, state_slot, keep, q, k, v, g_, beta):
            calls.append((pool is L.rec_state, state_slot.tolist(), keep.tolist()))
            R = pool.shape[0]
            slots = torch.stack([torch.where(keep, state_slot, torch.full_like(state_slot, R)), state_slot])
            o, new = kd.emulate(pool, slots, q, k, v, g_, beta)
            pool.copy_(new)
            return o

        out = {}
        for route in ("kernel", "einsum"):
            with monkeypatch.context() as mp:
                if route == "kernel":
                    mp.setattr(kd, "takes", lambda *a: True)
                    mp.setattr(kd, "decode_step", stand_in)
                L.conv_state.copy_(conv0)
                L.rec_state.copy_(rec0)
                with torch.no_grad():
                    y = LA._mix_rows(m, L, x, positions, torch.zeros(B, dtype=torch.long), slot)
                out[route] = (y.clone(), L.rec_state.clone())
        assert calls == [(True, slot.tolist(), [False, True, True, True])]  # position 0 starts from a zero state
        assert torch.allclose(out["kernel"][0].float(), out["einsum"][0].float(), rtol=0, atol=1e-2)
        assert torch.allclose(out["kernel"][1], out["einsum"][1], rtol=0, atol=1e-5)
    finally:
        eng.close()


@needs_nki
@pytest.mark.parametrize("grid", [1, 2])
def test_nki_simulator_matches_emulation(monkeypatch, grid):
    """grid 2 with spl 1: each program runs its half of the rows (the served form at LNC 2)."""
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    pool, slot, keep, q, k, v, g, beta = inputs(H=8)
    slots = _slots(pool, slot, keep)
    o_want, pool_want = kd.emulate(pool, slots, q, k, v, g, beta)
    kern = kd.kda_decode_kernel if grid == 1 else kd.kda_decode_kernel[2]
    extra = {} if grid == 1 else {"spl": 1}
    o, pool_out = nki.simulate(kern)(
        pool=pool.clone(), slots=slots, q=q, k=k, v=v, g=g, beta=beta, ident=torch.eye(128), rev=kd.REV, **extra
    )
    o, pool_out = torch.as_tensor(o), torch.as_tensor(pool_out)
    real = keep.nonzero().flatten()
    assert (o[real] - o_want[real]).abs().max().item() < 1e-5
    assert (pool_out[slot[real]] - pool_want[slot[real]]).abs().max().item() < 1e-5
