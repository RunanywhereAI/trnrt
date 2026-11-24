"""kernels/kda_decode.py: the KDA decode step kernel's emulation against models/linear_attn.py's decode form, its
route (the decode form takes it on the device; the KDA shell's row cap decides which steps reach the decode form),
the kernel's source (no PSUM accumulation group: on trn2 such a group lost its started term in each program's first
row), and the kernel in the NKI simulator against its emulation, at grid 1 and split over the two programs of an LNC 2
core (rows that start from a zero state, a shared padding write row, an odd row count: a partial chunk)."""

from __future__ import annotations

import importlib.util
import inspect

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
    assert kd.takes(dev, 128, 128, True, 2)
    assert not kd.takes(cpu, 128, 128, True, 2)  # the CPU keeps the einsums
    assert not kd.takes(dev, 128, 128, False, 2)  # a scalar (GDN) gate
    assert not kd.takes(dev, 64, 128, True, 2) and not kd.takes(dev, 128, 64, True, 2)


@pytest.mark.parametrize("H", [1, 3, 4, 8])
def test_takes_only_the_rings_two_heads_a_row(H):
    """The kernel's ring (RING rows: two chunks of CH // 2 rows and PD ahead) is sized for H 2 and _body asserts it, so a
    layer of another per-rank head count keeps the einsums instead of failing in the trace."""
    assert kd.CH // 2 * 2 + kd.PD == kd.RING
    assert not kd.takes(torch.empty(2, 8, device="meta"), 128, 128, True, H)


def test_the_decode_form_passes_its_per_rank_head_count():
    src = inspect.getsource(LA._mix_rows)
    assert 'kda_decode.takes(x, dk, dv, sp.kind == "kda" and g.dim() == 3, nv)' in src
    assert "nk, nv, dk, dv = layer.nk, layer.nv, sp.head_k_dim, sp.head_v_dim" in src


def test_the_shell_cap_sends_16_and_24_row_steps_to_the_decode_form(monkeypatch):
    """At a shell cap of 12 rows, every step up to 12 rows takes the KDA shell and never reaches the decode form; 16-
    and 24-row steps take the decode form."""
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


KERNEL_SPAN = ("try:  # the Neuron venv", "    kda_decode_kernel = None")


def test_the_kernel_has_no_psum_accumulation_group():
    """Every matmul of the kernel is a lone start (accumulate=False): the r1 - v accumulation group of the first form
    lost its started term on trn2 (head 2's r1 = 0 in each program's first row), which the simulator does not model."""
    import inspect

    src = inspect.getsource(kd)
    span = src[src.index(KERNEL_SPAN[0]) : src.index(KERNEL_SPAN[1])]
    assert span.count("nisa.nc_matmul(") == 7  # the matvecs, the outer product and the prologue's four
    assert "accumulate=True" not in span and "accumulate=(" not in span
    assert span.count("accumulate=False") == span.count("nisa.nc_matmul(")


@needs_nki
@pytest.mark.parametrize(("grid", "B"), [(2, 24), (2, 16), (2, 8), (2, 5), (1, 7)])
def test_nki_simulator_matches_emulation(monkeypatch, grid, B):
    """grid 2 with spl 1: each program runs its half of the rows (the served form at LNC 2); B 5 leaves each program a
    partial chunk. H = 2 heads (the ring's sizing; the served shape at tp 32)."""
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    pool, slot, keep, q, k, v, g, beta = inputs(B=B, R=64, H=2, seed=1000 + 10 * B + grid)
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
