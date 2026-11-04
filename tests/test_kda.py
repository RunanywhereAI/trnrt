"""Kimi Delta Attention outside the decode shell: the prefill kernels and the decode form's conv window.

- kernels/delta_rule.py, the NKI chunked delta rule: its algorithm (emulate) against the float64 token-by-token
  recurrence and the torch path (linear_attn.chunk_scan) on the host, from a non-zero state, including the strongest
  decay the safe gate allows; the kernel in the NKI simulator against emulate at the served unit group.
- linear_attn.chunk_scan against the recurrence, and stable on nearly parallel keys (the regime of trained weights).
- kernels/short_conv.py: never taken on the host; in the simulator, the prefill chunk's causal conv bit for bit
  linear_attn._causal_conv's at grids 1 and 2.
- The decode form's conv window: read and written as one-hot matmuls at two rows and more (no indirect DMA), exactly
  conv_state[slot] and its index_put_; a row alone keeps the direct copy; _mix_rows' decode form equals the index form;
  and through a tiny GLM-5.3-Flash engine (CPU, piecewise prefill, whole-step decode) the prefill's chunk form writes
  the window the decode then reads (the composition), with the same tokens and logprobs as the index form, and a
  window read in the wrong order changes the decode and no prompt's first token (the positive path).
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from trnrt.kernels import delta_rule as dr
from trnrt.models import linear_attn as LA
from trnrt.models.linear_attn import chunk_scan, recurrent_step

needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")


# -- the delta rule --------------------------------------------------------------------------------------------------


def inputs(T, H, seed=0, lower_bound=-5.0, correlated=False, D=128):
    """Torch-path inputs as linear_attn builds them for KDA: l2-normalised q (scaled) and k, per-channel gates in
    [lower_bound, 0) (the safe gate, lower_bound * sigmoid)."""
    g = torch.Generator().manual_seed(seed)
    if correlated:  # nearly parallel keys, beta near 1, slow decay: the regime of trained weights
        base = torch.randn(H, D, generator=g)
        k = torch.nn.functional.normalize(base + 0.05 * torch.randn(T, H, D, generator=g), dim=-1)
        beta = 0.9 + 0.1 * torch.rand(T, H, generator=g)
    else:
        k = torch.nn.functional.normalize(torch.randn(T, H, D, generator=g), dim=-1)
        beta = torch.rand(T, H, generator=g)
    q = torch.nn.functional.normalize(torch.randn(T, H, D, generator=g), dim=-1) * D**-0.5
    v = torch.randn(T, H, D, generator=g)
    gate = lower_bound * torch.sigmoid(torch.randn(T, H, D, generator=g) * 2 - (4 if correlated else 0))
    S0 = torch.randn(H, D, D, generator=g) * 0.1
    return q, k, v, gate, beta, S0


def recurrence(q, k, v, g, beta, S0):
    """Token by token in float64 (the reference of the chunked forms)."""
    S, outs = S0.double().unsqueeze(0), []
    for t in range(q.shape[0]):
        o, S = recurrent_step(*(x[t : t + 1].double() for x in (q, k, v, g, beta)), S)
        outs.append(o)
    return torch.cat(outs), S[0]


@pytest.mark.parametrize("H,T", [(2, 300), (1, 128), (2, 4096)])
@pytest.mark.parametrize("correlated", [False, True])
def test_emulation_matches_the_recurrence(H, T, correlated):
    """The kernel's factorisation (16-row reference sub-chunks, the transposed blocked inverse, the chunk as an affine
    map of S) equals the float64 recurrence to float32 rounding, padding a ragged last chunk, from a non-zero state;
    chunk_scan (the torch path) too. 4096 rows: a served prefill chunk at two heads a rank."""
    if T > 1024 and not correlated:
        pytest.skip("one long case is enough")
    q, k, v, gate, beta, S0 = inputs(T, H, correlated=correlated)
    want, S_want = recurrence(q, k, v, gate, beta, S0)
    got, S_got = dr.emulate(q, k, v, gate, beta, S0)
    scale = want.abs().max().item()
    assert (got.double() - want).abs().max().item() < 2e-5 * max(1.0, scale)
    assert (S_got.double() - S_want).abs().max().item() < 2e-5 * max(1.0, S_want.abs().max().item())
    ref, S_ref = chunk_scan(q, k, v, gate, beta, S0, 64)
    assert (got - ref).abs().max().item() < 2e-5 * max(1.0, scale)


def test_strong_decay_stays_finite():
    """Every gate at the lower bound (-5 per token, e^-640 over a chunk): the reference sub-chunks keep every factor
    finite, the result is the recurrence's."""
    q, k, v, gate, beta, S0 = inputs(256, 1, seed=4)
    gate = torch.full_like(gate, -4.999)
    want, S_want = recurrence(q, k, v, gate, beta, S0)
    got, S_got = dr.emulate(q, k, v, gate, beta, S0)
    assert torch.isfinite(got).all() and torch.isfinite(S_got).all()
    assert (got.double() - want).abs().max().item() < 2e-5 * max(1.0, want.abs().max().item())


def test_constants():
    c = dr.consts_np()
    I, U, BL, BU = (torch.tensor(c[:, i]) for i in (dr.C_I, dr.C_U, dr.C_BL, dr.C_BU))
    assert torch.equal(I, torch.eye(128)) and torch.equal(BU, BL.T)
    assert torch.equal(U, torch.triu(torch.ones(128, 128)))
    for i, s in enumerate(dr.LEVELS):
        lo = torch.tensor(c[:, dr.C_LO + i])
        assert lo.sum().item() == 128 // (2 * s) * s * s


@needs_nki
@pytest.mark.parametrize("H,T,ug", [(2, 256, 6), (2, 256, None), (3, 128, 6), (2, 512, 6)])
def test_nki_simulator_matches_emulation(H, T, ug, monkeypatch):
    """The kernel at the model's unit group (6: linear_attn's prefill call) and at the module default."""
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    q, k, v, gate, beta, S0 = inputs(T, H, seed=2, correlated=True)
    args = dr.kernel_inputs(q, k, v, gate, beta, S0, device=torch.device("cpu"), ug=ug)
    o, S = nki.simulate(dr.kernel())(**args)
    want, S_want = dr.emulate(q, k, v, gate, beta, S0)
    o, S = torch.as_tensor(o), torch.as_tensor(S)
    assert (o - want).abs().max().item() < 1e-5 * max(1.0, want.abs().max().item())
    assert (S - S_want).abs().max().item() < 1e-5 * max(1.0, S_want.abs().max().item())


# -- chunk_scan -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [4, 16, 32, 64])
def test_chunk_scan_matches_the_recurrence(chunk):
    """chunk_scan (sub-chunks, (I - N)^-1, padding) equals the delta rule token by token, from a non-zero state, with
    per-channel decay."""
    g = torch.Generator().manual_seed(3)
    T, H, Dk, Dv = 37, 3, 8, 5
    q = torch.nn.functional.normalize(torch.randn(T, H, Dk, generator=g), dim=-1) * Dk**-0.5
    k = torch.nn.functional.normalize(torch.randn(T, H, Dk, generator=g), dim=-1)
    v = torch.randn(T, H, Dv, generator=g)
    gate = -torch.rand(T, H, Dk, generator=g) * 3
    beta = torch.rand(T, H, generator=g)
    S0 = torch.randn(H, Dk, Dv, generator=g)
    S, want = S0.unsqueeze(0), []
    for t in range(T):
        o, S = recurrent_step(q[t : t + 1], k[t : t + 1], v[t : t + 1], gate[t : t + 1], beta[t : t + 1], S)
        want.append(o)
    got, S_got = chunk_scan(q, k, v, gate, beta, S0, chunk)
    assert (got - torch.cat(want)).abs().max().item() < 1e-4
    assert (S_got - S[0]).abs().max().item() < 1e-4


@pytest.mark.parametrize("chunk", [16, 32, 64])
def test_chunk_scan_is_stable_on_correlated_keys(chunk):
    """Nearly parallel keys, beta near 1 and slow decay: the sub-chunk's triangular system has an inverse of modest size
    whose Neumann terms are huge. Inverting it by repeated squaring over the whole sub-chunk was off by 1.2e3 (32
    tokens) and 7.6e18 (64) here; the blocked inverse (linear_attn._unit_lower_inverse) is exact to rounding."""
    g = torch.Generator().manual_seed(0)
    T, H, Dk, Dv = 128, 2, 32, 16
    base = torch.randn(H, Dk, generator=g)
    k = torch.nn.functional.normalize(base + 0.05 * torch.randn(T, H, Dk, generator=g), dim=-1)
    q = torch.nn.functional.normalize(torch.randn(T, H, Dk, generator=g), dim=-1) * Dk**-0.5
    v = torch.randn(T, H, Dv, generator=g)
    gate = -0.01 * torch.rand(T, H, Dk, generator=g)
    beta = 0.9 + 0.1 * torch.rand(T, H, generator=g)
    S, want = torch.zeros(1, H, Dk, Dv, dtype=torch.float64), []
    for t in range(T):
        o, S = recurrent_step(*(x[t : t + 1].double() for x in (q, k, v, gate, beta)), S)
        want.append(o)
    got, S_got = chunk_scan(q, k, v, gate, beta, torch.zeros(H, Dk, Dv), chunk)
    assert (got - torch.cat(want).float()).abs().max().item() < 1e-5
    assert (S_got - S[0].float()).abs().max().item() < 1e-4


# -- the short conv of a prefill chunk ------------------------------------------------------------------------------


def _conv_inputs(T: int, C: int, K: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    xe = (torch.randn(K - 1 + T, C, generator=g) * 0.5).to(torch.bfloat16)
    w = (torch.randn(C, K, generator=g) * 0.3).to(torch.bfloat16)
    return xe, w


def test_the_short_conv_kernel_is_never_taken_on_the_host():
    from trnrt.kernels import short_conv as sc

    xe, w = _conv_inputs(16, 256)
    assert sc.KERNEL == "nki" and not sc.takes(xe, w, 16)  # a CPU tensor: the torch path
    assert not sc.takes(xe.to("meta"), w.to("meta"), 1)  # one row: a decode form


@needs_nki
@pytest.mark.parametrize("T,C", [(1024, 256), (300, 384), (128, 128), (4096, 768)])
def test_short_conv_kernel_equals_causal_conv(monkeypatch, T, C):
    """y = sum_j xe[j : j + T] w[:, j] in fp32: the kernel's grid 1 and grid 2 outputs equal _causal_conv's bits
    (768: a rank's q, k and v channels at tp 32)."""
    import nki

    from trnrt.kernels import short_conv as sc
    from trnrt.models.linear_attn import _causal_conv

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    xe, w = _conv_inputs(T, C)
    want = _causal_conv(xe, w, T)
    args = dict(xe=xe, w=w, ib=torch.eye(128).to(torch.bfloat16), i32=torch.eye(128), rev=sc.REV)
    one = torch.as_tensor(nki.simulate(sc.short_conv_kernel)(**args))
    two = torch.as_tensor(nki.simulate(sc.short_conv_kernel[2])(**args))
    assert one.shape == (T, C) and one.dtype == torch.float32
    assert torch.equal(one, want)
    assert torch.equal(two, want)


# -- the decode form's conv window ------------------------------------------------------------------------------------


def pool(seed=0, S=25, K=4, D=768, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(S, K - 1, D, generator=g).to(dtype), g


@pytest.mark.parametrize("B", [1, 2, 4, 6, 8])
def test_read_and_write_are_the_index_forms(B):
    cs, g = pool(B)
    slot = torch.randperm(cs.shape[0] - 1, generator=g)[:B] + 1  # distinct, never the scratch row 0
    assert torch.equal(LA._conv_read(cs, slot), cs[slot])
    win = torch.randn(B, *cs.shape[1:], generator=g).to(cs.dtype)
    want, got = cs.clone(), cs.clone()
    LA._write_rows(want, slot, win)
    LA._conv_write(got, slot, win)
    assert torch.equal(got[1:], want[1:])  # _write_rows' one-row form also writes the scratch row
    assert torch.equal(got[slot], win)


def test_rows_sharing_the_scratch_slot_change_only_it():
    cs, g = pool(7)
    slot = torch.tensor([3, 0, 9, 0])  # two padded rows on the scratch row
    win = torch.randn(4, *cs.shape[1:], generator=g).to(cs.dtype)
    got = cs.clone()
    LA._conv_write(got, slot, win)
    assert torch.equal(got[3], win[0]) and torch.equal(got[9], win[2])
    keep = torch.ones(cs.shape[0], dtype=torch.bool)
    keep[[0, 3, 9]] = False
    assert torch.equal(got[keep], cs[keep])


def _index_form(mp):
    """The index form of the window's read and write (conv_state[slot] and _write_rows), patched in."""
    mp.setattr(LA, "_conv_read", lambda cs, slot: cs[slot])
    mp.setattr(LA, "_conv_write", lambda cs, slot, win: LA._write_rows(cs, slot, win))


@pytest.fixture(scope="module")
def flash(tmp_path_factory):
    from tests import test_glm5_next as G

    path = str(tmp_path_factory.mktemp("kda_conv_decode"))
    G.build(path, seed=5)
    return path


def make_engine(path, budget=64):
    from tests.test_glm5_next import engine

    return engine(path, max_num_seqs=8, max_prefill_tokens=budget)


@pytest.mark.parametrize("B", [2, 4, 8])
def test_the_decode_form_is_the_index_form(flash, monkeypatch, B):
    """_mix_rows' decode form on a KDA layer of the tiny engine, from the same state rows: output, conv_state and
    rec_state equal to the index form's. One row starts a sequence (position 0: the window is masked to zeros)."""
    eng = make_engine(flash)
    try:
        m = eng.model
        L = next(x for x in m.layers if hasattr(x, "conv_dim"))
        g = torch.Generator().manual_seed(B)
        S = L.conv_state.shape[0]
        conv0 = torch.randn(L.conv_state.shape, generator=g).to(L.conv_state.dtype)
        rec0 = torch.randn(L.rec_state.shape, generator=g).to(L.rec_state.dtype) * 0.1
        x = torch.randn(B, m.cfg.hidden_size, generator=g).to(m.dtype)
        positions = torch.arange(B) + 5
        positions[0] = 0
        slot = torch.randperm(S - 1, generator=g)[:B] + 1
        slot_mapping = torch.zeros(B, dtype=torch.long)
        out = {}
        for form in ("index", "onehot"):
            with monkeypatch.context() as mp:
                if form == "index":
                    _index_form(mp)
                L.conv_state.copy_(conv0)
                L.rec_state.copy_(rec0)
                with torch.no_grad():
                    y = LA._mix_rows(m, L, x, positions, slot_mapping, slot)
                out[form] = (y.clone(), L.conv_state[1:].clone(), L.rec_state[1:].clone())
        for name, a, b in zip(("output", "conv_state", "rec_state"), out["onehot"], out["index"]):
            assert torch.equal(a, b), name
        assert not torch.equal(out["index"][1], conv0[1:])  # the decode wrote its windows
    finally:
        eng.close()


def test_one_row_keeps_the_index_form(flash, monkeypatch):
    """One row compiles to a direct copy already, so a one-row decode never calls _conv_read / _conv_write; two rows do
    (the positive control)."""

    def boom(*a):
        raise AssertionError("the one-hot path")

    eng = make_engine(flash)
    try:
        m = eng.model
        L = next(x for x in m.layers if hasattr(x, "conv_dim"))
        monkeypatch.setattr(LA, "_conv_read", boom)
        monkeypatch.setattr(LA, "_conv_write", boom)
        x = torch.zeros(2, m.cfg.hidden_size, dtype=m.dtype)
        with torch.no_grad():
            LA._mix_rows(m, L, x[:1], torch.tensor([3]), torch.zeros(1, dtype=torch.long), torch.tensor([2]))
            with pytest.raises(AssertionError, match="the one-hot path"):
                LA._mix_rows(m, L, x, torch.tensor([3, 4]), torch.zeros(2, dtype=torch.long), torch.tensor([2, 5]))
    finally:
        eng.close()


def serve(path, monkeypatch, form: str, budget: int = 64, reverse: bool = False):
    from tests.test_glm5_next import prompts
    from trnrt.engine.request import SamplingParams

    with monkeypatch.context() as mp:
        if form == "index":
            _index_form(mp)
        if reverse:  # the mutation: each window's K - 1 inputs read in reverse order
            real = LA._conv_read
            mp.setattr(LA, "_conv_read", lambda cs, slot: real(cs, slot).flip(1))
        eng = make_engine(path, budget)
        try:
            sp = SamplingParams(max_new_tokens=6, ignore_eos=True, logprobs=0)
            reqs = [eng.add_request(p, sp) for p in prompts(13, [12, 14, 6, 11, 7, 13, 10, 8])]
            while eng.has_work():
                eng.step()
            return [(r.output_ids, [x[0] for x in r.logprobs]) for r in reqs]
        finally:
            eng.close()


def test_the_engine_serves_the_index_forms_tokens_and_logprobs(flash, monkeypatch):
    """Eight requests, each prefilled in one chunk (budget 64) and then decoded at 2 to 8 rows: the chunk form writes
    the window in conv_state's own layout and the decode reads it through the one-hot form (and hands it on)."""
    assert serve(flash, monkeypatch, "onehot") == serve(flash, monkeypatch, "index")


def test_a_reversed_window_changes_the_decode_not_the_prefill(flash, monkeypatch):
    want = serve(flash, monkeypatch, "onehot")
    broken = serve(flash, monkeypatch, "onehot", reverse=True)
    assert all(b[0][0] == w[0][0] for b, w in zip(broken, want))  # every prompt's first token: prefill
    assert broken != want  # the decode reads the window through _conv_read
