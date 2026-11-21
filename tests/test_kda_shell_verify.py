"""kernels/kda_shell.py verify(): a speculative verify of B sequences x Q positions through the KDA shell, and the build
switch that turns the shell off (models/hybrid.py KDA_SHELL, EngineConfig.kda_shell).

Host (hybrid.SHELL_HOST: each position's arithmetic as kda_shell.emulate):
- the composition against the served XLA attention block whose KDA half is linear_attn's own verify form (_mix_rows
  with state_slot [B, 1 + Q]: the causal conv over [history | x_1 .. x_Q], verify_scan), the states and conv windows
  written after every position included, with a positive control (position 1 from the read state) that must fail;
- position 0 against decode() bit for bit;
- the device call: ONE run() of every row in sequence order, vq = Q, position 0's read row (the zero row at a sequence
  start) and the zero row for later positions, each position's write row, the pend whole, one all-reduce;
- the gate: a verify of at most KDA_SHELL_VERIFY positions, B Q rows, under KDA_SHELL_MAXB; layer() and the split layer
  route a verify to verify();
- with KDA_SHELL off a KDA layer gets no blobs and no call takes the shell (EngineConfig.kda_shell's route:
  tests/test_kda_routes.py).
The kernel's one call against Q decode calls in the simulator is tests/test_kda_shell.py's (Neuron venv)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests import test_kda_shell as TKS
from trnrt.kernels import kda_half as KH
from trnrt.kernels import kda_shell as KS
from trnrt.kernels import mk_kda_layer as MK
from trnrt.models import hybrid as HY
from trnrt.models import linear_attn as LA

H, HC, BF = TKS.H, TKS.HC, TKS.bf
STATE_SLOT = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # [B 3, 1 + Q]: read, after position 0, after position 1
POSITIONS = torch.tensor([[0, 1], [9, 10], [4, 5]])  # sequence 0 starts here: zero states, its read row unread


@pytest.fixture(autouse=True)
def _torch_threads():
    n = torch.get_num_threads()
    torch.set_num_threads(4)  # test_kda_shell's pin: the emulations' bf16 reductions round by thread count
    yield
    torch.set_num_threads(n)


def _layer(monkeypatch, on: bool = True, npg: int = 2, seed: int = 1, R: int = 12):
    """test_kda_shell's stand-in KDA layer as an nn.Module, then hybrid.init_layer with KDA_SHELL on or off."""
    monkeypatch.setattr(HY, "KDA_SHELL", on)
    monkeypatch.setattr(HY, "LinearSpec", SimpleNamespace)  # the stand-in spec counts as a linear-attention spec
    monkeypatch.setattr(KH, "half_programs", lambda: npg)
    _, L, g = TKS.stand_in(seed, R=R)
    layer = nn.Module()
    for k, v in vars(L).items():
        setattr(layer, k, v)
    layer.spec.gate_rank = 128
    layer.pack_moe = False

    def p(*shape):
        return nn.Parameter(torch.empty(*shape, dtype=BF), requires_grad=False)

    layer.conv_state[-1] = 0  # the zero row a sequence start reads (shell.start_row): zeros, as the state pool keeps it
    layer.rec_state[-1] = 0
    HY.init_layer(layer, TKS._cfg(), layer.spec, 0, 32, p)
    layer.hc_attn_fn.data.copy_((torch.randn(2 * HC + HC * HC, HC * H, generator=g) * 0.02).to(BF))
    layer.hc_attn_base.data.copy_(torch.randn(2 * HC + HC * HC, generator=g) * 0.5)
    layer.hc_attn_scale.data.copy_(torch.rand(3, generator=g) + 0.5)
    return layer, g


def _model():
    return SimpleNamespace(cfg=TKS._cfg(), dtype=BF, page_size=256, _w=getattr, _attn_all_reduce=lambda x: x)


def _pend(g, T):
    return (
        torch.rand(T, HC, generator=g) * 2,
        torch.softmax(torch.randn(T, HC, HC, generator=g), -1),
        (torch.randn(T, H, generator=g) * 0.3).to(BF),
    )


@pytest.mark.parametrize(("npg", "mix"), [(1, False), (2, False), (2, True)])
def test_verify_is_the_xla_verify_form(monkeypatch, npg, mix):
    layer, g = _layer(monkeypatch, npg=npg, seed=7)
    monkeypatch.setattr(HY, "SHELL_HOST", True)
    B, Q = STATE_SLOT.shape[0], STATE_SLOT.shape[1] - 1
    T = B * Q
    model = _model()
    streams = (torch.randn(T, HC * H, generator=g) * 0.5).to(BF)
    pend = _pend(g, T) if mix else None
    positions = POSITIONS.reshape(T)
    conv0, rec0 = layer.conv_state.clone(), layer.rec_state.clone()
    with torch.no_grad():
        S = streams.view(T, HC, H)
        if mix:  # the XLA path mixes the previous block's output in first (hybrid.layer's non-shell branch)
            S = HY.mix_out(*pend, S, True)
        served = HY._block(
            model,
            layer,
            "attn",
            S.reshape(T, HC * H),
            layer.in_norm,
            lambda xn: LA._mix_rows(model, layer, xn, positions, torch.zeros(T, dtype=torch.long), STATE_SLOT),
            False,
        )
        want_conv, want_rec = layer.conv_state.clone(), layer.rec_state.clone()
        layer.conv_state.copy_(conv0)
        layer.rec_state.copy_(rec0)
        Sv, (post, comb, Y) = KS.verify(
            model, layer, streams.view(T, HC, H), positions, STATE_SLOT, pend=HY._shell_pend(layer, pend)
        )
        shelled = HY.mix_out(post, comb, Y.float().sum(0).to(BF), Sv, True)
    d = (shelled.float() - served.float()).abs()
    assert torch.allclose(shelled.float(), served.float(), rtol=2e-2, atol=4e-3), d.max()
    assert d.mean() < 2e-3 * served.float().abs().mean(), d.mean()
    rows = STATE_SLOT[:, 1:].reshape(-1)
    rest = torch.ones(layer.rec_state.shape[0], dtype=torch.bool)
    rest[rows] = False
    assert torch.equal(layer.conv_state[rest], conv0[rest]) and torch.equal(layer.rec_state[rest], rec0[rest])
    assert torch.equal(layer.conv_state[rows, :-1], want_conv[rows, :-1])  # the shifted history, bit for bit
    assert torch.allclose(layer.conv_state[rows].float(), want_conv[rows].float(), rtol=1e-2, atol=1e-3)
    dr = (layer.rec_state[rows] - want_rec[rows]).abs()
    assert torch.allclose(layer.rec_state[rows], want_rec[rows], rtol=1e-3, atol=1e-4), dr.max()
    # the states after position 1 are not after position 0's: the second position really read the first's write
    assert not torch.allclose(want_rec[STATE_SLOT[:, 2]], want_rec[STATE_SLOT[:, 1]], rtol=1e-2, atol=1e-3)


def test_position_one_from_the_read_state_is_caught(monkeypatch):
    """Positive control: position 1 starting from the sequence's read state (position 0's update lost: two independent
    decodes) is NOT the verify form at the state bound above."""
    layer, g = _layer(monkeypatch, npg=2, seed=8)
    B, Q = 3, 2
    T = B * Q
    model = _model()
    streams = (torch.randn(T, HC * H, generator=g) * 0.5).to(BF)
    positions = POSITIONS.reshape(T)
    conv0, rec0 = layer.conv_state.clone(), layer.rec_state.clone()
    R = rec0.shape[0]
    with torch.no_grad():
        HY._block(
            model,
            layer,
            "attn",
            streams,
            layer.in_norm,
            lambda xn: LA._mix_rows(model, layer, xn, positions, torch.zeros(T, dtype=torch.long), STATE_SLOT),
            False,
        )
        want = layer.rec_state[STATE_SLOT[:, 2]].clone()
        layer.conv_state.copy_(conv0)
        layer.rec_state.copy_(rec0)
        Ss = streams.view(B, Q, HC, H)
        rd = torch.where(POSITIONS[:, 0] > 0, STATE_SLOT[:, 0], torch.full_like(STATE_SLOT[:, 0], R - 1))  # S0 twice
        for q, wr in ((0, torch.tensor([10, 11, 0])), (1, STATE_SLOT[:, 2])):
            KS._host(model, layer, Ss[:, q], torch.stack([rd, wr]).to(torch.int32), None)
    assert not torch.allclose(layer.rec_state[STATE_SLOT[:, 2]], want, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("mix", [False, True])
def test_position_zero_is_decode_bit_for_bit(monkeypatch, mix):
    layer, g = _layer(monkeypatch, npg=2, seed=9)
    monkeypatch.setattr(HY, "SHELL_HOST", True)
    B, Q = 3, 2
    T = B * Q
    model = _model()
    streams = (torch.randn(T, HC, H, generator=g) * 0.5).to(BF)
    pend = HY._shell_pend(layer, _pend(g, T)) if mix else None
    positions = POSITIONS.reshape(T)
    conv0, rec0 = layer.conv_state.clone(), layer.rec_state.clone()
    with torch.no_grad():
        Sv, (pv, cv, Yv) = KS.verify(model, layer, streams, positions, STATE_SLOT, pend=pend)
        w1_rec, w1_conv = layer.rec_state[STATE_SLOT[:, 1]].clone(), layer.conv_state[STATE_SLOT[:, 1]].clone()
        layer.conv_state.copy_(conv0)
        layer.rec_state.copy_(rec0)
        # decode on position 0's rows: it reads and writes one row, so give it the read row's contents at row w1
        layer.rec_state[STATE_SLOT[:, 1]] = rec0[STATE_SLOT[:, 0]]
        layer.conv_state[STATE_SLOT[:, 1]] = conv0[STATE_SLOT[:, 0]]
        p0 = None if pend is None else (pend[0][0::Q], pend[1][0::Q], pend[2][:, 0::Q])
        Sd, (pd, cd, Yd) = KS.decode(model, layer, streams[0::Q], positions[0::Q], STATE_SLOT[:, 1], pend=p0)
    assert torch.equal(Sv[0::Q], Sd) and torch.equal(pv[0::Q], pd) and torch.equal(cv[0::Q], cd)
    assert torch.equal(Yv[:, 0::Q], Yd)
    assert torch.equal(layer.rec_state[STATE_SLOT[:, 1]], w1_rec)
    assert torch.equal(layer.conv_state[STATE_SLOT[:, 1]], w1_conv)


def test_the_device_call_passes_every_row_once(monkeypatch):
    """One run() of all B Q rows in sequence order with vq Q; slots [position 0's read row (the zero row at a sequence
    start) and the zero row for later positions; state_slot[:, 1 + q]]; the pend whole; the partials all-reduced
    once."""
    layer, _ = _layer(monkeypatch, npg=2)
    calls = []

    def fake_run(S, blobs, conv, rec, slots, pend=None, **kw):
        calls.append(dict(rows=S[:, 0, 0].tolist(), slots=slots.tolist(), pend=pend, blobs=blobs, **kw))
        T = S.shape[0]
        return None, torch.zeros(T, HC), torch.zeros(T, HC, HC), torch.ones(2, T, H, dtype=BF), conv, rec

    monkeypatch.setattr(KS, "run", fake_run)
    reduced = []
    model = SimpleNamespace(cfg=TKS._cfg(), _attn_all_reduce=lambda x: reduced.append(x) or x)
    T = 6
    S = torch.arange(T, dtype=BF).view(T, 1, 1).expand(T, HC, H).contiguous()
    Z = layer.rec_state.shape[0] - 1  # the zero row
    pend = (torch.zeros(T, HC), torch.zeros(T, HC, HC), torch.zeros(2, T, H, dtype=BF))
    Sv, _ = KS.verify(model, layer, S, POSITIONS.reshape(T), STATE_SLOT, pend=pend)
    assert len(calls) == 1 and calls[0]["rows"] == list(range(T))
    assert calls[0]["slots"] == [[Z, Z, 4, Z, 7, Z], [2, 3, 5, 6, 8, 9]]
    assert calls[0]["vq"] == 2 and calls[0]["pend"] is pend and calls[0]["grid"] == 2
    assert calls[0]["blobs"][torch.bfloat16] is layer.ks_blob_bf16
    assert Sv is S  # the stand-in returns no S_n
    assert len(reduced) == 1 and reduced[0].shape == (2, T, H)


def test_run_gives_a_verify_the_row_batched_recurrence_of_its_positions(monkeypatch):
    """run(vq=Q) passes recb 2 Q - 1 (the kernel reads vq = Q from it); a decode call keeps forms()' recb."""
    seen = []
    monkeypatch.setattr(KS, "kda_shell_kernel", SimpleNamespace())
    T, R, CD = 6, 12, TKS.CD
    S = torch.zeros(T, HC, H, dtype=BF)
    blobs = {torch.bfloat16: torch.zeros(1, dtype=BF), torch.float32: torch.zeros(1)}
    conv, rec = torch.zeros(R, 3, CD, dtype=BF), torch.zeros(R, 2, 128, 128)
    slots = torch.zeros(2, T, dtype=torch.int32)
    for vq in (1, 2, 3):
        args = KS.run(
            S,
            blobs,
            conv,
            rec,
            slots,
            H=H,
            nv=2,
            hc=HC,
            iters=20,
            hc_eps=1e-6,
            rms_eps=1e-5,
            lb=-5.0,
            grid=2,
            fin=2,
            build_only=True,
            vq=vq,
        )
        seen.append((args["recb"], args["cbat"]))
    assert seen == [(KS.forms(T)[0], KS.forms(T)[1]), (3, KS.forms(T)[1]), (5, KS.forms(T)[1])]


def test_the_gate_takes_a_verify_of_up_to_three_positions(monkeypatch):
    layer, _ = _layer(monkeypatch, npg=2)
    monkeypatch.setattr(HY, "SHELL_HOST", True)
    model = SimpleNamespace(dp=1)
    h = torch.zeros(24, HC * H)

    def takes(state_slot, rows=6):
        return HY.kda_shell_takes(model, layer, h[:rows], state_slot, True, False)

    monkeypatch.setattr(HY, "KDA_SHELL_MAXB", 24)
    assert HY.KDA_SHELL_VERIFY == 3
    assert takes(torch.arange(6)) and takes(STATE_SLOT)
    assert takes(torch.arange(32).view(8, 4), rows=24)  # B 8 x Q 3
    assert not takes(torch.arange(15).view(3, 5), rows=12)  # Q 4 above the cap
    assert not takes(STATE_SLOT, rows=4)  # B Q != rows
    monkeypatch.setattr(HY, "KDA_SHELL_MAXB", 4)
    assert not takes(STATE_SLOT)  # MAXB caps the B Q rows
    monkeypatch.setattr(HY, "KDA_SHELL_MAXB", 24)
    monkeypatch.setattr(HY, "KDA_SHELL", False)
    assert not takes(torch.arange(6)) and not takes(STATE_SLOT)


def test_layer_and_the_split_layer_call_verify_for_a_verify():
    assert "call = kda_shell.verify if state_slot.dim() == 2 else kda_shell.decode" in inspect.getsource(HY.layer)
    assert "call = KS.verify if state_slot.dim() == 2 else KS.decode" in inspect.getsource(MK.decode_layer_split)


def test_off_adds_no_blobs(monkeypatch):
    layer, _ = _layer(monkeypatch, on=False)
    assert not any(n.startswith("ks_") for n, _ in layer.named_parameters())
    assert not hasattr(layer, "ks_dims") and not any(n.startswith("ks_") for n in getattr(layer, "_decode_only", ()))
    on, _ = _layer(monkeypatch, on=True)
    assert [n for n, _ in on.named_parameters() if n.startswith("ks_")] == ["ks_blob_bf16", "ks_blob_f32"]


# -- the kernel's one verify call against Q decode calls, in the simulator ---------------------------------------------


def _verify_forms(B: int, Q: int, grid: int, mix: bool, seed: int):
    """One case of B sequences x Q positions (rows sequence-major; sequence 0 at position 0) in the simulator: the
    kernel's verify (one call, vq = Q) and its reference, Q decode calls of all T = B Q rows (the prologue's rounding
    depends on a call's row count), the pools threaded: call q's position-q rows read state_slot[:, q] (position 0's, or
    the zero row) and write state_slot[:, q + 1], every other row writes a scratch row. Returns (one call, the
    reference), each (S_n, post, comb, P, conv, rec) with each row's outputs from its position's call, the state_slot
    and the scratch rows."""
    from trnrt.kernels import shell as SH

    T = B * Q
    c = TKS.case(T, seed, grid)
    L = c["L"]
    R = L.rec_state.shape[0]
    g = torch.Generator().manual_seed(seed + 1)
    perm = torch.randperm(R - 1, generator=g) + 1
    perm = perm[perm != SH.start_row(R)]
    st = perm[: B * (1 + Q)].view(B, 1 + Q)
    scratch = perm[B * (1 + Q) : B * (1 + Q) + B]
    pos = torch.arange(B).view(B, 1) * 5 + 7 + torch.arange(Q).view(1, Q)
    pos[0] = torch.arange(Q)
    t = KS.arrange(
        c["wcat"],
        L.conv_w,
        L.f_b,
        L.g_b,
        L.dt_bias,
        L.out,
        c["fn"],
        L.in_norm,
        L.A_log,
        L.o_norm,
        c["base"],
        c["scale"],
        TKS.NV,
        grid,
        HC,
    )
    blobs = KS.pack(t, H, TKS.NV, TKS.K, grid, HC)
    pend = c["pend"] if mix else None
    kw = {
        "H": H,
        "nv": TKS.NV,
        "hc": HC,
        "iters": TKS.ITERS,
        "hc_eps": TKS.HC_EPS,
        "rms_eps": TKS.EPS,
        "lb": TKS.LB,
        "pend": pend,
        "simulate": True,
        "grid": grid,
    }
    start = torch.full_like(st[:, 0], SH.start_row(R))
    rd = torch.stack([torch.where(pos[:, 0] > 0, st[:, 0], start)] + [start] * (Q - 1), 1).reshape(T)
    one = KS.run(
        c["S"],
        blobs,
        L.conv_state.clone(),
        L.rec_state.clone(),
        torch.stack([rd, st[:, 1:].reshape(T)]).to(torch.int32),
        vq=Q,
        **kw,
    )
    conv, rec, outs = L.conv_state.clone(), L.rec_state.clone(), []
    for q in range(Q):
        r = torch.stack([torch.where(pos[:, q] > 0, st[:, q], start)] * Q, 1)  # every row reads position q's row
        w = torch.stack([st[:, q + 1] if u == q else scratch for u in range(Q)], 1)  # only position q's rows keep
        o = KS.run(c["S"], blobs, conv, rec, torch.stack([r.reshape(T), w.reshape(T)]).to(torch.int32), **kw)
        conv, rec = o[4], o[5]
        outs.append(o)

    def pick(i, d):
        return torch.stack([o[i].unflatten(d, (B, Q)).select(d + 1, q) for q, o in enumerate(outs)], d + 1).flatten(
            d, d + 1
        )

    ref = (None if pend is None else pick(0, 0), pick(1, 0), pick(2, 0), pick(3, 1), conv, rec)
    return one, ref, st, scratch


@pytest.mark.skipif(not TKS.NKI, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize(("B", "grid", "mix"), [(1, 2, False), (2, 2, True), (4, 2, False), (8, 2, True), (3, 1, True)])
def test_the_verify_in_one_call_is_the_decode_calls_bit_for_bit(B, grid, mix, monkeypatch):
    """vq 3 (recb 5): position q >= 1's conv taps and state from position q - 1's in SBUF, against decode calls of the
    same rows with the pools between them: the same fp32 operations per element, so every output, and every pool row
    but the reference's scratch rows, equal bit for bit. B 2 x Q 3 is 6 rows, XNB_SKIP_T: the per-row xn form inside a
    verify. A kernel ignoring vq would start position q from its (unread) slots' zero row."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    Q = 3
    one, ref, st, scratch = _verify_forms(B, Q, grid, mix, seed=900 + 7 * B + grid)
    keep = torch.ones(one[5].shape[0], dtype=torch.bool)
    keep[scratch] = False
    for name, got, want in zip(("S", "post", "comb", "P", "conv", "rec"), one, ref, strict=True):
        if got is None and want is None:
            continue
        a, b = (got[keep], want[keep]) if name in ("conv", "rec") else (got, want)
        assert torch.equal(a, b), (name, (a.float() - b.float()).abs().max())
    for q in range(1, Q):
        assert not torch.equal(one[5][st[:, q + 1]], one[5][st[:, q]])  # position q's states are not position q - 1's
