"""kernels/dsa_slots_n_p.py: the pipelined prefill slot attention (EngineConfig.dsa_slots_pipe_rows) against the split
kernel (kernels/dsa_slots_n_s.py). In the NKI simulator at grid 2: o and lse bit for bit over the rows the run-time
count covers, fp8 and bf16 latents, the bf16 and fp32 bias, a fully masked row, a partial last iteration with padding,
RPI 4 / 16 / 32 / 64, one and two slot chunks, the schedule forms; a planted P K accumulation order (reversed) differs,
so the comparison sees an accumulation-order change. On the host: the route sends only _cp_slots_attend's call, and
only on the split kernel's route."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from tests.test_dsa_slots import _inputs, kernel_args
from trnrt.kernels import dsa_slots as ds

needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")


def test_rev_is_its_own():
    from trnrt.kernels import dsa_slots_n as dsn
    from trnrt.kernels import dsa_slots_n_p as dp

    assert dp.REV not in (dsn.REV, ds.REV)
    assert dp._kernel_rev() == dp.REV


def _run(kern, kw, rev):
    import nki

    out = nki.simulate(kern[2])(**kw, rev=rev)
    return out if isinstance(out, (tuple, list)) else (out,)


def _planted_back(X, it, k, j):
    """dsa_slots_n_p._back with its P K accumulation order reversed (the comparison's positive control)."""
    from trnrt.kernels import dsa_slots_n_p as dp

    nisa, nl = dp.nisa, dp.nl
    H, R, NCH = X["H"], X["R"], X["NCH"]
    L, x = j % X["D"], j % 2
    IB, PP, PO = X["IB"], X["PP"], X["PO"]
    Kv, P = X["K"][L], X["P"][x]
    nb = NCH * dp.KPP
    for n_ in range(nb):
        i = nb - 1 - n_
        ch = i // dp.KPP
        t = i % dp.KPP
        jj = i * 128
        gg = 512 // H
        pp = PP[:, (i % gg) * H : (i % gg + 1) * H]
        nisa.nc_matmul(dst=pp, stationary=P[:, jj : jj + 128], moving=IB[0:H, 0:H], accumulate=False)
        pT = X["PTb"][i % 2]
        nisa.activation(dst=pT, op=nl.copy, data=pp)
        nisa.nc_matmul(dst=PO, stationary=pT, moving=Kv[ch][:, t * R : (t + 1) * R], accumulate=(n_ > 0))
    rden = X["RDEN"][x]
    nisa.reciprocal(dst=rden, data=X["DEN"][x])
    ob = X["OB"][x]
    nisa.tensor_scalar(dst=ob, data=PO, op0=nl.multiply, operand0=rden, engine=dp.VEP)
    dp._dma(X["o"].ap(pattern=[[R, H], [1, R]], offset=k * H * R, scalar_offset=it, indirect_dim=0), ob, X["stk"])
    ls = X["LS"][x]
    nisa.activation(dst=ls, op=nl.log, data=X["DEN"][x])
    nisa.tensor_tensor(dst=ls, data1=ls, data2=X["MX"][x], op=nl.add, engine=dp.VEP)
    dp._dma(X["lse"].ap(pattern=[[1, H], [1, 1]], offset=k * H, scalar_offset=it, indirect_dim=0), ls, X["stk"])


SCHED = [
    (2, 0, 2, 1, 0),
    (3, 1, 3, 1, 0),
    (3, 0, 2, 1, 0),
    (2, 1, 3, 1, 0),
    (2, 0, 2, 8, 0),
    (2, 0, 2, 16, 1),
    (2, 0, 2, 4, 2),
    (2, 0, 2, 1, 1),
]  # (la, sfirst, npt, sbat, sdge)
CASES = [
    (1, 1, 20, 20, 16, 128),
    (0, 1, 18, 13, 16, 128),
    (1, 0, 12, 7, 4, 128),
    (1, 1, 40, 33, 32, 128),
    (1, 1, 9, 9, 4, 256),
    (1, 1, 128, 97, 64, 128),
]  # (fp8, bb, N, n, rpi, NS); the last the served rpi


@needs_nki
@pytest.mark.parametrize("sched", [*SCHED, (2, 0, 2, 32, 1)])
@pytest.mark.parametrize(("fp8", "bb", "N", "n", "rpi", "NS"), CASES)
def test_pipelined_is_the_split_kernel_bit_for_bit(monkeypatch, fp8, bb, N, n, rpi, NS, sched):
    from trnrt.kernels import dsa_slots_n_p as dp
    from trnrt.kernels import dsa_slots_n_s

    la, sfirst, npt, sbat, sdge = sched
    if sbat > rpi // 2 or (sched != SCHED[0] and (rpi, NS) not in ((32, 128), (16, 128), (64, 128))):
        pytest.skip("each schedule form on the larger shapes; a batch no wider than a program's rows")
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    H, R = 8, 512
    q, kc, rows, bias = _inputs(N, H, R, NS, fp8, seed=N + n + rpi)
    bias[3] = ds.NEG_INF  # a fully masked row
    args, Np = kernel_args(q, kc, rows, bias, rpi)
    if bb:
        args["bias"] = args["bias"].to(torch.bfloat16)
    it = torch.tensor([[-(-n // rpi)]], dtype=torch.int32)
    kw = dict(args, n_iter=it, scale=0.05, fp8=fp8, want_lse=1, qkw=1)
    a = _run(dsa_slots_n_s.dsa_slots_n_s_kernel, kw, dsa_slots_n_s.REV_BB if bb else dsa_slots_n_s.REV)
    kw = dict(kw, la=la, sfirst=sfirst, npt=npt, sbat=sbat, sdge=sdge, ldge=min(sdge, 3))
    b = _run(dp.dsa_slots_n_p_kernel, kw, dp.REV)
    rows_run = int(it) * rpi
    for xa, yb, nm in zip(a, b, ("o", "lse")):
        x = torch.as_tensor(xa).reshape(Np, H, -1)[:rows_run]
        y = torch.as_tensor(yb).reshape(Np, H, -1)[:rows_run]
        assert torch.equal(x.view(torch.int32), y.view(torch.int32)), (nm, int((x != y).sum()))


@needs_nki
def test_planted_accumulation_order_differs(monkeypatch):
    from trnrt.kernels import dsa_slots_n_p as dp
    from trnrt.kernels import dsa_slots_n_s

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    H, R, N, rpi = 8, 512, 20, 16
    q, kc, rows, bias = _inputs(N, H, R, 128, 1, seed=55)
    args, Np = kernel_args(q, kc, rows, bias, rpi)
    it = torch.tensor([[-(-N // rpi)]], dtype=torch.int32)
    kw = dict(args, n_iter=it, scale=0.05, fp8=1, want_lse=1, qkw=1)
    a = _run(dsa_slots_n_s.dsa_slots_n_s_kernel, kw, dsa_slots_n_s.REV)
    monkeypatch.setattr(dp, "_back", _planted_back)
    p = _run(dp.dsa_slots_n_p_kernel, dict(kw, sbat=1), dp.REV ^ 0x1)
    po = torch.as_tensor(p[0]).reshape(Np, H, R)[:N]
    ao = torch.as_tensor(a[0]).reshape(Np, H, R)[:N]
    assert not torch.equal(po.view(torch.int32), ao.view(torch.int32)), "the plant must move bits"
    assert torch.allclose(po, ao, rtol=1e-4, atol=1e-5), "the plant is the same sum in another order"


def test_route_sends_only_the_cp_single_buffer_call(monkeypatch):
    """With the route, _cp_slots_attend (attention_cp_rows' single slot buffer, prefill only) takes the pipelined
    kernel when the split kernel's route holds (grid 2, platform.lnc_split naming dsa_slots_n); off, at grid 1, or with
    the split off, dsa_slots_n.attend as before."""
    from trnrt import platform
    from trnrt.kernels import dsa_slots_n as dsn
    from trnrt.kernels import dsa_slots_n_p as dp
    from trnrt.models import mla

    seen = []
    monkeypatch.setattr(dp, "attend", lambda *a, **k: seen.append(("p", k)) or "p")
    monkeypatch.setattr(dsn, "attend", lambda *a, **k: seen.append(("n", k)) or "n")
    monkeypatch.setattr(platform, "nki_grid", lambda: 2)
    split = {"dsa_slots_n"}
    monkeypatch.setattr(platform, "lnc_split", lambda kernel: kernel in split)
    assert dp.PIPE_ROWS == 0
    assert mla._cp_slots_attend(1, 2, 3, 4, 0.1, 5) == "n"  # the default: off
    monkeypatch.setattr(dp, "PIPE_ROWS", 64)
    assert mla._cp_slots_attend(1, 2, 3, 4, 0.1, 5) == "p"
    split.clear()  # the split off
    assert mla._cp_slots_attend(1, 2, 3, 4, 0.1, 5) == "n"
    split.add("dsa_slots_n")
    monkeypatch.setattr(platform, "nki_grid", lambda: 1)  # grid 1
    assert mla._cp_slots_attend(1, 2, 3, 4, 0.1, 5) == "n"
    assert [s for s, _ in seen] == ["n", "p", "n", "n"]
    assert seen[1][1] == {"lse": True}


def test_engine_config_refuses_an_odd_pipe():
    from trnrt.config import EngineConfig

    for v in (-2, 3):
        with pytest.raises(ValueError, match="dsa_slots_pipe_rows"):
            EngineConfig(model_path="m", dsa_slots_pipe_rows=v)
    assert EngineConfig(model_path="m", dsa_slots_pipe_rows=64).dsa_slots_pipe_rows == 64
