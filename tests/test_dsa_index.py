"""kernels/dsa_index.py, the decode-side indexer scores: the host path against the dense definition, and (where the NKI
package is installed) the kernel under the NKI simulator."""

import importlib.util

import pytest
import torch

from trnrt.kernels import dsa_index

HI, D, PPP = 4, 128, 8


def case(B=2, pages=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    npg = B * pages + 1
    pkc = (torch.randn(npg, PPP * D, generator=g) * 0.5).to(torch.bfloat16)
    table = (torch.randperm(npg - 1, generator=g)[: B * pages].view(B, pages) + 1).to(torch.int32)
    q = (torch.randn(B, HI, D, generator=g) * 0.1).to(torch.bfloat16)
    w = torch.randn(B, HI, generator=g) * D ** -0.5
    cand = torch.zeros(B, pages * PPP)
    cand[:, -3:] = dsa_index.NEG_INF
    return q, w, pkc, table, cand


def dense(q, w, pkc, table, cand):
    B, npg = table.shape
    k = pkc[table.long()].reshape(B, npg * PPP, D).float()
    s = torch.einsum("bhd,bpd->bhp", q.float(), k)
    return torch.einsum("bh,bhp->bp", w, torch.relu(s)) + cand


def test_host_scores_are_the_dense_definition():
    q, w, pkc, table, cand = case()
    got = dsa_index.scores(q, w, pkc, dsa_index.page_groups(table), cand)
    want = dense(q, w, pkc, table, cand)
    ok = want > -1e29
    assert torch.allclose(got[ok], want[ok], rtol=1e-5, atol=1e-6)
    assert ((got < -1e29) == ~ok).all()


def test_page_groups_layout():
    """ppg[p, b, g] is page p NG + g of row b; a table that is no multiple of 128 pages is refused."""
    table = torch.arange(2 * 256, dtype=torch.int32).view(2, 256)
    ppg = dsa_index.page_groups(table)
    assert ppg.shape == (128, 2, 2) and ppg.dtype == torch.int32
    assert torch.equal(ppg[5, 1], table[1, 10:12]) and torch.equal(ppg.permute(1, 0, 2).reshape(2, -1), table)
    with pytest.raises(ValueError):
        dsa_index.page_groups(table[:, :200])


@pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("B", [1, 2, 5])
def test_nki_simulator_matches_the_host(B, monkeypatch):
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    q, w, pkc, table, cand = case(B=B, pages=128, seed=3 + B)
    ppg = dsa_index.page_groups(table)
    want = dsa_index.scores(q, w, pkc, ppg, cand)
    qT = q.reshape(B * HI, D).t().contiguous()
    wb = w.float().repeat(1, 4).unsqueeze(1).expand(B, 128, 4 * HI).contiguous()
    eye = torch.eye(128).to(torch.bfloat16)
    got = torch.as_tensor(nki.simulate(dsa_index.dsa_index_kernel)(
        qT=qT, wb=wb, pkc=pkc, ppg=ppg, cand=cand, identb=eye, rev=dsa_index.REV, dge=dsa_index.DGE, spl=1,
        nr=dsa_index.NR, qr=dsa_index.QR, dbg=0)).float()
    ok = want > -1e29
    assert ((got - want).abs()[ok].max() / want[ok].abs().max()).item() < 1e-5
    assert ((got < -1e29) == ~ok).all()
