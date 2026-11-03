"""The context-parallel DSA indexer's decode scores in pieces of rows (models/mla.py CP_INDEX_ROWS, _cp_scores_rows;
--dsa-index-rows): kernels/dsa_index.py one call per n rows when a call has more. Rows are independent in the kernel, so
the pieces' scores are one call's; separate calls only change how each call's query transpose is lowered.

On the host (dsa_index's emulation, the served per-rank shape: 32 indexer heads of 128, 8 local pools per page row, a
128-page bucket):
- the pieces equal one call bit for bit at 12, 16, 24 and 32 rows, every row with its own pages; reading every piece's
  pages from its first row (a wrong slice) fails that equality (the positive control);
- _cp_scores takes the pieces only with CP_INDEX_ROWS set and a call of more rows than it (a spy), and its scores are
  the same with and without them; unset (0, the served value) every call is one kernel call.
"""

import types

import pytest
import torch

from trnrt.kernels import dsa_index
from trnrt.models import mla

HI, D, PPK, NPG = 32, 128, 8, 128  # indexer heads, head dim, local pools per page row, the bucket's pages per row


def inputs(T, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(T, HI, D, generator=g) * 0.3).bfloat16()
    w = torch.rand(T, HI, generator=g) * D ** -0.5
    pkc = (torch.randn(600, PPK * D, generator=g) * 0.3).bfloat16()
    table = torch.stack([torch.randperm(599, generator=g)[:NPG] + 1 for _ in range(T)])  # each row its own pages
    P = NPG * PPK
    nloc = torch.randint(P // 2, P + 1, (T,), generator=g)
    cand = torch.where(torch.arange(P).view(1, P) < nloc.view(-1, 1), 0.0, float("-inf"))
    return q, w, pkc, table, cand, nloc


@pytest.mark.parametrize("T", [12, 16, 24, 32])
def test_pieces_equal_one_call(monkeypatch, T):
    monkeypatch.setattr(mla, "CP_INDEX_ROWS", 8)
    q, w, pkc, table, cand, _ = inputs(T)
    want = dsa_index.scores(q, w, pkc, dsa_index.page_groups(table), cand)
    got = mla._cp_scores_rows(q, w, pkc, table, cand)
    assert got.shape == want.shape == (T, NPG * PPK)
    assert torch.equal(got, want)


def test_a_piece_read_with_its_first_rows_pages_fails(monkeypatch):
    """Positive control: page groups built from each piece's first row for all of its rows (a wrong slice)."""
    monkeypatch.setattr(mla, "CP_INDEX_ROWS", 8)
    q, w, pkc, table, cand, _ = inputs(16)
    want = dsa_index.scores(q, w, pkc, dsa_index.page_groups(table), cand)
    real = dsa_index.page_groups
    monkeypatch.setattr(dsa_index, "page_groups", lambda t: real(t[:1].expand_as(t).contiguous()))
    assert not torch.equal(mla._cp_scores_rows(q, w, pkc, table, cand), want)


def _layer(pkc):
    dsa = types.SimpleNamespace(head_dim=D)
    return types.SimpleNamespace(spec=types.SimpleNamespace(mla=types.SimpleNamespace(dsa=dsa)),
                                 pool_key=pkc.view(-1, D), v_cache=None)


@pytest.mark.parametrize("rows,T,pieces", [(0, 12, []), (8, 8, []), (8, 12, [12]), (8, 16, [16])])
def test_cp_scores_takes_the_pieces_only_past_the_setting(monkeypatch, rows, T, pieces):
    q, w, pkc, table, _, nloc = inputs(T, seed=T)
    pk = torch.zeros(T, NPG * PPK, D, dtype=torch.bfloat16)  # its shape only: the kernel path reads the page rows
    scale = D ** -0.5
    monkeypatch.setattr(mla, "CP_INDEX_ROWS", 0)
    want = mla._cp_scores(None, _layer(pkc), q, w, pk, table, nloc, scale)
    monkeypatch.setattr(mla, "CP_INDEX_ROWS", rows)
    seen = []
    real = mla._cp_scores_rows
    monkeypatch.setattr(mla, "_cp_scores_rows", lambda q_, *a: (seen.append(q_.shape[0]), real(q_, *a))[1])
    got = mla._cp_scores(None, _layer(pkc), q, w, pk, table, nloc, scale)
    assert seen == pieces
    assert torch.equal(got, want)
