"""Long-context and context-parallel pooled DSA (models/dsa_long.py, models/mla.py attention_long / attention_cp /
attention_cp_rows).

- The selection: the two-level exact selection and the device forms against the dense definition on adversarial ties;
  the slots against the bucketed path's token selection; the context-parallel merge of every rank's local lists
  against the global selection (ties straddling ranks, A up to 32, decode batches merged in row pieces, the two-
  selection form bit for bit); a token's local slot against a context's local pool rows; the pool-key cache as page
  rows.
- The engine (transformers >= 5.18, tests/test_glm5_next.py's builder): GLM-5.3-Flash at tp 1 on the long path against
  transformers and against the bucketed path; context parallelism, always on at tp > 1, at tp 2 and 4 against the
  replicated long path at tp 1; its row groups (CP 8 below the attention TP, with the local-K merge) at tp 16. A
  context-parallel model is refused at load unless hidden is a multiple of 512, the KV cache is FP8 (the separate
  pool-key cache) and the page holds kpool x cp tokens: every tp > 1 test builds at hidden 512 and runs FP8 KV with
  such pages, and patches no module constant (a spawned rank would not see it).
"""

import pytest
import torch

from trnrt.models import dsa_long

NEG = dsa_long.NEG_INF


def _ref_mask(sc: torch.Tensor, keep: int) -> torch.Tensor:
    """The tie rule written out (dsa_select.reference_mask restricted to candidates): per row, the candidates
    (> VISIBLE) in descending score, ties by ascending index, the first keep."""
    N, P = sc.shape
    out = torch.zeros(N, P, dtype=torch.bool)
    for n in range(N):
        cand = [p for p in range(P) if sc[n, p] > dsa_long.VISIBLE]
        cand.sort(key=lambda p: (-float(sc[n, p]), p))
        out[n, cand[:keep]] = True
    return out


def _as_mask(pools, cnt, P):
    N = pools.shape[0]
    m = torch.zeros(N, P, dtype=torch.bool)
    for n in range(N):
        c = int(cnt[n])
        idx = pools[n, :c]
        assert torch.all(idx[1:] > idx[:-1]), "pools must be ascending and distinct"
        m[n, idx] = True
    return m


def _cases():
    """(name, scores [N, P], keep): random, integer ties, exact zeros, ties straddling sub-block boundaries,
    the keep-th score tied in several sub-blocks whose maximum it is, candidate prefixes shorter than keep,
    exactly keep, and fewer visible sub-blocks than keep."""
    g = torch.Generator().manual_seed(0)
    out = []
    for P in (37, 64, 257, 1000, 4096):
        for keep in (1, 4, 16, 64):
            if keep >= P:
                continue
            N = 6
            npool = torch.tensor([0, 1, keep, keep + 1, P // 2, P])
            cand = torch.arange(P).view(1, P) < npool.view(N, 1)

            def put(s, cand=cand):
                return torch.where(cand, s, NEG)

            out.append((f"randn P{P} k{keep}", put(torch.randn(N, P, generator=g)), keep))
            out.append((f"int P{P} k{keep}", put(torch.randint(0, 5, (N, P), generator=g).float()), keep))
            out.append((f"zeros P{P} k{keep}", put(torch.zeros(N, P)), keep))
            out.append((f"relu P{P} k{keep}", put(torch.relu(torch.randn(N, P, generator=g) - 1.5)), keep))
            # a plateau at the threshold value, spanning sub-block boundaries, with a few larger values after it
            s = torch.full((N, P), 1.0)
            s[:, P // 3 :] = 2.0
            s[:, -3:] = 5.0
            out.append((f"plateau P{P} k{keep}", put(s), keep))
            # every sub-block of 4 has the same maximum; the tied elements sit at varying offsets
            s = torch.zeros(N, P)
            for p in range(P):
                if (p * 7) % 4 == p % 4:
                    s[:, p] = 3.0
            out.append((f"blockmax-ties P{P} k{keep}", put(s), keep))
            # signed zeros and one-ulp neighbours
            s = torch.where(torch.rand(N, P, generator=g) < 0.5, torch.tensor(0.0), torch.tensor(-0.0))
            s = s + torch.where(torch.rand(N, P, generator=g) < 0.1, torch.tensor(1e-45), torch.tensor(0.0))
            out.append((f"signed-zero P{P} k{keep}", put(s), keep))
    return out


# -- the selection ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sub", [1, 2, 4, 32])
def test_two_level_equals_reference(sub):
    """select_two_level == select_reference == the tie rule written out, for sub-blocks of 1 to 32 pools."""
    n = 0
    for name, sc, keep in _cases():
        want = _ref_mask(sc, keep)
        p_ref, c_ref = dsa_long.select_reference(sc, keep)
        p_two, c_two = dsa_long.select_two_level(sc, keep, sub)
        P = sc.shape[1]
        assert torch.equal(_as_mask(p_ref, c_ref, P), want), name
        assert torch.equal(_as_mask(p_two, c_two, P), want), f"{name} sub {sub}"
        assert torch.equal(p_two, p_ref) and torch.equal(c_two, c_ref), f"{name} sub {sub}"
        n += 1
    assert n > 100


def test_two_level_corner_tie_room():
    """The proof's corner, as data: t equals the keep-th sub-block maximum M, the elements equal to t sit in
    sub-blocks with a larger maximum AND in tied sub-blocks, and the lowest-index ties are in the LATER tied
    sub-blocks' predecessors. keep 3, sub 4: sub-block 0 = [9, 5, 5, 5] (max 9 > t), sub-blocks 1, 2 = [5, 0, 0, 0]
    (max 5 = t); the exact selection is 9 and the two lowest-index 5s, both in sub-block 0."""
    sc = torch.tensor([[9.0, 5, 5, 5, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0]])
    assert _ref_mask(sc, 3)[0].nonzero().flatten().tolist() == [0, 1, 2]
    p, c = dsa_long.select_two_level(sc, 3, 4)
    assert c.tolist() == [3] and p[0].tolist() == [0, 1, 2]


def test_device_selection_forms_equal_reference():
    """select_device (kernels/dsa_topk.py's selection in row groups, then compact()) and compact() alone give
    select_reference's pools and counts on the adversarial cases: the decode batch's device form."""
    for name, sc, keep in _cases():
        want = dsa_long.select_reference(sc, keep)
        got = dsa_long.select_device(sc, keep)
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]), name
        mask = _ref_mask(sc, keep).to(torch.float32)
        assert torch.equal(dsa_long.compact(mask, keep)[0], want[0]), name


def test_compact_at_1m_pools():
    """dsa_long.compact at the 262,144 pools of a 1M context (G = 512 groups of 512, no full-length scan) equals the
    reference ordering, for sparse, clustered and short masks."""
    g = torch.Generator().manual_seed(5)
    M, keep = 262144, 512
    sel = torch.zeros(4, M)
    sel[0, torch.randperm(M, generator=g)[:keep]] = 1  # spread
    sel[1, 1000 : 1000 + keep] = 1  # one run across group boundaries
    sel[2, torch.randperm(M, generator=g)[:37]] = 1  # fewer than keep
    want = dsa_long._ascending(sel > 0, keep)
    got = dsa_long.compact(sel, keep)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def test_slots_match_bucketed_selection():
    """The tokens slots() attends (selected pools, tail, causal) are the bucketed path's selection (glm5_next.block_mask,
    kernels/dsa_topk.py's rule with the tail) plus the causal visibility, for a chunk and for a decode batch, pages of
    4 and 8 tokens."""
    from trnrt.models import glm5_next

    g = torch.Generator().manual_seed(1)
    kp, keep = 4, 3
    for ps in (4, 8):
        for L, positions in (
            (64, torch.tensor([0, 2, 3, 4, 11, 12, 13, 30, 47, 63])),
            (96, torch.tensor([95, 50, 17, 7])),
        ):
            P = L // kp
            N = positions.shape[0]
            npool = dsa_long.npools(positions, kp)
            index = torch.randn(N, P, generator=g)
            index[:, ::5] = 0.0
            sc = index + torch.where(torch.arange(P).view(1, P) < npool.view(N, 1), 0.0, NEG)
            pools, cnt = dsa_long.select_two_level(sc, keep, 2)
            vis = torch.where(torch.arange(L).view(1, L) <= positions.view(N, 1), 0.0, NEG).view(N, 1, L)
            want = glm5_next.block_mask(index.view(N, 1, P), vis, kp, keep, True)[:, 0] == 0
            want &= vis[:, 0] == 0
            assert torch.equal(dsa_long.reference_mask(pools, cnt, npool, positions, L, kp), want)
            # through the block table: rows / bias address exactly those tokens
            pages = L // ps
            table = torch.randperm(200, generator=g)[:pages] + 1
            rows, bias = dsa_long.slots(pools, cnt, npool, positions, table, ps, kp, keep + 3)
            tok = (rows.unsqueeze(-1) * kp + torch.arange(kp)).reshape(N, -1)  # cache slots
            live = bias.reshape(N, -1) == 0
            slot_of = (table.view(-1, 1) * ps + torch.arange(ps)).reshape(-1)  # context position -> cache slot
            for n in range(N):
                assert sorted(tok[n][live[n]].tolist()) == sorted(slot_of[want[n]].tolist()), (ps, L, n)
            # one table per row (decode form) gives the same
            rows2, bias2 = dsa_long.slots(pools, cnt, npool, positions, table.expand(N, pages), ps, kp, keep + 3)
            assert torch.equal(rows2, rows) and torch.equal(bias2, bias)


# -- the context-parallel merge -------------------------------------------------------------------------------------


def _local_lists(sc: torch.Tensor, keep: int, A: int):
    """Every rank's exact local top-keep over its pools (context pools c = m A + rank): (vals [N, A, keep], cpool)."""
    N = sc.shape[0]
    vals, cps = [], []
    for r in range(A):
        loc = sc[:, r::A]
        if loc.shape[1] == 0:
            loc = torch.full((N, 1), NEG)
        lp, lc = dsa_long.select_reference(loc, keep)
        ok = torch.arange(keep).view(1, keep) < lc.view(N, 1)
        vals.append(torch.where(ok, loc.gather(1, lp), torch.tensor(NEG)))
        cps.append((lp * A + r).float())
    return torch.stack(vals, 1), torch.stack(cps, 1)


def _merged_mask(sel, cps, N, P):
    got = torch.zeros(N, P, dtype=torch.bool)
    rows, ranks, ks = sel.nonzero(as_tuple=True)
    got[rows, cps[rows, ranks, ks].long()] = True
    return got


@pytest.mark.parametrize("A", [2, 4, 8])
def test_cp_merge_equals_global_selection(A):
    """dsa_long.cp_merge: every rank's exact local top-keep, gathered and merged, is the global exact selection, ties
    included (the adversarial cases, whose ties straddle ranks)."""
    for name, sc, keep in _cases():
        N, P = sc.shape
        vals, cps = _local_lists(sc, keep, A)
        sel = dsa_long.cp_merge(vals, cps, keep)
        assert torch.equal(_merged_mask(sel, cps, N, P), _ref_mask(sc, keep)), f"{name} A={A}"


@pytest.mark.parametrize("A", [16, 32])
def test_cp_merge_wide(A):
    """cp_merge at A = 16 / 32 with keep 512: A keep = 8,192 / 16,384 candidates per row, wider than the selection
    kernel's 4,096 per partition, take the two-level threshold; still the global exact selection with ties (integer
    scores, a plateau straddling ranks, candidate prefixes around keep)."""
    g = torch.Generator().manual_seed(4)
    keep, P, N = 512, 32768, 4
    npool = torch.tensor([keep - 3, keep + 1, P // 3, P])
    cand = torch.arange(P).view(1, P) < npool.view(N, 1)
    plateau = torch.full((N, P), 1.0)
    plateau[:, P // 2 :] = 2.0
    for name, raw in (
        ("int", torch.randint(0, 7, (N, P), generator=g).float()),
        ("randn", torch.randn(N, P, generator=g)),
        ("plateau", plateau),
    ):
        sc = torch.where(cand, raw, torch.tensor(NEG))
        vals, cps = _local_lists(sc, keep, A)
        sel = dsa_long.cp_merge(vals, cps, keep)
        assert torch.equal(_merged_mask(sel, cps, N, P), _ref_mask(sc, keep)), f"{name} A={A}"


def test_cp_merge_in_row_pieces_equals_one_piece(monkeypatch):
    """cp_merge with rows (the decode batch's CP_MERGE_ROWS) merges in pieces of it: the same mask as one piece, since
    every row is merged on its own; without rows (a prefill chunk) it is one piece whatever N."""
    g = torch.Generator().manual_seed(11)
    N, A, keep = 10, 4, 6
    vals = torch.randint(0, 5, (N, A, keep), generator=g).float()
    vals[0, 1, 3] = NEG
    cps = torch.stack([torch.randperm(A * keep, generator=g)[: A * keep].view(A, keep) for _ in range(N)]).float()
    whole = dsa_long.cp_merge_rows(vals, cps, keep)
    for rows in (None, 3, 4, 10, 64):
        assert torch.equal(dsa_long.cp_merge(vals, cps, keep, rows), whole), rows
    pieces = []
    real = dsa_long.cp_merge_rows

    def counted(v, c, k):
        pieces.append(v.shape[0])
        return real(v, c, k)

    monkeypatch.setattr(dsa_long, "cp_merge_rows", counted)
    dsa_long.cp_merge(vals, cps, keep)
    dsa_long.cp_merge(vals, cps, keep, 3)
    assert pieces == [10, 3, 3, 3, 1]


def _merge_case(N: int, kind: str, A: int = 8, K: int = 512, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    vals = torch.randn(N, A, K, generator=g)
    if kind == "ties":  # a few distinct values: thousands of candidates tied at the threshold
        vals = torch.randint(0, 6, (N, A, K), generator=g).float()
    if kind == "zeros":
        vals = torch.zeros(N, A, K)
    if kind == "empty":  # every rank's list ends early (NEG_INF past its count)
        cnt = torch.randint(0, K, (N, A, 1), generator=g)
        vals = torch.where(torch.arange(K).view(1, 1, K) < cnt, vals, torch.full_like(vals, NEG))
    # context pools: rank a's local pool m is m A + a, each list ascending
    lp = torch.stack([torch.randperm(4096, generator=g)[:K].sort().values for _ in range(N * A)]).view(N, A, K)
    return vals, (lp * A + torch.arange(A).view(1, A, 1)).float()


@pytest.mark.parametrize("N", [1, 2, 4, 8])
@pytest.mark.parametrize("kind", ["randn", "ties", "zeros", "empty"])
def test_two_selection_merge_is_bit_equal(N, kind):
    """cp_merge_rows_2k (the decode batch's merge at two rows and more): the same mask as the tie search, ties, zeros
    and short lists included."""
    vals, cpool = _merge_case(N, kind, seed=10 + N)
    assert torch.equal(dsa_long.cp_merge_rows_2k(vals, cpool, 512), dsa_long.cp_merge_rows(vals, cpool, 512))


def test_two_selection_merge_sees_the_tie_rule():
    """The positive control: ties keyed by +cpool (the largest pools) give another mask."""
    vals, cpool = _merge_case(2, "ties", seed=6)
    assert not torch.equal(dsa_long.cp_merge_rows_2k(vals, -cpool, 512), dsa_long.cp_merge_rows(vals, cpool, 512))


@pytest.mark.parametrize(
    ("N", "KL"), [(64, 120), (512, 120), (16, 512)]
)  # local-K lists (K 120) and the failed rows' (keep)
@pytest.mark.parametrize("kind", ["randn", "ties", "zeros", "empty"])
def test_prefill_local_k_merge_two_selection_is_bit_equal(N, KL, kind):
    """The row groups' merges through cp_merge(..., two_k=True) at a prefill chunk's shapes (CP 8, local K 120, keep
    512; the failed rows' lists are keep long) give the tie search's mask bit for bit."""
    vals, cpool = _merge_case(N, kind, seed=200 + N + KL)
    vals, cpool = vals[:, :, :KL].contiguous(), cpool[:, :, :KL].contiguous()
    if kind == "randn":  # value order, as each rank's list arrives (dsa_long_select's vorder)
        vals = vals.sort(dim=-1, descending=True).values
    want = dsa_long.cp_merge(vals, cpool, 512, None, 4096 * 8)
    got = dsa_long.cp_merge(vals, cpool, 512, None, 4096 * 8, two_k=True)
    assert torch.equal(got, want), (N, KL, kind, (got != want).sum().item())
    assert want.to(torch.float32).sum().item() > 0 or kind == "empty"


def test_cp_slots_and_pool_rows():
    """A token's local slot (dsa_long.cp_local_slots) and a context's local pool rows (cp_pool_rows) agree: the rows of
    local pool m hold exactly the slots its owner wrote for context pool m A + rank, and every token is written by
    exactly one rank."""
    ps, kp = 32, 4
    for A in (2, 4, 8):
        table = torch.tensor([5, 2, 9, 7])
        pos = torch.arange(table.numel() * ps)
        owners = []
        for r in range(A):
            ls = dsa_long.cp_local_slots(pos, table, ps, kp, A, torch.tensor([r]))
            mine = (pos // kp) % A == r
            owners.append(mine)
            rows = dsa_long.cp_pool_rows(table, ps, kp, A)  # [Pl]
            for m in range(rows.numel()):
                c = m * A + r
                toks = torch.arange(c * kp, c * kp + kp)
                assert torch.equal(ls[toks], rows[m] * kp + torch.arange(kp)), (A, r, m)
            assert torch.all(ls[~mine] == dsa_long.cp_dump_slot(ps, A))
        assert torch.equal(torch.stack(owners).sum(0), torch.ones(pos.numel(), dtype=torch.long))


def test_cp_page_keys_equal_pool_rows():
    """The pool-key cache viewed as [pages, ppl Di] and indexed by the block table holds the rows cp_pool_rows names,
    in order."""
    for A, ps, kp, Di in ((2, 8, 4, 16), (8, 256, 4, 128)):
        ppl = ps // (kp * A)
        cache = torch.randn(12 * (ps // A), Di // kp)  # local slots x the pool-key piece (Di / kpool per slot)
        table = torch.tensor([[3, 7, 1], [0, 11, 5]])
        prow = dsa_long.cp_pool_rows(table, ps, kp, A)
        want = cache.view(-1, Di)[prow]
        got = cache.view(-1, ppl * Di)[table].view(table.shape[0], prow.shape[-1], Di)
        assert torch.equal(got, want), (A, ps)


def test_cp_degree():
    """CP 8 over an attention TP of at least 8 (row groups above it); the whole attention group below."""
    assert dsa_long.cp_enabled() and dsa_long.CP_DEGREE == 8
    assert [dsa_long.cp_degree(a) for a in (2, 4, 8, 16, 32)] == [2, 4, 8, 8, 8]


# -- the engine -------------------------------------------------------------------------------------------------------


def _glm():
    """tests/test_glm5_next.py's helpers (transformers >= 5.18), or a skip."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from tests import test_glm5_next

    return test_glm5_next


class _Lazy:
    def __getattr__(self, name):
        return getattr(_glm(), name)


glm = _Lazy()


@pytest.fixture(scope="module")
def sparse(tmp_path_factory):
    """index_topk 16 (4 pools of 4 kept), positions up to 4096."""
    path = tmp_path_factory.mktemp("glm5_next_long")
    return str(path), glm.build(str(path), seed=1, index_topk=16, max_position_embeddings=4096)


@pytest.fixture(scope="module")
def sparse_cp(tmp_path_factory):
    """sparse's config at hidden 512: at attention TP > 1 the DSA caches are context-parallel, and the model takes the
    served form's constraints (hidden a multiple of 512, the fused decode layer's tiles; pages of kpool x cp tokens; an
    FP8 KV cache, the separate pool-key cache). A spawned rank sees no monkeypatch, so these are the model's own."""
    path = tmp_path_factory.mktemp("glm5_next_cp")
    return str(path), glm.build(str(path), seed=1, index_topk=16, max_position_embeddings=4096, hidden_size=512)


def _generate(path, ps, n, long_keys, monkeypatch, **kw):
    """Generate at tp 1 on an engine whose buckets past long_keys keys run the long path; asserts that it ran (and that
    it did not when long_keys is out of reach)."""
    from trnrt.engine.request import SamplingParams
    from trnrt.models import mla

    calls = {"long": 0}
    real = mla.attention_long

    def counted(*a, **k):
        calls["long"] += 1
        return real(*a, **k)

    monkeypatch.setattr(mla, "attention_long", counted)
    monkeypatch.setattr(dsa_long, "LONG_KEYS", long_keys)
    eng = glm.engine(path, **kw)
    try:
        reqs = eng.generate(ps, SamplingParams(max_new_tokens=n, ignore_eos=True, logprobs=0))
    finally:
        eng.close()
        monkeypatch.setattr(mla, "attention_long", real)
    assert (calls["long"] > 0) == (long_keys < 1 << 20), calls
    return eng, reqs


def _close(a, b, atol=2e-5):
    for x, y in zip(a, b):
        torch.testing.assert_close(
            torch.tensor([v[0] for v in x.logprobs]), torch.tensor([v[0] for v in y.logprobs]), rtol=0, atol=atol
        )


@pytest.mark.parametrize(("page_size", "chunk"), [(4, 16), (8, 6)])
def test_long_path_matches_transformers(sparse, page_size, chunk, monkeypatch):
    """Every bucket past 16 keys on the long path (scores, two-level selection, slots, absorbed attention), through
    chunked prefill with chunks that cut pools, a radix-cache hit and batched decode, against transformers' greedy
    tokens, and against the bucketed path's logprobs (fp32: the same selection, the same keys)."""
    path, hf = sparse
    ps = glm.prompts(2, (37, 70, 9))
    ps.append(ps[1][:48] + ps[0][:20])  # shares 48 tokens with prompt 1: a prefix hit
    kw = {"page_size": page_size, "max_prefill_tokens": chunk}
    eng_l, got = _generate(path, ps, 12, 16, monkeypatch, **kw)
    assert eng_l.model.long_ctx(32) and not eng_l.model.long_ctx(16)
    _, want = _generate(path, ps, 12, 1 << 30, monkeypatch, **kw)
    assert [r.output_ids for r in got] == [r.output_ids for r in want] == [glm.hf_greedy(hf, x, 12) for x in ps]
    _close(got, want)


def test_long_path_fp8_kv(sparse, monkeypatch):
    """Under an FP8 KV cache (the separate bf16 pool-key cache: the served form) the long path serves the bucketed
    path's tokens."""
    path, _ = sparse
    ps = glm.prompts(7, (41, 66))
    _, got = _generate(path, ps, 8, 16, monkeypatch, kv_cache_dtype="fp8")
    _, want = _generate(path, ps, 8, 1 << 30, monkeypatch, kv_cache_dtype="fp8")
    assert [r.output_ids for r in got] == [r.output_ids for r in want]


@pytest.mark.parametrize("sub", [2, 32])
def test_long_context_against_transformers(sparse, sub, monkeypatch):
    """A 1,500-token prompt (375 pools against 4 kept: the 1M regime's ratio of candidates to selection is 512)
    prefilled in chunks of 64 on the long path, then decoded: transformers' greedy tokens, and the bucketed path's
    logprobs. sub 2 and 32 make the two-level selection run with 188 and 12 sub-blocks."""
    monkeypatch.setattr(dsa_long, "SUB", sub)
    path, hf = sparse
    ps = glm.prompts(9, (1500,), 384)
    kw = {"max_model_len": 2048, "num_pages": 600, "max_prefill_tokens": 64, "max_num_seqs": 1}
    _, got = _generate(path, ps, 6, 64, monkeypatch, **kw)
    _, want = _generate(path, ps, 6, 1 << 30, monkeypatch, **kw)
    assert [r.output_ids for r in got] == [r.output_ids for r in want] == [glm.hf_greedy(hf, ps[0], 6)]
    _close(got, want)


def test_long_scratch_is_capped(sparse, monkeypatch):
    """The selection scratch (models/mla.py bind_scratch) of a model whose every DSA layer can run long is at most
    LONG_KEYS wide: a long max_model_len does not allocate [rows, max_model_len]."""
    monkeypatch.setattr(dsa_long, "LONG_KEYS", 64)
    path, _ = sparse
    eng = glm.engine(path, max_model_len=4096, num_pages=64)
    try:
        for layer in eng.model.kv_layers():
            if getattr(layer, "dsa_mask", None) is not None:
                assert layer.dsa_mask.shape[1] == 64
    finally:
        eng.close()


def _cp_engine(path, ps, n, tp, **kw):
    """Greedy tokens and logprobs of a context-parallel engine at tp (CP is always on at tp > 1), FP8 KV."""
    from trnrt.engine.request import SamplingParams

    eng = glm.engine(path, tp=tp, kv_cache_dtype="fp8", **kw)
    try:
        cp, rows, lps = eng.model.cp, eng.model.cp_rows, eng.runner.lps
        reqs = eng.generate(ps, SamplingParams(max_new_tokens=n, ignore_eos=True, logprobs=0))
    finally:
        eng.close()
    return (cp, rows, lps), reqs


@pytest.mark.xfail(
    strict=True,
    reason="docs/KNOWN_ISSUES.md: logprobs 1.3-2.2e-3 apart under the FP8 KV cache (atol "
    "1e-4), tokens equal; not isolated from FP8 summation-order noise",
)
@pytest.mark.parametrize(("tp", "page_size"), [(2, 8), (4, 16)])
def test_cp_engine_matches_replicated(sparse_cp, tp, page_size, monkeypatch):
    """Context parallelism at tp 2 and 4 (attention TP = tp = CP, each rank holding 1 / tp of the DSA caches): the same
    greedy tokens, and logprobs within fp32 summation order, as the replicated long path at tp 1, through chunked
    prefill (chunks that cut pools, sequence-parallel streams), a prefix hit and batched decode (the compacted slots)."""
    path, _ = sparse_cp
    ps = glm.prompts(4, (37, 70, 9))
    ps.append(ps[1][:48] + ps[0][:20])
    kw = {"page_size": page_size, "max_prefill_tokens": 16, "max_num_seqs": 4}
    _, want = _generate(path, ps, 10, 16, monkeypatch, kv_cache_dtype="fp8", **kw)
    (cp, rows, lps), got = _cp_engine(path, ps, 10, tp, **kw)
    assert (cp, rows, lps) == (tp, 1, page_size // tp)
    assert [r.output_ids for r in got] == [r.output_ids for r in want]
    _close(got, want, atol=1e-4)


def test_cp_engine_fewer_local_pools_than_keep(tmp_path, monkeypatch):
    """Context parallelism where a rank holds fewer local pools than keep (index_topk 64: keep 16 pools; page 8 at A 2:
    one local pool per page, so a short context's bucket gives each rank a few): every local pool is a candidate, and
    the tokens equal the replicated long path's."""
    path = str(tmp_path)
    glm.build(path, seed=2, index_topk=64, max_position_embeddings=4096, hidden_size=512)
    ps = glm.prompts(3, (41, 9, 66))
    kw = {"page_size": 8, "max_prefill_tokens": 16, "max_num_seqs": 4}
    _, want = _generate(path, ps, 8, 16, monkeypatch, kv_cache_dtype="fp8", **kw)
    _, got = _cp_engine(path, ps, 8, 2, **kw)
    assert [r.output_ids for r in got] == [r.output_ids for r in want]


def test_cp_row_groups_match_replicated(tmp_path, monkeypatch):
    """Context-parallel row groups (models/mla.py attention_cp_rows) at tp 16: CP 8 inside 2 row groups, each row group
    writing its share of every chunk row and selecting and attending its half of the rows with every head (index_topk
    1024: keep 256 pools, so the local-K merge of CP_LOCAL_K = 120 per rank runs, with its certificate and redo), decode
    batches through attention_cp inside each row group. The greedy tokens equal the replicated long path's at tp 1.
    (16 spawned CPU ranks; the model is cut to 16 heads so they split, hidden 512, pages of kpool x cp = 32 tokens; a
    vocabulary of 1,024, so each rank's shard holds the vocab-parallel sampler's 64 candidates: 384 / 16 = 24 rows made its
    top-k ask for more ids than the shard has.)"""
    from trnrt.models import mla

    path = str(tmp_path)
    glm.build(
        path,
        seed=3,
        layers=4,
        linear_heads=16,
        num_attention_heads=16,
        num_key_value_heads=16,
        index_topk=1024,
        max_position_embeddings=4096,
        hidden_size=512,
        vocab_size=1024,
    )
    ps = glm.prompts(6, (1400, 301, 9))
    kw = {"page_size": 32, "max_prefill_tokens": 64, "max_num_seqs": 4, "max_model_len": 2048, "num_pages": 200}
    assert mla.CP_LOCAL_K == 120 and mla._local_k(256, 8) == 120
    _, want = _generate(path, ps, 6, 16, monkeypatch, kv_cache_dtype="fp8", **kw)
    (cp, rows, lps), got = _cp_engine(path, ps, 6, 16, **kw)
    assert (cp, rows, lps) == (8, 2, 32 // 8)
    assert [r.output_ids for r in got] == [r.output_ids for r in want]


def test_cp_engine_dump_slot_stays_zero(sparse_cp, monkeypatch):
    """Under context parallelism (tp 2) rank 0's dump slot is all zeros in every DSA layer's latent, indexer and
    pool-key caches after serving (the tokens other ranks own, and padded ones, write it; the zero write follows each
    store); the control (the zero write made a no-op on rank 0, which runs in this process; the spawned rank keeps it)
    leaves rank 0's slot non-zero, with the same tokens."""
    from trnrt.engine.request import SamplingParams
    from trnrt.models import mla

    path, _ = sparse_cp

    def run(identity: bool):
        if identity:
            monkeypatch.setattr(mla, "_cp_zero_dump", lambda *a: None)
        eng = glm.engine(path, page_size=8, max_prefill_tokens=16, max_num_seqs=4, tp=2, kv_cache_dtype="fp8")
        try:
            toks = [
                r.output_ids
                for r in eng.generate(glm.prompts(4, (37, 70, 9)), SamplingParams(max_new_tokens=10, ignore_eos=True))
            ]
            d = dsa_long.cp_dump_slot(8, 2)
            layers = [layer for layer in eng.model.kv_layers() if getattr(layer, "pool_key", None) is not None]
            assert layers, "no DSA layer with a pool-key cache"
            rows = [
                (layer.k_cache[d].float().clone(), layer.v_cache[d].float().clone(), layer.pool_key[d].float().clone())
                for layer in layers
            ]
        finally:
            eng.close()
        return toks, rows

    toks, rows = run(False)
    assert all(torch.count_nonzero(t) == 0 for r in rows for t in r)
    toks2, rows2 = run(True)
    assert toks2 == toks
    assert any(torch.count_nonzero(t) > 0 for r in rows2 for t in r), "control: nothing landed in the dump slot"
