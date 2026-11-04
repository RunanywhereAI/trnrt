"""Context-parallel DSA (models/mla.py attention_cp / attention_cp_rows) at function level, on the host, the group's
all-reduce emulated over A simulated ranks (each rank's zero-padded contribution, summed):

- the merge's two list gathers as ONE zero-padded fp32 gather (_cp_merge_gather): bit for bit the two separate
  gathers, signed zeros, NEG_INF padding and pool indices up to 2^24 - 1 included, for several rank orders of the sum,
  with a planted swapped pack as the positive control;
- the indexer query sharded by head and gathered back with the latent queries in the same gather (_idx_q_lat_rows):
  every rank's indexer query is the replicated projection's (bf16 and FP8 block-scaled weights) and its q_all is the
  latent queries' own gather bit for bit; a dtype mismatch is refused, not cast; a rank in another rank's place is
  caught;
- the dump slot (_cp_zero_dump): zeroed by a one-row write after each store, every other row bit for bit; both CP sites
  store as served, then zero (an AST pin with planted breaks); padded and unowned writes both land zero;
- dsa_long.compact's count clamped to [0, keep] on a malformed (NaN) mask;
- the decode-only bf16 copies of the FP8 DSA projections (_wd), kept out of the piecewise graphs' tensors;
- the row caps of the fused DSA layer and the plain shell.
"""

from __future__ import annotations

import ast
import inspect
import types

import pytest
import torch
import torch.nn.functional as F

from trnrt.models import dsa_long, mla
from trnrt.models.decoder import DecoderForCausalLM
from trnrt.models.quant import dequant

NEG_INF = -1e30


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()])


def _model(a: int, A: int, reduce, index_of=lambda a: a, dtype=torch.bfloat16):
    oh = torch.zeros(A)
    oh[index_of(a)] = 1
    return types.SimpleNamespace(
        cp=A,
        dtype=dtype,
        long_grp_index=torch.tensor([index_of(a)]),
        long_grp_onehot=oh,
        cp_group=None,
        attn_group=None,
        _all_reduce=reduce,
    )


def _emulate(fn, A: int, index_of=lambda a: a):
    """fn(model, a) on every simulated rank: a first pass records each rank's zero-padded contribution, a second returns
    their sum as the group's all-reduce does."""
    parts = []
    for a in range(A):
        fn(_model(a, A, lambda t, pg: (parts.append(t.clone()), t)[1], index_of), a)
    total = parts[0]
    for p in parts[1:]:
        total = total + p
    return [fn(_model(a, A, lambda t, pg: total, index_of), a) for a in range(A)]


# -- the merge's lists in one gather -------------------------------------------------------------------------------


def _lists(A: int, T: int, keep: int, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    lval, cpool = [], []
    for a in range(A):
        v = torch.randn(T, keep, generator=g)
        v[:, keep - 5 * (a + 1) :] = NEG_INF  # padding past a short list
        v[0, 0], v[0, 1] = 0.0, -0.0
        c = torch.randint(0, 1 << 24, (T, keep), generator=g).float()
        c[1, 0] = (1 << 24) - 1
        lval.append(v)
        cpool.append(c)
    return lval, cpool


@pytest.mark.parametrize("order", [None, "reversed", "shuffled"])
def test_the_merged_gather_is_the_two_gathers_bit_for_bit(order):
    A, T, keep = 8, 4, 512
    lval, cpool = _lists(A, T, keep)
    perm = {None: list(range(A)), "reversed": list(range(A))[::-1], "shuffled": [3, 0, 7, 1, 6, 2, 5, 4]}[order]

    def gather(xs):
        parts = []
        for a in range(A):
            mla._cp_gather(_model(a, A, lambda t, pg: (parts.append(t.clone()), t)[1]), xs[a])
        total = parts[perm[0]]
        for r in perm[1:]:
            total = total + parts[r]
        return total

    def merged(lv, cp):
        parts = []
        for a in range(A):
            mla._cp_merge_gather(_model(a, A, lambda t, pg: (parts.append(t.clone()), t)[1]), lv[a], cp[a], keep)
        total = parts[perm[0]]
        for r in perm[1:]:
            total = total + parts[r]
        return mla._cp_merge_gather(_model(0, A, lambda t, pg: total), lv[0], cp[0], keep)

    gv, gc = merged(lval, cpool)
    assert torch.equal(_bits(gv), _bits(gather(lval))) and torch.equal(_bits(gc), _bits(gather(cpool)))
    # the positive control: the two lists' slices swapped in the pack give other bits
    swapped = merged(cpool, lval)
    assert not torch.equal(_bits(swapped[0]), _bits(gv))


# -- the indexer query sharded by head, gathered with the latent queries ------------------------------------------

HI, DI, K, T, NH, R = 8, 16, 32, 3, 2, 8


def _idx_layer(fp8: bool, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(HI * DI, K, generator=g) * 0.2
    dsa = types.SimpleNamespace(n_heads=HI, head_dim=DI)
    layer = types.SimpleNamespace(spec=types.SimpleNamespace(mla=types.SimpleNamespace(dsa=dsa)))
    if fp8:  # blocks of 8 along K, scales per output row
        layer.idx_wq = (w * 4).to(torch.float8_e4m3fn)
        layer.idx_wq_scale = torch.rand(HI * DI, K // 8, generator=g) * 0.5 + 0.25
    else:
        layer.idx_wq = w.bfloat16()
    return layer


def _q_lats(A: int, seed: int = 2):
    """Every rank's q_lat [T, NH, R] bf16, with signed zeros planted (a zero-padded sum adds zeros to each value)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for a in range(A):
        q = torch.randn(T, NH, R, generator=g).bfloat16()
        q[0, 0, a] = -0.0
        q[1, 1, a] = 0.0
        out.append(q)
    return out


@pytest.mark.parametrize("fp8", [False, True])
def test_every_rank_gets_the_replicated_query_and_its_own_latent_gather(fp8):
    A = 4
    layer = _idx_layer(fp8)
    q_resid = torch.randn(T, K, generator=torch.Generator().manual_seed(1)).bfloat16()
    ql = _q_lats(A)
    ref = F.linear(q_resid, dequant(layer.idx_wq, getattr(layer, "idx_wq_scale", None), torch.bfloat16)).view(T, HI, DI)
    coal = _emulate(lambda m, a: mla._idx_q_lat_rows(m, layer, q_resid, ql[a]), A)
    lat = _emulate(lambda m, a: mla._cp_gather(m, ql[a]).permute(1, 0, 2, 3).reshape(T, A * NH, R), A)
    for a in range(A):
        qi, q_all = coal[a]
        assert qi.shape == ref.shape and torch.equal(qi, ref), (fp8, a)
        assert torch.equal(_bits(q_all), _bits(lat[a])), a
        for b in range(A):  # head b NH + j of q_all is rank b's head j
            assert torch.equal(q_all[:, b * NH : (b + 1) * NH], ql[b]), (a, b)


def test_a_dtype_mismatch_is_refused_not_cast():
    layer = _idx_layer(False)
    q_resid = torch.randn(T, K, generator=torch.Generator().manual_seed(1)).bfloat16()
    with pytest.raises(ValueError, match="differ"):
        mla._idx_q_lat_rows(_model(0, 4, lambda t, pg: t), layer, q_resid, _q_lats(4)[0].float())


def test_a_rank_on_another_ranks_heads_is_caught():
    """The positive control: rank 1 projecting rank 0's head block (and filling rank 0's place in the gather) gives a
    query that differs from the replicated one."""
    A = 4
    layer = _idx_layer(False)
    q_resid = torch.randn(T, K, generator=torch.Generator().manual_seed(1)).bfloat16()
    ql = _q_lats(A)
    ref = F.linear(q_resid, layer.idx_wq).view(T, HI, DI)
    bad = _emulate(lambda m, a: mla._idx_q_lat_rows(m, layer, q_resid, ql[a]), A, index_of=lambda a: 0 if a == 1 else a)
    assert not torch.equal(bad[2][0], ref)


# -- the dump slot --------------------------------------------------------------------------------------------------

FP8_MODEL = types.SimpleNamespace(fp8_max=448.0)
FP8_MODEL._store = lambda cache, slots, x: DecoderForCausalLM._store(FP8_MODEL, cache, slots, x)


@pytest.mark.parametrize("form", ["fp8 store", "index_put bf16", "index_put fp32"])
def test_dump_row_zeroed_and_the_rest_kept_bit_for_bit(form):
    g = torch.Generator().manual_seed(0)
    dt = {"fp8 store": torch.float8_e4m3fn, "index_put bf16": torch.bfloat16, "index_put fp32": torch.float32}[form]
    cache = torch.randn(40, 1, 6, generator=g) * 4
    cache[31] = float("nan")  # the dump row, NaN from an unowned token's write
    cache[4, 0, 2] = float("nan")  # a NaN in a live row: kept (its own request's cache)
    cache = cache.to(dt)
    before = cache.clone()
    like = torch.zeros(3, 6, dtype=torch.bfloat16)  # the stored rows' dtype (k_cache's c, bf16)
    mla._cp_zero_dump(FP8_MODEL if form == "fp8 store" else None, cache, 31, like)
    assert cache.dtype == dt and cache.shape == before.shape
    assert torch.count_nonzero(_bits(cache)[31]) == 0
    rest = torch.arange(40) != 31
    assert torch.equal(_bits(cache)[rest], _bits(before)[rest])
    assert torch.isnan(cache[4, 0, 2].float())


SERVED = (
    "model._store(layer.k_cache, ls, c.unsqueeze(1))",
    "model._store(layer.v_cache, ls, ki.unsqueeze(1))",
    "layer.pool_key.index_put_((slots4,), keys.reshape(T * kp, Di // kp).to(layer.pool_key.dtype))",
)
ZEROS = (
    "_cp_zero_dump(model, layer.k_cache, dump, c)",
    "_cp_zero_dump(model, layer.v_cache, dump, ki)",
    "_cp_zero_dump(None, layer.pool_key, dump, keys)",
)


def _pinned(src: str) -> bool:
    """The function's k_cache / v_cache stores and its pool-key index_put_ are the served lines, and each is followed, as
    the next statement of its block, by the zero write of the same cache (and nothing else writes those caches)."""
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef))
    pairs, writes = [], []
    for n in ast.walk(fn):
        for field in ("body", "orelse"):
            stmts = getattr(n, field, [])
            block = [ast.unparse(x) for x in stmts if isinstance(x, ast.stmt)] if isinstance(stmts, list) else []
            pairs += list(zip(block, block[1:] + [""]))
    for a, b in pairs:
        if a.startswith(("model._store(layer.k_cache", "model._store(layer.v_cache", "layer.pool_key.index_put_")):
            writes.append((a, b))
    return sorted(writes) == sorted(zip(SERVED, ZEROS))


@pytest.mark.parametrize("fn", [mla.attention_cp, mla.attention_cp_rows])
def test_both_cp_sites_store_as_served_then_zero_the_dump_row(fn):
    assert _pinned(inspect.getsource(fn))


@pytest.mark.parametrize("site", range(3))
@pytest.mark.parametrize("plant", ["drop the zero", "select in the store", "zero before the store"])
def test_a_planted_change_breaks_the_pin(site, plant):
    import textwrap

    src = textwrap.dedent(inspect.getsource(mla.attention_cp))
    store, zero = SERVED[site], ZEROS[site]
    assert src.count(store) == 1 and src.count(zero) == 1
    if plant == "drop the zero":
        bad = src.replace("    " + zero + "\n", "", 1)
    elif plant == "select in the store":
        bad = src.replace(
            store,
            store.replace("c.unsqueeze(1)", "torch.where(real.view(-1, 1), c, 0).unsqueeze(1)")
            .replace("ki.unsqueeze(1)", "torch.where(real.view(-1, 1), ki, 0).unsqueeze(1)")
            .replace("keys.reshape(", "torch.where(real4, keys, 0).reshape("),
            1,
        )
    else:
        i, j = src.index(store), src.index(zero)
        line_s = src[src.rindex("\n", 0, i) + 1 : src.index("\n", i) + 1]
        line_z = src[src.rindex("\n", 0, j) + 1 : src.index("\n", j) + 1]
        bad = src.replace(line_s + line_z, line_z + line_s, 1)
    assert bad != src and not _pinned(bad)


def test_padded_and_unowned_writes_both_land_zero():
    """The sites' own slot formulas (real = slot_mapping >= page_size, the owner's local slot from
    dsa_long.cp_local_slots, the dump slot for the rest), their store, then the zero write: a NaN padded row and a NaN
    token another rank owns both land in the dump row, which ends zero, in the indexer / latent rows (ls) and the
    pool-key pieces (slots4); a NaN token this rank owns keeps its own NaN in its own slot; no other row moves."""
    ps, A, kp, rank = 16, 4, 4, torch.tensor([1])
    table = torch.tensor([3, 5, 7, 9])  # one sequence's pages (never the null page 0)
    positions = torch.tensor([0, 4, 5, 8, 12, 0])  # pools 0, 1, 1, 2, 3 and a padded row at position 0
    slot_mapping = torch.tensor([3 * ps, 3 * ps + 4, 3 * ps + 5, 3 * ps + 8, 3 * ps + 12, 1])  # the last: padded
    real = slot_mapping >= ps
    dump = dsa_long.cp_dump_slot(ps, A)
    ls = torch.where(real, dsa_long.cp_local_slots(positions, table, ps, kp, A, rank), torch.full_like(positions, dump))
    owned = (positions // kp) % A == 1
    assert owned.tolist() == [False, True, True, False, False, False] and not real[-1]
    assert bool((ls[~owned] == dump).all()) and bool((ls[owned] != dump).all())
    x = torch.full(
        (6, 8), float("nan")
    )  # every row NaN: the padded row (5), unowned real tokens (0, 3, 4), owned (1, 2)
    cache = torch.randn(10 * ps // A, 8)
    before = cache.clone()
    cache.index_put_((ls,), x)
    mla._cp_zero_dump(None, cache, dump, x)
    assert torch.count_nonzero(cache[dump]) == 0 and torch.isnan(cache[ls[owned]]).all()
    other = torch.ones(cache.shape[0], dtype=torch.bool)
    other[ls] = False
    assert torch.equal(cache[other], before[other])
    first = positions - positions % kp
    pos4 = (first.unsqueeze(-1) + torch.arange(kp)).reshape(-1)
    slots4 = torch.where(
        real.repeat_interleave(kp), dsa_long.cp_local_slots(pos4, table, ps, kp, A, rank), torch.full_like(pos4, dump)
    )
    own4 = owned.repeat_interleave(kp)
    assert bool((slots4[~own4] == dump).all())
    pk = torch.randn(10 * ps // A, 2)
    pk.index_put_((slots4,), torch.full((6 * kp, 2), float("nan")))
    mla._cp_zero_dump(None, pk, dump, x)
    assert torch.count_nonzero(pk[dump]) == 0 and torch.isnan(pk[slots4[own4]]).all()


def test_compact_count_is_in_range_for_a_nan_mask():
    """dsa_long.compact's count clamped to [0, keep]: a mask with a NaN lane (a device top-k's output over NaN scores)
    gives an in-range count and in-range ids on any host (the NaN's int cast is host-dependent: 0 on arm64, int64 min
    on x86); a mask with more than keep ones is the control that fails without the clamp on every host; a well-formed
    mask's count and ids are unchanged."""
    keep, M = 8, 64
    sel = torch.zeros(3, M)
    sel[0, [3, 9, 20]] = 1.0
    sel[1, [1, 2, 5]] = 1.0
    sel[1, 40] = float("nan")
    sel[2, :12] = 1.0  # 12 ones > keep
    idx, cnt = dsa_long.compact(sel, keep)
    assert cnt.tolist()[0] == 3 and idx[0, :3].tolist() == [3, 9, 20]
    assert all(0 <= int(c) <= keep for c in cnt) and bool(((idx >= 0) & (idx < M)).all())
    assert int(cnt[2]) == keep and idx[2].tolist() == list(range(keep))
    raw = sel.view(3, -1).sum(-1).to(torch.int64)  # the count before the clamp, as compact cast it
    assert int(raw[2]) == 12 > keep, "control: the unclamped count of an over-full mask exceeds keep"


# -- the decode-only bf16 copies and the row caps -------------------------------------------------------------------


def _fp8_layer(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    layer = types.SimpleNamespace()
    for n, shape in (("w_a", (256, 384)), ("q_b", (128, 256)), ("o", (384, 128))):
        w = torch.randn(*shape, generator=g)
        s = torch.rand(shape[0], shape[1] // 128, generator=g) * 0.02 + 0.001
        setattr(layer, n, (w / s.repeat_interleave(128, -1)).clamp(-240, 240).to(torch.float8_e4m3fn))
        setattr(layer, n + "_scale", s)
        setattr(layer, n + "_dec", dequant(getattr(layer, n), s, torch.bfloat16))  # as load_layer fills it
    model = types.SimpleNamespace(dtype=torch.bfloat16)
    model._w = lambda layer_, name: dequant(getattr(layer_, name), getattr(layer_, name + "_scale"), model.dtype)
    return model, layer


def test_the_copy_in_decode_calls_and_the_dequant_otherwise():
    """A decode call reads a DSA projection's bf16 copy; every other call dequantizes the FP8 weight in-graph (the same
    values); a layer without copies dequantizes in decode too."""
    model, layer = _fp8_layer()
    assert mla.DECODE_BF16_NAMES == ("w_a", "q_b", "o")
    for n in mla.DECODE_BF16_NAMES:
        assert mla._wd(model, layer, n, True) is getattr(layer, n + "_dec")
        got = mla._wd(model, layer, n, False)
        assert got is not getattr(layer, n + "_dec") and torch.equal(got, getattr(layer, n + "_dec")), n
    bare = types.SimpleNamespace(**{k: v for k, v in vars(layer).items() if not k.endswith("_dec")})
    assert torch.equal(mla._wd(model, bare, "w_a", True), model._w(bare, "w_a"))


def test_the_copies_stay_out_of_the_piecewise_graphs_tensors():
    """decoder._layer_split (every piecewise prefill graph's explicit tensors) leaves the decode-only copies out, so a
    layer keeping them has the kind and tensors of one without; the whole-step decode traces the module and reads them."""
    from torch import nn

    def layer(copies: bool):
        L = nn.Module()
        L.spec = "dsa"
        L.w_a = nn.Parameter(torch.zeros(4, 8, dtype=torch.float8_e4m3fn), requires_grad=False)
        L.w_a_scale = nn.Parameter(torch.ones(4, 1), requires_grad=False)
        if copies:
            L.w_a_dec = nn.Parameter(torch.zeros(4, 8, dtype=torch.bfloat16), requires_grad=False)
            L._decode_only = ("w_a_dec",)
        return L

    with_, without = (types.SimpleNamespace(layers=[layer(c)]) for c in (True, False))
    split = DecoderForCausalLM._layer_split
    assert list(split(with_, 0)[1]) == list(split(without, 0)[1]) == ["w_a", "w_a_scale"]
    kind = DecoderForCausalLM.layer_kind
    assert kind(types.SimpleNamespace(layers=with_.layers, _layer_split=lambda i: split(with_, i)), 0) == kind(
        types.SimpleNamespace(layers=without.layers, _layer_split=lambda i: split(without, i)), 0
    )


def test_row_caps():
    """The fused DSA layer takes decode calls of up to LAYER_MK_MAXB rows, the plain shell up to SHELL_MAXB; hybrid's
    fused-DSA branch asks the same cap, so a step past it keeps the plain path."""
    from trnrt.models import hybrid

    assert (mla.LAYER_MK_MAXB, mla.SHELL_MAXB) == (8, 6)
    assert [mla._mk_rows_on(t) for t in (1, 6, 8, 9)] == [True, True, True, False]
    assert [mla._shell_rows_on(t) for t in (1, 6, 8)] == [True, True, False]
    assert [hybrid._mk4_rows_on(t) for t in (1, 6, 8, 9)] == [True, True, True, False]


# -- the decode scores of a rank's local pools (kernels/dsa_index.py over the pool-key page rows) -------------------

CT_, PAGES, PPK, DI_, HI_ = 2, 128, 8, 128, 4


def _scores_case(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    pool_key = (torch.randn(PAGES * 3 * PPK, DI_, generator=g) * 0.5).bfloat16()  # 3 x the pages a row uses
    table = torch.stack([torch.randperm(PAGES * 3, generator=g)[:PAGES] for _ in range(CT_)])
    # the reference pool rows from dsa_long.cp_pool_rows, not the page-row view the kernel path uses: the two
    # derivations could disagree about the layout (page_size 64, kpool 4, cp 2: 8 per page)
    prow = dsa_long.cp_pool_rows(table, PPK * 4 * 2, 4, 2)
    pk = pool_key[prow]  # [T, Pl, Di]
    q = torch.randn(CT_, HI_, DI_, generator=g).bfloat16()
    w = torch.randn(CT_, HI_, generator=g)
    nloc = torch.tensor([PAGES * PPK - 37, 300])
    dsa = types.SimpleNamespace(head_dim=DI_)
    layer = types.SimpleNamespace(
        spec=types.SimpleNamespace(mla=types.SimpleNamespace(dsa=dsa)), pool_key=pool_key, v_cache=None
    )
    return layer, q, w, pk, table, nloc


def test_kernel_scores_equal_the_xla_scores():
    """_cp_scores of a decode call of whole 128-page groups (dsa_index's emulation on the host) gives dsa_long.scores over
    the gathered pool keys up to fp32 rounding, NEG_INF past each row's local count."""
    layer, q, w, pk, table, nloc = _scores_case()
    scale = DI_**-0.5
    want = dsa_long.scores(q, w, pk, nloc, scale)
    got = mla._cp_scores(None, layer, q, w, pk, table, nloc, scale)
    live = want > -1e29
    assert torch.equal(live, got > -1e29)
    torch.testing.assert_close(got[live], want[live], rtol=2e-6, atol=2e-6)
    assert (got[~live] < -1e29).all()


def test_a_shuffled_pool_order_is_caught():
    layer, q, w, pk, table, nloc = _scores_case(1)
    scale = DI_**-0.5
    want = dsa_long.scores(q, w, pk.view(CT_, PAGES, PPK, DI_).flip(2).reshape(CT_, -1, DI_), nloc, scale)
    got = mla._cp_scores(None, layer, q, w, pk, table, nloc, scale)
    live = got > -1e29
    assert (got[live] - want[live]).abs().max() > 1e-2


def test_prefill_and_partial_groups_keep_xla(monkeypatch):
    layer, q, w, pk, table, nloc = _scores_case(2)
    calls = []
    monkeypatch.setattr(dsa_long, "scores", lambda *a: calls.append(1) or torch.zeros(1))
    mla._cp_scores(None, layer, q, w, pk, table[:, :120], nloc, 1.0)  # not whole 128-page groups
    mla._cp_scores(None, layer, q, w, pk, table[0], nloc, 1.0)  # a prefill chunk's table [P]
    assert calls == [1, 1]
