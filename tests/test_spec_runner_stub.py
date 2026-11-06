"""ModelRunner's speculative setup and warmup, on a stub model whose graph calls are recorders.

The graphs of the whole-call decode, prefill and verify are recorders returning the outputs a real model's would (shapes
only), so what runs for real is the runner: its bucket families and page ladders, the state pool it builds, the
speculative board and the small graphs around the verify (engine/spec_async.py, executed on the CPU here, so their
argument lists and shapes must agree with what the runner passes them), the MTP passes and the order of it all.

For each shipped speculative preset the warmup must request exactly the graphs its families name: the plain decode only at
the plain buckets and the page ladder, the verify family only at the speculative buckets and the speculative ladder, and
nothing at a page rung outside the ladder (the 3072-page rung exists for plain decode only)."""

from __future__ import annotations

import types

import pytest
import torch

from trnrt import __main__ as cli
from trnrt.config import prefill_pages
from trnrt.engine import model_runner as mr
from trnrt.engine.sampler import VERIFY_COLS

H = 8
COLS = 2 + 2 * 20


class StubModel:
    """What ModelRunner.__init__ and the warmup ask of a model; every graph call is replaced by a recorder (see _build)."""

    mtp = object()
    window = None
    tp_rank = 0
    tp_size = 1
    prefill_sp = False

    def __init__(self):
        self.norm = torch.zeros(H)
        self.sp_onehot = torch.zeros(1)

    def state_layers(self):
        return [0]

    def state_shapes(self):
        return [((3, 8), (2, 4, 4))]

    def bind_state(self, conv, rec):
        self.state_pools = (conv, rec)

    def kv_shapes(self):
        return [((1, 4), (1, 4))]

    def bind_kv_cache(self, *a, **k):
        self.kv_call = (a, k)

    def token_state_shapes(self):
        return []

    def aux_kv_shapes(self):
        return []

    def bind_aux_kv(self, caches):
        pass

    def parameters(self):
        return iter(())

    def _sp_on(self):
        return False

    def mtp_kv_takes(self, q, p):
        return q >= 64

    def forward_decode(
        self,
        input_ids,
        positions,
        block_table,
        context_lens,
        slot_mapping,
        temperature,
        top_p,
        top_k,
        min_p,
        noise,
        board,
        read_slot,
        write_slot,
        bitmask=None,
        penalties=None,
        swa_table=None,
        swa_first=None,
        state_slot=None,
        ngram_ids=None,
    ):
        raise AssertionError("replaced by a recorder")

    def forward_prefill(
        self,
        input_ids,
        positions,
        block_table,
        slot_mapping,
        last_index,
        temperature,
        top_p,
        top_k,
        min_p,
        noise,
        board,
        write_slot,
        bitmask=None,
        penalties=None,
        plp_targets=None,
        swa_table=None,
        swa_first=None,
        state_slot=None,
        ngram_ids=None,
    ):
        raise AssertionError("replaced by a recorder")

    def forward_extend(
        self,
        input_ids,
        positions,
        block_table,
        slot_mapping,
        temperature,
        top_p,
        top_k,
        min_p,
        noise,
        u_accept,
        draft,
        swa_table=None,
        swa_first=None,
        state_slot=None,
        ngram_ids=None,
    ):
        raise AssertionError("replaced by a recorder")

    def forward_mtp_k(
        self,
        input_ids,
        positions,
        block_table,
        slot_mapping,
        hsrc,
        hidx,
        last_index,
        src_norm,
        pos_rest=None,
        slot_rest=None,
        swa_table=None,
        swa_first=None,
        swa_rest=None,
        swa_first_rest=None,
        sp_onehot=None,
    ):
        B = input_ids.shape[0]
        k = 1 if pos_rest is None else 1 + pos_rest.shape[1]
        return torch.zeros(B, k), torch.zeros(B, H)

    def forward_mtp_kv(self, input_ids, positions, block_table, slot_mapping, hsrc, hidx, src_norm, sp_onehot=None):
        return torch.zeros(1)


def _recorders(log):
    def decode(*a):
        log.append(("decode", a[0].shape[0]))
        return torch.zeros(a[0].shape[0], COLS), torch.zeros(a[0].shape[0], H)

    def prefill(*a):
        log.append(("prefill", a[0].shape[0]))
        return torch.zeros(1, COLS), torch.zeros(a[0].shape[0], H)

    def extend(*a):
        B, Q = a[0].shape
        log.append(("verify", B))
        return torch.zeros(B * Q, VERIFY_COLS), torch.zeros(B * Q, H)

    return decode, prefill, extend


def build(preset: str, monkeypatch, *extra):
    ecfg = cli.engine_config(["--model", "m", "--preset", preset, "--device", "cpu", "--tp", "1", *extra])
    log: list = []
    monkeypatch.setattr(mr, "_piecewise", lambda *a, **k: _recorders(log))
    runner = mr.ModelRunner(StubModel(), types.SimpleNamespace(), ecfg, 8, torch.device("cpu"))
    return runner, ecfg, log


def keys_of(runner):
    return set(runner.compile_seconds)


def expected(c):
    """The graph keys a speculative engine's warmup builds, from its configuration alone."""
    Q, k = 1 + c.spec_k, c.spec_k
    dec, pb = c.resolved_decode_batch_buckets(), c.resolved_page_buckets()
    sp = tuple(sorted(c.spec_page_buckets or pb))
    plain_b = [b for b in dec if not c.plain_bucket_min or b >= c.plain_bucket_min]
    spec_b = [b for b in dec if not c.spec_bucket_max or b <= c.spec_bucket_max]
    out = set()
    if c.spec_max_batch:
        out |= {("decode", B, P) for B in plain_b for P in pb}
    for B in spec_b:
        for P in sp:
            out |= {
                ("verify", B, Q, P),
                ("spec_prep", B, Q, P),
                ("spec_post", B, Q),
                ("spec_host_init", B),
                ("spec_ckpt", B, Q),
                ("mtp_board", B, Q, P, B * Q, k),
            }
    fill = c.mtp_fill_rows
    for C in c.resolved_prefill_token_buckets():
        for P in prefill_pages(c.resolved_prefill_page_buckets(), c.prefill_page_min_rows, C):
            out.add(("prefill", C, P))
            Pm = P if not c.spec_page_buckets else min(x for x in sp if x >= P)
            q = C if not fill or fill >= C else fill
            out |= {("mtp", 1, q, Pm, C, k), ("mtp_prefill_ids", 1, q), ("mtp_post", 1, Q), ("spec_init", 1)}
            if c.mtp_kv_only and c.mtp_chunk_q and q >= 64:
                out.add(("mtp_kv", 1, q, Pm, C))
    out |= {("copy", n) for n in (4, 8, 16, 32)}  # the checkpoint copies
    return out


PRESETS = ["mtp-batch2", "mtp-batch8", "mtp-batch12", "mtp-batch24"]


@pytest.mark.parametrize("preset", PRESETS)
def test_the_warmup_requests_exactly_the_graphs_the_families_name(preset, monkeypatch):
    runner, ecfg, _ = build(preset, monkeypatch)
    runner.warmup()
    got, want = keys_of(runner), expected(ecfg)
    copies = {k for k in got if k[0] == "copy"}
    got -= copies
    want = {k for k in want if k[0] != "copy"}
    assert got - want == set(), sorted(got - want)[:6]
    assert want - got == set(), sorted(want - got)[:6]
    assert copies, "the checkpoint copies are warmed"


def test_mtp_batch8_builds_the_families_it_names(monkeypatch):
    """Literal counts, from the preset's own numbers: decode buckets 1 2 4 6 8, plain from 4, verify up to 4, pages 1024 3072
    4096 and the speculative ladder 1024 4096."""
    runner, _, _ = build("mtp-batch8", monkeypatch)
    runner.warmup()
    keys = keys_of(runner)
    assert {k[1:] for k in keys if k[0] == "decode"} == {(B, P) for B in (4, 6, 8) for P in (1024, 3072, 4096)}
    assert {k[1:] for k in keys if k[0] == "verify"} == {(B, 3, P) for B in (1, 2, 4) for P in (1024, 4096)}
    assert not [k for k in keys if k[0] in ("verify", "spec_prep", "mtp_board") and 3072 in k], (
        "a speculative graph at the 3072 rung"
    )
    assert {k for k in keys if k[0] == "spec_post"} == {("spec_post", B, 3) for B in (1, 2, 4)}


def test_mtp_batch2_is_static_speculation_with_no_plain_decode_graph(monkeypatch):
    runner, _, _ = build("mtp-batch2", monkeypatch)
    runner.warmup()
    keys = keys_of(runner)
    assert not [k for k in keys if k[0] == "decode"], "a static speculative engine never launches a plain decode"
    assert {k[1:] for k in keys if k[0] == "verify"} == {(B, 3, P) for B in (1, 2, 4, 6, 8) for P in (1024, 4096)}
    assert not [k for k in keys if k[0] == "mtp_kv"], "mtp-batch2 has no KV-only windows"


def test_mtp_batch24_has_two_disjoint_families(monkeypatch):
    runner, _, _ = build("mtp-batch24", monkeypatch)
    runner.warmup()
    keys = keys_of(runner)
    assert {k[1] for k in keys if k[0] == "decode"} == {16, 24}
    assert {k[1] for k in keys if k[0] == "verify"} == {1, 2, 4}
    with pytest.raises(RuntimeError, match="exceeds its warmed buckets"):
        mr.pick_family(5, runner.spec_buckets, "verify")
    assert runner.state.rows == 58 and runner.state.spec_seats == 4 and runner.state.mixed_seats


@pytest.mark.parametrize("preset", PRESETS)
def test_the_state_pool_has_the_rows_of_the_preset(preset, monkeypatch):
    runner, ecfg, _ = build(preset, monkeypatch)
    assert runner.state.rows == mr.state_pool_rows(ecfg) == {"mtp-batch24": 58}.get(preset, 34)


def test_a_speculative_engine_without_speculation_builds_none_of_it(monkeypatch):
    runner, ecfg, _ = build("batch12", monkeypatch)
    assert not runner.spec_async and runner._spec_fns == {} and runner._mtp is None and runner.async_ckpt_t == 0
    assert runner.plain_buckets == runner.spec_buckets == runner.decode_buckets
    assert runner.spec_page_buckets == tuple(sorted(runner.page_buckets)) and not runner.spec_page_set
    assert runner.mtp_q_buckets == () and runner.state.rows == mr.state_pool_rows(ecfg)
