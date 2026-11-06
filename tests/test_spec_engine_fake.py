"""The engine's speculative step with the scheduler, the page pool, the radix cache and the state pool, against a fake runner.

The fake keeps the device's side of the bargain, in the form of a model that is a pure function of the token history
(tests/test_scheduler.py), so what every request must generate is known exactly whatever the acceptance:

- the speculative board (engine/spec_async.py) holds each request's newest token, its position, the state row to read and
  whether its drafts are real; a blind verify reads it, not the host;
- every state row holds the number of tokens it has processed, and every graph call checks that the row it reads holds the
  count it should: a plain decode at position p reads a row of p tokens, a verify at base b reads the row `cur` of b tokens
  and writes rows[j] = b + 1 + j, the next base's row is rows[acc], a restored or copied checkpoint of b tokens is read by a
  chunk that starts at b;
- every KV position a call writes lies on a page the scheduler reserved, and a blind verify's `start` is an upper bound of
  the position the board holds.

So what this tests is the composition: that the host's bookkeeping of a step it has not read back (the pages, the start bound,
the blind steps in flight, the plain -> speculative flip, the speculative seats, the decode checkpoints reserved before the
accepted count is known) stays consistent with what the device does with the same step."""

from __future__ import annotations

import itertools
import random
import types
from collections import defaultdict

import pytest
import torch

from tests.test_scheduler import EOS, VOCAB, next_token, reference_output
from trnrt.config import EngineConfig
from trnrt.engine import engine as E
from trnrt.engine import state_pool as SP
from trnrt.engine.host_cache import HostTier
from trnrt.engine.kv_pool import PagePool
from trnrt.engine.radix_cache import RadixCache
from trnrt.engine.request import SamplingParams, Status
from trnrt.engine.sampler import NUM_TOP_LOGPROBS, VERIFY_COLS
from trnrt.engine.scheduler import PENDING, Scheduler, SchedulerConfig

PS = 4
K = 2
Q = 1 + K


class Dev:
    """A device output: .cpu() is the host's copy."""

    def __init__(self, t: torch.Tensor):
        self.t = t

    def cpu(self) -> torch.Tensor:
        return self.t


class _Model:
    def state_shapes(self):
        return [((3, 8), (2, 4, 4))]

    def bind_state(self, conv, rec):
        self.conv, self.rec = conv, rec

    def aux_state_shapes(self):
        return []

    def bind_aux_state(self, aux):
        pass


class FakeRunner:
    """ModelRunner's interface as the engine uses it, with the device's semantics (see the module docstring)."""

    def __init__(self, cfg: EngineConfig, state, rng: random.Random, accept: tuple[float, ...]):
        self.cfg, self.state, self.rng, self.accept = cfg, state, rng, accept
        self.decode_buckets = (1, 2, 4, 8, 16)
        self.async_ckpt_t = PS
        self.launch_bound = types.SimpleNamespace(done_through=lambda t: None)
        self.watchdog = None
        self.last_hidden = None
        self.content: dict[int, int] = {}  # state row -> tokens it has processed
        self.hist: dict[
            str, list[int]
        ] = {}  # rid -> the device's token history (every token the device chain produced)
        self.board: dict[str, dict] = {}
        self.last_rows: list[str] = []
        self.calls: dict[str, int] = defaultdict(int)
        self.copies = 0
        self.host_kv: dict[int, int] = {}  # host tier slot -> the page it was saved from
        self.host_ckpt: dict[int, int] = {}  # host tier slot -> the tokens its state row had processed

    # -- helpers -------------------------------------------------------------------

    def _row(self, req) -> int:
        return self.state.row(req)

    def _kv_ok(self, req, positions_end: int) -> None:
        """Every KV position a call writes (up to positions_end) is on a page the scheduler reserved, except what lies past
        max_model_len (padding writes go to the null page)."""
        top = min(positions_end, self.cfg.max_model_len)
        assert len(req.pages) * PS >= top, f"{req.rid}: KV up to {top} but only {len(req.pages)} pages"

    def synth_active(self) -> bool:
        return False

    def _chain(self, r, start: int) -> list[int]:
        """The device's token history of r up to its newest token, at position `start`: the chain its calls produced, plus,
        when the newest token is a prompt token no call has seen (a prompt whose chunks stopped one token short, or a
        recompute after a preemption), the host's token."""
        h = self.hist[r.rid]
        toks = r.token_ids[: start + 1]
        if len(h) == start:
            assert PENDING not in toks
            h.append(toks[start])
        assert len(h) == start + 1, f"{r.rid}: a call at {start}, the device chain is {len(h)} tokens"
        known = [t for t in toks if t != PENDING]
        assert known == h[: len(known)], f"{r.rid}: the host's tokens differ from the device's"
        return h

    def forget(self, req) -> None:
        self.hist.pop(req.rid, None)
        self.board.pop(req.rid, None)
        self.state.release(req)

    def copy_state(self, src, dst) -> None:
        self.copies += len(src)
        for a, b in zip(src, dst, strict=True):
            self.content[b] = self.content.get(a)

    # -- the host tier's copies (engine/host_cache.py): a restored checkpoint is the state of the tokens it was saved at ------

    def kv_save(self, slot, page) -> None:
        self.host_kv[slot] = page

    def kv_load(self, slot, page) -> None:
        assert slot in self.host_kv, f"a restore of host slot {slot}, which holds nothing"
        self.calls["kv_load"] += 1

    def kv_drop(self, slot) -> None:
        self.host_kv.pop(slot, None)

    def state_save(self, slot, row) -> None:
        assert self.content.get(row) is not None, f"a checkpoint saved from row {row}, which holds no state"
        self.host_ckpt[slot] = self.content[row]

    def state_load(self, slot, row) -> None:
        assert slot in self.host_ckpt
        self.content[row] = self.host_ckpt[slot]
        self.calls["state_load"] += 1

    def state_drop(self, slot) -> None:
        self.host_ckpt.pop(slot, None)

    # -- plain graphs -------------------------------------------------------------------

    def prefill(self, s) -> Dev:
        self.calls["prefill"] += 1
        r = s.req
        row = self._row(r)
        if s.start > 0:
            assert self.content.get(row) == s.start, (
                f"{r.rid}: a chunk at {s.start} reads a row of {self.content.get(row)}"
            )
        self._kv_ok(r, s.end)
        if s.mid is not None:
            self.content[s.mid[1]] = s.start + s.mid[0]
        self.content[row] = s.end
        toks = r.token_ids[: s.end]
        assert PENDING not in toks
        self.hist[r.rid] = list(toks)
        out = torch.zeros(1, 2 + 2 * NUM_TOP_LOGPROBS)
        if s.sample:
            t = next_token(self.hist[r.rid])
            self.hist[r.rid].append(t)
            out[0, 0] = t
        return Dev(out)

    def decode(self, seqs) -> Dev:
        self.calls["decode"] += 1
        out = torch.zeros(len(seqs), 2 + 2 * NUM_TOP_LOGPROBS)
        for i, s in enumerate(seqs):
            r = s.req
            row = self._row(r)
            h = self._chain(r, s.start)
            assert self.content.get(row) == s.start, (
                f"{r.rid}: decode at {s.start} reads a state of {self.content.get(row)}"
            )
            self._kv_ok(r, s.end)
            t = next_token(h)
            h.append(t)
            self.content[row] = s.end
            out[i, 0] = t
        return Dev(out)

    # -- the blind step ----------------------------------------------------------------

    def spec_after_prefill(self, seqs, hkey) -> None:
        (s,) = seqs
        if s.sample:
            r = s.req
            self.board[r.rid] = {"base": s.end, "cur": self._row(r), "nd": 1}
            r.spec_ready = True

    def verify_async(self, seqs, q: int):
        assert q == Q
        self.calls["verify"] += 1
        B = next(b for b in self.decode_buckets if b >= len(seqs))
        out = torch.zeros(B * Q, VERIFY_COLS)
        res = torch.zeros(B, Q + 2)
        self.last_rows = []
        for b, s in enumerate(seqs):
            r = s.req
            rows = self.state.rows_of(r)[:Q]
            if not r.spec_ready:  # spec_host_init: its newest token is a token the host holds
                assert PENDING not in r.token_ids[: s.start + 1], f"{r.rid}: a host init with a token still in flight"
                self._chain(r, s.start)
                self.board[r.rid] = {"base": s.start, "cur": self._row(r), "nd": 0}
                r.spec_ready = True
            bd = self.board[r.rid]
            base, h = bd["base"], self.hist[r.rid]
            assert base <= s.start, f"{r.rid}: the board's base {base} is past the host's upper bound {s.start}"
            assert len(h) == base + 1
            self._kv_ok(r, base + Q)
            assert self.content.get(bd["cur"]) == base, (
                f"{r.rid}: verify at {base} reads a state of {self.content.get(bd['cur'])}"
            )
            drafted = bd["nd"] > 0
            acc = self.rng.choices(range(K + 1), weights=self.accept)[0] if drafted else 0
            emitted = []
            for _ in range(acc + 1):
                emitted.append(next_token(h))
                h.append(emitted[-1])
            old = base
            for j in range(Q):
                self.content[rows[j]] = old + 1 + j
            t = self.async_ckpt_t
            if s.ckpt_row:
                c = (old + acc + 1) // t * t
                if c > old:
                    self.content[s.ckpt_row] = self.content[rows[c - 1 - old]]
            bd.update(base=old + 1 + acc, cur=rows[acc])
            res[b, : acc + 1] = torch.tensor(emitted, dtype=torch.float32)
            res[b, Q], res[b, Q + 1] = acc, float(drafted)
            for j in range(Q):  # the logprob columns carry which column they are, so a mix-up shows
                out[b * Q + j, 3:6] = torch.tensor([100.0 * j + 3, 100.0 * j + 4, 100.0 * j + 5])
            self.last_rows.append(r.rid)
        return Dev(out), Dev(res)

    def mtp_async(self) -> None:
        for rid in self.last_rows:
            self.board[rid]["nd"] = 1


def make_engine(
    *,
    num_pages=96,
    seats=6,
    ckpts=10,
    spec_seats=None,
    spec_max_batch=0,
    max_len=160,
    accept=(0.2, 0.3, 0.5),
    seed=0,
    max_prefill=24,
    host=False,
):
    cfg = EngineConfig(
        model_path="m",
        page_size=PS,
        max_num_seqs=seats,
        max_model_len=max_len,
        max_prefill_tokens=max_prefill,
        spec_method="mtp",
        spec_k=K,
        spec_max_batch=spec_max_batch,
        state_spec_seats=spec_seats,
        state_checkpoints=ckpts,
    )
    eng = E.LLMEngine.__new__(E.LLMEngine)
    eng.cfg = cfg
    eng.pool = PagePool(num_pages)
    eng.radix = RadixCache(eng.pool, PS)
    state = SP.StatePool(_Model(), seats, torch.device("cpu"), torch.bfloat16, ckpts, 1 + K, spec_seats)
    scfg = SchedulerConfig(
        page_size=PS,
        max_num_seqs=seats,
        max_prefill_tokens=max_prefill,
        max_model_len=max_len,
        eos_token_ids=(EOS,),
        admission_decode_tokens=8,
    )
    eng.scheduler = Scheduler(scfg, eng.pool, eng.radix, draft_fn=eng._propose)
    eng.scheduler.states = state
    eng.radix.free_ckpt = state.free_ckpt
    eng.runner = FakeRunner(cfg, state, random.Random(seed), accept)
    eng.host_tier = None
    if host:  # as LLMEngine.__init__ wires the host tier under the radix cache
        eng.host_tier = HostTier(eng.runner, PS, 64, 16)
        eng.radix.offload, eng.radix.offload_ckpt = eng.host_tier.offload, eng.host_tier.offload_ckpt
        eng.radix.offload_ckpt_hash = eng.host_tier.offload_ckpt_hash
        eng.scheduler.host = eng.host_tier
    eng.tokenizer = None
    eng._ids = itertools.count()
    eng._inflight = None
    eng.last_step = E.StepStats()
    eng.spec_proposed = eng.spec_accepted = eng.spec_verifies = 0
    eng.spec_pos_reached, eng.spec_pos_accepted = [], []
    eng._dyn_plain = False
    eng.dyn_flips = eng.dyn_seat_moves = eng.dyn_plain_steps = eng.dyn_spec_steps = 0
    return eng


def prompts(n, seed=1, shared=True):
    rng = random.Random(seed)
    system = [rng.randrange(8, VOCAB) for _ in range(16)]
    return [
        ([*system] if shared and i % 2 == 0 else []) + [rng.randrange(8, VOCAB) for _ in range(rng.randrange(3, 30))]
        for i in range(n)
    ]


def run(eng, plist, max_new=(1, 40), seed=2):
    rng = random.Random(seed)
    reqs = [eng.add_request(p, SamplingParams(max_new_tokens=rng.randrange(*max_new), logprobs=0)) for p in plist]
    steps = 0
    while eng.has_work():
        eng.step()
        steps += 1
        assert steps < 5000
    return reqs


def check_done(eng, reqs):
    for r in reqs:
        assert r.status is Status.FINISHED and r.spec_inflight == 0, r.rid
        assert r.output_ids == reference_output(r.prompt_ids, r.params.max_new_tokens), r.rid
    pool, radix = eng.pool, eng.radix
    assert pool.num_free + radix.total_pages() == pool.num_usable, "a page leaked"
    st = eng.runner.state
    assert not st._held, "a request still holds state rows"
    assert len(st._ckpt_free) + radix.num_ckpts == st.num_ckpt_rows, "a checkpoint row leaked or is held twice"


@pytest.mark.parametrize("accept", [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.2, 0.3, 0.5)])
def test_a_static_speculative_engine_generates_the_reference_whatever_the_acceptance(accept):
    eng = make_engine(accept=accept)
    reqs = run(eng, prompts(12))
    check_done(eng, reqs)
    assert eng.runner.calls["verify"] > 0 and eng.spec_verifies > 0
    assert all(r.spec_proposed % K == 0 for r in reqs) and eng.spec_accepted <= eng.spec_proposed


def test_acceptance_is_counted_per_verify_and_per_draft_position():
    eng = make_engine(accept=(0.0, 0.0, 1.0))  # every draft accepted
    reqs = run(eng, prompts(4, shared=False), max_new=(20, 30))
    check_done(eng, reqs)
    assert eng.spec_accepted == eng.spec_proposed > 0
    assert eng.spec_pos_reached[0] == eng.spec_pos_accepted[0] and eng.spec_proposed == K * eng.spec_verifies
    assert all(r.spec_accepted == r.spec_proposed for r in reqs)
    assert eng.spec_pos_reached[0] >= eng.spec_pos_reached[-1]


def test_logprobs_of_a_blind_step_come_from_the_columns_the_synchronous_verify_reads():
    """Accepted positions take the draft's logprob (column 5), then the replacement's (column 4) when a draft was rejected,
    or the bonus token's (column 3) after a fully accepted run. The fake marks each column with 100 j + c."""
    for accept, col in (
        ((1.0, 0.0, 0.0), 4),
        ((0.0, 0.0, 1.0), 3),
    ):  # no draft accepted: the replacement; all: the bonus
        eng = make_engine(accept=accept)
        (r,) = run(eng, [[10, 11, 12, 13, 14]], max_new=(12, 13))
        assert len(r.logprobs) == len(r.output_ids)
        got = [lp for lp, _, _ in r.logprobs[1:]]  # the first token comes from the prefill (a zero column)
        assert set(got) <= {100.0 * j + c for j in range(Q) for c in (3, 4, 5)}
        last = [g for g in got if g % 100 == col]
        assert last, "the step's final token takes its own column"
        assert all(g % 100 == col or g % 100 == 5 for g in got), (accept, got)


def test_a_pool_too_short_for_all_requests_preempts_and_still_generates_the_reference():
    eng = make_engine(num_pages=40, seats=4, ckpts=6, max_len=120)
    reqs = run(eng, prompts(10, seed=3), max_new=(10, 50))
    check_done(eng, reqs)
    assert eng.scheduler.num_preemptions > 0, "the pool was meant to be short enough to preempt"


def test_decode_checkpoints_reserved_before_the_accepted_count_is_known_are_resolved_or_returned():
    eng = make_engine(ckpts=8)
    reqs = run(eng, prompts(6, seed=4), max_new=(30, 60))
    check_done(eng, reqs)
    assert eng.radix.num_ckpts > 0
    # a second turn on a finished request's tokens resumes from a state checkpoint the device wrote under a blind step:
    # the fake checks the restored row holds exactly the tokens the chunk resumes at
    again = [eng.add_request(r.token_ids[:], SamplingParams(max_new_tokens=6)) for r in reqs[:3]]
    while eng.has_work():
        eng.step()
    for a in again:
        assert a.status is Status.FINISHED and a.output_ids == reference_output(a.prompt_ids, 6)
    # past its own prompt: only a decode checkpoint written under a blind step can serve that
    assert any(a.num_cached_tokens > r.num_prompt for a, r in zip(again, reqs[:3], strict=True))


@pytest.mark.parametrize("hysteresis", [0, 1, 2])
def test_dynamic_speculation_flips_between_the_two_modes_without_changing_a_token(hysteresis):
    eng = make_engine(seats=8, ckpts=10, spec_seats=3, spec_max_batch=3, max_len=200)
    eng.cfg.spec_hysteresis = hysteresis
    rng = random.Random(5)
    reqs = []
    steps = 0
    pl = prompts(14, seed=6)
    for p in pl:  # arrivals spread over the run: the batch grows and shrinks across the limit
        reqs.append(eng.add_request(p, SamplingParams(max_new_tokens=rng.randrange(6, 45))))
        for _ in range(rng.randrange(0, 6)):
            eng.step()
            steps += 1
    while eng.has_work():
        eng.step()
        steps += 1
        assert steps < 8000
    check_done(eng, reqs)
    assert eng.dyn_flips >= 2, "the batch never crossed the limit"
    assert eng.dyn_plain_steps > 0 and eng.dyn_spec_steps > 0
    assert eng.runner.calls["decode"] > 0 and eng.runner.calls["verify"] > 0


def test_a_flip_moves_requests_out_of_plain_seats_and_copies_their_state():
    eng = make_engine(seats=8, ckpts=10, spec_seats=3, spec_max_batch=3, max_len=200)
    reqs = [eng.add_request(p, SamplingParams(max_new_tokens=60)) for p in prompts(8, seed=7, shared=False)]
    for _ in range(3000):
        eng.step()
        if eng.dyn_seat_moves:
            break
    assert eng.dyn_seat_moves > 0, "no request was ever moved into a speculative seat"
    while eng.has_work():
        eng.step()
    check_done(eng, reqs)
    assert eng.runner.copies >= eng.dyn_seat_moves


def test_the_fake_device_refuses_a_flip_that_forgets_the_accepted_row():
    """The fake's own state check can fail: a plain -> speculative or speculative -> plain flip that leaves the request's
    current row at row 0 instead of the accepted one hands the next call a state of the wrong length. Without this the
    composition checks above could be passing vacuously."""
    eng = make_engine(spec_seats=2, spec_max_batch=1, seats=4)
    real = eng.runner.state.set_current
    eng.runner.state.set_current = lambda req, i: real(req, 0)
    for p in prompts(6, seed=9, shared=False):
        eng.add_request(p, SamplingParams(max_new_tokens=40))

    def drive():
        for _ in range(3000):
            eng.step()
            if not eng.has_work():
                break

    with pytest.raises(AssertionError, match="reads a state"):
        drive()


@pytest.mark.parametrize("old", range(13))
@pytest.mark.parametrize("acc", range(K + 1))
def test_a_reserved_decode_checkpoint_is_kept_at_the_page_boundary_the_step_crossed_or_returned(old, acc):
    """The step's verify computed positions old .. old + acc; the state after c = floor((old + acc + 1) / page) * page tokens
    was copied into the reserved row when c > old: that row becomes the request's decode checkpoint at c, else it goes back."""
    eng = make_engine()
    st = eng.runner.state
    req = eng.add_request(list(range(10, 10 + old + 1)), SamplingParams(max_new_tokens=5))
    row = st.alloc_ckpt()
    req.ckpt_pending.append((E.ASYNC_CKPT + row, row, True))
    free = len(st._ckpt_free)
    s = types.SimpleNamespace(req=req, ckpt_row=row)
    eng._resolve_async_ckpt(s, acc)
    c = (old + acc + 1) // PS * PS
    if c > old:
        assert req.ckpt_pending == [(c, row, True)] and len(st._ckpt_free) == free
    else:
        assert req.ckpt_pending == [] and len(st._ckpt_free) == free + 1 and row in st._ckpt_free


def test_a_newer_decode_checkpoint_supersedes_the_requests_older_one():
    eng = make_engine()
    st = eng.runner.state
    req = eng.add_request(list(range(10, 20)), SamplingParams(max_new_tokens=5))
    old_row, row = st.alloc_ckpt(), st.alloc_ckpt()
    req.ckpt_pending += [(4, old_row, True), (E.ASYNC_CKPT + row, row, True)]
    eng._resolve_async_ckpt(types.SimpleNamespace(req=req, ckpt_row=row), 2)  # old 9: positions 9 .. 11, c = 12
    assert req.ckpt_pending == [(12, row, True)] and old_row in st._ckpt_free


def test_a_step_that_ended_its_request_first_leaves_the_reserved_row_to_the_release():
    eng = make_engine()
    req = eng.add_request(list(range(10, 20)), SamplingParams(max_new_tokens=5))
    s = types.SimpleNamespace(req=req, ckpt_row=7)  # the scheduler's _release already freed row 7 with the request
    eng._resolve_async_ckpt(s, 1)
    assert req.ckpt_pending == []


def _decoding(eng, n):
    """n running requests that decode."""
    reqs = []
    for i in range(n):
        r = eng.add_request([10 + i, 11, 12], SamplingParams(max_new_tokens=5))
        eng.scheduler.waiting.remove(r)
        r.status, r.num_computed = Status.RUNNING, r.num_tokens - 1
        eng.scheduler.running.append(r)
        reqs.append(r)
    return reqs


@pytest.mark.parametrize(
    ("hysteresis", "plain_now", "n", "want"),
    [
        (1, False, 3, False),  # at the limit: speculate
        (1, False, 4, True),  # past it: plain
        (1, True, 3, True),  # plain stays plain until at most limit - h decode
        (1, True, 2, False),
        (0, True, 3, False),  # no hysteresis: back at the limit
        (2, True, 2, True),
        (2, True, 1, False),
    ],
)
def test_dynamic_speculation_flips_at_the_limit_with_hysteresis(hysteresis, plain_now, n, want):
    eng = make_engine(seats=8, spec_max_batch=3)
    eng.cfg.spec_hysteresis = hysteresis
    eng._dyn_plain = plain_now
    _decoding(eng, n)
    assert eng._dyn_want_plain() is want


def test_a_request_in_a_plain_seat_forces_plain_mode_and_returns_only_when_they_all_fit_speculative_seats():
    eng = make_engine(seats=8, spec_seats=3, spec_max_batch=3)
    st = eng.runner.state
    st.prefer_spec = False  # plain mode's admission: plain seats first
    reqs = _decoding(eng, 2)
    for r in reqs:
        st.rows_of(r)
    assert not any(st.is_spec(r) for r in reqs)
    eng._dyn_plain = False
    assert eng._dyn_want_plain() is True, (
        "a verify needs a speculative seat: a request in a plain seat is stepped plain"
    )
    eng._dyn_plain = True
    assert eng._dyn_want_plain() is False, "two requests fit the three free speculative seats: the flip moves them"
    _decoding(eng, 2)  # two more decoding requests, none holding a seat yet: five running, three speculative seats
    assert eng._dyn_want_plain() is True


def test_a_prefix_restored_from_the_host_tier_then_speculated_equals_a_cold_request():
    """A finished request's pages and checkpoints are evicted from the device into the host tier, and the same prompt
    comes back: its prefix is restored (pages loaded, the deepest checkpoint's state copied into its row), the rest
    prefilled, and the blind steps that follow generate what a cold request does. The fake checks that the restored row
    holds exactly the tokens the chunk after it resumes at."""
    eng = make_engine(host=True, num_pages=96, ckpts=10)
    prompt = prompts(2, seed=11)[0][:30] + list(range(200, 230))
    cold = eng.add_request(prompt, SamplingParams(max_new_tokens=24))
    while eng.has_work():
        eng.step()
    assert cold.output_ids == reference_output(prompt, 24)
    eng.radix.evict(eng.radix.total_pages())  # everything the tree holds goes to the host tier
    assert eng.radix.total_pages() == 0 and eng.host_tier.entries and eng.host_tier.ckpts
    again = eng.add_request(prompt, SamplingParams(max_new_tokens=24))
    while eng.has_work():
        eng.step()
    assert again.num_cached_tokens > 0 and eng.runner.calls["kv_load"] > 0 and eng.runner.calls["state_load"] > 0
    assert again.output_ids == cold.output_ids == reference_output(prompt, 24)
    assert again.spec_proposed > 0, "the restored request speculated"
