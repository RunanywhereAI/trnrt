"""Continuous-batching scheduler over a paged KV pool with a radix prefix cache.

Every step runs all decoding sequences (one query token each) and, within a token budget,
chunks of prefill. Decodes never pause for a prefill, which bounds inter-token latency by
the chunk size; the managed Neuron scheduler instead runs one prefill per step and pauses
decodes during it (vllm-neuron docs, "neuron-scheduler").

Admission reserves KV for what the running requests will need (SchedulerConfig.admission_decode_tokens),
so a KV-short pool runs fewer requests instead of admitting prompts it must later preempt. When the
pool still runs dry the youngest running request is preempted; its computed prefix goes into the
radix cache, so the recompute resumes from its last state checkpoint.

State checkpoints. GLM-5.3-Flash's linear-attention (KDA) layers resume a prefix only from a state
checkpoint (engine/state_pool.py, RadixNode.ckpt), so a prefill leaves one at:
- the radix junction past the last checkpoint of a request's matched path (another request shares
  those tokens: SGLang v0.5.21 unified_cache/components/mamba.py mamba_branching_seqlen, vLLM v0.30.0
  Scheduler._mamba_block_aligned_split's shared-prefix junction);
- the prompt's last page boundary (vLLM's "replay boundary"): the prompt-end checkpoint, from which a
  session's next turn resumes when its replay of this turn's reply differs from the cached tokens;
and decode leaves one at every page boundary it crosses (SGLang mamba_track_interval at the page size),
the newest superseding the older, so a finished turn leaves the state after its last full page for the
next turn to resume from.

A checkpoint target strictly inside a prefill chunk is written by the chunk's own graph
(ScheduledSeq.mid: the linear-attention chunk scan's state_slot [row, ckpt_row, k]) instead of
splitting the chunk there, which would cost one more prefill call per request (~0.45 s on the
1024-row bucket). So a step holds at most one entry per request.

Junctions are also taken AHEAD: a junction exists only once a second request has matched the first
one's KV, so that second request would recompute the whole shared prefix just to leave the
checkpoint, and only a third one would hit. When the requests sharing a prefix are already queued or
running together (a burst sharing a system prompt, a closed-loop benchmark), the junction is known
before either computes it: the longest page-aligned prefix a request shares with another waiting or
running request becomes a checkpoint target of the one computing it first (at its admission, or, when
the other arrives while it is still prefilling, added to its targets then), and the other waits until
it can resume from it.
"""

from __future__ import annotations

import bisect
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .request import Request, Status

if TYPE_CHECKING:
    from .host_cache import HostTier
    from .kv_pool import PagePool
    from .radix_cache import RadixCache
    from .state_pool import StatePool

PENDING = -1  # placeholder for a token sampled by a step whose results are not read yet
# A blind speculative draft (engine/spec_async.py): the drafts are on the device, not known to the host.
BLIND = -2


class NeedSync(Exception):
    """Scheduling needs a preemption while a step is in flight. A preempted request must be
    recomputed from REAL tokens, so the caller commits the in-flight step and retries."""


@dataclass
class SchedulerConfig:
    page_size: int
    max_num_seqs: int
    max_prefill_tokens: int  # prefill tokens per step, across all chunks
    max_model_len: int
    eos_token_ids: tuple[int, ...] = ()
    # Admission: a waiting request is admitted only if the free plus evictable pages cover its whole
    # prompt and up to admission_decode_tokens of its decode, on top of what the running requests still
    # need: the rest of their prompts plus up to admission_decode_tokens of decode each (_still_needs).
    # A KV-short pool then runs fewer requests instead of admitting prompts it must later preempt, and
    # every preemption recomputes from the request's last state checkpoint. (vLLM's v1 scheduler admits
    # whenever the first chunk fits.)
    admission_decode_tokens: int = 256
    # EngineConfig.max_running: running requests at most (None: max_num_seqs). Admission only, never the graphs' seats.
    max_running: int | None = None


def shared_pages(a: list[int], b: list[int], ps: int, limit: int) -> int:
    """Whole pages a and b share from position 0, at most `limit`."""
    n = min(limit, len(a) // ps, len(b) // ps)
    if n <= 0 or a[:ps] != b[:ps]:
        return 0
    if a[: n * ps] == b[: n * ps]:
        return n
    lo, hi = 1, n  # a[:lo pages] equal, a[:hi pages] not
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[lo * ps : mid * ps] == b[lo * ps : mid * ps]:
            lo = mid
        else:
            hi = mid
    return lo


def page_chain(req: Request, ps: int) -> list[int]:
    """req's prompt-so-far as chained page hashes, h[k] = hash((h[k-1], page k)) (the host cache's own formula), over its
    leading full pages that hold no in-flight placeholder (-1: replaced in place once its step lands, so never cached).
    Built once and extended as tokens append (the request's token list only grows; a new list object or a shorter one
    rebuilds it), so shared_pages_req compares two prefixes by a binary search over two int lists instead of copying
    slices: at ~150K tokens a slice compare costs ~1 ms, and an admission makes one per running request."""
    tok = req.token_ids
    st = req.__dict__.get("_page_chain")
    if st is None or st[0] is not tok or st[1] > len(tok) or st[3] != ps:
        st = [tok, 0, [], ps, False]  # token list, tokens hashed, chain, page size, stopped at a placeholder page
        req.__dict__["_page_chain"] = st
    if not st[4]:
        chain, n, h = st[2], st[1], st[2][-1] if st[2] else 0
        while n + ps <= len(tok):
            page = tuple(tok[n : n + ps])
            if -1 in page:  # a placeholder: the chain stops here until the next rebuild
                st[4] = True
                break
            h = hash((h, page))
            chain.append(h)
            n += ps
        st[1] = n
    return st[2]


def shared_pages_req(a: Request, b: Request, ps: int, limit: int) -> int:
    """shared_pages(a.token_ids, b.token_ids, ps, limit) from the requests' page chains: equal entries at page k mean
    equal pages 0..k (up to a 64-bit hash collision), and prefix equality is monotone in k, so the longest shared prefix
    within both chains is a binary search. Past either chain's end (a placeholder page) it compares the rest by slices,
    as shared_pages does."""
    ta, tb = a.token_ids, b.token_ids
    n = min(limit, len(ta) // ps, len(tb) // ps)
    if n <= 0:
        return 0
    ca, cb = page_chain(a, ps), page_chain(b, ps)
    m = min(n, len(ca), len(cb))
    if m == 0:
        return shared_pages(ta, tb, ps, n)
    if ca[0] != cb[0]:
        return 0
    if ca[m - 1] != cb[m - 1]:
        lo, hi = 1, m  # pages [0, lo) shared, [0, hi) not
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ca[mid - 1] == cb[mid - 1]:
                lo = mid
            else:
                hi = mid
        return lo
    if m == n:
        return n
    k = m  # the first m pages are shared; the rest by slices
    while k < n and ta[k * ps : (k + 1) * ps] == tb[k * ps : (k + 1) * ps]:
        k += 1
    return k


@dataclass
class ScheduledSeq:
    req: Request
    start: int  # first position computed this step
    num_tokens: int
    sample: bool  # this chunk reaches the newest token, so it produces the next one
    # Speculative draft slots verified after the newest token (decode only), [BLIND] * k: the drafts themselves are on the
    # device. The step then covers positions start .. start + len(draft) and may emit up to len(draft) + 1 tokens.
    draft: list[int] = field(default_factory=list)
    # Recurrent-state checkpoint rows (engine/state_pool.py): copied into the request's state row
    # before this entry runs (a prefix hit), and receiving a copy of it after (the state at end).
    restore: int | None = None
    save: int | None = None
    # (k, row): the state after this entry's first k tokens goes to checkpoint row `row`, written by the
    # entry's own graph, not copied after it.
    mid: tuple[int, int] | None = None
    # A blind speculative step's decode-checkpoint row, reserved by the engine (LLMEngine._reserve_async_ckpts) for the
    # state after the page boundary the step may cross.
    ckpt_row: int | None = None

    @property
    def end(self) -> int:
        return self.start + self.num_tokens


@dataclass
class StepPlan:
    decodes: list[ScheduledSeq] = field(default_factory=list)
    prefills: list[ScheduledSeq] = field(default_factory=list)

    def seqs(self) -> list[ScheduledSeq]:
        return self.decodes + self.prefills

    def __bool__(self) -> bool:
        return bool(self.decodes or self.prefills)


class Scheduler:
    MAX_AHEAD = 4  # junction checkpoints one admission may add ahead (each costs a row and a copy)

    def __init__(self, cfg: SchedulerConfig, pool: PagePool, radix: RadixCache, draft_fn=None):
        self.cfg = cfg
        # draft_fn(req) -> [BLIND] * k for a decoding request that speculates in this step, else [] (speculative decoding,
        # engine/spec_async.py).
        self.draft_fn = draft_fn
        self.pool = pool
        self.radix = radix
        # Set by the engine before the first schedule(): the state pool whose checkpoint rows back the
        # radix nodes (engine/state_pool.StatePool: alloc_ckpt / free_ckpt), and the host cache under the
        # radix cache (engine/host_cache.HostTier), when one is configured.
        self._states: StatePool | None = None
        self.host: HostTier | None = None
        # Set by an overlapping engine while a launched step's results are unread.
        self.in_flight = False
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []  # admission order, oldest first
        self.num_preemptions = 0
        # In-chunk checkpoints placed, and skipped for want of a free row (the chunk is not split instead).
        self.ckpt_mid_placed = 0
        self.ckpt_mid_skipped = 0
        self._new_pending: list = []  # (request, pending checkpoint) the current plan created

    @property
    def states(self) -> StatePool:
        if self._states is None:
            raise RuntimeError("the scheduler has no state pool: set Scheduler.states before the first schedule()")
        return self._states

    @states.setter
    def states(self, pool: StatePool) -> None:
        self._states = pool

    # -- public API ---------------------------------------------------------------

    def close_session(self, sid: str) -> bool:
        return self.radix.session_close(sid)

    def add(self, req: Request) -> None:
        if req.num_tokens >= self.cfg.max_model_len:
            raise ValueError(f"prompt of {req.num_tokens} tokens does not fit max_model_len={self.cfg.max_model_len}")
        if req.session_id is not None:
            req.session_gen = self.radix.session_open(req.session_id)
        self._mark_running_junctions(req)
        self.waiting.append(req)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> StepPlan:
        """The next step's work. A NeedSync aborts the plan (rollback)."""
        self._new_pending = []
        try:
            return self._schedule()
        except NeedSync:
            self.rollback()
            raise

    def rollback(self) -> None:
        """Undo the checkpoint bookkeeping of the last plan, which will not run: the rows taken for its
        checkpoints go back and a prefill's checkpoint target is wanted again. A request admitted from a
        checkpoint keeps req.ckpt_restore, so the retry's first chunk restores it."""
        for req, item in self._new_pending:
            if item in req.ckpt_pending:
                req.ckpt_pending.remove(item)
                self.states.free_ckpt(item[1])
                if not item[2]:
                    bisect.insort(req.ckpt_targets, item[0])
        self._new_pending = []

    def update(self, plan: StepPlan, tokens: list) -> list[Request]:
        """Apply one step's results synchronously: advance() then commit()."""
        self.advance(plan)
        return self.commit(plan, tokens)

    def advance(self, plan: StepPlan) -> None:
        """Move every scheduled request past this step BEFORE its results are known, so the
        next step can be scheduled while this one runs (overlap scheduling). A sampling
        entry gets a PENDING placeholder where its new token will go. A blind speculative entry moves nothing: its tokens
        and its progress stay on the device until its commit, and the request counts the step in flight."""
        for s in plan.seqs():
            if s.draft:
                s.req.spec_inflight += 1
                continue
            s.req.num_computed = s.end
            if s.sample:
                s.req.token_ids.append(PENDING)

    def commit(self, plan: StepPlan, tokens: list) -> list[Request]:
        """Fill in a step's results. `tokens[i]` is the sampled id for `plan.seqs()[i]` (ignored for
        an entry that does not sample), or for a speculative entry the list it emitted (accepted drafts, then one more).
        An entry whose request already finished in an earlier commit is ignored. Returns requests finished by this commit."""
        now = time.monotonic()
        finished = []
        for s, tok in zip(plan.seqs(), tokens):
            req = s.req
            if s.draft:
                req.spec_inflight -= 1
                if req.status is Status.RUNNING and self._commit_blind(req, tok, now) is not None:
                    finished.append(req)
                continue
            if req.status is not Status.RUNNING:
                continue  # finished or aborted while this step was in flight
            if not s.sample:
                self._cache(req)  # share a prefill chunk's full pages right away
                continue
            at = s.end  # index of the placeholder
            sampled = int(tok)
            req.token_ids[at] = sampled
            if req.first_token_time is None:
                req.first_token_time = now
            reason = self._finish_reason(req, sampled, length=at + 1)
            if reason is not None:
                # Drop placeholders of steps launched after this one, and count only KV
                # that belongs to tokens the request keeps.
                del req.token_ids[at + 1 :]
                req.num_computed = min(req.num_computed, at)
                req.finish_reason = reason
                req.finish_time = now
                finished.append(req)
                self._release(req, Status.FINISHED)
            elif s.num_tokens > 1:
                self._cache(req)
        return finished

    def _commit_blind(self, req: Request, tok: list, now: float) -> str | None:
        """A blind speculative entry's emitted tokens (accepted drafts, then the replacement or the bonus), in order:
        appended as a plain step's are, the newest one's position becoming num_computed (the KV of the newest token and
        of the accepted drafts is written; a rejected draft's lies beyond). Returns the finish reason."""
        reason = None
        for t in tok:
            req.token_ids.append(int(t))
            reason = self._finish_reason(req, int(t), length=len(req.token_ids))
            if reason is not None:
                break
        req.num_computed = len(req.token_ids) - 1
        if req.first_token_time is None:
            req.first_token_time = now
        if reason is not None:
            req.finish_reason = reason
            req.finish_time = now
            self._release(req, Status.FINISHED)
        return reason

    def finish_now(self, req: Request, reason: str) -> None:
        """Finish a running request outside update(), e.g. on a stop string."""
        self._drop_pending(req)
        req.finish_reason = reason
        req.finish_time = time.monotonic()
        self._release(req, Status.FINISHED)

    def abort(self, req: Request) -> None:
        if req.status is Status.WAITING:
            self.waiting.remove(req)
            req.status = Status.FINISHED
        elif req.status is Status.RUNNING:
            req.finish_reason = "abort"
            self._drop_pending(req)
            self._release(req, Status.FINISHED)
        req.finish_reason = "abort"

    def track(self, req: Request, start: int, n: int) -> int | None:
        """Decode checkpoint: when positions start .. start + n - 1 cross a page boundary b, a fresh
        row for the state after b tokens (the caller copies it there), registered as pending at b.
        It supersedes the request's older decode checkpoint, which is freed: only the newest serves
        the next turn. So a finished request leaves the state after prompt + generated[:-1] (the last
        sampled token is never fed), floored to a page, and the session's next turn, whose prompt
        replays this one's reply, resumes there and recomputes the tail under a page. Rows are never
        reused while pending, so a copy still in flight cannot overwrite a checkpoint the tree
        already holds."""
        ps = self.cfg.page_size
        b = (start + n) // ps * ps
        if b <= start:
            return None
        row = self._alloc_ckpt()
        if row is None:
            return None
        for item in [x for x in req.ckpt_pending if x[2]]:
            req.ckpt_pending.remove(item)
            self.states.free_ckpt(item[1])
        self._pend(req, (b, row, True))
        return row

    @staticmethod
    def shared_tokens(req: Request) -> int:
        """Leading positions of `req` whose KV pages belong to the radix tree: other requests
        may read them, so they are never rewritten in place."""
        n, node = 0, req.node
        while node is not None and node.parent is not None:
            n += len(node.key)
            node = node.parent
        return n

    # -- planning -----------------------------------------------------------------

    def _seats(self) -> int:
        """Requests that may run at once: the compiled seats (max_num_seqs), or fewer under max_running."""
        cap = self.cfg.max_running
        return self.cfg.max_num_seqs if cap is None else min(self.cfg.max_num_seqs, cap)

    def _schedule(self) -> StepPlan:
        plan = StepPlan()
        ps = self.cfg.page_size

        # 1. Every decoding sequence gets its one token, preempting the youngest if KV
        #    runs out. Iterating oldest-first while preempting youngest-first means a
        #    victim is never a sequence already scheduled this step.
        for req in list(self.running):
            if req.status is not Status.RUNNING or not req.is_decoding:
                continue
            draft = self.draft_fn(req) if self.draft_fn is not None else []
            # A blind speculative row (engine/spec_async.py): the steps still in flight may each have advanced the request by
            # up to 1 + k positions, so its newest token is at most this far on, and its pages cover the verify past that
            # bound. The request finishes by max_model_len, so pages past it are never needed (padding writes go to the null
            # page).
            start = req.num_computed
            end = start + 1
            if draft:
                start = min(start + (1 + len(draft)) * req.spec_inflight, self.cfg.max_model_len - 1)
                end = min(start + 1 + len(draft), self.cfg.max_model_len)
            while not self._reserve(req, end):
                if self.in_flight:
                    raise NeedSync()
                victim = self._youngest_running()
                self._preempt(victim)
                if victim is req:
                    break
            if req.status is Status.RUNNING:
                save = None if draft else self.track(req, req.num_computed, 1)
                plan.decodes.append(ScheduledSeq(req, start, 1 + len(draft), True, draft, save=save))

        # 2. Continue partial prefills, oldest first.
        budget = self.cfg.max_prefill_tokens
        for req in self.running:
            if budget <= 0:
                break
            if req.is_decoding:
                continue
            n = min(req.remaining_prefill, budget)
            if not self._reserve(req, req.num_computed + n):
                break
            plan.prefills.append(self._chunk(req, req.num_computed, n))
            budget -= n

        # 3. Admit waiting requests, first come first served, while budget and capacity remain.
        owed = sum(self._still_needs(r) for r in self.running)
        deferred = []  # waiting for a running request to checkpoint the prefix they share
        while self.waiting and budget > 0 and len(self.running) < self._seats():
            req = self.waiting[0]
            if self._prefix_in_progress(req):
                deferred.append(self.waiting.popleft())
                continue
            # Leave at least one token to compute: the step must produce logits.
            limit = (req.num_tokens - 1) // ps
            match, kv_pages = self.radix.match_checkpoint(req.token_ids, max_pages=limit)
            if self.host is not None:
                match, kv_pages = self._restore(req, match, kv_pages, limit)
            cached = match.num_pages * ps
            n = min(req.num_tokens - cached, budget)
            need = -(-(cached + n) // ps) - match.num_pages
            self.radix.lock(match.node)  # so eviction below cannot take this prefix
            # The reservation: what it may still need on top of what the running requests may. A request
            # that would not fit even alone is still admitted when nothing else runs.
            if self.running:
                full = -(-self._target_tokens(req) // ps) - match.num_pages
                if self.pool.num_free + self.radix.evictable_pages < full + owed:
                    self.radix.unlock(match.node)
                    break
            if self.pool.num_free < need:
                self.radix.evict(need - self.pool.num_free)
            new_pages = self.pool.alloc(need)
            if new_pages is None:
                self.radix.unlock(match.node)
                break
            self.waiting.popleft()
            req.pages = list(match.pages) + new_pages
            req.node = match.node
            req.num_computed = cached
            if req.num_cached_tokens < 0:
                req.num_cached_tokens = cached
            req.status = Status.RUNNING
            self.running.append(req)
            # None at the root: the state starts from zero at position 0. Kept on the request
            # until the engine launches the copy (a plan given up with NeedSync is rescheduled).
            req.ckpt_restore = match.node.ckpt
            req.ckpt_targets = self._ckpt_targets(req, cached, kv_pages * ps, self._junctions_ahead(req, cached))
            plan.prefills.append(self._chunk(req, cached, n))
            budget -= n
            owed += self._still_needs(req)
        self.waiting.extendleft(reversed(deferred))
        return plan

    def _target_tokens(self, req: Request) -> int:
        """The tokens a request is reserved pages for: its known tokens (the prompt, and what it has
        generated) plus up to admission_decode_tokens of what it may still generate."""
        left = max(0, req.params.max_new_tokens - len(req.output_ids))
        return min(self.cfg.max_model_len, req.num_tokens + min(left, self.cfg.admission_decode_tokens))

    def _still_needs(self, req: Request) -> int:
        """Pages a running request will still allocate up to its _target_tokens."""
        return max(0, -(-self._target_tokens(req) // self.cfg.page_size) - len(req.pages))

    def _chunk(self, req: Request, start: int, n: int) -> ScheduledSeq:
        """One step's prefill of positions start .. start + n. A checkpoint target at its end becomes
        `save`, a copy of the state row after the entry; the deepest target strictly inside it becomes
        `mid`, written by the entry's own graph (earlier ones inside are dropped). Each takes a fresh row,
        pending until its pages are in the tree (_cache); without a free row the target is dropped. The
        entry restores the request's checkpoint if it was admitted from one and that copy has not run."""
        end = start + n
        mid = None
        inside = [t for t in req.ckpt_targets if start < t < end]
        if inside:
            row = self._alloc_ckpt()
            if row is None:
                self.ckpt_mid_skipped += 1
            else:
                mid = (inside[-1] - start, row)
                self._pend(req, (inside[-1], row, False))
                self.ckpt_mid_placed += 1
        save = None
        if end in req.ckpt_targets:
            save = self._alloc_ckpt()
            if save is not None:
                self._pend(req, (end, save, False))
        req.ckpt_targets = [t for t in req.ckpt_targets if t > end]
        return ScheduledSeq(req, start, n, end == req.num_tokens, restore=req.ckpt_restore, save=save, mid=mid)

    def _reserve(self, req: Request, total_tokens: int) -> bool:
        need = -(-total_tokens // self.cfg.page_size) - len(req.pages)
        if need <= 0:
            return True
        if self.pool.num_free < need:
            self.radix.evict(need - self.pool.num_free)
        pages = self.pool.alloc(need)
        if pages is None:
            return False
        req.pages.extend(pages)
        return True

    def _youngest_running(self) -> Request:
        """Preemption victim: the youngest running request."""
        return max(self.running, key=lambda r: r.arrival_time)

    # -- recurrent-state checkpoints ------------------------------------------------------

    @staticmethod
    def _prompt_pages(req: Request, ps: int) -> int:
        """Pages of the prompt a checkpoint may end on: those before the newest token."""
        return (req.num_tokens - 1) // ps

    def _shared_with(self, req: Request, other: Request) -> int:
        """Page-aligned prompt tokens req and other share (checkpointable: before either's newest)."""
        ps = self.cfg.page_size
        return shared_pages_req(req, other, ps, min(self._prompt_pages(req, ps), self._prompt_pages(other, ps))) * ps

    def _mark_running_junctions(self, req: Request) -> None:
        """A request arriving while another is still prefilling a prefix they share: make the end of the
        shared pages a checkpoint target of the one computing it, if it has not passed it yet, so `req`
        resumes from it."""
        for other in self.running:
            if other.is_decoding or other.status is not Status.RUNNING:
                continue
            j = self._shared_with(req, other)
            if j > other.num_computed and j not in other.ckpt_targets and all(p != j for p, _, _ in other.ckpt_pending):
                bisect.insort(other.ckpt_targets, j)

    def _prefix_in_progress(self, req: Request) -> bool:
        """Another running request is computing a prefix `req` shares with it and will checkpoint a
        position of it past what `req` could resume from now (a junction target not yet reached, or a
        saved checkpoint not yet in the tree): admitting `req` now would compute those tokens a second
        time, so it waits until it can resume from them."""
        best = None
        for other in self.running:
            if other.status is not Status.RUNNING:
                continue
            soon = [p for p, _, _ in other.ckpt_pending]
            if not other.is_decoding:
                soon += [p for p in other.ckpt_targets if p > other.num_computed]
            if not soon:
                continue
            j = self._shared_with(req, other)
            soon = [p for p in soon if 0 < p <= j]
            if not soon:
                continue
            if best is None:  # what req resumes from now (its checkpoint match)
                ps = self.cfg.page_size
                best = (
                    self.radix.match_checkpoint(req.token_ids, max_pages=self._prompt_pages(req, ps))[0].num_pages * ps
                )
            if max(soon) > best:
                return True
        return False

    def _junctions_ahead(self, req: Request, cached: int) -> list[int]:
        """The page-aligned prefixes `req` shares with other waiting or running requests, past what it
        resumes from: distinct lengths, the shortest (most widely shared) MAX_AHEAD."""
        out = set()
        for other in list(self.waiting) + self.running:
            if other is not req:
                j = self._shared_with(req, other)
                if j > cached:
                    out.add(j)
        return sorted(out)[: self.MAX_AHEAD]

    def _ckpt_targets(self, req: Request, cached: int, kv_matched: int, ahead=()) -> list[int]:
        """Positions past `cached` where this prefill should leave a checkpoint (page multiples
        below the newest token): the radix junction (KV other requests share runs past the last
        checkpoint), the junctions with requests still queued or running (`ahead`), and the
        prompt's last page boundary."""
        ps = self.cfg.page_size
        last = (req.num_tokens - 1) // ps * ps
        out = {last}
        if kv_matched > cached:
            out.add(kv_matched)
        out.update(ahead)
        return sorted(p for p in out if cached < p <= last)

    def _alloc_ckpt(self) -> int | None:
        """A free checkpoint row, evicting a checkpoint of an unlocked node if none is free, then, with
        the host cache, one on a running request's path (radix_cache.py evict_checkpoints `keep`)."""
        row = self.states.alloc_ckpt()
        if row is None and self.radix.evict_checkpoints(1):
            row = self.states.alloc_ckpt()
        if row is None and self.radix.offload_ckpt is not None:
            keep = {r.ckpt_restore for r in self.running if r.ckpt_restore is not None}
            if self.radix.evict_checkpoints(1, keep=keep):
                row = self.states.alloc_ckpt()
        return row

    def _pend(self, req: Request, item: tuple[int, int, bool]) -> None:
        """A checkpoint row this plan writes, pending until its pages are in the tree (_cache)."""
        req.ckpt_pending.append(item)
        self._new_pending.append((req, item))

    def _restore(self, req: Request, match, kv_pages: int, limit: int):
        """Extend a device match with what the host cache holds. The model resumes only from a
        checkpoint, so this restores the pages up to the deepest host checkpoint past the device one
        (none: nothing): copied into fresh device pages and inserted into the tree, then that
        checkpoint into a fresh row of the state pool. Returns the new match."""
        host = self.host
        if host is None:
            raise RuntimeError("_restore needs the host cache")
        ids, ps = req.token_ids, self.cfg.page_size
        full = self.radix.match_prefix(ids, max_pages=kv_pages)
        chain = page_chain(req, ps) if getattr(host, "ps", None) == ps else None  # the host cache's own hashes
        slots = host.lookup(ids, kv_pages, limit, chain=chain)
        ck = host.find_ckpt(ids, match.num_pages + 1, kv_pages + len(slots), chain=chain)
        keep = max(0, ck[0] - kv_pages) if ck is not None else 0
        host.unpin(slots[keep:])
        slots = slots[:keep]
        if ck is None:
            return match, kv_pages
        self.radix.lock(full.node)
        try:
            if slots:
                if self.pool.num_free < len(slots):
                    self.radix.evict(len(slots) - self.pool.num_free)
                pages = self.pool.alloc(len(slots))
                if pages is None:
                    host.unpin(slots)
                    return match, kv_pages
                host.load(slots, pages)
                n = kv_pages + len(slots)
                res = self.radix.insert(ids[: n * ps], list(full.pages) + pages)
                for mine, tree in zip(pages, res.pages[kv_pages:]):
                    if mine != tree:  # someone inserted the same pages meanwhile: keep the tree's
                        self.pool.free([mine])
            row = self._alloc_ckpt()
            if row is not None:
                host.load_ckpt(ck[1], row)
                if not self.radix.attach_checkpoint(ids, ck[0], row):
                    self.states.free_ckpt(row)
        finally:
            self.radix.unlock(full.node)
        return self.radix.match_checkpoint(ids, max_pages=limit)

    # -- release ------------------------------------------------------------------

    def _finish_reason(self, req: Request, tok: int, length: int) -> str | None:
        """`length`: the request's token count up to and including `tok` (later
        placeholders of in-flight steps do not count)."""
        p = req.params
        if not p.ignore_eos and tok in self.cfg.eos_token_ids:
            return "stop"
        if tok in p.stop_token_ids:
            return "stop"
        if length - req.num_prompt >= p.max_new_tokens:
            return "length"
        if length >= self.cfg.max_model_len:
            return "length"
        return None

    def _cacheable(self, req: Request) -> int:
        """Leading positions of `req` the radix tree may take (_cache) and _release keeps out of the freed tail:
        those computed (num_computed, which an overlapped step has already advanced) whose tokens are all known
        (placeholders are the newest positions: advance appends them, commit fills the oldest first), so a
        placeholder never enters a radix key; and never fewer than the tree already holds for this request (its
        lock's path): an abort while a step is in flight (_drop_pending) cuts num_computed below the pages the last
        prefill chunk's _cache gave the tree, and _release must not free one of those as its tail."""
        n = len(req.token_ids)
        while n and req.token_ids[n - 1] == PENDING:
            n -= 1
        return max(min(req.num_computed, n), self.shared_tokens(req))

    def _drop_pending(self, req: Request) -> None:
        if PENDING in req.token_ids:
            del req.token_ids[req.token_ids.index(PENDING) :]
            req.num_computed = min(req.num_computed, len(req.token_ids) - 1)

    def _cache(self, req: Request) -> None:
        """Insert the request's full computed pages into the radix tree and move its
        lock to the new deepest node. Duplicate pages (the same prefix computed by
        another request first) are freed in favour of the tree's. Pending checkpoints whose
        pages are now in the tree join their node."""
        ps = self.cfg.page_size
        n_full = self._cacheable(req) // ps
        if n_full == 0:
            return
        res = self.radix.insert(req.token_ids[: n_full * ps], req.pages[:n_full])
        for i, page in enumerate(res.pages):
            if req.pages[i] != page:
                self.pool.free([req.pages[i]])
                req.pages[i] = page
        self.radix.lock(res.node)
        if req.node is not None:
            self.radix.unlock(req.node)
        req.node = res.node
        if req.ckpt_pending:
            keep = []
            prompt_end = (req.num_prompt - 1) // ps * ps
            for item in req.ckpt_pending:
                pos, row, decode = item
                if pos > n_full * ps:
                    keep.append(item)
                elif not self.radix.attach_checkpoint(
                    req.token_ids, pos // ps, row, anchor=not decode and pos == prompt_end
                ):
                    self.states.free_ckpt(row)  # another request's checkpoint got there first
            req.ckpt_pending = keep

    def _release(self, req: Request, status: Status) -> None:
        self._cache(req)
        for _, row, _ in req.ckpt_pending:  # past the KV the request keeps: never valid
            self.states.free_ckpt(row)
        req.ckpt_pending, req.ckpt_targets, req.ckpt_restore = [], [], None
        if (
            status is Status.FINISHED
            and req.session_id is not None
            and req.node is not None
            and req.finish_reason != "abort"
        ):
            self.radix.session_register(req.session_id, req.session_gen, req.node)
        tail = req.pages[self._cacheable(req) // self.cfg.page_size :]  # what _cache did not take
        if tail:
            self.pool.free(tail)
        if req.node is not None:
            self.radix.unlock(req.node)
            self.radix.free_dominated(req.node)
        req.pages = []
        req.node = None
        req.status = status
        self.running.remove(req)

    def _preempt(self, req: Request) -> None:
        self._release(req, Status.WAITING)
        req.num_computed = 0
        req.spec_ready = False  # its recompute rebuilds the speculative board row (engine/spec_async.py)
        req.spec_cur = (
            None  # its recompute writes row 0; a flip to plain before its next verify must not pick an old row
        )
        req.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(req)
