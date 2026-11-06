"""A request's sampling parameters and its state as the scheduler moves it through the engine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .radix_cache import RadixNode


@dataclass(frozen=True)
class SamplingParams:
    max_new_tokens: int = 128
    temperature: float = 0.0  # 0 means greedy
    top_p: float = 1.0
    top_k: int = 0  # 0 means disabled
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logprobs: int | None = None  # None: off; n >= 0: chosen token plus top-n (n <= 20)
    stop: tuple[str, ...] = ()  # stop strings, matched on detokenized output
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False
    seed: int | None = None
    # vLLM's thinking_token_budget. Not served: unsupported() refuses any value but None.
    thinking_token_budget: int | None = None


def unsupported(params: SamplingParams) -> str | None:
    """What in a request's sampling the engine does not serve, or None. Every graph samples on the device from the
    logits alone; penalties and a thinking budget's forced tokens need graph variants with host inputs, which the
    engine neither compiles nor warms. add_request refuses them before anything is queued (the API answers 400)."""
    # Only values that change the output are refused: clients send the neutral ones (penalties of 1 / 0 / 0, no
    # budget) by default.
    if params.repetition_penalty != 1.0 or params.frequency_penalty != 0.0 or params.presence_penalty != 0.0:
        return "repetition / frequency / presence penalties"
    if params.thinking_token_budget is not None:
        return "a thinking token budget"
    return None


class Status(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(eq=False)
class Request:
    rid: str
    prompt_ids: list[int]
    params: SamplingParams
    arrival_time: float = field(default_factory=time.monotonic)
    # SGLang's top-level session_id: a finished request's prefix stays referenced by its
    # session (soft eviction protection) until the session is closed.
    session_id: str | None = None
    session_gen: int = -1

    # Prompt followed by generated tokens. After a preemption the whole list is the
    # "prompt" of the recompute, so it is kept as one list.
    token_ids: list[int] = field(init=False)
    status: Status = Status.WAITING
    # Pages holding KV for positions [0, num_computed), in position order.
    pages: list[int] = field(default_factory=list)
    num_computed: int = 0
    # Deepest radix node this request has locked, if any.
    node: RadixNode | None = None
    # Tokens of the prompt served from the prefix cache at first admission.
    num_cached_tokens: int = -1
    num_preemptions: int = 0
    finish_reason: str | None = None
    # Per output token, when params.logprobs is set: (logprob, top ids, top logprobs).
    logprobs: list[tuple[float, list[int], list[float]]] = field(default_factory=list)
    first_token_time: float | None = None
    finish_time: float | None = None
    # Thinking budget state: inside the reasoning span, its token count, output tokens
    # already scanned, tokens still to force, and whether the span has ended.
    think_in: bool = False
    think_count: int = 0
    think_seen: int = 0
    think_force: list[int] = field(default_factory=list)
    think_done: bool = False
    # Recurrent-state checkpoints (engine/scheduler.py): prefill positions still to save at, and
    # saved rows waiting for their pages to enter the radix tree, as (position, row, from decode).
    ckpt_targets: list[int] = field(default_factory=list)
    ckpt_pending: list[tuple[int, int, bool]] = field(default_factory=list)
    ckpt_restore: int | None = None  # the checkpoint row a prefix hit restores, until the copy is launched
    # Speculative decoding (engine/spec_async.py, engine/engine.py).
    spec_proposed: int = 0  # draft tokens proposed for this request, and how many verification accepted
    spec_accepted: int = 0
    mtp_draft: list[int] = field(default_factory=list)  # MTP drafts for the next step (the synchronous path)
    spec_inflight: int = 0  # blind speculative steps launched and not yet committed
    spec_ready: bool = False  # its speculative board row is the device's (else the host initializes it)
    # Index (into its state rows) of the row that holds its state after its last collected blind step; None: the state
    # pool's own current row (dynamic speculation, EngineConfig.spec_max_batch).
    spec_cur: int | None = None

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            raise ValueError("empty prompt")
        self.token_ids = list(self.prompt_ids)

    @property
    def host_bound(self) -> bool:
        """Needs every previous token on the host before its next step (no overlap): penalties,
        or a thinking budget still running."""
        return self.has_penalties or (self.params.thinking_token_budget is not None and not self.think_done)

    @property
    def has_penalties(self) -> bool:
        p = self.params
        return p.repetition_penalty != 1.0 or p.frequency_penalty != 0.0 or p.presence_penalty != 0.0

    @property
    def num_prompt(self) -> int:
        return len(self.prompt_ids)

    @property
    def output_ids(self) -> list[int]:
        """Generated tokens known on the host. Under overlap scheduling the newest entries
        of token_ids can be placeholders (-1) for steps still in flight; they are left out."""
        out = self.token_ids[self.num_prompt :]
        while out and out[-1] == -1:
            out = out[:-1]
        return out

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def is_decoding(self) -> bool:
        """Every known token except the newest has KV in cache, so the next step is one
        query token. After a preemption the recompute is a prefill over every known token."""
        return self.num_computed == self.num_tokens - 1

    @property
    def remaining_prefill(self) -> int:
        return self.num_tokens - self.num_computed
