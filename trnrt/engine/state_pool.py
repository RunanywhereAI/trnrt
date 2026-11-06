"""Device-resident recurrent state of the linear-attention (KDA) layers: rows for running requests,
and state checkpoints for the prefix cache.

A linear-attention layer (models/linear_attn.py) keeps, per sequence, the last K - 1 inputs
of its short conv and one [Dk, Dv] fp32 matrix per head, instead of KV pages. Each such layer
gets two pools on the device, conv [rows, K - 1, conv_dim] (model dtype) and recurrent
[rows, heads, Dk, Dv] (fp32, as transformers' mamba_ssm_dtype float32), and every graph call
passes a state_slot per sequence: decode reads and writes its row once per token, every
prefill chunk carries the row forward, so chunked prefill and decode compose.

Rows, in order: row 0, the scratch row that padded batch rows read and write; one row per
request that can run at once; the checkpoint rows; and the zero row, the last.

Rows are not reset on the host. A sequence whose first computed position is 0 starts from
zero state inside the graph: it reads the zero row (kernels/shell.py start_row), which is
zeros from allocation and never written (no free list holds it, and copy_state refuses it as
a target, check_writes). So a request admitted into a reused row, or recomputed from scratch
after a preemption, never sees its predecessor's state. The zero row keeps that read in range:
otherwise it is one row past the pool, which the KDA kernels' hardware-DMA reads skip
(oob_mode skip), and that skipped read was the site of a device hang in serving warmup.

Speculative decoding: a request in a speculative seat holds `rows_per_req` = 1 + k consecutive rows. The verify graph
reads its CURRENT row and writes the state after each of its 1 + k positions into its rows in order, and the row after the
last accepted position becomes current (`set_current`), so a rejected draft costs no copy. vLLM keeps the same layout for its
gated-delta-net layers (`vllm/v1/attention/backends/gdn_attn.py`: num_spec + 1 state slots per request; see
THIRD_PARTY_NOTICES.md, "Designs followed").

Seats: under dynamic speculation (`EngineConfig.spec_max_batch`) at most that many requests speculate at once, so only
`spec_seats` seats need 1 + k rows and the others are plain seats of one row. A request takes the kind the engine prefers
(`prefer_spec`: speculative in speculative mode, plain in plain mode), else the other kind. The engine's plain-to-speculative
flip, right after it drains the step in flight, moves every running request out of a plain seat into a speculative one
(`move`, then `ModelRunner.copy_state` of its current row). That is what keeps the row count of a cap-12 pool at 34 (k 2,
4 speculative seats, 12 checkpoint rows: 1 + 4 x 3 + 8 + 12 + 1). With every seat speculative, each holds `rows_per_req` rows.

State checkpoints (the prefix cache of these models): the checkpoint rows hold the state of
all linear layers after a page-aligned prefix, one per radix node that has one
(engine/radix_cache.py RadixNode.ckpt; SGLang v0.5.21 mem_cache/unified_cache/components/
mamba.py keeps one state per node the same way). They are written and read by copying whole
rows on the device (ModelRunner.copy_state).
"""

from __future__ import annotations

import torch

from .request import Status

SCRATCH_ROW = 0


class StatePool:
    def __init__(
        self,
        model,
        rows: int,
        device: torch.device,
        conv_dtype: torch.dtype,
        ckpt_rows: int,
        rows_per_req: int = 1,
        spec_seats: int | None = None,
    ):
        """rows: requests that can run at once (the seats), each holding rows_per_req rows (a plain seat of a mixed pool holds
        one); ckpt_rows: rows for state checkpoints. Plus the scratch row and the zero row. spec_seats
        (`EngineConfig.state_spec_seats`): only that many of the seats hold rows_per_req rows, the rest one row."""
        self.per_req = rows_per_req
        seats = rows if spec_seats is None else int(spec_seats)
        if not 0 < seats <= rows:
            raise ValueError(f"spec seats {seats} must be in 1 .. {rows} (the seats)")
        self.spec_seats = seats
        self.mixed_seats = seats < rows and rows_per_req > 1  # two seat kinds (else every seat is rows_per_req rows)
        self.prefer_spec = (
            True  # the kind a request's first use takes (the engine sets it per step: speculative mode or not)
        )
        n_run = seats * rows_per_req + (rows - seats)
        self.num_ckpt_rows = ckpt_rows
        self.zero_row = 1 + n_run + ckpt_rows
        self.rows = self.zero_row + 1
        shapes = model.state_shapes()
        self.conv = [torch.zeros((self.rows, *c), dtype=conv_dtype, device=device) for c, _ in shapes]
        self.rec = [torch.zeros((self.rows, *r), dtype=torch.float32, device=device) for _, r in shapes]
        model.bind_state(self.conv, self.rec)
        # Running rows go out in sets of rows_per_req consecutive ids, lowest first (pop() takes the end of the list).
        self._free: list[list[int]] = [
            list(range(1 + i * rows_per_req, 1 + (i + 1) * rows_per_req)) for i in range(seats - 1, -1, -1)
        ]
        base = 1 + seats * rows_per_req  # plain seats (mixed_seats): one row each, after the speculative seats
        self._pfree: list[list[int]] = [[base + j] for j in range(rows - seats - 1, -1, -1)]
        self._held: dict[str, list] = {}  # rid -> [rows, current index, request]
        # hold: released rows wait in _cooling until release_cooling() (kv_pool.PagePool.hold, for the same reason: an
        # overlapped step may still be reading or writing them).
        self.hold = False
        self._cooling: list[list[int]] = []
        self._ckpt_free = list(range(self.zero_row - 1, n_run, -1))

    def pools(self) -> list[torch.Tensor]:
        return self.conv + self.rec

    def check_writes(self, rows) -> None:
        """Refuse a write into the zero row: copy_state's targets and every row handed out."""
        if any(int(r) == self.zero_row for r in rows):
            raise RuntimeError(f"state row {self.zero_row} is the zero row: never written")

    def rows_of(self, req) -> list[int]:
        """The request's rows, allocated on first use. When none are free, rows held by requests that are not running
        (finished, or preempted and waiting) are reclaimed: every call is built from running requests only, so a running
        request's rows are never taken, and a reclaimed holder restarts from zero state or a checkpoint if it ever runs
        again. A finished holder's last step may still be in flight; it executes before any later call."""
        held = self._held.get(req.rid)
        if held is not None:
            return held[0]
        order = (self._free, self._pfree) if self.prefer_spec or not self.mixed_seats else (self._pfree, self._free)
        if not any(order):
            self._reclaim()
        free = next((f for f in order if f), None)
        if free is None:
            raise RuntimeError(f"recurrent-state pool exhausted ({len(self._held)} requests hold rows)")
        rs = free.pop()
        self._held[req.rid] = [rs, 0, req]
        return rs

    def _list(self, rs: list[int]) -> list[list[int]]:
        """The free list a set of rows goes back to: a speculative seat's or a plain seat's."""
        return self._free if len(rs) == self.per_req or not self.mixed_seats else self._pfree

    def _is_spec_rows(self, rs: list[int]) -> bool:
        return len(rs) == self.per_req

    def _reclaim(self, spec: bool | None = None) -> None:
        """Rows held by requests that are not running back to their free lists (only the given kind's when spec is not
        None): every call is built from running requests only."""
        for rid, (rs, _, other) in list(self._held.items()):
            if other.status is not Status.RUNNING and (spec is None or self._is_spec_rows(rs) == spec):
                del self._held[rid]
                self._list(rs).append(rs)

    def holds(self, req) -> bool:
        return req.rid in self._held

    def is_spec(self, req) -> bool:
        """Whether the request holds a speculative seat (rows_per_req rows); False when it holds none yet."""
        held = self._held.get(req.rid)
        return held is not None and self._is_spec_rows(held[0])

    def spec_available(self) -> int:
        """Speculative seats a move or a first use could take: free ones plus those of non-running holders."""
        return len(self._free) + sum(
            1 for rs, _, o in self._held.values() if o.status is not Status.RUNNING and self._is_spec_rows(rs)
        )

    def move(self, req, to_spec: bool) -> tuple[int, int] | None:
        """Re-seat a request in a seat of the other kind (speculative: rows_per_req rows, plain: one). Returns (its
        current row, the new seat's first row) for the caller to copy (ModelRunner.copy_state), the new first row
        becoming current; None when no seat of that kind is free. Only between steps: an asynchronous step reads its rows
        from the device board (engine/spec_async.py), which the engine rebuilds from the host after a move."""
        held = self._held.get(req.rid)
        if held is None or self._is_spec_rows(held[0]) == to_spec:
            return None
        free = self._free if to_spec else self._pfree
        if not free:
            self._reclaim(spec=to_spec)
        if not free:
            return None
        new = free.pop()
        src = held[0][held[1]]
        self._held[req.rid] = [new, 0, req]
        self._give_back(held[0])
        return src, new[0]

    def _give_back(self, rs: list[int]) -> None:
        if self.hold:
            self._cooling.append(rs)
        else:
            self._list(rs).append(rs)

    def row(self, req) -> int:
        """The request's current row: the one its next call reads and (outside verify) writes."""
        rs = self.rows_of(req)
        return rs[self._held[req.rid][1]]

    def set_current(self, req, i: int) -> None:
        """After a verify: row i of the request's rows holds the state its next call starts from."""
        self.rows_of(req)
        self._held[req.rid][1] = i

    def release(self, req) -> None:
        held = self._held.pop(req.rid, None)
        if held is not None:
            self._give_back(held[0])

    def release_cooling(self) -> None:
        for rs in self._cooling:
            self._list(rs).append(rs)
        self._cooling.clear()

    def alloc_ckpt(self) -> int | None:
        return self._ckpt_free.pop() if self._ckpt_free else None

    def free_ckpt(self, row: int) -> None:
        self._ckpt_free.append(row)

    def free_ckpt_rows(self) -> list[int]:
        return self._ckpt_free

    @property
    def num_free_ckpts(self) -> int:
        return len(self._ckpt_free)

    def bytes_per_row(self) -> int:
        return sum(t[0].numel() * t.element_size() for t in self.pools())
