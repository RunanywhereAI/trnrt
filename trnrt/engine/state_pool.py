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
    def __init__(self, model, rows: int, device: torch.device, conv_dtype: torch.dtype, ckpt_rows: int):
        """rows: requests that can run at once (max_num_seqs), one row each; ckpt_rows: rows for
        state checkpoints. Plus the scratch row and the zero row."""
        self.num_ckpt_rows = ckpt_rows
        self.zero_row = 1 + rows + ckpt_rows
        self.rows = self.zero_row + 1
        shapes = model.state_shapes()
        self.conv = [torch.zeros((self.rows, *c), dtype=conv_dtype, device=device) for c, _ in shapes]
        self.rec = [torch.zeros((self.rows, *r), dtype=torch.float32, device=device) for _, r in shapes]
        model.bind_state(self.conv, self.rec)
        # pop() hands out the lowest free row first.
        self._free = list(range(rows, 0, -1))
        self._held: dict[str, tuple[int, object]] = {}  # rid -> (row, request)
        self._ckpt_free = list(range(self.zero_row - 1, rows, -1))

    def pools(self) -> list[torch.Tensor]:
        return self.conv + self.rec

    def check_writes(self, rows) -> None:
        """Refuse a write into the zero row: copy_state's targets and every row handed out."""
        if any(int(r) == self.zero_row for r in rows):
            raise RuntimeError(f"state row {self.zero_row} is the zero row: never written")

    def row(self, req) -> int:
        """The request's row, allocated on first use. When none is free, rows held by requests
        that are not running (finished, or preempted and waiting) are reclaimed: every call is
        built from running requests only, so a running request's row is never taken, and a
        reclaimed holder restarts from zero state or a checkpoint if it ever runs again. A
        finished holder's last step may still be in flight; it executes before any later call."""
        held = self._held.get(req.rid)
        if held is not None:
            return held[0]
        if not self._free:
            for rid, (r, other) in list(self._held.items()):
                if other.status is not Status.RUNNING:
                    del self._held[rid]
                    self._free.append(r)
        if not self._free:
            raise RuntimeError(f"recurrent-state pool exhausted ({len(self._held)} requests hold rows)")
        r = self._free.pop()
        self._held[req.rid] = (r, req)
        return r

    def release(self, req) -> None:
        held = self._held.pop(req.rid, None)
        if held is not None:
            self._free.append(held[0])

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
