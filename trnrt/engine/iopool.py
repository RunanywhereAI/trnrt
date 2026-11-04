"""A decode call's per-call device IO at POOL_K fixed address sets. Host-side only: the traced graphs and their keys
are unchanged.

Why. The Neuron runtime keeps, per NEFF, a cache of DMA ring sets for its IO tensors (NEURON_RT_IO_RING_CACHE_SIZE),
and needs a new set whenever a call's IO tensors sit at addresses no cached set was built for. Two things move a
decode call's IO every call:
  - ModelRunner._dev uploads every host argument to a new device tensor (torch.from_numpy(a).to(device));
  - libtorch_neuronx_lite allocates every output that does not alias an input with torch.empty, per call
    (compile/backend.py, Executable.__call__).
The weights, caches and in-place buffers stay put (GLM-5.3-Flash's whole-step decode graph: ~1,100 of them, 102
outputs aliased), but a ring set covers them all. Its saturated decode NEFFs held ~287 MB of IO rings per NeuronCore
(~8.6 MB a set, inferred, at 32 cached sets) against ~12.5 MB at one set, and an engine at 16 concurrent requests
failed to build one more set (NRT_RESOURCE) with 3.75 GB of HBM in IO rings.

The pool (calls named in POOL_NAMES). Each call of a graph key takes slot j = its count mod K:
  - inputs: each host argument is copied (copy_) into slot j's device buffer for that argument, allocated once;
  - outputs: the executor proxy (wrap(), installed by the backend on every Executable) puts slot j's buffer in place
    of each fresh, non-aliased output before the NEFF runs. A buffer is keyed by (executor, slot, occurrence): a call
    that runs one NEFF twice (layer pieces of one shape) gets two buffers, so a piece never writes the buffer the
    previous piece left for it as input.
So each NEFF of a pooled key runs with K address sets (times its occurrences per call), all built at warmup.
Reuse is safe by construction or falls back to a fresh tensor (a ring miss, never a hazard):
  - an input buffer is rewritten only after slot j's previous call is done. A worker's `done` follows rank 0's count
    and may run ahead of its own device by one call's tail (LaunchBound.done_to), so a worker also needs the call
    after it done, or waits for its own output;
  - an output buffer is handed out as a new view each time; while the previous view is still alive (the engine has
    not let go of that step's output, or a piece still has it as its input) the call gets a fresh tensor instead, as
    it does when the buffer is one of the call's own arguments. The liveness check is what keeps values right; the
    occurrence key is what keeps a multi-piece call pooled rather than falling back every call.
"""

from __future__ import annotations

import time
import weakref

import torch

# Slots per pooled key. The overlapped decode loop launches step c + 1 before it reads step c back, so a worker rank
# at K = 2 waits for the previous step before every call (the pipeline serialises); from K = 3 no rank waits, and 4
# keeps one slot of margin.
POOL_K = 4
POOL_NAMES = frozenset({"decode"})

# The call running now (one engine thread per process runs every graph call): set by Pool.begin, read by the
# executor proxies the call's NEFFs run through. None between calls and for calls that are not pooled.
_CTX = None


class _Ctx:
    __slots__ = ("args", "j", "key", "occ", "pool")

    def __init__(self, pool, key, j):
        self.pool, self.key, self.j = pool, key, j
        self.occ: dict = {}  # id(proxy) -> its executions so far in this call
        self.args: set = set()  # storage identities of the call's device arguments, filled by note_args


def _sid(t) -> int:
    """A tensor's storage identity. Not data_ptr(): a device tensor's is 0 (libtorch_neuronx_lite tells its tensors
    apart by storage too, compile/backend.py)."""
    return t.untyped_storage()._cdata


class Pool:
    """One per ModelRunner (rank)."""

    def __init__(self, k: int = POOL_K, names=POOL_NAMES, rank: int = 0):
        self.k, self.names, self.rank = int(k), frozenset(names), rank
        self.count: dict = {}  # key -> calls so far
        self.inputs: dict = {}  # (key, j, path) -> device buffer
        self.outs: dict = {}  # (id(proxy), key, j, occurrence, output index) -> [buffer, weakref to the view handed out]
        self.last: dict = {}  # (key, j) -> LaunchBound index of the slot's previous call's last execution
        self.calls = self.waits = self.reallocs = 0
        self.wait_s = self.wait_max = 0.0
        self.fallback = {"held": 0, "argument": 0, "shape": 0}

    def begin(self, name: str, key, bound) -> int | None:
        """Open a call: its slot (None when this call is not pooled), after waiting for the slot's previous call."""
        global _CTX  # noqa: PLW0603  # the call in flight, process-wide: the proxy reads it (tests read it too)
        j = None
        if self.k > 0 and name in self.names:
            n = self.count.get(key, 0)
            self.count[key] = n + 1
            j = n % self.k
            self.calls += 1
            self.ready(key, j, bound)
        _CTX = _Ctx(self, key, j) if j is not None else None
        return j

    def ready(self, key, j: int, bound) -> None:
        """Slot j's input buffers may be rewritten: its previous call is done on this rank's device."""
        i = self.last.get((key, j))
        if i is None or bound is None:
            return
        margin = 0 if self.rank == 0 else 1  # a worker's done may run one call's tail ahead (LaunchBound.done_to)
        if bound.done > i + margin:
            return
        t0 = time.perf_counter()
        bound.wait_for(i)
        dt = time.perf_counter() - t0
        self.waits += 1
        self.wait_s += dt
        self.wait_max = max(self.wait_max, dt)

    def put(self, slot, host: torch.Tensor, device) -> torch.Tensor:
        """host (a CPU tensor) on the device at slot's fixed buffer."""
        buf = self.inputs.get(slot)
        if buf is not None and buf.shape == host.shape and buf.dtype == host.dtype:
            buf.copy_(host)
            return buf
        if buf is not None:
            self.reallocs += 1
        buf = self.inputs[slot] = host.to(device, copy=True)  # never the host array's own memory (a CPU device)
        return buf

    def note_args(self, args) -> None:
        """The call's device arguments: a pooled output never overwrites one of them."""
        ctx = _CTX
        if ctx is None:
            return

        def walk(a):
            if isinstance(a, torch.Tensor):
                ctx.args.add(_sid(a))
            elif isinstance(a, (tuple, list)):
                for x in a:
                    walk(x)

        walk(args)

    def end(self, key, j: int | None, bound) -> None:
        global _CTX  # noqa: PLW0603  # see begin
        _CTX = None
        if j is not None and bound is not None:
            self.last[(key, j)] = bound.n - 1

    def stats(self) -> dict:
        return {
            "k": self.k,
            "names": sorted(self.names),
            "rank": self.rank,
            "calls": self.calls,
            "waits": self.waits,
            "wait_s": round(self.wait_s, 4),
            "wait_max_ms": round(1000 * self.wait_max, 3),
            "reallocs": self.reallocs,
            "fallback": dict(self.fallback),
            "slots": {str(k): min(n, self.k) for k, n in list(self.count.items())},
        }


class PooledExecutor:
    """Stands in for an Executable's `executor` (torch.classes.neuron.Executor): execute(inputs, outputs) with slot
    j's buffers in place of the fresh outputs, while a pooled call is open; otherwise as it was."""

    def __init__(self, inner, exe):
        self.inner, self.exe = inner, exe
        io_map = getattr(exe, "io_map", None) or {}
        self.aliased = frozenset(int(i) for i in io_map)

    def __getattr__(self, name):  # anything else the Executable or a hook reaches for
        return getattr(self.inner, name)

    def execute(self, inputs, outputs):
        ctx = _CTX
        if ctx is not None:
            o = ctx.occ.get(id(self), 0)
            ctx.occ[id(self)] = o + 1
            self._pooled(ctx, o, outputs)
        return self.inner.execute(inputs, outputs)

    def _pooled(self, ctx, o: int, outputs) -> None:
        pool = ctx.pool
        for i, fresh in enumerate(outputs):
            if i in self.aliased or not isinstance(fresh, torch.Tensor):
                continue
            k = (id(self), ctx.key, ctx.j, o, i)
            e = pool.outs.get(k)
            if e is None:  # first use: the fresh tensor becomes the slot's buffer
                e = pool.outs[k] = [fresh, None]
            buf = e[0]
            if buf.shape != fresh.shape or buf.dtype != fresh.dtype:
                pool.fallback["shape"] += 1
                continue
            if self._held(e):  # its previous view is alive: a step not let go of, or the next piece's input
                pool.fallback["held"] += 1
                continue
            if _sid(buf) in ctx.args:  # the buffer is an argument of this very call
                pool.fallback["argument"] += 1
                continue
            v = buf.view(buf.shape)
            e[1] = weakref.ref(v)
            outputs[i] = v

    @staticmethod
    def _held(e) -> bool:
        return e[1] is not None and e[1]() is not None


def wrap(exe):
    """The backend's Executable with its executor behind PooledExecutor; anything else (CPU mode's gm.forward, a
    capture) unchanged."""
    if hasattr(exe, "executor") and hasattr(exe, "io_map") and not isinstance(exe.executor, PooledExecutor):
        exe.executor = PooledExecutor(exe.executor, exe)
    return exe
