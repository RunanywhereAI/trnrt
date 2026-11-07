"""engine/iopool.py: a pooled call's per-call IO at K fixed addresses.

The runtime caches a DMA ring set per distinct set of IO addresses (NEURON_RT_IO_RING_CACHE_SIZE), and every decode
call's host arguments (ModelRunner._dev) and fresh outputs (libtorch_neuronx_lite's Executable) sat at new addresses.
The pool's hazards are the reasons it can fall back: a slot's inputs rewritten while its previous call still runs, an
output buffer handed out while the engine still holds its previous step, and a piece writing the buffer the previous
piece left as its input. Each has a test here, and each guard a mutation that must fail.

FakeExecutable is libtorch_neuronx_lite's Executable.__call__ (compile/backend.py) in kind: a fresh torch.empty per
output not in io_map, the input itself for an aliased one, executor.execute(inputs, outputs), the first
output_count returned. Its kernel writes the way a device does, zeroing its output before it reads the inputs, so an
output that IS an input loses it. The runner tests' calls go through the real ModelRunner._exec_inner and _dev.
"""

from __future__ import annotations

import collections
import gc
import importlib.util
import types

import numpy as np
import pytest
import torch

from trnrt.engine import iopool

# ModelRunner's call body and LaunchBound drive the runner tests; the pool-only tests need neither.
HAVE_RUNNER = importlib.util.find_spec("trnrt.engine.model_runner") is not None
needs_runner = pytest.mark.skipif(not HAVE_RUNNER, reason="needs trnrt.engine.model_runner")
if HAVE_RUNNER:
    from trnrt.engine import model_runner as mr

KEY = ("decode", 4, 8)


class FakeExecutor:
    def __init__(self):
        self.sets = []  # every execution's IO addresses

    def execute(self, inputs, outputs):
        h, w, _board = inputs
        self.sets.append(tuple(t.data_ptr() for t in (*inputs, *outputs)))
        out = outputs[0]
        out.zero_()  # a device writes its output as it goes: an output that is also an input is lost here
        out.add_(h * w + 1)
        outputs[1].add_(1)  # the board, in place (io_map: output 1 is input 2)


class FakeExecutable:
    def __init__(self, rows: int = 4):
        self.executor = FakeExecutor()
        self.io_map = {1: 2}
        self.original_output_count = 1
        self.rows = rows

    def __call__(self, *inputs):
        outputs = [
            inputs[self.io_map[i]] if i in self.io_map else torch.empty(self.rows, dtype=torch.float32)
            for i in range(2)
        ]
        self.executor.execute(tuple(inputs), outputs)
        return outputs[: self.original_output_count]


def runner(k: int, pieces: int = 2, rank: int = 0):
    """A ModelRunner as far as _exec_inner and _dev reach: its decode runs `pieces` launches of ONE executable, each
    piece's output the next one's input (layer pieces of one shape)."""
    exe = FakeExecutable()
    exe.executor = iopool.PooledExecutor(exe.executor, exe)
    w, board = torch.full((4,), 2.0), torch.zeros(4)

    def decode(x):
        h = x  # the call's argument itself, as a graph takes it
        for _ in range(pieces):
            h = exe(h, w, board)[0]
        return h

    r = types.SimpleNamespace(
        tp_send=None,
        _decode=decode,
        _prefill=None,
        _spec_fns={},
        compile_seconds={},
        exec_counts={},
        watchdog=None,
        model=types.SimpleNamespace(tp_rank=rank),
        device=torch.device("cpu"),
        calls=collections.defaultdict(int),
        launch_bound=mr.LaunchBound(),
        iopool=iopool.Pool(k=k, names={"decode"}, rank=rank),
        exe=exe,
        board=board,
    )
    r._dev = types.MethodType(mr.ModelRunner._dev, r)
    r.call = lambda x: types.MethodType(mr.ModelRunner._exec_inner, r)("decode", KEY, [x])
    return r


def inputs(n: int):
    return [np.arange(4, dtype=np.float32) + 10 * c for c in range(n)]


def expected(x: np.ndarray, pieces: int = 2) -> torch.Tensor:
    h = torch.from_numpy(x).clone()
    for _ in range(pieces):
        h = h * 2 + 1
    return h


@needs_runner
def test_a_pooled_decode_runs_with_k_address_sets_and_the_same_values():
    r = runner(k=3)
    for x in inputs(12):
        out = r.call(x)
        assert torch.equal(out, expected(x))
        del out
        r._last_out = None  # the engine has read the step back and let go of it
    sets = set(r.exe.executor.inner.sets)
    assert len(sets) == 3 * 2  # 3 slots x 2 pieces (each piece its own buffers)
    assert r.iopool.fallback == {"held": 0, "argument": 0, "shape": 0} and r.iopool.reallocs == 0


def no_occurrence_key(monkeypatch):
    """One buffer per (executor, slot), whatever the piece: piece 1 is offered piece 0's buffer, its own input."""
    pooled = iopool.PooledExecutor._pooled
    monkeypatch.setattr(iopool.PooledExecutor, "_pooled", lambda self, ctx, o, outputs: pooled(self, ctx, 0, outputs))


def run_six(r):
    wrong = 0
    for x in inputs(6):
        out = r.call(x)
        wrong += not torch.equal(out, expected(x))
        del out
        r._last_out = None
    return wrong


@needs_runner
def test_without_the_occurrence_key_the_pieces_fall_back(monkeypatch):
    """The liveness check still refuses piece 0's buffer (piece 1 holds it as its input): values right, but piece 1
    takes a fresh tensor every call, so its addresses move again."""
    no_occurrence_key(monkeypatch)
    r = runner(k=3)
    assert run_six(r) == 0
    assert r.iopool.fallback["held"] == 6 and len(set(r.exe.executor.inner.sets)) > 6


@needs_runner
def test_without_the_liveness_check_too_a_piece_overwrites_its_input(monkeypatch):
    """Both guards off: piece 1 writes its output into its input, and the kernel zeroes it before reading it."""
    no_occurrence_key(monkeypatch)
    monkeypatch.setattr(iopool.PooledExecutor, "_held", staticmethod(lambda e: False))
    assert run_six(runner(k=3)) == 6


@needs_runner
def test_without_the_liveness_check_a_held_step_is_overwritten(monkeypatch):
    monkeypatch.setattr(iopool.PooledExecutor, "_held", staticmethod(lambda e: False))
    r = runner(k=2)
    xs = inputs(4)
    held = r.call(xs[0])
    want = expected(xs[0]).clone()
    for x in xs[1:]:
        r.call(x)
        r._last_out = None
    assert not torch.equal(held, want)


@needs_runner
def test_a_held_output_is_never_overwritten():
    r = runner(k=2)
    xs = inputs(8)
    held = r.call(xs[0])  # the engine still holds step 0 (as overlap holds the step in flight)
    want = expected(xs[0]).clone()
    for x in xs[1:]:
        out = r.call(x)
        assert torch.equal(out, expected(x))
        del out
        r._last_out = None
        assert torch.equal(held, want)
    assert r.iopool.fallback["held"] >= 1
    del held
    gc.collect()
    n = r.iopool.fallback["held"]
    for x in inputs(4):
        r.call(x)
        r._last_out = None
    assert r.iopool.fallback["held"] == n  # let go of: the slot is pooled again


def test_a_buffer_that_is_an_argument_of_the_call_is_not_written():
    pool = iopool.Pool(k=1, names={"decode"})
    exe = FakeExecutable()
    proxy = iopool.PooledExecutor(exe.executor, exe)
    exe.executor = proxy
    buf = torch.zeros(4)
    pool.outs[(id(proxy), KEY, 0, 0, 0)] = [buf, None]
    pool.begin("decode", KEY, None)
    pool.note_args([buf])  # e.g. a kept output passed back in by name
    out = exe(buf, torch.ones(4), torch.zeros(4))[0]
    pool.end(KEY, 0, None)
    assert out.data_ptr() != buf.data_ptr() and pool.fallback["argument"] == 1


class FakeDevice:
    """Launches queue in order; one completes only when it or a later one is read."""

    def __init__(self):
        self.launched = self.completed = self.reads = 0

    def launch(self):
        out = FakeOut(self, self.launched)
        self.launched += 1
        return out


class FakeOut:
    device = types.SimpleNamespace(type="neuron")

    def __init__(self, dev, idx):
        self.dev, self.idx = dev, idx

    def cpu(self):
        self.dev.completed = max(self.dev.completed, self.idx + 1)
        self.dev.reads += 1
        return self


@needs_runner
@pytest.mark.parametrize(("rank", "done", "waits"), [(0, 5, 1), (0, 6, 0), (1, 6, 1), (1, 7, 0)])
def test_a_slot_is_rewritten_only_after_its_previous_call_is_done(rank, done, waits):
    """Slot 0's previous call ended with execution 5. Rank 0's done is its own read-back: done 6 covers it. A
    worker's done is rank 0's count (done_to), which may be one call's tail ahead of its own device: 6 is not enough
    there, 7 is."""
    dev, bound = FakeDevice(), mr.LaunchBound(bound=64)
    launch = bound.counted(dev.launch)
    outs = [launch() for _ in range(9)]
    bound.done_to(done) if rank else bound._mark(done - 1)
    pool = iopool.Pool(k=2, names={"decode"}, rank=rank)
    pool.last[(KEY, 0)] = 5
    pool.ready(KEY, 0, bound)
    assert pool.waits == waits
    if waits:
        assert dev.completed >= 6 and bound.done >= 6
    del outs


@needs_runner
@pytest.mark.parametrize(("k", "waits"), [(2, True), (3, False), (iopool.POOL_K, False)])
def test_k_against_the_decode_overlap(k, waits):
    """The served decode loop: rank 0 launches step c + 1, then reads step c back (overlap), and sends each call with
    its done count. A worker at K = 2 must wait for the step before every call (the pipeline serialises); from K = 3
    it never waits. Rank 0 never waits at any K >= 2."""
    for rank in (0, 1):
        dev, bound = FakeDevice(), mr.LaunchBound(bound=64)
        launch = bound.counted(dev.launch)
        pool = iopool.Pool(k=k, names={"decode"}, rank=rank)
        outs, rank0_done = [], 0
        for c in range(20):
            if rank:
                bound.done_to(rank0_done)
            j = pool.begin("decode", KEY, bound)
            outs.append(launch())
            pool.end(KEY, j, bound)
            if c >= 1:  # rank 0 reads step c - 1 back after launching step c
                if rank == 0:
                    bound.done_through(outs[c - 1])
                rank0_done = c
        assert (pool.waits > 0) == (waits and rank == 1), (k, rank, pool.waits)


@needs_runner
def test_a_call_that_raises_closes_its_slot():
    r = runner(k=2)

    def boom(x):
        raise RuntimeError("launch failed")

    r._decode = boom
    with pytest.raises(RuntimeError):
        r.call(inputs(1)[0])
    assert iopool._CTX is None and r.iopool.last.get((KEY, 0)) is not None


def test_wrap_installs_the_proxy_once_and_leaves_plain_callables():
    exe = FakeExecutable()
    inner = exe.executor
    assert isinstance(iopool.wrap(exe).executor, iopool.PooledExecutor)
    assert iopool.wrap(exe).executor.inner is inner  # once
    f = lambda *a: a  # noqa: E731  (CPU mode's gm.forward)
    assert iopool.wrap(f) is f


def test_calls_not_named_are_not_pooled():
    pool = iopool.Pool(k=2, names={"decode"})
    j = pool.begin("prefill", ("prefill", 1024), None)
    assert j is None and iopool._CTX is None
    pool.end(("prefill", 1024), j, None)
    assert pool.calls == 0


def test_a_speculative_engine_pools_every_call_of_its_blind_step_and_only_those():
    plain = iopool.pool_names(False)
    assert plain == {"decode"} == iopool.POOL_NAMES
    spec = iopool.pool_names(True)
    assert plain < spec and "mtp_kv" not in spec
    assert spec == plain | {
        "verify",
        "spec_prep",
        "spec_post",
        "mtp",
        "mtp_post",
        "mtp_board",
        "spec_ckpt",
        "spec_init",
        "spec_host_init",
        "mtp_prefill_ids",
    }, "a graph of the blind step was added to or dropped from the pool"
    assert iopool.pool_names(True, kv_only=True) == spec | {"mtp_kv"}
    assert iopool.pool_names(False, kv_only=True) == plain  # the window graph exists only with speculation
    pool = iopool.Pool(k=2, names=spec)
    for name in sorted(spec):
        assert pool.begin(name, (name, 1), None) == 0, name
        pool.end((name, 1), 0, None)
    assert pool.begin("prefill", ("prefill", 512), None) is None and pool.begin("copy", ("copy", 4), None) is None


def test_the_pool_remembers_whether_warmup_fills_every_slot():
    assert iopool.Pool(k=4).warm is False
    assert iopool.Pool(k=4, warm=True).warm is True


@needs_runner
def test_the_pool_works_where_every_data_ptr_is_0(monkeypatch):
    """A device tensor's data_ptr() is 0 (libtorch_neuronx_lite): an argument check by data_ptr would take every
    pooled buffer for an argument of the call and fall back every time. By storage, the outputs stay pooled."""
    monkeypatch.setattr(torch.Tensor, "data_ptr", lambda self: 0)
    r = runner(k=3)
    storages = set()
    for x in inputs(12):
        out = r.call(x)
        assert torch.equal(out, expected(x))
        storages.add(out.untyped_storage()._cdata)
        del out
        r._last_out = None
    assert r.iopool.fallback == {"held": 0, "argument": 0, "shape": 0} and len(storages) == 3
