"""kernels/dsa_long_select_t.py at LNC 2: both programs of a logical core must run the same device-loop trip count. A
count that differs between them hangs the call on trn2 (an execution timeout, then a collectives error on every rank):
the kernel once forced program 1's count to 0 while program 0 ran the failed-row tile, which hung the first prefill
whose local top-K certificate failed. The simulator cannot show a hang, so this test runs the kernel's Python once per
program against a stand-in NKI that tracks the loop count's data (dma_copy / iota into the count tile, register_load)
and records each program's fori_loop trip, with trip planted at 1 (and 0). The control re-inserts the old override
(program 1's count set to 0) into the kernel's source: the check must then see the trips disagree."""

import importlib.util
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KDIR = os.path.join(ROOT, "trnrt", "kernels")
SEL_T = os.path.join(KDIR, "dsa_long_select_t.py")
TRIPS: dict[int, int] = {}


class Tile:
    """A stand-in tile: any shape, any view of it, and the one value the count tile carries."""

    def __init__(self, shape=(1,), val=None):
        self.shape, self.val = tuple(shape), val

    def __getitem__(self, _):
        return Tile(self.shape, self.val)

    def ap(self, *a, **k):
        return Tile(self.shape, self.val)


class Any:
    """Every NKI name the test does not model: callable (returns a tile), and usable as a value (dtypes, engines)."""

    def __init__(self, name):
        self.name = name

    def __call__(self, *a, **k):
        return Tile()


class Reg:
    val = None


class FakeNL(types.ModuleType):
    def __init__(self):
        super().__init__("nki.language")
        self.pid, self.npg = 0, 2

    def program_id(self, axis=0):
        return self.pid

    def num_programs(self, axes=0):
        return self.npg

    def program_ndim(self):
        return 1

    def ndarray(self, shape, dtype=None, buffer=None):
        return Tile(shape)

    def fori_loop(self, lo, reg, body):
        TRIPS[self.pid] = reg.val - lo  # the trip this program's device loop would run (the body is not traced here)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Any(name)


class FakeISA(types.ModuleType):
    def __init__(self):
        super().__init__("nki.isa")

    def dma_copy(self, dst=None, src=None, **k):
        if isinstance(dst, Tile) and isinstance(src, Tile):
            dst.val = src.val

    def iota(self, dst=None, pattern=None, offset=0, channel_multiplier=0, **k):
        if isinstance(dst, Tile):
            dst.val = offset  # the count tile: a [1, 1] iota is its offset

    def register_alloc(self, *a, **k):
        return Reg()

    def register_load(self, reg, t):
        reg.val = t.val

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Any(name)


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _load(monkeypatch, src: str):
    """dsa_long_select (fresh) and the given dsa_long_select_t source, both against the stand-in NKI."""
    import trnrt.kernels as pkg

    nl, isa = FakeNL(), FakeISA()
    nki = types.ModuleType("nki")
    nki.jit = lambda f=None, **k: f if f is not None else (lambda g: g)
    nki.language, nki.isa = nl, isa
    for name, mod in (("nki", nki), ("nki.language", nl), ("nki.isa", isa)):
        monkeypatch.setitem(sys.modules, name, mod)
    path = os.path.join(KDIR, "dsa_long_select.py")
    spec = importlib.util.spec_from_file_location("trnrt.kernels.dsa_long_select", path)
    dls = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "trnrt.kernels.dsa_long_select", dls)
    monkeypatch.setattr(pkg, "dsa_long_select", dls, raising=False)
    spec.loader.exec_module(dls)
    mod = types.ModuleType("trnrt.kernels._dsa_long_select_t_trips")
    mod.__package__, mod.__file__ = "trnrt.kernels", SEL_T
    exec(compile(src, SEL_T, "exec"), mod.__dict__)  # noqa: S102  # the kernel source is the subject: run against the stand-in NKI
    assert mod.dsa_long_select_t_kernel is not None, "the kernel was not defined against the stand-in NKI"
    return mod, nl


def _trips(mod, nl, trip: int) -> dict[int, int]:
    TRIPS.clear()
    for pid in (0, 1):
        nl.pid = pid
        mod.dsa_long_select_t_kernel(
            qT=Tile((1, 128, 4, 128)),
            w=Tile((1, 128, 4)),
            npool=Tile((1, 128, 1)),
            pk=Tile((8192, 128)),
            identb=Tile((128, 128)),
            trip=Tile((1, 1), trip),
            keep=512,
            scale=0.1,
            sub=64,
            lsub=8,
            one=0,
            rev=0,
        )
    return dict(TRIPS)


OLD_OVERRIDE = "        if idle:\n            nisa.iota(dst=cnt, pattern=[[0, 1]], offset=0, channel_multiplier=0)\n"
COUNT_LINE = "        nisa.dma_copy(dst=cnt, src=trip)\n"


@pytest.mark.parametrize("trip", [1, 0])
def test_both_programs_run_the_same_trip(monkeypatch, trip):
    mod, nl = _load(monkeypatch, _read(SEL_T))
    got = _trips(mod, nl, trip)
    assert got == {0: trip, 1: trip}, got


def test_the_old_override_is_caught(monkeypatch):
    """Positive control: program 1's count overwritten with 0 (the form that hung) must make the trips disagree at 1."""
    src = _read(SEL_T)
    assert src.count(COUNT_LINE) == 1
    old = src.replace(COUNT_LINE, COUNT_LINE + OLD_OVERRIDE)
    mod, nl = _load(monkeypatch, old)
    got = _trips(mod, nl, 1)
    assert got == {0: 1, 1: 0}, got
