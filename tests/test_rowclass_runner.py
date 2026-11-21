"""The runner side of the DSA selection's static row classes (engine/model_runner.py, file end: rowclass_init /
rowclass_order / rowclass_page). The row-class selection kernel (kernels/dsa_sel2gc.py) scores only a slot's first
128 NG_s pages, so the runner must put the batch in class-slot order and send a batch that does not fit to the next page
bucket. On the host, with the kernel module stood in for (its LAYOUTS and its RUNNER guard):
- the order: longest first, stable, only for a batch size with a layout;
- the guard: a fitting batch keeps the classed bucket; one that does not fit takes the next bucket; one that would fit
  sorted but arrived unsorted is refused; a row exactly at its slot's edge fits, one page more does not;
- the construction checks: a layout at a batch size or bucket that does not exist, no larger bucket to fall back to,
  classes not widest first, and a verify or MTP call that could reach the classed bucket all refuse, and the kernel's
  guard (dsa_sel2gc.RUNNER) stays unset after a refusal; a CPU engine has no row classes;
- the engine orders every plain decode chunk with rowclass_order before the runner's decode, the one plain launch."""

from __future__ import annotations

import ast
import importlib.machinery
import inspect
import sys
import types

import pytest
import torch

import trnrt.kernels
from trnrt.engine import engine as E
from trnrt.engine import model_runner as MR

pytestmark = pytest.mark.usefixtures("decode_routes")  # set_decode_routes's writes are undone

LAY = {24: (24,) * 6 + (8,) * 6 + (4,) * 12, 16: (24,) * 4 + (8,) * 12}  # the 9-16 build's layouts
PS = 256
MTP2 = types.SimpleNamespace(spec_method="mtp", spec_k=2)


def _runner(**kw):
    r = types.SimpleNamespace(
        plain_buckets=(16, 24),
        spec_buckets=(1, 2, 4),
        page_buckets=(1024, 3072, 4096),
        spec_page_buckets=(1024, 4096),
        ps=PS,
        rc={},
        rc_steps=0,
        rc_fallbacks=0,
        device=torch.device("meta"),
    )
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def _seqs(pages):
    """Sequences whose decode context ends in the last token of `pages` pages (end = pages x PS), tagged by index."""
    return [types.SimpleNamespace(end=p * PS, tag=i) for i, p in enumerate(pages)]


@pytest.fixture
def sel2gc(monkeypatch):
    """A stand-in for kernels/dsa_sel2gc.py: the layouts set_decode_routes would leave, and its runner guard unset."""
    name = "trnrt.kernels.dsa_sel2gc"
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.LAYOUTS = dict(LAY)
    mod.RUNNER = None
    monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(trnrt.kernels, "dsa_sel2gc", mod, raising=False)
    return mod


def test_order_longest_first_stable():
    r = _runner(rc=LAY)
    s = [
        *_seqs([10, 1500, 300, 1500, 2900, 300]),
        types.SimpleNamespace(end=1500 * PS - 255, tag=6),
    ]  # 6: 1,500 pages too
    got = MR.rowclass_order(r, s)
    assert [x.tag for x in got] == [4, 1, 3, 6, 2, 5, 0]
    assert [x.tag for x in s] == list(range(7)), "the scheduler's list is not mutated"


def test_order_only_with_a_layout():
    s = _seqs([10, 2000, 30])
    assert MR.rowclass_order(_runner(rc={}), s) is s
    assert MR.rowclass_order(_runner(rc={24: LAY[24]}), s) is s  # 3 rows -> B16: no layout there
    assert [x.tag for x in MR.rowclass_order(_runner(rc={16: LAY[16]}), s)] == [1, 2, 0]


def test_page_guard():
    r = _runner(rc=LAY)
    fit = MR.rowclass_order(
        r, _seqs([3072, 3000, 2000, 1500, 1025, 1024, 1024, 1000, 512, 500, *[400] * 10, 512, 1, 1, 1])
    )
    assert MR.rowclass_page(r, fit, 24, 3072) == 3072 and r.rc_steps == 1
    edge = MR.rowclass_order(r, _seqs([3072] * 6 + [1024] * 6 + [512] * 12))  # every row exactly at its slot's edge
    assert MR.rowclass_page(r, edge, 24, 3072) == 3072
    over = MR.rowclass_order(
        r, _seqs([3072] * 6 + [1024] * 5 + [1025] + [512] * 12)
    )  # 7 rows past 1,024 pages, 6 slots
    assert MR.rowclass_page(r, over, 24, 3072) == 4096 and r.rc_fallbacks == 1
    small = MR.rowclass_order(r, _seqs([1024] * 4 + [513] + [512] * 19))  # 5 rows past 512 pages but under NG8
    assert MR.rowclass_page(r, small, 24, 3072) == 3072
    with pytest.raises(RuntimeError, match="not in class-slot order"):
        MR.rowclass_page(r, list(reversed(fit)), 24, 3072)
    assert MR.rowclass_page(r, list(reversed(fit)), 24, 1024) == 1024  # not the classed bucket: today's pick
    assert MR.rowclass_page(r, _seqs([3000, 10]), 4, 3072) == 3072  # no layout at B4
    assert MR.rowclass_page(_runner(), _seqs([3000, 10]), 24, 3072) == 3072  # classes off
    assert (r.rc_steps, r.rc_fallbacks) == (3, 1)


def test_init_installs_the_guard(sel2gc):
    assert MR.rowclass_init(_runner(), MTP2) == LAY
    assert sel2gc.RUNNER == "trnrt.engine.model_runner.rowclass_page"
    assert inspect.getsource(MR).count("dsa_sel2gc.RUNNER" + " = ") == 1, "one assignment, in rowclass_init"


def test_init_without_layouts_or_on_cpu_is_off(sel2gc):
    sel2gc.LAYOUTS = {}
    assert MR.rowclass_init(_runner(), MTP2) == {} and sel2gc.RUNNER is None
    sel2gc.LAYOUTS = dict(LAY)
    assert MR.rowclass_init(_runner(device=torch.device("cpu")), MTP2) == {} and sel2gc.RUNNER is None


def test_init_without_the_kernel_module_is_off(monkeypatch):
    monkeypatch.setattr(MR, "_row_class_layouts", dict)  # (a tree without kernels/dsa_sel2gc.py has none)
    assert MR.rowclass_init(_runner(), MTP2) == {}


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ({"plain_buckets": (24,)}, "B 16 is not a plain decode bucket"),
        ({"page_buckets": (1024, 3072)}, "needs a larger page bucket"),
        ({"page_buckets": (1024, 4096)}, "needs a larger page bucket"),
        ({"spec_page_buckets": (1024, 3072, 4096), "spec_buckets": (1, 2, 8)}, "would skip the guard"),  # 8 x Q 3 = 24
        ({"spec_page_buckets": (1024, 3072, 4096), "spec_buckets": (16,)}, "would skip the guard"),
    ],
)
def test_init_refuses(sel2gc, change, why):
    with pytest.raises(ValueError, match=why):
        MR.rowclass_init(_runner(**change), MTP2)
    assert sel2gc.RUNNER is None, "a refused construction must leave the kernel's guard unset"


def test_init_refuses_narrow_first(sel2gc):
    sel2gc.LAYOUTS = {16: (8,) * 12 + (24,) * 4}
    with pytest.raises(ValueError, match="widest first"):
        MR.rowclass_init(_runner(), MTP2)


def test_a_spec_ladder_without_the_bucket_passes(sel2gc):
    r = _runner(spec_buckets=(1, 2, 4, 8))  # 8 x 3 = 24, but no verify runs at 3072 (spec ladder 1024, 4096)
    assert MR.rowclass_init(r, MTP2) == LAY
    r = _runner(spec_buckets=(16, 24), spec_page_buckets=(1024, 3072, 4096))  # no spec method: no verify or MTP graph
    assert MR.rowclass_init(r, types.SimpleNamespace(spec_method=None, spec_k=4)) == LAY


def test_the_engine_orders_every_plain_decode_chunk_before_the_decode():
    """Structural: in LLMEngine._launch, the chunk handed to runner.decode is rowclass_order's, and the same list is the
    one the launch records and whose states it saves (outputs and saves follow the chunk's order)."""
    f = next(
        n
        for n in ast.walk(ast.parse(inspect.getsource(E.LLMEngine)))
        if isinstance(n, ast.FunctionDef) and n.name == "_launch"
    )
    src = ast.unparse(f)
    assert "chunk = rowclass_order(self.runner, plain[i:i + max_b])" in src
    assert "launches.append((chunk, self.runner.decode(chunk), 'decode'))" in src
    assert "self._save_states(chunk)" in src


def test_the_runner_decode_takes_the_guards_page_bucket():
    dec = next(
        n
        for n in ast.walk(ast.parse(inspect.getsource(MR.ModelRunner)))
        if isinstance(n, ast.FunctionDef) and n.name == "decode"
    )
    assert "P = rowclass_page(self, seqs, B, pick_bucket(" in ast.unparse(dec)
