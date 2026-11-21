"""The speculative configuration (trnrt/__main__.py PRESETS latency, EngineConfig's speculation fields):

- the latency build's speculation values are the shipped configuration's (tests/test_builds.py checks the rest), and the
  balanced and throughput builds have none of them;
- no plain decode graph (plain_decode False) is allowed only where no plain decode can launch: each reason trips on its
  own, with the reason named, and a plain launch on such an engine fails loud;
- every inconsistent combination is refused, each with a positive control that the valid one builds.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from trnrt import __main__ as cli
from trnrt.engine import model_runner as mr

pytestmark = pytest.mark.usefixtures("decode_routes")  # set_decode_routes's writes are undone

if TYPE_CHECKING:
    from trnrt.config import EngineConfig

LATENCY_SPEC = {
    "spec_method": "mtp",
    "spec_k": 2,
    "spec_max_batch": 8,
    "spec_hysteresis": 1,
    "spec_bucket_max": 8,
    "plain_decode": False,
    "plain_bucket_min": 0,
    "spec_page_buckets": (1024, 2048, 4096),
    "mtp_fill_rows": 1024,
    "mtp_chunk_q": 64,
    "mtp_kv_only": True,
    "dyn_mtp_kv": True,
    "state_seats": 8,
    "state_spec_seats": None,
}


@pytest.fixture(autouse=True)
def _no_synthetic_acceptance_from_the_shell(monkeypatch):
    # cli reads TRNRT_SPEC_SYNTH_AL; a developer's exported value must not change what these tests build.
    monkeypatch.delenv("TRNRT_SPEC_SYNTH_AL", raising=False)


def _cfg(*argv) -> EngineConfig:
    return cli.engine_config(["--model", "m", *argv])


def test_the_latency_build_speculates_as_shipped():
    c = _cfg("--preset", "latency")
    assert {k: getattr(c, k) for k in LATENCY_SPEC} == LATENCY_SPEC


@pytest.mark.parametrize("argv", [(), ("--preset", "balanced"), ("--preset", "throughput")])
def test_speculation_is_off_in_the_other_builds(argv):
    c = _cfg(*argv)
    assert c.spec_method is None and not c.spec_max_batch and not c.mtp_kv_only and not c.dyn_mtp_kv
    assert c.spec_synth_al is None and c.plain_decode and c.spec_page_buckets is None and not c.mtp_chunk_q


def test_an_explicit_argument_wins_over_a_preset():
    c = _cfg("--preset", "latency", "--host-cache-gb", "12", "--no-mtp-kv-only", "--plain-decode")
    assert (c.host_cache_gb, c.mtp_kv_only, c.plain_decode) == (12.0, False, True)


@pytest.mark.parametrize(
    ("argv", "reason"),
    [
        (("--spec-max-batch", "4", "--spec-bucket-max", "4"), "spec_max_batch 4 < the largest decode bucket 8"),
        (
            ("--max-running", "12", "--state-seats", "12", "--max-num-seqs", "16"),
            "the running cap 12 > spec_max_batch 8",
        ),
        (("--spec-max-batch", "8", "--state-spec-seats", "8"), "state_spec_seats is set"),
        (("--decode-buckets", "1,2,4,8,12", "--max-running", "8"), "spec_max_batch 8 < the largest decode bucket 12"),
    ],
)
def test_each_reason_against_no_plain_decode_trips_on_its_own(argv, reason):
    c = _cfg("--preset", "latency", *argv)
    why = mr.plain_none_refusal(c)
    assert why is not None and reason in why, why
    with pytest.raises(ValueError, match="no plain decode buckets"):
        mr.family_buckets(c, c.resolved_decode_batch_buckets())


def test_no_plain_decode_without_dynamic_speculation_is_refused():
    c = _cfg("--preset", "latency", "--spec-max-batch", "0", "--spec-bucket-max", "0", "--no-dyn-mtp-kv")
    why = mr.plain_none_refusal(c)
    assert why is not None and "no dynamic speculation" in why


def test_the_latency_build_has_no_plain_launch_path():
    c = _cfg("--preset", "latency")
    assert mr.plain_none_refusal(c) is None
    assert mr.family_buckets(c, c.resolved_decode_batch_buckets()) == ((), (1, 2, 4, 6, 8))
    with pytest.raises(RuntimeError, match="no plain decode buckets"):
        mr.pick_family(3, (), "plain decode")
    assert mr.pick_family(3, (4, 8), "plain decode") == 4
    with pytest.raises(RuntimeError, match="exceeds its warmed buckets"):
        mr.pick_family(9, (4, 8), "plain decode")


def test_dynamic_plain_mode_cannot_start_under_the_running_cap():
    """LLMEngine._dyn_want_plain is False at every decoding count the latency build admits (at most its running cap,
    8), and True one above it (the detector can trip)."""
    from trnrt.engine.engine import LLMEngine
    from trnrt.engine.request import Status

    cfg = _cfg("--preset", "latency")

    def stub(n):
        reqs = [SimpleNamespace(status=Status.RUNNING, is_decoding=True) for _ in range(n)]
        return SimpleNamespace(
            scheduler=SimpleNamespace(running=reqs),
            cfg=cfg,
            _dyn_plain=False,
            runner=SimpleNamespace(state=SimpleNamespace(mixed_seats=False)),
        )

    assert not any(LLMEngine._dyn_want_plain(stub(n)) for n in range(cfg.max_running + 1))
    assert LLMEngine._dyn_want_plain(stub(cfg.max_running + 1))


def test_the_engines_plain_decode_launches_are_known():
    """Structural: the engine's plain decode launch is _launch's alone, and ModelRunner.decode picks its bucket through
    pick_family over plain_buckets. A new plain launch path fails this test, so it gets the same review."""
    import ast
    import inspect

    from trnrt.engine import engine as E

    sites = set()
    for fn in ast.walk(ast.parse(inspect.getsource(E))):
        if isinstance(fn, ast.FunctionDef):
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "decode"
                    and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "runner"
                ):
                    sites.add(fn.name)
    assert sites == {"_launch"}, sites
    cls = next(
        n for n in ast.parse(inspect.getsource(mr)).body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner"
    )
    dec = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "decode")
    calls = [n for n in ast.walk(dec) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "pick_family"]
    assert calls and ast.unparse(calls[0].args[1]) == "self.plain_buckets"


def test_the_synthetic_acceptance_comes_from_the_flag_or_the_environment(monkeypatch):
    monkeypatch.delenv("TRNRT_SPEC_SYNTH_AL", raising=False)
    assert _cfg("--preset", "latency").spec_synth_al is None
    assert _cfg("--preset", "latency", "--spec-synth-al", "2.5").spec_synth_al == 2.5
    monkeypatch.setenv("TRNRT_SPEC_SYNTH_AL", "2.518506")
    assert _cfg("--preset", "latency").spec_synth_al == 2.518506
    assert _cfg("--preset", "latency", "--spec-synth-al", "2.0").spec_synth_al == 2.0
    with pytest.raises(ValueError, match="needs spec_method"):
        _cfg()  # the variable is set but speculation is off: refused, not silently ignored


@pytest.mark.parametrize(
    ("argv", "why"),
    [
        (("--spec-max-batch", "4"), "spec_max_batch needs spec_method"),
        (("--spec-page-buckets", "1024,4096"), "spec_page_buckets needs spec_method"),
        (("--mtp-kv-only",), "mtp_kv_only needs spec_method"),
        (("--spec-method", "mtp", "--page-size", "96"), "power-of-two page size"),
        (("--spec-method", "mtp", "--spec-synth-al", "3.5"), r"in \[1, spec_k \+ 1\]"),
        (("--spec-method", "mtp", "--spec-bucket-max", "4"), "need spec_max_batch"),
        (("--spec-method", "mtp", "--state-spec-seats", "4"), "needs spec_max_batch"),
        (("--spec-method", "mtp", "--dyn-mtp-kv"), "needs spec_max_batch"),
        (("--spec-method", "mtp", "--mtp-kv-only"), "needs mtp_chunk_q"),
        (("--spec-method", "mtp", "--spec-max-batch", "4", "--state-spec-seats", "3"), "must cover spec_max_batch 4"),
        (("--spec-method", "mtp", "--spec-max-batch", "4", "--state-spec-seats", "99"), "fit the"),
        (("--state-seats", "4"), "below the running cap 24"),
        (("--state-seats", "99"), "above max_num_seqs"),
        (("--no-plain-decode",), "plain_decode False .* needs spec_method"),
    ],
)
def test_an_inconsistent_combination_is_refused(argv, why, monkeypatch):
    monkeypatch.delenv("TRNRT_SPEC_SYNTH_AL", raising=False)
    with pytest.raises(ValueError, match=why):
        _cfg(*argv)


def test_the_valid_neighbours_of_the_refused_combinations_build():
    """Speculation's neighbours on the latency build (the speculative one), with plain decode back for the rows above a
    smaller speculation limit; a plain build's own."""
    spec = ("--preset", "latency", "--plain-decode")
    ok = (
        (*spec, "--spec-max-batch", "4", "--state-spec-seats", "4"),
        (*spec, "--spec-max-batch", "4", "--spec-bucket-max", "4", "--plain-bucket-min", "4"),
        (*spec, "--spec-synth-al", "3.0"),
        (*spec, "--mtp-chunk-q", "64", "--mtp-kv-only"),
        ("--state-seats", "28"),
        ("--preset", "latency", "--plain-decode"),
    )
    for argv in ok:
        _cfg(*argv)


@pytest.mark.parametrize("build", ["balanced", "throughput"])
def test_speculation_on_a_row_class_build_is_refused_and_names_the_latency_build(build):
    with pytest.raises(ValueError, match="--preset latency"):
        _cfg("--preset", build, "--spec-method", "mtp")


def test_the_sequence_parallel_gather_is_xla_or_nki():
    from trnrt.config import EngineConfig

    assert EngineConfig(model_path="m", sp_gather="nki").sp_gather == "nki"
    with pytest.raises(ValueError, match="'xla' or 'nki'"):
        EngineConfig(model_path="m", sp_gather="rdma")


@pytest.fixture
def kernel_routes(monkeypatch):
    """Whatever set_decode_routes writes is undone after the test."""
    from trnrt.kernels import dense_half, dsa_satwf, sp_gather
    from trnrt.models import decoder, hybrid, mla

    for mod, name in (
        (hybrid, "KDA_SHELL_MAXB"),
        (hybrid, "MOE_HALF_BELOW"),
        (mla, "LAYER_MK_MAXB"),
        (mla, "CP_INDEX_ROWS"),
        (sp_gather, "MODE"),
        (sp_gather, "TP_EXPERTS"),
        (dsa_satwf, "SEL2G"),
        (dense_half, "SKIP_T"),
        (decoder, "MTP_CHUNK_Q"),
        (decoder, "DYN_MTP_KV"),
    ):
        monkeypatch.setattr(mod, name, getattr(mod, name))


def test_build_shard_builds_the_mtp_layer_only_for_an_engine_that_speculates(monkeypatch, kernel_routes):
    """load_model gets mtp=True only with spec_method mtp: a plain engine never builds or loads the checkpoint's MTP layer, so
    its graphs (a decode call returning only its output) are the plain engine's."""
    import json
    import os

    from trnrt.engine import engine as E
    from trnrt.models.config import ModelConfig

    with open(os.path.join(os.path.dirname(__file__), "fixtures", "GLM-5.3-Flash-config.json")) as f:
        mcfg = ModelConfig.from_config_dict(json.load(f))
    seen = {}

    class Stop(Exception):
        pass

    def fake_load(*a, **k):
        seen.update(k)
        raise Stop

    monkeypatch.setattr(E, "load_model", fake_load)
    monkeypatch.setattr(E.ModelConfig, "from_pretrained", classmethod(lambda cls, p: mcfg))
    for argv, want in (((), False), (("--preset", "latency"), True)):
        seen.clear()
        cfg = _cfg(*argv, "--device", "cpu", "--tp", "1")
        with pytest.raises(Stop):
            E.build_shard(cfg, "m", 8)
        assert seen["mtp"] is want, argv
