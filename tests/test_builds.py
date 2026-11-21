"""The builds (trnrt/__main__.py PRESETS, BANDS, DEPRECATED, BUILD_FIELDS) and --max-running:

- every engine-argument default is the balanced build's; --preset balanced names it and changes nothing;
- each build sets exactly its documented values, row for row (a change to a build shows up here), and the
  recurrent-state pool it implies has the rows its graphs are compiled at (34, 50 and 50);
- auto (and --users alone) picks the build of the band that holds the users per engine, the band edges included;
  auto without --users, and --users beside another build, are refused;
- the names of earlier releases run the nearest build, with one warning per process;
- an argument given explicitly wins over the build; --max-running and the host cache's arguments change nothing else;
- capture accepts the builds, auto with --users, and the old names, so `compile capture -- --preset NAME ...` and the
  server read the same configuration;
- engine.build_shard hands the build's row caps to the model modules (set_decode_routes) before anything else;
- out-of-range values are refused; --moe-kernel, unset, follows the device;
- every build has its BUILD_FIELDS entry, and docs/CONFIGURATION.md documents the builds.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import logging
import os
import subprocess
import sys

import pytest

from trnrt import __main__ as cli
from trnrt.config import EngineConfig
from trnrt.engine import engine as E
from trnrt.engine import model_runner as mr
from trnrt.models import hybrid, mla

pytestmark = pytest.mark.usefixtures("decode_routes")  # set_decode_routes's writes are undone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Every build, in EngineConfig's names.
COMMON = {
    "page_size": 256,
    "max_model_len": 1_048_576,
    "kv_cache_dtype": "fp8",
    "max_prefill_tokens": 4096,
    "prefill_token_buckets": (512, 1024, 2048, 4096),
    "prefill_page_buckets": (1024, 4096),
    "prefill_page_min_rows": 2048,
    "dsa_layer_max_rows": 24,
    "ffn_half_below": 25,
    "dsa_index_rows": 8,
    "dense_half_skip_rows": (2, 18, 24),
    "sp_gather": "nki",
    "sp_gather_tp_experts": True,
    "dsa_sel2g": True,
    "tp": 32,
}
BALANCED = {
    **COMMON,
    "max_num_seqs": 32,
    "max_running": 24,
    "decode_batch_buckets": (8, 16, 24),
    "page_buckets": (1024, 3072, 4096),
    "kv_cache_gb": 4.0,
    "state_checkpoints": 24,
    "state_seats": 24,
    "state_spec_seats": None,
    "kda_shell_max_rows": 12,
    "spec_method": None,
    "spec_max_batch": 0,
    "plain_decode": True,
    "host_cache_gb": 16.0,
    "host_cache_ckpt_frac": 0.6,
    "host_cache_batch": True,
    "host_cache_save_once": True,
    "pool_ascending": True,
}
EXPECTED = {
    "latency": {
        **COMMON,
        "max_num_seqs": 16,
        "max_running": 8,
        "decode_batch_buckets": (1, 2, 4, 6, 8),
        "page_buckets": (1024, 2048, 4096),
        "kv_cache_gb": 4.0,
        "state_checkpoints": 8,
        "state_seats": 8,
        "state_spec_seats": None,
        "kda_shell_max_rows": 24,
        "spec_method": "mtp",
        "spec_k": 2,
        "spec_max_batch": 8,
        "spec_hysteresis": 1,
        "spec_bucket_max": 8,
        "plain_decode": False,
        "plain_bucket_min": 0,
        "spec_page_buckets": (1024, 2048, 4096),
        "mtp_chunk_q": 64,
        "mtp_fill_rows": 1024,
        "mtp_kv_only": True,
        "dyn_mtp_kv": True,
        "host_cache_gb": 4.0,
        "host_cache_ckpt_frac": 0.5,
        "host_cache_batch": False,
        "host_cache_save_once": False,
        "pool_ascending": False,
    },
    "balanced": BALANCED,
    "throughput": {**BALANCED, "decode_batch_buckets": (16, 24), "kv_cache_gb": 5.25},
}
# The recurrent-state pool's rows: the scratch row, the seats' rows (1 + spec_k for a seat that speculates), the
# checkpoint rows and the zero row.
STATE_ROWS = {"latency": 34, "balanced": 50, "throughput": 50}


@pytest.fixture(autouse=True)
def _no_synthetic_acceptance_from_the_shell(monkeypatch):
    monkeypatch.delenv("TRNRT_SPEC_SYNTH_AL", raising=False)


def _cfg(*argv) -> EngineConfig:
    return cli.engine_config(["--model", "m", *argv])


def _diff(a: EngineConfig, b: EngineConfig) -> dict:
    return {f.name: getattr(b, f.name) for f in dataclasses.fields(a) if getattr(a, f.name) != getattr(b, f.name)}


def test_the_builds_are_three():
    assert sorted(cli.PRESETS) == ["balanced", "latency", "throughput"]
    assert sorted(cli.BUILD_FIELDS) == sorted(cli.PRESETS)
    assert cli.DEFAULT_PRESET == "balanced"


def test_every_build_names_the_same_fields_and_each_is_engine_configs():
    """A route a build fixes without a command-line argument (BUILD_FIELDS): every build names the same fields, so no
    engine inherits one from the EngineConfig default; each name is an EngineConfig field, and an unknown one is refused."""
    names = {n: set(v) for n, v in cli.BUILD_FIELDS.items()}
    assert len({frozenset(v) for v in names.values()}) == 1, names
    known = {f.name for f in dataclasses.fields(EngineConfig)}
    assert set(cli.BUILD_FIELDS["latency"]) <= known
    for name in cli.PRESETS:
        got = cli.build_fields(name)
        assert got == cli.BUILD_FIELDS[name]
        c = _cfg("--preset", name)
        assert all(getattr(c, k) == v for k, v in got.items()), name


def test_a_build_field_engine_config_lacks_is_refused(monkeypatch):
    monkeypatch.setitem(cli.BUILD_FIELDS, "latency", {**cli.BUILD_FIELDS["latency"], "kda_shel": False})
    with pytest.raises(ValueError, match="kda_shel"):
        cli.build_fields("latency")


def test_the_kda_routes_of_each_build():
    assert {n: cli.BUILD_FIELDS[n]["kda_shell"] for n in cli.PRESETS} == {
        "latency": True,
        "balanced": False,
        "throughput": False,
    }
    for f in ("mhc_seam", "shell_mix_row_split", "state_ckpt_mid_export"):
        assert {n: cli.BUILD_FIELDS[n][f] for n in cli.PRESETS} == {
            "latency": False,
            "balanced": False,
            "throughput": True,
        }, f


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_a_build_sets_its_documented_values(name):
    c = _cfg("--preset", name)
    for field, want in EXPECTED[name].items():
        assert getattr(c, field) == want, (name, field)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_state_pool_has_the_rows_the_graphs_are_compiled_at(name):
    assert mr.state_pool_rows(_cfg("--preset", name)) == STATE_ROWS[name]


def test_the_defaults_are_the_balanced_build():
    assert _diff(_cfg(), _cfg("--preset", "balanced")) == {}
    assert _diff(_cfg(), _cfg("--preset=balanced")) == {}


def test_the_decode_families_of_each_build():
    """latency: every decode bucket a verify and no plain decode graph; the others: plain decode only."""
    got = {
        n: mr.family_buckets(c, c.resolved_decode_batch_buckets())
        for n, c in ((n, _cfg("--preset", n)) for n in EXPECTED)
    }
    assert got == {
        "latency": ((), (1, 2, 4, 6, 8)),
        "balanced": ((8, 16, 24), (8, 16, 24)),
        "throughput": ((16, 24), (16, 24)),
    }


@pytest.mark.parametrize(
    ("users", "build"),
    [(1, "latency"), (8, "latency"), (9, "balanced"), (16, "balanced"), (17, "throughput"), (64, "throughput")],
)
def test_auto_picks_the_band_of_the_users(users, build):
    assert cli.band_preset(users) == build
    for argv in (("--preset", "auto", "--users", str(users)), ("--users", str(users))):
        assert cli.parse_engine_args(["--model", "m", *argv]).preset == build
        assert _diff(_cfg("--preset", build), _cfg(*argv)) == {}


def test_auto_needs_users_and_users_needs_auto():
    with pytest.raises(SystemExit, match="needs --users"):
        _cfg("--preset", "auto")
    with pytest.raises(SystemExit, match="does not go with --preset latency"):
        _cfg("--preset", "latency", "--users", "4")
    with pytest.raises(SystemExit):
        _cfg("--users", "0")


@pytest.mark.parametrize(("old", "new"), sorted(cli.DEPRECATED.items()))
def test_an_earlier_name_runs_the_nearest_build_with_a_warning(old, new, caplog, monkeypatch):
    monkeypatch.setattr(cli, "_warned", set())
    with caplog.at_level(logging.WARNING, logger="trnrt"):
        assert _diff(_cfg("--preset", new), _cfg("--preset", old)) == {}
        _cfg("--preset", old)
    assert caplog.text.count(f"--preset {old} is deprecated: it runs the {new} build") == 1


def test_the_earlier_names_map_by_batch_size():
    assert cli.DEPRECATED == {
        "batch8": "latency",
        "mtp-batch8": "latency",
        "mtp-always": "latency",
        "batch12": "balanced",
        "mtp-batch12": "balanced",
        "mtp-batch24": "throughput",
    }


def test_an_explicit_argument_wins_over_the_build():
    cfg = _cfg("--max-running", "4", "--preset", "latency", "--decode-buckets", "1,2,4", "--host-cache-gb", "8")
    assert cfg.max_running == 4 and cfg.decode_batch_buckets == (1, 2, 4) and cfg.host_cache_gb == 8.0
    assert cfg.max_num_seqs == 16 and cfg.kda_shell_max_rows == 24  # the rest still the build's


def test_the_host_only_arguments_change_nothing_else():
    """--max-running and the host cache's arguments are host only: each moves its own field and nothing else."""
    base = _cfg()
    for argv, want in (
        (("--max-running", "16"), {"max_running": 16}),
        (("--host-cache-gb", "8"), {"host_cache_gb": 8.0}),
        (("--host-cache-ckpt-frac", "0.5"), {"host_cache_ckpt_frac": 0.5}),
        (("--no-host-cache-save-once",), {"host_cache_save_once": False}),
        (("--no-pool-ascending",), {"pool_ascending": False}),
    ):
        assert _diff(base, _cfg(*argv)) == want, argv


def test_capture_takes_the_builds_auto_and_the_old_names():
    from trnrt.compile.farm import check_engine_args

    for name in (*cli.PRESETS, *cli.DEPRECATED):
        check_engine_args(["--model", "m", "--preset", name])
    check_engine_args(["--model", "m", "--preset", "auto", "--users", "12"])
    check_engine_args(["--model", "m", "--users", "12", "--no-host-cache-save-once", "--host-cache-ckpt-frac", "0.5"])
    with pytest.raises(SystemExit):
        check_engine_args(["--model", "m", "--preset", "batch16"])


def test_the_moe_kernel_follows_the_device_unless_given():
    """Unset, --moe-kernel is EngineConfig's own choice: the NKI kernels on Neuron (and in a capture, on the meta
    device), XLA on CPU, where NKI kernels do not run; given, it wins."""
    neuron, cpu = _cfg(), _cfg("--device", "cpu")
    assert neuron.moe_kernel is None and neuron.resolved_moe_kernel() == "nki"
    assert dataclasses.replace(neuron, device="meta").resolved_moe_kernel() == "nki"
    assert cpu.resolved_moe_kernel() == "xla"
    assert _cfg("--device", "cpu", "--moe-kernel", "nki").resolved_moe_kernel() == "nki"
    assert _cfg("--moe-kernel", "xla").resolved_moe_kernel() == "xla"


@pytest.mark.parametrize(
    "argv",
    [
        ["--max-running", "0"],
        ["--kda-shell-max-rows", "0"],
        ["--dsa-index-rows", "-1"],
        ["--ffn-half-below", "0"],
        ["--dsa-layer-max-rows", "0"],
    ],
)
def test_the_command_line_refuses_out_of_range_values(argv):
    with pytest.raises(SystemExit):
        cli.parse_engine_args(["--model", "m", *argv])


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"max_running": 0}, "max_running=0: at least 1"),
        ({"kda_shell_max_rows": 0}, "kda_shell_max_rows=0: at least 1"),
        ({"dsa_layer_max_rows": 0}, "dsa_layer_max_rows=0: at least 1"),
        ({"ffn_half_below": 0}, "ffn_half_below=0: at least 1"),
        ({"dsa_index_rows": -1}, r"dsa_index_rows=-1: 0 \(one call\) or more"),
        ({"host_cache_ckpt_frac": 1.0}, r"host_cache_ckpt_frac=1.0: in \(0, 1\)"),
        ({"host_cache_ckpt_frac": 0.0}, r"host_cache_ckpt_frac=0.0: in \(0, 1\)"),
        ({"host_cache_save_once": True}, "host_cache_save_once needs host_cache_batch"),
        ({"plain_decode": False}, "plain_decode False .* needs spec_method 'mtp'"),
    ],
)
def test_engine_config_refuses_out_of_range_values(kw, message):
    with pytest.raises(ValueError, match=message):
        EngineConfig(model_path="m", **kw)


def test_engine_config_defaults_are_the_modules_own_row_caps():
    """A fresh import (another process: a test may have set the routes in this one) has the model modules' own caps,
    which EngineConfig's defaults restate; a build raises them (set_decode_routes)."""
    code = (
        "from trnrt.models import hybrid, mla; "
        "print(hybrid.KDA_SHELL_MAXB, mla.LAYER_MK_MAXB, hybrid.MOE_HALF_BELOW, mla.CP_INDEX_ROWS, mla.SHELL_MAXB)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=ROOT),
    )
    assert out.stdout.split() == ["8", "8", "16", "0", "6"]
    base = EngineConfig(model_path="m")
    assert (base.kda_shell_max_rows, base.dsa_layer_max_rows, base.ffn_half_below, base.dsa_index_rows) == (8, 8, 16, 0)


@pytest.fixture
def routes(monkeypatch):
    """Whatever set_decode_routes writes is undone after the test."""
    from trnrt.kernels import dense_half, dsa_satwf, sp_gather
    from trnrt.models import decoder

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


def test_set_decode_routes_moves_the_caps_the_graphs_read_and_a_later_engine_inherits_nothing(routes):
    from trnrt.kernels import dense_half, dsa_satwf, sp_gather
    from trnrt.models import decoder

    def caps():
        return hybrid.KDA_SHELL_MAXB, mla.LAYER_MK_MAXB, hybrid.MOE_HALF_BELOW, mla.CP_INDEX_ROWS

    def kernels():
        return sp_gather.MODE, sp_gather.TP_EXPERTS, dsa_satwf.SEL2G, dense_half.SKIP_T

    E.set_decode_routes(_cfg("--preset", "latency"))
    assert caps() == (24, 24, 25, 8) and (decoder.MTP_CHUNK_Q, decoder.DYN_MTP_KV) == (64, "1")
    assert kernels() == ("nki", True, True, frozenset((2, 18, 24)))
    E.set_decode_routes(_cfg())
    assert caps() == (12, 24, 25, 8) and (decoder.MTP_CHUNK_Q, decoder.DYN_MTP_KV) == (0, "0")
    assert kernels() == ("nki", True, True, frozenset((2, 18, 24)))
    assert [hybrid.kda_shell_rows(t) for t in (8, 12, 13)] == [True, True, False]
    assert [mla._mk_rows_on(t) for t in (16, 24, 25)] == [True, True, False]
    assert [mla._shell_rows_on(t) for t in (6, 8)] == [True, False]  # the plain shell's cap is not a setting


def test_build_shard_sets_the_routes_first():
    """Every rank's build_shard (the engine's ranks and a capture's) sets the routes before anything else it does."""
    f = next(
        n for n in ast.parse(inspect.getsource(E)).body if isinstance(n, ast.FunctionDef) and n.name == "build_shard"
    )
    body = [s for s in f.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    assert ast.unparse(body[0]) == "set_decode_routes(cfg)"


def test_the_docs_document_the_builds():
    with open(os.path.join(ROOT, "docs", "CONFIGURATION.md")) as f:
        doc = f.read()
    for s in (
        "--preset latency",
        "--preset balanced",
        "--preset throughput",
        "--preset auto",
        "--users",
        "--max-running",
    ):
        assert s in doc, s
    for old in cli.DEPRECATED:
        assert f"`{old}`" in doc, old
