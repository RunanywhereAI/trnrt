"""The two named batch configurations beside the served defaults (trnrt/__main__.py PRESETS, --max-running):

- every default is the served configuration's: the engine arguments, EngineConfig's own defaults, and the model
  modules' row caps on a fresh import (the decode kernel routes the served graphs were captured with);
- --preset batch12 sets exactly its documented values and nothing else; an argument given explicitly wins over it;
  capture accepts it, so `compile capture -- --preset batch12 ...` and the server read the same configuration;
- engine.build_shard hands the configuration's row caps to the model modules (set_decode_routes) before anything else,
  on every rank; the defaults leave the served values, the preset's raise them;
- out-of-range values are refused;
- the README documents both configurations.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import os
import subprocess
import sys

import pytest

from trnrt import __main__ as cli
from trnrt.config import EngineConfig
from trnrt.engine import engine as E
from trnrt.models import hybrid, mla

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The served configuration's values of everything the presets and --max-running touch.
SERVED = dict(max_num_seqs=8, decode_batch_buckets=(1, 2, 4, 6, 8), max_running=None, kda_shell_max_rows=8,
              dsa_layer_max_rows=8, ffn_half_below=16, dsa_index_rows=0)
# batch12 as the README documents it, in EngineConfig's names.
BATCH12 = dict(max_num_seqs=16, decode_batch_buckets=(1, 2, 4, 6, 8, 12), max_running=12, kda_shell_max_rows=12,
               dsa_layer_max_rows=16, ffn_half_below=17, dsa_index_rows=8)


def _diff(a: EngineConfig, b: EngineConfig) -> dict:
    return {f.name: getattr(b, f.name) for f in dataclasses.fields(a) if getattr(a, f.name) != getattr(b, f.name)}


def test_the_defaults_are_the_served_configuration():
    cfg = cli.engine_config(["--model", "m"])
    assert {k: getattr(cfg, k) for k in SERVED} == SERVED
    base = EngineConfig(model_path="m")
    assert {k: getattr(base, k) for k in SERVED if k not in ("max_num_seqs", "decode_batch_buckets")} == \
        {k: v for k, v in SERVED.items() if k not in ("max_num_seqs", "decode_batch_buckets")}


def test_the_model_modules_import_with_the_served_row_caps():
    """A fresh import (another process: a test above may have set the routes in this one): the caps the served graphs
    were traced with, which EngineConfig's defaults restate."""
    code = ("from trnrt.models import hybrid, mla; "
            "print(hybrid.KDA_SHELL_MAXB, mla.LAYER_MK_MAXB, hybrid.MOE_HALF_BELOW, mla.CP_INDEX_ROWS, mla.SHELL_MAXB)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT,
                         env=dict(os.environ, PYTHONPATH=ROOT))
    assert out.stdout.split() == ["8", "8", "16", "0", "6"]
    base = EngineConfig(model_path="m")
    assert (base.kda_shell_max_rows, base.dsa_layer_max_rows, base.ffn_half_below, base.dsa_index_rows) == (8, 8, 16, 0)


def test_batch12_sets_exactly_its_values():
    plain = cli.engine_config(["--model", "m"])
    assert _diff(plain, cli.engine_config(["--model", "m", "--preset", "batch12"])) == BATCH12
    assert _diff(plain, cli.engine_config(["--model", "m", "--preset=batch12"])) == BATCH12
    assert sorted(cli.PRESETS) == ["batch12"]


def test_an_explicit_argument_wins_over_the_preset():
    cfg = cli.engine_config(["--model", "m", "--max-running", "4", "--preset", "batch12", "--decode-buckets", "1,2"])
    assert cfg.max_running == 4 and cfg.decode_batch_buckets == (1, 2)
    assert cfg.max_num_seqs == 16 and cfg.kda_shell_max_rows == 12  # the rest still the preset's


def test_batch4_is_the_served_graphs_with_a_running_cap():
    """--max-running 4 changes nothing but the cap: every other field is the served configuration's."""
    assert _diff(cli.engine_config(["--model", "m"]), cli.engine_config(["--model", "m", "--max-running", "4"])) == \
        {"max_running": 4}


def test_capture_takes_the_preset_and_the_settings():
    from trnrt.compile.farm import check_engine_args

    check_engine_args(["--model", "m", "--preset", "batch12"])
    check_engine_args(["--model", "m", "--max-running", "4", "--kda-shell-max-rows", "12", "--dsa-layer-max-rows",
                       "16", "--ffn-half-below", "17", "--dsa-index-rows", "8"])
    with pytest.raises(SystemExit):
        check_engine_args(["--model", "m", "--preset", "batch16"])


@pytest.mark.parametrize("argv", [["--max-running", "0"], ["--kda-shell-max-rows", "0"], ["--dsa-index-rows", "-1"],
                                  ["--ffn-half-below", "0"], ["--dsa-layer-max-rows", "0"]])
def test_the_command_line_refuses_out_of_range_values(argv):
    with pytest.raises(SystemExit):
        cli.parse_engine_args(["--model", "m", *argv])


@pytest.mark.parametrize("kw", [dict(max_running=0), dict(kda_shell_max_rows=0), dict(dsa_layer_max_rows=0),
                                dict(ffn_half_below=0), dict(dsa_index_rows=-1)])
def test_engine_config_refuses_out_of_range_values(kw):
    with pytest.raises(ValueError):
        EngineConfig(model_path="m", **kw)


@pytest.fixture
def routes(monkeypatch):
    """Whatever set_decode_routes writes is undone after the test."""
    for mod, name in ((hybrid, "KDA_SHELL_MAXB"), (hybrid, "MOE_HALF_BELOW"), (mla, "LAYER_MK_MAXB"),
                      (mla, "CP_INDEX_ROWS")):
        monkeypatch.setattr(mod, name, getattr(mod, name))


def _caps():
    return hybrid.KDA_SHELL_MAXB, mla.LAYER_MK_MAXB, hybrid.MOE_HALF_BELOW, mla.CP_INDEX_ROWS


def test_set_decode_routes_moves_the_caps_the_graphs_read(routes):
    E.set_decode_routes(cli.engine_config(["--model", "m", "--preset", "batch12"]))
    assert _caps() == (12, 16, 17, 8)
    assert [hybrid.kda_shell_rows(t) for t in (8, 12, 13)] == [True, True, False]
    assert [mla._mk_rows_on(t) for t in (8, 12, 16, 17)] == [True, True, True, False]
    assert [hybrid._mk4_rows_on(t) for t in (12, 17)] == [True, False]
    assert [mla._shell_rows_on(t) for t in (6, 8)] == [True, False]  # the plain shell's cap is not a setting
    E.set_decode_routes(cli.engine_config(["--model", "m"]))  # a later engine with the defaults inherits nothing
    assert _caps() == (8, 8, 16, 0)
    assert [hybrid.kda_shell_rows(t) for t in (8, 9)] == [True, False]


def test_build_shard_sets_the_routes_first():
    """Every rank's build_shard (the engine's ranks and a capture's) sets the routes before anything else it does."""
    f = next(n for n in ast.parse(inspect.getsource(E)).body if isinstance(n, ast.FunctionDef) and n.name == "build_shard")
    body = [s for s in f.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    assert ast.unparse(body[0]) == "set_decode_routes(cfg)"


def test_the_readme_documents_both_configurations():
    with open(os.path.join(ROOT, "README.md")) as f:
        readme = f.read()
    assert "--preset batch12" in readme and "--max-running 4" in readme
    for flag, value in (("--max-num-seqs", "16"), ("--decode-buckets", "1,2,4,6,8,12"), ("--max-running", "12"),
                        ("--kda-shell-max-rows", "12"), ("--dsa-layer-max-rows", "16"), ("--ffn-half-below", "17"),
                        ("--dsa-index-rows", "8")):
        assert flag in readme and value in readme, flag
