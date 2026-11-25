"""benchmarks/: the points and builds tables, the two scorers, the acceptance-mode check, and every script's dry run.

The scorers are checked against hand-computed values on small synthetic runs; the scripts by their --dry-run output, which
prints the exact command a run would execute. Nothing here starts a server or a client.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "benchmarks"
GOLDEN = 2.518506


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


points = _load("bench_points", B / "points.py")
agentx_score = _load("bench_agentx_score", B / "agentx" / "score.py")
fixed_score = _load("bench_fixed_score", B / "fixed_seq" / "score.py")
mode = _load("bench_acceptance_mode", B / "accuracy" / "acceptance_mode.py")


def test_every_point_has_a_build_whose_arguments_the_engine_accepts():
    from trnrt.__main__ import parse_engine_args

    pts = points.points()
    assert pts, "points.csv is empty"
    for name, p in pts.items():
        a = parse_engine_args(["--model", "/m", *p["engine_args"].split()])
        assert a.preset in ("latency", "balanced", "throughput"), name
        if p["build"] == "latency-golden":
            assert a.preset == "latency" and a.spec_synth_al == pytest.approx(GOLDEN), name
        else:
            assert a.spec_synth_al is None, name


def test_the_agentx_points_follow_the_builds_bands():
    """AgentX points at 1-8 users run the latency build at the golden length (InferenceX scores speculative decoding at a
    fixed acceptance length), 9-16 the balanced build, 17 and up the throughput build: --preset auto's bands."""
    from trnrt.__main__ import resolve_preset

    for name, p in points.points().items():
        if p["benchmark"] != "agentx":
            continue
        c = int(p["concurrency"])
        assert resolve_preset("auto", c) == p["build"].removesuffix("-golden"), name
        assert (p["build"] == "latency-golden") == (c <= 8), name


def test_fixed_length_points_run_the_benchmark_mode_only_where_named():
    """fixed-off-* decode without drafts, fixed-on-* with real acceptance, fixed-golden-* at the golden length."""
    for name, p in points.points().items():
        if p["benchmark"] == "fixed_seq":
            assert (p["build"] == "latency-golden") == name.startswith("fixed-golden-"), name
            assert (p["build"] == "latency") == name.startswith("fixed-on-"), name
            assert (p["build"] in ("balanced", "throughput")) == name.startswith("fixed-off-"), name


def _rec(start_s, end_s, isl, osl, ttft_ms, e2e_ms, itl_ms, phase="profiling", error=None):
    r = {
        "metadata": {
            "benchmark_phase": phase,
            "request_start_ns": int(start_s * 1e9),
            "request_end_ns": int(end_s * 1e9),
        },
        "metrics": {
            "input_sequence_length": {"value": isl},
            "output_sequence_length": {"value": osl},
            "time_to_first_token": {"value": ttft_ms},
            "request_latency": {"value": e2e_ms},
            "inter_token_latency": {"value": itl_ms},
        },
    }
    if error:
        r["error"] = error
    return r


def test_agentx_score_on_a_small_run(tmp_path):
    recs = [
        _rec(0, 5, 999, 9, 1, 1, 1, phase="warmup"),  # dropped: warmup
        _rec(10, 30, 1000, 100, 500, 20000, 20),
        _rec(20, 50, 2000, 200, 1500, 30000, 40),
        _rec(30, 70, 3000, 300, 1000, 40000, 30),
        _rec(40, 41, 10, 0, 0, 0, 0, error={"type": "InvalidInferenceResultError"}),  # dropped: error
    ]
    (tmp_path / "profile_export.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n{half a line")
    s = agentx_score.score(tmp_path, 8, {"sa": 0.58, "cb": 2.235}, "t")
    assert (s["profiled"], s["total"], s["warmup_dropped"], s["error_dropped"]) == (3, 5, 1, 1)
    assert s["errors"] == {"InvalidInferenceResultError": 1}
    assert s["duration_s"] == 60.0 and s["requests_per_min"] == 3.0
    assert s["total_tok_s_per_chip"] == round(6600 / 60 / 8, 1)
    # ITL 20 / 30 / 40 ms: p90 = 38 ms by linear interpolation, so P90 interactivity = 1 / 0.038
    assert s["itl_p90_ms"] == 38.0 and s["p90_interactivity_tok_s_per_user"] == round(1 / 0.038, 3)
    assert s["ttft_p50_s"] == 1.0 and s["e2e_p50_s"] == 30.0
    tps = 6600 / 60 / 8
    assert s["total_tokens_per_usd_millions"] == {
        "sa": round(tps * 3600 / 0.58 / 1e6, 3),
        "cb": round(tps * 3600 / 2.235 / 1e6, 3),
    }


def test_agentx_score_refuses_a_run_without_profiled_records(tmp_path):
    (tmp_path / "profile_export.jsonl").write_text(json.dumps(_rec(0, 1, 1, 1, 1, 1, 1, phase="warmup")) + "\n")
    with pytest.raises(SystemExit):
        agentx_score.score(tmp_path, 8, {"sa": 0.58}, "t")


def test_fixed_seq_score():
    r = {
        "total_token_throughput": 800.0,
        "output_throughput": 100.0,
        "p90_tpot_ms": 40.0,
        "median_tpot_ms": 25.0,
        "median_ttft_ms": 1500.0,
        "p90_ttft_ms": 3000.0,
        "duration": 400.0,
        "num_prompts": 40,
        "completed": 39,
        "max_concurrency": 4,
        "total_input_tokens": 39 * 7400,
        "total_output_tokens": 39 * 920,
    }
    s = fixed_score.score(r, 8, {"sa": 0.58}, "t")
    assert s["total_tok_s_per_chip"] == 100.0 and s["p90_tok_s_per_user"] == 25.0 and s["median_tok_s_per_user"] == 40.0
    assert s["failed"] == 1 and s["mean_input_tokens"] == 7400
    assert s["total_tokens_per_usd_millions"] == {"sa": round(100 * 3600 / 0.58 / 1e6, 4)}


@pytest.mark.parametrize("bad", ["sa", "sa=", "=1", "sa=-1", "sa=nan", "sa=x"])
def test_a_bad_price_is_refused(bad):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        agentx_score.parse_price(bad)


def test_the_acceptance_mode_check():
    golden = {"status": "ok", "benchmark_mode": {"synthetic_acceptance_length": GOLDEN, "drafts_verified": False}}
    real = {"status": "ok"}
    assert mode.check(real, "real") is None and mode.check(golden, "golden") is None
    assert "benchmark mode" in mode.check(golden, "real")
    assert "golden length" in mode.check(real, "golden")
    other = {"benchmark_mode": {"synthetic_acceptance_length": 3.0}}
    assert mode.check(other, "golden") is not None and mode.check(other, "real") is not None
    both = {"benchmark_mode": {"synthetic_acceptance_length": [GOLDEN, 3.0]}}  # a router over engines in two modes
    assert mode.check(both, "golden") is not None


def test_the_acceptance_mode_cli_refuses_with_status_2():
    golden = json.dumps({"benchmark_mode": {"synthetic_acceptance_length": GOLDEN}})
    assert mode.main(["--expect", "real", "--health-json", golden]) == 2
    assert mode.main(["--expect", "golden", "--health-json", golden]) == 0
    assert mode.main(["--expect", "real", "--health-json", "not json"]) == 2


needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _dry(script: str, *args: str, env: dict[str, str] | None = None) -> str:
    import os

    r = subprocess.run(
        ["bash", str(B / script), "--dry-run", *args],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}", **(env or {})},
        check=False,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout


@needs_bash
@pytest.mark.parametrize("point", list(points.points()))
def test_serve_dry_run_is_one_engine_of_the_points_build(point):
    out = _dry("serve.sh", point, env={"CACHE": "/c", "NKI": "/n", "MODEL": "/m"})
    p = points.points()[point]
    assert f"serve --model /model --port 8000 --engines 1 {p['engine_args']}" in out


@needs_bash
def test_compile_dry_run_captures_each_build_then_compiles_once():
    out = _dry("compile.sh", "latency-golden", "throughput", env={"OUT": "/o"})
    caps = [ln for ln in out.splitlines() if " compile capture " in ln]
    assert len(caps) == 2
    assert caps[0].endswith("--preset latency --spec-synth-al 2.518506") and caps[1].endswith("--preset throughput")
    assert out.count(" compile compile ") == 1 and out.count(" compile strip ") == 1


@needs_bash
def test_docker_args_reach_every_container():
    out = _dry("compile.sh", "balanced", env={"OUT": "/o", "DOCKER_ARGS": "--cpuset-cpus=0-95"})
    runs = [ln for ln in out.splitlines() if ln.startswith("+ docker run")]
    assert runs and all(ln.startswith("+ docker run --rm --cpuset-cpus=0-95 ") for ln in runs)
    out = _dry(
        "serve.sh", "agentx-c16", env={"CACHE": "/c", "NKI": "/n", "MODEL": "/m", "DOCKER_ARGS": "--cpuset-cpus=0-95"}
    )
    assert "--ipc=host --cpuset-cpus=0-95 -v /c:/cache" in out


@needs_bash
def test_client_dry_runs_take_the_points_concurrency():
    out = _dry("agentx/client.sh", "agentx-c24", "http://h:8000", "/nonexistent/out", env={"TOKENIZER": "/t"})
    assert (
        "--concurrency 24 " in out
        and "semianalysis_cc_traces_weka_062126" in out
        and "--endpoint /v1/chat/completions" in out
    )
    out = _dry(
        "fixed_seq/client.sh", "fixed-on-c4", "http://h:8000", "/nonexistent/out", env={"TOKENIZER": "/t", "INFX": "/i"}
    )
    assert (
        "--num-prompts 40 --max-concurrency 4 " in out
        and "--num-warmups 8 " in out
        and out.rstrip().endswith("--use-chat-template)")
    )
    out = _dry(
        "fixed_seq/client.sh",
        "fixed-off-c4",
        "http://h:8000",
        "/nonexistent/out",
        env={"TOKENIZER": "/t", "INFX": "/i"},
    )
    assert "--use-chat-template" not in out
    out = _dry(
        "fixed_seq/client.sh",
        "fixed-golden-c4",
        "http://h:8000",
        "/nonexistent/out",
        env={"TOKENIZER": "/t", "INFX": "/i"},
    )
    assert "on build latency-golden" in out and out.rstrip().endswith("--use-chat-template)")


@needs_bash
def test_a_client_refuses_a_point_of_the_other_benchmark():
    r = subprocess.run(
        ["bash", str(B / "agentx" / "client.sh"), "--dry-run", "fixed-off-c4", "http://h:8000", "/nonexistent/out"],
        capture_output=True,
        text=True,
        env={"TOKENIZER": "/t", "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin"},
        check=False,
    )
    assert r.returncode == 2 and "not an AgentX point" in r.stderr


record = _load("bench_record", B / "record.py")


def test_a_result_names_no_machine():
    s = record.scrub(
        {
            "url": "http://10.1.2.3:8000/v1/chat/completions",
            "log": "client on 172.31.0.9 wrote /home/someone/runs/agentx-c8/profile_export.jsonl.",
            "tokenizer_id": "/models/GLM-5.3-Flash",
            "lnl": "2.11.0.1.0.1284+f49d8626",
            "neuronx_cc": "2.27.5334.0",
            "nested": [{"/data/x/y": "ok"}],
        }
    )
    assert s["url"] == "http://<server>/v1/chat/completions"
    assert s["log"] == "client on <ip> wrote <path>/profile_export.jsonl."
    assert s["tokenizer_id"] == "<path>/GLM-5.3-Flash"
    assert s["lnl"] == "2.11.0.1.0.1284+f49d8626" and s["neuronx_cc"] == "2.27.5334.0"  # versions keep their digits
    assert s["nested"] == [{"<path>/y": "ok"}]


def test_a_result_keeps_the_neuron_runtime_versions(tmp_path):
    # versions.sh's real output on 25 Nov: the runtime's four-part versions read as IPv4 addresses, and an earlier
    # scrub replaced every one of them with <ip>
    versions = {
        "image": {"id": "sha256:c32de67d", "trnrt_revision": "704e7b6b"},
        "image_contents": {
            "deb": {"aws-neuronx-runtime-lib": "2.34.10.0-ac18d186d", "aws-neuronx-tools": "2.32.28.0-526c2b7f6"}
        },
        "host": {"neuron_driver": "2.30.2.0"},
    }
    score = tmp_path / "score.json"
    score.write_text(json.dumps({"label": "agentx-c16", "requests_per_min": 1.0}))
    (tmp_path / "v.json").write_text(json.dumps(versions))
    out = tmp_path / "result.json"
    argv = ["--point", "agentx-c16", "--score", str(score), "--versions", str(tmp_path / "v.json"), "--commit", "abc"]
    assert record.main([*argv, "--note", "client at 10.0.0.7", "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["versions"] == versions
    assert r["note"] == "client at <ip>"  # everywhere else an address is still scrubbed


def test_a_result_that_matches_a_private_pattern_is_refused(tmp_path, monkeypatch):
    pats = tmp_path / "p.json"
    pats.write_text(json.dumps({"patterns": [["a host", r"\bboxname\b", False, "host"]]}))
    score = tmp_path / "score.json"
    score.write_text(json.dumps({"label": "ran on boxname", "requests_per_min": 1.0}))
    out = tmp_path / "result.json"
    monkeypatch.setenv("TRNRT_TRACE_PATTERNS", str(pats))
    assert record.main(["--point", "agentx-c8", "--score", str(score), "--commit", "abc", "--out", str(out)]) == 2
    assert not out.exists()
    score.write_text(json.dumps({"label": "agentx-c8", "requests_per_min": 1.0}))
    assert record.main(["--point", "agentx-c8", "--score", str(score), "--commit", "abc", "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["build"] == "latency-golden" and r["synthetic_acceptance"] is True and r["concurrency"] == 8
