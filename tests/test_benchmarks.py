"""benchmarks/: the points and builds tables, the two scorers, the acceptance-mode check, and every script's dry run.

The scorers are checked against hand-computed values on small synthetic runs; the scripts by their --dry-run output, which
prints the exact command a run would execute. Nothing here starts a server or a client.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from trnrt.metrics import spec_stats

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


def test_agentx_score_refuses_a_profiled_record_without_a_token_count(tmp_path):
    bad = _rec(20, 50, 2000, 200, 1500, 30000, 40)
    del bad["metrics"]["output_sequence_length"]  # counted as 0 it would understate the throughput, with no sign
    lines = [json.dumps(_rec(10, 30, 1000, 100, 500, 20000, 20)), json.dumps(bad)]
    (tmp_path / "profile_export.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(SystemExit, match="output_sequence_length"):
        agentx_score.score(tmp_path, 8, {"sa": 0.58}, "t")


def test_agentx_score_tolerates_only_a_half_written_last_line(tmp_path):
    good = [json.dumps(_rec(10, 30, 1000, 100, 500, 20000, 20)), json.dumps(_rec(20, 50, 2000, 200, 1500, 30000, 40))]
    (tmp_path / "profile_export.jsonl").write_text("\n".join([good[0], "{half a line", good[1]]) + "\n")
    with pytest.raises(SystemExit, match="line 2 of 3"):
        agentx_score.score(tmp_path, 8, {"sa": 0.58}, "t")
    (tmp_path / "profile_export.jsonl").write_text("\n".join([*good, "{half a line"]))
    assert agentx_score.score(tmp_path, 8, {"sa": 0.58}, "t")["total"] == 2


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


def _spec(synth_al):
    """The speculation block an MTP engine's /v1/stats carries, as the server builds it (trnrt/metrics.py)."""
    cfg = SimpleNamespace(spec_method="mtp", spec_k=2, spec_synth_al=synth_al, spec_max_batch=0)
    e = SimpleNamespace(
        cfg=cfg, spec_proposed=0, spec_accepted=0, spec_verifies=0, spec_pos_reached=[], spec_pos_accepted=[]
    )
    return spec_stats(e)


def _stats(*synth_als, missing=None):
    """A router's /v1/stats over engines with these benchmark-mode lengths (None: real acceptance); `missing` an engine
    that did not answer."""
    engines = [
        {"url": f"http://e{i}", "up": True, "stats": {"gauges": {"kv_pages_used": 1, "spec": _spec(al)}}}
        for i, al in enumerate(synth_als)
    ]
    if missing is not None:
        engines[missing] = {"url": f"http://e{missing}", "up": False, "stats": None, "error": "did not answer"}
    return {"engines": engines, "router": {"workers": [e["url"] for e in engines]}}


def test_the_acceptance_mode_check_reads_every_engine():
    assert mode.check(_stats(None), "real") is None and mode.check(_stats(GOLDEN), "golden") is None
    assert mode.check(_stats(None, None), "real") is None and mode.check(_stats(GOLDEN, GOLDEN), "golden") is None
    assert "benchmark mode" in mode.check(_stats(GOLDEN), "real")
    assert "golden length" in mode.check(_stats(None), "golden") and mode.check(_stats(3.0), "golden") is not None
    # a router over engines in two modes: refused both ways (the router's /health block named only the synthetic one)
    assert mode.check(_stats(GOLDEN, None), "golden") is not None and mode.check(_stats(GOLDEN, None), "real")
    # no speculation at all (balanced, throughput): real acceptance
    plain = {"engines": [{"url": "http://e0", "up": True, "stats": {"gauges": {"kv_pages_used": 1}}}]}
    assert mode.check(plain, "real") is None and mode.check(plain, "golden") is not None
    # one engine's own /v1/stats, and its /health (whose block is there from the start)
    assert mode.check({"gauges": {"spec": _spec(GOLDEN)}}, "golden") is None
    assert mode.check({"status": "ok"}, "real") is None
    assert mode.check({"status": "ok", "benchmark_mode": {"synthetic_acceptance_length": [GOLDEN]}}, "golden") is None


@pytest.mark.parametrize(
    "body",
    [
        [],
        None,
        "x",
        {"engines": []},
        {"engines": "x"},
        _stats(None, missing=0),
        {"gauges": None},
        {"gauges": {"spec": "x"}},
        {"gauges": {"spec": {"synth_al": "2.5"}}},
        {"gauges": {"spec": {"synth_al": True}}},
        {"status": "ok", "benchmark_mode": {"synthetic_acceptance_length": None}},
        {"status": "ok", "benchmark_mode": []},
        {"status": "ok", "workers": ["http://e0"], "benchmark_mode": {"synthetic_acceptance_length": [GOLDEN]}},
    ],
)
def test_a_body_the_mode_cannot_be_read_from_is_refused_both_ways(body):
    assert isinstance(mode.modes(body), str)
    assert "cannot be read" in mode.check(body, "real") and "cannot be read" in mode.check(body, "golden")


def test_the_acceptance_mode_cli_retries_only_what_it_cannot_read():
    golden = json.dumps(_stats(GOLDEN))
    assert mode.main(["--expect", "real", "--body-json", golden]) == 2
    assert mode.main(["--expect", "golden", "--body-json", golden]) == 0
    assert mode.main(["--expect", "real", "--body-json", "not json"]) == 2
    bodies = [OSError("refused"), _stats(GOLDEN, GOLDEN, missing=1), _stats(GOLDEN, GOLDEN)]  # starting, then up
    calls, naps = [], []

    def get(url):
        calls.append(url)
        b = bodies[min(len(calls), len(bodies)) - 1]
        if isinstance(b, Exception):
            raise b
        return b

    assert mode.main(["--expect", "golden", "--url", "http://h:8000"], get=get, sleep=naps.append) == 0
    assert len(calls) == 3 and len(naps) == 2
    calls.clear()
    bodies[:] = [_stats(None)]  # a wrong mode is refused at once, not waited on
    assert mode.main(["--expect", "golden", "--url", "http://h:8000"], get=get, sleep=naps.append) == 2
    assert len(calls) == 1
    bodies[:] = [OSError("refused")]  # never readable: refused once --wait runs out
    assert mode.main(["--expect", "real", "--url", "http://h:8000", "--wait", "0"], get=get, sleep=naps.append) == 2


needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


# The scripts' own variables: one exported in the shell that runs the tests would change the commands they print
SCRIPT_VARS = {"PORT", "IMAGE", "DOCKER_ARGS", "CPUS", "OUT", "CACHE", "NKI", "MODEL", "TOKENIZER", "INFX", "PY", "REV"}
SCRIPT_VARS |= {"AGENTX", "AGENTX_PYTHON", "VENV", "MAX_MODEL_LEN", "HF_TOKEN"}


def _dry(script: str, *args: str, env: dict[str, str] | None = None, cwd: Path | None = None) -> str:
    import os

    base = {k: v for k, v in os.environ.items() if k not in SCRIPT_VARS}
    r = subprocess.run(
        ["bash", str(B / script), "--dry-run", *args],
        capture_output=True,
        text=True,
        env={**base, "PATH": f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}", **(env or {})},
        cwd=cwd,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout


@needs_bash
def test_an_exported_script_variable_does_not_reach_a_dry_run(monkeypatch):
    monkeypatch.setenv("PORT", "9999")
    monkeypatch.setenv("IMAGE", "other")
    out = _dry("serve.sh", "agentx-c16", env={"CACHE": "/c", "NKI": "/n", "MODEL": "/m"})
    assert "--port 8000 " in out and " trnrt serve " in out


@needs_bash
def test_the_fixed_length_client_resolves_a_relative_tokenizer_before_it_changes_directory(tmp_path):
    (tmp_path / "tok").mkdir()
    out = _dry(
        "fixed_seq/client.sh",
        "fixed-off-c1",
        "http://h:8000",
        "/nonexistent/out",
        env={"TOKENIZER": "tok", "INFX": "/i"},
        cwd=tmp_path,
    )
    assert f"--tokenizer {tmp_path.resolve()}/tok" in out


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


@contextlib.contextmanager
def _server(routes: dict[str, object]):
    """A local HTTP server answering GET path -> JSON body, for the scripts that query a server before their run."""
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = routes.get(self.path)
            self.send_response(404 if body is None else 200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if body is not None:
                self.wfile.write(json.dumps(body).encode())

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()


FAKE_LM_EVAL = """#!/usr/bin/env bash
# lm-eval's stand-in: writes a results file under --output_path, resolved against the current directory as lm-eval does
if [ "$1" = "-m" ]; then
  while [ $# -gt 0 ]; do [ "$1" = "--output_path" ] && out=$2; shift; done
  mkdir -p "$out/glm"
  printf '{"results": {"gsm8k": {"exact_match,strict-match": %s, "exact_match,flexible-extract": %s}}}' \\
    "$FAKE_STRICT" "$FAKE_STRICT" > "$out/glm/results_1.json"
  exit 0
fi
exec python3 "$@"
"""


@needs_bash
@pytest.mark.skipif(shutil.which("curl") is None, reason="needs curl")
def test_gsm8k_takes_a_relative_out_dir_and_exits_by_the_bar(tmp_path):
    import os

    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text(FAKE_LM_EVAL)
    (venv / "bin" / "python").chmod(0o755)
    base = {k: v for k, v in os.environ.items() if k not in SCRIPT_VARS}
    env = {**base, "VENV": str(venv), "PATH": f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}"}
    routes = {"/v1/stats": {"gauges": {"kv_pages_used": 0}}, "/v1/models": {"data": [{"id": "glm"}]}}
    with _server(routes) as url:
        for run, strict, rc, word in (("runs/pass", "0.97", 0, "PASS"), ("runs/fail", "0.90", 1, "FAIL")):
            r = subprocess.run(
                ["bash", str(B / "accuracy" / "gsm8k.sh"), url, run, "4"],
                capture_output=True,
                text=True,
                env={**env, "FAKE_STRICT": strict},
                cwd=tmp_path,
                check=False,
            )
            assert r.returncode == rc, r.stdout + r.stderr
            assert f"({word} at 0.94)" in r.stdout and (tmp_path / run / "glm" / "results_1.json").exists()
        # a server in the benchmark mode is refused before lm-eval runs
        routes["/v1/stats"] = {"gauges": {"spec": _spec(GOLDEN)}}
        r = subprocess.run(
            ["bash", str(B / "accuracy" / "gsm8k.sh"), url, "runs/refused", "4"],
            capture_output=True,
            text=True,
            env={**env, "FAKE_STRICT": "0.97"},
            cwd=tmp_path,
            check=False,
        )
        assert r.returncode == 2 and "benchmark mode" in r.stderr and not (tmp_path / "runs" / "refused").exists()


preflight = _load("bench_preflight", B / "preflight.py")


def _info(build, **over):
    """/server_info of one engine of `build`, as serve/native.py shapes it (the fields the preflight reads)."""
    e = preflight.expected(points.builds()[build]["engine_args"])
    return {"dp_size": 1, "tp_size": 32, "preset": e["preset"], "spec_synth_al": e["spec_synth_al"], **over}


@pytest.mark.parametrize("build", list(points.builds()))
def test_the_preflight_accepts_one_engine_of_each_build_and_refuses_the_others(build):
    args = points.builds()[build]["engine_args"]
    assert preflight.mismatch(_info(build), args) is None
    for other in points.builds():
        if other != build:
            assert preflight.mismatch(_info(other), args) is not None, (build, other)
    assert "2 engines" in preflight.mismatch(_info(build, dp_size=2), args)  # trnrt serve's default: two engines
    assert "tp_size" in preflight.mismatch(_info(build, tp_size=64), args)
    assert preflight.mismatch(None, args) is not None


def test_the_preflight_waits_for_a_starting_server_then_reads_every_engines_mode():
    calls, naps = [], []
    golden = _info("latency-golden")

    def get(url, path):
        calls.append(path)
        if len(calls) == 1:
            raise OSError("connection refused")  # still starting
        return golden if path == "/server_info" else _stats(GOLDEN)

    assert preflight.main(["agentx-c4", "http://h:8000"], get=get, sleep=naps.append) == 0
    assert calls == ["/server_info", "/server_info", "/v1/stats"] and len(naps) == 1
    calls.clear()
    # the right configuration but an engine on real acceptance (/v1/stats asks every engine): refused
    assert preflight.main(
        ["agentx-c4", "http://h:8000"], get=lambda u, p: golden if p == "/server_info" else _stats(None)
    )
    # the wrong build: refused at once, without waiting or reading the mode
    seen = []

    def wrong(url, path):
        seen.append(path)
        return _info("balanced")

    assert preflight.main(["agentx-c4", "http://h:8000"], get=wrong, sleep=naps.append) == 2 and seen == [
        "/server_info"
    ]
    assert preflight.main(["no-such-point", "http://h:8000"], get=wrong) == 2


def test_the_preflight_refuses_a_server_that_never_answers():
    def down(url, path):
        raise OSError("connection refused")

    assert preflight.main(["agentx-c16", "http://h:8000", "--wait", "0"], get=down) == 2


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
