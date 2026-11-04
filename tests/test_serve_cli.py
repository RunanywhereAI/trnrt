"""`python -m trnrt serve` and `python -m trnrt engine` start-up (trnrt/__main__.py): engines run with the compile-cache
assertion unless --allow-compile, a cache miss ends start-up with its likely cause, serve starts its engines one at a
time and stops them all past --startup-timeout, and the argv serve hands each engine parses there. Fake engine
processes and a fake /health: nothing is loaded or served."""

import os
import signal
import sys
import time

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("uvicorn")

import trnrt.__main__ as cli  # noqa: E402

MISS = (
    "NEURON_LIBTORCH_ASSERT_CACHE_HIT: compile cache miss for key 0123abcd. All graphs must be pre-compiled when "
    "assert_cache_hit is enabled."
)


def _last(cmd: list[str], flag: str) -> str:
    """The value argparse takes for `flag` in `cmd`: the last (an engine's command repeats serve's --port, --host)."""
    return cmd[len(cmd) - cmd[::-1].index(flag)]


class FakeEngine:
    """An engine process `serve` started: its /health answers 200 from its `ready`-th poll on, once `ready_s` seconds
    have passed since its start (never if ready is None); or the process exits with `exit_status` then instead."""

    def __init__(self, host, cmd, env, ready=1, exit_status=None, ready_s=0.0):
        self.host, self.cmd, self.env = host, cmd, env
        self.port = int(_last(cmd, "--port"))
        self.ready, self.exit_status, self.ready_s, self.t0 = ready, exit_status, ready_s, time.monotonic()
        self.polls, self.returncode, self.signals = 0, None, []

    def poll(self):
        return self.returncode

    def health(self) -> int:
        self.polls += 1
        if self.ready is None or self.polls < self.ready or time.monotonic() - self.t0 < self.ready_s:
            raise httpx.ConnectError("connection refused")  # still loading and warming: no server yet
        if self.exit_status is not None:
            self.returncode = self.exit_status
            raise httpx.ConnectError("connection refused")
        self.host.events.append(("healthy", self.port))
        return 200

    def send_signal(self, sig):
        self.signals.append(sig)
        self.returncode = -sig

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -signal.SIGKILL


class FakeHost:
    def __init__(self):
        self.events, self.engines, self.plan, self.router, self.router_kw = [], [], [], None, None


def _unset(monkeypatch, name):
    monkeypatch.setenv(name, "")  # recorded, so the variable is restored (here: unset again) after the test
    monkeypatch.delenv(name)


@pytest.fixture
def host(monkeypatch):
    """subprocess.Popen, httpx.get and uvicorn.run faked; `host.plan[i]` holds engine i's FakeEngine arguments."""
    import uvicorn

    h = FakeHost()

    def popen(cmd, env=None, preexec_fn=None):
        e = FakeEngine(h, cmd, env, **(h.plan[len(h.engines)] if len(h.engines) < len(h.plan) else {}))
        h.engines.append(e)
        h.events.append(("start", e.port))
        return e

    def get(url, timeout=None):
        port = int(url.split(":")[2].split("/")[0])
        for e in h.engines:
            if e.port == port and e.returncode is None:
                return httpx.Response(e.health())
        raise httpx.ConnectError("connection refused")

    def run(app, host=None, port=None, **kw):
        h.router, h.router_kw = app, kw
        h.events.append(("router", port))

    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.setattr(cli, "HEALTH_POLL_S", 0.0)
    monkeypatch.setattr(cli.signal, "signal", lambda *a: None)  # serve's SIGTERM handler: not in the test process
    _unset(monkeypatch, cli.CACHE_HIT_ENV)
    return h


def test_serve_starts_one_engine_at_a_time_with_the_cache_hit_assertion(host):
    host.plan = [{"ready": 3}, {"ready": 2}]
    assert cli.cmd_serve(["--model", "m", "--port", "9000"]) == 0
    # engine 1 starts only once engine 0 answers /health, and the router only once both do
    assert host.events == [("start", 9001), ("healthy", 9001), ("start", 9002), ("healthy", 9002), ("router", 9000)]
    for i, e in enumerate(host.engines):
        assert e.env[cli.CACHE_HIT_ENV] == "1"
        assert _last(e.cmd, "--core-base") == str(32 * i) and _last(e.cmd, "--host") == "127.0.0.1"
        assert "--engines" not in e.cmd and "--startup-timeout" not in e.cmd and "--allow-compile" not in e.cmd
        assert e.signals == [signal.SIGTERM]  # the router returned: serve stops its engines


def test_serve_gives_the_router_its_keep_alive_and_not_the_engines(host):
    """uvicorn's default keep-alive (5 s) is under a client pool's (aiohttp: 15 s): the router keeps 75 s by default."""
    assert cli.cmd_serve(["--model", "m"]) == 0
    assert host.router_kw["timeout_keep_alive"] == 75
    host.engines.clear()
    assert cli.cmd_serve(["--model", "m", "--timeout-keep-alive", "120"]) == 0
    assert host.router_kw["timeout_keep_alive"] == 120
    assert all("--timeout-keep-alive" not in e.cmd for e in host.engines)  # serve's own flag: an engine would refuse it


def test_router_main_passes_its_keep_alive(monkeypatch):
    import uvicorn

    from trnrt.serve import router

    got = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: got.update(kw))
    for extra, want in (([], 75), (["--timeout-keep-alive", "5"], 5)):
        monkeypatch.setattr(sys, "argv", ["router", "--worker-urls", "http://127.0.0.1:9", *extra])
        router.main()
        assert got["timeout_keep_alive"] == want


def test_allow_compile_turns_the_assertion_off_in_the_engines(host, monkeypatch):
    monkeypatch.setenv(cli.CACHE_HIT_ENV, "1")  # the flag decides, whatever the caller's environment holds
    assert cli.cmd_serve(["--model", "m", "--allow-compile"]) == 0
    assert [e.env[cli.CACHE_HIT_ENV] for e in host.engines] == ["0", "0"]
    assert all("--allow-compile" in e.cmd for e in host.engines)  # the engine command sets it from the flag too
    monkeypatch.setenv(cli.CACHE_HIT_ENV, "0")
    host.engines.clear()
    assert cli.cmd_serve(["--model", "m"]) == 0  # and without the flag it is on
    assert [e.env[cli.CACHE_HIT_ENV] for e in host.engines] == ["1", "1"]


def test_an_engine_that_never_answers_stops_every_engine_at_the_startup_timeout(host, monkeypatch):
    host.plan = [{"ready": None}]  # engine 0 never answers /health
    monkeypatch.setattr(cli, "HEALTH_POLL_S", 0.01)
    t = time.monotonic()
    with pytest.raises(SystemExit) as e:
        cli.cmd_serve(["--model", "m", "--port", "9000", "--startup-timeout", "0.2"])
    assert time.monotonic() - t < 10
    assert "--startup-timeout" in str(e.value.code), e.value.code
    assert [x for x in host.events if x[0] == "start"] == [("start", 9001)]  # engine 1 never started
    assert host.engines[0].signals == [signal.SIGTERM] and host.router is None


def test_the_startup_timeout_counts_from_each_engines_start(host, monkeypatch):
    """Each engine has the whole timeout: each is healthy 0.4 s after its start, within a 0.7 s timeout that one budget
    for both (0.8 s) would overrun."""
    host.plan = [{"ready_s": 0.4}, {"ready_s": 0.4}]
    monkeypatch.setattr(cli, "HEALTH_POLL_S", 0.01)
    assert cli.cmd_serve(["--model", "m", "--startup-timeout", "0.7"]) == 0
    assert host.router is not None and len(host.engines) == 2


def test_a_cache_miss_ends_serve_with_its_cause(host):
    host.plan = [{"ready": 2, "exit_status": cli.CACHE_MISS_STATUS}]
    with pytest.raises(SystemExit) as e:
        cli.cmd_serve(["--model", "m"])
    msg = str(e.value.code)
    assert "engine 0" in msg and "not in the compile cache" in msg and "--allow-compile" in msg, msg
    assert len(host.engines) == 1 and host.router is None
    # control: another failure is reported as what it is, not as a cache miss
    host.engines.clear()
    host.plan = [{"ready": 2, "exit_status": 1}]
    with pytest.raises(SystemExit) as e:
        cli.cmd_serve(["--model", "m"])
    assert "exited during start-up (status 1)" in str(e.value.code) and "cache" not in str(e.value.code)


def test_serve_reports_an_engine_its_host_refused(host):
    """An engine whose host checks refused it (engine/start_checks.py) exits with START_REFUSED_STATUS: serve says so,
    pointing at the engine's own message, and starts nothing after it."""
    host.plan = [{"ready": 2, "exit_status": cli.START_REFUSED_STATUS}]
    with pytest.raises(SystemExit) as e:
        cli.cmd_serve(["--model", "m"])
    msg = str(e.value.code)
    assert "engine 0 did not start" in msg and "check of the host refused it" in msg and "cache" not in msg, msg
    assert len(host.engines) == 1 and host.router is None


def test_an_engine_that_dies_while_the_next_starts_ends_serve(host, monkeypatch):
    host.plan = [{"ready": 1}, {"ready": None}]
    fake_get = httpx.get

    def get(url, timeout=None):
        if len(host.engines) == 2 and host.engines[1].polls == 3:
            host.engines[0].returncode = 1  # engine 0, healthy, exits while engine 1 is still warming
        return fake_get(url, timeout=timeout)

    monkeypatch.setattr(httpx, "get", get)
    with pytest.raises(SystemExit) as e:
        cli.cmd_serve(["--model", "m", "--startup-timeout", "5"])
    assert "engine 0 exited during start-up (status 1)" in str(e.value.code), e.value.code
    assert host.engines[1].signals == [signal.SIGTERM] and host.router is None


class _FakeLLMEngine:
    seen_env: list = []
    cfgs: list = []
    fail = None

    def __init__(self, cfg):
        import os

        type(self).seen_env.append(os.environ.get(cli.CACHE_HIT_ENV))
        type(self).cfgs.append(cfg)

    def warmup(self):
        if type(self).fail is not None:
            raise type(self).fail
        return {"seconds": 0.0, "graphs": 0}


def _wrapped_miss() -> RuntimeError:
    """LNL's miss as torch.compile raises it: wrapped, the miss on the cause only."""
    try:
        raise RuntimeError(MISS)
    except RuntimeError as inner:
        try:
            raise RuntimeError("backend='neuron_libtorch' raised an error") from inner
        except RuntimeError as outer:
            return outer


@pytest.fixture
def engine(monkeypatch):
    import trnrt.engine.engine as eng
    import trnrt.serve.api as api

    _FakeLLMEngine.seen_env, _FakeLLMEngine.cfgs, _FakeLLMEngine.fail = [], [], None
    monkeypatch.setattr(eng, "LLMEngine", _FakeLLMEngine)
    monkeypatch.setattr(api, "serve", lambda *a, **k: None)
    monkeypatch.setattr(api, "build_app", lambda *a, **k: None)
    _unset(monkeypatch, cli.CACHE_HIT_ENV)  # cmd_engine sets it in this process: restored after the test
    monkeypatch.setenv("NEURON_LIBTORCH_CACHE_ROOT", "/data/cache")
    return _FakeLLMEngine


ENGINE_ARGV = ["--model", "m", "--device", "cpu", "--tp", "1"]


def test_engine_asserts_cache_hits_and_reports_a_miss_with_its_cause(engine, capsys):
    engine.fail = _wrapped_miss()
    assert cli.cmd_engine(ENGINE_ARGV) == cli.CACHE_MISS_STATUS
    assert engine.seen_env == ["1"]  # set before the engine (and its spawned ranks) started
    err = capsys.readouterr().err
    assert MISS in err and "NEURON_LIBTORCH_CACHE_ROOT=/data/cache" in err and "--allow-compile" in err, err
    assert "other engine arguments" in err and "checkpoint revision" in err and "Neuron SDK" in err


def test_engine_passes_other_failures_through_and_allow_compile_turns_the_assertion_off(engine):
    engine.fail = RuntimeError("an unrelated failure")
    with pytest.raises(RuntimeError, match="unrelated"):
        cli.cmd_engine(ENGINE_ARGV)
    engine.fail = None
    assert cli.cmd_engine(ENGINE_ARGV) == 0
    assert cli.cmd_engine(ENGINE_ARGV + ["--allow-compile"]) == 0
    assert engine.seen_env == ["1", "1", "0"]


def test_an_engine_its_host_refuses_builds_nothing(engine, monkeypatch, capsys):
    """cmd_engine asks the host checks about its own cores before anything starts: a refusal is printed and ends it
    with START_REFUSED_STATUS, no engine built and no environment set for one."""
    from trnrt.engine import start_checks

    asked = []

    def refuse(device, core_base, tp):
        asked.append((device, core_base, tp))
        return "the Neuron devices of this engine's cores 32-63 (devices 8, 9) are in use by another process: pid 7"

    monkeypatch.setattr(start_checks, "refusal", refuse)
    assert cli.cmd_engine(["--model", "m", "--core-base", "32"]) == cli.START_REFUSED_STATUS
    assert asked == [("neuron", 32, 32)] and engine.cfgs == [] and engine.seen_env == []
    assert "trnrt engine: not starting: the Neuron devices of this engine's cores 32-63" in capsys.readouterr().err
    assert cli.CACHE_HIT_ENV not in os.environ  # refused before the engines' environment was set


def test_cache_miss_reads_the_chain_and_only_a_miss():
    assert cli.cache_miss(_wrapped_miss()) == MISS
    assert cli.cache_miss(RuntimeError("x\n" + MISS + "\ny")) == MISS
    assert cli.cache_miss(RuntimeError("compile failed")) is None
    # LNL's refusal of a contradictory environment names the variable but is no miss
    assert (
        cli.cache_miss(
            ValueError("NEURON_LIBTORCH_DISABLE_COMPILE_CACHE and NEURON_LIBTORCH_ASSERT_CACHE_HIT cannot both be set")
        )
        is None
    )


@pytest.fixture
def stand_in(host, engine, monkeypatch):
    """serve's engine commands run in this process: `python -m trnrt engine ARGV` as trnrt.__main__.main(ARGV), with the
    stand-in LLMEngine. An engine that parses its argv starts and answers /health; one that does not exits with
    argparse's status 2."""

    def popen(cmd, env=None, preexec_fn=None):
        assert cmd[1:4] == ["-m", "trnrt", "engine"], cmd
        e = FakeEngine(host, cmd, env)
        try:
            e.returncode = cli.main(cmd[3:]) or None
        except SystemExit as x:
            e.returncode = x.code
        host.engines.append(e)
        host.events.append(("start", e.port))
        return e

    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    return engine


def test_serve_hands_each_engine_an_argv_it_parses(stand_in):
    """The argv serve builds for each engine through cmd_engine's parser and engine_config: the `--host 127.0.0.1` serve
    appends is the engine's listen address, not an abbreviation of --host-cache-gb."""
    assert cli.cmd_serve(["--model", "m", "--device", "cpu", "--port", "9000"]) == 0
    assert [(c.host_cache_gb, c.tp_core_base) for c in stand_in.cfgs] == [(4.0, 0), (4.0, 32)]
    stand_in.cfgs.clear()
    assert cli.cmd_serve(["--model", "m", "--device", "cpu", "--host-cache-gb", "2", "--host", "0.0.0.0"]) == 0
    assert [c.host_cache_gb for c in stand_in.cfgs] == [2.0, 2.0]  # a full name still reaches the engines


def test_with_abbreviations_allowed_the_engines_fail_to_start(stand_in, monkeypatch, capsys):
    """The positive control: argparse's default (allow_abbrev=True) puts the bug back."""
    import argparse

    init = argparse.ArgumentParser.__init__

    def abbreviating(self, *a, **k):
        init(self, *a, **dict(k, allow_abbrev=True))

    monkeypatch.setattr(argparse.ArgumentParser, "__init__", abbreviating)
    with pytest.raises(SystemExit) as e:
        cli.cmd_serve(["--model", "m", "--device", "cpu"])
    assert "engine 0 exited during start-up (status 2)" in str(e.value.code), e.value.code
    assert "--host-cache-gb: invalid float value: '127.0.0.1'" in capsys.readouterr().err


def test_serve_refuses_an_abbreviated_argument(host):
    """serve passes its argv on to the engines less --engines and --startup-timeout, removed by their full names: an
    abbreviation of either (or of an engine argument) is refused at once rather than handed on."""
    for argv in (["--startup", "10"], ["--eng", "1"], ["--max-model", "4096"]):
        with pytest.raises(SystemExit) as e:
            cli.cmd_serve(["--model", "m", *argv])
        assert e.value.code == 2
    assert host.engines == []
