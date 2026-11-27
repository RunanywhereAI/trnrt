"""benchmarks/accuracy/needle.py: the needle check's verdicts against a stand-in server, and its HTTP transport.

The stand-in models what the check depends on: documents of pages, a device pool that evicts the least recently used
document's tail pages to a host tier, restores of the evicted suffix, the host-tier counters on /metrics (or none, as
the server today), /v1/stats with or without the benchmark mode, and a model that reads the facts out of its prompt.

- With the counters, as the check was first written: a correct model passes and every needle crossed a restore; a wrong
  code at a depth whose page came back from the host fails, naming it; a needle that never crossed a restore is
  NO-RESULT, never PASS and never FAIL.
- Without them: the answers alone are judged. A correct model passes and the verdict says that the restore was not
  observable; a wrong code after the churn fails.
- Refused before any request: the benchmark mode (on any engine of the server's /v1/stats) and a /v1/stats that does
  not answer. A cold miss and a request error are NO-RESULT.
- The smoke's ladder finds a request profile on a small document before anything long is sent.
- The transport reads the server's own metric names (built by trnrt/metrics.py, not restated here), reads nothing from a
  server without /metrics, and sends one session id with every request.

    pytest tests/test_benchmarks_needle.py -q
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from trnrt.metrics import spec_lines, spec_stats

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("bench_needle", ROOT / "benchmarks" / "accuracy" / "needle.py")
assert _spec and _spec.loader
N = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(N)

PAGE = 64
CFG = {
    "depths": [0.1, 0.5, 0.9],
    "page": PAGE,
    "seed": 7,
    "run_id": "t",
    "context_words": 3000,
    "churn": 6,
    "evict_factor": 40.0,
    "margin_pages": 2,
    "separate": False,
    "smoke_words": 300,
}
NEEDLE = re.compile(r"The access code for vault (\w+) is (\d{8})\.")


def asked(prompt: str) -> list[str]:
    return re.findall(r"vaults \(([^)]*)\)", prompt)[-1].split(", ")


class Fake:
    """The server. A document of P pages is on the device (its first `dev` pages), on the host (the rest, evicted from
    the least recently used document, its tail first), or new."""

    def __init__(
        self,
        capacity_docs=1.4,
        wrong_after_restore=None,
        wrong_always=False,
        cold_wrong=False,
        counters=True,
        never_evict=False,
        boom_at=None,
        think_mode=None,
        wrong_when_cached=False,
        stats_body=None,
        stats_down=False,
        spec=False,
    ):
        self.think_mode, self.log, self.wrong_when_cached = think_mode, [], wrong_when_cached
        self.cap_tokens = None
        self.capacity_docs = capacity_docs
        self.docs: dict[str, dict] = {}  # key -> {total, dev, tick, seen}
        self.tick = 0
        self.saved = self.restored = 0
        self.wrong_after_restore, self.wrong_always, self.cold_wrong = wrong_after_restore, wrong_always, cold_wrong
        self.counters, self.never_evict, self.spec = counters, never_evict, spec
        self.boom_at, self.n_asks = boom_at, 0
        self.stats_body = stats_body if stats_body is not None else engine_stats(None)
        self.stats_down = stats_down
        self.flipped = 0

    def stats(self):
        if self.stats_down:
            raise ConnectionRefusedError("connection refused")
        return self.stats_body

    def metrics(self):
        out = {"num_requests_running": 0.0}
        if self.counters:
            out["hicache_pages_saved_total"] = float(self.saved)
            out["hicache_pages_restored_total"] = float(self.restored)
        if self.spec:  # 10 verifies, 20 draft tokens and 15 accepted per request: accept length 2.5
            out["spec_decode_num_drafts_total"] = 10.0 * self.n_asks
            out["spec_decode_num_draft_tokens_total"] = 20.0 * self.n_asks
            out["spec_decode_num_accepted_tokens_total"] = 15.0 * self.n_asks
        return out

    def _make_room(self, need, keep):
        if self.never_evict:
            return
        cap = int(self.cap_tokens)
        used = sum(d["dev"] for d in self.docs.values())
        for key, d in sorted(self.docs.items(), key=lambda kv: kv[1]["tick"]):  # least recently used first
            if used + need <= cap:
                break
            if key == keep or d["dev"] == 0:
                continue
            drop = min(d["dev"], used + need - cap)
            d["dev"] -= drop
            used -= drop
            self.saved += drop

    def _smoke(self, prompt, extra):
        found = dict(NEEDLE.findall(prompt))
        text = "\n".join(f"{n}: {found.get(n, '00000000')}" for n in asked(prompt))
        if self.think_mode == "never":
            text = ""
        elif self.think_mode == "low_only":  # the template opens a reasoning block: only low effort answers in budget
            text = text if extra.get("reasoning_effort") == "low" else ""
        elif self.think_mode == "no_effort_server":
            if "reasoning_effort" in extra:
                raise RuntimeError("HTTP 400: the model's chat template does not use reasoning_effort")
            text = text if extra.get("max_tokens", 96) >= 4096 else ""
        return {"text": text, "prompt_tokens": int(len(prompt.split()) * 1.3), "cached_tokens": 0}

    def ask(self, prompt, extra=None):
        self.n_asks += 1
        extra = extra or {}
        key = prompt.split("\n\n", 1)[0]
        self.log.append((key, dict(extra), len(prompt.split())))
        if self.boom_at is not None and self.n_asks == self.boom_at:
            raise ConnectionResetError("server went away")
        if key.startswith("Smoke"):  # the smoke's document: answered by the thinking mode, outside the cache model
            return self._smoke(prompt, extra)
        tokens = int(len(prompt.split()) * 1.3)
        total = tokens // PAGE
        if self.cap_tokens is None:
            self.cap_tokens = int(total * self.capacity_docs)
        d = self.docs.setdefault(key, {"total": total, "dev": 0, "tick": 0})
        d["total"] = max(d["total"], total)
        on_device = d["dev"]
        restored_now = 0
        if on_device < d["total"]:
            seen_before = d.get("seen", False)
            self._make_room(d["total"] - on_device, key)
            if seen_before:  # evicted earlier: the missing suffix comes back from the host
                restored_now = d["total"] - on_device
                self.restored += restored_now
            d["dev"] = d["total"]
        d["seen"] = True
        self.tick += 1
        d["tick"] = self.tick
        cached = (d["total"] if restored_now == 0 and on_device == d["total"] else on_device + restored_now) * PAGE
        if "Summarize" in prompt.split("\n\n")[-1]:
            return {"text": "Archive.", "prompt_tokens": tokens, "cached_tokens": cached}
        found = dict(NEEDLE.findall(prompt))
        sents = re.split(r"(?<=\.) ", key)
        lines = []
        for n in asked(prompt):
            code = found.get(n, "00000000")
            idx = next(i for i, s in enumerate(sents) if s.startswith(f"The access code for vault {n} "))
            depth = idx / len(sents)
            crossed_restore = restored_now > 0 and restored_now >= d["total"] - depth * d["total"]
            planted = self.wrong_after_restore is not None and abs(depth - self.wrong_after_restore) < 0.06
            if (
                self.cold_wrong
                or self.wrong_always
                or (self.wrong_when_cached and cached > 0)
                or (planted and crossed_restore)
            ):
                code = "12345678"
                self.flipped += 1
            lines.append(f"{n}: {code}")
        return {"text": "\n".join(lines), "prompt_tokens": tokens, "cached_tokens": cached}


class Late(Fake):
    """Right on the cold asks, wrong on every later question, whatever the cache did (ask 1 is the smoke, 2 the cold
    question, 3 the device hit)."""

    def ask(self, prompt, extra=None):
        out = super().ask(prompt, extra)
        if self.n_asks > 3 and "Answer from the document" in prompt and not prompt.startswith("Smoke"):
            out["text"] = "Alder: 12345678\nBrindle: 12345678\nCormorant: 12345678"
        return out


def drive(fake, **over):
    return N.run(fake.ask, fake.metrics, fake.stats, {**CFG, **over})


def engine_stats(synth_al):
    """One engine's /v1/stats gauges with the speculation block the server builds (trnrt/metrics.py spec_stats):
    synth_al None is real acceptance."""
    cfg = SimpleNamespace(spec_method="mtp", spec_k=2, spec_synth_al=synth_al, spec_max_batch=0)
    engine = SimpleNamespace(
        cfg=cfg, spec_proposed=0, spec_accepted=0, spec_verifies=0, spec_pos_reached=[], spec_pos_accepted=[]
    )
    return {"gauges": {"kv_pages_used": 1, "spec": spec_stats(engine)}}


def router_stats(*synth_als):
    """The router's /v1/stats over engines with these lengths (None: real acceptance)."""
    return {
        "engines": [{"url": f"http://e{i}", "up": True, "stats": engine_stats(al)} for i, al in enumerate(synth_als)]
    }


def test_the_documents_share_no_prefix_hold_the_needles_at_their_depths_and_the_churn_ones_hold_none():
    doc, needles = N.build(7, 3000, [0.1, 0.5, 0.9], N.tag("Archive volume one", CFG), True)
    sents = doc.split(". ")
    assert len(needles) == 3 and len({c for _, _, c in needles}) == 3 and len({n for _, n, _ in needles}) == 3
    for d, name, code in needles:
        pos = next(i for i, s in enumerate(sents) if f"vault {name} is {code}" in s)
        assert abs(pos / len(sents) - d) < 0.03, (d, pos / len(sents))
    other, none = N.build(8, 3000, [0.1, 0.5, 0.9], N.tag("Unrelated ledger 0-0", CFG), False)
    assert none == [] and "access code" not in other and doc[:30] != other[:30]
    # Another run's copy of the same document differs from its first sentence on, so it shares no cached prefix.
    again, _ = N.build(7, 3000, [0.1, 0.5, 0.9], N.tag("Archive volume one", {"run_id": "u"}), True)
    assert again.split(". ", 1)[0] != doc.split(". ", 1)[0] and again.split(". ", 1)[1] == doc.split(". ", 1)[1]


def test_the_parser_counts_a_code_only_under_its_own_vaults_name():
    needles = [(0.1, "Alder", "11111111"), (0.5, "Brindle", "22222222"), (0.9, "Cormorant", "33333333")]
    assert N.parse("Alder: 11111111\nBrindle: 22222222\nCormorant: 33333333", needles) == {
        "Alder": True,
        "Brindle": True,
        "Cormorant": True,
    }
    # Right codes under the wrong vaults.
    assert N.parse("Alder: 22222222\nBrindle: 11111111\nCormorant: 33333333", needles) == {
        "Alder": False,
        "Brindle": False,
        "Cormorant": True,
    }
    # A vault that is not answered is wrong; codes without names count for nothing.
    assert N.parse("Alder: 11111111", needles) == {"Alder": True, "Brindle": False, "Cormorant": False}
    assert not any(N.parse("the codes are 11111111, 22222222, 33333333", needles).values())


def test_with_the_counters_a_correct_model_passes_and_every_needle_crossed_a_restore():
    f = Fake()
    r = drive(f)
    assert r["verdict"] == "PASS" and r["host_tier_observed"] is True, r
    # Not vacuous: the pages did come back from the host.
    assert f.restored >= r["doc_pages"] * 0.9 and all(r["crossed"].values()) and len(r["crossed"]) == 3
    assert r["revisit"][0]["cached_tokens"] > 0 and r["churned"] >= 1


def test_separate_mode_gives_each_needle_a_restore_of_its_own():
    r = drive(Fake(), separate=True)
    assert r["verdict"] == "PASS", r
    assert len(r["revisit"]) == 3
    assert all(len(x["needles"]) == 1 and x["pages_restored"] >= r["doc_pages"] * 0.9 for x in r["revisit"])


def test_PLANT_with_the_counters_a_wrong_code_at_depth_half_after_a_restore_fails_naming_it():
    f = Fake(wrong_after_restore=0.5)
    r = drive(f)
    assert f.flipped >= 1, "the plant never fired: vacuous"
    assert r["verdict"] == "FAIL" and "0.5" in r["why"] and "right cold" in r["why"], r
    assert all(all(c["ok"].values()) for c in r["cold"])


def test_PLANT_a_partial_restore_that_missed_depth_point_one_is_no_result_but_a_wrong_code_that_crossed_fails():
    # One churn document: the tail 70% of ours is evicted and restored, its head stays on the device.
    r = drive(Fake(capacity_docs=1.3), churn=1)
    assert r["verdict"] == "NO-RESULT" and "depth 0.1" in r["why"], r
    assert r["crossed"] == {"0.1": False, "0.5": True, "0.9": True}
    f2 = Fake(capacity_docs=1.3, wrong_after_restore=0.9)
    r2 = drive(f2, churn=1)
    assert f2.flipped >= 1 and r2["verdict"] == "FAIL" and "0.9" in r2["why"] and "0.1" in r2["why"], r2
    # A wrong answer planted where no restore reached never fires, and the verdict does not pretend otherwise.
    f3 = Fake(capacity_docs=1.3, wrong_after_restore=0.1)
    r3 = drive(f3, churn=1)
    assert f3.flipped == 0 and r3["verdict"] == "NO-RESULT"


def test_with_the_counters_a_model_wrong_after_a_revisit_that_restored_nothing_is_no_result():
    r = drive(Late(never_evict=True))
    assert r["verdict"] == "NO-RESULT" and "did not cross a restore" in r["why"] and r["pages_restored"] == 0, r


def test_without_the_counters_a_correct_model_passes_on_its_answers_and_says_the_restore_was_not_observable():
    f = Fake(counters=False)
    r = drive(f)
    assert r["verdict"] == "PASS" and r["host_tier_observed"] is False, r
    assert "not observable" in r["why"] and "only the answers were judged" in r["why"]
    assert f.restored >= r["doc_pages"] * 0.9  # the stand-in did restore; the check only could not see it
    assert r["churned"] == CFG["churn"]  # no counter to stop a round early: every churn document is sent
    assert "crossed" not in r and "pages_restored" not in r["revisit"][0]
    assert f"{r['revisit'][0]['cached_tokens']} of {r['revisit'][0]['prompt_tokens']}" in r["why"]


def test_PLANT_without_the_counters_a_wrong_code_after_the_churn_fails_naming_its_depth():
    f = Fake(counters=False, wrong_after_restore=0.5)
    r = drive(f)
    assert f.flipped >= 1, "the plant never fired: vacuous"
    assert r["verdict"] == "FAIL" and "0.5" in r["why"] and "right cold" in r["why"] and "not observable" in r["why"]
    assert "churn's eviction" in r["why"]  # the device hit was right: what failed came after the churn
    # Without the counters, a wrong revisit fails whatever the cache did (with them it was NO-RESULT, above).
    r = drive(Late(never_evict=True, counters=False))
    assert r["verdict"] == "FAIL", r


def test_the_benchmark_mode_is_refused_before_any_request():
    # an engine in the mode, a router over two, a router with one engine in it and one not
    for body in (engine_stats(2.5), router_stats(2.5, 3.0), router_stats(None, 2.5)):
        f = Fake(stats_body=body)
        r = drive(f)
        assert r["verdict"] == "NO-RESULT" and r["why"].startswith("refused:") and "benchmark mode" in r["why"], r
        assert f.log == [] and f.n_asks == 0
    r = drive(Fake(stats_body=router_stats(None, None)))  # control: real acceptance runs
    assert r["verdict"] == "PASS", r


def test_a_health_that_does_not_answer_a_cold_miss_and_a_request_error_are_no_result():
    f = Fake(stats_down=True)
    r = drive(f)
    assert r["verdict"] == "NO-RESULT" and "/v1/stats did not answer" in r["why"] and f.n_asks == 0, r
    f = Fake(stats_body={"status": "ok", "workers": ["http://e0"]})  # a router's /health shows no engine's mode
    r = drive(f)
    assert r["verdict"] == "NO-RESULT" and "cannot be read" in r["why"] and f.n_asks == 0, r
    r = drive(Fake(cold_wrong=True))
    assert r["verdict"] == "NO-RESULT" and "cold" in r["why"] and "wrong" in r["why"]
    r = drive(Fake(boom_at=3))
    assert r["verdict"] == "NO-RESULT" and "ConnectionResetError" in r["why"] and "seconds" in r


def test_the_cli_writes_the_verdict_file_and_exits_by_the_verdict(tmp_path, capsys):
    cases = (
        (Fake(), 0, "PASS", "restored"),
        (Fake(wrong_after_restore=0.5), 1, "FAIL", "restored"),
        (Fake(counters=False), 0, "PASS", "host restore not observable"),
        (Fake(counters=False, wrong_after_restore=0.5), 1, "FAIL", "host restore not observable"),
        (Fake(stats_body=engine_stats(2.5)), 2, "NO-RESULT", "refused"),
        (Fake(cold_wrong=True), 2, "NO-RESULT", "cold"),
    )
    for i, (fake, rc, word, said) in enumerate(cases):
        vf = tmp_path / f"v-{i}"
        argv = ["--verdict", str(vf), "--context-words", "3000", "--page", str(PAGE), "--churn", "6", "--run-id", "t"]
        got = N.main(argv, transport=(fake.ask, fake.metrics, fake.stats))
        out = capsys.readouterr().out
        assert got == rc and vf.read_text().split()[0] == word and said in out, (i, out)
        assert json.loads((tmp_path / "needle.json").read_text())["verdict"] == word


def test_the_cli_refuses_a_depth_outside_the_document_before_any_request(tmp_path):
    f = Fake()
    for depths in ("0.1,1.5", "-0.1"):
        argv = ["--verdict", str(tmp_path / "v"), "--depths", depths]
        assert N.main(argv, transport=(f.ask, f.metrics, f.stats)) == 2
    assert f.n_asks == 0 and not (tmp_path / "v").exists()


def test_the_verdict_file_is_written_even_when_out_cannot_be(tmp_path, capsys):
    f = Fake(counters=False)
    blocker = tmp_path / "file"
    blocker.write_text("x")  # --out under a regular file cannot be created
    vf = tmp_path / "v"
    argv = ["--verdict", str(vf), "--context-words", "3000", "--page", str(PAGE), "--churn", "6", "--run-id", "t"]
    assert N.main([*argv, "--out", str(blocker / "sub")], transport=(f.ask, f.metrics, f.stats)) == 0
    assert vf.read_text().split()[0] == "PASS" and "needle.json not written" in capsys.readouterr().err


def test_the_cli_builds_its_transport_for_the_router_with_one_session_and_refuses_abbreviations(tmp_path, monkeypatch):
    seen = []

    def transport(*args):
        seen.append(args)
        f = Fake(counters=False)
        return f.ask, f.metrics, f.stats

    monkeypatch.setattr(N, "make_transport", transport)
    argv = ["--verdict", str(tmp_path / "v"), "--context-words", "3000", "--page", str(PAGE), "--churn", "6"]
    assert N.main([*argv, "--run-id", "r1"]) == 0
    server, api, _model, _timeout, session = seen[0]
    assert server == "http://127.0.0.1:8000" and api == "chat" and session == "needle-r1"
    assert N.main(argv) == 0 and seen[1][4] != seen[0][4]  # a random run id by default
    with pytest.raises(SystemExit):
        N.main(["--verd", str(tmp_path / "v")])


def test_the_transport_reads_usage_the_servers_metric_names_and_health_and_sends_the_session(monkeypatch):
    engine = SimpleNamespace(
        cfg=SimpleNamespace(spec_method="mtp", spec_max_batch=0),
        spec_verifies=10,
        spec_proposed=20,
        spec_accepted=15,
        spec_pos_accepted=[8, 7],
    )
    body = "\n".join(
        [
            "# TYPE trnrt:hicache_pages_saved_total counter",
            "trnrt:hicache_pages_saved_total 12",
            "trnrt:hicache_pages_restored_total 3",
            *spec_lines(engine),  # the server's own spec_decode lines
            'trnrt:request_success_total{finished_reason="stop"} 2',
            "vllm:num_requests_running 9",
            "other 5",
        ]
    ).encode()
    calls = []

    def fake_http(url, payload, timeout):
        calls.append((url, payload))
        if url.endswith("/metrics"):
            return body
        if url.endswith("/v1/stats"):
            return json.dumps(engine_stats(None)).encode()
        return json.dumps(
            {
                "choices": [{"message": {"content": "hi"}, "text": "yo"}],
                "usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 64}},
            }
        ).encode()

    monkeypatch.setattr(N, "http", fake_http)
    ask, metrics, stats = N.make_transport("http://s:1/", "chat", "m", 5.0, "needle-t")
    got = ask("p")
    assert {k: v for k, v in got.items() if k != "raw"} == {"text": "hi", "prompt_tokens": 100, "cached_tokens": 64}
    assert got["raw"]["content"] == "hi" and calls[0][0] == "http://s:1/v1/chat/completions"
    assert calls[0][1]["temperature"] == 0 and calls[0][1]["model"] == "m" and calls[0][1]["session_id"] == "needle-t"
    ask, _, _ = N.make_transport("http://s:1", "completions", None, 5.0, "needle-t")
    assert ask("p")["text"] == "yo" and calls[1][0] == "http://s:1/v1/completions"
    assert "model" not in calls[1][1] and calls[1][1]["session_id"] == "needle-t"
    m = metrics()
    assert m == {
        "hicache_pages_saved_total": 12.0,
        "hicache_pages_restored_total": 3.0,
        "spec_decode_num_drafts_total": 10.0,
        "spec_decode_num_draft_tokens_total": 20.0,
        "spec_decode_num_accepted_tokens_total": 15.0,
    }
    assert set(N.SPEC) <= set(m)  # every speculative counter the check reads is one the server writes
    assert stats() == engine_stats(None)
    assert "max_tokens" not in calls[0][1]  # every question's budget comes from its profile, each churn's is 1


def test_the_speculative_counters_growth_gives_the_acceptance_length():
    before = {"spec_decode_num_drafts_total": 10.0, "spec_decode_num_accepted_tokens_total": 15.0}
    after = {"spec_decode_num_drafts_total": 20.0, "spec_decode_num_accepted_tokens_total": 35.0}
    assert N.spec_delta(before, after)["accept_length"] == 3.0
    assert N.spec_delta({}, {}) == {}  # no speculation, or no /metrics
    r = drive(Fake(spec=True))
    assert r["verdict"] == "PASS" and r["cold"][0]["spec"]["accept_length"] == 2.5, r


def test_the_metrics_of_a_server_without_metrics_are_empty_and_other_errors_raise(monkeypatch):
    def fake_http(url, payload, timeout):
        code = 404 if "router" in url else 500
        raise urllib.error.HTTPError(url, code, "x", {}, io.BytesIO(b'{"detail": "x"}'))

    monkeypatch.setattr(N, "http", fake_http)
    _, metrics, _ = N.make_transport("http://router:1", "chat", None, 5.0, None)
    assert metrics() == {}  # the router of `trnrt serve`: no counters, so only the answers are judged
    _, metrics, _ = N.make_transport("http://engine:1", "chat", None, 5.0, None)
    with pytest.raises(urllib.error.HTTPError):
        metrics()


def test_an_answer_lost_to_reasoning_is_what_the_smoke_ladder_catches():
    """GLM-5.3-Flash's chat template always opens a reasoning block and max_tokens counts it, so a short budget can come
    back with no content. The smoke finds a profile that answers on a small document first."""
    f = Fake(think_mode="low_only")
    doc, needles = N.build(1, 300, [0.5], "Smoke volume", True)
    assert f.ask(doc + N.question(needles))["text"] == ""  # without a profile: empty
    f = Fake(think_mode="low_only")
    r = drive(f)
    assert r["verdict"] == "PASS", r
    assert r["smoke"]["profile"] == N.PROFILES[0] and r["smoke"]["failures"] == []
    longs = [(k, e) for k, e, w in f.log if not k.startswith("Smoke") and w > 1000]
    # The questions carry the profile; the churn needs only its prefill.
    assert longs and all(e.get("reasoning_effort") == "low" or e.get("max_tokens") == 1 for _, e in longs)
    assert f.log[0][0].startswith("Smoke") and not any(k.startswith("Smoke") for k, _, _ in f.log[1:])


def test_a_server_that_refuses_the_effort_falls_through_to_the_long_answer_profile():
    r = drive(Fake(think_mode="no_effort_server"))
    assert r["verdict"] == "PASS" and r["smoke"]["profile"] == N.PROFILES[1], r
    assert len(r["smoke"]["failures"]) == 1 and "HTTP 400" in r["smoke"]["failures"][0]


def test_PLANT_no_profile_answers_is_no_result_and_nothing_long_is_sent():
    f = Fake(think_mode="never")
    r = drive(f)
    assert r["verdict"] == "NO-RESULT" and "Nothing long was sent" in r["why"] and r["smoke"]["profile"] is None
    assert len(f.log) == len(N.PROFILES) and all(k.startswith("Smoke") for k, _, _ in f.log), f.log


def test_the_transport_merges_the_profile_reads_reasoning_when_content_is_empty_and_names_an_http_error(monkeypatch):
    bodies = []

    def fake_http(url, payload, timeout):
        bodies.append(payload)
        if payload.get("reasoning_effort") == "none":
            detail = b'{"detail": "reasoning_effort must be one of low, high, max"}'
            raise urllib.error.HTTPError(url, 400, "bad", {}, io.BytesIO(detail))
        return json.dumps(
            {"choices": [{"message": {"content": "", "reasoning_content": "Alder: 1"}}], "usage": {"prompt_tokens": 7}}
        ).encode()

    monkeypatch.setattr(N, "http", fake_http)
    ask, _, _ = N.make_transport("http://s:1", "chat", None, 5.0, None)
    out = ask("p", {"reasoning_effort": "low", "max_tokens": 1536})
    assert bodies[0]["reasoning_effort"] == "low" and bodies[0]["max_tokens"] == 1536 and bodies[0]["temperature"] == 0
    assert "session_id" not in bodies[0]
    assert out["text"] == "Alder: 1"  # content empty: the reasoning text is judged
    with pytest.raises(RuntimeError, match=r"HTTP 400.*must be one of"):
        ask("p", {"reasoning_effort": "none"})


def test_the_device_hit_diagnostic_says_whether_the_failure_is_the_host_tiers():
    r = drive(Fake(wrong_after_restore=0.5))  # wrong only after a host restore: the device hit at once is right
    assert r["device_hit"]["cached_tokens"] > 0 and all(r["device_hit"]["ok"].values())
    assert r["verdict"] == "FAIL" and "host restore path is implicated" in r["why"]
    r = drive(Fake(wrong_when_cached=True))  # wrong on any prefix-cache hit, device or host
    assert not all(r["device_hit"]["ok"].values())
    assert r["verdict"] == "FAIL" and "not specific to a host restore" in r["why"]
    assert all(drive(Fake())["device_hit"]["ok"].values())  # a correct model is right on the device hit too
