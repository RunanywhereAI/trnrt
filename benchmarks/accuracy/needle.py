"""Needle retrieval after a long document's cached KV was evicted and came back: a long-context accuracy check.

    python3 -I benchmarks/accuracy/needle.py --verdict PATH [--server URL] [--context-words 60000]
        [--depths 0.1,0.5,0.9] [--churn 48] [--api chat|completions] [--page 256] [--margin-pages 2] [--seed 7]
        [--run-id ID] [--model NAME] [--timeout 1800] [--out DIR] [--separate]

Why. GLM-5.3-Flash's sparse attention reads only the pools of a long context that its indexer selects. A cached page
that comes back wrong (restored from the host tier, `--host-cache-gb`) changes no output while attention does not select
it, so comparing the outputs of repeated requests cannot show it. A question about a fact on that page forces the
selection: the pool that holds the fact must be selected, and its bytes must be right.

How. Three facts ("The access code for vault <name> is <8 digits>.") are planted at the given depths of a long document.
Every document opens with the run's id (--run-id, random by default), so no earlier run shares a prefix with it, and
every request carries one `session_id`, so the router of `trnrt serve` sends them all to one engine: the churn must
evict from the pool that holds the document.
0. The server's /health must not report the benchmark mode (acceptance_mode.py). Its drafts are not verified, so its
   answers are not the model's, and the check refuses before any request.
1. SMOKE: a small document with one needle, asked cold through a ladder of request profiles. GLM-5.3-Flash's chat
   template always opens a reasoning block and `max_tokens` counts reasoning, so a short budget can come back with no
   content. The first profile that answers correctly (PROFILES: reasoning effort "low", then the template's default
   effort with a long budget) is used for every question; churn requests use max_tokens 1, since only their prefill
   matters. If no profile answers, NO-RESULT after seconds, before anything long is sent. The text judged is `content`,
   or `reasoning_content` when content is empty.
2. COLD: the document and a question for the three codes. Every code must be right, else NO-RESULT: the model cannot
   retrieve them even from a fresh prefill, so there is nothing to judge. The same question is then asked again at
   once, a device prefix hit with no host tier involved. It is reported and named in the verdict's text, and changes
   no verdict.
3. CHURN: up to --churn other documents of the same length, so the prefix cache evicts the document's pages to the host
   tier. The cache evicts least recently used leaves first, so what leaves the device is a suffix of the document.
4. REVISIT: the same question again. The evicted suffix comes back from the host tier, and the codes must still be
   right.
--separate: one question and one churn round per depth, so each needle has a restore of its own.

Judging the restore. When /metrics has the host-tier counters (COUNTERS), a needle at depth d sits on page d x P of the
document's P pages (P = prompt tokens / --page), and it crossed a restore iff the revisit restored at least
(1 - d) x P + --margin-pages pages: the 0.1 needle needs nearly the whole document back from the host, the 0.9 needle
only its last tenth. A churn round then stops early once the tier saved --evict-factor times the document's pages.
The server exports no such counters today (trnrt/metrics.py), and the router serves no /metrics at all. Then the churn
sends all --churn documents and only the answers are judged, and the verdict says that a restore from the host tier was
not observable. The revisit's cached tokens (usage.prompt_tokens_details) still show how much of its prompt came from
the prefix cache, device and host together.

Verdicts, with exit status 0, 1 and 2:
- PASS: every needle right cold and right after the revisit (with the counters: and every needle crossed a restore).
- FAIL: a needle right cold is wrong after the revisit (with the counters: a needle that crossed a restore).
- NO-RESULT: the benchmark mode, a /health that does not answer, no smoke profile answers, a cold code wrong, a request
  error, or (with the counters only) a needle that did not cross a restore: the document was still partly on the
  device, so raise --churn. The depths that did cross are listed.
The verdict file's first word is the verdict; the full JSON goes to needle.json in --out (default: the verdict file's
directory). Point --server at one engine's own port (8001 with `trnrt serve`'s default --port) to read its /metrics as
well: the growth of the speculative-decoding counters over each question is then reported. Stdlib only. The check
(`run`) takes its transport as functions, so tests/test_benchmarks_needle.py drives it without a server.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import secrets
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acceptance_mode

WORDS = (  # noqa: SIM905  one paragraph of words reads better than 79 list items
    "amber anchor basin beacon birch bridge cable canyon cedar chalk cliff cobalt copper coral creek crest dune ember "
    "fable fern field flint forge frost garnet glacier granite harbor hazel heron iron ivory jasper juniper kettle "
    "lantern larch lichen linen maple marble meadow mist moss nectar oak onyx orchard otter pebble pine plume prairie "
    "quartz quill raven reef ridge river sable saffron sage shale silver slate spruce stone summit tallow thistle "
    "timber topaz tundra umber valley velvet willow yarrow zephyr"
).split()
NAMES = ("Alder", "Brindle", "Cormorant", "Dovetail", "Elmwood", "Fenwick", "Garrison", "Hollis", "Ingleby", "Juniper")
# The host tier's counters, read from /metrics under the trnrt: prefix when the server exports them.
COUNTERS = ("hicache_pages_saved_total", "hicache_pages_restored_total")
# The speculative-decoding counters (trnrt/metrics.py spec_lines, vLLM's spec_decode names).
SPEC = ("spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total")
# The smoke's ladder: short reasoning first, then the template's default effort with a long answer budget (the repo's
# GSM8K recipe caps its output at 16,384 tokens). A field the server refuses (HTTP 400) moves to the next profile.
PROFILES = [{"reasoning_effort": "low", "max_tokens": 1536}, {"max_tokens": 16384}]
CHURN_ASK = "\n\nSummarize the first sentence in one word."
METRIC = re.compile(r"^trnrt:(\w+)\s+([0-9.eE+-]+)\s*$")


def sentence(rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(9, 17))).capitalize() + "."


def haystack(seed: int, words: int, tag: str) -> list[str]:
    """Sentences of random words, `tag` first so that no two documents share a prefix."""
    rng = random.Random(seed)  # noqa: S311  seeded document text, not cryptography
    out, n = [f"{tag}."], 1
    while n < words:
        s = sentence(rng)
        out.append(s)
        n += len(s.split())
    return out


def tag(name: str, cfg: dict) -> str:
    """A document's first sentence: its name, and the run's id when there is one."""
    return f"{name}, run {cfg['run_id']}" if cfg.get("run_id") else name


def build(seed: int, words: int, depths: list[float], name: str, plant: bool = True):
    """(document text, needles [(depth, vault name, code)]). The needles go in at the given fractions of the sentences;
    a churn document (plant False) has none."""
    sents = haystack(seed, words, name)
    rng = random.Random(seed * 7919 + 1)  # noqa: S311  seeded needles, not cryptography
    needles = []
    if plant:
        names = list(NAMES)
        rng.shuffle(names)
        for d, vault in zip(depths, names):
            needles.append((d, vault, str(rng.randint(10_000_000, 99_999_999))))
    n = len(sents)
    # From the back, so an insertion does not move the later positions.
    for d, vault, code in sorted(needles, key=lambda x: -x[0]):
        sents.insert(max(1, min(n, int(n * d))), f"The access code for vault {vault} is {code}.")
    return " ".join(sents), needles


def question(needles) -> str:
    names = ", ".join(n for _, n, _ in needles)
    return (
        "\n\nAnswer from the document above only. Give the access code of each of these vaults "
        f"({names}), one per line, as `<vault name>: <8 digits>`."
    )


def parse(text: str, needles) -> dict[str, bool]:
    """Vault name -> whether the code stated for THAT vault is right: the text from the name to the next vault name must
    contain the code, so a code under another vault's name does not count."""
    out = {}
    names = [n for _, n, _ in needles]
    for _, name, code in needles:
        i = text.find(name)
        if i < 0:
            out[name] = False
            continue
        later = [text.find(o, i + len(name)) for o in names if o != name]
        j = min([x for x in later if x >= 0] or [len(text)])
        out[name] = code in text[i:j]
    return out


def parse_metrics(text: str) -> dict[str, float]:
    """The unlabelled `trnrt:` series of a Prometheus text body, by name without the prefix. Labelled series (the
    per-position acceptance counts, the per-graph counters) are not read."""
    out = {}
    for ln in text.splitlines():
        m = METRIC.match(ln)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def spec_delta(before: dict, after: dict) -> dict:
    """The speculative-decoding counters' growth over one request: verifies, draft tokens proposed, draft tokens
    accepted, and the mean acceptance length 1 + accepted / verifies. Empty when /metrics has none (no speculation, or
    the router, which serves no /metrics)."""
    d = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in SPEC if k in after}
    if d.get("spec_decode_num_drafts_total", 0) > 0:
        d["accept_length"] = round(
            1 + d.get("spec_decode_num_accepted_tokens_total", 0) / d["spec_decode_num_drafts_total"], 4
        )
    return d


def refusal(health) -> str | None:
    """Why the check must not run on this server, or None: its /health does not answer, or it reports the benchmark
    mode (synthetic acceptance), whose text is not the model's."""
    try:
        body = health()
    except (OSError, ValueError) as e:  # urllib's errors are OSErrors; a body that is not JSON a ValueError
        return f"the server's /health did not answer: {type(e).__name__}: {str(e)[:200]}"
    why = acceptance_mode.check(body, "real")
    return f"refused: {why}" if why else None


def smoke(ask, cfg: dict) -> tuple[dict | None, list[str]]:
    """A small document with one needle, asked cold through the ladder: (the first profile that answers correctly, the
    failures seen)."""
    doc, needles = build(cfg["seed"] + 5000, cfg["smoke_words"], [0.5], tag("Smoke volume", cfg), True)
    seen = []
    for prof in cfg.get("profiles") or PROFILES:
        try:
            a = ask(doc + question(needles), prof)
        except Exception as e:  # noqa: BLE001  a refused field (HTTP 400) moves to the next profile
            seen.append(f"{prof}: {type(e).__name__}: {str(e)[:160]}")
            continue
        if all(parse(a["text"], needles).values()):
            return prof, seen
        seen.append(f"{prof}: answered {a['text'][:100]!r} (want {needles[0][2]})")
    return None, seen


def run(ask, metrics, health, cfg: dict) -> dict:
    """The check. health() -> the server's /health body; ask(prompt, extra=None) -> {text, prompt_tokens, cached_tokens,
    raw}, with `extra` merged into the request body; metrics() -> {name: value} of /metrics' unlabelled trnrt: series
    ({} where there is no /metrics). Never raises: the verdict says what happened."""
    depths, page = cfg["depths"], cfg["page"]
    res: dict = {"verdict": "NO-RESULT", "why": "", "depths": depths}
    t0 = time.time()
    try:
        why = refusal(health)
        if why:
            res["why"] = why
            return res
        m0 = metrics()
        observed = all(c in m0 for c in COUNTERS)
        res["host_tier_observed"] = observed
        prof, seen = smoke(ask, cfg)
        res["smoke"] = {"profile": prof, "failures": seen}
        if prof is None:
            res["why"] = (
                "the smoke (a small cold document, one needle) got no correct answer under any request profile: "
                + " | ".join(seen)
                + ". Nothing long was sent."
            )
            return res
        doc, needles = build(cfg["seed"], cfg["context_words"], depths, tag("Archive volume one", cfg), True)
        groups = [needles] if not cfg["separate"] else [[n] for n in needles]
        cold = []
        for g in groups:
            mb = metrics()
            a = ask(doc + question(g), prof)
            cold.append(
                {
                    "needles": [n for _, n, _ in g],
                    "ok": parse(a["text"], g),
                    "prompt_tokens": a["prompt_tokens"],
                    "cached_tokens": a["cached_tokens"],
                    "answer": a["text"][:200],
                    "raw": a.get("raw"),
                    "spec": spec_delta(mb, metrics()),
                }
            )
        pt = cold[0]["prompt_tokens"]
        doc_pages = max(1, pt // page)
        res.update(prompt_tokens=pt, doc_pages=doc_pages, cold=cold)
        wrong_cold = [n for c in cold for n, ok in c["ok"].items() if not ok]
        if wrong_cold:
            res["why"] = f"cold (a fresh prefill) the model got the code of {wrong_cold} wrong: nothing to judge by"
            return res
        # Diagnostic: the same question at once, while the document is still on the device: a prefix-cache hit with no
        # host tier in the path. If it is wrong too, the failure is not the host tier's. It changes no verdict.
        mb = metrics()
        a = ask(doc + question(groups[0]), prof)
        dh = {
            "ok": parse(a["text"], groups[0]),
            "cached_tokens": a["cached_tokens"],
            "prompt_tokens": a["prompt_tokens"],
            "answer": a["text"][:200],
            "raw": a.get("raw"),
            "spec": spec_delta(mb, metrics()),
            "same_text_as_cold": a.get("raw") is not None and a.get("raw") == cold[0].get("raw"),
        }
        res["device_hit"] = dh
        revisit: list[dict] = []
        restored_total = saved_total = churned = 0
        saved_c, restored_c = COUNTERS
        for gi, g in enumerate(groups):
            base = metrics()
            for k in range(cfg["churn"]):  # one churn round per question group
                cname = tag(f"Unrelated ledger {gi}-{k}", cfg)
                cdoc, _ = build(cfg["seed"] + 1000 * (gi + 1) + k + 1, cfg["context_words"], depths, cname, False)
                ask(cdoc + CHURN_ASK, {"max_tokens": 1})  # only the prefill matters
                churned += 1
                if observed and metrics()[saved_c] - base[saved_c] >= doc_pages * cfg["evict_factor"]:
                    break
            mid = metrics()
            a = ask(doc + question(g), prof)
            end = metrics()
            r = {
                "needles": [n for _, n, _ in g],
                "ok": parse(a["text"], g),
                "cached_tokens": a["cached_tokens"],
                "prompt_tokens": a["prompt_tokens"],
                "answer": a["text"][:200],
                "raw": a.get("raw"),
                "spec": spec_delta(mid, end),
                "same_text_as_cold": a.get("raw") is not None
                and a.get("raw") == (cold[min(gi, len(cold) - 1)] or {}).get("raw"),
            }
            if observed:
                r["pages_restored"] = end[restored_c] - mid[restored_c]
                r["pages_saved_by_churn"] = mid[saved_c] - base[saved_c]
                restored_total += r["pages_restored"]
                saved_total += r["pages_saved_by_churn"]
            revisit.append(r)
        res.update(revisit=revisit, churned=churned)
        depth_of = {n: d for d, n, _ in needles}
        wrong = [(depth_of[n], n) for r in revisit for n, ok in r["ok"].items() if not ok]
        if not all(dh["ok"].values()):
            dh_note = (
                f" [a device prefix hit at once after the cold ask (cached {dh['cached_tokens']}, no host tier "
                "involved) was wrong too: the failure is not specific to a host restore]"
            )
        elif wrong:
            dh_note = f" [a device prefix hit at once after the cold ask was right (cached {dh['cached_tokens']}): " + (
                "the host restore path is implicated]"
                if observed
                else "what failed came with the churn's eviction and the revisit's restore]"
            )
        else:
            dh_note = ""
        if observed:
            judge_crossed(res, revisit, depth_of, wrong, doc_pages, cfg["margin_pages"], dh_note)
            res.update(pages_restored=restored_total, pages_saved_by_churn=saved_total)
        else:
            cached = ", ".join(f"{r['cached_tokens']} of {r['prompt_tokens']}" for r in revisit)
            unseen = (
                "a restore from the host tier was not observable (no host-tier counters on /metrics), so only the "
                f"answers were judged; the revisit resumed {cached} prompt tokens from the prefix cache (device and "
                "host together)"
            )
            if wrong:
                codes = ", ".join(f"{d:g} ({n})" for d, n in wrong)
                res.update(
                    verdict="FAIL",
                    why=f"after {churned} churn documents, the code at depth {codes} is wrong (it was right cold); "
                    + unseen
                    + dh_note,
                )
            else:
                res.update(
                    verdict="PASS", why=f"all {len(depth_of)} codes right after {churned} churn documents; {unseen}"
                )
    except Exception as e:  # noqa: BLE001  a verdict file must exist whatever happened
        res.update(verdict="NO-RESULT", why=f"{type(e).__name__}: {str(e)[:300]}")
    finally:
        res["seconds"] = round(time.time() - t0, 1)
    return res


def judge_crossed(res: dict, revisit: list[dict], depth_of: dict, wrong: list, doc_pages: int, margin: int, note: str):
    """The verdict when the host-tier counters are on /metrics: only a needle whose page came back from the host is
    judged, and every needle must have crossed a restore for a PASS."""
    crossed = {}  # vault name -> did the revisit restore the pages from the needle's page on
    for r in revisit:
        for n in r["needles"]:
            r.setdefault("crossed", {})[n] = crossed[n] = r["pages_restored"] >= (1 - depth_of[n]) * doc_pages + margin
    res["crossed"] = {f"{depth_of[n]:g}": c for n, c in crossed.items()}
    wrong_crossed = [f"{d:g} ({n})" for d, n in wrong if crossed[n]]
    not_crossed = [f"{depth_of[n]:g} ({n})" for n, c in crossed.items() if not c]
    restored = [r["pages_restored"] for r in revisit]
    if wrong_crossed:
        res.update(
            verdict="FAIL",
            why=f"after a restore of {restored} host pages of {doc_pages}, the code at depth {', '.join(wrong_crossed)} "
            "is wrong (it was right cold)"
            + (f"; depth {', '.join(not_crossed)} did not cross a restore" if not_crossed else "")
            + note,
        )
    elif not_crossed:
        did = ", ".join(f"{depth_of[n]:g}" for n, c in crossed.items() if c) or "none"
        res["why"] = (
            f"depth {', '.join(not_crossed)} did not cross a restore: the revisit(s) restored {restored} of {doc_pages} "
            f"pages (a needle at depth d needs (1 - d) x {doc_pages} + {margin} back from the host): the document was "
            f"still partly on the device or the tier did not serve it (raise --churn); crossed: {did}"
        )
    else:
        res.update(
            verdict="PASS",
            why=f"all {len(crossed)} codes right after crossing a host restore ({restored} of {doc_pages} pages "
            f"restored, {res['churned']} churn documents)",
        )


# -- transport ---------------------------------------------------------------------------------------------------------


def http(url: str, body: dict | None, timeout: float) -> bytes:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})  # noqa: S310  as below
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310  the operator's own server
        return r.read()


def make_transport(server: str, api: str, model: str | None, timeout: float, max_tokens: int, session_id: str | None):
    """(ask, metrics, health) over HTTP for `run`."""
    base = server.rstrip("/")

    def ask(prompt: str, extra: dict | None = None) -> dict:
        body: dict = {"max_tokens": max_tokens, "temperature": 0}
        if api == "chat":
            body["messages"] = [{"role": "user", "content": prompt}]
            url = base + "/v1/chat/completions"
        else:
            body["prompt"] = prompt
            url = base + "/v1/completions"
        if model:
            body["model"] = model
        if session_id:
            body["session_id"] = session_id
        body.update(extra or {})
        try:
            j = json.loads(http(url, body, timeout))
        # The server's reason (a 400 for a field it refuses) is what the smoke reports.
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}") from None
        ch = j["choices"][0]
        u = j.get("usage") or {}
        if api == "chat":
            msg = ch.get("message") or {}
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            text = content or reasoning
            raw = {
                "content": content[:400],
                "reasoning": reasoning[:400],
                "finish": ch.get("finish_reason"),
                "completion_tokens": u.get("completion_tokens"),
            }
        else:
            text = ch.get("text") or ""
            raw = {"content": text[:400], "finish": ch.get("finish_reason")}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        return {"text": text, "prompt_tokens": int(u.get("prompt_tokens", 0)), "cached_tokens": int(cached), "raw": raw}

    def metrics() -> dict[str, float]:
        try:
            body = http(base + "/metrics", None, 30)
        except urllib.error.HTTPError as e:
            if e.code == 404:  # the router serves no /metrics (each engine does, on its own port): nothing to read
                return {}
            raise
        return parse_metrics(body.decode())

    def health() -> dict:
        return json.loads(http(base + "/health", None, 30))

    return ask, metrics, health


def main(argv: list[str] | None = None, transport=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    ap.add_argument(
        "--server",
        default="http://127.0.0.1:8000",
        help="the router of `trnrt serve`, or one engine's own port to read its /metrics too",
    )
    ap.add_argument("--verdict", required=True, help="the file whose first word is the verdict")
    ap.add_argument("--context-words", type=int, default=60000)
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--churn", type=int, default=48, help="at most this many other documents per churn round")
    ap.add_argument(
        "--evict-factor",
        type=float,
        default=40.0,
        help="with the host-tier counters, a churn round ends once the tier saved this many times the document's pages",
    )
    ap.add_argument("--api", choices=["chat", "completions"], default="chat")
    ap.add_argument("--page", type=int, default=256, help="the server's --page-size")
    ap.add_argument("--margin-pages", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--run-id", default=None, help="opens every document and names the session (default: random), so runs differ"
    )
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--max-tokens", type=int, default=96, help="the base budget; the smoke's profile sets the questions'"
    )
    ap.add_argument("--smoke-words", type=int, default=1200)
    ap.add_argument("--separate", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds per request")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run_id = a.run_id or secrets.token_hex(4)
    cfg = {
        "depths": [float(x) for x in a.depths.split(",")],
        "page": a.page,
        "seed": a.seed,
        "run_id": run_id,
        "context_words": a.context_words,
        "churn": a.churn,
        "evict_factor": a.evict_factor,
        "margin_pages": a.margin_pages,
        "separate": a.separate,
        "smoke_words": a.smoke_words,
    }
    if len(cfg["depths"]) > len(NAMES):
        print(f"at most {len(NAMES)} depths", file=sys.stderr)
        return 2
    ask, metrics, health = transport or make_transport(
        a.server, a.api, a.model, a.timeout, a.max_tokens, f"needle-{run_id}"
    )
    r = run(ask, metrics, health, cfg)
    tier = {
        True: f" restored {r.get('pages_restored')} crossed {r.get('crossed')}",
        False: " host restore not observable",
    }.get(r.get("host_tier_observed"), "")  # absent: refused before /metrics was read
    line = (
        f"{r['verdict']} needle | run {run_id} depths {cfg['depths']} prompt_tokens {r.get('prompt_tokens')} "
        f"doc_pages {r.get('doc_pages')}{tier} churned {r.get('churned')} | {r['why'][:400]}"
    )
    print(line, flush=True)
    out = a.out or os.path.dirname(os.path.abspath(a.verdict))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "needle.json"), "w") as f:
        json.dump(r, f, indent=1)
    tmp = a.verdict + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(line + "\n")
    os.replace(tmp, a.verdict)  # atomic: a reader sees the old verdict or the new one, never half
    return {"PASS": 0, "FAIL": 1}.get(r["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
