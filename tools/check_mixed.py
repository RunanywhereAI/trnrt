"""Greedy text through the served engine, for the bytes gate (A0) and decode parity: the same engine `trnrt engine`
builds from the same arguments (so the same graphs from the same compile cache), fed natural-text prompts of different
lengths, `--concurrency` in flight, greedy, --new-tokens each, every output and its top-2 logprobs written to --out.

    python tools/check_mixed.py --model DIR --core-base 0 --concurrency 8 --requests 384 --new-tokens 64 \
        --lengths 3840,1500,3000,700,2048,2500,1024,3200,128,192,256,384,128,192,256,384 --turns --out a.json
    python tools/check_mixed.py --compare reference.json a.json

Prompts are consecutive windows of LONG_TEXT (repeated), 997 tokens apart, their lengths cycling through --lengths, so
prefill chunks of some requests meet the decodes of others in most steps. --turns adds two follow-up turns per prompt
that resume from the prefix cache (state checkpoints). --compare prints, per request, whether the outputs are equal and
where they first differ, the share of output tokens that agree position by position, and the decode-parity verdict.
Every engine argument (trnrt.__main__.add_engine_args) has the served value as its default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# decode-parity thresholds (see compare): taken from the logit format and the short-context parity, never from a
# candidate under test
DP_MEAN, DP_P99 = 0.03, 0.30
DP_EPS = 1e-3  # relative slack for the fp32 subtraction of the log-sum-exp that turns logits into logprobs

# one passage of ordinary English (written for this check, about 1000 tokens)
LONG_TEXT = " ".join([
    "Water moves through the environment in a continuous cycle. Heat from the sun warms the surface of",
    "oceans, lakes and rivers, and some of that water turns into vapor and rises into the air. As the",
    "vapor climbs, the air around it cools, and the vapor condenses into tiny droplets that gather into",
    "clouds. When the droplets grow large enough, they fall back to the ground as rain or snow. Some of",
    "that water runs off the land into streams, some soaks into the soil and becomes groundwater, and",
    "some is taken up by plants, which release it again through their leaves. Over time, nearly all of",
    "it finds its way back to the sea, and the cycle begins again. The same water that fell as rain on",
    "an ancient forest may today be part of a glacier, a cup of tea, or a cloud drifting over a city.",
    "The first computers filled entire rooms. They were built from vacuum tubes, which glowed like light",
    "bulbs and failed often, and engineers spent much of their time replacing the ones that burned out.",
    "Programs were entered by setting switches or by feeding in cards with holes punched in them. The",
    "invention of the transistor changed everything: it did the same job as a tube, but it was smaller,",
    "used less power and lasted far longer. Later, many transistors were placed on a single chip of",
    "silicon, and the number that could fit on one chip doubled roughly every two years. Machines that",
    "once needed a building of their own shrank to the size of a desk, then a book, and finally a phone",
    "that fits in a pocket and is more powerful than the room-sized computers that came before it.",
    "A good loaf of bread needs only four ingredients: flour, water, salt and yeast. The baker mixes them",
    "into a dough and kneads it, stretching and folding it until it becomes smooth and elastic. Kneading",
    "develops gluten, a network of proteins that traps the gas the yeast produces as it feeds on the",
    "sugars in the flour. The dough is then left to rise in a warm place, often for several hours, and",
    "during that time it slowly doubles in size. After it is shaped and has risen once more, it goes",
    "into a hot oven. The heat makes the gas expand one last time, sets the structure of the crumb, and",
    "browns the outside into a crisp crust. Many bakers say that patience matters more than any recipe.",
    "Trains have connected distant towns for almost two hundred years. The early locomotives burned coal",
    "to boil water, and the steam pushed pistons that turned the wheels. Railways allowed farmers to send",
    "their crops to markets far away and let people travel in a day a distance that once took a week.",
    "Today many trains run on electricity drawn from wires above the track, and the fastest ones carry",
    "passengers at more than three hundred kilometers per hour. Even so, the basic idea has not changed:",
    "steel wheels rolling on steel rails waste very little energy, which is why a single locomotive can",
    "pull a long line of heavy cars across a continent while using far less fuel than a fleet of trucks.",
    "Bees are small, but they play an enormous role in feeding the world. As a bee visits flowers to",
    "collect nectar and pollen, grains of pollen stick to the fine hairs on its body and are carried to",
    "the next flower, which allows the plant to produce seeds and fruit. Apples, almonds, berries and",
    "many vegetables depend on this help. A single colony can contain tens of thousands of workers, and",
    "each one has a job that changes as it grows older: cleaning the hive, feeding the young, building",
    "wax comb, guarding the entrance, and finally flying out to forage. Together they behave almost like",
    "one large animal, sharing food and information and keeping the temperature of the hive steady.",
])



def logit_quantum(run: dict) -> float | None:
    """One step of a run's logit grid: its smallest positive top-1 minus top-2 logprob margin below 1 nat (a difference
    of two bf16 logits of one row), rounded to a power of two; None without one."""
    import math

    ms = [lp[2][0] - lp[2][1] for r in run["requests"] for lp in (r.get("logprobs") or ()) if len(lp[2]) > 1]
    pos = [m for m in ms if 1e-4 < m < 1.0]
    return 2.0 ** round(math.log2(min(pos))) if pos else None


def compare(a_path: str, b_path: str, ulp: float | None = None, out: dict | None = None) -> int:
    """a: the reference run, b: the run under test (both written by this tool). Besides equality, for runs written with
    logprobs: every position before a pair's first difference has the same input tokens in both runs, so the
    chosen-token logprobs there are a teacher-forced comparison; it is split by where b computed the token (a prefill
    chunk or a decode call), and at each first difference a's top-1 minus top-2 logprob margin says how close that
    choice was.

    The verdict (decode parity). The logits are bf16, so top-1 minus top-2 margins come on a grid: with q the reference's
    logit quantum (its smallest positive margin, rounded to a power of two; or --logit-ulp), PASS iff:
    (1) every first difference is at a tie within one step: reference margin <= q AND the run under test's margin <= q;
    (2) the teacher-forced |dlogprob| before each first difference never exceeds 2 q;
    (3) its mean <= DP_MEAN and its p99 <= DP_P99.
    A run without logprobs FAILS. Exit status 1 on FAIL. Two runs of the same engine on the same arguments read
    equal=N/N: that is the bytes gate (A0), and the parity figures are then all zero."""
    with open(a_path) as f:
        a = json.load(f)
    with open(b_path) as f:
        b = json.load(f)
    forced = bool(b.get("forced"))  # b ran on a's tokens (--force / --score): every position is teacher-forced
    if a.get("forced") and not (forced and os.path.basename(a["forced"]) == os.path.basename(b["forced"])):
        raise SystemExit("a forced reference compares only with a run forced on the same tokens")
    if forced and not a.get("forced") and os.path.basename(a_path) != os.path.basename(b["forced"]):
        # a free run read as teacher-forced at every position would compare positions whose inputs differ
        raise SystemExit(f"{b_path} ran on {os.path.basename(b['forced'])}'s tokens, not on {os.path.basename(a_path)}'s")
    same = agree = total = 0
    dl: dict[str, list[float]] = {"mixed decode row": [], "decode call": [], "prefill chunk": []}
    ds: dict[str, list[float]] = {k: [] for k in dl}  # the same, signed: under test minus reference
    margins = []
    tf: list[float] = []  # |dlogprob| at every teacher-forced position that is not a difference: the verdict's pool
    no_lp = 0
    for x, y in zip(a["requests"], b["requests"]):
        if x["prompt_len"] != y["prompt_len"]:
            raise SystemExit("the two runs did not use the same prompts")
        ox, oy = x["output_ids"], y["output_ids"]
        first = next((i for i, (p, q) in enumerate(zip(ox, oy)) if p != q), None)
        n = min(len(ox), len(oy))
        agree += sum(p == q for p, q in zip(ox, oy))
        total += n
        same += ox == oy
        lx, ly = x.get("logprobs"), y.get("logprobs")
        where = y.get("where")
        if not (lx and ly):
            no_lp += 1
        # Positions whose inputs are equal in both runs: up to and including the first difference, or every one when b
        # was forced on a's tokens; the differences among them (b's own top-1 against a's) are checked by margin.
        flips = [i for i in range(n) if ox[i] != oy[i]] if forced else ([] if first is None else [first])
        if lx and ly:
            for i in range(n if forced or first is None else first + 1):
                kind = where[i] if where else "decode call"
                d = ly[i][2][0] - lx[i][2][0] if i in flips else ly[i][0] - lx[i][0]  # at a difference: the top-1s
                dl.setdefault(kind, []).append(abs(d))
                ds.setdefault(kind, []).append(d)
                if i not in flips:
                    tf.append(abs(d))
            for i in flips:
                top = lx[i][2]
                margins.append((x["prompt_len"], i, top[0] - top[1], ly[i][2][0] - ly[i][2][1],
                                where[i] if where else "?"))
        state = "equal" if ox == oy else (f"{len(flips)} differences, first at output {first}" if forced
                                          else f"first difference at output {first}")
        print(f"  prompt {x['prompt_len']:>5}: {state}  {x['text'][:60]!r}" + ("" if ox == oy else f" | {y['text'][:60]!r}"))
    for p, i, mx, my, w in margins:
        print(f"  margin at a difference, prompt {p} output {i} ({w} in the run under test): "
              f"top-1 minus top-2 logprob {mx:.4f} (reference) / {my:.4f} (under test)")
    for kind, v in dl.items():
        if v:
            v = sorted(v)
            sg = ds[kind]
            print(f"  teacher-forced |dlogprob| over {kind}s: n={len(v)} max {v[-1]:.4f} p99 {v[int(0.99 * (len(v) - 1))]:.4f}"
                  f" mean {sum(v) / len(v):.5f}; signed (under test minus reference) mean {sum(sg) / len(sg):+.5f}")
    why = []
    if no_lp:
        why.append(f"{no_lp} request pairs without logprobs")
    v = sorted(tf)
    mean = sum(v) / len(v) if v else float("nan")
    p99 = v[int(0.99 * (len(v) - 1))] if v else float("nan")
    if not mean <= DP_MEAN:
        why.append(f"mean |dlogprob| {mean:.4f} > {DP_MEAN}")
    if not p99 <= DP_P99:
        why.append(f"p99 |dlogprob| {p99:.4f} > {DP_P99}")
    q = ulp if ulp is not None else logit_quantum(a)
    src = "--logit-ulp" if ulp is not None else "the reference's smallest positive top-2 margin"
    if q is None:
        why.append("no logit quantum: the reference has no positive top-2 margin below 1 nat (pass --logit-ulp)")
        q = float("nan")
    tol = q * (1 + DP_EPS)
    mx_abs = v[-1] if v else float("nan")
    if not mx_abs <= 2 * tol:
        why.append(f"max |dlogprob| {mx_abs:.4f} > 2 steps ({2 * q:.4f})")
    clear = [(p, i, mx, my) for p, i, mx, my, _ in margins if not (mx <= tol and my <= tol)]
    if clear:
        why.append(f"{len(clear)} differences not at a one-step tie (margins > {q:.4f}): "
                   + ", ".join(f"prompt {p} output {i} margins {mx:.3f} / {my:.3f}" for p, i, mx, my in clear[:4]))
    verdict = "FAIL" if why else "PASS"
    print(f"decode parity: {verdict} mean_abs={mean:.4f} p99_abs={p99:.4f} max_abs={mx_abs:.4f} tokens={len(v)} "
          f"divergences={len(margins)} (logit step q={q:.4f} from {src}; thresholds: each divergence a tie within q on "
          f"both sides, max <= 2q, mean <= {DP_MEAN}, p99 <= {DP_P99})" + ("" if not why else ": " + "; ".join(why)))
    print(f"RESULT equal={same}/{len(a['requests'])} token_agreement={agree / max(total, 1):.4f} ({agree}/{total}) "
          f"decode_parity={verdict} forced={int(forced)} mean_abs={mean:.4f} p99_abs={p99:.4f} max_abs={mx_abs:.4f} q={q:.4f} "
          f"divergences={len(margins)}")
    if out is not None:  # the figures compare_rel holds against each other
        out.update(mean=mean, p99=p99, max=mx_abs, divergences=len(margins), verdict=verdict)
    return 1 if why else 0


def prompt_sets(tokenizer, lengths: list[int], n: int) -> list:
    """n prompts: consecutive windows of LONG_TEXT 997 tokens apart, their lengths cycling through `lengths`."""
    ids = tokenizer(LONG_TEXT)["input_ids"]
    while len(ids) < max(lengths) + n * 997:
        ids = ids + ids
    return [ids[i * 997 : i * 997 + lengths[i % len(lengths)]] for i in range(n)]


def run_turns(eng, prompts: list, reqs: list, n_new: int) -> dict:
    """--turns: after every prompt, one at a time (so a prefix-cache hit runs on the same path on both sides of a
    comparison), two turns of each: a, its prompt to its prompt-end page boundary + new text, which hits the prompt-end
    checkpoint; b, its prompt + its output + new text, which hits the finish checkpoint. Token ids, so the prefixes match
    exactly. Appends to prompts and reqs; returns {request index: {"source", "kind"}}."""
    from trnrt.engine.request import SamplingParams

    ps = eng.cfg.page_size
    n_text, turn_of = len(prompts), {}
    sp = SamplingParams(max_new_tokens=n_new, temperature=0.0, ignore_eos=True, logprobs=2)
    for i in range(n_text):
        new = list(prompts[(i + 7) % n_text][:64])
        pe = (len(prompts[i]) - 1) // ps * ps  # the prompt-end checkpoint's position
        for kind, ids in (("a", list(prompts[i][:pe]) + new), ("b", list(prompts[i]) + list(reqs[i].output_ids) + new)):
            r = eng.add_request(ids, sp)
            turn_of[len(reqs)] = {"source": i, "kind": kind}
            reqs.append(r)
            prompts.append(ids)
            while r.finish_time is None:
                eng.step()
    return turn_of


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":  # --compare REF RUN [--logit-ulp Q]
        q = float(sys.argv[sys.argv.index("--logit-ulp") + 1]) if "--logit-ulp" in sys.argv else None
        sys.exit(compare(sys.argv[2], sys.argv[3], q))
    from trnrt.__main__ import add_engine_args, engine_config

    # allow_abbrev=False: engine_config below reads the same argv without abbreviations, and would ignore one this took
    ap = argparse.ArgumentParser(prog="tools/check_mixed.py", description=__doc__.split("\n")[0], allow_abbrev=False)
    add_engine_args(ap)
    ap.add_argument("--core-base", type=int, default=0, help="the engine's first logical NeuronCore")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=None, help="prompts (default: --concurrency)")
    ap.add_argument("--lengths", default="8192,1500,6000,700,4096,3000,8000,2500,128,192,256,384,128,192,256,384",
                    help="comma list of prompt lengths in tokens, cycled over the requests")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--no-warmup", action="store_true", help="skip the bucket warmup (each graph loads at its first call)")
    ap.add_argument("--turns", action="store_true",
                    help="after the prompts, one at a time, two prefix-cache turns of each (16 new tokens, see run_turns)")
    ap.add_argument("--permute", default=None,
                    help="submit the requests in another order: 'reverse', or an int seed; the output keeps the canonical "
                         "order, so --compare pairs requests as without it (batch-row permutation moves bits)")
    args = ap.parse_args()
    if args.permute and args.turns:
        raise SystemExit("--permute with --turns: not built (the turns key on the submission position)")
    from trnrt.engine.engine import LLMEngine
    from trnrt.engine.request import SamplingParams

    conc = args.concurrency
    eng = LLMEngine(engine_config(sys.argv[1:], args.core_base))
    prompts = prompt_sets(eng.tokenizer, [int(x) for x in args.lengths.split(",")], args.requests or conc)
    n_req = len(prompts)
    if not args.no_warmup:
        w = eng.warmup()
        print(f"bucket warmup {w['seconds']:.1f}s over {w['graphs']} graphs", flush=True)
    sp = SamplingParams(max_new_tokens=args.new_tokens, temperature=0.0, ignore_eos=True, logprobs=2)
    order = list(range(n_req))  # submission order; the output stays canonical
    if args.permute == "reverse":
        order.reverse()
    elif args.permute:
        import random

        random.Random(int(args.permute)).shuffle(order)
    t0 = time.perf_counter()
    at, live, started = {}, [], 0
    while started < n_req or live:
        while len(live) < conc and started < n_req:
            i = order[started]
            r = eng.add_request(prompts[i], sp)
            at[i] = r
            live.append(r)
            started += 1
        eng.step()
        live = [r for r in live if r.finish_time is None]
    wall = time.perf_counter() - t0
    reqs = [at[i] for i in range(n_req)]
    turn_of = run_turns(eng, prompts, reqs, min(16, args.new_tokens)) if args.turns else {}
    calls: dict = {}
    rows: dict = {}  # decode calls by their batch rows (key[1]): which batch sizes really ran
    for key, n in eng.runner.calls.items():
        calls[key[0]] = calls.get(key[0], 0) + n
        if key[0] == "decode" and len(key) > 1:
            rows[f"decode:B{key[1]}"] = rows.get(f"decode:B{key[1]}", 0) + n
    out = {"mixed": False, "wall_s": round(wall, 1), "calls": calls, "rows": rows, "forced": None, "permute": args.permute,
           "requests": [{"prompt_len": len(p), "output_ids": r.output_ids, "text": eng.tokenizer.decode(r.output_ids),
                         "logprobs": [[lp, list(map(int, ids)), list(map(float, lps))] for lp, ids, lps in r.logprobs],
                         "where": ["prefill chunk"] + ["decode call"] * (len(r.output_ids) - 1),
                         "turn": turn_of.get(idx)}
                        for idx, (p, r) in enumerate(zip(prompts, reqs))]}
    eng.close()
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"RESULT wall_s={out['wall_s']} calls={calls} requests={len(reqs)} rows={rows}", flush=True)
    for x in out["requests"][:4]:
        print(f"  {x['prompt_len']:>5}: {x['text'][:200]!r}", flush=True)


if __name__ == "__main__":
    main()
