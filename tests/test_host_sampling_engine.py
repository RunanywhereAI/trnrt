"""Host-input sampling on the real engine: penalties, min_tokens and thinking budgets on a tiny random GLM-5.3-Flash
(fp32, CPU), with the engine's own scheduler, overlap and graphs (engine/host_inputs.py, engine/guided.py). Guided
decoding through xgrammar is tests/test_grammar_engine.py.

- with the flags off the engine refuses what needs them, and with them on but no request using them every call is the
  graph (and every token the one) the engine without them gives;
- a penalized request follows vLLM's formula, step by step, against its own raw logprobs, while the requests beside it
  decode exactly as alone;
- min_tokens keeps the end-of-sequence and stop ids out of the first min_tokens tokens;
- a thinking budget forces the reasoning's end at the budget.
"""

from __future__ import annotations

import pytest

pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")

from tests import test_glm5_next as T
from trnrt.engine.request import SamplingParams

THINK = ("<think>", "</think>")


def save_tokenizer(path: str, vocab: int, words: dict[str, int], eos: int = 1) -> None:
    """A word-level tokenizer over the model's whole vocabulary: id i is the word t<i>, except the ids `words` names,
    decoded by joining the words with no space (tests/test_e2e.py's)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    table = {f"t{i}": i for i in range(vocab)}
    for w, i in words.items():
        table.pop(f"t{i}")
        table[w] = i
    tk = Tokenizer(models.WordLevel(table, unk_token="t0"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tk.decoder = decoders.Fuse()
    t = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="t0", eos_token=f"t{eos}")
    t.save_pretrained(path)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """The tiny random GLM, with a tokenizer that has GLM's reasoning tags at ids 300 and 301."""
    path = str(tmp_path_factory.mktemp("glm5_next_host"))
    T.build(path, seed=4)
    save_tokenizer(path, 384, {THINK[0]: 300, THINK[1]: 301})
    return path


BASE = {"max_num_seqs": 4, "max_model_len": 256, "num_pages": 256}
FLAGS = {"structured_outputs": True, "penalties": True, "penalty_buckets": (16, 64)}


def run(path, prompts, params, **kw):
    eng = T.engine(path, **{**BASE, **kw})
    try:
        reqs = eng.generate(prompts, params)
        return reqs, set(eng.runner.calls)
    finally:
        eng.close()


def test_the_flags_off_engine_refuses_what_needs_them(model):
    eng = T.engine(model, **BASE)
    try:
        for p, flag in (
            (SamplingParams(presence_penalty=0.3), "--penalties"),
            (SamplingParams(min_tokens=2), "--structured-outputs"),
            (SamplingParams(thinking_token_budget=4), "--structured-outputs"),
        ):
            with pytest.raises(ValueError, match=flag):
                eng.add_request([5, 6, 7], p)
        assert not eng.has_work()
    finally:
        eng.close()


def test_with_the_flags_on_and_no_request_using_them_every_call_is_todays(model):
    """The same tokens, greedy and sampled, and the same call keys: no variant was launched."""
    ps = T.prompts(21, (9, 17, 12))
    params = [
        SamplingParams(max_new_tokens=12, temperature=0.0, ignore_eos=True),
        SamplingParams(max_new_tokens=12, temperature=0.9, top_p=0.9, seed=7, ignore_eos=True),
        SamplingParams(max_new_tokens=12, temperature=0.0, ignore_eos=True, logprobs=3),
    ]
    want, keys_off = run(model, ps, params)
    got, keys_on = run(model, ps, params, **FLAGS)
    assert [r.output_ids for r in got] == [r.output_ids for r in want]
    assert [r.logprobs for r in got] == [r.logprobs for r in want]
    assert keys_on == keys_off and all(len(k) == 3 for k in keys_on if k[0] in ("decode", "prefill")), sorted(keys_on)


def test_a_penalized_request_follows_the_formula_beside_unchanged_neighbours(model):
    """Greedy with frequency and presence penalties: each token is the argmax of its raw logprobs (logprobs=20, before
    penalties) minus vLLM's penalties over the output so far, and the plain requests decoding beside it (every other
    step it waits for its token) produce what they produce alone."""
    ps = T.prompts(22, (11, 8, 15))
    plain = SamplingParams(max_new_tokens=16, temperature=0.0, ignore_eos=True)
    pen = SamplingParams(
        max_new_tokens=16, temperature=0.0, ignore_eos=True, logprobs=20, frequency_penalty=1.5, presence_penalty=1.5
    )
    alone, _ = run(model, ps, [plain, plain, plain])
    got, keys = run(model, ps, [plain, pen, plain], **FLAGS)
    assert [got[0].output_ids, got[2].output_ids] == [alone[0].output_ids, alone[2].output_ids]
    out = got[1].output_ids
    assert out != alone[1].output_ids, "the penalties changed nothing: the case does not test them"
    for i, (tok, (_, ids, lps)) in enumerate(zip(out, got[1].logprobs)):
        seen = out[:i]
        score = {t: lp - 1.5 * seen.count(t) - 1.5 * (t in seen) for t, lp in zip(ids, lps)}
        assert tok in score, f"step {i}: the penalized argmax fell outside the raw top 20"
        assert tok == max(score, key=score.__getitem__), (i, tok)
    tags = {k[3:] for k in keys if len(k) > 3}
    assert ("grammar", "pen16") in tags, sorted(keys)


def test_a_repetition_penalty_counts_the_prompt(model):
    """repetition_penalty divides positive and multiplies negative logits of every token in the prompt or the output: at
    a large value the greedy output never takes such a token again (the unpenalized output does: the control)."""
    (p,) = T.prompts(23, (24,))
    free, _ = run(model, [p], SamplingParams(max_new_tokens=24, temperature=0.0, ignore_eos=True))
    out = free[0].output_ids
    assert any(t in p or t in out[:i] for i, t in enumerate(out)), "control: the free output repeats"
    got, _ = run(
        model,
        [p],
        SamplingParams(max_new_tokens=24, temperature=0.0, ignore_eos=True, repetition_penalty=60.0),
        penalties=True,
    )
    out = got[0].output_ids
    assert len(out) == 24 and not [t for i, t in enumerate(out) if t in p or t in out[:i]]


def test_min_tokens_keeps_eos_and_stop_ids_out_of_the_first_tokens(model):
    ps = T.prompts(24, (10, 13))
    free, _ = run(model, ps, SamplingParams(max_new_tokens=30, temperature=0.0, ignore_eos=True))
    # The stop ids: the free output's first tokens, so a request that may stop on them would stop at once.
    stops = tuple({free[0].output_ids[0], free[1].output_ids[0], free[0].output_ids[3]})
    eos = 1
    params = SamplingParams(max_new_tokens=30, temperature=0.0, min_tokens=8, stop_token_ids=stops)
    got, keys = run(model, ps, params, structured_outputs=True)
    for r in got:
        assert not set(r.output_ids[:8]) & {*stops, eos}, r.output_ids
        assert len(r.output_ids) >= 8
    first = SamplingParams(max_new_tokens=30, temperature=0.0, stop_token_ids=stops)
    ctl, _ = run(model, ps, first, structured_outputs=True)
    assert len(ctl[0].output_ids) == 1, "control: without min_tokens the first stop id ends it at once"
    assert any(k[3:] == ("grammar",) for k in keys)


def test_a_thinking_budget_forces_the_reasoning_end_at_the_budget(model):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    close = tok.convert_tokens_to_ids(THINK[1])
    ps = [[*p, 300] for p in T.prompts(25, (9, 14, 6))]  # each prompt opens the reasoning block
    free, _ = run(model, ps, SamplingParams(max_new_tokens=20, temperature=0.0, ignore_eos=True))
    assert all(close not in r.output_ids[:12] for r in free), "control: the free model does not close it itself"
    budgets = [0, 3, 7]
    params = [
        SamplingParams(max_new_tokens=20, temperature=0.0, ignore_eos=True, thinking_token_budget=b) for b in budgets
    ]
    plain = SamplingParams(max_new_tokens=20, temperature=0.0, ignore_eos=True)
    got, _ = run(model, [*ps, ps[0]], [*params, plain], structured_outputs=True)
    for r, b, f in zip(got, budgets, free):
        assert r.output_ids[:b] == f.output_ids[:b], "the reasoning before the budget is the free model's"
        assert r.output_ids[b] == close and r.think_done and r.think_count == b, (b, r.output_ids)
    assert got[3].output_ids == free[0].output_ids, "the request beside them is unchanged"


def test_a_static_speculative_engine_refuses_the_flags():
    from trnrt.config import EngineConfig

    for flag in ("structured_outputs", "penalties"):
        with pytest.raises(ValueError, match="spec_max_batch"):
            EngineConfig(model_path="m", spec_method="mtp", **{flag: True})
        EngineConfig(model_path="m", spec_method="mtp", spec_max_batch=2, **{flag: True})  # dynamic: served


def test_a_host_bound_row_never_launches_with_its_newest_token_in_flight(model):
    """The ordering rule (engine/scheduler.py): a decode of a penalized request carries its newest token from the host,
    never the board's placeholder (its penalties would miss that token), while the plain requests beside it keep the
    overlap: some of their decodes take the token from the board."""
    from trnrt.engine.scheduler import PENDING

    eng = T.engine(model, **BASE, **FLAGS)
    seen = {"bound": 0, "overlap": 0}
    orig = eng.runner.decode

    def spy(seqs):
        for s in seqs:
            inflight = s.req.token_ids[s.start] == PENDING
            if s.req.params.has_penalties:
                assert not inflight, f"{s.req.rid} launched with its newest token in flight"
                seen["bound"] += 1
            else:
                seen["overlap"] += inflight
        return orig(seqs)

    eng.runner.decode = spy
    try:
        ps = T.prompts(26, (9, 12, 7))
        plain = SamplingParams(max_new_tokens=12, temperature=0.0, ignore_eos=True)
        pen = SamplingParams(max_new_tokens=12, temperature=0.0, ignore_eos=True, presence_penalty=0.5)
        eng.generate(ps, [plain, pen, plain])
    finally:
        eng.close()
    assert seen["bound"] >= 11 and seen["overlap"] > 0, seen
