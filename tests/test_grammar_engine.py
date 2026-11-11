"""Guided decoding through xgrammar on the real engine (engine/grammar.py, engine/guided.py, engine/host_inputs.py): a
tiny random GLM-5.3-Flash (fp32, CPU) whose word-level tokenizer holds the pieces of a little JSON, so a grammar can
finish in a few tokens. The engine accepts every sampled token into the request's matcher and raises if one is refused,
so every test here also checks that no forbidden token was ever sampled.

- a choice, a JSON schema and a regex come out exactly as constrained, greedy and sampled, ending on EOS ("stop");
- a free JSON object is always a valid prefix of one, and parses when the grammar finished;
- under a reasoning block the prompt opened, the grammar constrains only after the reasoning's end (here forced by a
  thinking budget);
- the requests decoding beside a guided one are unchanged;
- under dynamic speculation a guided request turns the engine to plain decode steps while it runs, and greedy output
  of the others still equals plain greedy.
"""

from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
pytest.importorskip("xgrammar")

from tests import test_glm5_next as T
from tests.test_host_sampling_engine import THINK, save_tokenizer
from trnrt.engine.grammar import GrammarSpec
from trnrt.engine.request import SamplingParams

PIECES = ["{", "}", '"', ":", ",", "a", "b", "x", "yes", "no", "y", "[", "]", "0", "1", "true", *THINK]
WORDS = {w: 300 + i for i, w in enumerate(PIECES)}
BASE = {"max_num_seqs": 4, "max_model_len": 256, "num_pages": 256, "structured_outputs": True}
SCHEMA = {"type": "object", "properties": {"a": {"enum": ["x", "b"]}}, "required": ["a"], "additionalProperties": False}


def prompts(seed, lengths):
    """Random prompts drawn below the word ids (300 on), so none holds a reasoning tag: a prompt that opened a reasoning
    block would hold its grammar back until a close tag (guided.in_reasoning)."""
    return T.prompts(seed, lengths, vocab=min(WORDS.values()))


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("glm5_next_grammar"))
    T.build(path, seed=5)
    save_tokenizer(path, 384, WORDS)
    return path


def run(path, prompts, params, **kw):
    eng = T.engine(path, **{**BASE, **kw})
    try:
        reqs = eng.generate(prompts, params)
        return eng, reqs
    finally:
        eng.close()


def text(eng, r) -> str:
    return eng.tokenizer.decode(r.output_ids, skip_special_tokens=True)


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_a_choice_a_schema_and_a_regex_come_out_as_constrained(model, temperature):
    ps = prompts(31, (9, 14, 7, 11))
    specs = [
        GrammarSpec.choice(["no", "a,b"]),
        GrammarSpec("json_schema", json.dumps(SCHEMA)),
        GrammarSpec("regex", "(ab|ba){1,2}x"),
        GrammarSpec.choice(["true", "[1]"]),
    ]
    params = [SamplingParams(max_new_tokens=24, temperature=temperature, seed=3, grammar=g) for g in specs]
    eng, got = run(model, ps, params)
    outs = [text(eng, r) for r in got]
    assert outs[0] in ("no", "a,b") and outs[3] in ("true", "[1]"), outs
    assert json.loads(outs[1]) in ({"a": "x"}, {"a": "b"}), outs
    assert re.fullmatch("(ab|ba){1,2}x", outs[2]), outs
    for r in got:
        assert r.finish_reason == "stop" and r.output_ids[-1] == 1 and r.matcher.is_terminated(), r.output_ids


def test_a_free_json_object_is_always_a_valid_prefix(model):
    ps = prompts(32, (8, 12, 10))
    params = [
        SamplingParams(
            max_new_tokens=30, temperature=1.0, seed=s, grammar=GrammarSpec("json_schema", '{"type": "object"}')
        )
        for s in range(3)
    ]
    eng, got = run(model, ps, params)
    for r in got:  # the engine raises on a token its matcher refuses; a finished grammar parses
        t = text(eng, r)
        assert t.startswith("{"), t
        if r.finish_reason == "stop":
            assert isinstance(json.loads(t), dict), t


def test_under_reasoning_the_grammar_waits_for_the_reasoning_end(model):
    open_, close = WORDS[THINK[0]], WORDS[THINK[1]]
    ps = [[*p, open_] for p in prompts(33, (9, 12))]
    g = GrammarSpec.choice(["no", "a,b"])
    params = [SamplingParams(max_new_tokens=24, temperature=0.0, thinking_token_budget=b, grammar=g) for b in (0, 4)]
    eng, got = run(model, ps, params)
    for r, b in zip(got, (0, 4)):
        out = r.output_ids
        assert out[b] == close and r.think_count == b, out
        assert eng.tokenizer.decode(out[b + 1 :], skip_special_tokens=True) in ("no", "a,b"), out
        assert r.finish_reason == "stop"


def test_the_requests_beside_a_guided_one_are_unchanged(model):
    ps = prompts(34, (9, 14, 7))
    plain = SamplingParams(max_new_tokens=14, temperature=0.0, ignore_eos=True)
    guided = SamplingParams(max_new_tokens=14, temperature=0.0, grammar=GrammarSpec("json_schema", json.dumps(SCHEMA)))
    _, alone = run(model, ps, plain, structured_outputs=False)
    eng, got = run(model, ps, [plain, guided, plain])
    assert [got[0].output_ids, got[2].output_ids] == [alone[0].output_ids, alone[2].output_ids]
    assert json.loads(text(eng, got[1])) in ({"a": "x"}, {"a": "b"})


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_a_grammar_the_vocabulary_cannot_finish_ends_the_request_not_the_engine(model, ignore_eos):
    """xgrammar allows a token whose bytes start a match without knowing whether the vocabulary can finish it: "y" for
    the regex "yz" (no token spells "z"). The request then ends on EOS ("stop"), also under ignore_eos (the commit does
    not stop on that EOS: the dead end does), and the request beside it is unaffected."""
    ps = prompts(36, (9, 11))
    plain = SamplingParams(max_new_tokens=10, temperature=0.0, ignore_eos=True)
    grammar = GrammarSpec("regex", "yz")
    stuck = SamplingParams(max_new_tokens=10, temperature=0.0, ignore_eos=ignore_eos, grammar=grammar)
    _, alone = run(model, ps, plain, structured_outputs=False)
    eng, got = run(model, ps, [plain, stuck])
    assert got[0].output_ids == alone[0].output_ids
    assert text(eng, got[1]) == "y" and got[1].finish_reason == "stop" and got[1].grammar_dead
    assert got[1].output_ids == [WORDS["y"], 1], got[1].output_ids  # one EOS, not one a step to the length


def test_a_malformed_grammar_is_refused_at_admission(model):
    eng = T.engine(model, **BASE)
    try:
        with pytest.raises(ValueError, match="invalid regex"):
            eng.add_request([5, 6], SamplingParams(grammar=GrammarSpec("regex", "(unclosed")))
        assert not eng.has_work()
    finally:
        eng.close()


@pytest.fixture(scope="module")
def mtp_model(tmp_path_factory):
    from tests.test_spec_model import add_mtp

    path = str(tmp_path_factory.mktemp("glm5_next_grammar_mtp"))
    T.build(path, seed=2, index_topk=16)
    add_mtp(path)
    save_tokenizer(path, 384, WORDS)
    return path


def test_under_dynamic_speculation_a_guided_request_never_speculates(mtp_model, monkeypatch):
    """Two requests, within spec_max_batch: the engine would speculate both, but runs plain while the guided one runs
    (a verify takes no mask), then speculates the other alone; its greedy output equals plain greedy."""
    from trnrt.models import decoder

    monkeypatch.setattr(decoder, "MTP_CHUNK_Q", decoder.MTP_CHUNK_Q)  # the engine writes its routes (test_spec_model)
    monkeypatch.setattr(decoder, "DYN_MTP_KV", decoder.DYN_MTP_KV)
    ps = prompts(35, (9, 23))
    plain = SamplingParams(max_new_tokens=30, temperature=0.0, ignore_eos=True)
    guided = SamplingParams(max_new_tokens=20, temperature=0.0, grammar=GrammarSpec("json_schema", json.dumps(SCHEMA)))
    kw = {"max_prefill_tokens": 16, "max_model_len": 512, "num_pages": 512}
    _, want = run(mtp_model, ps[:1], plain, structured_outputs=False, **kw)
    spec = {"spec_method": "mtp", "spec_k": 2, "spec_max_batch": 2, "dyn_mtp_kv": True}
    eng, got = run(mtp_model, ps, [plain, guided], **kw, **spec)
    assert got[0].output_ids == want[0].output_ids
    assert json.loads(text(eng, got[1])) in ({"a": "x"}, {"a": "b"})
    assert eng.dyn_plain_steps > 0 and eng.dyn_spec_steps > 0, "plain while the guided request ran, then speculative"
