"""The pieces of host-input sampling that need no model: the packed token mask and the sampler's masked, penalized draw
(engine/sampler.py shaped_logits), the per-call inputs and variant tags (engine/host_inputs.py), a request's host-bound
rules (engine/request.py), and the API's constraint fields (serve/structured.py)."""

from __future__ import annotations

import json
import types

import numpy as np
import pytest
import torch

from trnrt.engine import grammar, host_inputs
from trnrt.engine.grammar import GrammarSpec
from trnrt.engine.request import Request, SamplingParams, unsupported
from trnrt.engine.sampler import NUM_TOP_LOGPROBS, sample, shaped_logits, unpack_bitmask
from trnrt.serve.structured import OBJECT, grammar_spec

V = 300
N = NUM_TOP_LOGPROBS


def _draw(logits, temperature, seed, bitmask=None, penalties=None, k=64):
    B = logits.shape[0]
    g = torch.Generator().manual_seed(seed)
    return sample(
        logits,
        torch.full((B,), float(temperature)),
        torch.full((B,), 0.95),
        torch.zeros(B, dtype=torch.int64),
        torch.zeros(B),
        torch.rand(B, k, generator=g),
        bitmask,
        penalties,
    )


def test_the_packed_mask_round_trips_through_the_graphs_unpacking():
    m = grammar.full_mask(3, V)
    assert m.shape == (3, grammar.words(V)) and unpack_bitmask(torch.from_numpy(m), V).all()
    grammar.allow_only(m[0], 37)
    grammar.forbid(m[1], [0, 31, 32, 299])
    want = np.ones((3, V), bool)
    want[0] = False
    want[0, 37] = True
    want[1, [0, 31, 32, 299]] = False
    assert np.array_equal(unpack_bitmask(torch.from_numpy(m), V).numpy(), want)
    assert grammar.allowed(m[0], 37) and not grammar.allowed(m[0], 36) and not grammar.allowed(m[1], 31)


@pytest.mark.parametrize("temperature", [0.0, 0.7, 1.5])
def test_a_masked_draw_never_takes_a_forbidden_token_and_reports_raw_logprobs(temperature):
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(6, V, generator=g) * 3
    mask = grammar.full_mask(6, V)
    rng = np.random.default_rng(0)
    allowed = []
    for i in range(6):
        keep = rng.choice(V, size=[1, 3, 40, 200, 299, 5][i], replace=False)
        mask[i] = 0
        for t in keep:
            mask[i].view(np.uint32)[t // 32] |= np.uint32(1 << (t % 32))
        allowed.append(set(keep.tolist()))
    raw = _draw(logits, temperature, 0)
    for seed in range(20):
        out = _draw(logits, temperature, seed, torch.from_numpy(mask))
        tok = out[:, 0].long().tolist()
        assert all(t in allowed[i] for i, t in enumerate(tok)), (seed, tok)
        lse = torch.logsumexp(logits, -1)
        assert torch.allclose(out[:, 1], logits[torch.arange(6), out[:, 0].long()] - lse, atol=1e-5)
        assert torch.equal(out[:, 2:], raw[:, 2:]), "top logprobs are the raw (unmasked) ones"
    if temperature == 0.0:  # greedy: the best allowed token
        best = [max(allowed[i], key=lambda t, i=i: float(logits[i, t])) for i in range(6)]
        assert _draw(logits, 0.0, 0, torch.from_numpy(mask))[:, 0].long().tolist() == best


def _reference_penalized(logits, prompt, out, rep, freq, pres):
    """vLLM's penalties, written out per token: repetition over prompt and output tokens, then frequency and presence
    over the output alone."""
    x = logits.clone()
    counts = {t: out.count(t) for t in set(out)}
    for t in set(prompt) | set(out) if rep != 1.0 else set(out):
        if rep != 1.0:
            x[t] = x[t] / rep if x[t] > 0 else x[t] * rep
        c = counts.get(t, 0)
        x[t] -= freq * c + pres * (c > 0)
    return x


@pytest.mark.parametrize(("rep", "freq", "pres"), [(2.0, 0.0, 0.0), (1.0, 1.5, 0.0), (1.0, 0.0, 3.0), (1.5, -0.4, 2.0)])
def test_penalties_shape_the_draw_as_the_reference_formula_says(rep, freq, pres):
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(4, V, generator=g) * 2
    rng = np.random.default_rng(3)
    reqs = []
    for i in range(4):
        prompt = rng.integers(0, V, 30 + i).tolist()
        top = np.argsort(-logits[i].numpy())
        out = [*top[:3].tolist() * 2, *rng.choice(top[3:40], 4).tolist()]  # the best three twice, then a few more
        r = Request(
            f"r{i}",
            prompt,
            SamplingParams(repetition_penalty=rep, frequency_penalty=freq, presence_penalty=pres),
        )
        r.token_ids += out
        reqs.append((r, prompt, out))
    m = 256
    pen = tuple(torch.from_numpy(a) for a in host_inputs.penalty_inputs([r for r, _, _ in reqs], m))
    got = _draw(logits, 0.0, 0, penalties=pen)[:, 0].long().tolist()
    want = [int(_reference_penalized(logits[i], p, o, rep, freq, pres).argmax()) for i, (_, p, o) in enumerate(reqs)]
    assert got == want, "MISMATCH"
    unpen = logits.argmax(-1).tolist()
    assert all(g != u for g, u in zip(got, unpen)), "NOCHANGE: the penalties left a row's best token: no test of them"


def test_a_penalty_never_brings_a_forbidden_token_back():
    logits = torch.zeros(1, V)
    logits[0, 5] = 1.0
    mask = grammar.full_mask(1, V)
    grammar.forbid(mask[0], [7])
    r = Request("r", [1, 2], SamplingParams(frequency_penalty=-100.0))
    r.token_ids += [7, 7]  # a negative frequency penalty lifts token 7 by 200
    pen = tuple(torch.from_numpy(a) for a in host_inputs.penalty_inputs([r], 8))
    shaped = shaped_logits(logits, None, pen)
    assert int(shaped.argmax()) == 7, "control: the penalty alone picks 7"
    assert int(_draw(logits, 0.0, 0, torch.from_numpy(mask), pen)[0, 0]) == 5


def _ecfg(structured=False, penalties=False, buckets=(16, 64)):
    return types.SimpleNamespace(structured_outputs=structured, penalties=penalties, penalty_buckets=buckets)


def test_variant_tags_follow_the_flags():
    assert host_inputs.variant_tags(_ecfg(), 100) == []
    assert host_inputs.variant_tags(_ecfg(True), 100) == [("grammar",)]
    assert host_inputs.variant_tags(_ecfg(False, True), 100) == [("pen16",), ("pen64",), ("pen100",)]
    assert host_inputs.variant_tags(_ecfg(True, True, (16, 500)), 100) == [
        ("grammar",),
        ("grammar", "pen16"),
        ("grammar", "pen100"),
    ]


def test_a_call_takes_a_variant_only_when_a_row_needs_it():
    plain = Request("a", [1, 2, 3], SamplingParams())
    pen = Request("b", [1, 2, 3], SamplingParams(presence_penalty=0.5))
    mt = Request("c", [4, 5], SamplingParams(min_tokens=2, stop_token_ids=(9,)))
    eos = (1,)
    assert host_inputs.extras(_ecfg(True, True), 100, eos, None, [plain, None]) == ([], ())
    extra, tag = host_inputs.extras(_ecfg(True, False), 100, eos, None, [plain, mt, None])
    assert tag == ("grammar",) and extra[1] is None
    rows = unpack_bitmask(torch.from_numpy(extra[0]), 100)
    assert rows[0].all() and rows[2].all() and not rows[1, 1] and not rows[1, 9] and int(rows[1].sum()) == 98
    mt.token_ids += [7, 8]  # min_tokens reached: no mask any more
    assert host_inputs.extras(_ecfg(True, False), 100, eos, None, [plain, mt]) == ([], ())
    extra, tag = host_inputs.extras(_ecfg(False, True), 100, eos, None, [plain, pen])
    assert tag == ("pen16",) and extra[0] is None
    extra, tag = host_inputs.extras(_ecfg(True, True), 100, eos, None, [plain, pen])
    assert tag == ("grammar", "pen16") and unpack_bitmask(torch.from_numpy(extra[0]), 100).all()
    many = Request("d", list(range(40)), SamplingParams(repetition_penalty=1.2))
    assert host_inputs.extras(_ecfg(False, True), 100, eos, None, [many])[1] == ("pen64",)
    many.token_ids += list(range(40, 90))
    assert host_inputs.extras(_ecfg(False, True), 100, eos, None, [many])[1] == ("pen100",)


def test_min_tokens_keeps_eos_under_ignore_eos_and_counts_tokens_in_flight():
    r = Request("r", [4, 5], SamplingParams(min_tokens=3, ignore_eos=True, stop_token_ids=(9,)))
    extra, _ = host_inputs.extras(_ecfg(True), 100, (1,), None, [r])
    row = unpack_bitmask(torch.from_numpy(extra[0]), 100)[0]
    assert row[1] and not row[9]
    r.token_ids += [6, -1]  # one known, one in flight: the next token is output index 2, still below 3
    assert host_inputs.extras(_ecfg(True), 100, (1,), None, [r])[1] == ("grammar",)
    r.token_ids.append(-1)
    assert host_inputs.extras(_ecfg(True), 100, (1,), None, [r]) == ([], ())


def test_a_forced_token_is_the_only_one_allowed():
    r = Request("r", [4, 5], SamplingParams(thinking_token_budget=2, min_tokens=5))
    r.think_force = [42, 43]
    extra, tag = host_inputs.extras(_ecfg(True), 100, (42,), None, [r])
    row = unpack_bitmask(torch.from_numpy(extra[0]), 100)[0]
    assert tag == ("grammar",) and row.nonzero().flatten().tolist() == [42]


def test_host_bound_rows_and_rows_that_never_speculate():
    assert not Request("a", [1], SamplingParams()).host_bound
    assert Request("a", [1], SamplingParams(frequency_penalty=0.1)).host_bound
    mt = Request("a", [1], SamplingParams(min_tokens=2))
    assert not mt.host_bound and mt.needs_host_inputs, "min_tokens depends on the count alone, not the tokens"
    b = Request("a", [1], SamplingParams(thinking_token_budget=10))
    b.think_in = True
    assert not b.host_bound and b.needs_host_inputs
    b.think_count = 8
    assert not b.host_bound
    b.think_count = 9
    assert b.host_bound, "one reasoning token left: the next one may be the last before the force"
    b.think_done = True
    assert not b.host_bound and not b.needs_host_inputs
    m = types.SimpleNamespace(done=False, is_terminated=lambda: m.done)
    g = Request("a", [1], SamplingParams(grammar=GrammarSpec("regex", "a")))
    g.matcher = m
    assert g.host_bound and g.needs_host_inputs
    m.done = True
    assert not g.host_bound and not g.needs_host_inputs


def test_the_engine_refuses_what_it_was_not_started_with():
    p = SamplingParams(grammar=GrammarSpec("regex", "a"))
    assert "--structured-outputs" in unsupported(p)
    assert unsupported(p, structured=True) is None
    for q in (SamplingParams(min_tokens=1), SamplingParams(thinking_token_budget=0)):
        assert "--structured-outputs" in unsupported(q) and unsupported(q, structured=True) is None
    pen = SamplingParams(repetition_penalty=1.1)
    assert "--penalties" in unsupported(pen, structured=True) and unsupported(pen, penalties=True) is None
    assert unsupported(SamplingParams(repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0)) is None


@pytest.mark.parametrize(
    ("body", "want"),
    [
        ({}, None),
        ({"response_format": {"type": "text"}}, None),
        ({"response_format": {"type": "json_object"}}, GrammarSpec("json_schema", OBJECT)),
        (
            {"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "integer"}}}},
            GrammarSpec("json_schema", '{"type": "integer"}'),
        ),
        (
            {"response_format": {"type": "json_schema", "schema": {"type": "null"}}},
            GrammarSpec("json_schema", '{"type": "null"}'),
        ),
        ({"guided_json": {"type": "string"}}, GrammarSpec("json_schema", '{"type": "string"}')),
        ({"guided_json": '{"type": "string"}'}, GrammarSpec("json_schema", '{"type": "string"}')),
        ({"guided_regex": "[a-z]+"}, GrammarSpec("regex", "[a-z]+")),
        ({"guided_choice": ["a.b", "c"]}, GrammarSpec("regex", r"(a\.b|c)")),
        ({"guided_grammar": 'root ::= "x"'}, GrammarSpec("ebnf", 'root ::= "x"')),
        ({"structured_outputs": {"regex": "x+"}}, GrammarSpec("regex", "x+")),
        ({"structured_outputs": {"json_object": True}}, GrammarSpec("json_schema", OBJECT)),
        ({"json_schema": '{"type": "boolean"}'}, GrammarSpec("json_schema", '{"type": "boolean"}')),
        ({"regex": "y"}, GrammarSpec("regex", "y")),
        ({"ebnf": 'root ::= "z"'}, GrammarSpec("ebnf", 'root ::= "z"')),
    ],
)
def test_every_constraint_field_maps_to_one_grammar(body, want):
    got = grammar_spec(body)
    assert got == want
    if got is not None and got.kind == "json_schema":
        json.loads(got.value)


@pytest.mark.parametrize(
    "body",
    [
        {"response_format": {"type": "json_object"}, "regex": "x"},
        {"guided_json": {}, "guided_regex": "x"},
        {"response_format": {"type": "structural_tag"}},
        {"response_format": {"type": "json_schema"}},
        {"response_format": "json"},
        {"guided_choice": []},
        {"guided_choice": ["a", 1]},
        {"guided_regex": ""},
        {"json_schema": "{not json"},
        {"structured_outputs": {"structural_tag": "x"}},
    ],
)
def test_a_malformed_or_doubled_constraint_is_refused(body):
    with pytest.raises(ValueError):  # noqa: PT011  the message varies with the field; the API answers any with 400
        grammar_spec(body)
