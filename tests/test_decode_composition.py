"""The decode step's composition across layers: a layer's FFN block leaves its stream mix to the next layer's KDA shell
call (hybrid.defers, the mix-in: the pending (post, comb, y) handed over in the shell's slot form), and the shells'
host forms (models/hybrid.py SHELL_HOST: kda_shell's, ffn_half's and dense_half's emulations, written as the served
forms) compose to the plain decode.

A tiny GLM-5.3-Flash engine (tests/test_glm5_next.py's random model at hidden 512, so the KDA shell takes its layers;
bf16, CPU, tp 1) serves the same prompts with the shells' host forms on and off: the same greedy tokens and logprobs
within bf16 noise, with the mix-in taken (the next layer's shell call received the pending FFN output). The positive
control: a hand-off that drops y (its slots zeroed) changes the output, so the agreement cannot hold because the
hand-off carried nothing. Prefill -> decode (states and caches carried) through the same engine."""

import pytest
import torch

from trnrt.kernels import kda_shell as KS
from trnrt.kernels import mk_kda_layer as MK
from trnrt.models import hybrid as HY

pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")


@pytest.fixture(scope="module")
def wide(tmp_path_factory):
    from tests import test_glm5_next as G

    path = str(tmp_path_factory.mktemp("decode_composition"))
    G.build(path, seed=21, hidden_size=512)
    return path


def serve(path, monkeypatch, host: bool, drop_y: bool = False, seen: list | None = None):
    from tests.test_glm5_next import engine, prompts
    from trnrt.engine.request import SamplingParams

    with monkeypatch.context() as mp:
        mp.setattr(HY, "SHELL_HOST", host)
        # the split fused KDA + MoE layer has no host form (its second call is a kernel): the shell call and the FFN
        # half as two calls, which hand over the same way
        mp.setattr(MK, "layer_takes", lambda model, layer, T: False)
        if drop_y:  # the mutation: the hand-off without the previous block's output
            real_pend = HY._shell_pend

            def dropped(layer, pend):
                out = real_pend(layer, pend)
                return None if out is None else (out[0], out[1], torch.zeros_like(out[2]))
            mp.setattr(HY, "_shell_pend", dropped)
        if seen is not None:
            real = KS.decode

            def spy(model, layer, S, positions, state_slot, pend=None):
                seen.append(pend is not None)
                return real(model, layer, S, positions, state_slot, pend=pend)
            mp.setattr(KS, "decode", spy)
        eng = engine(path, dtype=torch.bfloat16, max_num_seqs=4, max_prefill_tokens=16)
        try:
            sp = SamplingParams(max_new_tokens=4, ignore_eos=True, logprobs=0)
            reqs = [eng.add_request(p, sp) for p in prompts(31, [9, 14, 5, 12])]
            while eng.has_work():
                eng.step()
            return [(r.output_ids, [x[0] for x in r.logprobs]) for r in reqs]
        finally:
            eng.close()


def test_the_shells_compose_to_the_plain_decode(wide, monkeypatch):
    want = serve(wide, monkeypatch, host=False)
    seen = []
    got = serve(wide, monkeypatch, host=True, seen=seen)
    assert seen and any(seen), "no KDA shell call received a pending FFN output: the mix-in never ran"
    assert not all(seen), "every shell call had a pending mix: the step's first KDA layer should not"
    assert [g[0] for g in got] == [w[0] for w in want]
    for (_, a), (_, b) in zip(got, want):
        assert torch.allclose(torch.tensor(a), torch.tensor(b), rtol=0, atol=5e-2), (a, b)


def test_a_hand_off_without_y_changes_the_decode(wide, monkeypatch):
    want = serve(wide, monkeypatch, host=True)
    broken = serve(wide, monkeypatch, host=True, drop_y=True)
    assert all(b[0][0] == w[0][0] for b, w in zip(broken, want))  # every prompt's first token: prefill, no hand-off
    assert broken != want
