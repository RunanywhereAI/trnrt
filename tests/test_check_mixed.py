"""tools/check_mixed.py's compare (host only): a run against itself is the bytes gate's PASS (equal N/N, every
parity figure zero); a divergence at a clear margin FAILS (the gate can fail); the prompts are LONG_TEXT windows 997
tokens apart, their lengths cycling."""

from __future__ import annotations

import copy
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("check_mixed", os.path.join(HERE, "..", "tools", "check_mixed.py"))
CM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CM)


def _run(n_req: int = 3, n_out: int = 6) -> dict:
    reqs = []
    for k in range(n_req):
        ids = [100 + k * 10 + i for i in range(n_out)]
        # top-1 / top-2 logprobs on a 1/8 grid (bf16 logits): the reference's quantum is 0.125
        lps = [[-0.25, [t, t + 1], [-0.25, -0.375 - 0.125 * (i % 3)]] for i, t in enumerate(ids)]
        reqs.append({"prompt_len": 128 + k, "output_ids": ids, "text": "", "logprobs": lps,
                     "where": ["prefill chunk"] + ["decode call"] * (n_out - 1)})
    return {"forced": None, "requests": reqs}


def _write(tmp_path, name: str, run: dict) -> str:
    p = tmp_path / name
    p.write_text(json.dumps(run))
    return str(p)


def test_a_run_against_itself_passes_equal(tmp_path, capsys):
    a = _write(tmp_path, "a.json", _run())
    assert CM.compare(a, a) == 0
    out = capsys.readouterr().out
    assert "RESULT equal=3/3 token_agreement=1.0000" in out and "decode_parity=PASS" in out


def test_a_clear_divergence_fails(tmp_path, capsys):
    ref = _run()
    bad = copy.deepcopy(ref)
    r = bad["requests"][1]
    r["output_ids"][2] += 1000  # a different token where the reference's top-1 led by 2 grid steps (not a tie)
    r["logprobs"][2] = [-0.25, [r["output_ids"][2], 1], [-0.25, -1.0]]
    assert CM.compare(_write(tmp_path, "a.json", ref), _write(tmp_path, "b.json", bad)) == 1
    out = capsys.readouterr().out
    assert "RESULT equal=2/3" in out and "decode_parity=FAIL" in out


def test_prompts_are_windows_997_apart():
    class Tok:
        def __call__(self, text):
            return {"input_ids": list(range(5000))}

    ps = CM.prompt_sets(Tok(), [10, 20], 4)
    assert [len(p) for p in ps] == [10, 20, 10, 20] and [p[0] for p in ps] == [0, 997, 1994, 2991]


def test_permute_with_turns_is_refused(monkeypatch):
    monkeypatch.setattr("sys.argv", ["check_mixed.py", "--model", "x", "--out", "o.json", "--turns", "--permute", "reverse"])
    with pytest.raises(SystemExit, match="not built"):
        CM.main()
