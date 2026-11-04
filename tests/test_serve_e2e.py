"""`python -m trnrt serve` end to end on CPU: the real command line in a subprocess, two engines of a tiny random
GLM-5.3-Flash-architecture model (tests/test_glm5_next.py build) behind the router, and the OpenAI client.

- A streaming chat request whose reply is one tool call, parsed into OpenAI tool_calls deltas.
- The session's second turn, on the same engine, resumed from the prefix cache (usage cached_tokens > 0).
- SIGTERM to serve: the router stops, then each engine; serve exits 143 (128 + SIGTERM) and no process it started
  outlives it.

The tool call is made deterministic. Greedy decoding of a fixed random model is deterministic, so the test first runs
one engine of the same configuration in this process (trnrt.__main__.engine_config of the same arguments) to learn the
ten tokens it generates after the prompt. The tokenizer then names the first nine of them `<tool_call>`, `get_weather`,
`<arg_key>`, `city`, `</arg_key>`, `<arg_value>`, `Paris`, `</arg_value>`, `</tool_call>`, and the tenth becomes the
end-of-sequence token, so the served engines write the call as text and stop after it. The renaming leaves the logits
alone: no token the prompts contain is renamed.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB = 384  # tests/test_glm5_next.py SMALL
GEN = 3  # the token the chat template ends a prompt with (its generation prompt), and starts an assistant turn with
CALL = [
    "<tool_call>",
    "get_weather",
    "<arg_key>",
    "city",
    "</arg_key>",
    "<arg_value>",
    "Paris",
    "</arg_value>",
    "</tool_call>",
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]
# Each message's content as is, an assistant turn opened by the generation prompt's token, and the generation prompt
# last: a turn's prompt is a prefix of the next turn's.
TEMPLATE = (
    "{%- for m in messages -%}{%- if m.role == 'assistant' -%}t3 {% endif -%}"
    "{%- if m.content -%}{{ m.content }} {% endif -%}{%- endfor -%}"
    "{%- if add_generation_prompt -%}t3{%- endif -%}"
)
# The engine arguments on CPU: fp32, pages of 4 tokens, one 64-token prefill bucket, decode batches of 1 and 2.
CPU_ARGS = [
    "--device",
    "cpu",
    "--dtype",
    "float32",
    "--tp",
    "1",
    "--page-size",
    "4",
    "--max-model-len",
    "256",
    "--max-num-seqs",
    "2",
    "--max-prefill-tokens",
    "64",
    "--decode-buckets",
    "1,2",
    "--prefill-buckets",
    "64",
    "--page-buckets",
    "64",
    "--prefill-page-buckets",
    "64",
    "--prefill-page-min-rows",
    "0",
    "--kv-cache-gb",
    "0.02",
    "--kv-cache-dtype",
    "auto",
    "--state-checkpoints",
    "8",
    "--host-cache-gb",
    "0",
    "--moe-kernel",
    "xla",
]
START_S = 150.0  # the router answering /health, both engines up


def _words(ids) -> str:
    return " ".join(f"t{i}" for i in ids)


def _scripted_prompt(engine_args: list[str]) -> tuple[list[int], list[int]]:
    """A 23-token user message and the 10 tokens one engine of `engine_args` generates greedily after it (with the
    generation prompt's token), all distinct and none in the prompt or the unknown token 0."""
    import trnrt.__main__ as cli
    from trnrt.engine.engine import LLMEngine
    from trnrt.engine.request import SamplingParams

    eng = LLMEngine(cli.engine_config(engine_args))
    try:
        eng.warmup()
        for seed in range(40):
            user = torch.randint(4, VOCAB, (23,), generator=torch.Generator().manual_seed(seed)).tolist()
            out = eng.generate([user + [GEN]], SamplingParams(max_new_tokens=10, ignore_eos=True))[0].output_ids
            if len(set(out)) == 10 and not set(out) & (set(user) | {GEN, 0}):
                return user, out
    finally:
        eng.close()
    pytest.fail("no prompt of 40 generates 10 distinct new tokens")


def _save_tokenizer(path: str, gen: list[int]) -> None:
    """Word-level "t0".."t383" with the generated tokens renamed (the module docstring), decoded without separators."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    names = {i: f"t{i}" for i in range(VOCAB)}
    names.update(zip(gen, CALL))
    tk = Tokenizer(models.WordLevel({w: i for i, w in names.items()}, unk_token="t0"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="t0", eos_token=names[gen[9]])
    tok.chat_template = TEMPLATE
    tok.save_pretrained(path)
    with open(os.path.join(path, "generation_config.json")) as f:
        g = json.load(f)
    g["eos_token_id"] = gen[9]  # the engine's end-of-sequence token (models/config.py reads it from here)
    with open(os.path.join(path, "generation_config.json"), "w") as f:
        json.dump(g, f)


def _free_port_run(n: int) -> int:
    """A port p with p .. p + n - 1 all free (serve's router on p, engine i on p + 1 + i)."""
    for _ in range(50):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            p = s.getsockname()[1]
        if p + n > 65535:
            continue
        try:
            socks = []
            for q in range(p, p + n):
                s = socket.socket()
                socks.append(s)
                s.bind(("127.0.0.1", q))
            return p
        except OSError:
            pass
        finally:
            for s in socks:
                s.close()
    raise RuntimeError("no run of free ports")


def _descendants(pid: int) -> list[int]:
    rows = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, check=True).stdout
    kids: dict[int, list[int]] = {}
    for line in rows.splitlines():
        c, p = (int(x) for x in line.split())
        kids.setdefault(p, []).append(c)
    out, todo = [], [pid]
    while todo:
        for c in kids.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_serve_end_to_end_on_cpu(tmp_path):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    glm = pytest.importorskip("tests.test_glm5_next")

    model = str(tmp_path / "model")
    glm.build(model, seed=1, index_topk=16)
    engine_args = ["--model", model, *CPU_ARGS]
    user, gen = _scripted_prompt(engine_args)
    _save_tokenizer(model, gen)
    tool = [i for i in range(4, VOCAB) if i not in set(gen) | set(user)][:6]  # turn 2's tool result

    port = _free_port_run(3)
    log_path = tmp_path / "serve.log"
    env = dict(os.environ, PYTHONPATH=ROOT, HF_HUB_OFFLINE="1", TRNRT_START_LOCK=str(tmp_path / "start-lock"))
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "trnrt",
                "serve",
                *engine_args,
                "--port",
                str(port),
                "--host",
                "127.0.0.1",
                "--served-model-name",
                "glm",
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def tail() -> str:
        return log_path.read_text(errors="replace")[-4000:]

    base = f"http://127.0.0.1:{port}"
    try:
        t0 = time.monotonic()
        while True:
            assert proc.poll() is None, f"serve exited (status {proc.returncode}) before the router answered:\n{tail()}"
            assert time.monotonic() - t0 < START_S, f"no router after {START_S:.0f} s:\n{tail()}"
            try:
                if httpx.get(base + "/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)

        client = openai.OpenAI(base_url=base + "/v1", api_key="none", max_retries=0, timeout=60)
        first = [{"role": "user", "content": _words(user)}]
        chunks = list(
            client.chat.completions.create(
                model="glm",
                messages=first,
                tools=TOOLS,
                stream=True,
                temperature=0,
                max_tokens=16,
                stream_options={"include_usage": True},
                extra_body={"session_id": "agent-1"},
            )
        )
        calls, content, finish = {}, "", None
        for ch in chunks:
            for c in ch.choices:
                content += c.delta.content or ""
                finish = c.finish_reason or finish
                for tc in c.delta.tool_calls or []:
                    d = calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
                    d["id"] = tc.id or d["id"]
                    d["name"] += (tc.function.name or "") if tc.function else ""
                    d["arguments"] += (tc.function.arguments or "") if tc.function else ""
        assert list(calls) == [0] and calls[0]["name"] == "get_weather", (calls, content, tail())
        assert json.loads(calls[0]["arguments"]) == {"city": "Paris"} and calls[0]["id"], calls
        assert content.strip() == "" and finish == "tool_calls", (content, finish)
        assert chunks[-1].usage is not None and chunks[-1].usage.prompt_tokens == len(user) + 1, chunks[-1].usage

        second = first + [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": calls[0]["id"],
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": calls[0]["arguments"]},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": calls[0]["id"], "content": _words(tool)},
        ]
        r = client.chat.completions.create(
            model="glm", messages=second, tools=TOOLS, temperature=0, max_tokens=4, extra_body={"session_id": "agent-1"}
        )
        assert r.usage.prompt_tokens == len(user) + 1 + len(tool) + 1, r.usage  # turn 1, the tool result, t3
        cached = r.usage.prompt_tokens_details.cached_tokens
        assert 0 < cached <= len(user) + 1, (r.usage, tail())  # resumed from turn 1's prompt-end checkpoint
        h = httpx.get(base + "/health", timeout=10).json()
        assert h["routed"]["new_session"] == 1 and h["routed"]["session"] == 1, h  # both turns on one engine

        kids = _descendants(proc.pid)
        assert len(kids) >= 2, kids  # the two engines
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(90) == 128 + signal.SIGTERM, tail()
        deadline = time.monotonic() + 10
        while any(_alive(p) for p in kids) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert [p for p in kids if _alive(p)] == [], tail()
    finally:  # whatever failed, nothing it started outlives the test (serve and its engines share its process group)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(10)
