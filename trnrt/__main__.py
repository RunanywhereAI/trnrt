"""The trnRT command line.

    python -m trnrt serve --model DIR [--port 8000]       # two engines and the router in front of them
    python -m trnrt engine --model DIR --core-base 0       # one engine and its OpenAI API, no router
    python -m trnrt compile {shapes,decisions,capture,compile,strip} ...   # the compile cache (trnrt/compile/farm.py)

Every default is the served configuration of GLM-5.3-Flash on one trn2.48xlarge: two engines of 32 tensor-parallel
ranks each (logical NeuronCores 0-31 and 32-63 at LNC 2), 1M-token context, FP8 KV cache, 256-token pages, decode
batches of up to 8, 4,096-token prefill chunks, and 4 GiB per rank of host memory under the prefix cache. The bucket
ladders are graph shapes: a server finds its graphs in the compile cache only if `compile capture` ran with the same
arguments.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

GiB = 2**30


def _ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x)


def add_engine_args(ap: argparse.ArgumentParser) -> None:
    """The engine's arguments, with the served values as defaults."""
    g = ap.add_argument_group("engine")
    g.add_argument("--model", required=True, help="the checkpoint directory (or a Hugging Face repo id)")
    g.add_argument("--device", default="neuron", choices=["neuron", "cpu"])
    g.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    g.add_argument("--tp", type=int, default=32, help="tensor-parallel ranks per engine, one logical NeuronCore each")
    g.add_argument("--max-model-len", type=int, default=1_048_576)
    g.add_argument("--page-size", type=int, default=256)
    g.add_argument("--max-num-seqs", type=int, default=8, help="requests decoding at once per engine")
    g.add_argument("--max-prefill-tokens", type=int, default=4096, help="prefill tokens per step (the chunk size)")
    g.add_argument("--decode-buckets", type=_ints, default=(1, 2, 4, 6, 8))
    g.add_argument("--prefill-buckets", type=_ints, default=(1024, 2048, 4096))
    g.add_argument("--page-buckets", type=_ints, default=(512, 1024, 4096))
    g.add_argument("--prefill-page-buckets", type=_ints, default=(1024, 4096))
    g.add_argument("--prefill-page-min-rows", type=int, default=2048)
    g.add_argument("--kv-cache-gb", type=float, default=4.0, help="KV pool per rank, GiB")
    g.add_argument("--kv-cache-dtype", default="fp8", choices=["auto", "fp8"])
    g.add_argument("--state-checkpoints", type=int, default=16, help="state-pool rows kept for prefix checkpoints")
    g.add_argument("--host-cache-gb", type=float, default=4.0,
                   help="host memory under the prefix cache per rank, GiB (0: off)")
    g.add_argument("--moe-kernel", default="nki", choices=["nki", "xla"])
    g.add_argument("--num-layers", type=int, default=None, help="debugging: the first n layers only")


def engine_config(argv: list[str], core_base: int = 0):
    """The EngineConfig of the serve or engine arguments `argv` (also `compile capture`'s, after its `--`)."""
    import torch

    from .config import EngineConfig

    ap = argparse.ArgumentParser(prog="trnrt")
    add_engine_args(ap)
    a, _ = ap.parse_known_args(argv)  # the server's own arguments (--port, --engines, ...) are not the engine's
    return EngineConfig(
        model_path=a.model, device=a.device, dtype=getattr(torch, a.dtype), kv_cache_dtype=a.kv_cache_dtype,
        page_size=a.page_size, kv_cache_gb=a.kv_cache_gb, max_num_seqs=a.max_num_seqs, max_model_len=a.max_model_len,
        max_prefill_tokens=a.max_prefill_tokens, decode_batch_buckets=a.decode_buckets,
        prefill_token_buckets=a.prefill_buckets, page_buckets=a.page_buckets,
        prefill_page_buckets=a.prefill_page_buckets, prefill_page_min_rows=a.prefill_page_min_rows,
        host_cache_gb=a.host_cache_gb, state_checkpoints=a.state_checkpoints, tp=a.tp, tp_core_base=core_base,
        moe_kernel=a.moe_kernel, num_layers=a.num_layers)


def cmd_engine(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m trnrt engine", description="One engine and its OpenAI API.")
    add_engine_args(ap)
    ap.add_argument("--core-base", type=int, default=0, help="the engine's first logical NeuronCore")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--served-model-name", default="glm")
    a = ap.parse_args(argv)
    cfg = engine_config(argv, a.core_base)
    from .engine.engine import LLMEngine
    from .serve.api import build_app, serve

    t = time.perf_counter()
    eng = LLMEngine(cfg)
    print(f"engine up in {time.perf_counter() - t:.1f} s (cores {a.core_base}-{a.core_base + cfg.tp - 1})", flush=True)
    w = eng.warmup()
    print(f"bucket warmup {w['seconds']:.1f} s over {w['graphs']} graphs", flush=True)
    # The router keeps its connections to the engines alive and drops idle ones after 30 s: uvicorn closing them first
    # (its default keep-alive is 5 s) raced a request onto a closing connection.
    serve(build_app(eng, a.served_model_name), host=a.host, port=a.port, log_level="warning", timeout_keep_alive=600)
    return 0


def _dies_with_parent() -> None:
    """In a child before exec: SIGTERM when the parent exits (Linux PR_SET_PDEATHSIG), so an engine holding its
    NeuronCores never outlives the server that started it."""
    if sys.platform.startswith("linux"):
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def _without(argv: list[str], flag: str) -> list[str]:
    """argv without `flag VALUE` / `flag=VALUE`."""
    out, skip = [], False
    for x in argv:
        if skip:
            skip = False
        elif x == flag:
            skip = True
        elif not x.startswith(flag + "="):
            out.append(x)
    return out


def _wait_healthy(urls: list[str], procs: list[subprocess.Popen]) -> None:
    import httpx

    t0 = time.monotonic()
    todo = list(urls)
    while todo:
        for p in procs:
            if p.poll() is not None:
                raise SystemExit(f"an engine exited during start-up (status {p.returncode})")
        for u in list(todo):
            try:
                if httpx.get(u + "/health", timeout=2).status_code == 200:
                    todo.remove(u)
                    print(f"{u} ready after {time.monotonic() - t0:.0f} s", flush=True)
            except httpx.HTTPError:
                pass
        if todo:
            time.sleep(5)


def cmd_serve(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m trnrt serve",
                                 description="Engines on disjoint NeuronCores and the session-affine router in front.")
    add_engine_args(ap)
    ap.add_argument("--engines", type=int, default=2, help="engines of --tp ranks each, on consecutive cores")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000, help="the router's port; engine i listens on port + 1 + i")
    ap.add_argument("--served-model-name", default="glm")
    a = ap.parse_args(argv)
    # Two engines whose ranks load collective graphs at the same moment collide in the runtime's collectives
    # bootstrap: each graph's first call holds this host-wide lock (engine/start_lock.py).
    env = dict(os.environ)
    env.setdefault("TRNRT_START_LOCK", os.path.join("/tmp", f"trnrt-start-lock-{a.port}"))
    engine_argv = _without(argv, "--engines")
    procs: list[subprocess.Popen] = []
    urls = []
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))  # unwind through the finally below
    try:
        for i in range(a.engines):
            port = a.port + 1 + i
            cmd = [sys.executable, "-m", "trnrt", "engine", *engine_argv, "--core-base", str(i * a.tp),
                   "--port", str(port), "--host", "127.0.0.1"]
            procs.append(subprocess.Popen(cmd, env=dict(env, TRNRT_START_LOCK_OWNER=f"engine{i}"),
                                          preexec_fn=_dies_with_parent))
            urls.append(f"http://127.0.0.1:{port}")
        _wait_healthy(urls, procs)
        import uvicorn

        from .serve.router import Router, build_router_app

        uvicorn.run(build_router_app(Router(urls)), host=a.host, port=a.port)
    finally:
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for p in procs:
            try:
                p.wait(120)  # an engine's close stops its 31 workers
            except subprocess.TimeoutExpired:
                p.kill()
    return 0


def cmd_compile(argv: list[str]) -> int:
    from .compile import farm

    return farm.main(argv, engine_config=engine_config)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmds = {"serve": cmd_serve, "engine": cmd_engine, "compile": cmd_compile}
    if not argv or argv[0] not in cmds:
        print(__doc__.strip())
        return 0 if argv[:1] in ([], ["-h"], ["--help"]) else 2
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":  # spawned capture ranks re-import this module: only a direct run starts a command
    sys.exit(main())
