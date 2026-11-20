"""The router's longest event-loop gap while it relays ~1 MB sessionless requests (#30).

The two stub engines run in processes of their own (each with its own loop and GIL), so neither their work nor their
JSON parsing can stall the loop this measures. A 1 ms sampler task records the largest gap between its wakeups while a
client sends ``--requests`` requests, one at a time and ``--pause`` apart, from another thread. The body goes out
pre-encoded: the numbers are about the router, not the client.

    python tools/bench_router_loop_gap.py --requests 10 --size 1000000
    python tools/bench_router_loop_gap.py --stream

Run it on the commit under test; numbers are comparable on one machine only (note the CPU and Python version with the
result). The router relays one stream per request on one loop, so any long block here pauses every relayed stream.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import socket
import statistics
import sys
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from trnrt.serve.router import Router, build_router_app

SAMPLE_S = 0.001


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stub_engine() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/chat/completions", response_model=None)
    async def chat(req: Request):
        body = await req.json()
        if body.get("stream"):

            async def stream():
                yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                yield b"data: [DONE]\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")
        return JSONResponse({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    return app


def _serve_stub(port: int) -> None:
    """A stub engine in its own process: its loop and its GIL are not the measured loop's."""
    uvicorn.run(_stub_engine(), host="127.0.0.1", port=port, log_level="warning")


class _StubProcess:
    """A stub engine in a process of its own, so neither its loop nor its JSON parsing can delay the measured loop."""

    def __init__(self, port: int):
        self.port = port
        self.process = multiprocessing.get_context("spawn").Process(target=_serve_stub, args=(port,), daemon=True)

    def start(self) -> None:
        self.process.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if not self.process.is_alive():
                raise RuntimeError(f"stub engine on port {self.port} exited during startup")
            try:
                if httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=0.2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.01)
        raise RuntimeError(f"stub engine on port {self.port} did not become ready in 10 s")

    def stop(self) -> None:
        self.process.terminate()
        self.process.join(timeout=10)


async def _serve(app: FastAPI, port: int) -> tuple[uvicorn.Server, asyncio.Task]:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    deadline = asyncio.get_running_loop().time() + 10
    while not server.started:
        if task.done():
            if task.cancelled():
                raise RuntimeError("router server exited during startup")
            cause = task.exception()
            if cause is None:
                raise RuntimeError("router server exited during startup")
            # Chained so the bind or configuration error that ended the server is not lost (and retrieved).
            raise RuntimeError("router server exited during startup") from cause
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError("router server did not become ready in 10 s")
        await asyncio.sleep(0.01)
    return server, task


async def _sample(stop: asyncio.Event, gaps: list[tuple[float, float, float]]) -> None:
    """Record each 1 ms wakeup's expected and actual loop time, and how late it woke: the delay interval is
    [expected, actual], ~0 on an idle loop."""
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        t0 = loop.time()
        await asyncio.sleep(SAMPLE_S)
        t1 = loop.time()
        gaps.append((t0 + SAMPLE_S, t1, max(0.0, t1 - t0 - SAMPLE_S)))


def _client(port: int, body: bytes, requests: int, stream: bool, pause: float) -> list[tuple[float, float, float]]:
    """Send the requests from a thread with its own loop; return each request's (start, end, seconds), monotonic."""
    windows = []
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    async def run() -> None:
        async with httpx.AsyncClient(timeout=120) as c:
            for _ in range(requests):
                t0 = time.monotonic()
                if stream:
                    async with c.stream("POST", url, content=body, headers={"content-type": "application/json"}) as r:
                        async for _chunk in r.aiter_raw():
                            pass
                        r.raise_for_status()
                else:
                    r = await c.post(url, content=body, headers={"content-type": "application/json"})
                    r.raise_for_status()
                windows.append((t0, time.monotonic(), time.perf_counter() - t0))
                await asyncio.sleep(pause)

    asyncio.run(run())
    return windows


async def _main(requests: int, size: int, stream: bool, pause: float, session: bool) -> None:
    ports = [_free_port() for _ in range(3)]
    engines = [_StubProcess(p) for p in ports[:2]]
    for e in engines:
        e.start()
    router = Router([f"http://127.0.0.1:{p}" for p in ports[:2]])
    server, task = await _serve(build_router_app(router, poll_s=0), ports[2])

    payload = {
        "model": "glm",
        "stream": stream,
        "messages": [{"role": "user", "content": "bench " * (size // 6)}],
    }
    if session:
        payload["session_id"] = "bench"  # session-routed: no prefix tree, the second cost this measures on its own
    body = json.dumps(payload).encode()
    await asyncio.to_thread(_client, ports[2], body, 1, stream, pause)  # warm up the pools

    gaps: list[tuple[float, float, float]] = []
    stop = asyncio.Event()
    sampler = asyncio.create_task(_sample(stop, gaps))
    windows = await asyncio.to_thread(_client, ports[2], body, requests, stream, pause)
    stop.set()
    await sampler

    server.should_exit = True
    await task
    for e in engines:
        e.stop()

    # A request's longest loop block: the largest delay whose [expected, actual] interval overlaps its window
    # (a delay that starts at the request's end or ends at its start counts as overlapping).
    per_request = [
        max((g for expected, actual, g in gaps if expected <= t1 and actual >= t0), default=0.0)
        for t0, t1, _ in windows
    ]
    per_request.sort()
    p95 = statistics.quantiles(per_request, n=20, method="inclusive")[18] if len(per_request) > 1 else per_request[0]
    seconds = [dt for _, _, dt in windows]
    every = sorted(g for _, _, g in gaps)
    print(
        f"python {'.'.join(map(str, sys.version_info[:3]))}, body {len(body):,} B, {'stream' if stream else 'json'}, "
        f"{'session' if session else 'sessionless'}, {requests} requests {pause * 1e3:.0f} ms apart"
    )
    print(
        f"longest loop block per request: p50 {statistics.median(per_request) * 1e3:6.2f} ms  "
        f"p95 {p95 * 1e3:6.2f} ms  max {max(per_request) * 1e3:6.2f} ms"
    )
    print(
        f"all samples: max {max(every) * 1e3:6.2f} ms, {sum(1 for g in every if g > 0.005)} of {len(every)} >5 ms"
        f"  |  request: p50 {statistics.median(seconds) * 1e3:7.1f} ms  max {max(seconds) * 1e3:7.1f} ms"
    )


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main() -> None:
    ap = argparse.ArgumentParser(prog="bench_router_loop_gap", allow_abbrev=False)
    ap.add_argument("--requests", type=_positive_int, default=10)
    ap.add_argument("--size", type=int, default=1_000_000, help="characters of message content")
    ap.add_argument("--pause", type=float, default=0.05, help="seconds between requests")
    ap.add_argument("--session", action="store_true", help="send session_id (skips the prefix tree)")
    ap.add_argument("--stream", action="store_true")
    args = ap.parse_args()
    asyncio.run(_main(args.requests, args.size, args.stream, args.pause, args.session))


if __name__ == "__main__":
    main()
