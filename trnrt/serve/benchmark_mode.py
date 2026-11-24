"""The announcement of the benchmark mode (`--spec-synth-al`, engine/spec_synth.py): synthetic acceptance.

In that mode an accepted draft position emits the draft token whether or not the target agrees, so the text a request
returns is not the model's. The mode exists only to compare serving throughput with InferenceX's published runs, which
fix the acceptance length the same way. A run in it must never be read for quality, so the server says so wherever a
client or an operator looks:

- a WARNING when the engine's app starts and when the router first sees an engine in the mode, and a status line from
  `trnrt serve`;
- the header `x-trnrt-synthetic-acceptance: <mean acceptance length>` on every HTTP response of the engine, and on every
  response of the router once its polls have seen an engine in the mode;
- a `benchmark_mode` block in GET /health (engine and router). GET /v1/stats already carries the length as
  `gauges.spec.synth_al`, which the accuracy tools read to refuse a run (benchmarks/accuracy).

All of it is host code: nothing here is traced, so no graph changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

HEADER = "x-trnrt-synthetic-acceptance"


def banner(al: float) -> str:
    """The one-line warning for logs and status lines."""
    return (
        f"BENCHMARK MODE: synthetic acceptance at mean acceptance length {al:g}. Accepted drafts are emitted "
        "unverified, so the text this server returns is not the model's. Use it only to compare serving throughput "
        "with a published run at the same acceptance length, never to judge output quality."
    )


def health_block(lengths: list[float]) -> dict:
    """GET /health's `benchmark_mode` block: the configured mean acceptance length(s)."""
    return {
        "synthetic_acceptance_length": lengths[0] if len(lengths) == 1 else lengths,
        "drafts_verified": False,
        "use": "throughput comparison only; output text is not the model's",
    }


def header_value(lengths: list[float]) -> str:
    return ",".join(f"{x:g}" for x in lengths)


class HeaderMiddleware:
    """Pure ASGI middleware that adds HEADER to every HTTP response while `value()` is not None.

    Pure ASGI rather than Starlette's BaseHTTPMiddleware: that one wraps the app's response in its own stream, which
    changes how a client disconnect reaches a streaming handler, and the API and the router both act on disconnects
    (they abort the request on the engine)."""

    def __init__(self, app, value: Callable[[], str | None]):
        self.app = app
        self.value = value

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_header(message):
            if message["type"] == "http.response.start":
                v = self.value()
                if v is not None:
                    headers = [h for h in message.get("headers", []) if h[0].lower() != HEADER.encode()]
                    message = {**message, "headers": [*headers, (HEADER.encode(), v.encode())]}
            await send(message)

        await self.app(scope, receive, send_with_header)
