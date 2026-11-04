"""Session-affine router over the data-parallel engines of one host (two tensor-parallel engines on a trn2.48xlarge).

Each worker is a full trnRT server (serve/api.py) on its own NeuronCores, with its own prefix cache.

A request that carries a session id (the body field `session_id`) is routed by its session alone:
- a known session goes to the worker it was pinned to, ALWAYS, whatever the load. An agent's next turn shares its whole
  context with the previous one, and only that worker's prefix cache holds it: at 150-230K tokens a move is a cold
  prefill of the whole context, which no load imbalance is worth;
- unless that worker is down: a refused connection or a 503 (from /health, polled every `poll_s`, or from a request)
  marks it down at once; a slow or failed /health only after `down_after` in a row (3), so one poll stalled by a long
  step or a GC pause moves no session. Then the session fails over to the least-loaded live worker, is pinned there,
  and the router logs it. If its home worker is back before the session COMPLETES a turn on the new one, it goes home
  (its context is still only there); after one completed turn it stays;
- a NEW session goes to the least-loaded live worker: in-flight requests plus the sessions pinned there and active in the
  last `session_idle_s` (each holds a long context in that worker's KV), then the KV in use (the worker's /v1/stats,
  polled) breaking ties.
A session's own side requests do not follow its pin. An agent client may ask for a title or a summary under its own
system prompt, without tools, with the session's id, while the main turns run: pinned, those requests overlapped the
main turns' prefills on one worker while the other idled. So each session request carries a key, (sys_hash,
first_hash): hashes of its leading system messages and of the first 4,096 characters of the first message after them
(_session_key: stable while a conversation grows; sys_hash differs under another system prompt), and whether it sends
tools:
- the session's key (or none known, or none computed): its pin, as above;
- another key with the session's sys_hash, or with tools: the same agent's new history (a compaction, a front-truncated
  history) or the main agent's first turn after a side request: the session is re-keyed. If its pin was made by a
  tool-less request (tentative: maybe a side request that came first) and this one sends tools, it is re-pinned like a
  new session, to the least-loaded worker; otherwise the pin stays (the worker holds its system part);
- another sys_hash without tools: a side request (a title or summary prompt has its own system block), served by the
  least-loaded live worker; the pin and key stay.
A client that sends no tools keeps its pin through compactions. One case stays open for such a client: if its first
request is a side request (another system block), its main turns are taken as side requests and go by load.
Session-routed requests skip the prefix tree: no match, no insert (both walk the request text in Python, ~1 MB of it for
a 200K-token context). The session map forgets sessions idle past `session_ttl_s` (4 x session_idle_s), and the least
recently used past `max_sessions` (100,000), on each poll and each new pin.

A request without a session id (most OpenAI clients send none) is routed by an approximate prefix tree of the request
text the router sent to each worker, locality first (the cache-aware policy of SGLang's model gateway):
1. If its longest match on a live worker covers at least `sticky_chars` characters (32,768, ~8K tokens) PAST its leading
   system messages, to that worker whatever the load: an agent's growing conversation stays where its KV is. The system
   messages do not count: one client's sessions all share them (its instructions, environment and file tree), so
   counting them would pin every new session to the worker that served the first.
2. Else if the load is imbalanced (max - min in-flight > balance_abs and max > balance_rel * min), to the least-loaded
   worker: load wins over a short or new prefix.
3. Else to the worker whose tree matches the longest prefix, if that match covers at least `cache_threshold` of the
   request; its prefix cache very likely still holds the KV.
4. Else to the least-loaded worker, the smallest tree (the most room for new prefixes) breaking ties.
A worker known to be down is never picked while another is up.

The tree is text-level, like SGLang's router, so it needs no tokenizer; it is pruned to `max_tree_chars` per worker by
evicting least-recently-used leaves (128M characters: above 32 sessions of a 1M-token context's ~4 MB of text, so long
sessions' prefixes are not evicted under load).

The router's own endpoints:
- GET /health: 200 {"status": "ok"} while every worker is up; 200 {"status": "degraded"} while some are (requests and
  sessions go to the live ones, so a load balancer in front keeps sending); 503 {"status": "down"} once none is. Each
  body names the workers down ("down": their URLs) and carries the routing counters. Up and down are the router's view
  (above): the /health polls and the requests' refused connections and 503s;
- GET /v1/models: the answer of the first live worker that gives one (least loaded first), 503 if none does.

    python -m trnrt.serve.router --worker-urls http://127.0.0.1:8111,http://127.0.0.1:8112 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import time

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

SESSION_FIELD = "session_id"  # the request body field naming the session (serve/api.py takes it too)
# TRNRT_ROUTER_LOG_SESSIONS=1, for debugging: a line per new pin and per side request, and the failover and return-home
# lines, each with the session's id, so a log reader can attribute requests to engines. Off (the default) the router
# logs no session id, which is the client's own identifier: no line per request, and the failover and return-home lines
# without it.
LOG_SESSIONS_ENV = "TRNRT_ROUTER_LOG_SESSIONS"
SESSION_KEY_CHARS = 4096
ROUTES = ("/v1/completions", "/v1/chat/completions")


class _Node:
    __slots__ = ("children", "text", "last", "parent")

    def __init__(self, text: str = "", parent: "_Node | None" = None):
        self.children: dict[str, _Node] = {}
        self.text = text
        self.last = 0.0
        self.parent = parent


class PrefixTree:
    """Character radix tree with LRU leaf pruning."""

    def __init__(self, max_chars: int):
        self.root = _Node()
        self.max_chars = max_chars
        self.size = 0

    def match(self, s: str) -> int:
        node, i = self.root, 0
        now = time.monotonic()
        while i < len(s):
            child = node.children.get(s[i])
            if child is None:
                break
            n = 0
            while n < len(child.text) and i + n < len(s) and child.text[n] == s[i + n]:
                n += 1
            child.last = now
            i += n
            if n < len(child.text):
                break
            node = child
        return i

    def insert(self, s: str) -> None:
        node, i = self.root, 0
        now = time.monotonic()
        while i < len(s):
            child = node.children.get(s[i])
            if child is None:
                leaf = _Node(s[i:], node)
                leaf.last = now
                node.children[s[i]] = leaf
                self.size += len(leaf.text)
                break
            n = 0
            while n < len(child.text) and i + n < len(s) and child.text[n] == s[i + n]:
                n += 1
            if n < len(child.text):  # split
                upper = _Node(child.text[:n], node)
                upper.last = now
                node.children[s[i]] = upper
                child.text = child.text[n:]
                child.parent = upper
                upper.children[child.text[0]] = child
                child = upper
            child.last = now
            i += n
            node = child
        while self.size > self.max_chars:
            self._evict_one()

    def _evict_one(self) -> None:
        leaves, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if not n.children and n is not self.root:
                leaves.append(n)
            stack.extend(n.children.values())
        if not leaves:
            self.size = 0
            return
        victim = min(leaves, key=lambda n: n.last)
        self.size -= len(victim.text)
        del victim.parent.children[victim.text[0]]


class Router:
    def __init__(self, workers: list[str], cache_threshold: float = 0.5, balance_abs: int = 32,
                 balance_rel: float = 1.5, max_tree_chars: int = 128_000_000, session_idle_s: float = 1800.0,
                 clock=time.monotonic, sticky_chars: int = 32_768, down_after: int = 3,
                 session_ttl_s: float | None = None, max_sessions: int = 100_000, log_sessions: bool | None = None):
        self.workers = workers
        # whether lines name sessions (LOG_SESSIONS_ENV; None: from the environment)
        self.log_sessions = os.environ.get(LOG_SESSIONS_ENV) == "1" if log_sessions is None else log_sessions
        self.trees = [PrefixTree(max_tree_chars) for _ in workers]
        self.load = [0] * len(workers)
        self.cache_threshold, self.balance_abs, self.balance_rel = cache_threshold, balance_abs, balance_rel
        self.sticky_chars = sticky_chars
        self.routed = {"sticky": 0, "cache": 0, "balance": 0, "smallest": 0, "session": 0, "new_session": 0,
                       "failover": 0, "home": 0, "side": 0, "rekey": 0}
        self.session_idle_s, self.clock = session_idle_s, clock
        self.session_ttl_s = session_ttl_s if session_ttl_s is not None else 4 * session_idle_s
        self.max_sessions = max_sessions
        self.sessions: dict[str, int] = {}  # session id -> the worker it is pinned to (insertion order = LRU order)
        self.seen: dict[str, float] = {}  # session id -> its last request (self.clock)
        self.moved: dict[str, list] = {}  # failed-over session -> [home worker, turns completed on the new one]
        self.keys: dict[str, tuple] = {}  # session id -> its history's key (_session_key: sys_hash, first_hash)
        self.tentative: set[str] = set()  # sessions pinned by a tool-less request (maybe a side request that came first)
        self.up = [True] * len(workers)  # False once down (see note_failure) until /health answers again
        self.fails = [0] * len(workers)  # consecutive soft failures (a slow or failed /health)
        self.down_after = down_after
        self.kv_used = [None] * len(workers)  # KV pages in use / total, from the worker's /v1/stats; None unknown

    @staticmethod
    def session_of(body: dict) -> str | None:
        sid = body.get(SESSION_FIELD)
        return sid if isinstance(sid, str) and sid else None

    def _live(self) -> list[int]:
        live = [i for i, ok in enumerate(self.up) if ok]
        return live or list(range(len(self.workers)))  # all down: try them anyway rather than refuse

    def active_sessions(self, w: int) -> int:
        cutoff = self.clock() - self.session_idle_s
        return sum(1 for s, i in self.sessions.items() if i == w and self.seen.get(s, 0.0) >= cutoff)

    def least_loaded(self) -> int:
        """The live worker with the fewest in-flight requests plus pinned active sessions, then the least KV in use."""
        return min(self._live(), key=lambda i: (self.load[i] + self.active_sessions(i),
                                                1.0 if self.kv_used[i] is None else self.kv_used[i], i))

    def pick_session(self, sid: str) -> int:
        w = self.sessions.get(sid)
        mv = self.moved.get(sid)
        if mv is not None and mv[1] == 0 and self.up[mv[0]] and mv[0] != w:
            # failed over, but no turn completed on the new worker yet: its context is still only at home, back up
            w = mv[0]
            del self.moved[sid]
            self.routed["home"] += 1
            print(f"trnrt.router: {self._session(sid)} back to worker {w} ({self.workers[w]}), up again", flush=True)
        elif w is not None and self.up[w]:
            self.routed["session"] += 1
        elif w is not None:
            old, w = w, self.least_loaded()
            self.routed["failover"] += 1
            self.moved[sid] = [mv[0] if mv is not None else old, 0]
            print(f"trnrt.router: {self._session(sid)} failed over from worker {old} ({self.workers[old]}, down) to "
                  f"worker {w} ({self.workers[w]})", flush=True)
        else:
            self.prune()  # a new pin: bound the map first (and on every poll), not on every request
            w = self.least_loaded()
            self.routed["new_session"] += 1
            if self.log_sessions:  # one line per pin, the last per session naming its worker
                print(f"trnrt.router: session {sid!r} pinned to worker {w} ({self.workers[w]})", flush=True)
        self.sessions.pop(sid, None)
        self.sessions[sid] = w  # re-inserted: the map's order is least recently used first
        self.seen[sid] = self.clock()
        return w

    def pick_keyed(self, sid: str, key: tuple | None, tools: bool) -> int:
        """A session request by its key (sys_hash, first_hash) and tools (see the module docstring): its pin, or a side
        request's least-loaded worker, or a new history re-keyed (re-pinned like a new session when the main agent's
        tools arrive on a tentative pin)."""
        known = self.keys.get(sid)
        if key is None or known is None or key == known:
            new = sid not in self.sessions
            if key is not None and known is None:
                self.keys[sid] = key
            w = self.pick_session(sid)
            if new and not tools:
                self.tentative.add(sid)
            elif tools:
                self.tentative.discard(sid)
            return w
        if tools or key[0] == known[0]:
            self.keys[sid] = key
            self.routed["rekey"] += 1
            if tools and sid in self.tentative:  # its pin came from a side request: place the main history anew
                self.tentative.discard(sid)
                self.sessions.pop(sid, None)
                self.moved.pop(sid, None)
            return self.pick_session(sid)
        self.routed["side"] += 1
        w = self.least_loaded()
        if self.log_sessions:
            print(f"trnrt.router: side request of session {sid!r} to worker {w} ({self.workers[w]})", flush=True)
        return w

    def _session(self, sid: str) -> str:
        """A log line's name for session `sid`: its id only under LOG_SESSIONS_ENV."""
        return f"session {sid!r}" if self.log_sessions else "a session"

    def turn_done(self, sid: str | None, w: int) -> None:
        """A request of session `sid` completed on worker w: once a failed-over session has a turn on its new worker,
        its context lives there and it stays."""
        mv = self.moved.get(sid) if sid is not None else None
        if mv is not None and self.sessions.get(sid) == w:
            mv[1] += 1
            if mv[0] != w:
                del self.moved[sid]

    def prune(self) -> None:
        """Forget sessions idle past session_ttl_s, and the least recently used past max_sessions."""
        cutoff = self.clock() - self.session_ttl_s
        for sid in [s for s in self.sessions if self.seen.get(s, 0.0) < cutoff]:
            self._forget(sid)
        while len(self.sessions) > self.max_sessions:
            self._forget(next(iter(self.sessions)))

    def _forget(self, sid: str) -> None:
        self.sessions.pop(sid, None)
        self.seen.pop(sid, None)
        self.moved.pop(sid, None)
        self.keys.pop(sid, None)
        self.tentative.discard(sid)

    def note_ok(self, w: int) -> None:
        if not self.up[w]:
            print(f"trnrt.router: worker {w} ({self.workers[w]}) is up", flush=True)
        self.up[w], self.fails[w] = True, 0

    def note_failure(self, w: int, hard: bool) -> None:
        """A refused connection or a 503 (hard) marks a worker down at once; a timeout or another error (soft) only
        after down_after in a row: one /health stalled by a long step or a GC pause must not move every session pinned
        there (each move is a cold prefill of a long context)."""
        self.fails[w] += 1
        if hard or self.fails[w] >= self.down_after:
            self.mark_down(w)

    def mark_down(self, w: int) -> None:
        if self.up[w]:
            print(f"trnrt.router: worker {w} ({self.workers[w]}) is down", flush=True)
        self.up[w] = False

    def pick(self, text: str, session: str | None = None, head: int = 0, key: tuple | None = None,
             tools: bool = False) -> int:
        """The worker for a request. `head`: the length of the text's leading part that every session shares (a chat
        request's system messages, _request_head): a match counts as sticky only past it, so a long shared system
        prompt does not pin every new session to the worker that served the first one. `key`, `tools`: a session
        request's _session_key and whether it sends tools (pick_keyed)."""
        if session is not None:
            return self.pick_keyed(session, key, tools)
        live = self._live()
        matches = [t.match(text) for t in self.trees]
        best = max(live, key=lambda i: (matches[i], -self.load[i]))
        lo, hi = min(self.load), max(self.load)
        if matches[best] >= head + self.sticky_chars:
            # locality first: a long shared prefix (an agent's growing conversation sent without a session id) stays
            # where its KV is, whatever the load
            self.routed["sticky"] += 1
            w = best
        elif hi - lo > self.balance_abs and hi > self.balance_rel * max(lo, 1):
            self.routed["balance"] += 1
            w = min(live, key=lambda i: self.load[i])
        elif text and matches[best] / len(text) >= self.cache_threshold:
            self.routed["cache"] += 1
            w = best
        else:
            self.routed["smallest"] += 1
            # Load first, then the smaller tree. Tree size alone degenerates once the trees reach max_tree_chars: the
            # worker that receives a request is pruned back below the other one's size, so it wins the next request
            # too, and only the balance rule (an imbalance over balance_abs) ever sends one elsewhere.
            w = min(live, key=lambda i: (self.load[i], self.trees[i].size))
        self.trees[w].insert(text)
        return w

    def status(self) -> str:
        """"ok" while every worker is up, "degraded" while some are, "down" once none is."""
        n = sum(self.up)
        return "ok" if n == len(self.workers) else "degraded" if n else "down"

    def health(self) -> dict:
        return {"workers": self.workers, "load": self.load, "routed": self.routed, "up": self.up, "fails": self.fails,
                "kv_used": self.kv_used, "sessions": [sum(1 for i in self.sessions.values() if i == w)
                                                      for w in range(len(self.workers))],
                "active_sessions": [self.active_sessions(w) for w in range(len(self.workers))],
                "sessions_known": len(self.sessions)}


def _kv_used(stats: dict) -> float | None:
    g = stats.get("gauges") or {}
    used, total = g.get("kv_pages_used"), g.get("kv_pages_total")
    return used / total if isinstance(used, (int, float)) and isinstance(total, (int, float)) and total > 0 else None


def _leading_system(msgs: list) -> int:
    """How many of a chat request's messages lead it as system (or developer) messages."""
    k = 0
    while k < len(msgs) and isinstance(msgs[k], dict) and msgs[k].get("role") in ("system", "developer"):
        k += 1
    return k


def _request_head(path: str, body: dict) -> int:
    """The length of _request_text's prefix through a chat request's leading system messages (0 for a prompt): the part
    every session of one client shares (its instructions, environment and file tree)."""
    if not path.endswith("chat/completions"):
        return 0
    msgs = body.get("messages") or []
    k = _leading_system(msgs)
    return len(json.dumps(msgs[:k], ensure_ascii=False)) - 1 if k else 0  # "[A, B" is a prefix of "[A, B, C]"


def _session_key(path: str, body: dict) -> tuple[str, str]:
    """A session request's history key, (sys_hash, first_hash): hashes of its leading system messages and of the first
    SESSION_KEY_CHARS characters of the first message after them (for a completion: "" and its prompt's first
    characters). Stable while an agent's conversation grows (both parts are the conversation's start); sys_hash differs
    under another system prompt (a title or summary request), first_hash under another conversation start (a
    compaction). No dump of the whole request (~1 MB at 200K tokens)."""
    h = lambda b: hashlib.sha1(b.encode("utf-8", "replace")).hexdigest()  # noqa: E731
    if not path.endswith("chat/completions"):
        p = body.get("prompt", "")
        return h(""), h((p if isinstance(p, str) else json.dumps(p))[:SESSION_KEY_CHARS])
    msgs = body.get("messages") or []
    k = _leading_system(msgs)
    first = msgs[k] if k < len(msgs) and isinstance(msgs[k], dict) else {}
    c = first.get("content")
    c = c if isinstance(c, str) else "" if c is None else json.dumps(c, ensure_ascii=False)
    return h(json.dumps(msgs[:k], ensure_ascii=False)), h(str(first.get("role", "")) + "\x00" + c[:SESSION_KEY_CHARS])


def _request_text(path: str, body: dict) -> str:
    if path.endswith("chat/completions"):
        return json.dumps(body.get("messages", []), ensure_ascii=False)
    p = body.get("prompt", "")
    return p if isinstance(p, str) else json.dumps(p)


def build_router_app(router: Router, poll_s: float = 2.0, health_timeout_s: float = 5.0) -> FastAPI:
    # A streaming request holds its connection to the worker until the last token, so the pool's connection cap is a cap
    # on requests in flight at every worker together. httpx's default (Limits(max_connections=100), shared by all hosts)
    # held a whole host at 100 requests in flight: the rest waited inside client.send() here, which clients measured as
    # time to first token. So no cap, and a keep-alive expiry that closes idle connections here before an engine does.
    client = httpx.AsyncClient(timeout=None, limits=httpx.Limits(
        max_connections=None, max_keepalive_connections=512, keepalive_expiry=30.0))

    async def poll() -> None:
        """Each worker's /health and /v1/stats (KV in use) every `poll_s` seconds, and the session map's pruning. A 200
        brings a worker up; a refused connection or a 503 marks it down at once; a timeout or another failure counts
        toward router.down_after in a row (Router.note_failure). A refused connect in a request also marks it down."""
        while True:
            for i, u in enumerate(router.workers):
                try:
                    code = (await client.get(u + "/health", timeout=health_timeout_s)).status_code
                    if code == 200:
                        router.note_ok(i)
                    else:
                        router.note_failure(i, hard=code == 503)
                except httpx.ConnectError:
                    router.note_failure(i, hard=True)
                except httpx.HTTPError:
                    router.note_failure(i, hard=False)
                if router.up[i]:
                    try:
                        s = await client.get(u + "/v1/stats", timeout=health_timeout_s)
                        router.kv_used[i] = _kv_used(s.json()) if s.status_code == 200 else None
                    except (httpx.HTTPError, ValueError):
                        router.kv_used[i] = None
            router.prune()
            await asyncio.sleep(poll_s)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = asyncio.create_task(poll()) if poll_s > 0 else None
        try:
            yield
        finally:
            if task is not None:
                task.cancel()

    app = FastAPI(title="trnRT router", lifespan=lifespan)

    @app.get("/health")
    async def health():
        """200 while any worker is up ("ok" or "degraded"), 503 once none is (module docstring)."""
        status = router.status()
        down = [u for u, ok in zip(router.workers, router.up) if not ok]
        return JSONResponse({"status": status, "down": down, **router.health()},
                            status_code=503 if status == "down" else 200)

    @app.get("/v1/models")
    async def models():
        """Every worker serves the same model: the first live one that answers, least loaded first. A refused
        connection marks a worker down, as in a request."""
        for w in sorted(router._live(), key=lambda i: (router.load[i], i)):
            try:
                r = await client.get(router.workers[w] + "/v1/models", timeout=health_timeout_s)
            except httpx.ConnectError:
                router.note_failure(w, hard=True)
                continue
            except httpx.HTTPError:
                continue
            if r.status_code == 200:
                with contextlib.suppress(ValueError):
                    return JSONResponse(r.json())
        raise HTTPException(status_code=503, detail="no engine answered /v1/models")

    async def proxy(path: str, http: Request):
        body = await http.json()
        sid = router.session_of(body)
        key = _session_key(path, body) if sid is not None else None
        tools = bool(body.get("tools"))
        w = router.pick(_request_text(path, body) if sid is None else "", session=sid,
                        head=_request_head(path, body) if sid is None else 0, key=key, tools=tools)
        # An unreachable worker, or one that answers 503 (a failed engine refuses every request before it queues it,
        # serve/api.py): note it, re-pick once (a session fails over if it went down).
        for attempt in (0, 1):
            url = router.workers[w] + path
            router.load[w] += 1
            try:
                if body.get("stream"):
                    resp = await client.send(client.build_request("POST", url, json=body), stream=True)
                    if resp.status_code == 503:
                        try:
                            await resp.aread()
                        finally:
                            await resp.aclose()
                        r = resp
                else:
                    r = await client.post(url, json=body)
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                router.load[w] -= 1
                router.note_failure(w, hard=isinstance(e, httpx.ConnectError))  # refused: a dead port
                if attempt == 1 or all(not ok for ok in router.up):
                    raise HTTPException(status_code=503, detail=f"worker {w} unreachable")
                w = router.pick(_request_text(path, body) if sid is None else "", session=sid,
                                head=_request_head(path, body) if sid is None else 0, key=key, tools=tools)
                continue
            except BaseException:
                router.load[w] -= 1
                raise
            if (resp if body.get("stream") else r).status_code == 503:
                router.load[w] -= 1
                router.note_failure(w, hard=True)
                if attempt == 1 or all(not ok for ok in router.up):
                    return Response(r.content, status_code=503, media_type=r.headers.get("content-type"))
                w = router.pick(_request_text(path, body) if sid is None else "", session=sid,
                                head=_request_head(path, body) if sid is None else 0, key=key, tools=tools)
                continue
            break
        if body.get("stream"):
            ww = w

            async def relay():
                done = False
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                    done = True
                finally:
                    await resp.aclose()  # closing upstream aborts the request on the worker
                    router.load[ww] -= 1
                    if done and resp.status_code == 200:
                        router.turn_done(sid, ww)

            return StreamingResponse(relay(), status_code=resp.status_code,
                                     media_type=resp.headers.get("content-type", "text/event-stream"))
        router.load[w] -= 1
        if r.status_code == 200:
            router.turn_done(sid, w)
        return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))

    def route(path: str):
        async def handler(http: Request):  # annotated so FastAPI passes the request
            return await proxy(path, http)

        return handler

    for path in ROUTES:
        app.add_api_route(path, route(path), methods=["POST"])
    return app


def main() -> None:
    ap = argparse.ArgumentParser(prog="trnrt.serve.router", allow_abbrev=False)
    ap.add_argument("--worker-urls", required=True,
                    help="comma list of the engines' base URLs, e.g. http://127.0.0.1:8111,http://127.0.0.1:8112")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--timeout-keep-alive", type=int, default=75,
                    help="seconds an idle client connection stays open (above aiohttp's 15 s pool keep-alive)")
    args = ap.parse_args()
    import uvicorn

    router = Router([u.rstrip("/") for u in args.worker_urls.split(",") if u])
    uvicorn.run(build_router_app(router), host=args.host, port=args.port, timeout_keep_alive=args.timeout_keep_alive)


if __name__ == "__main__":
    main()
