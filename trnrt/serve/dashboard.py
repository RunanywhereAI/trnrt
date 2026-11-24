"""GET /dashboard: one self-contained page (inline CSS and JavaScript, no CDN, so it works where the host has no
internet access) that polls GET /v1/stats every 2 s and draws it. serve/stats.py defines every number on it. An engine
serves it for itself; the router serves it too, with a link to each engine (`?engine=i`), since the engines listen on
127.0.0.1 only."""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TrnRT serving</title>
<style>
:root {
  --bg: #f6f7f9; --panel: #ffffff; --ink: #14171c; --muted: #5d6673; --line: #e3e6eb; --grid: #eceef2;
  --accent: #2563eb; --c1: #2563eb; --c2: #d97706; --c3: #059669; --c4: #9333ea; --bad: #dc2626; --ok: #16a34a;
  --warn-bg: #fff7e6; --warn-ink: #8a5300;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0f1216; --panel: #171b21; --ink: #e7eaee; --muted: #9aa3ae; --line: #262c35; --grid: #20262e;
    --accent: #60a5fa; --c1: #60a5fa; --c2: #fbbf24; --c3: #34d399; --c4: #c084fc; --bad: #f87171; --ok: #4ade80;
    --warn-bg: #2a2110; --warn-ink: #f5c26b;
  }
}
:root[data-theme="dark"] {
  --bg: #0f1216; --panel: #171b21; --ink: #e7eaee; --muted: #9aa3ae; --line: #262c35; --grid: #20262e;
  --accent: #60a5fa; --c1: #60a5fa; --c2: #fbbf24; --c3: #34d399; --c4: #c084fc; --bad: #f87171; --ok: #4ade80;
  --warn-bg: #2a2110; --warn-ink: #f5c26b;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.wrap { max-width: 1400px; margin: 0 auto; padding: 16px; }
header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 16px; margin-bottom: 14px; }
h1 { font-size: 18px; margin: 0; font-weight: 650; }
h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin: 22px 0 8px;
  font-weight: 600; }
.sub { color: var(--muted); }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--muted); margin-right: 6px; }
.dot.live { background: var(--ok); } .dot.down { background: var(--bad); }
.spacer { flex: 1; }
button { background: var(--panel); color: var(--ink); border: 1px solid var(--line); border-radius: 6px;
  padding: 3px 10px; font: inherit; cursor: pointer; }
.tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 10px; }
.tile { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; min-width: 0; }
.tile .k { color: var(--muted); font-size: 12px; }
.tile .v { font-size: 24px; font-weight: 650; font-variant-numeric: tabular-nums; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }
.tile .s { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 10px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; font-size: 12px; }
th:first-child, td:first-child { text-align: left; }
tr:last-child td { border-bottom: none; }
td.l, th.l { text-align: left; }
.grp { border-left: 1px solid var(--line); }
.charts { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 10px; }
@media (max-width: 480px) { .charts { grid-template-columns: 1fr; } }
.chart { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px 6px; }
.chart .t { display: flex; justify-content: space-between; gap: 8px; font-weight: 600; font-size: 13px; }
.chart .lg { color: var(--muted); font-weight: 400; font-size: 12px; }
.chart .lg i { display: inline-block; width: 10px; height: 3px; vertical-align: middle; margin: 0 4px 0 10px; }
canvas { width: 100%; height: 180px; display: block; }
.note { color: var(--muted); font-size: 12px; margin-top: 18px; }
.warn { background: var(--warn-bg); color: var(--warn-ink); border-radius: 6px; padding: 6px 10px; margin: 8px 0 0; }
.prod { padding: 10px 12px; }
.prod .row { display: flex; flex-wrap: wrap; gap: 6px 18px; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
.bad { color: var(--bad); } .ok { color: var(--ok); }
a { color: var(--accent); }
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1 id="title">TrnRT serving</h1>
  <span class="sub" id="subtitle"></span>
  <span class="sub" id="engines"></span>
  <span class="spacer"></span>
  <span class="sub"><span class="dot" id="dot"></span><span id="status">connecting</span></span>
  <button id="theme" type="button" title="Light / dark">Theme</button>
</header>

<div class="tiles" id="tiles"></div>

<h2>Rates and percentiles by window</h2>
<div class="panel"><table id="win"></table></div>

<h2>Last 30 minutes (10-s points)</h2>
<div class="charts" id="charts"></div>

<h2>Recent requests</h2>
<div class="panel"><table id="recent"></table></div>

<h2>Data collection</h2>
<div class="panel prod" id="prod"></div>

<p class="note">
All times are measured inside the server, from when the engine receives a request (after the chat template and
tokenization) to its last token. They exclude the client's network round trip, so a client sees a longer TTFT and
end-to-end time. TTFT = first token - arrival. Decode tok/s is per stream: (output tokens - 1) ÷ (last token - first
token); p10 is the slowest tenth of streams. RPM, TPM and the percentiles count requests by when they finished. Output
tok/s (aggregate) counts tokens as the engine hands them out. TPM total = input + output; cached is the part of the input
served from the prefix cache. "Since start" is this engine process; "today" (UTC) includes earlier processes' records
when the production log is on. Each engine has its own numbers: behind the router, pick the engine above.
</p>
</div>

<script>
"use strict";
const $ = (id) => document.getElementById(id);
const POLL_MS = 2000;
const NA = "\u2013";  // a value not known yet
let last = null;

// ---- theme ----
function setTheme(t) {
  if (t) document.documentElement.setAttribute("data-theme", t);
  else document.documentElement.removeAttribute("data-theme");
  try { t ? localStorage.setItem("trnrt-theme", t) : localStorage.removeItem("trnrt-theme"); } catch (e) {}
  if (last) draw(last);
}
try { const t = localStorage.getItem("trnrt-theme"); if (t) document.documentElement.setAttribute("data-theme", t); } catch (e) {}
$("theme").onclick = () => {
  const dark = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim() === "#0f1216";
  setTheme(dark ? "light" : "dark");
};

// ---- formatting ----
const nf = (x, d = 1) => x == null || !isFinite(x) ? NA : x.toLocaleString(undefined, {minimumFractionDigits: d, maximumFractionDigits: d});
const big = (x) => {
  if (x == null || !isFinite(x)) return NA;
  const a = Math.abs(x);
  if (a >= 1e9) return nf(x / 1e9, 2) + "B";
  if (a >= 1e6) return nf(x / 1e6, 2) + "M";
  if (a >= 1e4) return nf(x / 1e3, 1) + "K";
  return nf(x, 0);
};
const sec = (x) => x == null ? NA : x < 10 ? nf(x, 2) + " s" : x < 100 ? nf(x, 1) + " s" : nf(x, 0) + " s";
const dur = (s) => {
  if (s == null) return NA;
  s = Math.floor(s);
  const d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600), m = Math.floor(s % 3600 / 60);
  return d ? `${d}d ${h}h` : h ? `${h}h ${m}m` : m ? `${m}m ${s % 60}s` : `${s}s`;
};
const clock = (t, secs) => new Date(t * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: secs ? "2-digit" : undefined, hour12: false});
const bytes = (n) => n >= 1 << 30 ? nf(n / (1 << 30), 2) + " GiB" : n >= 1 << 20 ? nf(n / (1 << 20), 1) + " MiB" : n >= 1024 ? nf(n / 1024, 0) + " KiB" : n + " B";
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

// ---- tiles ----
function tile(k, v, s) {
  return `<div class="tile"><div class="k">${esc(k)}</div><div class="v">${v}</div><div class="s">${s || "&nbsp;"}</div></div>`;
}
function tiles(d) {
  const g = d.gauges, w1 = d.windows["1m"], w5 = d.windows["5m"], st = d.windows.start;
  const cap = g.cap != null ? `cap ${g.cap} of ${g.seats} seats` : `${g.seats} seats, no cap`;
  const out = [
    tile("Running / waiting", `${g.running} / ${g.waiting}`, cap),
    tile("Requests / min", nf(w1.rpm), `5 min ${nf(w5.rpm)} · since start ${nf(st.rpm)}`),
    tile("Output tokens / min", big(w1.tpm.output), `5 min ${big(w5.tpm.output)}`),
    tile("Input tokens / min", big(w1.tpm.input), `cache hit ${nf(w1.cache_hit_pct, 0)}% · 5 min ${big(w5.tpm.input)}`),
    tile("Output tok/s (aggregate)", nf(w1.out_tok_s), `5 min ${nf(w5.out_tok_s)}`),
    tile("Tokens processed", big(d.totals.processed), `in ${big(d.totals.tokens.prompt)} · out ${big(d.totals.tokens.output)}`),
    tile("KV cache", nf(g.kv_used_pct, 0) + "%", `${big(g.kv_pages_used)} / ${big(g.kv_pages_total)} pages`),
    tile("Uptime", dur(d.uptime_s), `requests ok ${big(d.totals.requests.ok)} · aborted ${d.totals.requests.aborted} · failed ${d.totals.requests.failed}`),
  ];
  if (g.spec) out.push(tile("Spec acceptance", g.spec.acceptance == null ? NA : nf(100 * g.spec.acceptance, 0) + "%",
    `${esc(g.spec.method)} k=${g.spec.k} · ${nf(g.spec.accept_length, 2)} tokens/verify`));
  const sc = g.scheduling || {};
  out.push(tile("Scheduling", esc(sc.policy || "fcfs") + (sc.priority_scheduling ? " + priority" : ""),
    `eviction ${esc(sc.eviction_policy || "lru")} · queue limit ${sc.max_queued_requests ?? "none"} · preempted ${big(g.preemptions)}`));
  out.push(tile("Model", esc(g.model || NA), `TP ${g.tp} · ${big(g.max_model_len)} ctx`));
  $("tiles").innerHTML = out.join("");
  $("subtitle").textContent = `${g.model} · ${cap}`;
  const rt = d.router;
  $("engines").innerHTML = rt ? "engine " + rt.engines.map((e, i) => i === rt.engine ? `<b>${i}</b>`
    : `<a href="?engine=${i}">${i}</a>${e.up ? "" : " (down)"}`).join(" · ") : "";
}

// ---- window table ----
function windows(d) {
  const names = [["1m", "1 min"], ["5m", "5 min"], ["15m", "15 min"], ["start", "since start"], ["today", "today (UTC)"]];
  let h = `<tr><th class="l">window</th><th>requests</th><th>RPM</th>
    <th class="grp">input TPM</th><th>cached</th><th>output TPM</th><th>total TPM</th><th>out tok/s</th>
    <th class="grp">TTFT p50</th><th>p90</th><th>p99</th><th>mean</th>
    <th class="grp">decode p10</th><th>p50</th><th>p90</th><th>mean</th>
    <th class="grp">e2e p50</th><th>p90</th><th>p99</th></tr>`;
  for (const [k, label] of names) {
    const w = d.windows[k];
    if (!w) continue;
    const t = w.ttft, dc = w.decode_tps, e = w.e2e;
    const lbl = k === "today" && w.first ? `${label}<span class="sub"> from ${clock(w.first)}</span>` : label;
    h += `<tr><td>${lbl}</td><td>${big(w.ok)}</td><td>${nf(w.rpm)}</td>
      <td class="grp">${big(w.tpm.input)}</td><td>${nf(w.cache_hit_pct, 0)}%</td><td>${big(w.tpm.output)}</td>
      <td>${big(w.tpm.total)}</td><td>${nf(w.out_tok_s)}</td>
      <td class="grp">${sec(t.p50)}</td><td>${sec(t.p90)}</td><td>${sec(t.p99)}</td><td>${sec(t.mean)}</td>
      <td class="grp">${nf(dc.p10)}</td><td>${nf(dc.p50)}</td><td>${nf(dc.p90)}</td><td>${nf(dc.mean)}</td>
      <td class="grp">${sec(e.p50)}</td><td>${sec(e.p90)}</td><td>${sec(e.p99)}</td></tr>`;
  }
  $("win").innerHTML = h;
}

// ---- charts ----
const CHARTS = [
  {id: "c_rw", title: "Running and waiting", unit: "", lines: [["running", "running (mean)", "--c1"], ["waiting", "waiting (mean)", "--c2"]], cap: true},
  {id: "c_rpm", title: "Requests / min", unit: "", lines: [["rpm", "RPM", "--c1"]]},
  {id: "c_tps", title: "Output tok/s (aggregate)", unit: "", lines: [["out_tok_s", "tok/s", "--c3"]]},
  {id: "c_ttft", title: "TTFT (s)", unit: "s", lines: [["ttft_p50", "p50", "--c1"], ["ttft_p90", "p90", "--c2"]]},
  {id: "c_dec", title: "Decode tok/s per stream", unit: "", lines: [["decode_p50", "p50", "--c3"], ["decode_p10", "p10", "--c4"]]},
];
function setupCharts() {
  $("charts").innerHTML = CHARTS.map((c) => `<div class="chart"><div class="t"><span>${c.title}</span><span class="lg">${
    c.lines.map((l) => `<i style="background:var(${l[2]})"></i>${l[1]}`).join("") +
    (c.cap ? `<i style="background:var(--muted)"></i>cap` : "")}</span></div><canvas id="${c.id}"></canvas></div>`).join("");
  for (const c of CHARTS) {
    const cv = $(c.id);
    cv.addEventListener("mousemove", (ev) => { c.hover = ev.offsetX; if (last) drawChart(c, last); });
    cv.addEventListener("mouseleave", () => { c.hover = null; if (last) drawChart(c, last); });
  }
}
function niceMax(v) {
  if (!(v > 0)) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v))), m = v / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
}
function css(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
function drawChart(c, d) {
  const cv = $(c.id), dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight;
  if (!W) return;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  }
  const x = cv.getContext("2d");
  x.setTransform(dpr, 0, 0, dpr, 0, 0);
  x.clearRect(0, 0, W, H);
  const s = d.series, n = s.length;
  const L = 44, R = 8, T = 8, B = 22, pw = W - L - R, ph = H - T - B;
  let vmax = 0;
  for (const p of s) for (const l of c.lines) if (p[l[0]] != null) vmax = Math.max(vmax, p[l[0]]);
  const capv = c.cap ? d.gauges.cap ?? d.gauges.seats : null;
  if (capv != null) vmax = Math.max(vmax, capv);
  const ymax = niceMax(vmax * 1.05);
  const X = (i) => L + (n <= 1 ? 0 : i * pw / (n - 1)), Y = (v) => T + ph - v / ymax * ph;
  x.font = "11px -apple-system, Segoe UI, Roboto, sans-serif";
  x.fillStyle = css("--muted"); x.strokeStyle = css("--grid"); x.lineWidth = 1;
  x.textAlign = "right"; x.textBaseline = "middle";
  for (let k = 0; k <= 4; k++) {
    const v = ymax * k / 4, yy = Math.round(Y(v)) + 0.5;
    x.beginPath(); x.moveTo(L, yy); x.lineTo(W - R, yy); x.stroke();
    x.fillText(v >= 1000 ? big(v) : +v.toFixed(ymax < 5 ? 2 : ymax < 20 ? 1 : 0) + "", L - 6, yy);
  }
  x.textAlign = "center"; x.textBaseline = "top";
  for (let i = 0; i < n; i += 30) x.fillText(clock(s[i].t), X(i), H - B + 6);
  if (capv != null) {
    x.strokeStyle = css("--muted"); x.setLineDash([4, 4]);
    x.beginPath(); x.moveTo(L, Y(capv)); x.lineTo(W - R, Y(capv)); x.stroke(); x.setLineDash([]);
  }
  for (const l of c.lines) {
    x.strokeStyle = css(l[2]); x.lineWidth = 1.8; x.beginPath();
    let pen = false;
    for (let i = 0; i < n; i++) {
      const v = s[i][l[0]];
      if (v == null) { pen = false; continue; }
      if (!pen) { x.moveTo(X(i), Y(v)); pen = true; } else x.lineTo(X(i), Y(v));
    }
    x.stroke();
    for (let i = 0; i < n; i++) {  // isolated points would otherwise not show
      const v = s[i][l[0]], a = i > 0 ? s[i - 1][l[0]] : null, b = i < n - 1 ? s[i + 1][l[0]] : null;
      if (v != null && a == null && b == null) { x.fillStyle = css(l[2]); x.fillRect(X(i) - 1.5, Y(v) - 1.5, 3, 3); }
    }
  }
  if (c.hover != null && c.hover >= L && c.hover <= W - R && n > 1) {
    const i = Math.round((c.hover - L) / pw * (n - 1)), p = s[i];
    x.strokeStyle = css("--muted"); x.lineWidth = 1;
    x.beginPath(); x.moveTo(X(i) + 0.5, T); x.lineTo(X(i) + 0.5, T + ph); x.stroke();
    const txt = clock(p.t, true) + "  " + c.lines.map((l) => `${l[1]} ${p[l[0]] == null ? NA : nf(p[l[0]], c.unit === "s" ? 2 : 1)}`).join("  ");
    x.font = "11px -apple-system, Segoe UI, Roboto, sans-serif";
    const tw = x.measureText(txt).width + 12, tx = Math.min(Math.max(X(i) - tw / 2, L), W - R - tw);
    x.fillStyle = css("--panel"); x.strokeStyle = css("--line");
    x.fillRect(tx, T, tw, 18); x.strokeRect(tx + 0.5, T + 0.5, tw - 1, 17);
    x.fillStyle = css("--ink"); x.textAlign = "left"; x.textBaseline = "middle"; x.fillText(txt, tx + 6, T + 9);
  }
}

// ---- recent requests ----
function recent(d) {
  let h = `<tr><th class="l">finished</th><th class="l">request</th><th>prompt</th><th>cached</th><th>output</th>
    <th>TTFT</th><th>decode tok/s</th><th>ITL</th><th>e2e</th><th>running / waiting at arrival</th><th class="l">finish</th><th class="l">stream</th></tr>`;
  for (const r of d.recent) {
    const ch = r.prompt ? Math.round(100 * r.cached / r.prompt) : 0;
    const fin = ["stop", "length", "tool_calls"].includes(r.finish) ? esc(r.finish) : `<span class="bad">${esc(r.finish)}</span>`;
    h += `<tr><td>${clock(r.done, true)}</td><td class="l mono">${esc((r.rid || "").slice(0, 8))}</td>
      <td>${big(r.prompt)}</td><td>${big(r.cached)} <span class="sub">(${ch}%)</span></td><td>${big(r.output)}</td>
      <td>${sec(r.ttft)}</td><td>${nf(r.decode_tps)}</td><td>${r.itl == null ? NA : nf(1000 * r.itl, 0) + " ms"}</td>
      <td>${sec(r.e2e)}</td><td>${r.running_at_arrival ?? NA} / ${r.waiting_at_arrival ?? NA}</td>
      <td class="l">${fin}</td><td class="l">${r.stream ? "yes" : "no"}</td></tr>`;
  }
  if (!d.recent.length) h += `<tr><td colspan="12" class="l sub">no finished requests yet</td></tr>`;
  $("recent").innerHTML = h;
}

// ---- data collection ----
function prod(d) {
  const p = d.gauges.prod || {on: false};
  if (!p.on) {
    $("prod").innerHTML = `<div class="row"><span><b>Recording: off</b></span><span class="sub">set TRNRT_PROD_DIR to keep every request's record on disk</span></div>`;
    return;
  }
  const files = Object.entries(p.files || {}).map(([n, b]) => `<span class="mono">${esc(n)} ${bytes(b)}</span>`).join("") || `<span class="sub">no files yet today</span>`;
  const dropped = Object.values(p.dropped || {}).reduce((a, b) => a + b, 0);
  $("prod").innerHTML = `<div class="row">
      <span><b class="ok">Recording: on</b></span>
      <span>directory <span class="mono">${esc(p.dir)}/${esc(p.day)}</span></span>
      <span>written ${big(p.written.requests)} requests · ${big(p.written.gauges)} gauges</span>
      <span class="${dropped ? "bad" : ""}">dropped ${dropped}</span>
      <span class="${p.write_errors ? "bad" : ""}">write errors ${p.write_errors}</span>
      <span>queued ${p.queued}</span></div>
    <div class="row" style="margin-top:6px">${files}</div>
    ${p.last_error ? `<div class="warn">last error: ${esc(p.last_error)}</div>` : ""}`;
}

function draw(d) {
  tiles(d); windows(d); recent(d); prod(d);
  for (const c of CHARTS) drawChart(c, d);
}

async function poll() {
  try {
    const r = await fetch(location.pathname.replace(/dashboard\/?$/, "") + "v1/stats" + location.search, {cache: "no-store"});
    if (!r.ok) throw new Error("HTTP " + r.status);
    last = await r.json();
    draw(last);
    $("dot").className = "dot live";
    $("status").textContent = "live · " + clock(last.now, true);
  } catch (e) {
    $("dot").className = "dot down";
    $("status").textContent = "no data: " + e.message;
  } finally {
    setTimeout(poll, POLL_MS);
  }
}
setupCharts();
window.addEventListener("resize", () => { if (last) for (const c of CHARTS) drawChart(c, last); });
poll();
</script>
</body>
</html>
"""
