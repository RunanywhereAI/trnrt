"""Build the served engine's compile cache on CPU hosts: capture every graph, compile it, strip it.

    python -m trnrt compile shapes zai-org/GLM-5.3-Flash --revision <commit>
    python -m trnrt compile decisions --model zai-org/GLM-5.3-Flash --revision <commit>
    python -m trnrt compile capture -- <the server's engine arguments>
    python -m trnrt compile compile
    python -m trnrt compile strip

Every step works in one directory, --out (default ~/.cache/trnrt; trnrt/compile/cache.py):

- shapes: config.json, the tokenizer's JSON files and every safetensors header of the checkpoint
  (a local directory, or a Hugging Face repo id read without downloading the weights) into
  <out>/shapes: a few MB that stand in for the weights, since a graph depends on their shapes and
  dtypes and never on their values.
- decisions: the one exception, the load-time decisions that read weight values
  (trnrt/compile/capture.py "Load-time decisions"), computed from the real checkpoint by HTTP range
  reads (a few tensors per MoE layer) into the shape directory. The capture refuses to guess them.
- capture: every graph the served engine runs, one process per tensor-parallel rank on the meta
  device (no NeuronCore needed): graph.hlo, the metadata and command.txt per cache entry under
  <out>/cache/neuron/compile_cache/<key>/, and the run's records under <out>/capture/. The
  arguments after `--` must be the engine arguments the server will run with (capture refuses any
  other, since the engine's parser would drop it unseen), and the serving host must load
  the checkpoint revision the shape directory was read at: the cache keys hash the shapes, the
  compiler arguments and the package versions.
- compile: neuronx-cc on every captured entry that has no NEFF yet, longest HLO first, each in LNL's
  layout with the completion marker written last, so the entry is a cache hit. A compile can take tens
  of GB of host memory, so they run as many at once as MemAvailable admits (--mem-per-compile-gb each,
  at most --workers), and one the OOM killer takes runs once more, alone.
  neuronx-cc is a plain CLI on an HLO file and runs on any host with the Neuron compiler installed,
  CPU-only instance types included. Run it on the host that captured: graphs that call NKI kernels
  name the kernel binaries the capture compiled by absolute path (cache.py NKI_KERNEL_DIR).
- strip: drop the compiler's debug info from every NEFF in place (trnrt/compile/strip.py). The
  runtime holds a loaded NEFF in host memory several times over, and the debug info is a large part
  of a NEFF's bytes, so unstripped NEFFs of two engines at 1M-token buckets do not fit in host RAM.

Then the server runs with NEURON_LIBTORCH_CACHE_ROOT=<out>/cache and finds every graph compiled.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field

from . import cache
from .cache import COMMAND, HLO, LOG, neff_name

GiB = 2**30
MEM_PER_COMPILE_GB = 30.0  # compile --mem-per-compile-gb

# -- one entry ----------------------------------------------------------------------------------

# Arguments LNL's neuroncc_compile always sets itself; everything else on the command line is the
# entry's compiler_args (what LNL's cache key hashed, after _apply_platform_compiler_args).
_FIXED = {"--framework", "--target", "--output", "--logfile"}


def parse_command(text: str) -> tuple[str, list[str]]:
    """(target, compiler_args) from an LNL command.txt."""
    argv = shlex.split(text)
    if len(argv) < 3 or argv[1] != "compile":
        raise ValueError(f"not a neuronx-cc compile command: {text[:120]!r}")
    target, rest, i = None, [], 3  # argv[2] is the HLO path
    while i < len(argv):
        a = argv[i]
        name = a.split("=", 1)[0]
        if name in _FIXED:
            value = a.split("=", 1)[1] if "=" in a else argv[i + 1]
            if name == "--target":
                target = value
            i += 1 if "=" in a else 2
            continue
        rest.append(a)
        i += 1
    if target is None:
        raise ValueError("command has no --target")
    return target, rest


def entry_args(entry: str) -> tuple[str, list[str]]:
    """The target and compiler_args to compile `entry` with, from the command.txt the capture wrote
    (trnrt/compile/capture.py _write_command)."""
    with open(os.path.join(entry, COMMAND)) as f:
        return parse_command(f.read())


# A base64 run long enough to be a backend_config (they are kilobytes); decoded and parsed lazily.
_B64 = re.compile(rb"[A-Za-z0-9+/]{64,}={0,2}")


def kernel_configs(hlo_path: str) -> list[dict]:
    """The decoded backend_config of every NKI custom call in an HLO (func_name, klir_binary, ...)."""
    with open(hlo_path, "rb") as f:
        data = f.read()
    out = []
    for m in _B64.finditer(data):
        blob = m.group(0)
        if not blob.startswith(b"eyJ"):  # base64 of '{"'
            continue
        try:
            cfg = json.loads(base64.b64decode(blob + b"=" * (-len(blob) % 4)))
        except (binascii.Error, ValueError):
            continue
        if isinstance(cfg, dict):
            out.append(cfg)
    return out


def kernel_refs(hlo_path: str) -> set[str]:
    """Absolute kernel-binary paths an HLO's NKI custom calls point at (klir_binary.binary)."""
    refs = set()
    for cfg in kernel_configs(hlo_path):
        klir = cfg.get("klir_binary")
        if isinstance(klir, dict) and isinstance(klir.get("binary"), str):
            refs.add(klir["binary"])
    return refs


def missing_kernels(hlo_path: str) -> list[str]:
    return sorted(p for p in kernel_refs(hlo_path) if not os.path.exists(p))


def compiler_command(neuronx_cc: str, hlo: str, target: str, neff: str, log: str, args: list[str]) -> list[str]:
    """The command LNL's neuroncc_compile runs (compile/backend.py)."""
    return [neuronx_cc, "compile", hlo, "--framework", "XLA", "--target", target, "--output", neff,
            "--logfile", log, *args]


@dataclass
class Result:
    key: str
    ok: bool
    seconds: float
    returncode: int
    cpu_seconds: float = 0.0  # user + system of neuronx-cc and every descendant it waited for
    max_rss_gb: float = 0.0  # largest single process (wait4 ru_maxrss)
    peak_tree_rss_gb: float = 0.0  # largest sum over the live process tree, sampled (needs psutil)
    neff_bytes: int = 0
    args: list[str] = field(default_factory=list)
    error: str = ""
    attempt: int = 1  # 2: the run again, alone, of a compile the OOM killer took


def _tree_rss(pid: int) -> int:
    """Resident bytes of a process and its descendants; 0 without psutil."""
    try:
        import psutil
    except ImportError:
        return 0
    try:
        p = psutil.Process(pid)
        procs = [p, *p.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for q in procs:
        try:
            total += q.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def run_compile(entry: str, *, work_root: str | None = None, neuronx_cc: str | None = None,
                sample_seconds: float = 1.0, rss: list[int] | None = None) -> Result:
    """Compile `entry`'s graph.hlo with its own compiler arguments into the entry: the NEFF, the log
    and command.txt, then the completion marker LAST, which makes it an LNL cache hit. rss: a
    one-element list kept at the compiler's process-tree resident bytes, as last sampled."""
    key = cache.key_of(entry)
    try:
        target, args = entry_args(entry)
    except (OSError, ValueError) as err:  # a capture killed after renaming graph.hlo into place, before command.txt
        return Result(key, False, 0.0, -1, error=f"{type(err).__name__}: {err}: an incomplete entry; remove {entry} "
                                                 "and run capture again")
    hlo = os.path.join(entry, HLO)
    neff, log = os.path.join(entry, neff_name(key)), os.path.join(entry, LOG)
    cc = neuronx_cc or shutil.which("neuronx-cc")
    if cc is None:
        raise RuntimeError("neuronx-cc is not on PATH (activate the Neuron SDK's environment, or pass --neuronx-cc)")
    cmd = compiler_command(cc, hlo, target, neff, log, args)
    gone = missing_kernels(hlo)
    if gone:
        return Result(key, False, 0.0, -1, args=args,
                      error=f"{len(gone)} NKI kernel binaries missing, e.g. {gone[0]}: compile on the host that "
                            "captured, or copy its NKI kernel directory here")
    tmp = os.path.join(entry, f"{COMMAND}.tmp-{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(shlex.join(cmd))
    os.replace(tmp, os.path.join(entry, COMMAND))  # a kill here must not leave the entry without its command
    # neuronx-cc leaves its scratch tree (neuronxcc-<random>) in the cwd; its stderr goes to a file there,
    # not a pipe, which a long error report could fill and block the compiler on.
    cwd = tempfile.mkdtemp(prefix=f"ncc-{key[:8]}-", dir=work_root)
    err_path = os.path.join(cwd, "stderr.txt")
    peak = [0]
    t0 = time.time()
    with open(err_path, "wb") as err_f:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=err_f)
    done = threading.Event()

    def sample():
        while not done.wait(sample_seconds):
            now = _tree_rss(proc.pid)
            peak[0] = max(peak[0], now)
            if rss is not None:
                rss[0] = now

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    _, status, ru = os.wait4(proc.pid, 0)
    seconds = time.time() - t0
    done.set()
    th.join()
    proc.returncode = os.waitstatus_to_exitcode(status)
    with open(err_path, "rb") as f:
        err = f.read().decode(errors="replace")[-2000:]
    ok = proc.returncode == 0 and os.path.exists(neff)
    res = Result(key, ok, seconds, proc.returncode, cpu_seconds=ru.ru_utime + ru.ru_stime,
                 max_rss_gb=ru.ru_maxrss / 2**20, peak_tree_rss_gb=peak[0] / 2**30, args=args,
                 error="" if ok else (err or f"exit {proc.returncode}, no NEFF written"))
    if res.ok:
        res.neff_bytes = os.path.getsize(neff)
        shutil.rmtree(cwd, ignore_errors=True)  # a failed compile's scratch tree stays, for its logs
        cache.write_marker(entry, res.neff_bytes)  # last: a reader that sees the marker sees a complete entry
    return res


def oom_killed(r: Result) -> bool:
    """The OOM killer took neuronx-cc itself (-9, or 137 through a shell) or one of its subprocesses,
    which neuronx-cc reports as exit 70 and "[F137] neuronx-cc was forcibly killed - This most
    commonly occurs due to insufficient system memory"."""
    return r.returncode in (-9, 137) or (r.returncode == 70 and "[F137]" in r.error)


def host_mem_available(path: str = "/proc/meminfo") -> int | None:
    """MemAvailable, bytes: the kernel's estimate of what new processes can take without swapping.
    None without /proc/meminfo (not Linux)."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB
    except (OSError, ValueError, IndexError):
        pass
    return None


def admits(avail: int | None, held: list[int], workers: int, reserve: float) -> bool:
    """Whether one more compile may start beside the running ones, whose process trees hold `held`
    bytes: fewer than `workers` run, and MemAvailable `avail` (None: unknown, `workers` alone decide),
    less what the running compiles have yet to take of their `reserve` bytes each, still holds
    `reserve`. MemAvailable already counts what a running compile holds, so it owes `reserve` less
    that. With nothing running one starts regardless: waiting would free nothing of ours."""
    if not held:
        return True
    if len(held) >= workers:
        return False
    if avail is None:
        return True
    return avail - sum(max(0.0, reserve - h) for h in held) >= reserve


def compile_entries(entries: list[str], workers: int, work_root: str | None = None,
                    neuronx_cc: str | None = None, on_result=None, mem_per_compile: float = MEM_PER_COMPILE_GB * GiB,
                    mem_available=host_mem_available, poll_seconds: float = 1.0) -> list[Result]:
    """run_compile over `entries`, the largest graph.hlo first (size is the best cheap proxy for
    compile time), so the tail of the batch is short graphs. A compile starts when `admits` allows
    it: at most `workers` at once, each reserving `mem_per_compile` bytes of mem_available(), which
    is read again every `poll_seconds` while entries wait. An entry that fails, by an exception too,
    fails alone: its Result says why and the others go on. A compile the OOM killer took (oom_killed)
    runs once more after all the others, alone. on_result sees every attempt; the list returned holds
    each entry's last."""
    import collections
    import queue

    todo = collections.deque(sorted(entries, key=lambda d: -os.path.getsize(os.path.join(d, HLO))))
    done: queue.Queue = queue.Queue()
    running: dict[str, list[int]] = {}  # entry -> its compiler's process-tree resident bytes, as last sampled
    last: dict[str, Result] = {}

    def attempt(d: str, n: int, rss: list[int]) -> None:
        try:
            r = run_compile(d, work_root=work_root, neuronx_cc=neuronx_cc, rss=rss)
        except Exception as err:  # that entry's failure: the others go on
            r = Result(cache.key_of(d), False, 0.0, -1, error=f"{type(err).__name__}: {err}")
        r.attempt = n
        done.put((d, r))

    def start(d: str, n: int) -> None:
        running[d] = [0]
        threading.Thread(target=attempt, args=(d, n, running[d])).start()

    def collect(timeout: float | None) -> tuple[str, Result] | None:
        try:
            d, r = done.get(timeout=timeout)
        except queue.Empty:
            return None
        del running[d]
        last[d] = r
        if on_result is not None:
            on_result(r)
        return d, r

    workers = max(1, workers)
    killed = []
    while todo or running:
        while todo and admits(mem_available(), [h[0] for h in running.values()], workers, mem_per_compile):
            start(todo.popleft(), 1)
        got = collect(poll_seconds if todo else None)  # memory frees as compiles end, and goes as they grow
        if got is not None and oom_killed(got[1]):
            killed.append(got[0])
    for d in killed:  # with every other compile ended
        start(d, 2)
        collect(None)
    return list(last.values())


# -- the command line ----------------------------------------------------------------------------

def _keys_filter(a) -> set[str] | None:
    keys = set()
    if a.keys:
        keys |= set(a.keys.split(","))
    if a.keys_file:
        with open(a.keys_file) as f:
            keys |= set(json.load(f))
    return keys or None


def cmd_shapes(a, dirs) -> int:
    from . import capture

    dest = a.shape_dir or dirs["shapes"]
    if os.path.isdir(a.model):
        if a.revision:
            raise SystemExit("--revision is for a Hugging Face repo id; a local checkpoint is what it holds")
        capture.write_shape_dir(a.model, dest)
        where = a.model
    else:
        from huggingface_hub import model_info

        rev = model_info(a.model, revision=a.revision).sha  # pinned: the commit this directory was read at
        capture.write_shape_dir_hub(a.model, dest, rev)
        where = f"{a.model}@{rev}"
    with open(os.path.join(dest, capture.HEADERS)) as f:
        n = len(json.load(f))
    print(f"{dest}: config + {n} tensor headers from {where}", flush=True)
    return 0


def cmd_decisions(a, dirs) -> int:
    from .. import platform
    from . import capture

    shape_dir = a.shape_dir or dirs["shapes"]
    fp8_max = platform.fp8_max(capture.DEFAULT_TARGET)
    ranks = None if a.ranks == "all" else [int(x) for x in a.ranks.split(",")]
    rec = capture.compute_decisions(a.model, shape_dir, a.tp, fp8_max, ranks=ranks,
                                    prefixes=a.layers.split(",") if a.layers else None, procs=a.procs,
                                    revision=a.revision)
    mine = {k: v for k, v in rec["decisions"].items() if k.startswith(f"fp8_max={fp8_max:g} tp={a.tp} ")}
    vals = sorted({(v["moe_prefill_dq"], v["moe_prefill_down"]) for v in mine.values()})
    print(f"{shape_dir}/{capture.DECISIONS}: {len(mine)} (layer, rank) decisions at fp8_max={fp8_max:g} "
          f"tp={a.tp} from {rec['model']}@{rec['revision']} in {rec['seconds']} s; (dq, down) values {vals}; "
          f"experts read per decision: max {max((v['experts_read'] for v in mine.values()), default=0)}",
          flush=True)
    return 0


def check_engine_args(argv: list[str]) -> None:
    """Refuse capture arguments the engine does not take. engine_config reads its arguments with
    parse_known_args, since the server hands it its own as well, so it drops one it does not know:
    a mistyped bucket ladder would capture the default ladder's graphs. Here every argument must be
    an engine argument (trnrt/__main__.py add_engine_args), spelled out in full."""
    from ..__main__ import add_engine_args

    ap = argparse.ArgumentParser(prog="python -m trnrt compile capture --", allow_abbrev=False, add_help=False)
    add_engine_args(ap)
    _, unknown = ap.parse_known_args(argv)
    if unknown:
        raise SystemExit(f"capture: not engine arguments: {shlex.join(unknown)}. After `--` come the engine arguments "
                         "the server runs with, spelled out in full (the engine group of `python -m trnrt serve "
                         "--help`); the server's own, such as --engines and --port, do not change the graphs: leave "
                         "them out")


def cmd_capture(a, dirs, engine_config) -> int:
    from . import capture

    if engine_config is None:
        raise SystemExit("capture needs the serve command line's engine_config(argv) -> EngineConfig "
                         "(trnrt/__main__.py passes it to main)")
    check_engine_args(list(a.serve))
    shape_dir = a.shape_dir or dirs["shapes"]
    if not os.path.exists(os.path.join(shape_dir, capture.HEADERS)):
        raise SystemExit(f"{shape_dir}: no {capture.HEADERS}; write the shape directory first (`shapes`)")
    # The target before the configuration: anything in it that depends on the platform resolves for
    # the target the graphs are captured for, not for this host.
    capture.configure_env(a.target)
    cfg = engine_config(list(a.serve))
    ranks = None if a.ranks == "all" else [int(r) for r in a.ranks.split(",")]
    s = capture.capture_all(cfg, shape_dir, dirs["out"], a.target, ranks=ranks, procs=a.procs, argv=list(a.serve))
    for rec in s.pop("per_rank"):
        print(json.dumps(rec), flush=True)
    print(json.dumps({"summary": True, **s}), flush=True)
    return 1 if s["failed_ranks"] else 0


def cmd_compile(a, dirs) -> int:
    root = cache.local_root(dirs["cache_root"])
    if not cache.entries(root):
        raise SystemExit(f"{root}: no captured entries; run capture first, with this --out")
    todo = cache.pending(root)
    want = _keys_filter(a)
    if want is not None:
        todo = [d for d in todo if cache.key_of(d) in want]
    workers, reserve = a.workers or os.cpu_count() or 1, a.mem_per_compile_gb * GiB
    avail = host_mem_available()
    room = ("MemAvailable unknown: --workers alone limits them" if avail is None else
            f"MemAvailable {avail / GiB:.0f} GB, room for {int(avail // reserve)} now")
    print(f"{len(todo)} entries to compile under {root}: at most {workers} at once, each reserving "
          f"{a.mem_per_compile_gb:g} GB of host memory ({room})", flush=True)
    t0 = time.time()

    def show(r: Result) -> None:
        row = {**asdict(r), "oom_killed": oom_killed(r)}
        print(json.dumps(row), flush=True)
        if a.log:
            with open(a.log, "a") as f:
                f.write(json.dumps(row) + "\n")

    results = compile_entries(todo, workers, a.work_root, a.neuronx_cc, on_result=show, mem_per_compile=reserve,
                              mem_available=host_mem_available)
    bad = [r for r in results if not r.ok]
    summary = {"summary": True, "entries": len(results), "failed": len(bad),
               "retried": sum(r.attempt > 1 for r in results), "workers": workers,
               "mem_per_compile_gb": a.mem_per_compile_gb, "wall_seconds": round(time.time() - t0, 1)}
    if any(oom_killed(r) for r in bad):
        summary["hint"] = ("the OOM killer took compiles that ran alone: this host has too little memory for them "
                           "(see each one's peak_tree_rss_gb)")
    print(json.dumps(summary), flush=True)
    return 1 if bad else 0


def cmd_strip(a, dirs) -> int:
    from . import strip

    root = cache.local_root(dirs["cache_root"])
    if not any(os.path.exists(os.path.join(d, cache.MARKER)) for d in cache.entries(root)):
        raise SystemExit(f"{root}: no compiled entries; run compile first, with this --out")
    s = strip.strip_cache(root, keys=_keys_filter(a), workers=a.workers,
                          on_entry=lambda rep: print(json.dumps(rep), flush=True))
    print(json.dumps({"summary": True, "root": root, **s}), flush=True)
    return 1 if s["refused"] else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m trnrt compile", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", default=cache.DEFAULT_OUT,
                       help="the pipeline's work directory: shapes/, capture/ and the cache root cache/ "
                            "(default %(default)s)")

    sh = sub.add_parser("shapes", help="config and safetensors headers of a checkpoint")
    common(sh)
    sh.add_argument("model", help="a Hugging Face repo id, or a local checkpoint directory")
    sh.add_argument("--revision", default=None, help="the repo's commit (default: its default branch, resolved "
                                                     "and printed)")
    sh.add_argument("--shape-dir", default=None, help="default <out>/shapes")

    dc = sub.add_parser("decisions", help="the load-time decisions that read weight values")
    common(dc)
    dc.add_argument("--model", required=True, help="a Hugging Face repo id (range reads) or a local checkpoint dir")
    dc.add_argument("--revision", default=None, help="the repo's commit (default: its current one, recorded)")
    dc.add_argument("--shape-dir", default=None, help="default <out>/shapes")
    dc.add_argument("--tp", type=int, default=32, help="tensor-parallel ranks of the engine (default %(default)s)")
    dc.add_argument("--ranks", default="all", help="all, or a comma list")
    dc.add_argument("--layers", default=None, help="comma list of layer prefixes (default: every MoE layer)")
    dc.add_argument("--procs", type=int, default=16, help="worker processes (default %(default)s)")

    cp = sub.add_parser("capture", help="every graph the engine runs, on the meta device")
    common(cp)
    cp.add_argument("--shape-dir", default=None, help="default <out>/shapes")
    cp.add_argument("--ranks", default="all", help="all, or a comma list")
    cp.add_argument("--procs", type=int, default=None,
                    help="rank processes at once after rank 0 (default: tp, at most half the CPUs)")
    cp.add_argument("--target", default="trn2", help="the Neuron target to compile for (default %(default)s)")
    cp.add_argument("serve", nargs=argparse.REMAINDER, help="-- then the engine arguments the server runs with")

    c = sub.add_parser("compile", help="neuronx-cc on every captured entry without a NEFF")
    common(c)
    c.add_argument("--workers", type=int, default=None,
                   help="most compiles at once (default: one per CPU); host memory admits fewer: "
                        "--mem-per-compile-gb")
    c.add_argument("--mem-per-compile-gb", type=float, default=MEM_PER_COMPILE_GB,
                   help="host memory each compile reserves, GB (default %(default)g): neuronx-cc can take tens of "
                        "GB on one of these graphs. A compile starts only while MemAvailable, less what the running "
                        "ones have yet to take of theirs, holds it; one the OOM killer takes runs once more, alone")
    c.add_argument("--keys", default=None, help="only these keys (comma list)")
    c.add_argument("--keys-file", default=None, help="only these keys (a capture's keys.json)")
    c.add_argument("--work-root", default=None, help="where neuronx-cc's scratch trees go (default: the temp dir)")
    c.add_argument("--neuronx-cc", default=None, help="the compiler (default: neuronx-cc on PATH)")
    c.add_argument("--log", default=None, help="also append each result as a JSON line to this file")

    st = sub.add_parser("strip", help="drop the NEFFs' debug info, in place")
    common(st)
    st.add_argument("--keys", default=None, help="only these keys (comma list)")
    st.add_argument("--keys-file", default=None, help="only these keys (a capture's keys.json)")
    st.add_argument("--workers", type=int, default=4, help="NEFFs at once (default %(default)s)")
    return ap


def main(argv: list[str] | None = None, engine_config=None) -> int:
    """The `compile` command line. engine_config(serve_argv) -> EngineConfig builds the serving
    engine's configuration from the serve arguments after `--` (capture needs it; the EngineConfig is
    pickled to the rank processes). Returns the exit code."""
    a = build_parser().parse_args(argv)
    if getattr(a, "serve", None) and a.serve[:1] == ["--"]:
        a.serve = a.serve[1:]
    dirs = cache.out_dirs(a.out)
    if a.cmd == "capture":
        return cmd_capture(a, dirs, engine_config)
    return {"shapes": cmd_shapes, "decisions": cmd_decisions, "compile": cmd_compile, "strip": cmd_strip}[a.cmd](a, dirs)
