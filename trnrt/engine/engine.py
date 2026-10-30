from __future__ import annotations

import atexit
import itertools
import os
import time
import weakref
from dataclasses import dataclass

import torch

from ..config import EngineConfig, ModelConfig
from ..models.loader import load_model, resolve_model_path
from . import watchdog
from .kv_pool import PagePool
from .model_runner import ModelRunner
from .radix_cache import RadixCache
from .request import Request, SamplingParams, Status, unsupported
from .sampler import unpack
from .scheduler import NeedSync, Scheduler, SchedulerConfig

# Tensor-parallel engines not closed yet, held weakly: close() at exit (stopping the workers) without
# keeping every engine a process ever made alive. atexit.register(self.close) did, so a test session
# kept each tp engine's model, runner and process groups (and their gloo listeners) until it ended.
_OPEN: weakref.WeakSet = weakref.WeakSet()


@atexit.register
def _close_open() -> None:
    for eng in list(_OPEN):
        eng.close()


@dataclass
class StepStats:
    """The work the last step() launched (decodes, prefill tokens, and its graph calls by kind) and that step() call's
    wall time. The call that launches a step reads the previous one, so its seconds belong to the previous launch."""
    num_decode: int = 0
    num_prefill_tokens: int = 0
    seconds: float = 0.0
    decode_calls: int = 0
    prefill_calls: int = 0
    num_restored: int = 0  # state checkpoints copied into requests (prefix hits of a recurrent model)
    prefill_chunks: int = 0  # prefill entries (one sequence's chunk each)


def resolve_attention_tp(mcfg: ModelConfig, cfg: EngineConfig) -> int:
    """The token mixers' TP degree (models/decoder.py attention_tp)."""
    from ..models.decoder import attention_tp

    return attention_tp(mcfg, cfg.tp)


def _cp_degree(mcfg: ModelConfig, attn_tp: int) -> int:
    """The context-parallel degree of the DSA caches (DecoderForCausalLM.cp): the attention TP when context-parallel
    DSA is on and every attention layer can run the long path, else 1."""
    from ..config import LinearSpec
    from ..models import dsa_long, mla

    specs = [s for s in (mcfg.attn_layers or ()) if not isinstance(s, LinearSpec)]
    ok = dsa_long.cp_enabled() and attn_tp > 1 and specs and all(mla.long_capable(s) for s in specs)
    return dsa_long.cp_degree(attn_tp) if ok else 1


def pool_pages(cfg: EngineConfig, mcfg: ModelConfig) -> int:
    """Pages in the KV pool. The pool is a graph input shape (every layer graph's KV cache), so a capture
    (trnrt/compile/capture.py) must size it exactly as the engine does."""
    from ..models.decoder import kv_heads_per_rank, state_bytes_per_token_rank

    from .model_runner import kv_cache_torch_dtype

    # KV for one page on ONE rank: each rank holds its attention rank's share of the KV heads
    # (or one replicated head when the attention TP exceeds the head count).
    attn_tp = resolve_attention_tp(mcfg, cfg)
    page_bytes = (mcfg.kv_bytes_per_token(kv_cache_torch_dtype(cfg)) * cfg.page_size
                  * kv_heads_per_rank(mcfg.num_kv_heads, attn_tp) // mcfg.num_kv_heads)
    page_bytes += state_bytes_per_token_rank(mcfg, attn_tp, cfg.dtype,
                                             kv_cache_torch_dtype(cfg) == torch.float8_e4m3fn) * cfg.page_size
    # Context-parallel DSA (models/dsa_long.py): a rank holds 1 / cp of every page.
    page_bytes //= _cp_degree(mcfg, attn_tp)
    # A model of linear-attention layers only has no KV: pages then only count positions.
    return cfg.num_pages or (int(cfg.kv_cache_gb * 2**30 // page_bytes) + 1 if page_bytes
                             else cfg.max_num_seqs * cfg.max_pages_per_seq + 1)


def build_shard(cfg: EngineConfig, path: str, num_pages: int, rank: int = 0):
    """Device, model shard and runner for one tensor-parallel rank (the whole model at tp=1)."""
    mcfg = ModelConfig.from_pretrained(path)
    if cfg.num_layers is not None:
        mcfg = mcfg.truncated(cfg.num_layers)
    if cfg.device == "neuron":
        # Runtime knobs vllm-neuron sets for serving (neuron_worker.py,
        # _init_neuron_distributed_environment_and_runtime). Piecewise execution launches several graphs per step:
        # take the runtime's maximum queue depth (libnrt clamps larger values to 63, with a warning).
        os.environ.setdefault("NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS", "63")
        os.environ.setdefault("NEURON_RT_IO_RING_CACHE_SIZE", "32")
        # NOT NEURON_RT_DISABLE_EXECUTION_BARRIER=1, the third knob vllm-neuron sets there (neuron_worker.py): without
        # the runtime's per-execution barrier a collective mismatch returns silently wrong numbers, and ranks
        # alternating world and attention-group collectives under host jitter deadlock.
        from .. import platform

        platform.configure_runtime_env()  # NEURON_LOGICAL_NC_CONFIG, before the runtime
        import libtorch_neuronx_lite  # noqa: F401  registers the "neuron" device

        if rank == 0:
            print("trnrt platform", platform.check_runtime(), flush=True)

        device = torch.device("neuron:0")  # each rank sees exactly one logical NeuronCore
    else:
        device = torch.device(cfg.device)
    atp = resolve_attention_tp(mcfg, cfg)
    group = attn_group = cp_group = None
    if cfg.tp > 1:
        import torch.distributed as dist

        from . import tp

        group = dist.group.WORLD
        attn_group = tp.attention_group(cfg.tp, atp)
        # Context-parallel row groups (models/dsa_long.py): the cp consecutive ranks of each row group, created after
        # the attention groups on every rank (dist.new_group is collective).
        cpd = _cp_degree(mcfg, atp)
        if 1 < cpd < atp:
            cp_group = tp.attention_group(cfg.tp, cpd)
    from .model_runner import fp8_e4m3_max

    # Weights the checkpoint stores as FP8 stay FP8 on the device (fp32 block scales, dequantized in-graph).
    keep_fp8 = mcfg.quant_block is not None or mcfg.quant_expert_block is not None
    model = load_model(path, mcfg, cfg.dtype, device, cfg.max_model_len, rank, cfg.tp, group,
                       keep_fp8=keep_fp8, fp8_max=fp8_e4m3_max(device), vocab_parallel=True,
                       moe_kernel=cfg.resolved_moe_kernel(), attn_tp=atp, attn_group=attn_group,
                       max_num_seqs=cfg.max_num_seqs)
    if cp_group is not None:
        model.cp_group = cp_group
    runner = ModelRunner(model, mcfg, cfg, num_pages, device, kv_heads=model.nkv, model_path=path)
    return model, runner


class LLMEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.t_start = time.perf_counter()  # startup_seconds counts from here to the end of warmup()
        self.startup_seconds: float | None = None
        path = resolve_model_path(cfg.model_path)
        self.model_path = path
        self.mcfg = ModelConfig.from_pretrained(path)
        if cfg.num_layers is not None:
            self.mcfg = self.mcfg.truncated(cfg.num_layers)
        self.attn_tp = resolve_attention_tp(self.mcfg, cfg)
        num_pages = pool_pages(cfg, self.mcfg)
        self._workers = []
        self._tp_saved = None  # rank 0's process state before init_rank (tp.process_state), restored by close
        if cfg.tp > 1:
            import multiprocessing as mp

            from . import tp

            port = tp.free_port(cfg.tp_core_base if cfg.device == "neuron" else None)
            ctx = mp.get_context("spawn")
            self._workers = [ctx.Process(target=tp.worker_main, args=(r, cfg, path, num_pages, port), daemon=True)
                             for r in range(1, cfg.tp)]
            self._tp_saved = tp.process_state()
            try:
                for w in self._workers:
                    w.start()
                if cfg.device == "neuron":
                    tp.neuron_env(0, port, cfg.tp_core_base)
                tp.init_rank(0, cfg.tp, port)
            except BaseException:
                self._stop_tp()
                raise
        t = time.perf_counter()
        try:
            self.model, self.runner = build_shard(cfg, path, num_pages, 0)
            self.load_seconds = time.perf_counter() - t
        except BaseException:
            self._stop_tp()
            raise
        # The tokenizer (and so transformers) loads only AFTER tensor-parallel workers are spawned: a process that
        # imported transformers and then spawned a child traces torch.topk differently from every other process, which
        # gave rank 0 its own compile cache keys and doubled every compile. And only after build_shard, so rank 0 imports
        # libtorch_neuronx_lite before transformers, the order every spawned rank and trnrt/compile/capture.py have.
        from transformers import AutoTokenizer

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
        except (OSError, ValueError, TypeError):
            self.tokenizer = None  # token-id-only use; stop strings need one
        self.device = self.runner.device
        if cfg.tp > 1:
            from . import tp

            self.runner.tp_send = tp.send
            _OPEN.add(self)
        self.pool = PagePool(num_pages)
        self.radix = RadixCache(self.pool, cfg.page_size)
        scfg = SchedulerConfig(
            page_size=cfg.page_size,
            max_num_seqs=cfg.max_num_seqs,
            max_prefill_tokens=cfg.max_prefill_tokens,
            max_model_len=cfg.max_model_len,
            eos_token_ids=self.mcfg.eos_token_ids,
            admission_decode_tokens=cfg.admission_decode_tokens,
        )
        # The scheduler (engine/scheduler.py) runs the served prefix-cache rules for a recurrent model: a hit resumes from
        # the deepest state checkpoint at or below it, checkpoints are written inside prefill chunks and at every page
        # boundary in decode, and a checkpoint a newer one dominates is freed when its path is released.
        if self.runner.state is None:
            raise ValueError("trnRT serves models with linear-attention layers: this model has no recurrent state")
        self.scheduler = Scheduler(scfg, self.pool, self.radix)
        self.scheduler.states = self.runner.state  # checkpoint rows of the state pool back the radix nodes
        self.radix.free_ckpt = self.runner.state.free_ckpt
        # Host memory under the radix cache (engine/host_cache.py), per rank: half for state checkpoints, the rest for
        # KV pages. Evicted pages and checkpoints go there (a checkpoint on a running request's path too, when no row is
        # free), and a later hit copies them back.
        self.host_tier = None
        if cfg.host_cache_gb > 0:
            from .host_cache import HostTier

            budget = cfg.host_cache_gb * 2**30
            ckpts = max(1, int(budget / 2 // self.runner.state.bytes_per_row()))
            budget -= ckpts * self.runner.state.bytes_per_row()
            page_bytes = self.runner.kv_page_bytes()
            pages = int(budget // page_bytes) if page_bytes else cfg.max_num_seqs * cfg.max_pages_per_seq
            self.host_tier = HostTier(self.runner, cfg.page_size, max(1, pages), ckpts)
            self.radix.offload, self.radix.offload_ckpt = self.host_tier.offload, self.host_tier.offload_ckpt
            self.scheduler.host = self.host_tier
        self._ids = itertools.count()
        self.last_step = StepStats()
        self._inflight = None  # (plan, launches) of a step whose results are not read yet

    def add_request(self, prompt_ids: list[int], params: SamplingParams, rid: str | None = None,
                    session_id: str | None = None) -> Request:
        why = unsupported(params)
        if why:  # before anything is built or queued: the request never reaches a step
            raise ValueError(f"{why} is not supported; send the request without it")
        req = Request(rid or f"req-{next(self._ids)}", list(prompt_ids), params, session_id=session_id)
        self.scheduler.add(req)
        return req

    def flush_cache(self) -> None:
        """Drop every cached prefix (device pages and the host cache)."""
        if self.host_tier is not None:
            self.host_tier.clear()
        offload, self.radix.offload = self.radix.offload, None
        self.radix.evict(self.radix.total_pages())
        self.radix.offload = offload

    def close_session(self, session_id: str) -> bool:
        """SGLang /close_session: drop the session's prefix references (nothing is freed)."""
        return self.scheduler.close_session(session_id)

    def abort(self, req: Request) -> None:
        self.scheduler.abort(req)
        self.runner.forget(req)

    def dead_ranks(self) -> list[int]:
        """The tensor-parallel ranks whose worker process has exited (rank 0 is this process)."""
        return [r for r, w in enumerate(self._workers, 1) if w.exitcode is not None]

    def close_failed(self) -> None:
        """Once a rank has failed: nothing more goes to the workers, and every worker still running is stopped now. A
        broadcast to a dead peer raises, and one to a live peer waits behind a step that will never finish. close()
        still runs at exit, on the thread that made the engine (torch's thread count it restores is that thread's)."""
        if getattr(self, "runner", None) is not None:
            self.runner.tp_send = None
        self._stop_workers(0.0)

    def close(self) -> None:
        """Stop tensor-parallel workers (rank 0 tells them to exit), destroy the process group and
        give this process back the state init_rank changed. Idempotent."""
        runner = getattr(self, "runner", None)
        _OPEN.discard(self)
        if runner is not None and getattr(runner, "watchdog", None) is not None:
            runner.watchdog.close()
        wait = 30.0
        if self._workers and runner is not None and runner.tp_send is not None:
            if self.dead_ranks():  # a broadcast to a dead peer raises or never returns: stop the rest instead
                wait = 0.0
            else:
                runner.tp_send(None)
            runner.tp_send = None
        self._stop_tp(wait)

    def _stop_tp(self, wait: float = 30.0) -> None:
        """Workers that did not exit within `wait` seconds (and every worker of an engine whose
        construction failed, which no rank 0 will ever release) are terminated: a daemon worker
        otherwise lives as long as the process, blocked in a rendezvous or a broadcast."""
        import torch.distributed as dist

        self._stop_workers(wait)
        if self._tp_saved is not None:
            if dist.is_initialized():
                dist.destroy_process_group()
            from . import tp

            tp.restore_process_state(self._tp_saved)
            self._tp_saved = None

    def _stop_workers(self, wait: float) -> None:
        """Each worker gets `wait` seconds to exit once rank 0 released it (none otherwise), then the rest are
        terminated together and killed if that does not end them."""
        released = getattr(self, "runner", None) is not None and self.runner.tp_send is None
        started = [w for w in self._workers if w.pid is not None]
        for w in started:
            w.join(timeout=wait if released else 0)
        late = [w for w in started if w.is_alive()]
        for w in late:  # all at once: 31 hung workers of a failed engine take 10 s, not 310
            w.terminate()
        for w in late:
            w.join(timeout=10)
            if w.is_alive():  # still holding its NeuronCores
                w.kill()
                w.join(timeout=10)
        self._workers = []

    def warmup(self, reverse: bool = False) -> dict:
        out = self.runner.warmup(reverse=reverse)
        self.startup_seconds = time.perf_counter() - self.t_start
        return out

    def has_work(self) -> bool:
        return self.scheduler.has_work() or self._inflight is not None

    def step(self) -> list[Request]:
        """Run one scheduler step. Returns requests that finished in it.

        Step N+1 is scheduled and launched before step N's tokens are read, so host scheduling hides behind device
        time; its inputs come from the on-device token board. A step that must preempt (the scheduler raises NeedSync)
        reads the step in flight back first and schedules again on the next call.
        """
        t = time.perf_counter()
        self.last_step = StepStats()
        self.scheduler.in_flight = self._inflight is not None
        try:
            plan = self.scheduler.schedule()
        except NeedSync:
            finished = self._drain()
        else:
            if not plan:
                finished = self._drain()
            else:
                launches = self._launch(plan)
                self.scheduler.advance(plan)
                finished = self._drain()  # read step N while the device runs step N+1
                self._inflight = (plan, launches)
                self.scheduler.in_flight = True
        self.last_step.seconds = time.perf_counter() - t
        return finished

    def _drain(self) -> list[Request]:
        if self._inflight is None:
            return []
        plan, launches = self._inflight
        self._inflight = None
        self.scheduler.in_flight = False
        tokens = self._collect(plan, launches)
        return self._finish(plan, self.scheduler.commit(plan, tokens))

    def _launch(self, plan) -> list:
        """Every call of the step, launched before the first read-back: (seqs, device output, kind)."""
        max_b = self.runner.decode_buckets[-1]
        launches = []
        self._restore_states(plan)
        for i in range(0, len(plan.decodes), max_b):
            chunk = plan.decodes[i : i + max_b]
            launches.append((chunk, self.runner.decode(chunk), "decode"))
            self._save_states(chunk)
        for s in plan.prefills:
            launches.append(([s], self.runner.prefill(s), "prefill"))
            self._save_states([s])
        self.last_step = StepStats(len(plan.decodes), sum(s.num_tokens for s in plan.prefills),
                                   decode_calls=sum(k == "decode" for _, _, k in launches),
                                   prefill_calls=len(plan.prefills),
                                   num_restored=sum(s.restore is not None for s in plan.prefills),
                                   prefill_chunks=len(plan.prefills))
        return launches

    def _restore_states(self, plan) -> None:
        """Prefix hits of recurrent models: copy each hit's checkpoint into the request's row before
        any of the step's calls."""
        if self.runner.state is None:
            return
        hits = [s for s in plan.prefills if s.restore is not None]
        self.runner.copy_state([s.restore for s in hits], [self.runner.state.row(s.req) for s in hits])
        for s in hits:
            s.req.ckpt_restore = None

    def _save_states(self, seqs) -> None:
        """Checkpoints of recurrent models: copy each entry's state row after it ran (scheduler
        ScheduledSeq.save), before any later call writes the row again."""
        saves = [s for s in seqs if s.save is not None]
        if saves:
            self.runner.copy_state([self.runner.state.row(s.req) for s in saves], [s.save for s in saves])

    def _collect(self, plan, launches) -> list:
        """Read a step's outputs; returns tokens aligned with plan.seqs()."""
        results: dict[int, int] = {}
        for chunk, out, kind in launches:
            with watchdog.watched(self.runner.watchdog, f"reading back a {kind} call"):
                dev_out, out = out, out.cpu()
            self.runner.launch_bound.done_through(dev_out)  # this call and every launch before it are done
            tokens, lps, top_ids, top_lps = unpack(out, len(chunk))
            for i, s in enumerate(chunk):
                results[id(s)] = tokens[i]
                n = s.req.params.logprobs
                if s.sample and n is not None and s.req.status is Status.RUNNING:
                    s.req.logprobs.append((lps[i], top_ids[i][:n], top_lps[i][:n]))
        return [results[id(s)] for s in plan.seqs()]

    def _finish(self, plan, finished: list[Request]) -> list[Request]:
        finished += self._check_stop_strings(plan.seqs())
        for r in finished:
            self.runner.forget(r)
        return finished

    def _check_stop_strings(self, seqs) -> list[Request]:
        done = []
        for s in seqs:
            req = s.req
            stops = req.params.stop
            if not stops or not s.sample or req.status is not Status.RUNNING:
                continue
            # Only the tail can newly contain a stop string; decode enough of it to cover
            # the longest one plus multi-token characters.
            tail = self.tokenizer.decode(req.output_ids[-(max(len(x) for x in stops) + 8):])
            if any(x in tail for x in stops):
                self.scheduler.finish_now(req, "stop")
                done.append(req)
        return done

    def generate(self, prompts: list[list[int]], params: SamplingParams | list[SamplingParams]) -> list[Request]:
        plist = params if isinstance(params, list) else [params] * len(prompts)
        reqs = [self.add_request(p, sp) for p, sp in zip(prompts, plist)]
        while self.has_work():
            self.step()
        return reqs
