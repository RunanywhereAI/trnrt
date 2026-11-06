"""Pads scheduled work into static-shape buckets and runs the model graphs.

Every call is padded to a bucket: decode to (B sequences, P pages), prefill to
(C tokens, P pages). The page axis is sized to the longest context in the call, not to
max_model_len, so a step reads only the KV it can use. Padded rows write their KV into
the null page and their outputs are discarded.

A decode call is one graph (prep, every layer and the sampler). A prefill call is a prep graph,
one graph per run of up to piecewise_prefill_moe_group layers (shared by every run with the same
kinds of layer), and a post graph that samples.
"""

from __future__ import annotations

import inspect
import os
import time
import weakref
import zlib
from collections import defaultdict, deque

import numpy as np
import torch

from ..config import EngineConfig, ModelConfig, _pow2_ladder, pick_bucket, prefill_pages
from . import iopool, start_lock
from .kv_pool import NULL_PAGE
from .request import Request
from .scheduler import PENDING, ScheduledSeq

# Stands for this rank's device-resident token board in host_args. A string, because the
# args are pickled to the other tensor-parallel ranks and an object() would not survive.
BOARD = "__trnrt_board__"
# neuronx-cc's model type for every graph, after the platform arguments.
MODEL_TYPE_ARGS = ("--model-type=transformer",)
# Speculative decoding (engine/spec_async.py). SPEC stands for the speculative board; DEV prefixes the name of a tensor an
# earlier graph of THIS rank returned that a later graph of the same step reads (the small speculative graphs' outputs, the
# last verify's and MTP pass's outputs): fixed names, overwritten every step, the same on every rank. HIDDEN prefixes the
# name of a call's final hidden states, kept on the device for the MTP pass (the last HIDDEN_KEEP: a step drafts only from
# its own calls', and each is HBM). PARAM prefixes the name of one of the model's parameters (a norm weight).
SPEC = "__trnrt_spec__"
DEV = "__trnrt_dev__:"
HIDDEN = "__trnrt_hidden__:"
HIDDEN_KEEP = 8
PARAM = "__trnrt_param__:"


def scratch_keys(ecfg: EngineConfig, page_size: int) -> int:
    """The longest context one call reads (DSA's selection scratch, bind_kv_cache's max_keys): the larger of the decode
    and prefill ladders' largest page buckets x the page size."""
    return max(max(ecfg.resolved_page_buckets()), max(ecfg.resolved_prefill_page_buckets())) * page_size


def kv_cache_torch_dtype(ecfg: EngineConfig) -> torch.dtype:
    if ecfg.kv_cache_dtype == "auto":
        return ecfg.dtype
    if ecfg.kv_cache_dtype in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    raise ValueError(f"kv_cache_dtype must be auto or fp8, not {ecfg.kv_cache_dtype!r}")


def fp8_e4m3_max(device: torch.device) -> float:
    """Largest finite FP8 E4M3 value on the device (trnrt/platform.py fp8_max: 240 on trn2). On CPU
    torch.float8_e4m3fn is the fn variant (448)."""
    # meta: the capture of a device run (trnrt/compile/capture.py), which must trace the device's value.
    if device.type not in ("neuron", "meta"):
        return float(torch.finfo(torch.float8_e4m3fn).max)
    from .. import platform

    return platform.fp8_max(platform.runtime_target())


def neuronx_cc_args(kv_dtype: torch.dtype, fp8_weights: bool = False) -> list[str]:
    """Extra neuronx-cc arguments for every graph.

    trn2 FP8 is e4m3 with inf (max 240) and torch has no such dtype, so an fp8 KV cache or FP8 weights are
    float8_e4m3fn in the graph, which neuronx-cc rejects there: "[NCC_EVRF051] Data type F8E4M3FN is not supported
    on TRN1/TRN2 ... use the --experimental-unsafe-fp8e4m3fn-as-fp8e4m3 flag" (SDK 2.32). The values are clamped to
    240 on write, so the cast is exact. trnrt/platform.py adds --logical-nc-config and vllm-neuron's argument set."""
    from .. import platform

    args = platform.neuronx_cc_args(platform.runtime_target(), fp8=kv_dtype == torch.float8_e4m3fn or fp8_weights)
    return args + list(MODEL_TYPE_ARGS)


# neuronx-cc refuses a graph past 5M instructions (NCC_EBVF030). The whole-call verify graph of a 45-layer model, Q = 3
# positions x 16 rows, is in the range where the limit has to be raised; only that graph takes the argument, so no decode or
# prefill graph's compile key moves.
VERIFY_BIG_LIMIT = "--internal-max-instruction-limit=30000000"


def verify_cc_args(ecfg: EngineConfig) -> list[str]:
    """Extra neuronx-cc arguments for the whole-call verify graph (speculative decoding) only."""
    return [VERIFY_BIG_LIMIT] if ecfg.spec_method else []


def canonicalize(gm) -> None:
    """Drop keyword arguments equal to their defaults from every torch call in `gm`, so
    equivalent traces print identically (canonical_neuron_backend)."""
    from torch.fx import Node
    from torch.fx.operator_schemas import get_signature_for_torch_op

    def strip_defaults(node) -> None:
        if not node.kwargs:
            return
        sigs = get_signature_for_torch_op(node.target) or []
        if not sigs:
            return
        kept = {}
        for k, v in node.kwargs.items():
            # Defaults of the overloads that declare one for k (an overload may take k
            # without a default, e.g. a required out=); drop v only if all of them agree.
            declared = [sg.parameters[k] for sg in sigs if k in sg.parameters]
            ds = [p.default for p in declared if p.default is not inspect.Parameter.empty]
            is_default = not isinstance(v, Node) and (
                (bool(ds) and all(type(d) is type(v) and d == v for d in ds))
                # e.g. topk(..., out=None): no schema declares out (its out variant takes
                # values= / indices=), and None means "no output buffer"
                or (not declared and v is None))
            if not is_default:
                kept[k] = v
        node.kwargs = kept

    for node in gm.graph.nodes:
        if node.op == "call_function" and not isinstance(node.target, str):
            try:
                strip_defaults(node)
            except Exception:  # an op without an inspectable schema: leave as traced
                pass
    gm.recompile()


def canonical_neuron_backend():
    """libtorch_neuronx_lite's backend behind a pass that drops keyword arguments equal to
    their defaults from every torch call, so equivalent traces print identically.

    The compile cache key hashes str(graph) (libtorch_neuronx_lite compile/cache.py,
    create_cache_hash). Tensor-parallel rank 0 and the other ranks traced the same model to
    graphs that differed only in spelling: rank 0 recorded torch.topk(x, 64, dim=-1), rank 1
    torch.topk(x, 64, dim=-1, largest=True, sorted=True, out=None): a process that imported
    transformers and then spawned a child records the short form, every other process the long
    one. So every rank compiled its own copy, and a 32-rank model would run 32 neuronx-cc at once.
    Stripping defaults maps both forms to the short one: one key, one compile, the other
    ranks wait on the cache lock and load it. trnrt/compile/capture.py applies the same pass, so a
    graph captured on a CPU host gets the key the device run computes.
    """
    import torch._dynamo as dynamo

    inner = dynamo.lookup_backend("neuron_libtorch")

    def backend(gm, example_inputs, **kwargs):
        canonicalize(gm)
        return iopool.wrap(inner(gm, example_inputs, **kwargs))  # engine/iopool.py: its executor's proxy

    backend.__name__ = "trnrt_canonical_neuron_libtorch"
    return backend


def piecewise_groups(n_layers: int, group: int | None, alone=()) -> list[range]:
    """Consecutive runs of up to `group` layers; every layer in `alone` is a run of its own.

    With the XLA MoE path (CPU tests) the MoE layers get a graph each: neuronx-cc (2.27, SDK 2.32) lowers the
    in-graph expert-weight gathers into millions of tiny software DMAs once two attention + MoE layers share a graph.
    With the NKI MoE kernel the kernel's expert loads are one DMA per expert whatever the graph holds, so MoE layers
    group like any other (_piecewise's moe_group)."""
    g = group or -(-n_layers // 14)
    alone = set(alone)
    runs: list[range] = []
    i = 0
    while i < n_layers:
        j = i + 1
        if i not in alone:
            while j < min(i + g, n_layers) and j not in alone:
                j += 1
        runs.append(range(i, j))
        i = j
    return runs


# Layer-graph launches allowed in flight before the host waits: the runtime's execution queue fails with "Execution
# Queue Full" when the device is the bottleneck and too many heavy launches are queued; 8 was measured safe.
IN_FLIGHT = 6
# NEFF executions this rank may have outstanding, across ALL calls, before the next launch waits (LaunchBound). IN_FLIGHT
# above bounds only one call's layer pieces, and a step can make several prefill calls of several executions each, plus
# the previous step's (overlap launches step N + 1 before reading step N). 12 is two decode steps' launches, so a
# decode-only load never waits, and below 16, where back-to-back whole-model decode launches failed.
MAX_IN_FLIGHT = 12


class LaunchBound:
    """One counter over every NEFF execution a rank launches (ModelRunner wraps every compiled callable), which waits,
    before a launch would leave more than `bound` outstanding, on the oldest outstanding one: the queue executes in
    order, so reading an output back means everything launched before it is done (libtorch_neuronx_lite has no
    synchronize). Host-side only: the traced graphs and their keys are unchanged.

    Outputs are held weakly (a held prefill hidden state is tens of MB of HBM): when the one to wait on was freed, a
    later one is waited on instead (more than needed, never less). The last output is held strongly, so there is
    always one. Reads elsewhere (the engine reading a step back, done_through) count as waits."""

    def __init__(self, bound: int = MAX_IN_FLIGHT):
        self.bound = max(1, int(bound))
        self.n = 0  # executions launched
        self.done = 0  # executions known finished: every one with an index below
        self.waits = 0  # waits this bound made
        self._handles: deque = deque()  # (index, weak reference to its output), oldest first
        self._last = None
        self._aliases: dict = {}  # id(view) -> (weak reference to the view, its execution's index): alias()

    def counted(self, fn):
        """fn (a compiled callable) with every call counted and bounded."""

        def call(*args, **kw):
            self.before()
            out = fn(*args, **kw)
            self.after(out)
            return out

        return call

    def before(self) -> None:
        need = self.n + 1 - self.bound  # executions that must be done for this launch to keep the bound
        if need <= self.done:
            return
        self.wait_for(need - 1)

    def wait_for(self, i: int) -> None:
        """Wait until execution i is done (also engine/iopool.py, before it rewrites a slot's inputs)."""
        for j, ref in self._handles:  # the oldest output still held from execution i on (the newest always is)
            t = self._last if j == self.n - 1 else ref()
            if j >= i and t is not None:
                t.cpu()  # waits for execution j, and so for everything launched before it
                self.waits += 1
                self._mark(j)
                return

    def after(self, out) -> None:
        outs = out if isinstance(out, (tuple, list)) else (out,)
        t = next((o for o in outs if getattr(getattr(o, "device", None), "type", None) == "neuron"), None)
        if t is None:
            return  # host (CPU tests) and meta (the capture) outputs: no execution to bound
        self._handles.append((self.n, weakref.ref(t)))
        self._last = t
        self.n += 1

    def alias(self, view, out) -> None:
        """view, a tensor the caller made from out (decode's out[:n]), stands for out's execution when it is read back.
        Explicit, so it does not depend on how the device's tensors expose views."""
        for i, ref in self._handles:
            if ref() is out:
                self._aliases[id(view)] = (weakref.ref(view), i)
                return

    def done_to(self, count: int) -> None:
        """A worker (rank > 0): the first `count` executions are done, by rank 0's count sent with each call. Workers
        never read an output back, so without this their `done` moved only by the bound's own waits, and every
        bound-th launch waited on the step in flight; rank 0's call then waited in its launch for that rank.
        Why this is safe: a worker executes exactly the calls rank 0 sends, in order, so its counted executions pair
        one for one with rank 0's; and rank 0 reading back execution N means every rank passed N's collectives, so a
        worker's own execution N is done but for its tail after the last collective. Its count can therefore run
        ahead of the truth by at most that tail, which still bounds its queue at the bound plus about one. Never past
        its own launches."""
        c = min(int(count), self.n)
        if c > self.done:
            self._mark(c - 1)

    def done_through(self, t) -> None:
        """t, an output of one of these launches, an alias() of one or a view of one, was read back: it and everything
        before it are done. The engine reads decode's out[:n] back, a new tensor over the call's output: matched by
        identity alone, done would never move on the decode path and every bound-th execution would wait on the step
        still in flight, breaking the overlap. decode alias()es its view; a view's _base chain is matched too."""
        e = self._aliases.pop(id(t), None)
        if e is not None and e[0]() is t:
            self._mark(e[1])
            return
        cands = []
        while t is not None and len(cands) < 8:
            cands.append(t)
            t = getattr(t, "_base", None)
        for i, ref in self._handles:
            r = ref()
            if r is not None and any(r is c for c in cands):
                self._mark(i)
                return

    def _mark(self, i: int) -> None:
        self.done = max(self.done, i + 1)
        while self._handles and self._handles[0][0] < self.done:
            self._handles.popleft()
        if self._aliases:
            self._aliases = {k: e for k, e in self._aliases.items() if e[1] >= self.done and e[0]() is not None}


def _piecewise(model, compile_, moe_group: int, prefill_moe_group: int, compile_verify=None):
    """forward_decode compiled whole, forward_prefill as prep graph -> one graph per run of layers (shared by every
    run with the same kinds) -> post graph, with forward_prefill's signature, and, for a model that speculates,
    forward_extend (the verify call) compiled whole too, by compile_verify. Tables are chosen on the host exactly as
    DecoderForCausalLM._attn_inputs does; the compiled prep returns only the biases.

    Each graph execution holding a cross-chip collective has a fixed cost at 32 ranks, so a decode call is one graph.
    A prefill step is compute-bound, so that cost hardly matters there, while a big chunk times many MoE layers in one
    graph does not compile: neuronx-cc refuses a graph past 5M instructions (NCC_EBVF030).

    moe_group / prefill_moe_group: with the NKI MoE kernel (DecoderLayer.moe_blob), runs of up to that many layers of
    any kind; without it every MoE layer is a run of its own (piecewise_groups)."""
    kinds: dict[tuple, object] = {}
    blob = any(getattr(l, "moe_blob", False) for l in model.layers)

    def make_plan(mg):
        if blob:
            runs = piecewise_groups(len(model.layers), mg)
        else:
            runs = piecewise_groups(len(model.layers), None, [i for i, l in enumerate(model.layers) if l.moe])
        out = []
        for idxs in runs:
            key = tuple(model.layer_kind(i) for i in idxs)
            if key not in kinds:
                kinds[key] = compile_(model.group_fn(idxs))
            out.append((kinds[key], tuple(t for i in idxs for t in model.layer_tensors(i))))
        return out

    plan_p = make_plan(prefill_moe_group or moe_group)
    windows = [None] + ([model.window] if model.window is not None else [])
    seen: set = set()
    last: list = [None]

    def run(fn, *args, **kw):
        """Launch one piece. The FIRST launch of a piece at a given shape waits for everything
        before it and is waited for itself: when every graph came from the compile cache, the
        ranks ran ahead of each other and the first post graph failed on every rank with
        "replica group signature mismatch for group 0: likely caused by mismatched collectives
        between peers"; a compile in the same process lines the ranks up behind its lock."""
        key = (id(fn), tuple(tuple(x.shape) if isinstance(x, torch.Tensor) else x is None for x in args[:7]))
        first = key not in seen
        if first and last[0] is not None and last[0].device.type == "neuron":
            last[0].cpu()
        out = fn(*args, **kw)
        t = out[0] if isinstance(out, tuple) else out
        if first:
            if t.device.type == "neuron":
                t.cpu()
            seen.add(key)
        last[0] = t
        return out

    def flat(prep):
        def f(*a):
            h, attn = prep(*a)
            return (h, *(attn[w][0] for w in windows))
        return f

    def layers(h, positions, slot_mapping, biases, block_table, swa_table, state_slot=None):
        table_w = swa_table if swa_table is not None else block_table
        bias_w = biases.get(model.window) if model.window is not None else None
        kw = {"state_slot": state_slot} if state_slot is not None else {}  # linear-attention layers
        outs = []
        for fn, tensors in plan_p:
            h = run(fn, h, positions, slot_mapping, block_table, biases[None], table_w, bias_w, *tensors, **kw)
            outs.append(h)
            if len(outs) > IN_FLIGHT and h.device.type == "neuron":
                # libtorch_neuronx_lite has no synchronize; reading an earlier output back
                # waits for it, which bounds the launches queued behind it.
                outs[-1 - IN_FLIGHT].cpu()
        return h

    prep_p, post_p = compile_(flat(model.prep_prefill)), compile_(model.post_prefill)
    sig = inspect.signature(model.forward_prefill)
    whole_d = compile_(model.forward_decode)

    def with_h(out, h):
        """An MTP head drafts from the final hidden states (forward_mtp_k): a model with one hands them on beside the
        call's output, kept on the device (ModelRunner._exec_inner)."""
        return out if getattr(model, "mtp", None) is None else (out, h)

    def decode(*args):
        return run(whole_d, *args)

    def prefill(*args):
        a = sig.bind(*args)
        a.apply_defaults()
        a = a.arguments
        h, *b = run(prep_p, a["input_ids"], a["positions"], a["block_table"], a["swa_table"], a["swa_first"],
                    a["ngram_ids"])
        h = layers(h, a["positions"], a["slot_mapping"], dict(zip(windows, b)), a["block_table"], a["swa_table"],
                   a["state_slot"])
        return with_h(run(post_p, h, a["last_index"], a["temperature"], a["top_p"], a["top_k"], a["min_p"], a["noise"],
                          a["board"], a["write_slot"], a["bitmask"], a["penalties"], a["plp_targets"]), h)

    extend = None
    if getattr(model, "mtp", None) is not None:  # the verify call as one graph (prep, every layer, post)
        whole_e = (compile_verify or compile_)(model.forward_extend)

        def extend(*args):  # noqa: F811
            return run(whole_e, *args)

    return decode, prefill, extend


def _copy_rows(src, dst, *pools):
    """pool[dst] = pool[src] for every recurrent-state pool: a state checkpoint saved or restored
    (engine/state_pool.py). Padding pairs copy the scratch row onto itself (copy_buckets)."""
    for p in pools:
        p.index_put_((dst,), p[src])
    return dst.sum()


def alloc_kv(shapes, slots: int, dtype, device):
    """(k_caches, v_caches): one zeroed [slots, *shape] K and V cache per KV layer, every layer's K then every layer's
    V (the allocation order places them in device memory)."""
    ks = [torch.zeros((slots, *k), dtype=dtype, device=device) for k, _ in shapes]
    vs = [torch.zeros((slots, *v), dtype=dtype, device=device) for _, v in shapes]
    return ks, vs


def state_checkpoint_rows(ecfg: EngineConfig) -> int:
    """Rows of a recurrent model's state pool kept for prefix-cache checkpoints: the configured
    count, else twice max_num_seqs (SGLang v0.5.21 int8_mamba_ckpt_size's default: 2x the active
    mamba pool)."""
    return ecfg.state_checkpoints if ecfg.state_checkpoints is not None else 2 * ecfg.max_num_seqs


class ModelRunner:
    def __init__(self, model, mcfg: ModelConfig, ecfg: EngineConfig, num_pages: int, device: torch.device,
                 kv_heads: int | None = None, model_path: str | None = None):
        self.model = model
        self.tp_send = None  # rank 0 under tensor parallelism: broadcasts every graph call
        # A device hang (a rank that stopped taking part in the calls leaves the others blocked in their
        # collectives, with no error) ends the rank loudly, naming its last calls (engine/watchdog.py).
        from . import watchdog

        self.watchdog = watchdog.make(getattr(model, "tp_rank", 0), device.type) if ecfg.tp > 1 else None
        self.mcfg = mcfg
        self.ecfg = ecfg
        self.device = device
        self.ps = ecfg.page_size
        # Context-parallel DSA (models/dsa_long.py): each rank holds ps / cp slots of every page.
        self.lps = self.ps // getattr(model, "cp", 1)
        slots = num_pages * self.lps
        kv_dtype = kv_cache_torch_dtype(ecfg)
        shapes = model.kv_shapes()  # per layer, this rank's (nkv, Dk), (nkv, Dv)
        self.k_caches, self.v_caches = alloc_kv(shapes, slots, kv_dtype, device)
        # max_rows: the most query rows one graph call carries (a prefill chunk, or a decode batch).
        model.bind_kv_cache(self.k_caches, self.v_caches, self.ps,
                            fp8_max=fp8_e4m3_max(device) if kv_dtype == torch.float8_e4m3fn else None,
                            max_rows=max(max(ecfg.resolved_prefill_token_buckets()),
                                         max(ecfg.resolved_decode_batch_buckets()) * (1 + ecfg.spec_k if ecfg.spec_method else 1)),
                            max_keys=scratch_keys(ecfg, self.ps))
        # Token-slot state beside K and V, in the model dtype, addressed by the same slots, so the prefix cache and the
        # host cache treat it as KV: under an fp8 KV cache every DSA layer keeps its pool keys here (layer.pool_key,
        # mla.pool_cache_mode "separate"), which context-parallel DSA requires. One dict per KV layer, allocated and
        # bound in layer order.
        states = [{n: torch.zeros((slots, *shp), dtype=ecfg.dtype, device=device) for n, shp in d.items()}
                  for d in model.token_state_shapes()]
        if any(states):
            model.bind_token_states(states)
        self.s_caches = [t for d in states for t in d.values()]
        # Per-token caches beside K and V (paged like them: the host cache copies them with the page).
        self.aux_caches = [torch.zeros((slots, *a), dtype=kv_dtype, device=device) for a in model.aux_kv_shapes()]
        model.bind_aux_kv(self.aux_caches)
        self.decode_buckets = ecfg.resolved_decode_batch_buckets()
        # The two step families of dynamic speculation each have their own buckets (family_buckets): plain decode, and the
        # verify family. Both are the decode buckets when speculation is off or not split.
        self.plain_buckets, self.spec_buckets = family_buckets(ecfg, self.decode_buckets)
        self.prefill_buckets = ecfg.resolved_prefill_token_buckets()
        # Sequence-parallel prefill streams (models/decoder.py prefill_sp_enabled) split every chunk's rows evenly over
        # the tp ranks: off when a bucket does not divide.
        if getattr(model, "prefill_sp", False):
            odd = [b for b in self.prefill_buckets if b % model.tp_size]
            if odd:
                model.prefill_sp = False
                print(f"trnrt: sequence-parallel prefill streams off: prefill buckets {odd} do not divide over "
                      f"tp={model.tp_size}", flush=True)
        self.page_buckets = ecfg.resolved_page_buckets()
        self.spec_page_buckets = spec_page_buckets(ecfg, self.page_buckets)
        self.spec_page_set = self.spec_page_buckets != tuple(sorted(self.page_buckets))
        self.prefill_page_buckets = ecfg.resolved_prefill_page_buckets()
        self.prefill_page_min_rows = ecfg.prefill_page_min_rows
        # A prefill chunk's MTP pass fills the draft layer's KV over the chunk: in one call of the chunk's bucket, or, with
        # mtp_fill_rows, in calls of at most that many rows over the same hidden states (spec_after_prefill). The pass's
        # scratchpad reservation grows with its rows (the compiler's figure at 4096 pages: 1.0 GiB at 512 rows, 1.5 at
        # 1024, 2.5 at 2048, 5.5 at 4096) and the scratchpad is the largest over the loaded graphs.
        fill = ecfg.mtp_fill_rows
        self.after_prefill_rows = fill if (fill and ecfg.spec_method) else None
        fills = [c for c in self.prefill_buckets if c <= fill] + [fill] if fill else list(self.prefill_buckets)
        # spec_after_prefill picks its row count from these (the decode-side pass has its own graphs): a chunk of any
        # length takes its prefill call's bucket, or the fill size.
        self.mtp_q_buckets = tuple(sorted(set(fills))) if ecfg.spec_method else ()
        self.K = ecfg.sampling_candidates
        self._rngs: dict[str, np.random.Generator] = {}
        self._synth = None  # EngineConfig.spec_synth_al: the synthetic accept flags of each verify step (engine/spec_synth.py)
        if ecfg.spec_synth_al is not None:
            from .spec_synth import SynthAccept

            self._synth = SynthAccept(ecfg.spec_synth_al, ecfg.spec_k, ecfg.seed)
        # Token board for overlap scheduling: one slot per live request plus a scratch slot
        # that padded rows write to.
        n_slots = 2 * ecfg.max_num_seqs + 8
        self.board = torch.zeros(n_slots, dtype=torch.float32, device=device)
        self.scratch_slot = n_slots - 1
        self._free_slots = list(range(n_slots - 2, -1, -1))
        self._slot: dict[str, int] = {}
        self.compile_seconds: dict[tuple, float] = {}
        self.exec_counts: dict[tuple, int] = {}  # graph executions by key (vllm-neuron's neff_execution_count)
        self.calls: dict[tuple, int] = defaultdict(int)
        # meta: trnrt/compile/capture.py traces the device run's graphs on a host without NeuronCores, through
        # a backend that writes each graph's HLO into the compile cache instead of compiling it.
        if device.type in ("neuron", "meta"):
            import torch._dynamo as dynamo

            # One static graph per bucket; dynamo must not give up on recompiles.
            for knob in ("cache_size_limit", "recompile_limit", "accumulated_cache_size_limit",
                         "accumulated_recompile_limit"):
                if hasattr(dynamo.config, knob):
                    setattr(dynamo.config, knob, 1 << 20)
            os.environ.setdefault("NEURON_LIBTORCH_COMPILATION_TIMEOUT", "7200")
            fp8_weights = any(t.dtype == torch.float8_e4m3fn for t in model.parameters())
            if device.type == "neuron":
                be = canonical_neuron_backend()
            else:
                from ..compile import capture

                be = capture.backend()
            opts = dict(backend=be, fullgraph=True, dynamic=False,
                        options={"compiler_args": neuronx_cc_args(kv_dtype, fp8_weights)})
            compile_ = lambda f: torch.compile(f, **opts)  # noqa: E731
            compile_verify = None
            vargs = verify_cc_args(ecfg)
            if vargs:  # the whole-call verify graph's own compile function: no decode or prefill key moves
                opts_v = dict(opts, options={"compiler_args": neuronx_cc_args(kv_dtype, fp8_weights) + vargs})
                compile_verify = lambda f: torch.compile(f, **opts_v)  # noqa: E731
        else:
            compile_ = lambda f: f  # noqa: E731
            compile_verify = None
        # Every compiled callable counts its launches against one bound (LaunchBound); outside torch.compile, so no
        # traced graph changes. Host and meta outputs are not counted.
        bound = self.launch_bound = LaunchBound()
        # engine/iopool.py: a decode call's host arguments and outputs at fixed device addresses, so the runtime's IO
        # ring cache holds a few sets per NEFF instead of filling. Host-side only; on a NeuronCore only.
        # A speculative engine pools the blind step's calls too, and warms every slot of the pool (iopool.pool_names).
        self.iopool = (iopool.Pool(names=iopool.pool_names(bool(ecfg.spec_method), ecfg.mtp_kv_only),
                                   rank=getattr(model, "tp_rank", 0), warm=bool(ecfg.spec_method))
                       if device.type == "neuron" else None)

        def counted(c):
            return None if c is None else (lambda f: bound.counted(c(f)))

        compile_, compile_verify = counted(compile_), counted(compile_verify)
        # Linear-attention layers: their recurrent state per running request (engine/state_pool.py),
        # passed to decode / prefill as a trailing state_slot argument.
        self.state = None
        if model.state_layers():
            from .state_pool import StatePool

            # A verify writes the state after each of its 1 + k positions into the request's own rows (state_pool.py),
            # so a seat that speculates holds 1 + k rows; the state pool may be sized apart from max_num_seqs.
            per_req = 1 + ecfg.spec_k if ecfg.spec_method else 1
            self.state = StatePool(model, ecfg.state_seats or ecfg.max_num_seqs, device, ecfg.dtype,
                                   state_checkpoint_rows(ecfg), per_req, ecfg.state_spec_seats)
            if self.state.rows != state_pool_rows(ecfg):
                raise RuntimeError(f"state pool has {self.state.rows} rows, the configuration says "
                                   f"{state_pool_rows(ecfg)}: the pool's shape (every state graph's key) moved")
            self._state_arg = {n: list(inspect.signature(getattr(model, f)).parameters).index("state_slot")
                               for n, f in (("decode", "forward_decode"), ("prefill", "forward_prefill"))}
            if ecfg.spec_method:
                self._state_arg["verify"] = list(inspect.signature(model.forward_extend).parameters).index("state_slot")
            self._copy = compile_(_copy_rows)
            # At least 4 pairs: a 2-row copy of a 18.6 MB row measured 4.5 ms against 1.2 ms for 4 rows and 2.1 ms for
            # 8 (SDK 2.32, bf16), the same trap as the one-row write in linear_attn._write_rows.
            self.copy_buckets = _pow2_ladder(4, max(4, ecfg.max_num_seqs))
        self._decode, self._prefill, self._extend = _piecewise(model, compile_, ecfg.piecewise_moe_group,
                                                               ecfg.piecewise_prefill_moe_group, compile_verify)
        self._hidden: dict[str, torch.Tensor] = {}  # a call's final hidden states, kept on the device for the MTP pass
        self._hidden_c: dict[str, int] = {}  # a prefill call's hidden stash -> that call's row bucket C
        self._hidden_sp: set[str] = set()  # hidden states holding only this rank's rows of a prefill chunk
        self._prefill_c: int | None = None  # the C of the latest prefill call
        self._hcount = 0
        self.last_hidden: str | None = None
        self._devs: dict[str, torch.Tensor] = {}
        self.spec_k = ecfg.spec_k if ecfg.spec_method == "mtp" else 1  # MTP passes per draft graph
        has_mtp = getattr(model, "mtp", None) is not None
        if ecfg.spec_method and not has_mtp:
            raise ValueError("speculative decoding (spec_method mtp) needs a model with an MTP layer")
        self._mtp = compile_(model.forward_mtp_k) if ecfg.spec_method else None
        # The KV-only window graph (DecoderForCausalLM.forward_mtp_kv): an after-prefill MTP window that holds no row of the
        # last position writes only the draft layer's KV.
        self._mtp_kv = (compile_(model.forward_mtp_kv) if (ecfg.mtp_kv_only and self._mtp is not None) else None)
        # Speculative steps under overlap scheduling (engine/spec_async.py): the per-slot board and the small graphs around
        # the verify and MTP graphs.
        self.spec_async = self._mtp is not None
        self._spec_fns: dict = {}
        self.async_ckpt_t = 0
        if self.spec_async:
            from . import spec_async as _sa

            Q, k, ps = 1 + ecfg.spec_k, ecfg.spec_k, self.ps
            self.spec_board = torch.zeros(n_slots, _sa.width(Q, k), dtype=torch.float32, device=device)
            self._spec_fns = {
                "spec_prep": compile_(lambda b, si, t, r, pad, v: _sa.spec_prep(b, si, t, r, pad, v, Q, k, ps, 1, None)),
                "spec_post": (compile_(lambda b, si, o, ids, ra, v, us: _sa.spec_post(b, si, o, ids, ra, v, Q, k, us))
                              if self._synth is not None
                              else compile_(lambda b, si, o, ids, ra, v: _sa.spec_post(b, si, o, ids, ra, v, Q, k))),
                "mtp_post": compile_(lambda b, si, d, v: _sa.mtp_post(b, si, d, v, Q, k)),
                "spec_init": compile_(lambda b, tb, si, base, cur, v: _sa.spec_init(b, tb, si, base, cur, v, Q, k)),
                "spec_host_init": compile_(lambda b, si, tok, base, cur, v: _sa.spec_host_init(b, si, tok, base, cur, v,
                                                                                               Q, k)),
                "mtp_prefill_ids": compile_(_sa.mtp_prefill_ids),
            }

            def mtp_board(b, si, t, pad, v, hsrc, hidx, src_norm):
                # mtp_prep, the MTP passes and mtp_post as one graph (the decode side of the blind step): three launches
                # become one, and the board is updated in place at its end as mtp_post's is
                T, last, pos, slot, *r = _sa.mtp_prep(b, si, t, pad, v, Q, ps, 1, None, k)
                draft, xl = model.forward_mtp_k(T, pos, t, slot, hsrc, hidx, last, src_norm, *(r if r else (None, None)))
                _sa.mtp_post(b, si, draft, v, Q, k)
                return draft, xl

            self._spec_fns["mtp_board"] = compile_(mtp_board)
            # Decode checkpoints (the served prefix cache leaves one at every page boundary a request crosses): a blind
            # step's next verify has rewritten the state rows by the time the host reads the accepted count, so spec_ckpt
            # copies the state on the device right after spec_post.
            if self.state is not None:
                self.async_ckpt_t = ecfg.page_size
                self._spec_fns["spec_ckpt"] = compile_(
                    lambda b, si, ra, ck, v: _sa.spec_ckpt(b, si, ra, ck, v, Q, self.async_ckpt_t, self.state.pools(), 1, None))
        self._host_kv: dict[int, list] = {}  # host cache slot -> per-layer (K, V) page copies
        self._host_ckpt: dict[int, list] = {}  # host checkpoint slot -> every state pool's row
        self._last_out = None  # the output of the last graph call launched on this rank (_settle)

    # -- sampling inputs ----------------------------------------------------------

    def _sampling(self, reqs: list[Request | None]):
        n = len(reqs)
        temperature = np.zeros(n, np.float32)
        top_p = np.ones(n, np.float32)
        top_k = np.zeros(n, np.int64)
        min_p = np.zeros(n, np.float32)
        noise = np.full((n, max(self.K, 1)), 0.5, np.float32)
        for i, r in enumerate(reqs):
            if r is None or r.params.temperature <= 0:
                continue
            p = r.params
            temperature[i], top_p[i], top_k[i], min_p[i] = p.temperature, p.top_p, p.top_k, p.min_p
            rng = self._rngs.get(r.rid)
            if rng is None:
                rng = self._rngs[r.rid] = np.random.default_rng(
                    p.seed if p.seed is not None else (self.ecfg.seed, zlib.crc32(r.rid.encode())))
            noise[i] = rng.random(noise.shape[1], dtype=np.float32)
        return temperature, top_p, top_k, min_p, noise

    def pad_slots(self, shape) -> np.ndarray:
        """Cache slots for padded rows: slots 1 .. page_size - 1 of the null page in turn, never slot 0.

        A padded row sits at position 0 behind an all-null block table, so the one key it can see is the null page's
        slot 0, and nothing may write that slot while it is read: with several padded rows writing their KV to slot 0
        and reading it back in the same graph, the value read changed from execution to execution with identical
        inputs. Writes to slots 1 .. page_size - 1 are never read unmasked (every padded row's other keys and its pools
        are invisible), and those slots still count as padding where a graph tells real rows from padded ones by
        slot_mapping >= page_size (models/linear_attn.py). Slot 0 is then written by nothing after start-up."""
        n = int(np.prod(shape))
        if self.ps < 2:
            return np.full(shape, NULL_PAGE * self.ps, np.int64)
        return (NULL_PAGE * self.ps + 1 + np.arange(n, dtype=np.int64) % (self.ps - 1)).reshape(shape)

    def slot(self, req: Request) -> int:
        s = self._slot.get(req.rid)
        if s is None:
            s = self._slot[req.rid] = self._free_slots.pop()
        return s

    def synth_active(self) -> bool:
        """Whether the speculative steps take the host's synthetic accept flags (EngineConfig.spec_synth_al)."""
        return self._synth is not None

    def forget(self, req: Request) -> None:
        self._rngs.pop(req.rid, None)
        if self._synth is not None:
            self._synth.forget(req.rid)
        if self.state is not None:
            self.state.release(req)
        s = self._slot.pop(req.rid, None)
        if s is not None:
            self._free_slots.append(s)

    def _prefill_state(self, row: int, C: int) -> list[int]:
        """A prefill's state slot for a row (the warmup's scratch row): [row, the scratch row, C], the shape of an
        in-chunk checkpoint's slot (linear_attn's chunk form), so warmup traces the graphs serving runs."""
        return [row, 0, C]

    def _with_state(self, name: str, args: list, rows) -> list:
        """args plus the state_slot argument (padding the optional ones before it with None)."""
        if self.state is None:
            return args
        i = self._state_arg[name]
        if not isinstance(rows, str):  # a str: a device tensor an earlier graph made (DEV)
            rows = np.asarray(rows, np.int64)
        return args + [None] * (i - len(args)) + [rows]

    def _dev(self, a, slot=None):
        """slot: (key, pool slot, argument position...) of a pooled call (engine/iopool.py): a host array goes to that
        slot's fixed device buffer instead of a new tensor."""
        if isinstance(a, str) and a == BOARD:
            return self.board
        if isinstance(a, str) and a == SPEC:
            return self.spec_board
        if isinstance(a, str) and a.startswith(DEV):
            return self._devs[a]
        if isinstance(a, str) and a.startswith(HIDDEN):
            return self._hidden[a]
        if isinstance(a, str) and a.startswith(PARAM):
            return getattr(self.model, a[len(PARAM):])
        if a is None:
            return None
        if isinstance(a, tuple):
            return tuple(self._dev(x, None if slot is None else slot + (i,)) for i, x in enumerate(a))
        if slot is not None:
            return self.iopool.put(slot, torch.from_numpy(a), self.device)
        return torch.from_numpy(a).to(self.device)

    def _exec(self, name: str, key: tuple, host_args: list):
        """Run graph `name` on host-side arguments (numpy arrays, BOARD, None, or tuples of
        arrays). Under tensor parallelism rank 0 first sends the same call to every other
        rank, so all ranks execute identical graphs on identical inputs.

        A graph's first call loads its NEFFs and builds its collectives on every rank, so rank 0 holds the host-wide
        start lock (engine/start_lock.py) from before it sends the call until the call has run."""
        if key in self.compile_seconds or self.tp_send is None:
            return self._exec_inner(name, key, host_args)
        with start_lock.held(f"first {name} {key}"):
            return self._exec_inner(name, key, host_args)

    def _exec_inner(self, name: str, key: tuple, host_args: list):
        if self.tp_send is not None:  # with rank 0's count of finished executions (LaunchBound.done_to on the worker)
            self.tp_send((name, key, host_args, self.launch_bound.done))
        attr = {"decode": "_decode", "prefill": "_prefill", "verify": "_extend", "mtp": "_mtp", "mtp_kv": "_mtp_kv"}.get(name)
        fn = getattr(self, attr) if attr else self._spec_fns[name]
        if name == "prefill" and len(key) > 1 and isinstance(key[1], int):
            self._prefill_c = key[1]  # ("prefill", C, P): the call's row bucket
        pool = self.iopool  # engine/iopool.py: j, this call's slot, or None
        j = pool.begin(name, key, self.launch_bound) if pool is not None else None
        try:
            args = [self._dev(a, None if j is None else (key, j, i)) for i, a in enumerate(host_args)]
            if pool is not None:
                pool.note_args(args)
            first = key not in self.compile_seconds
            self.exec_counts[key] = self.exec_counts.get(key, 0) + 1
            t = time.perf_counter()
            wd = self.watchdog
            if wd is not None:
                wd.launched(name, key)
            if wd is None or first:  # a first call may compile or load its NEFFs: not watched
                out = fn(*args)
            else:
                with wd.blocking(f"launching {name} {key}"):  # blocks once the runtime's execution queue is full
                    out = fn(*args)
        finally:
            if pool is not None:
                pool.end(key, j, self.launch_bound)
        if name in self._spec_fns:  # the small speculative graphs: every output stays on the device by name
            outs = out if isinstance(out, tuple) else (out,)
            for i, o in enumerate(outs):
                self._devs[f"{DEV}{name}:{i}"] = o
            out = outs[0]
        elif isinstance(out, tuple) and name == "mtp":  # (drafts, MTP hidden): nothing reads the MTP hidden back
            out = out[0]
        elif isinstance(out, tuple):  # (output, hidden states kept on the device)
            out, h = out
            self.last_hidden = f"{HIDDEN}{self._hcount}"
            self._hcount += 1
            self._hidden[self.last_hidden] = h
            if name == "prefill" and self.model._sp_on():  # this rank's rows only (sequence-parallel streams)
                self._hidden_sp.add(self.last_hidden)
            if name == "prefill" and self._prefill_c is not None:
                self._hidden_c[self.last_hidden] = self._prefill_c
            if len(self._hidden) > HIDDEN_KEEP:
                old = next(iter(self._hidden))
                del self._hidden[old]
                self._hidden_sp.discard(old)
                self._hidden_c.pop(old, None)
        if name in ("verify", "mtp") and self.spec_async:
            self._devs[DEV + name] = out  # read by spec_post / mtp_post
        if first:
            if self.device.type == "neuron":
                out = out.cpu()  # count the compile, not just the launch
            self.compile_seconds[key] = time.perf_counter() - t
        self.calls[key] += 1
        self._last_out = out
        return out

    # -- warmup -------------------------------------------------------------------

    def warmup(self, reverse: bool = False) -> dict:
        """Compile (or load from the compile cache) every bucket graph before serving, so no
        request ever waits on neuronx-cc. Dummy calls write only to the null page.

        reverse: prefill buckets first and every ladder backwards. Two engines that warm the same graphs in opposite
        orders compile two of them at once (one graph compiles at a time per process; a second process needing the
        same graph waits on its lock).

        Speculative decoding: the verify family at its own buckets and pages (family_buckets, spec_page_buckets), each
        verify followed by the graphs of its blind step, and every prefill call by the MTP pass over its rows. With
        warm pools (iopool.Pool.warm) each such block runs once per pool slot, in the order a step runs it, so a graph's
        call count, and with it its slot, advances in step with its neighbours' as in serving."""
        t0 = time.perf_counter()
        spec_q = 1 + self.ecfg.spec_k if self.ecfg.spec_method else None

        def order(xs):
            return tuple(reversed(xs)) if reverse else tuple(xs)

        # Plain decode graphs: all of them without speculation, only with dynamic speculation under it (a static
        # speculative engine verifies every row, a draftless one in the verify graph, and never launches a plain decode).
        plain = not spec_q or bool(self.ecfg.spec_max_batch)

        def warm_decode():
            for B in order(self.decode_buckets):
                for P in order(tuple(sorted(set(self.page_buckets) | set(self.spec_page_buckets)))):  # each family at its own P
                    ids = np.zeros(B, np.int64)
                    table = np.full((B, P), NULL_PAGE, np.int64)
                    args = [np.zeros(B, np.int64), ids, table, np.ones(B, np.int64), self.pad_slots(B),
                            *self._sampling([None] * B), BOARD, np.full(B, -1, np.int64),
                            np.full(B, self.scratch_slot, np.int64)]
                    args = self._with_state("decode", args, [0] * B)  # the scratch row
                    for _ in range(self._pool_warm_reps("decode") if plain and B in self.plain_buckets and P in self.page_buckets else 0):
                        self._exec("decode", ("decode", B, P), args)
                    if spec_q and B in self.spec_buckets and P in self.spec_page_buckets:
                        z = np.zeros((B, spec_q), np.int64)
                        samp = self._sampling([None] * (B * spec_q))
                        args = [z, z, table, self.pad_slots((B, spec_q)), *samp,
                                np.full(B * spec_q, 0.5, np.float32), z.reshape(-1)]
                        args = self._with_state("verify", args, np.zeros((B, 1 + spec_q), np.int64))
                        for _ in range(self._pool_warm_reps("verify")):
                            self._exec("verify", ("verify", B, spec_q, P), args)
                            self._warm_mtp_board(B, spec_q, P)  # the decode-side pass: prep, the passes and post in one graph
                            self._warm_spec(B, P)

        def warm_prefill():
            for C in order(self.prefill_buckets):
                for P in order(prefill_pages(self.prefill_page_buckets, self.prefill_page_min_rows, C)):
                    args = [np.zeros(C, np.int64), np.zeros(C, np.int64), np.full(P, NULL_PAGE, np.int64),
                            self.pad_slots(C), np.zeros(1, np.int64), *self._sampling([None]), BOARD,
                            np.full(1, self.scratch_slot, np.int64)]
                    args = self._with_state("prefill", args, self._prefill_state(0, C))
                    self._exec("prefill", ("prefill", C, P), args)
                    if self._mtp is not None:
                        self._warm_after_prefill(C, P)
            for n in self.copy_buckets if self.state is not None and self.state.num_ckpt_rows else ():
                z = np.zeros(n, np.int64)
                self._copy_state(z, z)  # the scratch row onto itself

        for warm in ((warm_prefill, warm_decode) if reverse else (warm_decode, warm_prefill)):
            warm()
        if self.device.type == "neuron":
            torch.zeros(1, device=self.device).cpu()
        return {"graphs": len(self.compile_seconds), "seconds": time.perf_counter() - t0}

    def _pool_warm_reps(self, name: str) -> int:
        """How many times warmup runs a block whose calls the pool takes under `name`: the pool's K when the pool warms
        every slot, so each slot's buffers and the runtime's DMA ring set for each are built before the engine serves (a
        pooled serve otherwise builds slots 1 .. K - 1 on its first calls of each shape, and their rings after it said
        it was ready); 1 otherwise. Host only: no graph or key changes."""
        p = self.iopool
        return p.k if p is not None and p.warm and p.k > 0 and name in p.names else 1

    def _warm_after_prefill(self, C: int, P: int) -> None:
        """The MTP pass over a prefill call's rows, and the board graphs around it, at a prefill bucket C and page rung P,
        over the hidden states the call left (so the graph for that stash is the one serving runs). The windows of
        mtp_fill_rows: q = min(C, F) is the size of the split pass."""
        prefilled = self.last_hidden
        F = self.after_prefill_rows
        qs = [q for q in self.mtp_q_buckets for _ in range(self._pool_warm_reps("mtp"))]
        for q in qs:
            # a chunk of any length up to C fills the MTP KV from the rows of its call's bucket: q = C, or q = F when the
            # call is larger than F (spec_after_prefill picks q from the same ladder as the call's C)
            if q <= C and q == (C if not F or C <= F else F):
                self.last_hidden = prefilled
                Pm = self.spec_page(P)
                self._warm_mtp(1, q, Pm)
                if self._mtp_kv_on(q, Pm):  # the window graph beside the full one (the window with the final row is still the full graph)
                    self._warm_mtp_kv(q, Pm)
                sl = np.full(1, self.scratch_slot, np.int64)
                self._exec("mtp_prefill_ids", ("mtp_prefill_ids", 1, q),
                           [BOARD, sl, np.zeros((1, q), np.int64), np.full(1, -1, np.int64)])
                self._exec("mtp_post", ("mtp_post", 1, 1 + self.spec_k), [SPEC, sl, DEV + "mtp", np.zeros(1, np.float32)])
        sl, z = np.full(1, self.scratch_slot, np.int64), np.zeros(1, np.float32)
        for _ in range(self._pool_warm_reps("spec_init")):
            self._exec("spec_init", ("spec_init", 1), [SPEC, BOARD, sl, z, z, z])

    def spec_page(self, P: int) -> int:
        """The page rung a launch of the verify family or the MTP layer (or its warmup) uses for rung P: the smallest rung of
        the spec ladder that is >= P, refused if there is none (a speculative step must never pick a rung outside it, e.g.
        the 3072-page rung of the shipped configurations). With no separate spec ladder, P."""
        if not self.spec_page_set:
            return P
        q = pick_bucket(P, self.spec_page_buckets)
        if q not in self.spec_page_buckets:
            raise RuntimeError(f"a speculative launch at page bucket {q} is outside the spec page buckets {self.spec_page_buckets}")
        return q

    def _warm_spec(self, B: int, P: int) -> None:
        """The small speculative graphs (engine/spec_async.py) of a (B, P) verify, on scratch slots only (every row
        invalid: the board keeps its values), right after that verify and its MTP pass were warmed."""
        Q = 1 + self.spec_k
        sl = np.full(B, self.scratch_slot, np.int64)
        v = np.zeros(B, np.float32)
        table = np.full((B, P), NULL_PAGE, np.int64)
        rows = np.zeros((B, Q), np.int64)
        pad = self.pad_slots((B, Q))
        self._exec("spec_host_init", ("spec_host_init", B), [SPEC, sl, v, v, v, v])
        self._exec("spec_prep", ("spec_prep", B, Q, P), [SPEC, sl, table, rows, pad, v])
        post_args = [SPEC, sl, DEV + "verify", DEV + "spec_prep:0", rows.astype(np.float32), v]
        flags0 = [np.zeros((B, self.spec_k), np.float32)]
        self._exec("spec_post", ("spec_post", B, Q), post_args + (flags0 if self._synth is not None else []))
        if self.async_ckpt_t:  # no row reserved: the scratch row onto itself
            self._exec("spec_ckpt", ("spec_ckpt", B, Q), [SPEC, sl, rows.astype(np.float32), np.zeros(B, np.float32), v])

    def _mtp_board_key(self, B: int, Q: int, P: int, hkey: str) -> tuple:
        return ("mtp_board", B, Q, P, self._hidden[hkey].shape[0], self.spec_k)

    def _warm_mtp_board(self, B: int, q: int, P: int) -> None:
        """The fused decode-side MTP graph at (B, q, P) over the hidden states the previous warmup call left, on scratch slots
        only (every row invalid: the board keeps its values)."""
        sl = np.full(B, self.scratch_slot, np.int64)
        self._exec("mtp_board", self._mtp_board_key(B, q, P, self.last_hidden),
                   [SPEC, sl, np.full((B, P), NULL_PAGE, np.int64), self.pad_slots((B, q)), np.zeros(B, np.float32),
                    self.last_hidden, np.zeros((B, q), np.int64), PARAM + "norm"])

    def _warm_mtp(self, B: int, q: int, P: int) -> None:
        """One dummy MTP pass over the hidden states the previous warmup call left, so the graph for that (rows, B, q, P) is
        compiled; writes only to the null page."""
        z = np.zeros((B, q), np.int64)
        r = self.spec_k - 1
        rest = np.zeros((B, r), np.int64) if r else None
        args = [z, z, np.full((B, P), NULL_PAGE, np.int64), self.pad_slots((B, q)), self.last_hidden, z,
                np.zeros(B, np.int64), PARAM + "norm", rest, self.pad_slots((B, r)) if r else None]
        self._exec("mtp", self._mtp_key(B, q, P, self.last_hidden), self._with_sp_src(args, self.last_hidden))

    def _mtp_key(self, B: int, Q: int, P: int, hkey: str) -> tuple:
        sp = ("sp",) if hkey in self._hidden_sp else ()
        return ("mtp", B, Q, P, self._hidden[hkey].shape[0], self.spec_k) + sp

    def _mtp_kv_key(self, Q: int, P: int, hkey: str) -> tuple:
        """The KV-only window graph's key: one sequence, Q rows, P pages, the stash's rows; independent of spec_k (no later
        passes)."""
        sp = ("sp",) if hkey in self._hidden_sp else ()
        return ("mtp_kv", 1, Q, P, self._hidden[hkey].shape[0]) + sp

    def _mtp_kv_on(self, Q: int, P: int) -> bool:
        """Whether a window of Q rows over P pages with no final row runs the KV-only graph."""
        return self._mtp_kv is not None and self.model.mtp_kv_takes(Q, P)

    def _mtp_kv_args(self, ids, pos, table, slot, hkey: str, hidx) -> list:
        """forward_mtp_kv's arguments: ids [1, Q] straight from the host (no sampled token in a KV-only window), the
        sp_onehot buffer when the stash is a sequence-parallel chunk's."""
        args = [ids, pos, table, slot, hkey, hidx, PARAM + "norm"]
        return args + [PARAM + "sp_onehot"] if hkey in self._hidden_sp else args

    def _warm_mtp_kv(self, q: int, P: int) -> None:
        """One dummy KV-only window at (q, P) over the hidden states the previous warmup call left: the graph for (1, q, P)
        is compiled; writes only to the null page."""
        z = np.zeros((1, q), np.int64)
        self._exec("mtp_kv", self._mtp_kv_key(q, P, self.last_hidden),
                   self._mtp_kv_args(z, z, np.full((1, P), NULL_PAGE, np.int64), self.pad_slots((1, q)), self.last_hidden,
                                     z))

    def _with_sp_src(self, args: list, hkey: str) -> list:
        """forward_mtp_k's arguments, plus the sp_onehot buffer when the hidden states are a sequence-parallel prefill chunk's
        (this rank's rows only; models/decoder.py _mtp_pass)."""
        if hkey not in self._hidden_sp:
            return args
        return args + [None] * (14 - len(args)) + [PARAM + "sp_onehot"]

    # -- decode -------------------------------------------------------------------

    def decode(self, seqs: list[ScheduledSeq]) -> torch.Tensor:
        """B sequences x one token."""
        n = len(seqs)
        B = pick_family(n, self.plain_buckets, "plain decode")
        P = pick_bucket(-(-max(s.end for s in seqs) // self.ps), self.page_buckets)
        ids = np.zeros(B, np.int64)
        pos = np.zeros(B, np.int64)
        table = np.full((B, P), NULL_PAGE, np.int64)
        ctx = np.ones(B, np.int64)
        slot = self.pad_slots(B)
        read = np.full(B, -1, np.int64)
        write = np.full(B, self.scratch_slot, np.int64)
        rows: list[Request | None] = [None] * B
        for b, s in enumerate(seqs):
            r = s.req
            t = r.token_ids[s.start]
            write[b] = self.slot(r)
            if t == PENDING:
                read[b] = write[b]
            else:
                ids[b] = t
            pos[b] = s.start
            np_pages = min(len(r.pages), P)
            table[b, :np_pages] = r.pages[:np_pages]
            ctx[b] = s.end
            slot[b] = r.pages[s.start // self.ps] * self.ps + s.start % self.ps
            rows[b] = r
        args = [ids, pos, table, ctx, slot, *self._sampling(rows), BOARD, read, write]
        if self.state is not None:
            srow = np.zeros(B, np.int64)  # padded rows: the scratch row
            for b, s in enumerate(seqs):
                srow[b] = self.state.row(s.req)
            args = self._with_state("decode", args, srow)
        out = self._exec("decode", ("decode", B, P), args)
        v = out[:n]
        self.launch_bound.alias(v, out)  # the engine reads v back: it marks out's execution done (LaunchBound)
        return v

    # -- prefill ------------------------------------------------------------------

    def prefill(self, s: ScheduledSeq) -> torch.Tensor:
        """One sequence's chunk, padded to its bucket C."""
        C = pick_bucket(s.num_tokens, self.prefill_buckets)
        P = pick_bucket(-(-s.end // self.ps), prefill_pages(self.prefill_page_buckets, self.prefill_page_min_rows, C))
        r, n = s.req, s.num_tokens
        ids = np.zeros(C, np.int64)
        pos = np.zeros(C, np.int64)
        table = np.full(P, NULL_PAGE, np.int64)
        slot = self.pad_slots(C)
        write = np.full(1, self.scratch_slot, np.int64)
        ids[:n] = r.token_ids[s.start : s.end]
        pos[:n] = np.arange(s.start, s.end)
        np_pages = min(len(r.pages), P)
        table[:np_pages] = r.pages[:np_pages]
        pages = np.asarray(r.pages, np.int64)
        slot[:n] = pages[pos[:n] // self.ps] * self.ps + pos[:n] % self.ps
        last = np.array([n - 1], np.int64)
        if s.sample:
            write[0] = self.slot(r)
        args = [ids, pos, table, slot, last, *self._sampling([r if s.sample else None]), BOARD, write]
        if self.state is not None:
            # [row, ckpt_row, k]: an in-chunk checkpoint copies the state after k of the chunk's tokens into ckpt_row;
            # none: the scratch row receives the chunk's end, k = C.
            srow = np.array([self.state.row(r), 0, C], np.int64)
            if s.mid is not None:
                srow[1], srow[2] = s.mid[1], s.mid[0]
            args = self._with_state("prefill", args, srow)
        return self._exec("prefill", ("prefill", C, P), args)

    # -- speculative steps under overlap scheduling (engine/spec_async.py) ----------------

    def _spec_rows(self, seqs, Q: int, P: int):
        """Host arguments of a blind speculative call: board slots [B] (padding: the scratch slot), page tables [B, P], each
        request's Q state rows [B, Q] (padding: the scratch row 0), and the valid flags, plus the placement."""
        B = pick_family(len(seqs), self.spec_buckets, "verify")
        slot_idx = np.full(B, self.scratch_slot, np.int64)
        valid = np.zeros(B, np.float32)
        table = np.full((B, P), NULL_PAGE, np.int64)
        rows = np.zeros((B, Q), np.int64)
        for b, s in enumerate(seqs):
            r = s.req
            slot_idx[b] = self.slot(r)
            valid[b] = 1.0
            npg = min(len(r.pages), P)
            table[b, :npg] = r.pages[:npg]
            rs = self.state.rows_of(r)
            if len(rs) < Q:
                raise RuntimeError(f"verify of {Q} positions needs {Q} state rows per request, not {len(rs)}")
            rows[b] = rs[:Q]
        return B, slot_idx, valid, table, rows

    def verify_async(self, seqs: list[ScheduledSeq], Q: int):
        """A blind verify (engine/spec_async.py): spec_prep builds its ids, positions, KV slots and state rows from the
        board, the verify graph runs on them, spec_post writes the accepted count and the emitted tokens back. Each seq's
        start is an upper bound of its newest token's position (the page bucket covers it). Returns (the verify output
        [B * Q, VERIFY_COLS], the [B, Q + 2] (tokens, acc, nd) result) on the device."""
        P = self.spec_page(pick_bucket(-(-min(max(s.start + Q for s in seqs), self.ecfg.max_model_len) // self.ps),
                                       self.page_buckets))
        B, slot_idx, valid, table, rows = self._spec_rows(seqs, Q, P)
        cold = [(b, s) for b, s in enumerate(seqs) if not s.req.spec_ready]
        if cold:  # newest token a prompt token the host holds (engine/spec_async.py spec_host_init)
            tok, base, cur, v = (np.zeros(B, np.float32) for _ in range(4))
            for b, s in cold:
                tok[b] = s.req.token_ids[s.start]
                base[b] = s.start
                cur[b] = self.state.row(s.req)
                v[b] = 1.0
                s.req.spec_ready = True
            self._exec("spec_host_init", ("spec_host_init", B), [SPEC, slot_idx, tok, base, cur, v])
        self._exec("spec_prep", ("spec_prep", B, Q, P), [SPEC, slot_idx, table, rows, self.pad_slots((B, Q)), valid])
        rq: list[Request | None] = [None] * (B * Q)
        for b, s in enumerate(seqs):
            rq[b * Q : (b + 1) * Q] = [s.req] * Q
        samp = list(self._sampling(rq))
        u = np.array([self._rngs[r.rid].random() if r is not None and r.rid in self._rngs else 0.5 for r in rq],
                     np.float32)
        args = [DEV + "spec_prep:0", DEV + "spec_prep:2", table, DEV + "spec_prep:3", *samp, u, DEV + "spec_prep:1"]
        args = self._with_state("verify", args, DEV + "spec_prep:4")
        out = self._exec("verify", ("verify", B, Q, P), args)
        hkey = self.last_hidden
        sx = []
        if self._synth is not None:  # one step's flags a request, drawn now (the host knows nothing of the step's outcome)
            fl = np.zeros((B, self.spec_k), np.float32)
            for b, s in enumerate(seqs):
                fl[b] = self._synth.flags(s.req.rid, s.req.params.seed)
            sx = [fl]
        res = self._exec("spec_post", ("spec_post", B, Q),
                         [SPEC, slot_idx, DEV + "verify", DEV + "spec_prep:0", rows.astype(np.float32), valid] + sx)
        if self.async_ckpt_t and any(s.ckpt_row for s in seqs):
            ck = np.zeros(B, np.float32)  # the rows the engine reserved for a crossed decode checkpoint
            for b, s in enumerate(seqs):
                ck[b] = s.ckpt_row or 0
            self._exec("spec_ckpt", ("spec_ckpt", B, Q), [SPEC, slot_idx, rows.astype(np.float32), ck, valid])
        self._spec_last = (len(seqs), B, slot_idx, valid, table, P, hkey)
        return out, res

    def mtp_async(self) -> None:
        """The MTP pass after the last verify_async: its positions, ids and last index from the board, its drafts into the
        board, in one graph (mtp_board) over the verify's hidden states."""
        n, B, slot_idx, valid, table, P, hkey = self._spec_last
        Q = 1 + self.spec_k
        hidx = np.zeros((B, Q), np.int64)  # a padding row reads the first row
        for b in range(n):
            hidx[b] = b * Q + np.arange(Q)  # the verify's rows of this sequence
        self._exec("mtp_board", self._mtp_board_key(B, Q, P, hkey),
                   [SPEC, slot_idx, table, self.pad_slots((B, Q)), valid, hkey, hidx, PARAM + "norm"])

    def spec_after_prefill(self, seqs, hkey: str) -> None:
        """After a final prefill chunk: the request's board row (spec_init: the sampled token from the token board the
        prefill wrote, base = its position, cur = its state row) and the MTP pass over the chunk, whose last id is that
        token (mtp_prefill_ids). A chunk that is not final only fills the MTP KV (its next ids are prompt tokens)."""
        (x,) = seqs
        slot_idx = np.full(1, self.scratch_slot, np.int64)
        valid = np.zeros(1, np.float32)
        base = np.zeros(1, np.float32)
        cur = np.zeros(1, np.float32)
        if x.sample:
            slot_idx[0] = self.slot(x.req)
            valid[0] = 1.0
            base[0] = x.end
            cur[0] = self.state.row(x.req)
            self._exec("spec_init", ("spec_init", 1), [SPEC, BOARD, slot_idx, base, cur, valid])
            x.req.spec_ready = True
        # The MTP pass over the chunk: positions start .. end - 1 take the token after them (prompt tokens, the last the
        # sampled one), hidden rows of the prefill's output. With mtp_fill_rows F it runs in calls of at most F rows over the
        # SAME hidden stash (hidx indexes the chunk's rows in full, also with sequence-parallel stashes, which the graph
        # gathers whole): call c covers rows [c F, (c + 1) F) of the chunk, reads the MTP KV the earlier calls wrote (stream
        # order), and only the LAST call carries the sampled token, the later draft passes' real positions and mtp_post. An
        # earlier call's later passes write positions the next call rewrites (the draft layer's KV of a position is its own
        # input's projection: the calls fill exactly the one call's KV).
        F = self.after_prefill_rows
        m = x.num_tokens
        r_ = self.spec_k - 1  # the later passes' positions end .. end + r_ - 1
        # Every call of a chunk that needs more than one uses the SAME q (F's bucket): a tail call of a few rows would
        # otherwise pick a smaller bucket, a key warmup never builds (it builds q = min(C, F)) and a runtime compile of a
        # large graph. q follows the CALL that produced the stash, not the chunk in it: a prefill call is padded to its
        # bucket C and the graph key holds the stash (C rows, or C / tp under sequence-parallel prefill).
        C_call = self._hidden_c.get(hkey)
        if C_call is None:
            C_call = m  # (a stash the runner did not register)
        elif C_call < m:
            raise ValueError(f"spec_after_prefill: a chunk of {m} rows over a prefill call of bucket {C_call}")
        Qm = pick_bucket(F if F and C_call > F else C_call, self.mtp_q_buckets)
        # The page rung follows the call too: a prefill call of fewer rows than prefill_page_min_rows takes ONLY the largest
        # rung (config.prefill_pages) and warmup builds this pass for exactly those rungs.
        rungs = prefill_pages(self.prefill_page_buckets, self.prefill_page_min_rows, C_call) if hkey in self._hidden_c else self.prefill_page_buckets
        P = self.spec_page(pick_bucket(-(-(x.end - 1 + self.spec_k) // self.ps), rungs))
        ran_full = False  # a window ran the full pass (the one with a final row, or every window without KV-only windows)
        r = x.req
        pages = np.asarray(r.pages, np.int64)
        for a_ in range(0, m, F or m):
            ids = np.zeros((1, Qm), np.int64)
            pos = np.zeros((1, Qm), np.int64)
            slot = self.pad_slots((1, Qm))
            hidx = np.zeros((1, Qm), np.int64)
            last = np.zeros(1, np.int64)
            table = np.full((1, P), NULL_PAGE, np.int64)
            last_tok = np.full(1, -1, np.int64)  # the column taking the sampled token from the token board (-1: none)
            pos_r = np.zeros((1, r_), np.int64)
            slot_r = self.pad_slots((1, r_))
            b_ = min(m, a_ + F) if F else m
            cnt, final = b_ - a_, b_ == m
            has_last = bool(x.sample and final)
            if has_last:
                ids[0, : cnt - 1] = r.token_ids[x.start + 1 + a_ : x.end]
                last_tok[0] = cnt - 1
            else:  # a non-final call of a final chunk, or any call of a non-final one: all prompt tokens
                ids[0, :cnt] = r.token_ids[x.start + 1 + a_ : x.start + 1 + b_]
            pp = np.arange(x.start + a_, x.start + b_)
            pos[0, :cnt] = pp
            slot[0, :cnt] = pages[pp // self.ps] * self.ps + pp % self.ps
            hidx[0, :cnt] = a_ + np.arange(cnt)
            last[0] = cnt - 1
            npg = min(len(r.pages), P)
            table[0, :npg] = r.pages[:npg]
            for s in range(r_):  # the engine reserved these pages (a dry pool: a padding write, a worse draft)
                p = (x.end if final else x.start + b_) + s
                pos_r[0, s] = p
                if p // self.ps < len(pages):
                    slot_r[0, s] = pages[p // self.ps] * self.ps + p % self.ps
            if not has_last and self._mtp_kv_on(Qm, P):  # this window needs only the MTP layer's KV (no draft is read from it, no later pass of it survives: the next window rewrites those positions)
                self._exec("mtp_kv", self._mtp_kv_key(Qm, P, hkey), self._mtp_kv_args(ids, pos, table, slot, hkey, hidx))
                continue
            self._exec("mtp_prefill_ids", ("mtp_prefill_ids", 1, Qm), [BOARD, slot_idx, ids, last_tok])
            rest = [pos_r, slot_r] if r_ else [None, None]
            args = [DEV + "mtp_prefill_ids:0", pos, table, slot, hkey, hidx, last, PARAM + "norm", *rest]
            self._exec("mtp", self._mtp_key(1, Qm, P, hkey), self._with_sp_src(args, hkey))
            ran_full = True
        if ran_full or not self._mtp_kv_on(Qm, P):  # (mtp_post reads the last full pass's drafts; with only KV-only windows every row is invalid and it would write nothing)
            self._exec("mtp_post", ("mtp_post", 1, 1 + self.spec_k), [SPEC, slot_idx, DEV + "mtp", valid])

    # -- recurrent-state checkpoints (engine/state_pool.py) ---------------------------

    def copy_state(self, src: list[int], dst: list[int]) -> None:
        """Copy whole state rows on the device (every linear layer's conv and recurrent row), in order with the graph
        calls around it."""
        if not src:
            return
        self.state.check_writes(dst)
        top = self.copy_buckets[-1]
        for i in range(0, len(src), top):
            pairs = list(zip(src[i : i + top], dst[i : i + top]))
            n = pick_bucket(len(pairs), self.copy_buckets)
            a, b = np.zeros(n, np.int64), np.zeros(n, np.int64)  # padding: the scratch row onto itself
            for j, (x, y) in enumerate(pairs):
                a[j], b[j] = x, y
            if self.tp_send is not None:
                self.tp_send(("state_copy", None, [a, b]))
            self._copy_state(a, b)

    def _copy_state(self, a, b) -> None:
        """One copy call: a, b [n] host arrays of source and destination rows."""
        a, b = self._dev(a), self._dev(b)
        key = ("copy", a.shape[0])
        first = key not in self.compile_seconds
        t = time.perf_counter()
        out = self._copy(a, b, *self.state.pools())
        if first:
            if self.device.type == "neuron":
                out.cpu()
            self.compile_seconds[key] = time.perf_counter() - t
        self.calls[key] += 1
        self._last_out = out

    # -- host cache (engine/host_cache.py) --------------------------------------------
    # Every rank holds its own share of each page and state row, so each call is broadcast and every rank copies its own.

    def kv_save(self, slot: int, page: int) -> None:
        if self.tp_send is not None:
            self.tp_send(("kv_save", None, [slot, page]))
        self._kv_save(slot, page)

    def kv_load(self, slot: int, page: int) -> None:
        if self.tp_send is not None:
            self.tp_send(("kv_load", None, [slot, page]))
        self._kv_load(slot, page)

    def kv_drop(self, slot: int) -> None:
        if self.tp_send is not None:
            self.tp_send(("kv_drop", None, [slot]))
        self._kv_drop(slot)

    def _paged_caches(self) -> list[torch.Tensor]:
        return self.k_caches + self.v_caches + self.aux_caches + self.s_caches

    def _kv_save(self, slot: int, page: int) -> None:
        sl = slice(page * self.lps, (page + 1) * self.lps)  # a page is ps (cp: ps / cp) contiguous rows of every cache
        # copy=True: on a CPU engine .cpu() would return a view of the live cache.
        self._host_kv[slot] = [c[sl].to("cpu", copy=True) for c in self._paged_caches()]

    def _settle(self) -> None:
        """Wait for every graph call launched on this rank. An eager host-to-device copy into a device tensor does NOT
        wait for calls already queued on the NeuronCore: a copy into a KV slot issued right after launching a decode
        call that writes that slot was overwritten by the call's write in 200 of 200 tries (SDK 2.32). The host cache's
        restores are such copies, issued while an overlapped step may still be in flight (its last write of a finished
        request's freed tail page, which a restore can be given). Calls run in submission order, so reading the last
        one's output back waits for all of them; clearing it keeps a batch of restores to one wait."""
        out, self._last_out = self._last_out, None
        if out is not None and self.device.type == "neuron":
            out.cpu()

    def _kv_load(self, slot: int, page: int) -> None:
        self._settle()
        sl = slice(page * self.lps, (page + 1) * self.lps)
        for h, c in zip(self._host_kv[slot], self._paged_caches()):
            c[sl].copy_(h)

    def _kv_drop(self, slot: int) -> None:
        self._host_kv.pop(slot, None)

    def state_save(self, slot: int, row: int) -> None:
        """Host copy of one state row (a checkpoint the radix cache evicted, engine/host_cache.py)."""
        if self.tp_send is not None:
            self.tp_send(("state_save", None, [slot, row]))
        self._state_save(slot, row)

    def state_load(self, slot: int, row: int) -> None:
        if self.tp_send is not None:
            self.tp_send(("state_load", None, [slot, row]))
        self._state_load(slot, row)

    def state_drop(self, slot: int) -> None:
        if self.tp_send is not None:
            self.tp_send(("state_drop", None, [slot]))
        self._state_drop(slot)

    def _state_save(self, slot: int, row: int) -> None:
        self._host_ckpt[slot] = [p[row].to("cpu", copy=True) for p in self.state.pools()]

    def _state_load(self, slot: int, row: int) -> None:
        self._settle()  # as _kv_load
        for h, p in zip(self._host_ckpt[slot], [p[row] for p in self.state.pools()]):
            p.copy_(h)

    def _state_drop(self, slot: int) -> None:
        self._host_ckpt.pop(slot, None)

    def kv_page_bytes(self) -> int:
        """Host bytes one page takes on this rank (every layer's K and V, and the per-token caches and token-slot states
        beside them)."""
        return sum(self.lps * c[0].numel() * c.element_size() for c in self._paged_caches())

    # -- tensor-parallel worker -----------------------------------------------------

    def serve(self, recv) -> None:
        """Ranks > 0: execute every call rank 0 broadcasts, until it sends None."""
        local = {"state_copy": self._copy_state, "kv_save": self._kv_save, "kv_load": self._kv_load,
                 "kv_drop": self._kv_drop, "state_save": self._state_save, "state_load": self._state_load,
                 "state_drop": self._state_drop}
        while True:
            msg = recv()
            if msg is None:
                return
            name, key, host_args = msg[:3]
            if len(msg) > 3:  # an execution: rank 0's count of finished executions comes with it
                self.launch_bound.done_to(msg[3])
            if name in local:
                local[name](*host_args)
                continue
            out = self._exec(name, key, host_args)
            if self.calls[key] <= 1:
                out.cpu()  # surface a compile or runtime error on this rank, once


def state_pool_rows(ecfg: EngineConfig) -> int:
    """Rows of the recurrent-state pool an EngineConfig implies: the scratch row, the running seats' rows (a seat that
    speculates holds 1 + spec_k, the others one), the checkpoint rows and the zero row. The pool's shape is an input of every
    graph that carries the state, so a configuration's row count is part of its compiled graphs."""
    seats = ecfg.state_seats or ecfg.max_num_seqs
    per_req = 1 + ecfg.spec_k if ecfg.spec_method else 1
    spec = seats if ecfg.state_spec_seats is None else ecfg.state_spec_seats
    return 1 + spec * per_req + (seats - spec) + state_checkpoint_rows(ecfg) + 1


def family_buckets(ecfg: EngineConfig, buckets: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(plain decode buckets, verify-family buckets): the decode buckets each step family is warmed and picked at; both the
    decode buckets when EngineConfig.spec_bucket_max and plain_bucket_min are 0. With dynamic speculation the two families
    use disjoint row counts: the verify family (verify, the speculative board graphs, the decode-side MTP pass) only runs
    while at most spec_max_batch requests decode, plain decode only above it (down to spec_max_batch - spec_hysteresis + 1).
    spec_bucket_max keeps the verify family to buckets <= B; plain_bucket_min keeps plain decode to buckets >= B, a smaller
    batch padded up. Every kept graph's key is the same as without the knobs, and the dropped ones are never loaded."""
    lo, hi, m = ecfg.plain_bucket_min, ecfg.spec_bucket_max, ecfg.spec_max_batch
    if not lo and not hi:
        return tuple(buckets), tuple(buckets)
    plain = tuple(b for b in buckets if not lo or b >= lo)
    spec = tuple(b for b in buckets if not hi or b <= hi)
    if not plain or plain[-1] != buckets[-1]:
        raise ValueError(f"plain_bucket_min={lo} leaves the plain decode no bucket for the largest batch ({buckets})")
    if not spec or spec[-1] < m:
        raise ValueError(f"spec_bucket_max={hi}: the verify family's buckets {spec} do not cover spec_max_batch {m}")
    return plain, spec


def pick_family(n: int, buckets: tuple[int, ...], family: str) -> int:
    """pick_bucket over one step family's warmed buckets (family_buckets), raising a clear error past them: a launch outside
    the warmed set must never compile or look up a missing graph at run time."""
    if n > buckets[-1]:
        raise RuntimeError(f"a {family} launch of {n} rows exceeds its warmed buckets {buckets}: not compiled, refused")
    return pick_bucket(n, buckets)


def spec_page_buckets(ecfg: EngineConfig, page_buckets: tuple[int, ...]) -> tuple[int, ...]:
    """The page ladder of every graph the MTP layer or a verify runs in (verify, the speculative board graphs, the fused and
    unfused MTP passes, the KV-only windows): EngineConfig.spec_page_buckets, else the page buckets. Plain decode and prefill
    keep page_buckets. A rung the MTP layer's attention cannot be built at (the 3072-page rung of the shipped configurations:
    its context-parallel selection kernel needs a row count that is neither <= 128 nor a multiple of 128 there) can then
    exist for plain decode only."""
    pb = tuple(sorted(page_buckets))
    if not ecfg.spec_page_buckets:
        return pb
    sp = tuple(sorted(set(ecfg.spec_page_buckets)))
    if not set(sp) <= set(pb) or sp[-1] != pb[-1]:
        raise ValueError(f"spec_page_buckets {sp} must be page buckets {pb} including the largest (every context has a rung)")
    return sp
