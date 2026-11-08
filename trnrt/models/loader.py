"""Load GLM-5.3-Flash's Hugging Face safetensors checkpoint (native block-scaled FP8) into the model."""

from __future__ import annotations

import glob
import os

import torch
from safetensors import safe_open

from . import glm5_next, hybrid, linear_attn, mla
from .config import LinearSpec, ModelConfig
from .decoder import DecoderForCausalLM
from .quant import FP8, dequant, fit_e4m3_max, quantize_fp8_rows

ARCHITECTURES = {a: DecoderForCausalLM for a in glm5_next.ARCHITECTURES}

# Checkpoint tensor names (suffixes after model.layers.{i}.).
NAMES = {
    "router": "mlp.gate.weight",
    "router_bias": "mlp.gate.e_score_correction_bias",
    "shared": "mlp.shared_experts.",
}


def names(cfg: ModelConfig) -> dict:
    return dict(NAMES)


# What a Hugging Face repo id's download takes: the configs, the weights, the tokenizer's files, and the chat template,
# which GLM-5.3-Flash ships as chat_template.jinja alone (its tokenizer_config.json carries none).
DOWNLOAD_PATTERNS = ("*.json", "*.safetensors", "tokenizer*", "*.txt", "*.model", "*.jinja")


def resolve_model_path(model: str, revision: str | None = None) -> str:
    """A local directory, or a Hugging Face repo id downloaded to the local cache at `revision` (a commit, a branch or a
    tag; None: the repo's default branch). A revision pins a download only: with a local directory it is refused."""
    if os.path.isdir(model):
        if revision is not None:
            raise ValueError(f"revision {revision!r} given with a local checkpoint directory ({model}): a revision "
                             "only selects what a Hugging Face repo id downloads")
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model, revision=revision, allow_patterns=list(DOWNLOAD_PATTERNS))


class _Alias:
    """A safetensors handle that serves one tensor under another name."""

    def __init__(self, handle, name: str):
        self._h, self._name = handle, name

    def get_slice(self, _name: str):
        return self._h.get_slice(self._name)

    def get_tensor(self, _name: str):
        return self._h.get_tensor(self._name)


class _Checkpoint:
    """Lazy view of a safetensors checkpoint. A tensor-parallel rank reads only its own rows
    or columns of each weight (safetensors get_slice reads just that part through mmap), so
    host memory per rank is its shard, not the model."""

    def __init__(self, path: str):
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no .safetensors files in {path}")
        self._handles = [safe_open(fn, framework="pt") for fn in files]
        self._where = {name: h for h in self._handles for name in h.keys()}
        # The composite checkpoint keeps the language model under model.language_model.; read it
        # under the usual model. names.
        vl = "model.language_model."
        for name in [n for n in self._where if n.startswith(vl)]:
            self._where.setdefault("model." + name[len(vl):], _Alias(self._where[name], name))

    def __contains__(self, name: str) -> bool:
        return name in self._where

    def shape(self, name: str) -> list[int]:
        return self._where[name].get_slice(name).get_shape()

    def linear(self, base: str, rows: slice | None = None, cols: slice | None = None, block: int | None = None):
        """Weight `base`.weight in the model's format: (w, scale) with scale fp32 [n, k / bk] for an FP8
        checkpoint weight (128 x 128 blocks, see models/quant.py), else (w, None). rows / cols select a
        tensor-parallel shard. A column shard narrower than or misaligned with its blocks is cut exactly when
        it lies inside one block and returned dequantized (scale None) when it straddles block edges."""
        wname = base + ".weight"
        if base + ".weight_scale_inv" not in self:
            return self.get(wname, rows, cols), None
        bs = block or 128
        n_full = self.shape(wname)[0]
        w = self.get(wname, rows)
        s = _row_scales(self.get(base + ".weight_scale_inv").float(), n_full, bs)
        s = s[rows] if rows is not None else s
        bk = bs
        if cols is not None:
            if cols.start % bk == 0 and (cols.stop % bk == 0 or cols.stop == w.shape[1]):
                w, s = w[:, cols], s[:, cols.start // bk : -(-cols.stop // bk)]
            elif cols.start // bk == (cols.stop - 1) // bk:
                # A shard inside one block (the experts at tp=32: 64 of a 128-wide block) keeps that
                # block's scale: exact, one scale column over the shard.
                b = cols.start // bk
                w, s = w[:, cols], s[:, b : b + 1]
            else:
                # A shard straddling block edges cannot keep the checkpoint's blocks in [n, k / bk] form:
                # hand back the dequantized values and let _assign quantize them again at the rank's own
                # block width (one extra FP8 rounding).
                sc = s.repeat_interleave(bk, dim=1)[:, : w.shape[1]]
                return (w[:, cols].float() * sc[:, cols]), None
        return w, s

    def get(self, name: str, rows: slice | None = None, cols: slice | None = None) -> torch.Tensor:
        if name not in self._where:
            raise KeyError(name)
        if rows is None and cols is None:
            return self._where[name].get_tensor(name)  # get_slice()[:] refuses 0-dim tensors (scalar scales)
        sl = self._where[name].get_slice(name)
        if cols is None:
            return sl[rows]  # contiguous rows: read only those
        # A column slice through get_slice is a strided read and measured far slower than reading the
        # rows and cutting in torch, so read rows, then cut columns.
        return sl[rows or slice(None)][:, cols]


def load_model(path: str, cfg: ModelConfig, dtype: torch.dtype, device: torch.device, max_positions: int,
               tp_rank: int = 0, tp_size: int = 1, tp_group=None, keep_fp8: bool = False,
               fp8_max: float = 240.0, vocab_parallel: bool = False, moe_kernel: str = "xla",
               attn_tp: int | None = None, attn_group=None, mtp: bool = False):
    """keep_fp8: weights the checkpoint quantizes stay FP8 on the device (dequantized in-graph);
    otherwise they are dequantized to `dtype` here. fp8_max: the device's largest finite e4m3.
    attn_tp / attn_group: the attention TP (DecoderForCausalLM; None = its default). mtp: build and load the
    checkpoint's multi-token-prediction layer (DecoderForCausalLM's mtp; cfg.mtp_layers must be set)."""
    cls = ARCHITECTURES.get(cfg.architecture)
    if cls is None:
        raise NotImplementedError(f"architecture {cfg.architecture} is not supported (have: {sorted(ARCHITECTURES)})")
    # Build on the meta device and materialise piece by piece: a module is allocated on the
    # host, filled from the checkpoint and moved to `device` before the next one is
    # allocated, so host memory peaks at one layer per rank instead of the whole shard.
    with torch.device("meta"):
        model = cls(cfg, dtype, max_positions, tp_rank, tp_size, tp_group, keep_fp8=keep_fp8,
                    vocab_parallel=vocab_parallel, moe_kernel=moe_kernel, attn_tp=attn_tp, attn_group=attn_group,
                    mtp=mtp)
    model.materialize = lambda mod: _materialize(mod, device)
    _load_decoder(model, path, dtype, fp8_max)
    model.register_buffer("vocab_start_t", torch.tensor(model.vocab_start, dtype=torch.int64).to(device),
                          persistent=False)
    model.rank_buffers(device)
    del model.materialize
    stray = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.device != device]
    if stray:
        raise RuntimeError(f"{len(stray)} tensors not on {device} after loading, e.g. {stray[:3]}")
    return model.eval()


def _materialize(mod: torch.nn.Module, device: torch.device):
    """Give `mod`'s own (non-recursive) meta parameters host storage; returns a function
    that moves them to `device` once they are filled. The decode-only ones are zeroed: a packed blob's fill
    (kernels/shell.register_packed) writes only its views, so its padding would hold run-to-run varying bytes."""
    params = [(n, p) for n, p in mod.named_parameters(recurse=False) if p.device.type == "meta"]
    zero = set(getattr(mod, "_decode_only", ()))
    for n, p in params:
        new = torch.zeros if n in zero else torch.empty
        setattr(mod, n, torch.nn.Parameter(new(p.shape, dtype=p.dtype), requires_grad=False))

    def done():
        # Every host parameter, including ones that replaced the materialised ones while the
        # module was filled (DecoderLayer.pack_experts swaps four expert tensors for w_blob).
        for n, p in list(mod.named_parameters(recurse=False)):
            if p.device != device:
                setattr(mod, n, torch.nn.Parameter(p.data.to(device), requires_grad=False))

    return done


def _row_scales(scale_inv: torch.Tensor, n_rows: int, bs: int) -> torch.Tensor:
    """Per-row scales [n_rows, K / bs] from 128 x 128 block scales.

    A weight saved from T tensor-parallel shards may restart its row blocks at every shard (the scale
    rows are then T x ceil(n_rows / T / bs), not ceil(n_rows / bs)). The shard count is the one that
    explains the scale row count."""
    nb = -(-n_rows // bs)
    if scale_inv.shape[0] == nb:
        return scale_inv.repeat_interleave(bs, dim=0)[:n_rows]
    for T in range(2, 129):
        if n_rows % T == 0 and T * -(-(n_rows // T) // bs) == scale_inv.shape[0]:
            per, nbs = n_rows // T, -(-(n_rows // T) // bs)
            return torch.cat([scale_inv[t * nbs : (t + 1) * nbs].repeat_interleave(bs, dim=0)[:per] for t in range(T)])
    raise ValueError(f"{scale_inv.shape[0]} block-scale rows do not fit {n_rows} rows in {bs}-row blocks")


def _part(n_total: int, rank: int, size: int) -> slice:
    k = n_total // size
    return slice(rank * k, (rank + 1) * k)


def _concat(parts: list[tuple[torch.Tensor, torch.Tensor | None]]):
    """Row-concatenate (w, scale) pieces of one fused weight. Pieces that disagree on being
    quantized are all dequantized, so the result is consistent either way."""
    ws, ss = [p[0] for p in parts], [p[1] for p in parts]
    if all(x is not None for x in ss):
        return torch.cat(ws), torch.cat(ss)
    if all(x is None for x in ss):
        return torch.cat(ws), None
    return torch.cat([dequant(w, s, torch.float32) if s is not None else w.float() for w, s in parts]), None


def _assign(param, scale_param, w: torch.Tensor, s: torch.Tensor | None, dtype, bk: int | None,
            fp8_max: float, row_group: int | None = None) -> None:
    """Store checkpoint data (w, s) into a model weight: FP8 + scale when the model keeps it
    quantized, otherwise dequantized to the model dtype. row_group: fit_e4m3_max's."""
    if scale_param is None:
        param.data.copy_(dequant(w, s, torch.float32).to(dtype) if s is not None else w.to(dtype))
        return
    if s is None:  # checkpoint left this weight unquantized: quantize it ourselves
        w, s = quantize_fp8_rows(w.float(), bk)
    w, s = fit_e4m3_max(w.to(FP8), s, fp8_max, row_group)
    param.data.copy_(w)
    scale_param.data.copy_(s)


def _load_decoder(model: DecoderForCausalLM, path: str, dtype: torch.dtype, fp8_max: float = 240.0) -> None:
    cfg = model.cfg
    ck = _Checkpoint(path)
    materialize = getattr(model, "materialize", lambda mod: (lambda: None))

    def put_norm(param, name):
        """An RMSNorm weight."""
        param.data.copy_(ck.get(name).float().to(dtype))

    def put_vocab(param, name):
        """This rank's vocabulary rows; rows past the vocabulary (the last shard's padding)
        are zero, so their logits are 0 and are cut off after the gather."""
        start, rows, V = model.vocab_start, model.vocab_rows, cfg.vocab_size
        stop = min(start + rows, V)
        param.data.zero_()
        param.data[: stop - start].copy_(ck.get(name, slice(start, stop)).to(dtype))

    top_done = materialize(model)
    put_vocab(model.embed, "model.embed_tokens.weight")
    put_norm(model.norm, "model.norm.weight")
    if model.lm_head is not None:
        if "lm_head.weight" not in ck:
            raise KeyError("checkpoint has untied embeddings but no lm_head.weight")
        put_vocab(model.lm_head, "lm_head.weight")
    mtp = getattr(model, "mtp", None)
    if mtp is not None:  # the MTP head's own parameters live on the model (enorm, hnorm, eh_proj, shared_head.norm)
        m = cfg.mtp_prefix + "."
        put_norm(model.mtp_enorm, m + "enorm.weight")
        put_norm(model.mtp_hnorm, m + "hnorm.weight")
        put_norm(model.mtp_norm, m + "shared_head.norm.weight")
        w, sc = ck.linear(m + "eh_proj")  # bf16 in the released checkpoint; dequantized if not
        model.mtp_eh.data.copy_((dequant(w, sc, torch.float32) if sc is not None else w).to(dtype))
    top_done()
    r, n = model.tp_rank, model.tp_size  # MLP / experts shard
    ar, an = model.attn_rank, model.attn_tp  # token-mixer shard (attention TP)
    blk = cfg.quant_block
    nm = names(cfg)
    jobs = [(i, layer, f"model.layers.{i}.") for i, layer in enumerate(model.layers)]
    if mtp is not None:  # a whole decoder layer under the MTP prefix (its own indexer, the MoE)
        jobs.append(("mtp", mtp, cfg.mtp_prefix + "."))
    for i, layer, pre in jobs:
        layer_done = materialize(layer)
        sp = layer.spec
        try:
            put_norm(layer.in_norm, pre + "input_layernorm.weight")
            put_norm(layer.post_norm, pre + "post_attention_layernorm.weight")
            if isinstance(sp, LinearSpec):
                linear_attn.load_mixer(layer, ck, pre, cfg, ar, an, dtype, blk, fp8_max)
            else:
                mla.load_layer(layer, ck, pre, cfg, ar, dtype,
                               lambda p_, s_, w_, sc_: _assign(p_, s_, w_, sc_, dtype, blk, fp8_max), _concat, put_norm)
            if layer.moe:
                _load_experts(layer, ck, pre, cfg, r, n, dtype, fp8_max)
                if layer.shared_gate_up is not None:  # mlp.shared_experts.*
                    sh, base = _part(cfg.moe_intermediate_size * cfg.n_shared_experts, r, n), pre + nm["shared"]
                    w, sc = _concat([ck.linear(base + "gate_proj", sh, block=blk),
                                     ck.linear(base + "up_proj", sh, block=blk)])
                    _assign(layer.shared_gate_up, layer.shared_gate_up_scale, w, sc, dtype, blk, fp8_max)
                    w, sc = ck.linear(base + "down_proj", cols=sh, block=blk)
                    _assign(layer.shared_down, layer.shared_down_scale, w, sc, dtype, blk, fp8_max)
                if layer.pack_moe:
                    layer.pack_experts()
                if getattr(layer, "mh_blob_bf16", None) is not None:  # the decode FFN half's blobs
                    from ..kernels import moe_half

                    moe_half.fill(layer, dtype)
            else:
                im = _part(cfg.intermediate_size, r, n)
                w, sc = _concat([ck.linear(pre + "mlp.gate_proj", im, block=blk),
                                 ck.linear(pre + "mlp.up_proj", im, block=blk)])
                _assign(layer.gate_up, layer.gate_up_scale, w, sc, dtype, blk, fp8_max)
                w, sc = ck.linear(pre + "mlp.down_proj", cols=im, block=blk)
                _assign(layer.down, layer.down_scale, w, sc, dtype, blk, fp8_max)
                if getattr(layer, "dh_blob_bf16", None) is not None:  # the dense FFN half's blob
                    from ..kernels import dense_half

                    dense_half.fill(layer, dtype)
            if not getattr(layer, "plain", False):  # the MTP layer has no hyper-connections
                hybrid.load_layer(layer, ck, pre, cfg, r, n, dtype)
        except KeyError as e:
            raise KeyError(f"layer {i}: checkpoint is missing {e}") from None
        layer_done()


def _load_experts(layer, ck: _Checkpoint, pre: str, cfg: ModelConfig, r: int, n: int, dtype,
                  fp8_max: float = 240.0) -> None:
    """The hub checkpoint names each expert (mlp.experts.{e}.gate_proj.weight ...); transformers 5
    saves them fused as 3-D tensors (mlp.experts.gate_up_proj [E, 2I, H], gate rows first,
    and mlp.experts.down_proj [E, H, I]). Accept both; shard I across tensor-parallel ranks.
    FP8 blocks are fitted onto e4m3 max per (row, block) (models/quant.fit_e4m3_max)."""
    E, I = cfg.num_experts, cfg.moe_intermediate_size
    sl = _part(I, r, n)
    nm = names(cfg)
    layer.router.data.copy_(ck.get(pre + nm["router"]).to(dtype))
    if layer.router_bias is not None:
        layer.router_bias.data.copy_(ck.get(pre + nm["router_bias"]).float())
    m = pre + "mlp.experts."
    bk = cfg.quant_expert_block
    if m + "gate_up_proj" in ck:
        gu = ck.get(m + "gate_up_proj")
        _assign_experts(layer, torch.cat([gu[:, sl], gu[:, I:][:, sl]], dim=1), None,
                        ck.get(m + "down_proj")[:, :, sl], None, dtype, bk, fp8_max)
        return
    for e in range(E):
        g, gs = ck.linear(f"{m}{e}.gate_proj", sl, block=bk)
        u, us = ck.linear(f"{m}{e}.up_proj", sl, block=bk)
        d, ds = ck.linear(f"{m}{e}.down_proj", cols=sl, block=bk)
        gus = torch.cat([gs, us]) if gs is not None else None
        _assign(_Row(layer.w_gu, e), _Row(layer.w_gu_scale, e) if layer.w_gu_scale is not None else None,
                torch.cat([g, u]), gus, dtype, bk, fp8_max)
        _assign(*_down(layer, e), d, ds, dtype, bk, fp8_max)


class _Row:
    """One expert's slice of a stacked [E, ...] parameter, assignable like a parameter;
    transposed: a view in checkpoint ([out, in]) order of a weight stored [in, out]."""

    def __init__(self, param, e: int, transposed: bool = False):
        self.data = param.data[e].T if transposed else param.data[e]


def _down(layer, e: int):
    t = layer.down_t
    return _Row(layer.w_down, e, t), (_Row(layer.w_down_scale, e, t) if layer.w_down_scale is not None else None)


def _assign_experts(layer, gu, gus, down, downs, dtype, bk, fp8_max, row_group: int | None = None) -> None:
    for e in range(gu.shape[0]):
        _assign(_Row(layer.w_gu, e), _Row(layer.w_gu_scale, e) if layer.w_gu_scale is not None else None,
                gu[e], gus[e] if gus is not None else None, dtype, bk, fp8_max, row_group)
        _assign(*_down(layer, e), down[e], downs[e] if downs is not None else None, dtype, bk, fp8_max, row_group)
