"""The KDA attention half of a GLM-5.3-Flash decode layer: the layouts and NKI helpers that kernels/kda_shell.py's
kernel (the served decode form) builds on, and the half's torch emulation.

What the half computes for one KDA attention block, per rank, B decode rows (models/linear_attn.py's decode form):

    xn   = rms_norm(x, in_norm)                                    [B, H] bf16 (H = 4096)
    qkv  = xn in_qkv^T;  fa = xn f_a^T;  ga = xn g_a^T;  bl = xn in_b^T     one fused projection, Wcat [1026, H]
    conv = causal depthwise conv of (conv_state[slot], qkv), K = 4; conv_state[slot] <- the last 3 inputs
    q, k, v = silu(conv) split into nv = 2 heads of 128;  q, k l2-normed (fp32), q * 128^-0.5
    g    = lb sigmoid(exp(A_log) (fa f_b^T + dt_bias));  beta = sigmoid(bl);  z = ga g_b^T
    o    = the KDA recurrent step on rec_state[slot] (models/linear_attn.py recurrent_step's arithmetic, in place)
    og   = gated RMSNorm(o) * sigmoid(z)                           [B, 256] bf16
    y    = all_reduce(og out^T)                                    [B, H] bf16, summed over the tp ranks

In XLA these are ~25 ops, two NKI kernel calls and an all-reduce: on trn2 at B = 1 the half cost ~86 us of device
time (B = 2: ~197 us), of which the recurrent step's own kernel was ~9 us.

Layouts:
- x in h-layout, h = p W + j (W = H / 128 = 32): every vector op of the input norm on 128 lanes, and the projection's
  k-tile j is the stationary xn[:, j, :] [128, B] against the weight's rows {p W + j}. The per-program weight [H, NCp]
  viewed as [128, W, NCp] gives each partition its W rows contiguously (W NCp 2 bytes per partition, ~41 KB at NCp 641).
- The projection lands with rows (B) on the partitions; PE transposes put its q, k, v, f_a and g_a columns on the
  partitions ("feature tiles" [128, B]), where the conv, the gates, the l2 norms and the gated norm run on 128 lanes.
- Programs split HEADS, not rows: at LNC = 2 (grid 2) and nv = 2, program p owns head p. It streams only its head's
  q, k, v and b columns (plus f_a and g_a, which both heads need: duplicated, 2 MB), its conv channels, f_b / g_b rows,
  its recurrent state (disjoint pool bytes: no write race with the other program), and its head's rows of the output
  projection, over all columns. One NKI program streams 340-400 GB/s, a logical core's two programs 672-708 (trn2), so a
  fused kernel must split its weight bytes, not duplicate them. The output projection contracts over
  both heads: each program writes its fp32 partial ([npg, B, H]) and the caller adds them, then rounds to bf16 (the same
  fp32 sum PSUM would make). The first build exchanged the gated-norm output through shared HBM and a core barrier
  instead, and that cost ~17 us of a ~55 us call (neuron-explorer, B = 1).
- The recurrent step runs per (row, head), with columns indexed head-major (j B + b).

Exactness target: the same operations and rounding points as the torch path (bf16 rounding where torch materialises a
bf16 tensor), so device results differ from XLA's only in fp32 summation order and the scalar engine's functions. Three
deliberate differences: x / sqrt(s) is x * rsqrt(s) (NKI 0.6 has no divide; the rsqrt takes one Newton step, as
kernels/mhc_pre.py); sigmoid and silu are 1 / (1 + exp(-x)) and x of that (the exp and rsqrt tables only: see
_sigmoid); and the projection sums its k-tiles in h-layout order.
"""

from __future__ import annotations

import os
import zlib

import torch

from . import kda_oob as KO

P = 128


def emulate(x, in_norm, wcat, conv_w, conv_pool, f_b, g_b, dt_bias, A_log, o_norm, out_w, rec_pool, slots,
            lb: float, rms_eps: float, norm: bool = True, qkv=None):
    """The half's arithmetic in torch (CPU, one rank, no all-reduce): (y [B, H] bf16 before the all-reduce, the conv pool
    and the rec pool after the step). wcat [1026, H] = cat(in_qkv, f_a, g_a, in_b); slots [2, B] int32: read rows (>= R:
    start from zero, the row's position 0) and write rows. Written as models/linear_attn.py's decode form. norm False: x is
    already normalised (the kernel's norm=0, as linear_attn.mix calls it). qkv [B, CD] bf16: use these as the
    projection's q, k, v columns instead of its own (a test pins the kernel's own projection output this way)."""
    from ..models.decoder import rms_norm
    from ..models.linear_attn import _l2norm_kda, recurrent_step

    bf, f32 = torch.bfloat16, torch.float32
    B, H = x.shape
    CD = conv_w.shape[0]
    nv, D = A_log.shape[0], o_norm.shape[0]
    R = rec_pool.shape[0]
    xn = rms_norm(x, in_norm, rms_eps) if norm else x
    proj = torch.nn.functional.linear(xn, wcat)  # bf16
    qkv_, fa, ga, bl = proj[:, :CD], proj[:, CD:CD + D], proj[:, CD + D:CD + 2 * D], proj[:, CD + 2 * D:]
    qkv = qkv_ if qkv is None else qkv.to(qkv_.dtype)
    rd, wr = slots[0].long(), slots[1].long()
    keep = (rd < R).view(B, 1, 1)
    prev = conv_pool[rd.clamp(max=R - 1)]
    prev = torch.where(keep, prev, torch.zeros_like(prev)).to(qkv.dtype)
    xe = torch.cat([prev, qkv.unsqueeze(1)], dim=1)  # [B, K, CD]
    K = conv_w.shape[1]
    wf = conv_w.float()
    yc = xe[:, 0].float() * wf[:, 0]
    for j in range(1, K):
        yc = yc + xe[:, j].float() * wf[:, j]
    conv_out = conv_pool.clone()
    for b in range(B):
        if wr[b] < R:
            conv_out[wr[b]] = xe[b, 1:].to(conv_pool.dtype)
    y = torch.nn.functional.silu(yc.to(bf))
    q = y[:, :nv * D].reshape(B, nv, D)
    k = y[:, nv * D:2 * nv * D].reshape(B, nv, D)
    v = y[:, 2 * nv * D:].reshape(B, nv, D)
    f = torch.nn.functional.linear(fa, f_b).float() + dt_bias
    f = f.reshape(B, nv, D)
    rate = torch.exp(A_log).view(1, nv, 1)
    g = lb * torch.sigmoid(rate * f)
    beta = torch.sigmoid(bl).float()
    q, k = _l2norm_kda(q.float()), _l2norm_kda(k.float())
    z = torch.nn.functional.linear(ga, g_b).reshape(B, nv, D)
    q = q * D ** -0.5
    S = torch.zeros(B, nv, D, D, dtype=f32)
    S[keep.view(B)] = rec_pool[rd[keep.view(B)]].float()
    o, S2 = recurrent_step(q, k, v.float(), g, beta, S)
    rec_out = rec_pool.clone()
    for b in range(B):
        if wr[b] < R:
            rec_out[wr[b]] = S2[b]
    of = o.to(bf).float()
    of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + rms_eps)
    of = o_norm * of
    og = (of * torch.sigmoid(z.float())).to(bf).reshape(B, nv * D)
    return torch.nn.functional.linear(og, out_w), conv_out, rec_out


def arrange(wcat, conv_w, f_b, g_b, dt_bias, out_w, nv: int, npg: int) -> dict:
    """The kernel's per-program layouts of the half's weights (once, at load). Program p owns heads p hp .. (p + 1) hp - 1
    (hp = nv / npg):
      wp    [npg, H, NCp] bf16: its projection columns, NCp = 3 hp 128 + 256 + hp: q, k, v of its heads, f_a, g_a, b;
      convp [npg, 128, 3 hp, K]: its conv channels as tiles (q, k, v of its heads), a channel per partition;
      fbT / gbT [npg, 128, hp 128]: f_b / g_b rows of its heads, transposed (the contraction dim on the partitions);
      dtb   [npg, 128, hp] fp32: dt_bias of its heads;
      outT  [128, nv, H]: out^T by head (the contraction dim on the partitions); each program reads its column half."""
    NC, H = wcat.shape
    CD, K = conv_w.shape
    D = P
    hp = nv // npg
    if nv % npg or CD != 3 * nv * D or NC != CD + 2 * D + nv:
        raise ValueError(f"kda_half.arrange: nv {nv}, {npg} programs, wcat {tuple(wcat.shape)}, conv {tuple(conv_w.shape)}")
    wp, convp, fbT, gbT, dtb = [], [], [], [], []
    for p in range(npg):
        heads = range(p * hp, (p + 1) * hp)
        qkv = [i * nv * D + h * D + d for i in range(3) for h in heads for d in range(D)]
        hr = [h * D + d for h in heads for d in range(D)]
        rows = qkv + list(range(CD, CD + 2 * D)) + [CD + 2 * D + h for h in heads]
        wp.append(wcat[rows].t())
        convp.append(conv_w[qkv].reshape(3 * hp, D, K).permute(1, 0, 2))
        fbT.append(f_b[hr].t())
        gbT.append(g_b[hr].t())
        dtb.append(dt_bias[hr].float().reshape(hp, D).t())
    outT = out_w.t().reshape(nv, D, H).permute(1, 0, 2)
    st = lambda xs: torch.stack(xs).contiguous()  # noqa: E731
    return dict(wp=st(wp), convp=st(convp), fbT=st(fbT), gbT=st(gbT), dtb=st(dtb), outT=outT.contiguous())


try:  # the Neuron venv
    import nki
    import nki.isa as nisa
    import nki.language as nl

except ImportError:
    nki = None


PD = 2  # rows whose states are loaded ahead of the row being computed
HR = 4  # (row, head) pairs in flight: their PSUM and partition-0 row tiles (HR > SKEW)
SKEW = 2  # pairs between a pair's stage A and its stage B


def ring_rows(hp: int) -> int:
    """State ring entries the recurrent pipeline needs at hp heads per row: the row being loaded (PD ahead), the rows
    between, and the row whose stage B still reads its entry, SKEW pairs = ceil(SKEW / hp) rows back. A fixed
    4 is exact at 2 heads per rank; at one head per program (grid 2, nv 2) it is 5: with 4, row b + 2's load overwrote
    row b - 2's states before its stage B read them (the simulator, B = 8: state rows off by up to 0.46)."""
    return PD + (SKEW + hp - 1) // hp + 1
JQ = 4  # weight DMAs per program (W / JQ k-tiles each), each its own tensor so the projection starts on the first

if nki is not None:
    F32, BF16, I32 = nl.float32, nl.bfloat16, nl.int32

    def _sb(shape, dt=None):
        return nl.ndarray(shape, dtype=dt or F32, buffer=nl.sbuf)

    def _ps(shape):
        return nl.ndarray(shape, dtype=F32, buffer=nl.psum)

    def _dma(dst, src):
        """A DMA with every address static: descriptors made by the runtime before the kernel runs (dge_mode none, as
        kernels/dsa_long_pipe.py). Left to the compiler (dge_mode unknown), ~98% of this kernel's bytes went through the
        GpSimd engine's software DGE: 2,464 packets, ~34 us of DMA in a ~76 us call (neuron-explorer, B = 1, trn2)."""
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.none)

    def _sigmoid(dst, src, e):
        """dst = sigmoid(src) = 1 / (1 + exp(-src)): the scalar engine's exp, then + 1 and the reciprocal on the vector
        engine (e: an fp32 scratch of src's shape). So the kernel's activations use two tables, exp and rsqrt: each
        switch costs ~1.3 us on the scalar engine, and with sigmoid and silu as functions of their own neuron-explorer
        counted 7 per core per call (trn2)."""
        nisa.activation(dst=e, op=nl.exp, data=src, scale=-1.0)
        nisa.tensor_scalar(dst=e, data=e, op0=nl.add, operand0=1.0, engine=nisa.vector_engine)
        nisa.reciprocal(dst=dst, data=e)

    def _rsqrt_bcast(X, sq, n: int, scale: float, add: float):
        """rsqrt(scale (the sum over partitions of sq [128, n] fp32) + add) on every partition, [128, n] fp32: a ones-column
        matmul for the sum, the scalar engine's rsqrt and one Newton step on the vector engine (kernels/mhc_pre.py: its
        rsqrt alone is ~1e-5 off), a ones-row matmul (K = 1) for the broadcast."""
        ps = _ps((1, n))
        nisa.nc_matmul(dst=ps, stationary=X["ones_c"], moving=sq, accumulate=False)
        a = _sb((1, n))
        nisa.tensor_scalar(dst=a, data=ps, op0=nl.multiply, operand0=scale, op1=nl.add, operand1=add,
                           engine=nisa.vector_engine)
        r0 = _sb((1, n))
        nisa.activation(dst=r0, op=nl.rsqrt, data=a)
        nt = _sb((1, n))
        nisa.tensor_tensor(dst=nt, data1=r0, data2=r0, op=nl.multiply, engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=nt, data1=nt, data2=a, op=nl.multiply, engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=nt, data=nt, op0=nl.multiply, operand0=-0.5, op1=nl.add, operand1=1.5,
                           engine=nisa.vector_engine)
        r = _sb((1, n))
        nisa.tensor_tensor(dst=r, data1=r0, data2=nt, op=nl.multiply, engine=nisa.vector_engine)
        pb = _ps((128, n))
        nisa.nc_matmul(dst=pb, stationary=X["ones_r"], moving=r, accumulate=False)
        rb = _sb((128, n))
        nisa.tensor_copy(dst=rb, src=pb, engine=nisa.vector_engine)
        return rb

    def _load(X, b):
        """Row b's states of this program's heads into its ring entry; a read row >= R skips the DMA (zero state)."""
        B, hp, D = X["B"], X["hp"], X["D"]
        S = X["Sr"][b % X["ring"]]
        nisa.memset(dst=S, value=0.0, engine=nisa.gpsimd_engine)
        ridx = X["sl"].ap(pattern=[[2 * B, 1], [1, 1]], offset=b)
        nisa.dma_copy(dst=S, src=X["pool"].ap(pattern=[[D, 128], [D * D, hp], [1, D]], offset=X["h0"] * D * D,
                                              scalar_offset=ridx, indirect_dim=0), oob_mode=nisa.oob_mode.skip,
                          dge_mode=KO.HW)
        if X["nk"] is not None and KO.RCLAMP == 1:  # RCLAMP
            KO.zero(S, X["nk"], 128, hp * D, b, B)  # RCLAMP

    def _stage_a(X, c):
        """Pair c = b hp + j (column j B + b): r0, r1 - v and k's row on partition 0, then delta and o."""
        B, hp = X["B"], X["hp"]
        b, j = c // hp, c % hp
        col = j * B + b
        S, orow = X["Sr"][b % X["ring"]], X["Or"][b % X["ring"]]
        pr, dl, kr = X["Pr"][c % HR], X["Dl"][c % HR], X["Kr"][c % HR]
        nisa.nc_matmul(dst=pr[:, 0, :], stationary=X["QE"][:, col:col + 1], moving=S[:, j, :], accumulate=False)
        nisa.nc_matmul(dst=pr[:, 1, :], stationary=X["KE"][:, col:col + 1], moving=S[:, j, :], accumulate=False)
        nisa.nc_matmul(dst=pr[:, 1, :], stationary=X["NVT"][:, col:col + 1], moving=X["I"], accumulate=True)
        nisa.nc_matmul(dst=pr[:, 2, :], stationary=X["KT"][:, col:col + 1], moving=X["I"], accumulate=False)
        # delta = beta (v - r1) = -beta (r1 - v); o = r0 + (q . k) delta
        nisa.activation(dst=dl, op=nl.copy, data=pr[:, 1, :], scale=X["NBR"][:, col:col + 1])
        nisa.activation(dst=kr, op=nl.copy, data=pr[:, 2, :])
        nisa.scalar_tensor_tensor(dst=orow[:, j, :], data=dl, op0=nl.multiply, operand0=X["QK"][:, col:col + 1],
                                  op1=nl.add, operand1=pr[:, 0, :])

    def _stage_b(X, c):
        """Pair c: S' = diag(e) S + k delta^T; after a row's last head, its o rows go to columns (a K = 1 matmul each)
        and its states to the pool."""
        B, hp, D = X["B"], X["hp"], X["D"]
        b, j = c // hp, c % hp
        S, Sx, orow = X["Sr"][b % X["ring"]], X["Sn"][b % X["ring"]], X["Or"][b % X["ring"]]
        po = X["Po"][c % HR]
        nisa.nc_matmul(dst=po, stationary=X["Kr"][c % HR], moving=X["Dl"][c % HR], accumulate=False)
        nisa.scalar_tensor_tensor(dst=Sx[:, j, :], data=S[:, j, :], op0=nl.multiply,
                                  operand0=X["E"][:, j * B + b:j * B + b + 1], op1=nl.add, operand1=po)
        if j == hp - 1:
            for jj in range(hp):
                nisa.nc_matmul(dst=X["POT"][:, jj * B + b:jj * B + b + 1], stationary=orow[:, jj, :], moving=X["one"],
                               accumulate=False)
            widx = X["sl"].ap(pattern=[[2 * B, 1], [1, 1]], offset=B + b)
            nisa.dma_copy(dst=X["pool"].ap(pattern=[[D, 128], [D * D, hp], [1, D]], offset=X["h0"] * D * D,
                                           scalar_offset=widx, indirect_dim=0), src=Sx, oob_mode=nisa.oob_mode.skip,
                          dge_mode=KO.HW)

# -- end of the in-kernel helpers


def _kernel_rev() -> int:
    """CRC-32 of this module's in-kernel helpers, which kda_shell's kernel traces (it folds this REV into its own): the
    compile-cache key does not include NKI kernel source."""
    src = open(os.path.abspath(__file__)).read()
    a = src.index("try:  # the Neuron venv")
    return zlib.crc32(src[a:src.index("# -- end of the in-kernel helpers", a)].encode())


REV = _kernel_rev()


# -- the layer's side ---------------------------------------------------------------------------------------------------

def half_programs() -> int:
    """The program count the layer's arranged weights are laid out for: the runtime's NKI grid, or (a meta build before
    the runtime is configured) the target family's LNC."""
    from .. import platform

    try:
        return platform.nki_grid()
    except RuntimeError:
        t = platform.target()
        return platform.lnc(platform.family_of(t)) if t is not None else 1


def takes_spec(spec, H: int, nk: int, nv: int, npg: int) -> bool:
    """Whether the shell runs this KDA layer's decode half: per-key gates with a lower bound, 128-wide heads, a gate of
    rank 128 (not g_full), conv kernel 4, k heads = v heads, nv split evenly over the programs, H a multiple of 512 per
    program."""
    return (spec.kind == "kda" and spec.lower_bound is not None and spec.head_k_dim == P and spec.head_v_dim == P
            and spec.gate_rank == P and spec.conv_kernel == 4 and nk == nv and nv % npg == 0
            and H % (P * JQ) == 0 and (H // npg) % 512 == 0)


def arranged_shapes(H: int, nv: int, K: int, npg: int, dtype) -> dict:
    """{name: (shape, dtype)} of arrange()'s tensors."""
    hp = nv // npg
    return dict(wp=((npg, H, 3 * hp * P + 2 * P + hp), dtype), convp=((npg, P, 3 * hp, K), dtype),
                fbT=((npg, P, hp * P), dtype), gbT=((npg, P, hp * P), dtype), dtb=((npg, P, hp), torch.float32),
                outT=((P, nv, H), dtype))
