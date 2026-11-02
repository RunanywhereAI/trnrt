# Decode megakernels

GLM-5.3-Flash's decode step on Trainium2 runs almost entirely as fused NKI kernels, about one per half-layer instead
of dozens of XLA ops and small kernel calls per layer. This page covers which kernels a step runs, what each one
fuses, the contract that lets them be fused one half at a time, and the rules every kernel here follows. The code is
in [`trnrt/kernels/`](../trnrt/kernels); the model code that calls it is in [`trnrt/models/`](../trnrt/models).

Numbers carry the same tags as [RESULTS.md](RESULTS.md): **[M]** measured on trn2, **[D]** derived from measured
numbers or from the code, **[E]** an estimate.

## The model and the layout

GLM-5.3-Flash has 45 decoder layers:

- 34 Kimi Delta Attention (KDA) layers: a linear-attention token mixer with a fixed-size recurrent state per head;
- 11 DeepSeek Sparse Attention (DSA) layers, every fourth from layer 3: NoPE multi-head latent attention whose keys
  are chosen per query by a pooled lightning indexer;
- a 4-stream residual mixed by manifold-constrained hyper-connections (mHC) before and after every block;
- an FFN that is dense in layers 0-2 and, from layer 3 on, 288 routed experts (top 8, sigmoid routing) plus one
  shared expert, with the SwiGLU clamped at 10.

One engine is tensor-parallel over 32 logical NeuronCores (8 Trainium2 chips) at LNC = 2. Each logical core is two
physical cores, so every kernel runs as **two NKI programs** (grid 2). The DSA caches are additionally
context-parallel over groups of 8 ranks: each rank holds 1/8 of every sequence's latent, indexer keys and pool keys.
A decode call carries 1, 2, 4, 6 or 8 rows (requests), one graph per row count, and the whole step is one compiled
graph. Under `--preset batch12` (README, Configuration) it also carries 12 rows, on the same kernels.

## Why fuse

A decode step at these batch sizes reads weights and moves little else, so its floor is set by bytes and
collectives. What it actually pays for is fixed costs, and those scale with the number of operations:

- **A minimal NKI call in a dependent chain costs ~6.3 µs [M]**, before it does anything.
- **XLA glue between kernels is not free.** At one decode row, a MoE FFN half cost ~198 µs in XLA [M], about 45% of
  the step across its 42 MoE layers. Of that, ~67 µs was XLA ops (the fp32 router, its top-k, the gather and
  normalisation, the shared expert with its in-graph FP8 dequant), ~14 µs layout permutes, and only ~14 µs the expert
  kernel itself.
- **The dense layers were worse per byte.** The FFN block of layers 0-2 was two kernel calls around a slot sum,
  rms_norm, the in-graph dequant of 4.7M FP8 weights, two dots and their transposes: 188-191 µs of XLA glue per step
  [M].
- **Every kernel boundary is a scheduling barrier.** Work that could hide under a collective or a weight stream
  cannot when it sits in another call.

So each half-layer, an attention half or an FFN half, becomes one kernel. That is a **decode shell**: it starts right
after one all-reduce and ends right before the next.

## What a decode step runs

Per layer, at every served row count (1-8) [D, from the model code]:

| Layer | Attention half | FFN half |
|---|---|---|
| KDA, MoE (31 layers) | `kda_shell` | the FFN-half kernel (`ffn_half_pb`; at an even row count `mk_kda_layer` runs its body with the glue after it, one call) |
| KDA, dense (layers 0-2) | `kda_shell` | `dense_half` (the XLA path and the mHC kernels at 2 rows, where that is faster) |
| DSA (11 layers) | `dsa_satwfc`: **one kernel for the whole layer**, from the indexer's selection to the FFN half's partials (below) | |

That is **79 fused kernel calls for the 45 layers**, with an all-reduce between consecutive calls (in XLA), plus the
sampler's graph tail [D]. A 12-row call (`--preset batch12`) runs the same table: there a program's rows pass the one
128-partition chunk `mk_kda_layer`'s glue takes, so its second kernel runs the same glue in chunks of rows
(`mk_ffn_b5c_kernel`; calls of 8 rows or fewer keep `mk_ffn_b5_kernel`).

### The KDA shell (`kda_shell.py`)

One call takes the residual streams to the two programs' partials of the KDA half's output projection:

    mix_in   S = mix(post_prev, comb_prev, round_bf16(Y_prev[0] + Y_prev[1]), S)   the previous FFN block's pending mix
    mhc_pre  x, post, comb = mhc_pre(S)                                              this block's hyper-connection
    norm     xn = rms_norm(x)                                                        with row isolation (below)
    KDA      q, k, v, the forget gate, beta, the output gate; the short conv; one gated delta-rule step on the
             [128 x 128] fp32 state per head; the gated RMSNorm; this program's partial of o_proj
    returns  S, post, comb, P [2, T, H] bf16, and the updated conv and recurrent state rows (in place)

- Everything stays in one layout, h = p W + j over 128 partitions. The mHC projection contracts in that same layout
  from a weight laid out for it at load, so there is no transpose and no exchange between the programs.
- Both programs compute the mix and mhc_pre on all of S, which their projections need anyway. Each program then owns
  one KDA head: 64 heads over 32 ranks is 2 per rank.
- The **mix-in** is what removes a kernel per layer. The FFN block of layer n leaves its output mix undone and hands
  `(post, comb, y)` to layer n + 1's shell, which applies it first (`hybrid.defers`, `DecoderForCausalLM._run_layers`).

### The FFN half (`ffn_half_pb.py`, host side `ffn_half.py`)

    mix_in   S = the attention output's mix of S
    front    x, post of S, and the Sinkhorn's input (mHC's front)
    norm     xn = post_norm * (x r) in mHC's layout, then row isolation
    MoE      the router (sigmoid, selection bias, top 8 by max8 / find_index8, ids clamped), the shared expert, and the
             routed (token, expert) pairs on the experts' packed FP8 tiles
    returns  S, post, comb, and per-program fp32 partials of the routed and shared outputs

- mHC's Sinkhorn (20 rounds) is a long serial vector chain. Issued as one block after the MoE body, the scheduler kept
  it there. Here it is cut into pieces placed at static hooks between the MoE body's stages: after the router's
  matmuls, after the shared expert's gate/up, and after each routed group's down projection. That way it fills the
  gaps the matmuls leave.
- The two programs split the routed pair groups and the shared expert's rows. The router runs on both, because
  splitting it would need an exchange.
- `dense_half.py` is the same kernel with a dense SwiGLU body in place of the MoE body. Its front is the FFN half's
  statement for statement, and a test pins it to that source.

### The fused DSA layer (`dsa_satwfc.py`, `dsa_satwf.py`)

The DSA layer's projections (q, the latent, the indexer query and weights) run in XLA. From there, **one kernel**
does the rest of the layer:

1. **Select** (`dsa_shell`'s sel2): score this rank's 1/8 of the context's pools with the indexer and keep the top
   pools, then compact the selection.
2. **Merge across the context-parallel group:** an all-reduce, inside the kernel, of every rank's zero-padded list
   over the 8 ranks of its group.
3. **Attend** over the selected pools in this rank's latent shard, with the over-capacity fallback folded in.
4. **Hand off to the heads' owners:** one all_to_all sends each rank the partial outputs and log-sum-exps of its own
   query heads from the 8 ranks of its group.
5. **Combine** with a log-sum-exp per head, then W_UV and o_proj into this rank's partial (`cc_tail.py`).
6. **World all-reduce** of the attention output, inside the kernel (`cc_world.py`). The FFN half's prologue work that
   reads neither the send buffer nor the result is placed under it.
7. **The FFN half's body**, the same function the FFN-half kernel traces (`ffn_half_shared.py`).
8. **The attention block's Sinkhorn**, issued in three pieces after the three collectives' issue points so it runs
   under their waits. Its only reader is step 7's mix-in. At four rows it ran ~15.5 µs alone [M].

One kernel per DSA layer means three collectives (the list merge, the all_to_all and the world sum) with the
selection, attention, combine, projections and FFN half between and around them, and no kernel boundary at all.

## The contract: all-reduce to all-reduce

Every shell keeps the same boundary, so shells could land one at a time and each half could be gated on its own:

- **In:** the streams S and the previous block's pending mix `(post, comb, Y)`, where Y is the two programs'
  all-reduced partials.
- **Out:** S, this block's `(post, comb)`, and each program's bf16 partial of the half's output projection `[2, T, H]`.
  The caller all-reduces it as one collective, and the next call adds the two programs' halves in its prologue
  (fp32, rounded once).
- **The boundary tensors stay inside the step graph:** device tensors of one NEFF, never NEFF outputs. Only the state
  pools are outputs, aliased in place.
- **The first and last blocks:** the first block has no pending mix. Beside a half that is not a shell, the caller
  applies the pending mix the plain way.
- The fused DSA layer and the KDA layer's split form (`mk_kda_layer.py`) are the same contract spanning more than one
  half. The boundary is what crosses it, not one call per half.

## Rules every kernel here follows

**Weights.** Each shell's weights are laid out at load by an `arrange()` step, one packed blob per dtype per layer
(`shell.register_packed`). Tensor offsets are static integer arguments, so a call has a handful of inputs instead
of dozens; every graph input costs launch time on the host. The blobs are decode-only: the prefill graphs never see
them (`layer._decode_only`).

**Two programs.**
- One NKI program streams ~330 GB/s of weights on trn2, against XLA's ~545 over both cores [M]. A kernel that
  duplicates its weight stream in both programs therefore caps itself at one core.
- Split by heads, experts or column tiles. Duplicate only what is replicated anyway (the router, mHC).
- Avoid exchanging data between the programs mid-kernel when a partial will do. A gated-norm exchange in an early
  KDA build cost ~17 µs of a ~55 µs call [M].
- Allocate every buffer before any branch on the program id, so that it sits at the same address in both traces.

**DMA.**
- Every static DMA uses static descriptors (`dge_mode none`). Row gathers by slot use the hardware DGE with
  out-of-bounds skip.
- Left to the compiler, 98% of one kernel's bytes went through the GpSimd engine's software DGE; fixing it took the
  call from 64 to 54.8 µs [M].

**Scalar engine.**
- An activation-table switch costs ~1.3 µs [M]. The kernels use only exp and rsqrt: sigmoid and SiLU go through exp
  and a vector reciprocal.
- `rsqrt` alone is ~1e-5 off, so it takes one Newton step on the vector engine.

**Collectives inside a kernel.**
- Both programs issue each collective; one program's result is the one consumed.
- On the device, a program cannot rely on reading a collective's result, or a shared buffer the other program wrote
  from SBUF, even after a core barrier. HBM-to-HBM writes followed by a barrier are reliable. So program 0 builds the
  send buffers and writes the outputs.
- Every send buffer is fenced with a core barrier after it is filled, and every destination before it is read.
  Unfenced, the sums came back wrong and differed from run to run.
- Where an in-kernel combine replaces XLA's reduce-scatter, it sums in XLA's pairwise tree order,
  ((0+1)+(2+3))+((4+5)+(6+7)), so the result is bit-equal.

**NaN containment.**
- A batch's rows meet on the contraction of the identity-matmul transposes, where one row's NaN would spread to every
  row (NaN × 0 = NaN). Padding rows and warmup inputs can hold NaN, so every kernel that transposes over rows
  zeroes non-finite values first (`rowiso.py`).
- Top-k ids that index a gather are sanitised: NaN and -inf become the dtype's finite minimum, and the ids are clamped
  into range. A device top-k over a NaN row can return out-of-range ids, and the gather at them faults.

**Numerics.**
- Round where torch materialises a bf16 tensor (a linear's output, a cast), so a kernel differs from the XLA path only
  in fp32 summation order and the scalar engine's functions.
- A recurrent state keeps a 1-ulp projection difference rather than washing it out, so recurrent kernels are checked
  with the projection pinned.

**Kernel source.**
- No `raise` in anything a kernel calls; asserts hold at trace time.
- Plain loops: no comprehensions and no dict methods in kernel bodies.
- Every kernel takes a `rev` argument, a CRC of its source and its helpers' (`shell.kernel_rev`). The compile-cache
  key hashes the graph, not NKI source, so without `rev` an edited kernel would silently reuse a stale binary. A
  module constant a kernel body reads is part of that source.

## How a kernel is checked

1. **`emulate()`:** the kernel's computation in torch, written as the model's own served form. It is tested against
   the model code with the real per-rank shapes, and a wrong-constant control must fail it.
2. **`nki.simulate`** against the emulation, at grid 1 and grid 2, at several row counts including at least 5 (a
   pipeline ring bug showed only at 8 rows). Grid 1 is compared with grid 2 bit for bit, and planted mutations must be
   caught.
3. **On one NeuronCore:** against the emulation, then priced in chains against the XLA path it replaces.
4. **In the step:** the compiled graphs, then the accuracy gate on outputs.

The simulator is necessary but not sufficient. It accepted forms the device computes wrongly: accumulating matmuls
into slices of one PSUM tile, unfenced in-kernel collectives, and cross-program reads of shared scratch. It also
accepted forms the compiler rejects. So every new form gets a device run checked by value, with a control.

This repository's own gate: the 130 graphs this code builds, and their kernel binaries, are compared with the
engine it was extracted from. They must match modulo names and `rev` values ([`tools/graph_equiv.py`](../tools/graph_equiv.py)).

## Prefill kernels

Prefill runs piecewise: a prep graph, one graph per run of 12 layers, and a post graph, at chunks of 1,024 / 2,048 /
4,096 tokens. The streams are sequence-parallel across the 32 ranks. The kernels:

| Kernel | What it does |
|---|---|
| `delta_rule.py` | KDA's chunked gated delta rule over a chunk (the per-chunk form of flash-linear-attention's chunk_kda); its 16-token sub-chunks stay finite in fp32 only because the gate's lower bound (-5) bounds every log decay |
| `short_conv.py` | KDA's causal short convolution with its conv state |
| `moe_prefill.py` | the routed experts as a grouped GEMM over the whole chunk, on the same packed FP8 tiles the decode kernels read |
| `dsa_topk.py` | exact top-k of the pooled indexer's scores, as one kernel instead of ~32-48 dependent XLA rounds |
| `dsa_long_pipe_s.py`, `dsa_long_select_t.py` | the long-context selection pipeline under context parallelism, its query tiles split over the two physical cores |
| `dsa_slots_n_s.py` | the context-parallel attention over the selected pools, rows split over the two physical cores |
| `mhc_pre.py`, `mhc_mix.py`, `mhc_fnj.py` | the hyper-connections around each block |

The split forms (`_s`) exist because at grid 2 a kernel that never reads its program id runs all of its work on both
physical cores, or on one while the other waits. In a served 4,096-token prefill call, `dsa_slots_n` alone took 51.6
of a 12-layer piece's 465 ms before its rows were split [M].
