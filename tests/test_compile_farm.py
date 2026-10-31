"""The compile pipeline (trnrt/compile/) on CPU: the neuronx-cc command and kernel-reference parsing the
compile step relies on, the compile step itself with a stand-in compiler, the shape directory (safetensors
headers only) that stands in for the weights, the load-time decisions read from real weight values, and,
where libtorch_neuronx_lite is installed, a capture of a tiny GLM-5.3-Flash at tp=2.

Tests of the decisions and the header checkpoint run the model loader (trnrt.models, trnrt.kernels) and skip
without it; the capture test needs libtorch_neuronx_lite (a Neuron SDK environment) and transformers with
glm5_next.
"""

import base64
import importlib.util
import json
import os
import pathlib
import struct
import sys
import types

import pytest
import torch

FP8 = torch.float8_e4m3fn
KEY = "66437b3432fb818ccebad2e5a8628757"
# An LNL command.txt for a trn2 graph at LNC 2.
COMMAND = (f"/usr/local/bin/neuronx-cc compile /cache/neuron/compile_cache/{KEY}/graph.hlo "
           f"--framework XLA --target trn2 --output /cache/neuron/compile_cache/{KEY}/graph_{KEY}.neff "
           f"--logfile /cache/neuron/compile_cache/{KEY}/log-neuron-cc.txt "
           "--logical-nc-config=2 -O2 --model-type=transformer "
           "'--internal-hlo2tensorizer-options=--experimental-unsafe-fp8e4m3fn-as-fp8e4m3'")
ARGS = ["--logical-nc-config=2", "-O2", "--model-type=transformer",
        "--internal-hlo2tensorizer-options=--experimental-unsafe-fp8e4m3fn-as-fp8e4m3"]


def needs(*modules):
    """Skip unless every module imports (the model and kernel packages, or the Neuron SDK)."""
    for m in modules:
        try:
            spec = importlib.util.find_spec(m)
        except ModuleNotFoundError:
            spec = None
        if spec is None:
            return pytest.mark.skip(reason=f"needs {m}")
    return pytest.mark.skipif(False, reason="")


MODELS = needs("trnrt.models.loader", "trnrt.models.decoder", "trnrt.kernels.moe_prefill", "trnrt.kernels.moe_dedupe")


# -- the compile step -------------------------------------------------------------------------------

def test_parse_command_keeps_only_the_compiler_args():
    from trnrt.compile.farm import compiler_command, parse_command

    target, args = parse_command(COMMAND)
    assert target == "trn2"
    assert args == ARGS
    cmd = compiler_command("neuronx-cc", "/c/k/graph.hlo", target, "/c/k/graph_k.neff", "/c/k/log", args)
    assert parse_command(" ".join(cmd)) == (target, args)
    with pytest.raises(ValueError):
        parse_command("neuronx-cc list-operators --framework XLA")
    with pytest.raises(ValueError, match="no --target"):
        parse_command("neuronx-cc compile g.hlo --framework XLA -O2")


def _hlo_with_kernels(path, cfgs):
    """Bytes shaped like a serialised HLO whose NKI custom calls carry base64 JSON backend configs."""
    blob = b"\x08\x01\x12\x07HloModule\x1a"
    for c in cfgs:
        blob += b"\x22\x9a" + base64.b64encode(json.dumps(c).encode()) + b"\x00\x18AwsNeuronCustomNativeKernel"
    path.write_bytes(blob)


def test_kernel_refs_reads_backend_configs_inside_binary_hlo(tmp_path):
    from trnrt.compile.farm import kernel_refs, missing_kernels

    present = tmp_path / "kernel_a.colz"
    present.write_bytes(b"x")
    hlo = tmp_path / "graph.hlo"
    _hlo_with_kernels(hlo, [{"func_name": "trnrt.kernels.moe_dedupe.moe_dedupe", "kernel_format": "colz",
                             "klir_binary": {"binary": str(present)}},
                            {"func_name": "k2", "klir_binary": {"binary": "/var/tmp/nki-intermediate-cache/none/k2.colz"}},
                            {"func_name": "no binary here"}])
    assert kernel_refs(str(hlo)) == {str(present), "/var/tmp/nki-intermediate-cache/none/k2.colz"}
    assert missing_kernels(str(hlo)) == ["/var/tmp/nki-intermediate-cache/none/k2.colz"]


def test_entry_args_reads_command_txt(tmp_path):
    from trnrt.compile.farm import entry_args

    (tmp_path / "command.txt").write_text(COMMAND)
    assert entry_args(str(tmp_path)) == ("trn2", ARGS)


# A stand-in for neuronx-cc: checks the command line it gets, writes a NEFF to --output and a log to
# --logfile, or fails as told by the HLO's first bytes.
FAKE_CC = r'''
import sys
a = sys.argv[1:]
assert a[0] == "compile" and a[2:6] == ["--framework", "XLA", "--target", "trn2"], a
out, log = a[a.index("--output") + 1], a[a.index("--logfile") + 1]
hlo = open(a[1], "rb").read()
open(log, "w").write("log\n")
if hlo.startswith(b"FAIL"):
    sys.stderr.write("[NCC_TEST] made to fail\n")
    sys.exit(70)
open(out, "wb").write(b"NEFF" + hlo + " ".join(a[6 + 4:]).encode())
'''


def _fake_cc(tmp_path):
    p = tmp_path / "neuronx-cc"
    p.write_text(f"#!{sys.executable}\n{FAKE_CC}")
    p.chmod(0o755)
    return str(p)


def _captured(root, key, hlo=b"hlo", command=None):
    """A captured entry, as capture.capture_graph leaves it: graph.hlo, the metadata, command.txt."""
    d = root / key
    d.mkdir(parents=True)
    (d / "graph.hlo").write_bytes(hlo)
    (d / ".artifact_metadata_v0.json").write_text(json.dumps({"cache_key": key}))
    (d / "command.txt").write_text(command or COMMAND.replace(f"/cache/neuron/compile_cache/{KEY}", str(d))
                                   .replace(KEY, key))
    return d


def test_run_compile_completes_the_entry_and_writes_the_marker_last(tmp_path):
    from trnrt.compile import cache
    from trnrt.compile.farm import run_compile

    cc = _fake_cc(tmp_path)
    d = _captured(tmp_path / "cc", "a" * 32, b"hlo-a")
    assert not cache.is_complete(str(d))
    r = run_compile(str(d), neuronx_cc=cc, work_root=str(tmp_path))
    assert r.ok and r.returncode == 0 and r.args == ARGS, r
    neff = d / f"graph_{'a' * 32}.neff"
    assert neff.read_bytes().startswith(b"NEFFhlo-a") and r.neff_bytes == neff.stat().st_size
    assert cache.read_marker(str(d))["neff_size"] == str(neff.stat().st_size)
    assert (d / ".compilation_complete").stat().st_mtime_ns >= neff.stat().st_mtime_ns
    assert cache.is_complete(str(d))
    assert (d / "command.txt").read_text().split()[0] == cc  # the command that built it
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("ncc-")]  # scratch removed on success

    bad = _captured(tmp_path / "cc", "b" * 32, b"FAIL")
    r = run_compile(str(bad), neuronx_cc=cc, work_root=str(tmp_path))
    assert not r.ok and r.returncode == 70 and "NCC_TEST" in r.error
    assert not (bad / ".compilation_complete").exists() and not cache.is_complete(str(bad))


def test_run_compile_refuses_an_hlo_whose_kernel_binaries_are_missing(tmp_path):
    from trnrt.compile.farm import run_compile

    d = _captured(tmp_path / "cc", "c" * 32)
    _hlo_with_kernels(d / "graph.hlo", [{"func_name": "k", "klir_binary": {"binary": str(tmp_path / "gone.colz")}}])
    r = run_compile(str(d), neuronx_cc=_fake_cc(tmp_path))
    assert not r.ok and "NKI kernel binaries missing" in r.error
    assert not (d / ".compilation_complete").exists()


def test_entries_skip_a_capture_lowering_directory(tmp_path):
    """A capture lowers each graph in <key>.capture.<pid> and renames it into place: one a killed capture left
    behind holds graph.hlo and no command.txt, and is no entry."""
    from trnrt.compile import cache

    d = _captured(tmp_path, "a" * 32)
    left = tmp_path / f"{'b' * 32}.capture.4242"
    left.mkdir()
    (left / "graph.hlo").write_bytes(b"hlo")
    assert cache.entries(str(tmp_path)) == cache.pending(str(tmp_path)) == [str(d)]


def test_compile_command_line_compiles_the_pending_entries(tmp_path, capsys):
    from trnrt.compile import cache, farm

    cc = _fake_cc(tmp_path)
    root = tmp_path / "out" / "cache" / "neuron" / "compile_cache"
    keys = [c * 32 for c in "def"]
    for k in keys:
        _captured(root, k, f"hlo-{k}".encode())
    (tmp_path / "keys.json").write_text(json.dumps({k: [0, 1] for k in keys[:2]}))
    assert farm.main(["compile", "--out", str(tmp_path / "out"), "--neuronx-cc", cc, "--workers", "2",
                      "--keys-file", str(tmp_path / "keys.json")]) == 0
    assert [cache.is_complete(str(root / k)) for k in keys] == [True, True, False]
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["entries"] == 2 and summary["failed"] == 0
    assert farm.main(["compile", "--out", str(tmp_path / "out"), "--neuronx-cc", cc]) == 0  # the one left
    assert cache.pending(str(root)) == []


def test_an_incomplete_entry_fails_alone_run_after_run(tmp_path, capsys):
    """A capture killed between graph.hlo and command.txt leaves an entry with no command, or a cut one: each run
    reports it and goes on with the others, and exits nonzero. A killed capture's lowering directory is no entry."""
    from trnrt.compile import cache, farm

    cc = _fake_cc(tmp_path)
    out = tmp_path / "out"
    root = out / "cache" / "neuron" / "compile_cache"
    good = _captured(root, "a" * 32)
    (_captured(root, "b" * 32) / "command.txt").unlink()
    _captured(root, "c" * 32, command="/usr/local/bin/neuronx-cc compile")
    left = root / f"{'d' * 32}.capture.4242"
    left.mkdir()
    (left / "graph.hlo").write_bytes(b"hlo")
    for entries in (3, 2):  # the second run finds the good entry compiled
        assert farm.main(["compile", "--out", str(out), "--neuronx-cc", cc]) == 1
        rows = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
        bad = {r["key"]: r["error"] for r in rows if "key" in r and not r["ok"]}
        assert sorted(bad) == ["b" * 32, "c" * 32]
        assert all("run capture again" in e for e in bad.values()), bad
        assert (rows[-1]["entries"], rows[-1]["failed"]) == (entries, 2)
        assert cache.is_complete(str(good))
    assert not list(root.rglob("command.txt.tmp-*"))


def test_oom_killed_reads_neuronx_cc_own_report():
    """neuronx-cc reports a subprocess the OOM killer took as exit 70 with [F137]; any other exit 70 is a
    compile error."""
    from trnrt.compile.farm import Result, oom_killed

    f137 = "[F137] neuronx-cc was forcibly killed - This most commonly occurs due to insufficient system memory."
    assert oom_killed(Result("k", False, 1.0, 70, error=f137))
    assert oom_killed(Result("k", False, 1.0, -9))
    assert not oom_killed(Result("k", False, 1.0, 70, error="[NCC_EBVF030] Instructions generated exceeds"))


# -- the shape directory ----------------------------------------------------------------------------

def _st_file(path, tensors: dict, metadata=None):
    """A safetensors file written by hand (8-byte header length, JSON header, data), so the test needs
    no particular safetensors version for float8."""
    header, data = {}, b""
    for name, t in tensors.items():
        raw = t.contiguous().view(torch.uint8).numpy().tobytes() if t.numel() else b""
        dt = {torch.bfloat16: "BF16", torch.float32: "F32", FP8: "F8_E4M3", torch.int64: "I64"}[t.dtype]
        header[name] = {"dtype": dt, "shape": list(t.shape), "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    if metadata:
        header["__metadata__"] = metadata
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + data)


def test_shape_dir_holds_every_header_and_the_json_files(tmp_path):
    from trnrt.compile import capture

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _st_file(ckpt / "model-00001-of-00002.safetensors",
             {"model.embed_tokens.weight": torch.zeros(8, 4, dtype=torch.bfloat16),
              "model.layers.0.mlp.experts.0.down_proj.weight": torch.zeros(4, 2, dtype=FP8)}, {"format": "pt"})
    _st_file(ckpt / "model-00002-of-00002.safetensors",
             {"model.layers.0.mlp.experts.0.down_proj.weight_scale_inv": torch.ones(1, 1),
              "model.norm.weight": torch.ones(4, dtype=torch.bfloat16)})
    for fn in ("config.json", "generation_config.json", "tokenizer_config.json", "model.safetensors.index.json"):
        (ckpt / fn).write_text("{}")
    (ckpt / "README.md").write_text("x")
    shapes = tmp_path / "shapes"
    capture.write_shape_dir(str(ckpt), str(shapes))
    assert sorted(os.listdir(shapes)) == sorted(["config.json", "generation_config.json", "tokenizer_config.json",
                                                 capture.HEADERS])
    index = json.loads((shapes / capture.HEADERS).read_text())
    assert index == capture.header_index(str(ckpt)) == {
        "model.embed_tokens.weight": ["BF16", [8, 4]],
        "model.layers.0.mlp.experts.0.down_proj.weight": ["F8_E4M3", [4, 2]],
        "model.layers.0.mlp.experts.0.down_proj.weight_scale_inv": ["F32", [1, 1]],
        "model.norm.weight": ["BF16", [4]]}
    assert capture.moe_prefixes(index) == ["model.layers.0."]
    with pytest.raises(FileNotFoundError):
        capture.header_index(str(shapes))


def test_header_slices_have_the_sliced_shape():
    from trnrt.compile.capture import _HeaderHandle

    h = _HeaderHandle({"w": ["BF16", [6, 4]], "s": ["F32", []], "q": ["F8_E4M3", [4, 256]]})
    assert h.get_slice("w")[slice(2, 5)].shape == (3, 4)
    assert h.get_slice("w")[slice(2, 5)].dtype == torch.bfloat16
    assert h.get_slice("w").get_dtype() == "BF16" and h.get_slice("w").get_shape() == [6, 4]
    assert h.get_tensor("s").shape == ()
    assert h.get_slice("q")[:, 128:].dtype == FP8 and h.get_slice("q")[:, 128:].shape == (4, 128)
    assert h.keys() == ["w", "s", "q"]


def test_moe_prefixes_reads_both_namings_and_sorts_by_layer():
    from trnrt.compile.capture import moe_prefixes

    index = {f"model.language_model.layers.{i}.mlp.experts.0.down_proj.weight": ["F8_E4M3", [1, 1]] for i in (12, 3)}
    index["model.layers.45.mlp.experts.0.down_proj.weight"] = ["F8_E4M3", [1, 1]]  # the MTP layer
    index["model.language_model.layers.3.mlp.experts.1.down_proj.weight"] = ["F8_E4M3", [1, 1]]
    index["model.language_model.layers.1.mlp.down_proj.weight"] = ["BF16", [1, 1]]  # a dense MLP
    assert moe_prefixes(index) == ["model.layers.3.", "model.layers.12.", "model.layers.45."]


# -- GLM-5.3-Flash-shaped fixtures ------------------------------------------------------------------

def glm_config(layers=4, experts=4, hidden=256, moe_intermediate=128, dense_mlps=3):
    """config.json of zai-org/GLM-5.3-Flash cut to `layers` layers, few heads and experts and small widths;
    everything else as published: KDA layers with a DSA (MLA) layer every fourth, the first `dense_mlps` MLPs
    dense, sigmoid routing with a correction bias and one shared expert, mHC, FP8 in 128 x 128 blocks."""
    types_ = ["linear_attention" if i % 4 != 3 else "deepseek_sparse_attention" for i in range(layers)]
    t = dict(attention_bias=False, eos_token_id=[1], pad_token_id=0, first_k_dense_replace=dense_mlps, hc_eps=1e-6,
             hc_mult=4, hc_sinkhorn_iters=20, head_dim=0, hidden_act="silu", hidden_size=hidden, index_head_dim=128,
             index_kpool=4, index_kpool_always_select_tail=True, index_kpool_compress=True, index_n_heads=32,
             index_topk=2048, index_share_for_mtp_iteration=True, indexer_rope_interleave=True, intermediate_size=256,
             kv_lora_rank=512, linear_attn_config=dict(
                 num_heads=4, gate_lower_bound=-5.0, head_dim=128, short_conv_kernel_size=4,
                 kda_layers=[i for i, x in enumerate(types_) if x == "linear_attention"],
                 full_attn_layers=[i for i, x in enumerate(types_) if x != "linear_attention"]),
             layer_types=types_, mlp_layer_types=["dense"] * min(dense_mlps, layers) + ["sparse"] * (layers - dense_mlps),
             indexer_types=["full"] * layers, max_position_embeddings=1024, mhc=True, mla_use_nope=True,
             model_type="glm5_next_text", moe_intermediate_size=moe_intermediate, moe_router_dtype="float32",
             n_group=1, n_routed_experts=experts, n_shared_experts=1, norm_topk_prob=True, num_attention_heads=4,
             num_experts_per_tok=2, num_hidden_layers=layers, num_key_value_heads=4, num_nextn_predict_layers=0,
             q_lora_rank=1536, qk_head_dim=256, qk_nope_head_dim=256, qk_rope_head_dim=0, rms_norm_eps=1e-5,
             routed_scaling_factor=2.5, scoring_func="sigmoid", swiglu_limit=10.0, tie_word_embeddings=False,
             topk_group=1, topk_method="noaux_tc", v_head_dim=256, vocab_size=384)
    return {"architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next", "text_config": t,
            "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                                    "weight_block_size": [128, 128]}}


LAYER = 3  # the first MoE layer of glm_config
PRE = f"model.layers.{LAYER}."


def glm_expert_checkpoint(path, refit: bool, E=4, H=256, I=128):
    """config.json (glm_config) and layer LAYER's router and routed experts stored as zai-org/GLM-5.3-Flash
    stores them: under model.language_model., one tensor per expert and projection, FP8 e4m3fn with 128 x 128
    block scales (weight_scale_inv). refit: codes up to 448 (the checkpoint's range), so models/quant.fit_e4m3_max
    halves some rows for trn2's 240."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(glm_config(experts=E, hidden=H, moe_intermediate=I), f)
    g = torch.Generator().manual_seed(7)
    t, pre = {}, f"model.language_model.layers.{LAYER}.mlp."
    for e in range(E):
        for proj, (n, k) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            w = torch.randn(n, k, generator=g) * 0.02
            s = w.abs().view(n // 128, 128, k // 128, 128).amax((1, 3)) / (448.0 if refit else 200.0)
            t[f"{pre}experts.{e}.{proj}.weight"] = (w / s.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(FP8)
            t[f"{pre}experts.{e}.{proj}.weight_scale_inv"] = s.contiguous()
    t[pre + "gate.weight"] = torch.randn(E, H, generator=g).bfloat16()
    t[pre + "gate.e_score_correction_bias"] = torch.randn(E, generator=g)
    _st_file(pathlib.Path(path) / "model.safetensors", t)


def fp8_bytes(g, *s):  # finite e4m3 bytes (exponent bit 3 clear)
    return (torch.randint(0, 256, s, dtype=torch.uint8, generator=g) & 0xBF).view(FP8)


def experts(E=6, H=1024, seed=0):
    """Packed-layout experts with fp32 128 x 128 block scales as GLM-5.3-Flash loads them at tp=32 (gate and
    up rows each inside one row block, down one block of 64 input rows per 128 output columns)."""
    g = torch.Generator().manual_seed(seed)
    w_gu, w_down = fp8_bytes(g, E, 128, H), fp8_bytes(g, E, 64, H)
    s_gu = (torch.rand(E, 2, H // 128, generator=g) * 0.02 + 0.005).repeat_interleave(64, dim=1)
    s_down = (torch.rand(E, 1, H // 128, generator=g) * 0.02 + 0.005).repeat_interleave(128, dim=2)
    return w_gu, s_gu, w_down, s_down


def checkpoint_experts(E=6, H=1024, Im=64, seed=0):
    """The routed experts as the loader stores GLM-5.3-Flash's for one rank at tp=32: checkpoint weights
    quantized per 128 x 128 block to e4m3fn (largest value of a block 448, as the checkpoint's), the rank's
    Im = 64 rows of gate_proj and of up_proj and its 64 columns of down_proj each inside one block, scales
    expanded per row, and models/quant.fit_e4m3_max applied as the loader does (trn2's e4m3 stops at 240: each
    (row, block) whose largest code exceeds 240 is halved and its scale doubled). (w_gu, s_gu, w_down, s_down)
    in the model's layout."""
    from trnrt.models.quant import fit_e4m3_max

    g = torch.Generator().manual_seed(seed)

    def blocks(rows, cols, shard_rows, shard_cols):
        w = torch.randn(rows, cols, generator=g) * 0.02
        bs = w.abs().view(rows // 128, 128, cols // 128, 128).amax((1, 3)) / 448.0  # weight_scale_inv
        codes = (w / bs.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(FP8)
        cw = codes[shard_rows, shard_cols]
        rs = bs.repeat_interleave(128, 0)[shard_rows]  # [n, k / 128] per row
        if shard_cols.stop - shard_cols.start < 128:  # inside one block: that block's scale
            rs = rs[:, shard_cols.start // 128:shard_cols.start // 128 + 1]
        return fit_e4m3_max(cw, rs, 240.0)

    gus, gss, ds, dss = [], [], [], []
    for _ in range(E):
        wg, sg = blocks(128, H, slice(0, Im), slice(0, H))  # the rank's gate rows of a 2 x Im block
        wu, su = blocks(128, H, slice(Im, 2 * Im), slice(0, H))
        wd, sd = blocks(H, 128, slice(0, H), slice(0, Im))  # down [H, I]: the rank's Im columns
        gus.append(torch.cat([wg, wu]))
        gss.append(torch.cat([sg, su]))
        ds.append(wd.T.contiguous())
        dss.append(sd.T.contiguous())
    return torch.stack(gus), torch.stack(gss), torch.stack(ds), torch.stack(dss)


# -- the header checkpoint and the load-time decisions (need the model loader) ------------------------

@MODELS
def test_a_shape_dir_reads_like_the_checkpoint(tmp_path):
    """The loader's checkpoint over a shape directory answers every question about names, shapes and dtypes
    as it does over the checkpoint, slices included, under both namings of GLM-5.3-Flash's text model."""
    from trnrt.compile import capture
    from trnrt.models import loader

    ckpt, shapes = tmp_path / "ckpt", tmp_path / "shapes"
    glm_expert_checkpoint(str(ckpt), refit=True)
    capture.write_shape_dir(str(ckpt), str(shapes))
    capture.install_header_checkpoint()
    real, fake = loader._Checkpoint(str(ckpt)), loader._Checkpoint(str(shapes))
    assert set(real._where) == set(fake._where)
    for name in real._where:
        assert fake.shape(name) == real.shape(name), name
        a, b = real._where[name].get_slice(name), fake._where[name].get_slice(name)
        assert str(a.get_dtype()) == str(b.get_dtype()), name
        idx = tuple(slice(0, max(1, n // 2)) for n in a.get_shape())
        x, y = a[idx], b[idx]
        assert (tuple(x.shape), x.dtype) == (tuple(y.shape), y.dtype), name
        assert not y.any(), name  # zeros
    assert PRE + "mlp.experts.0.down_proj.weight" in fake  # read under model. as well


@MODELS
@pytest.mark.parametrize("fmt", ["glm", "glm-rows", "glm-down", "loaded"])
def test_decide_one_expert_at_a_time_matches_the_whole_blob(fmt):
    """capture.decide reads experts until both answers are known; they must be check_blob and
    `down_factors is not None` over every expert at once, as DecoderLayer.pack_experts asks them."""
    from trnrt.compile import capture
    from trnrt.kernels import moe_dedupe as mdd
    from trnrt.kernels import moe_prefill as mp

    ws = list(checkpoint_experts() if fmt == "loaded" else experts())
    if fmt == "glm-rows":  # one gate row of the last expert with a scale of its own
        ws[1] = ws[1].clone()
        ws[1][5, 3, 0] *= 2
    if fmt == "glm-down":  # one down column of the last expert at twice its chunk's scale
        ws[3] = ws[3].clone()
        ws[3][5, 0, 7] *= 2
    whole = mdd.pack(*ws)
    want = (mp.check_blob(whole, 1024), mp.down_factors(whole, 1024) is not None)
    got = capture.decide((mdd.pack(*(w[e:e + 1] for w in ws)) for e in range(6)), 1024)
    assert got[:2] == want
    assert got[2] == {"glm": 6, "glm-rows": 6, "glm-down": 6, "loaded": 1}[fmt]


@MODELS
@pytest.mark.parametrize("refit", [True, False])
def test_layer_decisions_match_the_loaded_layer(tmp_path, refit):
    """The decisions computed for a capture (one expert of the rank through the loader, every other expert
    zeros) are the ones the device's load reaches with every expert loaded: loader._load_experts on the real
    checkpoint, moe_dedupe.pack, check_blob and down_factors."""
    from trnrt.compile import capture
    from trnrt.config import ModelConfig
    from trnrt.kernels import moe_dedupe as mdd
    from trnrt.kernels import moe_prefill as mp
    from trnrt.models import loader

    glm_expert_checkpoint(str(tmp_path), refit)
    cfg = ModelConfig.from_pretrained(str(tmp_path))
    assert cfg.num_experts == 4 and cfg.quant_expert_block == 128 and cfg.router_bias
    got = capture._layer_decisions(str(tmp_path), None, PRE, 2, [0, 1], 240.0)
    for r in (0, 1):
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt)  # noqa: E731
        layer = types.SimpleNamespace(router=z(4, 256, dt=torch.bfloat16), down_t=True, router_bias=z(4),
                                      router_logit_bias=None, w_gu=z(4, 128, 256, dt=FP8), w_gu_scale=z(4, 128, 2),
                                      w_down=z(4, 64, 256, dt=FP8), w_down_scale=z(4, 1, 256))
        loader._load_experts(layer, loader._Checkpoint(str(tmp_path)), PRE, cfg, r, 2, torch.bfloat16, 240.0)
        blob = mdd.pack(layer.w_gu, layer.w_gu_scale, layer.w_down, layer.w_down_scale)
        d = got[capture.decision_key(240.0, 2, r, PRE)]
        assert (d["moe_prefill_dq"], d["moe_prefill_down"]) == (mp.check_blob(blob, 256),
                                                                 mp.down_factors(blob, 256) is not None)
        assert (d["moe_prefill_dq"], d["moe_prefill_down"]) == ((False, True) if refit else (True, False))
        assert d["experts_read"] == (1 if refit else 4)


@MODELS
def test_compute_decisions_writes_every_layer_and_rank(tmp_path):
    from trnrt.compile import capture

    ckpt, shapes = tmp_path / "ckpt", tmp_path / "shapes"
    glm_expert_checkpoint(str(ckpt), refit=True)
    capture.write_shape_dir(str(ckpt), str(shapes))
    rec = capture.compute_decisions(str(ckpt), str(shapes), 2, 240.0, procs=1)
    on_disk = capture.read_decisions(str(shapes))
    assert on_disk["decisions"] == rec["decisions"] and on_disk["model"] == str(ckpt)
    assert sorted(rec["decisions"]) == [capture.decision_key(240.0, 2, r, PRE) for r in (0, 1)]


@MODELS
def test_capture_replays_the_decisions_and_refuses_without_them(tmp_path, monkeypatch):
    """A layer packed from a shape directory (zeros) gets the recorded answers, and the tensors the real
    values give it (moe_prefill_dsc / moe_prefill_dfr: their presence, shapes and dtypes enter the graphs);
    an answer the record lacks raises."""
    from trnrt.compile import capture
    from trnrt.models import decoder

    monkeypatch.setattr(decoder, "MOE_PREFILL_KERNEL", "nki", raising=False)
    capture.install_header_checkpoint()
    names = ("w_gu", "w_gu_scale", "w_down", "w_down_scale")
    real = types.SimpleNamespace(pack_tiles=True, **dict(zip(names, checkpoint_experts())))
    decoder.DecoderLayer.pack_experts(real)  # no expert load from a shape directory: the real answers
    key = capture.decision_key(240.0, 32, 0, PRE)
    rec = {"decisions": {key: {"moe_prefill_dq": False, "moe_prefill_down": True}}, "path": "x"}

    def zeros_layer():
        return types.SimpleNamespace(pack_tiles=True, **{n: torch.zeros_like(t) for n, t in zip(
            names, checkpoint_experts())})

    fake = zeros_layer()
    capture._PENDING[0] = (fake, rec, key)
    decoder.DecoderLayer.pack_experts(fake)
    assert fake.moe_prefill_dq is real.moe_prefill_dq is False
    for n in ("moe_prefill_dsc", "moe_prefill_dfr"):
        a, b = getattr(fake, n), getattr(real, n)
        assert (tuple(a.shape), a.dtype) == (tuple(b.shape), b.dtype)
    other = zeros_layer()
    capture._PENDING[0] = (other, {"decisions": {}, "path": "x"}, key)
    with pytest.raises(RuntimeError, match="no moe_prefill_dq"):
        decoder.DecoderLayer.pack_experts(other)
    assert capture._ACTIVE[0] is None and capture._PENDING[0] is None


# -- the capture (needs libtorch_neuronx_lite) --------------------------------------------------------

def tiny_engine_config(argv):
    """engine_config for the capture test: the serve arguments are the model path alone."""
    from trnrt.config import EngineConfig

    return EngineConfig(model_path=argv[0], device="neuron", dtype=torch.bfloat16, tp=2, page_size=8,
                        kv_cache_dtype="fp8", kv_cache_gb=0.01, max_num_seqs=2, max_model_len=64, max_prefill_tokens=8,
                        decode_batch_buckets=(2,), prefill_token_buckets=(8,), page_buckets=(2,))


@needs("libtorch_neuronx_lite", "transformers.models.glm5_next", "trnrt.engine.model_runner", "trnrt.models.loader")
def test_capture_tp2_writes_one_entry_per_graph(tmp_path):
    """`compile capture` of a tiny GLM-5.3-Flash at tp=2 (fake process group, meta device): both ranks trace
    the same graphs, every entry carries what the compile step needs and none is compiled, and each rank's
    record lists its keys in the order its warmup met them."""
    import transformers as tf

    from trnrt.compile import capture, farm

    ckpt, out = tmp_path / "ckpt", tmp_path / "out"
    # every MLP dense: no routed experts, so no load-time decisions. At tp=2 the DSA caches are context-parallel:
    # hidden a multiple of 512, pages of kpool x cp = 8 tokens and an FP8 KV cache (tiny_engine_config), as served.
    c = glm_config(dense_mlps=4, hidden=512)
    c.pop("quantization_config")
    model = tf.Glm5NextForConditionalGeneration(tf.Glm5NextConfig(
        text_config=c["text_config"], vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                                                         out_hidden_size=256, projection_intermediate_size=64)))
    model.save_pretrained(str(ckpt), safe_serialization=True)
    assert farm.main(["shapes", str(ckpt), "--out", str(out)]) == 0
    assert farm.main(["capture", "--out", str(out), "--", str(ckpt)], engine_config=tiny_engine_config) == 0
    keys = json.loads((out / "capture" / "keys.json").read_text())
    assert keys and all(ranks == [0, 1] for ranks in keys.values())
    ranks = [json.loads((out / "capture" / f"rank{r}.json").read_text()) for r in (0, 1)]
    assert [r["rank"] for r in ranks] == [0, 1] and list(ranks[0]["keys"]) == list(ranks[1]["keys"])
    cache_dir = out / "cache" / "neuron" / "compile_cache"
    for k in keys:
        files = set(os.listdir(cache_dir / k))
        assert {"graph.hlo", ".artifact_metadata_v0.json", "command.txt"} <= files
        assert ".compilation_complete" not in files
        assert farm.entry_args(str(cache_dir / k))[0] == "trn2"
    cfg = json.loads((out / "capture" / "config.json").read_text())
    assert cfg["target"] == "trn2" and cfg["graphs"] == len(keys) and cfg["argv"] == [str(ckpt)]
    assert capture.HEADERS in os.listdir(out / "shapes")
