"""Check that this environment holds every Neuron piece trnRT imports, and that they work.

    python docker/check_image.py            # imports, the compiler CLI, the torch.compile backend
    python docker/check_image.py --device   # also compiles one graph and runs it on a NeuronCore

The container build runs the first form (docker/Dockerfile), so an image that lacks a piece is never
built; the second form needs a NeuronCore (`docker run --device=/dev/neuron0 ...`). Every failure is
printed and the exit code is 1: nothing here falls back.

What is checked is what trnrt/ imports (grep for libtorch_neuronx_lite and nki under trnrt/):
  - libtorch_neuronx_lite (LNL): registers the "neuron" torch device and the torch.compile backend
    "neuron_libtorch" that trnrt/engine/model_runner.py canonical_neuron_backend wraps; plus the
    compile-cache, capture and NKI-wrapping modules trnrt/compile/cache.py, trnrt/compile/capture.py and
    trnrt/kernels/*.py use;
  - nki (nki.isa, nki.language, nki.isa.constants.oob_mode): every kernel in trnrt/kernels/;
  - neuronx-cc: the compiler LNL and the compile farm (trnrt/compile/farm.py) run as a CLI;
  - the versions: neuronx-cc and LNL of the Neuron SDK README.md states (2.32: neuronx-cc 2.27, LNL 2.11), since a
    compile cache built by another compiler is not this one's; and a transformers with the glm5_next architecture
    (5.17 and later): the engine tests build their tiny GLM-5.3-Flash with it, and skip without it;
  - every trnrt module imports.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import pkgutil
import shutil
import subprocess
import sys

FAILED: list[str] = []
# distribution -> the version prefix it must have (README.md "What you need": Neuron SDK 2.32, neuronx-cc 2.27; SDK
# 2.32 ships libtorch-neuronx-lite 2.11)
WANT_VERSION = {"neuronx-cc": "2.27.", "libtorch-neuronx-lite": "2.11."}


def check(name: str, fn):
    try:
        out = fn()
    except BaseException as e:  # noqa: BLE001 - report every failure, then exit 1
        FAILED.append(name)
        print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
        return None
    print(f"ok   {name}" + (f": {out}" if out not in (None, "") else ""), flush=True)
    return out


def attr(module: str, *names: str):
    """`from module import names`: an attribute, or a submodule imported on demand."""
    m = importlib.import_module(module)
    for n in names:
        if not hasattr(m, n):
            importlib.import_module(f"{module}.{n}")
    return None


def dist_version(dist: str, version=importlib.metadata.version) -> str:
    """The installed version of `dist`; RuntimeError if WANT_VERSION gives it a prefix that it lacks."""
    v = version(dist)
    want = WANT_VERSION.get(dist)
    if want is not None and not v.startswith(want):
        raise RuntimeError(f"{dist} {v}, want {want}*")
    return v


def glm5_next():
    """transformers has GLM-5.3-Flash's architecture (model_type glm5_next: transformers 5.17 and later)."""
    import transformers

    importlib.import_module("transformers.models.glm5_next")
    return f"transformers {transformers.__version__}"


def cli(*argv: str) -> str:
    exe = shutil.which(argv[0])
    if exe is None:
        raise FileNotFoundError(f"{argv[0]} is not on PATH")
    p = subprocess.run([exe, *argv[1:]], capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)} exited {p.returncode}: {(p.stderr or p.stdout).strip()[-300:]}")
    return " ".join((p.stdout or p.stderr).split())[:160]


def trnrt_versions():
    import trnrt

    want = os.environ.get("TRNRT_VERSION")
    meta = importlib.metadata.version("trnrt")
    if meta != trnrt.__version__:
        raise RuntimeError(f"installed distribution {meta} != trnrt.__version__ {trnrt.__version__}")
    if want and want != trnrt.__version__:
        raise RuntimeError(f"TRNRT_VERSION={want} but trnrt.__version__ is {trnrt.__version__}")
    return f"trnrt {trnrt.__version__} from {os.path.dirname(trnrt.__file__)}"


def trnrt_modules():
    import trnrt

    names = [m.name for m in pkgutil.walk_packages(trnrt.__path__, "trnrt.")]
    for n in names:
        importlib.import_module(n)
    return f"{len(names)} modules"


def backend():
    """LNL registers "neuron_libtorch" at import only when /dev/neuron* exists (libtorch_neuronx_lite/__init__.py
    _register_compile_backends, SDK 2.32), so without a device only the function it registers is checked."""
    import glob

    import torch._dynamo as dynamo

    import libtorch_neuronx_lite  # noqa: F401  registers the "neuron" device and, with a device, the backend
    from libtorch_neuronx_lite.compile.backend import compile as lnl_compile

    if not glob.glob("/dev/neuron*"):
        dynamo.lookup_backend("neuron_libtorch_graph_capture")
        return f"no /dev/neuron*: {lnl_compile.__module__}.compile present, registered when a device is"
    dynamo.lookup_backend("neuron_libtorch")
    from trnrt.engine.model_runner import canonical_neuron_backend

    return f"neuron_libtorch registered, wrapped as {canonical_neuron_backend().__name__}"


def device_graph():
    """One small graph on a NeuronCore the way trnRT compiles its own (trnrt/engine/engine.py and
    model_runner.py: the runtime env first, then LNL, device neuron:0, trnRT's backend wrapper and
    compiler arguments), against the CPU."""
    import torch

    from trnrt import platform

    platform.configure_runtime_env()
    import libtorch_neuronx_lite  # noqa: F401
    from trnrt.engine.model_runner import canonical_neuron_backend, neuronx_cc_args

    def f(x, w):
        return torch.nn.functional.silu(x @ w) * 2.0 + 1.0

    g = torch.Generator().manual_seed(0)
    x, w = torch.randn(64, 128, generator=g), torch.randn(128, 256, generator=g)
    want = f(x, w)
    dev = torch.device("neuron:0")
    fn = torch.compile(f, backend=canonical_neuron_backend(), fullgraph=True, dynamic=False,
                       options={"compiler_args": neuronx_cc_args(torch.bfloat16)})
    got = fn(x.to(dev), w.to(dev)).cpu()
    err = (got - want).abs().max().item()
    if not err < 0.05 * want.abs().max().item():  # a bf16 matmul's error at most
        raise RuntimeError(f"max |device - cpu| {err}")
    return f"max |device - cpu| {err:.3g} (output max {want.abs().max().item():.3g}), {platform.describe().get('target')}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", action="store_true", help="also run a graph on a NeuronCore")
    args = ap.parse_args()

    check("python", lambda: sys.version.split()[0])
    check("trnrt version", trnrt_versions)
    for dist in ("torch", "libtorch-neuronx-lite", "nki", "neuronx-cc", "transformers", "tokenizers",
                 "safetensors", "huggingface_hub", "fastapi", "uvicorn", "pydantic", "numpy"):
        check(f"dist {dist}", lambda d=dist: dist_version(d))
    check("transformers glm5_next", glm5_next)
    check("nki", lambda: attr("nki", "jit", "simulate"))
    check("nki.isa", lambda: attr("nki.isa", "nc_matmul", "dma_copy", "activation", "memset"))
    check("nki.language", lambda: attr("nki.language", "ndarray", "sbuf", "psum", "shared_hbm"))
    check("nki.isa.constants.oob_mode", lambda: attr("nki.isa.constants", "oob_mode"))
    check("libtorch_neuronx_lite.nki.nki_hop.wrap_nki", lambda: attr("libtorch_neuronx_lite.nki.nki_hop", "wrap_nki"))
    check("libtorch_neuronx_lite.compile.cache", lambda: attr("libtorch_neuronx_lite.compile", "cache"))
    check("libtorch_neuronx_lite.compile.backend",
          lambda: attr("libtorch_neuronx_lite.compile.backend", "_parse_compiler_args",
                       "_apply_platform_compiler_args", "preprocess_and_validate_inputs"))
    check("libtorch_neuronx_lite.compile.capture_backend",
          lambda: attr("libtorch_neuronx_lite.compile.capture_backend", "run_fx_to_hlo_pipeline",
                       "setup_workdir_common"))
    check("libtorch_neuronx_lite.compile.hlo", lambda: attr("libtorch_neuronx_lite.compile.hlo", "load_hlo_module"))
    check("libtorch_neuronx_lite.compile.platform",
          lambda: attr("libtorch_neuronx_lite.compile.platform", "resolve_target", "get_platform_target"))
    check("libtorch_neuronx_lite.compile.schema", lambda: attr("libtorch_neuronx_lite.compile.schema", "create_metadata"))
    check("libtorch_neuronx_lite.pyhlo.xla_data_pb2", lambda: attr("libtorch_neuronx_lite.pyhlo", "xla_data_pb2"))
    check("libtorch_neuronx_lite.overrides",
          lambda: attr("libtorch_neuronx_lite.overrides", "neuron_collectives", "xla_collectives"))
    check("torch.compile backend neuron_libtorch", backend)
    check("neuronx-cc --version", lambda: cli("neuronx-cc", "--version"))
    check("trnrt modules", trnrt_modules)
    if args.device:
        check("neuron-ls", lambda: cli("neuron-ls"))
        check("graph on a NeuronCore", device_graph)
    if FAILED:
        print(f"{len(FAILED)} check(s) failed: {', '.join(FAILED)}", flush=True)
        return 1
    print("all checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
