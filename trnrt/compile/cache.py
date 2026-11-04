"""The local compile-cache layout libtorch_neuronx_lite (LNL) reads, and its keys.

LNL keeps one directory per graph under its cache, named by the graph's cache key:

    <NEURON_LIBTORCH_CACHE_ROOT>/neuron/compile_cache/<key>/
        graph.hlo                     the lowered graph
        .artifact_metadata_v0.json    io_map, output count, unused inputs, the key itself
        command.txt                   the neuronx-cc command line
        graph_<key>.neff              the compiled graph
        log-neuron-cc.txt             the compiler's log
        .compilation_complete         written LAST: "completed:<time>\\nneff_size:<bytes>\\n"

The compile pipeline (trnrt/compile/farm.py) works in one directory, `--out` (default ~/.cache/trnrt):

    <out>/shapes/                     the checkpoint's config and safetensors headers (or any --shape-dir)
    <out>/capture/                    rank<R>.json per rank, keys.json, config.json (capture.capture_all)
    <out>/cache/                      the LNL cache ROOT: the serving host sets
                                      NEURON_LIBTORCH_CACHE_ROOT=<out>/cache

A directory is a cache hit once graph.hlo, the NEFF, the metadata and `.compilation_complete` are all
present (LNL compile/cache.py _is_cache_complete), and LNL never recompiles it. So every writer here
writes the marker last: a reader that sees the marker sees a complete entry.

The key (LNL compile/cache.py create_cache_hash) hashes the FX graph's text, the input shapes /
dtypes / strides, the collectives' replica groups, the package versions, the compiler arguments and
the target. It does not hash weight values, so a graph captured on a CPU host from the checkpoint's
shapes alone gets the key the device run computes (trnrt/compile/capture.py).

Graphs that call NKI kernels need two more things on the host that COMPILES them: LNL's NKI compile
results (`<cache>/nki/`, inside the cache root) and the kernel binaries they point at, which the NKI
compiler writes outside the cache, under NKI_KERNEL_DIR. The HLO references those binaries by absolute
path (trnrt/compile/farm.py kernel_refs), and neuronx-cc fails (NCC_EVRF059, "Kernel file ...
referenced by AwsNeuronCustomNativeKernel instruction does not exist on the host") without them. The
simplest arrangement is to capture and compile on the same host; to compile elsewhere, copy
NKI_KERNEL_DIR along with the cache root.
"""

from __future__ import annotations

import os

MARKER = ".compilation_complete"  # LNL compile/cache.py NEURON_COMPILE_COMPLETE_FILE
HLO = "graph.hlo"  # NEURON_COMPILE_GRAPH_HLO_FILE
COMMAND = "command.txt"  # the neuronx-cc command line, as LNL's neuroncc_compile writes it
LOG = "log-neuron-cc.txt"
METADATA = ".artifact_metadata_v0.json"  # compile/cache.py _get_artifact_metadata_filename(0)
NKI_SUBDIR = "nki"  # LNL nki/nki_cache.py _NKI_CACHE_SUBDIR: its NKI compile results, under the cache
# Where the NKI compiler leaves kernel binaries (the path neuronx-cc names in NCC_EVRF059).
NKI_KERNEL_DIR = "/var/tmp/nki-intermediate-cache"
DEFAULT_ROOT = "~/.cache/neuron_libtorch"  # LNL's default NEURON_LIBTORCH_CACHE_ROOT
DEFAULT_OUT = "~/.cache/trnrt"  # the compile pipeline's work directory (farm.py --out)


def out_dirs(out: str) -> dict[str, str]:
    """The pipeline's directories under its work directory `out` (module docstring)."""
    out = os.path.abspath(os.path.expanduser(out))
    return {
        "out": out,
        "shapes": os.path.join(out, "shapes"),
        "capture": os.path.join(out, "capture"),
        "cache_root": os.path.join(out, "cache"),
    }


def cache_root() -> str:
    """The cache ROOT, as LNL resolves it (libtorch_neuronx_lite/envs.py): NEURON_LIBTORCH_CACHE_ROOT,
    else ~/.cache/neuron_libtorch. It is LNL's own variable, read by every process that captures,
    compiles or serves, so a deployment chooses it there (the serving host points it at the
    compiled cache)."""
    return os.path.expanduser(os.environ.get("NEURON_LIBTORCH_CACHE_ROOT", DEFAULT_ROOT))


def local_root(root: str | None = None) -> str:
    """The directory that holds the entries: <root>/neuron/compile_cache (root: cache_root())."""
    return os.path.join(os.path.expanduser(root) if root else cache_root(), "neuron", "compile_cache")


def neff_name(key: str) -> str:
    """LNL compile/cache.py get_neff_filename: graph_<key>.neff."""
    return f"graph_{key}.neff"


def key_of(entry: str) -> str:
    return os.path.basename(entry.rstrip("/"))


def neff_path(entry: str) -> str:
    return os.path.join(entry, neff_name(key_of(entry)))


def is_complete(entry: str) -> bool:
    """LNL's cache-hit test (_is_cache_complete): graph.hlo, the NEFF, the marker and artifact
    metadata of any version (.artifact_metadata_v<N>.json)."""
    import glob

    return all(os.path.exists(os.path.join(entry, f)) for f in (HLO, neff_name(key_of(entry)), MARKER)) and bool(
        glob.glob(os.path.join(glob.escape(entry), ".artifact_metadata_v*.json"))
    )


def entries(root: str) -> list[str]:
    """Every entry directory under `root` (an entry directory holds graph.hlo), sorted by key. A
    capture's private lowering directory (<key>.capture.<pid>, capture.capture_graph: renamed into
    place once complete, left behind by a killed capture) is not an entry."""
    if not os.path.isdir(root):
        return []
    out = []
    for k in sorted(os.listdir(root)):
        d = os.path.join(root, k)
        if ".capture." not in k and os.path.isdir(d) and os.path.exists(os.path.join(d, HLO)):
            out.append(d)
    return out


def pending(root: str) -> list[str]:
    """Entries that are captured (graph.hlo) but not compiled (no completion marker)."""
    return [d for d in entries(root) if not os.path.exists(os.path.join(d, MARKER))]


def marker_text(neff_size: int, when: float | None = None) -> str:
    """The completion marker in LNL's format (compile/cache.py _store_local)."""
    import time

    return f"completed:{time.time() if when is None else when}\nneff_size:{neff_size}\n"


def read_marker(entry: str) -> dict[str, str]:
    """The marker's fields ({"completed": ..., "neff_size": ...}); {} when there is none."""
    p = os.path.join(entry, MARKER)
    if not os.path.exists(p):
        return {}
    out = {}
    with open(p) as f:
        for line in f:
            k, sep, v = line.strip().partition(":")
            if sep:
                out[k] = v
    return out


def write_marker(entry: str, neff_size: int) -> None:
    """Write the completion marker atomically (a temporary file renamed into place). Callers write it
    after every other file of the entry."""
    p = os.path.join(entry, MARKER)
    tmp = f"{p}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(marker_text(neff_size))
    os.replace(tmp, p)


def cache_key(gm, example_inputs, options: dict) -> str:
    """LNL's cache key for a graph, computed as compile() computes it (libtorch_neuronx_lite
    compile/backend.py: preprocess_and_validate_inputs, _apply_platform_compiler_args,
    cache.create_cache_hash), on a copy, so the graph LNL then receives is untouched."""
    import copy

    from libtorch_neuronx_lite.compile import cache
    from libtorch_neuronx_lite.compile.backend import _apply_platform_compiler_args, preprocess_and_validate_inputs

    gm2, inputs = preprocess_and_validate_inputs(copy.deepcopy(gm), list(example_inputs), options)
    return cache.create_cache_hash(gm2, inputs, _apply_platform_compiler_args(options))
