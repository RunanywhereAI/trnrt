"""What trnRT needs to know about the Neuron chip it runs on, in one place.

trnRT targets trn2 (NeuronCore-v3) only. This module answers the target string, the logical NeuronCore config
(LNC), HBM per logical core, the largest finite FP8 E4M3 value, the kernels that split their work over the two
physical cores of a logical core, and the neuronx-cc arguments that follow from them. The rest of the engine asks
this module instead of branching on target strings itself.

The chip has to be known BEFORE the Neuron runtime starts, because the runtime's LNC is an environment variable read
at start-up and must match the compiler's: "AWS Neuron currently doesn't support setting the compiler flag to a
different LNC configuration than the Neuron Runtime environment variable" (aws-neuron-sdk,
about-neuron/arch/neuron-features/logical-neuroncore-config.rst). libtorch_neuronx_lite (LNL) can name the target
only through the runtime (compile/platform.py, _get_target_from_nrt), so `target()` reads what the Neuron driver
itself reads instead: /sys/class/dmi/id/product_name, the EC2 instance type (aws-neuron-driver neuron_arch.c,
narch_get_instance_type_name). After the runtime is up, `check_runtime()` compares that answer with LNL's.
"""

from __future__ import annotations

import glob
import os
import shlex
from dataclasses import dataclass

FAMILY = "trn2"
# HBM per logical NeuronCore in GiB at LNC 2 (LNL compile/platform.py HBM_MEMORY_GB).
HBM_GIB_PER_CORE = 24
NKI_GEN = {FAMILY: 3}  # nki/__init__.py: trn2 = gen3
PHYSICAL_CORES_PER_CHIP = 8
# neuronx-cc --logical-nc-config: "(Only available on trn2; Default: 2)" (compiler/neuronx-cc/api-reference-guide).
LNC = 2
# Device name the driver publishes under neuron<N>/info/architecture/device_name (aws-neuron-driver
# neuron_dhal_v*.c, arch_device_name_suffix).
DEVICE_NAME = "Trainium2"
DMI_PRODUCT = "/sys/class/dmi/id/product_name"
SYSFS_DEVICE_NAME = "/sys/devices/virtual/neuron_device/neuron*/info/architecture/device_name"
# vllm-neuron's neuronx-cc arguments for every graph (neuron_model_runner.py, compile_options), except the
# optimization level: trnRT keeps neuronx-cc's own default -O2, where vllm-neuron 0.24 defaults to -O1.
CC_ARGS = ("--auto-cast=none", "--verbose=35", "-O2")
HLO2TENSORIZER_ARGS = ("--modular-flow-mac-threshold=10",)
BACKEND_ARGS = "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop"
UNSAFE_FP8 = "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3"
HLO2TENSORIZER = "--internal-hlo2tensorizer-options="


def family_of(target: str) -> str:
    """'trn2' / 'trn2.48xlarge' / 'trn2u.48xlarge' -> 'trn2'; anything else is refused."""
    if target.startswith(FAMILY):
        return FAMILY
    raise ValueError(f"trnRT targets trn2 (NeuronCore-v3); this host is {target!r}")


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def instance_type() -> str | None:
    """The EC2 instance type, as the Neuron driver reads it (neuron_arch.c)."""
    return _read(DMI_PRODUCT)


def target() -> str | None:
    """The Neuron target string without starting the runtime, or None off Neuron.

    Order: LNL's own override NEURON_PLATFORM_TARGET_OVERRIDE (it also wins inside LNL, so both agree; a CPU compile
    host sets it to trn2); the EC2 instance type; the driver's device name."""
    if os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE"):
        return os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"]
    it = instance_type()
    if it and it.startswith(FAMILY):
        return FAMILY
    for path in sorted(glob.glob(SYSFS_DEVICE_NAME)):
        if _read(path) == DEVICE_NAME:
            return FAMILY
    return None


def lnc(fam: str) -> int:
    """The logical NeuronCore config trnRT runs trn2 at."""
    family_of(fam)
    return LNC


_LNC: int | None = None  # resolved by configure_runtime_env, before anything is traced

# The NKI kernels that split their work over the two physical cores of an LNC 2 logical core (each by program id:
# halves of the rows, query tiles or heads per program; each split is bit-identical to its unsplit form). A kernel
# left out runs the same work on both physical cores. Part of every graph key, since the kernels read it at trace.
LNC_SPLIT = frozenset(("delta_rule", "dsa_topk", "moe_dedupe", "kda_decode", "dsa_decode", "dsa_fused", "moe_ep",
                       "moe_prefill", "dsa_slots_n", "dsa_long_pipe"))


def lnc_split(kernel: str) -> bool:
    return kernel in LNC_SPLIT


def nki_grid() -> int:
    """Grid for an NKI kernel called inside an LNL graph: the runtime's LNC. "The LNC value must match the
    NEURON_LOGICAL_NC_CONFIG environment variable ... Mismatching the two will cause a runtime error"
    (nki/__init__.py). At LNC 2 both physical cores run a kernel that does not split its work by program_id,
    writing the same output.

    Called while dynamo traces a graph, so it must not touch the runtime (torch.classes.neuron.Runtime() is not
    traceable) or the filesystem: it returns the value configure_runtime_env cached."""
    if _LNC is None:
        raise RuntimeError("trnrt.platform.configure_runtime_env() must run before an NKI kernel is traced")
    return _LNC


def configure_runtime_env() -> None:
    """Call before libtorch_neuronx_lite is imported. On trn2 sets NEURON_LOGICAL_NC_CONFIG to the LNC that
    neuronx_cc_args passes to the compiler; off Neuron (CPU tests) the grid is 1."""
    global _LNC
    t = target()
    if t is not None:
        family_of(t)
        os.environ["NEURON_LOGICAL_NC_CONFIG"] = str(LNC)
    _LNC = LNC if t is not None else 1


def fp8_max(target_str: str | None = None) -> float:
    """Largest finite FP8 E4M3 value on trn2: vllm-neuron 0.24 (utils/dtype_utils.py): "TRN2 uses e4m3 (with inf),
    max finite = 240"."""
    return 240.0


def neuronx_cc_args(target_str: str, fp8: bool) -> list[str]:
    """Platform arguments for every neuronx-cc call: --logical-nc-config matching the runtime, plus vllm-neuron's
    argument set. The FP8 option joins the one --internal-hlo2tensorizer-options argument, since a second one
    would replace the first."""
    family_of(target_str)
    h2t = list(HLO2TENSORIZER_ARGS) + ([UNSAFE_FP8] if fp8 else [])
    return [f"--logical-nc-config={LNC}", *CC_ARGS, BACKEND_ARGS, HLO2TENSORIZER + " ".join(h2t)]


@dataclass(frozen=True)
class Platform:
    target: str
    instance_type: str | None
    lnc: int = LNC
    family: str = FAMILY

    @property
    def nki_gen(self) -> int:
        return NKI_GEN[FAMILY]

    @property
    def hbm_gib_per_core(self) -> float:
        return HBM_GIB_PER_CORE

    @property
    def fp8_max(self) -> float:
        return fp8_max()

    @property
    def logical_cores_per_chip(self) -> int:
        return PHYSICAL_CORES_PER_CHIP // self.lnc


def detect() -> Platform | None:
    t = target()
    if t is None:
        return None
    family_of(t)
    return Platform(target=t, instance_type=instance_type())


def runtime_target() -> str:
    """The target LNL compiles for (env override, else the runtime). Needs a live device."""
    from libtorch_neuronx_lite.compile.platform import get_platform_target

    return get_platform_target()


def check_runtime() -> dict:
    """After libtorch_neuronx_lite is imported: the runtime's target must be trn2, the family this module set the
    LNC for."""
    p = detect()
    rt = runtime_target()
    if p is not None and not rt.startswith(FAMILY):
        raise RuntimeError(f"runtime target {rt!r} is not trn2 ({DMI_PRODUCT}: {p.instance_type!r})")
    return describe(rt)


def versions() -> dict:
    from importlib import metadata

    out = {}
    for dist in ("neuronx-cc", "libtorch-neuronx-lite", "torch", "nki"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            pass
    return out


def describe(runtime: str | None = None) -> dict:
    """One record of the platform for logs."""
    p = detect()
    if p is None:
        return {"target": None}
    rec = {"instance_type": p.instance_type, "target": runtime or p.target, "lnc": p.lnc, "nki_gen": p.nki_gen,
           "hbm_gib_per_core": p.hbm_gib_per_core, "fp8_max": p.fp8_max,
           "logical_cores_per_chip": p.logical_cores_per_chip, "env_lnc": os.environ.get("NEURON_LOGICAL_NC_CONFIG"),
           "cc_args": shlex.join(neuronx_cc_args(runtime or p.target, fp8=False))}
    rec.update(versions())
    return rec
