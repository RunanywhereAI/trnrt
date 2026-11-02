"""tools/graph_equiv.py's command.txt compare: the compile-cache key it names is masked, and nothing else is. Needs neither
the HLO protos nor zstd, so it runs everywhere (tests/test_graph_equiv.py skips without them)."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import graph_equiv as G  # noqa: E402


def cmd(root, key, opt="-O2"):
    return (f"neuronx-cc compile --framework=XLA {root}/cache/neuron/compile_cache/{key}/graph.hlo "
            f"--output {root}/cache/neuron/compile_cache/{key}/graph_{key}.neff {opt} --target=trn2").encode()


A = cmd("/data/ref", "01a930a6627b50fddf657b7cd9b2fbd8")
B = cmd("/data/new.part", "d7195a0954865208b93651c86e76b0f3")


def test_two_trees_commands_differ_only_in_the_key():
    """command.txt names its own cache entry (graph_<key>.neff), and keys never match across two trees: masked the commands
    are equal; unmasked they differ (the control: the key is what differed)."""
    assert G.norm_command(A) == G.norm_command(B)
    assert G.norm_command(A, mask_key=False) != G.norm_command(B, mask_key=False)


def test_a_real_argument_change_still_differs():
    assert G.norm_command(A) != G.norm_command(cmd("/data/new.part", "d7195a0954865208b93651c86e76b0f3", opt="-O1"))
