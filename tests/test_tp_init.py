"""init_rank's process group comes from torch.distributed.distributed_c10d itself, so a library that replaced the
torch.distributed attribute cannot change how an engine's ranks meet. vllm_neuron does exactly that when it is imported
(some Neuron images import it with libtorch_neuronx_lite): its port_hold_patch rebinds rank 0 to another port that
only a patched rank finds, and an engine whose rank 0 ran in such a process waited in gloo's mesh for unpatched
workers that never came (a tp=2 CPU engine in the image's test suite, after a test module imported
libtorch_neuronx_lite at collection)."""

import socket

import torch.distributed as dist

from trnrt.engine import tp as TP


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_init_rank_does_not_take_a_replaced_init_process_group(monkeypatch):
    calls = []

    def replaced(*a, **k):  # what a monkey-patching library leaves in torch.distributed
        calls.append(k.get("init_method"))
        raise AssertionError("init_rank called the replaced torch.distributed.init_process_group")

    monkeypatch.setattr(dist, "init_process_group", replaced)
    TP.init_rank(0, 1, _port())
    try:
        assert dist.is_initialized() and dist.get_world_size() == 1
        assert calls == []
    finally:
        dist.destroy_process_group()
