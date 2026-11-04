# Running on AWS

TrnRT needs two hosts: a `trn2.48xlarge` to serve on, and any x86_64 host with Docker to build the compile cache on.
They can be the same machine, but the compile step needs only CPUs and memory, so a cheaper instance does it.

## 1. Get a trn2.48xlarge

| Option | When to use it | Link |
|---|---|---|
| EC2 Capacity Blocks for ML | Reserve a fixed window (days to weeks) at a known price. The benchmarks quote this price. | [Capacity Blocks](https://aws.amazon.com/ec2/capacityblocks/) · [user guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html) |
| On-demand | Short tests, if your account has trn2 quota | [On-demand pricing](https://aws.amazon.com/ec2/pricing/on-demand/) |
| Spot | Cheapest, but can be interrupted | [Spot pricing](https://aws.amazon.com/ec2/spot/pricing/) |

The instance: 16 Trainium2 chips with 96 GiB of HBM each, 192 vCPUs and 2 TB of memory
([trn2 instances](https://aws.amazon.com/ec2/instance-types/trn2/),
[Trainium2 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html)).

## 2. Prepare the host

- **AMI:** a Neuron [AWS Deep Learning AMI](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html)
  comes with the Neuron driver and tools. On a plain Ubuntu 24.04 AMI, install them with the
  [Neuron setup guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/index.html). The host needs the
  Neuron driver; the Neuron SDK, compiler and runtime ship inside TrnRT's image.
- **Docker:** [install Docker Engine](https://docs.docker.com/engine/install/).
- **Check the devices:** `neuron-ls` (from the Neuron tools) lists the 16 devices of a `trn2.48xlarge`, and the
  processes holding them
  ([neuron-ls guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-ls.html)).
- **Disk:** about 30 GB for the image and 328 GB for the checkpoint, plus the compile cache. Attach storage to match.
- **Hugging Face:** `pip install -U huggingface_hub` gives the `hf` command. Set `HF_TOKEN` to avoid the Hub's
  per-IP rate limits ([CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli)).

## 3. Deploy

Follow [DEPLOY.md](DEPLOY.md): build the image, build the compile cache, then serve.

## Neuron references

| | |
|---|---|
| Neuron SDK documentation | [awsdocs-neuron.readthedocs-hosted.com](https://awsdocs-neuron.readthedocs-hosted.com/) |
| Neuron release notes | [release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/index.html) |
| NKI, the kernel language TrnRT's kernels use | [NKI documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html) |
| Trainium memory hierarchy (SBUF, PSUM, HBM) | [memory hierarchy](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/about/memory-hierarchy-overview.html) |
| Profiling | [Neuron Explorer](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-explorer/index.html) |
| Neuron containers | [aws-neuron/deep-learning-containers](https://github.com/aws-neuron/deep-learning-containers) · TrnRT's base image: [pytorch-inference-vllm-neuronx](https://gallery.ecr.aws/neuron/pytorch-inference-vllm-neuronx) |
