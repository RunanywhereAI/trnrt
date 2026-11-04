# Deploy

Build the image, build the compile cache on any x86_64 host, then serve on a `trn2.48xlarge`. New to Trainium: [AWS.md](AWS.md) covers getting an instance. Flags and environment variables: [CONFIGURATION.md](CONFIGURATION.md). Requests: [API.md](API.md).

## What you need

| | |
|---|---|
| Serving host | one trn2.48xlarge: 16 Trainium2 chips with 96 GiB of HBM each (64 logical NeuronCores at LNC 2), 192 vCPUs, 2 TB of memory (AWS's published specification) |
| Compile host | any x86_64 host with Docker; no NeuronCore needed. `compile` runs as many neuronx-cc processes at once as host memory admits, 30 GB of available memory each by default (`--mem-per-compile-gb`), so its memory and cores set the compile time |
| Software | Docker, and the image built from `docker/`: Neuron SDK 2.32 (neuronx-cc 2.27), Python 3.13, its base image pinned by digest. The base image is about an 8 GB download |
| Model | `zai-org/GLM-5.3-Flash` at revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`, FP8 weights as released, 328 GB: the same revision for compiling and serving |
| Disk | on each host, the built image (26.7 GB); on the compile host, the compile output under `$OUT`; on the serving host, the checkpoint (328 GB) and the compile cache |
| Time | building the compile cache takes hours. Start-up loads the weights and warms every compiled graph, one engine at a time, and each engine has `--startup-timeout` (default 3600 s) to come up |
| Cost | one trn2.48xlarge for as long as it serves; [RESULTS.md](RESULTS.md) gives the cost per million tokens and per request at the two prices above |

## Steps

### 1. Get the code

```bash
git clone https://github.com/RunanywhereAI/trnrt.git
cd trnrt
```

Run the commands below from the repository's root.

### 2. Build the image

```bash
docker build -f docker/Dockerfile \
  --build-arg TRNRT_VERSION=$(python3 -c 'import trnrt; print(trnrt.__version__)') \
  --build-arg TRNRT_REVISION=$(git rev-parse HEAD) -t trnrt .
```

The build checks every Neuron piece TrnRT imports and its version, and that the image's transformers has `glm5_next`
(`docker/check_image.py`). Build the image on the compile host and on the serving host: the base image is about an
8 GB download, and the built image takes 26.7 GB of disk.

### 3. Build the compile cache (CPU host)

A graph depends on the weights' shapes and dtypes, and on MoE packing decisions read from a few weight values per
layer (`decisions`), never on the rest of the values. So the cache is built from the checkpoint's config and safetensors
headers (a few MB) plus those reads, without downloading the weights, on a host without NeuronCores:

```bash
OUT=/data/trnrt                               # everything goes under one directory
REV=eb9eb208eb0d988989d07a6a12d0fdeb5f52574a  # the revision the published numbers were measured on (below)
mkdir -p $OUT/nki $OUT/work                   # the NKI kernel binaries, shared by every step; neuronx-cc's scratch
# each step: `python -m trnrt` in a new container of the image, with $OUT at the same path and $OUT/nki mounted
trnrt() { docker run --rm -e HF_TOKEN -v $OUT:$OUT -v $OUT/nki:/var/tmp/nki-intermediate-cache trnrt "$@"; }
trnrt compile shapes zai-org/GLM-5.3-Flash --revision $REV --out $OUT
trnrt compile decisions --model zai-org/GLM-5.3-Flash --revision $REV --out $OUT   # a few range reads per MoE layer
trnrt compile capture --out $OUT -- --model $OUT/shapes                            # every graph, all 32 ranks
trnrt compile compile --out $OUT --work-root $OUT/work                              # neuronx-cc on each graph
trnrt compile strip --out $OUT                                                      # drop the NEFFs' debug info
```

Each step is its own `docker run --rm`, so whatever a step leaves inside its container is gone by the next. A graph
that calls an NKI kernel names the kernel's compiled binary by absolute path under `/var/tmp/nki-intermediate-cache`.
The capture writes those binaries, and neuronx-cc adds its own kernels there while compiling. So `capture` and
`compile` must see the same directory at that path: the host's `$OUT/nki`, mounted into every step above. Without it,
or on another host without a copy of it, `compile` fails every such graph ("NKI kernel binaries missing"). To split the
steps across hosts, copy `$OUT/nki` with the cache. A serving engine traces its graphs again and writes to that
directory too, so `serve` below mounts the compile host's copy.

- Hugging Face requests carry no token unless `HF_TOKEN` is set on the host (`-e HF_TOKEN` passes it in). Without one,
  the Hub rate-limits requests by IP.
- `compile` runs as many neuronx-cc processes at once as host memory admits. One can take tens of GB, so each reserves
  `--mem-per-compile-gb` (default 30) of MemAvailable, and one the OOM killer takes runs once more, alone.
  `--work-root $OUT/work` keeps neuronx-cc's scratch trees on the host: a failed compile's tree stays there with its
  logs, where the container's own temporary directory would be gone when the step ends.
- `strip` is not optional at 1M-token buckets: the runtime holds each loaded NEFF in host memory several times over,
  and two engines' unstripped NEFFs do not fit.
- The containers run as root, so everything under `$OUT` belongs to root; `sudo chown -R "$(id -u):$(id -g)" $OUT`
  hands it back once the steps are done.

The arguments after `capture --` must be the engine arguments the server runs with, spelled out in full. Capture refuses
any other; the server's own, such as `--engines` and `--port`, do not change the graphs. The defaults are the default
configuration (batch 12), so `--model` alone is enough. For batch 8 ([Batch 8](CONFIGURATION.md#batch-8)), pass `--preset batch8` to
both `capture` and `serve`.

### 4. Download the checkpoint and serve (trn2.48xlarge)

Copy `$OUT/cache` and `$OUT/nki` to the serving host, and download the checkpoint at the revision the cache was built
from (the `hf` command comes with `pip install -U huggingface_hub`):

```bash
hf download zai-org/GLM-5.3-Flash --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a --local-dir /models/GLM-5.3-Flash
```

That is the revision the published numbers were measured on; later revisions change the chat template. Give `serve`
the downloaded directory. Alternatively, `--model` takes the Hugging Face repo id and `serve` downloads the checkpoint
itself: then pass `--revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` too, since without it the download is the
repo's default branch (`--revision` with a local directory is refused). Then start the server:

```bash
docker run --rm --privileged --network host --ipc=host \
  -v /data/trnrt/cache:/cache -v /data/trnrt/nki:/var/tmp/nki-intermediate-cache -v /models/GLM-5.3-Flash:/model \
  -e NEURON_LIBTORCH_CACHE_ROOT=/cache \
  trnrt serve --model /model --port 8000
```

`serve` starts two engines, on logical NeuronCores 0-31 and 32-63, on ports 8001 and 8002 of the host, one at a time:
each loads its weights and warms every compiled graph from the cache, and the next starts once it answers `/health`.
Then the router answers on `--port`.

- **A graph missing from the cache fails start-up.** The engines run with `NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`: instead
  of compiling on the serving host (neuronx-cc, minutes to hours per graph), the engine stops and names the missing
  key, and `serve` stops every engine and exits nonzero. The cache was built for other arguments, another checkpoint
  revision, another TrnRT version or another Neuron SDK, or `NEURON_LIBTORCH_CACHE_ROOT` does not point at it.
  `--allow-compile` compiles the missing graphs on the spot instead; give it a `--startup-timeout` that allows for that
  (0: none).
- **`--startup-timeout`** (default 3600 s): how long each engine has, from its start, to answer `/health`. Past it,
  `serve` stops every engine and exits nonzero.
- **Each engine checks its host before it starts**, and refuses to start, with exit status 4 and a message saying what
  it found and how to fix it (`serve` then stops every engine and exits nonzero), when:
  - another process's Neuron runtime holds one of the engine's devices, as `neuron-ls` lists them (without `neuron-ls`
    this check is skipped with a warning; a listing it cannot read refuses the start). Stop that process, or set
    `TRNRT_SKIP_DEVICE_CHECK=1`;
  - the host has too little free contiguous memory for the Neuron runtime's DMA rings: fewer than 1,024 free order-10
    (4 MiB) blocks over all zones, or fewer than 256 in any NUMA node's Normal zone (`/proc/buddyinfo`). Clean page
    cache, such as a checkpoint just downloaded, is the usual cause: drop it on the host
    (`sync; echo 1 | sudo tee /proc/sys/vm/drop_caches`) and start again, or set `TRNRT_SKIP_HOST_MEM_CHECK=1`.
- `--ipc=host`: each engine's rank 0 hands its calls to the engine's other ranks through a 64 MiB segment in /dev/shm
  (`TRNRT_TP_SHM_BYTES`), one per engine, and Docker's default /dev/shm is 64 MB. Keep it in the command (or pass
  `--shm-size=1g`).
- The router's `/health` answers 200 while both engines are up, 200 with `"status": "degraded"` while one is (its
  requests and sessions go to the other), and 503 while neither is. The body names the engines that are down.
- SIGTERM (`docker stop`) stops the router, then each engine, and `serve` exits with status 143.
