"""The ahead-of-time compile pipeline: every graph the served engine runs, captured on a CPU host
(capture.py), compiled with neuronx-cc in the libtorch_neuronx_lite cache layout (farm.py, cache.py)
and stripped of its debug info (strip.py). `python -m trnrt compile` runs it (farm.main)."""
