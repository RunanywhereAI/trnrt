"""tools/graph_equiv.py's artifact-metadata normalization, on the host (no HLO protos needed): the cache key and the
capture's creation time are dropped, every other field kept, so two captures of one tree compare equal and a real
metadata change still differs."""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import graph_equiv as G  # noqa: E402


def test_norm_meta_drops_the_key_and_the_creation_time():
    a = {"cache_key": "k1", "created_timestamp": 1730530000.5, "compiler": "2.27", "flags": ["-O1"]}
    b = dict(a, cache_key="k2", created_timestamp=1730539999.0)
    assert G.norm_meta(json.dumps(a)) == G.norm_meta(json.dumps(b)) == {"compiler": "2.27", "flags": ["-O1"]}
    assert G.norm_meta(json.dumps(dict(a, compiler="2.28"))) != G.norm_meta(json.dumps(a))  # the control
    assert G.norm_meta(None) is None and G.norm_meta("not json") == "not json"
