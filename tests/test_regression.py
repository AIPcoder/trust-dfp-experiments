"""The simulator must reproduce the trajectories behind the paper's results exactly."""

import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_experiments  # noqa: E402


def test_actions_match_reference_run():
    """One full-length run (T = 5000): the star network, Byzantine attack, seed 8, the case that
    ends at a symmetric mixed equilibrium. Reference hash taken from the paper's run."""
    ref = json.load(open(os.path.join(ROOT, "tests", "reference_actions_sha256.json")))
    run_experiments.CFG["T"] = 5000
    key = ("star", "byzantine", "trust_only", 8)
    _, out = run_experiments.work(key)
    h = hashlib.sha256(np.ascontiguousarray(out["A"]).tobytes()).hexdigest()
    assert h == ref["|".join(map(str, key))]
    assert not out["NE"][-1]  # not a pure NE: the symmetric mixed-NE case explained in the README
