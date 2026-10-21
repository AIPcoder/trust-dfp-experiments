"""Nash gap of the empirical frequencies for every run -> <out>/gaps.npz  (needs runs.pkl)."""
import _bootstrap  # noqa: F401
import argparse, os, pickle
import numpy as np
from trustdfp.metrics import gap_series
from trustdfp import dfp_v8_lil as base

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); a = ap.parse_args()
D = pickle.load(open(os.path.join(a.out, "runs.pkl"), "rb")); R, runs = D["res"], D["meta"]["runs"]
G = {}
for topo in ("ring", "star"):
    for cfg in ("none", "trust_only"):
        G[f"{topo}|{cfg}"] = np.vstack([gap_series(R[(topo, at, cfg, r)]["A"].astype(int), R[(topo, at, cfg, r)]["d"])
                                        for at in base.ATTACK_TYPES for r in range(1, runs + 1)])
np.savez_compressed(os.path.join(a.out, "gaps.npz"), **G)
print("wrote", os.path.join(a.out, "gaps.npz"), {k: v.shape for k, v in G.items()})
