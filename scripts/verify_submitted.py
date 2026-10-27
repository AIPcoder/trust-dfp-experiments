"""Compare a run_m1.py run with the numbers printed in the submitted paper (Sec. IV, Fig. 1).

Needs the full setting: 100 seeds, T = 5000. Paper values are typed in by hand (not read from the run).
Exit code 0 = all reproduced within rounding.
"""

import argparse, sys
from pathlib import Path
import numpy as np

# (attack, mode, paper value, tolerance = half-unit of the last printed digit)
GAP = [
    ("equivocation", "trust_only", 0.0047, 0.00005),
    ("oscillating", "trust_only", 0.0136, 0.00005),
    ("equivocation", "none", 0.174, 0.0005),
    ("oscillating", "none", 0.463, 0.0005),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results_m1"))
    d = np.load(ap.parse_args().out / "m1_results.npz")
    attacks, modes = list(d["attacks"]), list(d["modes"])
    if d["gap"].shape[2:] != (100, 5000):
        sys.exit("Needs the full setting: python scripts/run_m1.py --seeds 100 --steps 5000")
    bad = 0
    for a, m, val, tol in GAP:
        got = d["gap"][attacks.index(a), modes.index(m), :, -1].mean()
        ok = abs(got - val) <= tol + 1e-12
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  Nash gap at T, {a:12s} {m:10s} paper={val:<7g} run={got:.5f}")
    for a in attacks:
        ai = attacks.index(a)
        e = d["estimation_error"][ai, modes.index("trust_only"), :, -1].mean()
        n = int((d["pure_ne_from"][ai, modes.index("trust_only")] > 0).sum())
        for ok, msg in ((e < 1e-5, f"est. error with trust < 1e-5  run={e:.3g}"), (n == 100, f"pure NE with trust 100/100  run={n}/100")):
            bad += not ok
            print(f"{'PASS' if ok else 'FAIL'}  {a:12s} {msg}")
    print("\nALL REPRODUCED" if not bad else f"\n{bad} MISMATCH(ES)")
    sys.exit(1 if bad else 0)
