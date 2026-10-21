"""Compare a full-setting run (--runs 20 --T 5000) with the numbers printed in the paper.
Exit code 0 = all reproduced within rounding. Paper values are typed in by hand (not read from the run).
"""

import _bootstrap  # noqa: F401
import argparse, sys
from summarize import summarize

# (topology or None, key, paper value, tolerance as half-unit of the last printed digit)
PAPER = [
    ("ring", "nash_gap_t1_gate", 0.963, 0.0005),
    ("ring", "nash_gap_T_gate", 0.0092, 0.00005),
    ("star", "nash_gap_t1_gate", 0.988, 0.0005),
    ("star", "nash_gap_T_gate", 0.0139, 0.00005),
    ("ring", "pure_ne_runs_gate", 80, 0),
    ("star", "pure_ne_runs_gate", 76, 0),
    ("ring", "est_err_T_gate", 6.6e-6, 0.05e-6),
    ("star", "est_err_T_gate", 3.2e-5, 0.05e-5),
    ("ring", "welfare_t1_gate", 0.17, 0.005),
    ("ring", "welfare_T_gate", 1.00, 0.005),
    ("star", "welfare_t1_gate", 0.16, 0.005),
    ("star", "welfare_T_gate", 0.98, 0.005),
    (None, "welfare_T_none_both", 0.44, 0.005),
    ("ring", "T_f_mean", 64.5, 0.05),
    ("ring", "T_f_sd", 24.6, 0.05),
    ("star", "T_f_mean", 63.8, 0.05),
    ("star", "T_f_sd", 22.9, 0.05),
    ("ring", "T_a_mean", 77, 0.5),
    ("ring", "T_a_sd", 33, 0.5),
    ("star", "T_a_mean", 114, 0.5),
    ("star", "T_a_sd", 46, 0.5),
    ("ring", "prop2_triples", 1974585, 0),
    ("star", "prop2_triples", 1974865, 0),
    ("ring", "prop2_violations", 0, 0),
    ("star", "prop2_violations", 0, 0),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    S = summarize(a.out)
    if (S["meta"]["runs"], S["meta"]["T"]) != (20, 5000):
        sys.exit("verify_paper_numbers needs the full setting: --runs 20 --T 5000")
    bad = 0
    for tp, key, val, tol in PAPER:
        got = S[key] if tp is None else S[tp][key]
        ok = abs(got - val) <= tol + 1e-12
        bad += not ok
        print(
            f"{'PASS' if ok else 'FAIL'}  {tp or 'both':5s} {key:22s} paper={val:<12g} run={got:<12.6g}"
        )
    print("\nALL REPRODUCED" if not bad else f"\n{bad} MISMATCH(ES)")
    sys.exit(1 if bad else 0)
