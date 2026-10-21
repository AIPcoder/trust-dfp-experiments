"""Headline statistics from runs.pkl / gaps.npz -> <out>/summary.json (and printed).
Every number quoted in the paper's experiment section is computed here."""

import _bootstrap  # noqa: F401
import argparse, json, os, pickle
import numpy as np
from trustdfp import simulator as base

ATS = list(base.ATTACK_TYPES)


def summarize(out):
    D = pickle.load(open(os.path.join(out, "runs.pkl"), "rb"))
    R, meta = D["res"], D["meta"]
    runs, T = meta["runs"], meta["T"]
    RUNS = range(1, runs + 1)
    G = np.load(os.path.join(out, "gaps.npz"))
    stack = lambda tp, cfg, key: np.vstack(
        [np.array([R[(tp, a, cfg, r)][key] for r in RUNS], dtype=float) for a in ATS]
    )
    S = {"meta": meta}
    for tp in ("ring", "star"):
        s = {}
        g, gn = G[f"{tp}|trust_only"], G[f"{tp}|none"]
        s["nash_gap_t1_gate"], s["nash_gap_T_gate"], s["nash_gap_T_none"] = (
            g[:, 0].mean(),
            g[:, -1].mean(),
            gn[:, -1].mean(),
        )
        NE = stack(tp, "trust_only", "NE")[:, -1].astype(bool)
        s["pure_ne_runs_gate"], s["n_runs"] = int(NE.sum()), int(NE.size)
        s["pure_ne_runs_none"] = int(stack(tp, "none", "NE")[:, -1].sum())
        order = [(a, r) for a in ATS for r in RUNS]
        s["not_pure_runs"] = [
            dict(attack=order[i][0], seed=order[i][1], nash_gap_T=float(g[i, -1]))
            for i in np.where(~NE)[0]
        ]
        s["est_err_T_gate"] = stack(tp, "trust_only", "E")[:, -1].mean()
        en = np.array(
            [stack(tp, "none", "E")[j * runs : (j + 1) * runs, -1].mean() for j in range(len(ATS))]
        )
        s["est_err_T_none_min"], s["est_err_T_none_max"] = en.min(), en.max()
        s["welfare_t1_gate"], s["welfare_T_gate"], s["welfare_T_none"] = (
            stack(tp, "trust_only", "W")[:, 0].mean(),
            stack(tp, "trust_only", "W")[:, -1].mean(),
            stack(tp, "none", "W")[:, -1].mean(),
        )
        TF = np.array([R[(tp, a, "trust_only", r)]["TF"] for a in ATS for r in RUNS], dtype=float)
        s["T_f_mean"], s["T_f_sd"], s["T_f_missing"] = (
            float(np.nanmean(TF)),
            float(np.nanstd(TF)),
            int(np.isnan(TF).sum()),
        )
        Ta = []
        for row in stack(tp, "trust_only", "NE").astype(bool):
            if not row[-1]:
                continue  # runs that end at a symmetric mixed NE have no settling time
            idx = np.where(~row)[0]
            Ta.append(idx[-1] + 2 if len(idx) else 1)
        s["T_a_mean"], s["T_a_sd"], s["T_a_runs"] = float(np.mean(Ta)), float(np.std(Ta)), len(Ta)
        chk = [R[(tp, a, "trust_only", r)] for a in ATS for r in RUNS]
        s["prop2_triples"] = int(sum(c.get("chk_n", 0) for c in chk))
        s["prop2_violations"] = int(sum(c.get("chk_viol", 0) for c in chk))
        s["prop2_max_ratio"] = float(max(c.get("chk_max", 0) for c in chk))
        S[tp] = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in s.items()}
    S["welfare_T_none_both"] = float(
        np.mean([S["ring"]["welfare_T_none"], S["star"]["welfare_T_none"]])
    )
    json.dump(S, open(os.path.join(out, "summary.json"), "w"), indent=2)
    return S


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    S = summarize(a.out)
    for tp in ("ring", "star"):
        print(f"--- {tp}")
        for k, v in S[tp].items():
            print(f"  {k}: {v}")
    print("welfare_T_none_both:", S["welfare_T_none_both"])
