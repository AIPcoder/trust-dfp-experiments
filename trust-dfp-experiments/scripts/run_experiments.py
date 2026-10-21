"""Main numerical experiment: trust-filtered DFP vs. no defense.

2 topologies (ring, star) x 4 attack types x 2 defenses ("none", "trust_only") x R seeds.
Writes <out>/runs.pkl. Full paper setting: --runs 20 --T 5000 (about 5 min on 4 cores).
Seeds are 1..R, so results are bit-for-bit repeatable on the same library versions.
"""
import _bootstrap  # noqa: F401
import argparse, os, pickle, time
from multiprocessing import Pool
import numpy as np
from trustdfp import potential_game as p

base = p.base
N = K = 5
CFG = {}

def eta_of(topo):  # min positive post-classification weight (Prop. 2 constant)
    return 0.5 if topo == "ring" else 1.0 / N

def work(args):
    topo, at, cfg, r = args
    T = CFG["T"]
    g, d, _ = p.build_potential_game(N=N, K=K, T=T, seed=r, defense_mode=cfg, attack_type=at,
                                     topology=topo, xi0=1.4, attack_prob=0.35, E_L=0.85, mu_attack=0.30)
    rng = np.random.default_rng(r)
    nb = base.legit_in_neighbors(N, topo)
    st = base.initialize_state(g, rng)
    st.diag_agent = next(i for i in range(N) if len(nb[i]) == 1)
    Vh = []
    for t in range(1, T + 1):
        st = base.dfp_step(t, st, g, nb, rng)
        if cfg == "trust_only":
            Vh.append(st.v.copy())
    hist = np.array([st.classification_hist[i] for i in range(N)]).all(axis=0)
    TF = None
    for t in range(T):
        if hist[t:].all():
            TF = t + 1
            break
    A = np.array(st.history_actions)
    F = np.array(st.history_f)
    D = np.array([base.closest_ne_distance(f, N, K) for f in F])
    E = np.array(st.history_est_error)
    opt = sum(1.0 / x for x in d)
    W = np.array([p.true_u(a, d).sum() for a in A]) / opt
    Ph = np.array([p.phi(a, d) for a in A]) / opt
    NE = np.array([len(set(a.tolist())) == N for a in A])
    out = dict(E=E, D=D, W=W, Ph=Ph, NE=NE, TF=TF, A=A.astype(np.int8), d=d)
    if cfg == "trust_only" and TF is not None and TF < T:   # Proposition 2 check
        th = 1 - eta_of(topo) ** (N - 1); C = 1 / th; rb = th ** (1 / (N - 1))
        Cp = (2 * C / (1 - rb)) * (1 + 1 / (np.e * rb * np.log(1 / rb)))
        cnt = 0; mx = 0.0; viol = 0
        for k in range(N):
            ys = np.array([np.abs(Vh[t][:, k, :] - F[t][k][None, :]).max() for t in range(T)])
            y0 = ys[TF - 1]
            for t in range(TF, T + 1):
                b = C * rb ** (t - TF) * y0 + Cp / t
                cnt += 1; ratio = ys[t - 1] / b; mx = max(mx, ratio); viol += ratio > 1
        out.update(chk_n=cnt, chk_max=mx, chk_viol=viol)
    return (topo, at, cfg, r), out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=20, help="seeds per cell (paper: 20)")
    ap.add_argument("--T", type=int, default=5000, help="rounds (paper: 5000)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="worker processes")
    ap.add_argument("--out", default="results", help="output directory")
    a = ap.parse_args()
    CFG["T"] = a.T
    os.makedirs(a.out, exist_ok=True)
    jobs = [(t, x, c, r) for t in ("ring", "star") for x in base.ATTACK_TYPES
            for c in ("none", "trust_only") for r in range(1, a.runs + 1)]
    t0 = time.time()
    with Pool(a.jobs, initializer=_init, initargs=(a.T,)) as pool:
        res = dict(pool.imap_unordered(work, jobs, chunksize=2))
    meta = dict(runs=a.runs, T=a.T)
    with open(os.path.join(a.out, "runs.pkl"), "wb") as f:
        pickle.dump(dict(meta=meta, res=res), f)
    print(f"done: {len(jobs)} simulations in {time.time()-t0:.0f}s -> {a.out}/runs.pkl")

def _init(T):
    CFG["T"] = T

if __name__ == "__main__":
    main()
