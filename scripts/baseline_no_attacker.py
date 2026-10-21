"""Reference run: plain DFP on the legitimate subgraph, no malicious agent (explains the 76/80 star result)."""

import _bootstrap  # noqa: F401
import argparse
import numpy as np
from trustdfp import potential_game as p

base = p.base


def run_clean(topo, seed, T=3000, N=5, K=5):
    g, d, _ = p.build_potential_game(N=N, K=K, T=T, seed=seed)
    rng = np.random.default_rng(seed)
    nb = base.legit_in_neighbors(N, topo)
    st = base.initialize_state(g, rng)
    A = []
    for t in range(1, T + 1):
        vs = st.v.copy()
        new = np.zeros_like(st.v)
        for i in range(N):
            w = 1.0 / (1 + len(nb[i]))
            for k in range(N):
                if k == i:
                    continue
                new[i, k] = w * vs[i, k] + sum(w * vs[l, k] for l in nb[i])
            new[i, i] = st.f[i]
        st.v = new
        a = np.array(
            [max(range(K), key=lambda x: (g.utility_fn(i, x, st.v), -x)) for i in range(N)]
        )
        for i in range(N):
            oh = np.zeros(K)
            oh[a[i]] = 1
            st.f[i] = ((t - 1) / t) * st.f[i] + oh / t
            st.v[i, i] = st.f[i]
        A.append(a)
    return np.array(A), d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=80)
    ap.add_argument("--T", type=int, default=3000)
    a = ap.parse_args()
    for topo in ("ring", "star"):
        ok = sum(
            len(set(run_clean(topo, r, T=a.T)[0][-1].tolist())) == 5 for r in range(1, a.runs + 1)
        )
        print(
            f"plain DFP, no malicious agent, {topo}: pure NE at T={a.T} in {ok}/{a.runs} runs",
            flush=True,
        )
