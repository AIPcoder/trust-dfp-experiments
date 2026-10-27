"""Experiment of the submitted paper: one malicious node, directed 5-player ring.

Two attacks (equivocation, oscillating) x two defenses (trust gate, none) x 100 seeds, T = 5000.
Dynamics come from trustdfp.simulator / trustdfp.potential_game (Algorithm 1, unchanged).
Writes <out>/m1_results.npz. Seeds are 1..100, so results are repeatable on the same library versions.
Runner structure adapted from the version supplied by S. Aydin (see README, "Credits").
"""

import _bootstrap  # noqa: F401
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from trustdfp import potential_game as dfp_pot
from trustdfp.metrics import gap_series

ATTACKS = ("equivocation", "oscillating")
MODES = ("trust_only", "none")
N_AGENTS = N_TARGETS = 5


def persistence_start(pure_ne):
    """First round (1-based) from which every round is a pure NE through the end; -1 if the last round is not one."""
    pure_ne = np.asarray(pure_ne, dtype=bool)
    if not pure_ne[-1]:
        return -1
    bad = np.flatnonzero(~pure_ne)
    return 1 if bad.size == 0 else int(bad[-1]) + 2


def run_one(spec):
    attack, mode, seed, steps = spec
    game, d, _ = dfp_pot.build_potential_game(
        N=N_AGENTS, K=N_TARGETS, T=steps, seed=seed, defense_mode=mode, attack_type=attack,
        topology="ring", xi0=1.4, attack_prob=0.35, E_L=0.85, mu_attack=0.30, sigma_trust=0.08,
    )
    state = dfp_pot.base.run_dfp(game, seed=seed)
    actions = np.asarray(state.history_actions, dtype=np.uint8)
    gap = gap_series(actions, d)
    error = np.asarray(state.history_est_error, dtype=np.float64)
    # With N = K, the pure NE are exactly the profiles in which all targets are distinct.
    pure_ne = np.asarray([len(set(a)) == N_AGENTS for a in actions], dtype=bool)
    return attack, mode, seed, gap, error, persistence_start(pure_ne), state.classification_round


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("results_m1"))
    opt = p.parse_args()
    jobs = [(a, m, s, opt.steps) for a in ATTACKS for m in MODES for s in range(1, opt.seeds + 1)]
    G = np.empty((len(ATTACKS), len(MODES), opt.seeds, opt.steps))
    E = np.empty_like(G)
    PN = np.full((len(ATTACKS), len(MODES), opt.seeds), -1, dtype=int)
    TF = PN.copy()
    with ProcessPoolExecutor(max_workers=opt.workers) as ex:
        for ix, (a, m, s, g, e, pn, tf) in enumerate(ex.map(run_one, jobs), 1):
            ai, mi, si = ATTACKS.index(a), MODES.index(m), s - 1
            G[ai, mi, si], E[ai, mi, si], PN[ai, mi, si] = g, e, pn
            TF[ai, mi, si] = -1 if tf is None else tf
            if ix % 25 == 0 or ix == len(jobs):
                print(f"Completed {ix}/{len(jobs)} runs", flush=True)
    opt.out.mkdir(parents=True, exist_ok=True)
    f = opt.out / "m1_results.npz"
    np.savez_compressed(
        f, gap=G, estimation_error=E, pure_ne_from=PN, classification_from=TF,
        attacks=np.array(ATTACKS), modes=np.array(MODES),
        seeds=np.arange(1, opt.seeds + 1), iterations=opt.steps,
    )
    print("Saved", f)
    for ai, a in enumerate(ATTACKS):
        for mi, m in enumerate(MODES):
            print(
                f"{a:13s} {m:12s} gap={G[ai, mi, :, -1].mean():.5f} error={E[ai, mi, :, -1].mean():.6g} "
                f"ends_in_pure_NE={np.sum(PN[ai, mi] > 0)}/{opt.seeds}", flush=True)


if __name__ == "__main__":
    main()
