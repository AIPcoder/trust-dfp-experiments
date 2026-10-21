"""
Decentralized Fictitious Play v8 — Trust-Filtered DFP, four attack types
══════════════════════════════════════════════════════════════════════════════
Faithful implementation of Algorithm 1 from `trust_fp.tex` ("Trustworthy
Decentralized Fictitious Play in Potential Games with Malicious Interference"),
extended with the four attack vectors of §3.1:

    A  Collision-inducing        (baseline / prior iterations)
    B  Byzantine uniform         (classical Byzantine fault model)
    C  Oscillating
    D  Byzantine equivocation (per-receiver splitting)             <-- NEW in v8

v8 adds exactly what the scoping discussion asked for, and nothing else:

  * Type D — the attacker maintains a PER-RECEIVER fabricated state instead
    of one shared broadcast matrix, so no two legitimate agents see the same
    fabricated content on the same round. This is invisible to the trust
    gate as specified, because beta_il(t) is accumulated per-channel and
    never cross-checked against what the same sender told a different
    receiver.

NOTE ON REMOVED TYPE C (Lead-lag / trajectory-tracking poisoning): an earlier
iteration of this file included a fifth attack type that pushed content ahead
of a running mean of the true legitimate v^l_k matrices. It was removed from
the taxonomy: its target is not stationary and not periodically stationary
(unlike the four types kept here), so it does not fit the same convergence
argument — it has no fixed point to plateau at, which made it structurally
hard to integrate into a single logically consistent account of §3.3's
findings. Its removal is a scoping decision, not a correctness finding about
the four remaining types.

Everything else (network model, trust gate, Algorithm 1 update rule, target-
assignment game, weight rule) is implemented exactly as stated in §1-§3 of
the paper, so that Table 1 / Figure 1 / Figure 2 can be regenerated from this
file alone.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Tuple
from itertools import permutations

# ──────────────────────────────────────────────────────────────────────────────
# GAME / CONFIG  (§3.1)
# ──────────────────────────────────────────────────────────────────────────────

XI_MODE = 'sqrt'   # 'lil': xi_t = xi0*sqrt(t*ln ln(t+16))  (only change vs dfp_v8.py)
ATTACK_TYPES = ["collision", "byzantine", "oscillating", "equivocation"]
ATTACK_LABEL = {
    "collision": "A: Collision-inducing",
    "byzantine": "B: Byzantine uniform",
    "oscillating": "C: Oscillating",
    "equivocation": "D: Equivocation",
}
ATTACK_COLOR = {
    "collision": "tab:blue",
    "byzantine": "tab:red",
    "oscillating": "tab:green",
    "equivocation": "tab:purple",
}


@dataclass
class DFPGame:
    N: int = 5
    K: int = 5
    utility_fn: Callable = None
    T: int = 500
    seed: int = 42
    warm_start_mode: str = "random_target"
    topology: str = "ring"                 # "ring" | "star"

    # ── Adversarial config ──────────────────────────────────────────────────
    defense_mode: str = "trust_only"       # "none" | "trust_only"
    E_L: float = 0.85
    mu_attack: float = 0.30
    sigma_trust: float = 0.08
    attack_prob: float = 0.35
    attack_type: str = "collision"
    bias_target: int = 0

    # ── Trust-gate params (Algorithm 1, xi_t = xi0*sqrt(t)) ─────────────────
    xi0: float = 1.4

    # ── Stress-test params, both neutral by default (§3.1 behavior unchanged
    # unless explicitly set) ─────────────────────────────────────────────────
    content_noise_scale: float = 1.0       # multiplies every attack type's own
                                            # fabricated-content noise std; 1.0
                                            # reproduces §3.1 exactly
    distance_compression: float = 0.0      # compresses each agent's estimated
                                            # target distances toward their mean
                                            # at game-build time; 0.0 = untouched


@dataclass
class DFPState:
    f: np.ndarray                          # (N,K) empirical frequencies
    v: np.ndarray                          # (N,N,K) local copies v^i_k
    actions: np.ndarray
    history_actions: list = field(default_factory=list)
    history_f: list = field(default_factory=list)
    history_est_error: list = field(default_factory=list)
    history_welfare: list = field(default_factory=list)
    history_attack: list = field(default_factory=list)
    beta: Dict[Tuple[int, int], float] = field(default_factory=dict)
    Nb: Dict[int, set] = field(default_factory=dict)
    classification_hist: Dict[int, list] = field(default_factory=dict)
    beta_gap_hist: list = field(default_factory=list)  # agent-0 diagnostic
    xi_hist: list = field(default_factory=list)
    classification_round: Optional[int] = None
    leak_flags: list = field(default_factory=list)      # bool per attack-round: did content leak in


# ──────────────────────────────────────────────────────────────────────────────
# NETWORK  — fixed ring among legit agents; single malicious agent, broadcast reach
# ──────────────────────────────────────────────────────────────────────────────

def legit_in_neighbors(N: int, topology: str) -> List[List[int]]:
    """
    Legitimate in-neighbor sets N^L_i for every legitimate agent i, per §1.2/§3.1.
    ring:  each agent observes exactly one predecessor, agent (i-1) mod N.
    star:  one hub (agent 0) is observed by, and observes, every other agent;
           every non-hub agent observes only the hub.
    """
    if topology == "ring":
        return [[(i - 1) % N] for i in range(N)]
    elif topology == "star":
        hub = 0
        out = []
        for i in range(N):
            if i == hub:
                out.append([j for j in range(N) if j != hub])
            else:
                out.append([hub])
        return out
    else:
        raise ValueError(f"Unknown topology: {topology}")

MAL_ID = -1  # sentinel id for the single malicious agent (not in L)


# ──────────────────────────────────────────────────────────────────────────────
# TARGET-ASSIGNMENT GAME  (§3.1, [5, 6])
# ──────────────────────────────────────────────────────────────────────────────

def build_target_assignment_game(N=5, K=5, T=500, signal_noise=0.5, signal_steps=10,
                                  seed=42, distance_compression=0.0, **kwargs):
    rng = np.random.default_rng(seed)
    agent_positions = rng.normal(0, np.sqrt(0.1), size=(N, 2))
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    target_positions = np.column_stack([np.cos(angles), np.sin(angles)])

    true_distances = np.array([
        [np.linalg.norm(agent_positions[i] - target_positions[k]) for k in range(K)]
        for i in range(N)
    ])
    signals = rng.normal(
        loc=target_positions[np.newaxis, np.newaxis, :, :],
        scale=signal_noise, size=(N, signal_steps, K, 2)
    )
    estimated_target_pos = signals.mean(axis=1)
    estimated_distances = np.array([
        [np.linalg.norm(agent_positions[i] - estimated_target_pos[i, k]) for k in range(K)]
        for i in range(N)
    ])
    if distance_compression > 0.0:
        # Stress-test only (§5): compress each agent's own five target distances
        # toward their mean, shrinking the utility margin between competing
        # targets. distance_compression=0.0 (default) leaves estimated_distances
        # bit-identical to §3.1's original game.
        mean_dist = estimated_distances.mean(axis=1, keepdims=True)
        estimated_distances = (1 - distance_compression) * estimated_distances + distance_compression * mean_dist
    dist_floor = signal_noise / np.sqrt(signal_steps)

    def utility_fn(agent_i, action_i, v):
        k = action_i
        d = max(estimated_distances[agent_i, k], dist_floor)
        prob_exclusive = 1.0
        for j in range(N):
            if j != agent_i:
                prob_exclusive *= float(1.0 - v[agent_i, j, k])
        return float(prob_exclusive / d)

    nash_eq = np.array(max(
        permutations(range(K)),
        key=lambda p: sum(1.0 / true_distances[i, p[i]] for i in range(N))
    ))

    game = DFPGame(N=N, K=K, utility_fn=utility_fn, T=T, seed=seed,
                    distance_compression=distance_compression, **kwargs)
    return game, nash_eq


def closest_ne_distance(f_snap, N, K):
    # min_{sigma in perms(K)} sum_i ||f_i - e_{sigma(i)}|| , solved as a linear
    # assignment problem (Hungarian algorithm) instead of enumerating all K!
    # permutations -- identical result, but O(N^3) instead of O(K!), which
    # matters since this is called once per timestep per seed when plotting.
    eye = np.eye(K)
    cost = np.linalg.norm(f_snap[:, None, :] - eye[None, :, :], axis=2)  # (N,K)
    row_ind, col_ind = linear_sum_assignment(cost)
    return cost[row_ind, col_ind].sum() / N


def unique_assignment(actions):
    return len(set(actions.tolist())) == len(actions)


# ──────────────────────────────────────────────────────────────────────────────
# WARM START
# ──────────────────────────────────────────────────────────────────────────────

def warm_start_concentration(N: int) -> float:
    return 1.0 - 1.0 / np.sqrt(N + 1)


def initialize_state(game: DFPGame, rng: np.random.Generator) -> DFPState:
    N, K = game.N, game.K
    c = warm_start_concentration(N)
    first_actions = rng.permutation(K)[:N]
    f = np.ones((N, K)) / K
    for i in range(N):
        f[i] = (1.0 - c) / (K - 1)
        f[i, first_actions[i]] = c

    v = np.ones((N, N, K)) / K
    for i in range(N):
        v[i, i] = f[i].copy()

    state = DFPState(f=f, v=v, actions=first_actions)
    for i in range(N):
        state.classification_hist[i] = []
    # malicious fabricated content: per-receiver dict of (N,K) matrices
    state.mal_fab = {}
    base = np.ones((N, K)) / K
    for i in range(N):
        state.mal_fab[i] = base.copy()
    return state


# ──────────────────────────────────────────────────────────────────────────────
# MALICIOUS ATTACK MODEL  — four attack types
# ──────────────────────────────────────────────────────────────────────────────

def update_malicious_content(state: DFPState, game: DFPGame, atk: bool, rng) -> None:
    """
    Updates state.mal_fab[i] for every legitimate receiver i (a full (N,K)
    matrix of fabricated player-belief rows), following the attack type.

    Types A, B, C: broadcast the SAME content to every receiver.
    Type D: content DIFFERS per receiver (equivocation) -- this is the one
            type where state.mal_fab[i] is not identical across i.
    """
    N, K = game.N, game.K
    ns = game.content_noise_scale  # 1.0 reproduces §3.1 exactly

    if game.attack_type == "collision":
        if atk:
            safe_target = rng.integers(0, K)
            while safe_target == game.bias_target and K > 1:
                safe_target = rng.integers(0, K)
            row = np.full(K, (1.0 - 0.85) / (K - 1))
            row[safe_target] = 0.85
            content = np.tile(row, (N, 1)) + rng.normal(0, 0.02 * ns, size=(N, K))
        else:
            content = state.mal_fab[0] + rng.normal(0, 0.008 * ns, size=(N, K))
        content = np.clip(content, 1e-6, None)
        content /= content.sum(axis=1, keepdims=True)
        for i in range(N):
            state.mal_fab[i] = content

    elif game.attack_type == "byzantine":
        if atk:
            content = np.full((N, K), 1.0 / K) + rng.normal(0, 0.01 * ns, size=(N, K))
        else:
            content = state.mal_fab[0] + rng.normal(0, 0.008 * ns, size=(N, K))
        content = np.clip(content, 1e-6, None)
        content /= content.sum(axis=1, keepdims=True)
        for i in range(N):
            state.mal_fab[i] = content

    elif game.attack_type == "oscillating":
        t_global = len(state.history_actions) + 1
        pole = int(t_global % 2 == 0)
        target_k = pole * (K - 1)
        if atk:
            row = np.full(K, (1.0 - 0.90) / (K - 1))
            row[target_k] = 0.90
            content = np.tile(row, (N, 1)) + rng.normal(0, 0.02 * ns, size=(N, K))
        else:
            row = np.full(K, (1.0 - 0.60) / (K - 1))
            row[target_k] = 0.60
            target = np.tile(row, (N, 1))
            content = state.mal_fab[0] + 0.02 * (target - state.mal_fab[0]) + rng.normal(0, 0.005 * ns, size=(N, K))
        content = np.clip(content, 1e-6, None)
        content /= content.sum(axis=1, keepdims=True)
        for i in range(N):
            state.mal_fab[i] = content

    elif game.attack_type == "equivocation":
        # Type D: PER-RECEIVER fabricated content. Receiver i is told that
        # every player concentrates on target (i mod K); no two receivers
        # see the same message. Only high-concentration content is sent
        # while attacking (by construction, per §3.3); while quiet, drift
        # gently and independently per receiver so consecutive attack
        # rounds still look locally plausible to each receiver individually.
        for i in range(N):
            target_k = i % K
            if atk:
                row = np.full(K, (1.0 - 0.85) / (K - 1))
                row[target_k] = 0.85
                content = np.tile(row, (N, 1)) + rng.normal(0, 0.02 * ns, size=(N, K))
            else:
                row = np.full(K, (1.0 - 0.55) / (K - 1))
                row[target_k] = 0.55
                target = np.tile(row, (N, 1))
                content = state.mal_fab[i] + 0.02 * (target - state.mal_fab[i]) + rng.normal(0, 0.006 * ns, size=(N, K))
            content = np.clip(content, 1e-6, None)
            content /= content.sum(axis=1, keepdims=True)
            state.mal_fab[i] = content
    else:
        raise ValueError(f"Unknown attack_type: {game.attack_type}")


# ──────────────────────────────────────────────────────────────────────────────
# CORE STEP — Algorithm 1
# ──────────────────────────────────────────────────────────────────────────────

def dfp_step(t: int, state: DFPState, game: DFPGame, neighbors: List[List[int]], rng) -> DFPState:
    N, K = game.N, game.K
    v_shared = state.v.copy()

    atk = bool(rng.random() < game.attack_prob)
    state.history_attack.append(atk)
    update_malicious_content(state, game, atk, rng)

    new_v = np.zeros_like(state.v)
    leaked_this_round = False

    for i in range(N):
        legit_ls = neighbors[i]              # legitimate in-neighbor(s): 1 on ring, up to N-1 on star hub
        full_in = list(legit_ls) + [MAL_ID]

        # ── receive content + trust observation ────────────────────────────
        received, trust_obs = {}, {}
        for l in legit_ls:
            received[l] = v_shared[l]
            trust_obs[l] = float(np.clip(rng.normal(game.E_L, game.sigma_trust), 0, 1))
        received[MAL_ID] = state.mal_fab[i]
        mean_m = game.mu_attack if atk else game.E_L
        trust_obs[MAL_ID] = float(np.clip(rng.normal(mean_m, game.sigma_trust), 0, 1))

        # ── accumulate trust beta_il(t) ─────────────────────────────────────
        for l in full_in:
            key = (i, l)
            state.beta[key] = state.beta.get(key, 0.0) + trust_obs[l]

        if i == state.diag_agent:
            diag_l = legit_ls[0]
            state.beta_gap_hist.append(state.beta[(i, diag_l)] - state.beta[(i, MAL_ID)])

        # ── trust gate: Nb_i(t) = {l in N_i : beta_ili(t) - beta_il(t) <= xi_t} ──
        xi_t = game.xi0 * np.sqrt(max(t, 1) * (np.log(np.log(max(t, 1) + 16.0)) if XI_MODE == 'lil' else 1.0))
        state.xi_hist.append(xi_t)
        if game.defense_mode == "trust_only":
            leader = max(full_in, key=lambda l: state.beta[(i, l)])
            Nb_i = {l for l in full_in if state.beta[(i, leader)] - state.beta[(i, l)] <= xi_t}
        else:
            Nb_i = set(full_in)
        state.Nb[i] = Nb_i

        # classification: correct iff malicious excluded and every legit in-neighbor kept
        correct = (MAL_ID not in Nb_i) and all(l in Nb_i for l in legit_ls)
        state.classification_hist[i].append(correct)

        if atk and (MAL_ID in Nb_i):
            leaked_this_round = True

        # ── weights: equal-split over Nb_i(t) ∪ {i} (§3.1) ───────────────────
        n = len(Nb_i)
        w_other = 1.0 / (1 + n)
        w_self = w_other

        for k in range(N):
            if k == i:
                continue
            acc = w_self * v_shared[i, k]
            for l in Nb_i:
                acc += w_other * received[l][k]
            new_v[i, k] = acc
        new_v[i, i] = state.f[i].copy()

    state.leak_flags.append(leaked_this_round)
    state.v = new_v

    # ── best response ────────────────────────────────────────────────────────
    new_actions = np.zeros(N, dtype=int)
    for i in range(N):
        best_a, best_u = -1, -np.inf
        for a in range(K):
            u = float(game.utility_fn(i, a, state.v))
            if u > best_u:
                best_u, best_a = u, a
        new_actions[i] = best_a

    # ── diagnostics ─────────────────────────────────────────────────────────
    err, cnt = 0.0, 0
    for ii in range(N):
        for jj in range(N):
            if ii != jj:
                err += np.linalg.norm(state.v[ii, jj] - state.f[jj])
                cnt += 1
    state.history_est_error.append(err / cnt if cnt else 0.0)
    welfare = sum(float(game.utility_fn(i, new_actions[i], state.v)) for i in range(N))
    state.history_welfare.append(welfare)

    # ── frequency update f_i,t = ((t-1)/t) f_i,t-1 + (1/t) 1_{a_i,t} ────────
    for i in range(N):
        one_hot = np.zeros(K)
        one_hot[new_actions[i]] = 1.0
        state.f[i] = ((t - 1) / t) * state.f[i] + (1.0 / t) * one_hot
        state.v[i, i] = state.f[i].copy()

    state.actions = new_actions
    state.history_actions.append(new_actions.copy())
    state.history_f.append(state.f.copy())
    return state


def run_dfp(game: DFPGame, seed: int) -> DFPState:
    rng = np.random.default_rng(seed)
    neighbors = legit_in_neighbors(game.N, game.topology)
    state = initialize_state(game, rng)
    # diagnostic agent: any agent with exactly one legit in-neighbor, so the
    # single trust-gap panel is comparable across ring and star (star's hub
    # itself has N-1 legit in-neighbors and isn't a fair single-channel view)
    state.diag_agent = next(i for i in range(game.N) if len(neighbors[i]) == 1)
    for t in range(1, game.T + 1):
        state = dfp_step(t, state, game, neighbors, rng)

    # classification round: first t at which every agent's classification
    # is correct AND remains so for the rest of the run
    N = game.N
    hist = np.array([state.classification_hist[i] for i in range(N)])   # (N,T) bool
    all_correct = hist.all(axis=0)                                       # (T,)
    cls_round = None
    for t in range(len(all_correct)):
        if all_correct[t:].all():
            cls_round = t + 1
            break
    state.classification_round = cls_round
    return state


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(state: DFPState, game: DFPGame) -> Dict[str, float]:
    N, K, T = game.N, game.K, game.T
    attack_rounds = np.array(state.history_attack)
    leak = np.array(state.leak_flags)
    leak_rate = leak[attack_rounds].mean() if attack_rounds.sum() > 0 else 0.0
    final_est_err = state.history_est_error[-1]
    ne_dist = closest_ne_distance(state.f, N, K)
    welfare = state.history_welfare[-1]
    return dict(leak_rate=leak_rate, final_est_error=final_est_err,
                ne_distance=ne_dist, welfare=welfare,
                unique_ne=unique_assignment(state.actions),
                classification_round=state.classification_round)


# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENT — R=20 seeds, all 5 attack types x {none, trust_only}, T=500
# ──────────────────────────────────────────────────────────────────────────────

TOPOLOGIES = ["ring", "star"]


def run_experiment(R=20, T=500, N=5, K=5, topologies=TOPOLOGIES):
    results = {}   # results[topology][attack_type][config] = list of DFPState (per seed)
    metrics = {}   # metrics[topology][attack_type][config] = list of dict (per seed)
    for topo in topologies:
        results[topo] = {}
        metrics[topo] = {}
        for atype in ATTACK_TYPES:
            results[topo][atype] = {"none": [], "trust_only": []}
            metrics[topo][atype] = {"none": [], "trust_only": []}
            for cfg in ["none", "trust_only"]:
                for r in range(1, R + 1):
                    game, nash_eq = build_target_assignment_game(
                        N=N, K=K, T=T, seed=r, defense_mode=cfg, attack_type=atype,
                        topology=topo, xi0=1.4, attack_prob=0.35, E_L=0.85, mu_attack=0.30)
                    state = run_dfp(game, seed=r)
                    results[topo][atype][cfg].append(state)
                    metrics[topo][atype][cfg].append(compute_metrics(state, game))
            print(f"  done: {topo}/{atype}")
    return results, metrics


def summarize_table(metrics, topologies=TOPOLOGIES):
    print(f"{'Topology':9s}{'Attack':28s}{'Config':12s}{'Leak rate':>16s}{'Final est.err':>18s}{'NE dist':>16s}{'Welfare':>14s}")
    rows = []
    for topo in topologies:
        for atype in ATTACK_TYPES:
            for cfg in ["none", "trust_only"]:
                ms = metrics[topo][atype][cfg]
                leak = np.array([m["leak_rate"] for m in ms])
                err = np.array([m["final_est_error"] for m in ms])
                ne = np.array([m["ne_distance"] for m in ms])
                wel = np.array([m["welfare"] for m in ms])
                uniq = np.mean([m["unique_ne"] for m in ms])
                print(f"{topo:9s}{ATTACK_LABEL[atype]:28s}{cfg:12s}"
                      f"{leak.mean():>8.3f}±{leak.std():<6.3f}"
                      f"{err.mean():>10.5f}±{err.std():<7.5f}"
                      f"{ne.mean():>7.3f}±{ne.std():<6.3f}"
                      f"{wel.mean():>7.2f}±{wel.std():<6.2f}   unique_NE={uniq:.2f}")
                rows.append((topo, atype, cfg, leak.mean(), leak.std(), err.mean(), err.std(),
                             ne.mean(), ne.std(), wel.mean(), wel.std(), uniq))
    return rows


def rows_to_latex(rows):
    lines = []
    label_map = {"collision": "A: Collision", "byzantine": "B: Byzantine",
                 "oscillating": "C: Oscillating", "equivocation": "D: Equivocation"}
    cfg_label = {"none": "none", "trust_only": "trust only"}
    for (topo, atype, cfg, lm, ls, em, es, nm, ns, wm, ws, uniq) in rows:
        name = label_map[atype] if cfg == "none" else ""
        lines.append(f"{topo} & {name} & {cfg_label[cfg]} & {lm:.3f} $\\pm$ {ls:.3f} & "
                     f"{em:.5f} $\\pm$ {es:.5f} & {nm:.3f} $\\pm$ {ns:.3f} & "
                     f"{wm:.2f} $\\pm$ {ws:.2f} \\\\")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def _ax_style(ax):
    ax.set_facecolor("white")
    ax.tick_params(colors="black", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#bbbbbb")
    ax.grid(True, alpha=0.35, color="#dddddd", which="both")


def _savefig(fig, fname):
    out = f"/mnt/user-data/outputs/{fname}"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out}")
    return out


def plot_figure1(results, N, K, T, topologies=TOPOLOGIES):
    n_topo = len(topologies)
    fig, axes = plt.subplots(n_topo, 2, figsize=(13, 4.3 * n_topo), squeeze=False)
    fig.patch.set_facecolor("white")

    for row, topo in enumerate(topologies):
        ax = axes[row][0]; _ax_style(ax)
        for atype in ATTACK_TYPES:
            c = ATTACK_COLOR[atype]
            for cfg, ls, alpha in [("trust_only", "-", 1.0), ("none", ":", 0.6)]:
                errs = np.array([s.history_est_error for s in results[topo][atype][cfg]])
                m = np.clip(errs.mean(axis=0), 1e-6, None)
                label = ATTACK_LABEL[atype] if cfg == "trust_only" else None
                ax.plot(np.arange(1, T + 1), m, color=c, ls=ls, lw=2, alpha=alpha, label=label)
        ax.set_yscale("log"); ax.set_xlabel("Time step $t$"); ax.set_ylabel("Avg. estimation error")
        ax.set_title(f"Estimation Error Convergence ({topo}, R=20)")
        ax.legend(fontsize=7)

        ax = axes[row][1]; _ax_style(ax)
        for atype in ATTACK_TYPES:
            c = ATTACK_COLOR[atype]
            for cfg, ls, alpha in [("trust_only", "-", 1.0), ("none", ":", 0.6)]:
                dists = np.array([[closest_ne_distance(f, N, K) for f in s.history_f]
                                   for s in results[topo][atype][cfg]])
                m = np.clip(dists.mean(axis=0), 1e-6, None)
                label = ATTACK_LABEL[atype] if cfg == "trust_only" else None
                ax.plot(np.arange(1, T + 1), m, color=c, ls=ls, lw=2, alpha=alpha, label=label)
        ax.set_yscale("log"); ax.set_xlabel("Time step $t$"); ax.set_ylabel("Distance to closest NE")
        ax.set_title(f"Convergence to Nash Equilibrium ({topo}, R=20)")
        ax.legend(fontsize=7)

    fig.suptitle("Solid: trust only | Dotted: none  |  mean over R=20 seeds, all four attack types, "
                 "ring (top) vs. star (bottom)" if n_topo > 1 else
                 "Solid: trust only | Dotted: none  |  mean over R=20 seeds, all four attack types",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig(fig, "dfp_v8_figure1_estimation_ne.png")


def plot_figure2(N, K, T, seed=42, attack_subset=None, fname="dfp_v8_figure2_attack_panels.png",
                  suptitle_extra="", topology="ring"):
    attack_subset = attack_subset if attack_subset is not None else ATTACK_TYPES
    n_rows = len(attack_subset)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows), squeeze=False)
    fig.patch.set_facecolor("white")
    clrs = [plt.cm.tab10(i) for i in range(N)]
    for row, atype in enumerate(attack_subset):
        beta_gap_state = None
        for col, cfg in enumerate(["none", "trust_only"]):
            game, _ = build_target_assignment_game(
                N=N, K=K, T=T, seed=seed, defense_mode=cfg, attack_type=atype,
                topology=topology, xi0=1.4, attack_prob=0.35, E_L=0.85, mu_attack=0.30)
            state = run_dfp(game, seed=seed)
            ax = axes[row, col]; _ax_style(ax)
            acts = np.array(state.history_actions)
            ts = np.arange(1, T + 1)
            for i in range(N):
                ax.step(ts, acts[:, i] + 1, where="post", color=clrs[i], lw=1.4)
            if state.classification_round is not None:
                ax.axvline(state.classification_round, color="black", ls=":", lw=1.2)
            ax.set_yticks(range(1, K + 1))
            if row == 0:
                ax.set_title("Actions — no defense" if col == 0 else "Actions — trust gate", fontsize=10)
            ax.set_ylabel(ATTACK_LABEL[atype], fontsize=9)
            ax.set_xlabel("Time step $t$")
            if col == 1:
                beta_gap_state = state

        ax = axes[row, 2]; _ax_style(ax)
        gap = np.array(beta_gap_state.beta_gap_hist)
        ts = np.arange(1, len(gap) + 1)
        ax.plot(ts, gap, color=ATTACK_COLOR[atype], lw=1.8,
                label=r"$\beta_{i,\bar l}(t)-\beta_{i,m}(t)$")
        xi_curve = 1.4 * np.sqrt(np.maximum(ts, 1))
        ax.plot(ts, xi_curve, color="black", ls="--", lw=1.3, label=r"$\xi_t$")
        if beta_gap_state.classification_round is not None:
            ax.axvline(beta_gap_state.classification_round, color="black", ls=":", lw=1.2,
                       label="class. round")
        if row == 0:
            ax.set_title("Trust-gate signal", fontsize=10)
        ax.set_xlabel("Time step $t$")
        ax.set_ylabel("Cumulative gap")
        ax.legend(fontsize=7)

    fig.suptitle(f"Per-attack-type illustration — {topology}, seed {seed}, T={T}{suptitle_extra}\n"
                 "Left: no defense | Middle: trust gate | Right: $\\beta$-gap signal driving classification",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _savefig(fig, fname)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pickle
    print("=" * 78)
    print("  Trust-Filtered DFP v8 — 4 attack types x {ring, star} x {none, trust_only}")
    print("=" * 78)
    R, T, N, K = 20, 500, 5, 5
    results, metrics = run_experiment(R=R, T=T, N=N, K=K, topologies=TOPOLOGIES)
    rows = summarize_table(metrics)
    print("\nLaTeX table rows:\n")
    print(rows_to_latex(rows))
    with open("/home/claude/work/metrics_v8.pkl", "wb") as f:
        pickle.dump(metrics, f)
    print("\nGenerating figures...")
    plot_figure1(results, N, K, T, topologies=TOPOLOGIES)
    plot_figure2(N, K, T, seed=42, attack_subset=["collision", "byzantine"],
                 fname="dfp_v8_panels_ab.png", suptitle_extra=" (Types A-B)", topology="ring")
    plot_figure2(N, K, T, seed=42, attack_subset=["oscillating", "equivocation"],
                 fname="dfp_v8_panels_cd.png", suptitle_extra=" (Types C-D)", topology="ring")
    plot_figure2(N, K, T, seed=42, attack_subset=["collision", "byzantine"],
                 fname="dfp_v8_panels_ab_star.png", suptitle_extra=" (Types A-B)", topology="star")
    plot_figure2(N, K, T, seed=42, attack_subset=["oscillating", "equivocation"],
                 fname="dfp_v8_panels_cd_star.png", suptitle_extra=" (Types C-D)", topology="star")
    print("Done.")
