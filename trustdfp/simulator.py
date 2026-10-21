"""Simulator for trust-filtered decentralized fictitious play (DFP).

Implements Algorithm 1 of the paper for N legitimate agents that play a repeated finite game while one
malicious agent injects false data on every communication channel.

Time index: the loop counter ``t`` of :func:`dfp_step` runs 1, 2, ..., so it equals the paper's t + 1.
The trust threshold is therefore xi_t = xi0 * sqrt(t) here, i.e. xi0 * sqrt(t_paper + 1) in the paper.

Four attack types (``attack_type``):
    collision     pushes every receiver's belief towards one target
    byzantine     sends near-uniform beliefs while attacking
    oscillating   alternates the pushed target every round
    equivocation  sends a different false belief to each receiver

Networks (``topology``): "ring" (each agent hears one predecessor) and "star" (agent 0 is the hub).
The game itself (utilities) is supplied by the caller, see ``trustdfp.potential_game``.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

XI_MODE = 'sqrt'   # 'sqrt': xi_t = xi0*sqrt(t) (used in the paper); 'lil': xi0*sqrt(t*ln ln(t+16))
ATTACK_TYPES = ["collision", "byzantine", "oscillating", "equivocation"]

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

    # ── Stress-test parameter (neutral by default) ──────────────────────────
    content_noise_scale: float = 1.0       # multiplies every attack type's own
                                            # fabricated-content noise std; 1.0 = default


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
    beta_gap_hist: list = field(default_factory=list)  # trust-gap diagnostic (agent `diag_agent`)
    xi_hist: list = field(default_factory=list)
    classification_round: Optional[int] = None
    leak_flags: list = field(default_factory=list)      # bool per attack-round: did content leak in
    diag_agent: int = 0                                 # agent whose trust gap is logged (set by the caller)
    mal_fab: Dict[int, np.ndarray] = field(default_factory=dict)  # fabricated content per receiver


# ──────────────────────────────────────────────────────────────────────────────
# NETWORK  (legitimate agents; one malicious agent reaches every legitimate agent)
# ──────────────────────────────────────────────────────────────────────────────

def legit_in_neighbors(N: int, topology: str) -> List[List[int]]:
    """
    Legitimate in-neighbor sets N^L_i for every legitimate agent i.
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


def closest_ne_distance(f_snap, N, K):
    # min_{sigma in perms(K)} sum_i ||f_i - e_{sigma(i)}|| , solved as a linear
    # assignment problem (Hungarian algorithm) instead of enumerating all K!
    # permutations -- identical result, but O(N^3) instead of O(K!), which
    # matters since this is called once per timestep per seed when plotting.
    eye = np.eye(K)
    cost = np.linalg.norm(f_snap[:, None, :] - eye[None, :, :], axis=2)  # (N,K)
    row_ind, col_ind = linear_sum_assignment(cost)
    return cost[row_ind, col_ind].sum() / N


# ──────────────────────────────────────────────────────────────────────────────
# INITIAL STATE
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

# ──────────────────────────────────────────────────────────────────────────────
# MALICIOUS ATTACK MODEL
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
    ns = game.content_noise_scale  # 1.0 = default

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
        # while attacking (by construction); while quiet, drift
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
# CORE STEP: one round of Algorithm 1
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

        # ── weights: equal-split over Nb_i(t) ∪ {i}  ───────────────────
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
