# trust-dfp-experiments

Code for the numerical experiments in *Trustworthy Decentralized Fictitious Play in Potential Games under
Malicious Interference* (A. K. Albas and S. Aydın, ICIT 2027 work-in-progress paper).

Repository: https://github.com/AIPcoder/trust-dfp-experiments · Archived releases: concept DOI (all versions)
[10.5281/zenodo.23546047](https://doi.org/10.5281/zenodo.23546047); version 1.0.0 is
[10.5281/zenodo.23546048](https://doi.org/10.5281/zenodo.23546048).

The repository contains two studies. **They are different, and only the first is in the submitted paper.**

| | Setting | Scripts | Check |
| --- | --- | --- | --- |
| **Submitted paper** (Sec. IV, Fig. 1) | ring, 2 attacks (equivocation, oscillating), with/without trust gate, 100 seeds, T = 5000 | `run_m1.py`, `plot_m1.py` | `verify_submitted.py` |
| **Extended study** (not in the submitted paper) | ring and star, 4 attacks, 20 seeds, T = 5000, Proposition 2 check | `run_experiments.py` and the scripts after it | `verify_paper_numbers.py` |

## Reproducing the submitted paper

Requires Python 3.10 or newer.

```bash
git clone https://github.com/AIPcoder/trust-dfp-experiments.git
cd trust-dfp-experiments
pip install -r requirements.txt
python scripts/run_m1.py --seeds 100 --steps 5000 --workers 2 --out results_m1   # 400 simulations, about 12 min on 2 cores
python scripts/plot_m1.py --out results_m1        # fig_ring_100_horizontal.pdf / .png
python scripts/verify_submitted.py --out results_m1
```

With this code the 100-seed run gives the values printed in the paper: mean Nash gap at t = 5000 of
0.0047 (equivocation) and 0.0136 (oscillating) with the trust gate, 0.174 and 0.463 without it; mean estimation
error with the trust gate 2.8e-6 and 9.4e-6 (below 1e-5); all 100 runs with the trust gate end in a pure NE.
Without the trust gate 71/100 (equivocation) and 0/100 (oscillating) runs end in a pure NE. `verify_submitted.py`
compares the run with the paper's numbers (typed in by hand) and prints PASS/FAIL. Seeds are 1 to 100 and are
fixed, so the same library versions give the same results; other versions can change the last digits.
`expected/submitted_summary.json` and `expected/fig_ring_100_horizontal.pdf` hold the reference output of this run;
`expected/reference_original_figure.pdf` is the figure of the submitted paper, for visual comparison.

Definitions. **Nash gap**: for the empirical-frequency profile `f` (each agent's long-run action frequencies),
`max_i [ max_k U_i(k) - sum_k f_i(k) U_i(k) ]`, where `U_i(k) = prod_{j != i}(1 - f_j(k)) / d_k` is agent *i*'s
expected utility for target *k* against the other agents' independent mixed strategies; it is 0 exactly at a
Nash equilibrium. **Estimation error**: average Euclidean distance between each agent's local estimate of
another agent's frequencies and the true frequencies. **Ends in a pure NE**: the actions of the last round are
one-to-one (with 5 agents and 5 targets these are exactly the pure NE); `pure_ne_from` in `m1_results.npz`
stores the first round from which every round is a pure NE (-1 if the last round is not one).

## Extended study (not part of the submitted paper)

The sections below describe the extended study. Its results, including the Proposition 2 check, the star
network and the welfare and settling-time figures, are not in the submitted paper.

## Setup of the experiment

* **Game.** Target assignment with 5 agents and 5 targets. Agent *i* earns `1/d_k` if no other agent picks
  target *k* and 0 otherwise, where `d_k` is a cost common to all agents (the mean agent-target distance).
  This is an exact potential game with `Φ(a) = Σ_{k ∈ {a_i}} 1/d_k`; its pure Nash equilibria are the
  one-to-one assignments. There are no private distance signals.
* **Algorithm.** Trust-filtered decentralized fictitious play (Algorithm 1 of the paper): trust scores
  `β_ij`, threshold `ξ_t = 1.4·sqrt(t+1)`, equal-split weights over the trusted neighbors and the agent itself.
* **Attacks.** One malicious agent; four attack types (collision, Byzantine, oscillating, equivocation).
  It attacks with probability 0.35 per round. Trust observations are Gaussian: mean 0.85 for honest
  channels and while the attacker is quiet, mean 0.30 while it attacks, standard deviation 0.08.
* **Networks.** Ring and star (agent 0 is the hub).
* **Design.** 2 networks × 4 attacks × 2 defenses (trust gate, none) × 20 seeds (1 to 20), T = 5000 rounds.

## Running it

Requires Python 3.10 or newer. The reference results below were produced with Python 3.13, numpy 2.5.3,
scipy 1.18.1 and matplotlib 3.11.2.

```bash
git clone https://github.com/AIPcoder/trust-dfp-experiments.git
cd trust-dfp-experiments
pip install -r requirements.txt
python -m pytest -q                                    # 8 tests, about 12 s
python scripts/run_experiments.py --runs 20 --T 5000 --out results   # 320 simulations
python scripts/make_gaps.py --out results
python scripts/summarize.py --out results              # prints every headline number
python scripts/make_figures.py --out results           # fig_nash.pdf, fig_est.pdf
python scripts/verify_paper_numbers.py --out results   # PASS/FAIL against the paper
```

On Linux and macOS, `make test`, `make quick` and `make full` run the same steps (`make verify` checks
the paper's numbers). On Windows, use the `python` commands above. Use `python` instead of `python3` if your
system has no `python3` command.

* **Runtime.** The full run took 462 s on 2 CPU cores, plus about 5 s for the Nash gaps. `--jobs N` sets the
  number of worker processes. A small version for a quick check: `--runs 2 --T 1000`.
* **Repeatability.** Seeds are fixed, so the same library versions give the same results.
  `expected/summary_full_run.json` and `expected/fig_*.pdf` hold the reference output.
  `tests/test_regression.py` checks one full-length trajectory against a stored hash.
  Other library versions can change the last digits; the checker allows for the paper's rounding.

## Results (full setting)

| Quantity | Ring | Star |
| --- | --- | --- |
| Nash gap, t = 1 to t = 5000 (trust gate) | 0.963 to 0.0092 | 0.988 to 0.0139 |
| Nash gap at t = 5000, no defense | 0.30 | 0.54 |
| Runs whose actions reach a pure NE | 80/80 | 76/80 |
| Estimation error at t = 5000, trust gate | 6.6e-6 | 3.2e-5 |
| Welfare, t = 1 to t = 5000 (no defense, both networks: 0.44) | 0.17 to 1.00 | 0.16 to 0.98 |
| Classification time T_f (rounds, mean ± sd) | 64.5 ± 24.6 | 63.8 ± 22.9 |
| Proposition 2 check: triples tested / violations | 1,974,585 / 0 | 1,974,865 / 0 |

### Why 76/80 on the star

The four star runs that do not end at a pure NE are collision (seed 18), Byzantine (seed 8) and oscillating
(seeds 8 and 20). In each, two spoke agents have identical information (everything reaches them through the
hub) and identical utilities, so they keep choosing the same two targets with a roughly even split (between
38/62 and 51/49 in these runs). Their frequencies converge to a symmetric *mixed* Nash equilibrium (Nash
gap at t = 5000 between 0.005 and 0.008) instead of a one-to-one assignment.

This comes from fictitious play in a symmetric game, not from the trust gate or the attacker: plain DFP
without a malicious agent also fails on the star (`scripts/baseline_no_attacker.py` gives 62/80 pure-NE runs at
T = 3000, against 80/80 on the ring). The ring has no such symmetric pair. Breaking the tie would require a
change to the algorithm, such as randomized tie-breaking, which is left for future work.

## Code guide

| Path | Contents |
| --- | --- |
| `trustdfp/simulator.py` | The simulator. `dfp_step` is one round of Algorithm 1: attacker content, trust observations, trust scores `β`, trusted set, weights, averaging, best response, frequency update. |
| `trustdfp/potential_game.py` | The exact-potential game, its potential, and the NE test. |
| `trustdfp/metrics.py` | Nash gap of the empirical frequencies. |
| `scripts/run_m1.py`, `plot_m1.py`, `verify_submitted.py` | The submitted paper's experiment, its figure, and the check against its numbers. |
| `scripts/run_experiments.py` | Extended study: runs all simulations, including the Proposition 2 check. |
| `scripts/make_gaps.py`, `summarize.py`, `make_figures.py` | Nash gaps, statistics, figures. |
| `scripts/verify_paper_numbers.py` | Compares an extended-study run with the numbers reported for that study. |
| `scripts/check_potential.py` | Checks the exact-potential identity on all 5^5 = 3125 action profiles. |
| `scripts/baseline_no_attacker.py` | Plain DFP without a malicious agent. |
| `tests/` | Potential-game identity, NE characterization, end-to-end run, regression check. |

Time index: the loop counter `t` in `simulator.dfp_step` starts at 1, so it equals the paper's `t + 1`.

## Limits

* The results depend on the common-cost game; with agent-specific costs `d_ik` the game is not an exact
  potential game (`scripts/check_potential.py` prints a defect of about 1.1).
* Proposition 2's bound is conservative: the largest observed ratio of the error to the bound is 0.019 on
  the ring and 1e-5 on the star.
* Only 5 agents and 5 targets, one malicious agent, ring and star networks.

## Credits and AI assistance

The algorithm, the game, the attack models, the experimental design and the paper are the authors' work
(Alp Kerem Albas and Sarper Aydın), and the simulator was written by the authors. The structure of
`scripts/run_m1.py` and `scripts/plot_m1.py` follows the 100-seed runner and plotting script by Sarper Aydın.

Claude (Anthropic) was used as a coding assistant for preparing this repository for release: refactoring the
code, writing the unit and regression tests in `tests/`, and drafting parts of this README. The authors designed
the algorithm and experiments, reviewed all code and tests, and ran every experiment.

The methods the code implements are cited in the paper: decentralized fictitious play (Eksin and Ribeiro),
trust-based consensus filtering (Akgün et al.), spoof-resilient networks (Gil et al.) and exact potential games
(Monderer and Shapley, 1996). Libraries: NumPy, SciPy, Matplotlib, pytest.

## Citation and license

See `CITATION.cff`, `CITATION.bib` and `CHANGELOG.md`. MIT license (`LICENSE`).
