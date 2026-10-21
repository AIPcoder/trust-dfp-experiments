# trust-dfp-experiments

Code for the numerical experiments in *Trustworthy Decentralized Fictitious Play in Potential Games under
Malicious Interference* (A. K. Albas and S. Aydın, ICIT 2027 work-in-progress paper).
It reproduces every number and both figures of the experiment section, and a script checks the output against
the numbers printed in the paper.

Repository: https://github.com/AIPcoder/trust-dfp-experiments · Archived release (DOI): to be added.

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
python -m pytest -q                                    # 6 tests, about 10 s
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
change to the algorithm, such as randomized tie-breaking, which the paper lists as future work.

## Code guide

| Path | Contents |
| --- | --- |
| `trustdfp/simulator.py` | The simulator. `dfp_step` is one round of Algorithm 1: attacker content, trust observations, trust scores `β`, trusted set, weights, averaging, best response, frequency update. |
| `trustdfp/potential_game.py` | The exact-potential game, its potential, and the NE test. |
| `trustdfp/metrics.py` | Nash gap of the empirical frequencies. |
| `scripts/run_experiments.py` | Runs all simulations, including the Proposition 2 check. |
| `scripts/make_gaps.py`, `summarize.py`, `make_figures.py` | Nash gaps, statistics, figures. |
| `scripts/verify_paper_numbers.py` | Compares a full run with the numbers in the paper. |
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

## Citation and license

See `CITATION.cff` and `CITATION.bib`. MIT license (`LICENSE`).
