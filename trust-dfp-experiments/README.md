# Trust-filtered decentralized fictitious play: numerical experiments

Code that reproduces every number and both figures in the numerical-experiment section of
**"Trustworthy Decentralized Fictitious Play in Potential Games under Malicious Interference"**
(A. K. Albas, S. Aydın; ICIT 2027 work-in-progress paper).

> Repository: `https://github.com/<OWNER>/<REPO>` (to be filled in) · Archived release (DOI): *to be added after the first release*

Anyone can rerun the study with three commands and check the output against the paper automatically.

## What is simulated

* **Game.** Target assignment with 5 agents and 5 targets as an *exact potential game*: agent *i* earns
  `1/d_k` if no other agent picks target *k*, else 0, with a common target cost `d_k` (mean agent-target distance).
  `Φ(a) = Σ_{k ∈ {a_i}} 1/d_k`; the pure Nash equilibria are the one-to-one assignments. No signal noise.
* **Algorithm.** Trust-filtered DFP (paper, Algorithm 1) with threshold `ξ_t = 1.4 √(t+1)`, equal-split weights.
  The base simulator `trustdfp/dfp_v8_lil.py` is used **unmodified** (a test checks its SHA-256).
* **Attacks.** Four types (collision, Byzantine uniform, oscillating, equivocation); attack probability 0.35.
* **Networks.** Ring and star (legitimate in-neighbour structure as in the 2021 experiments).
* **Design.** 2 networks × 4 attacks × 2 defenses (trust gate / none) × 20 seeds (1..20), T = 5000 rounds.

## Quick start

```bash
git clone https://github.com/<OWNER>/<REPO>.git && cd <REPO>
pip install -r requirements.txt          # Python >= 3.10; numpy, scipy, matplotlib, pytest
make test                                # 6 tests, about 10 s
make full                                # the paper's setting (320 simulations); see runtime below
make verify                              # compares the run with the paper's numbers: prints PASS/FAIL per number
```

`make quick` runs a small version (2 seeds, T = 1000) in under a minute for a sanity check.
Outputs go to `results/`: `runs.pkl` (all trajectories), `gaps.npz` (Nash gaps), `summary.json`
(every headline number), `fig_nash.pdf`, `fig_est.pdf` (the paper's Figs. 1 and 2).

**Runtime.** The full run took 462 s on 2 CPU cores (Python 3.13, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2),
plus 5 s for the gap series. Use `--jobs N` to set the number of processes.

**Determinism.** Seeds are fixed (1..20), so the results are repeatable on the same library versions. The reference
output of the authors' run is stored in `expected/summary_full_run.json`; `expected/fig_*.pdf` are the figures it produces.
Across library versions, last-digit floating-point differences are possible; `verify_paper_numbers.py` allows for
the rounding used in the paper.

## Scripts

| Script | Purpose |
| --- | --- |
| `scripts/run_experiments.py` | Runs all simulations, writes `runs.pkl` (incl. the Proposition 2 check) |
| `scripts/make_gaps.py` | Nash gap of the empirical frequencies for every run |
| `scripts/summarize.py` | All headline statistics → `summary.json` |
| `scripts/verify_paper_numbers.py` | Compares a full run with the numbers printed in the paper |
| `scripts/make_figures.py` | Figs. 1 and 2 |
| `scripts/check_potential.py` | Exhaustive exact-potential check on all 5^5 = 3125 profiles |
| `scripts/baseline_no_attacker.py` | Plain DFP without malicious agents (reference for the star result below) |

## Results reproduced (full setting)

| Quantity | Ring | Star |
| --- | --- | --- |
| Nash gap, t = 1 → t = 5000 (trust gate) | 0.963 → 0.0092 | 0.988 → 0.0139 |
| Nash gap at t = 5000, no defense | 0.30 | 0.54 |
| Runs whose actions reach a pure NE | 80/80 | 76/80 |
| Estimation error at t = 5000, gate | 6.6e-6 | 3.2e-5 |
| Welfare t = 1 → t = 5000 (no defense, both networks: 0.44) | 0.17 → 1.00 | 0.16 → 0.98 |
| Classification time T_f (rounds) | 64.5 ± 24.6 | 63.8 ± 22.9 |
| Proposition 2 check: triples / violations | 1,974,585 / 0 | 1,974,865 / 0 |

### Why 76/80 and not 80/80 on the star

The four remaining star runs (collision seed 18, Byzantine seed 8, oscillating seeds 8 and 20) are not failures of the
trust gate. In each, two spoke agents have identical information (everything reaches them through the hub) and identical
utilities, so they keep choosing the same two targets with a roughly even split (between 38/62 and 51/49 in these four runs). The frequencies then
converge to a symmetric *mixed* Nash equilibrium (Nash gap at t = 5000 between 0.005 and 0.008), not a pure one. This is a
property of DFP in symmetric games: plain DFP with no malicious agent also fails to settle on the star (see
`scripts/baseline_no_attacker.py`). The ring has no such symmetric pair. Breaking the tie would require changing the
algorithm (for example randomized tie-breaking), which the paper lists as future work.

## Layout

```
trustdfp/   dfp_v8_lil.py (base simulator, unmodified) · potential_game.py (exact-potential game) · metrics.py (Nash gap)
scripts/    run_experiments.py · make_gaps.py · summarize.py · verify_paper_numbers.py · make_figures.py · ...
tests/      exact-potential identity, NE characterization, unmodified-algorithm check, end-to-end smoke test
expected/   reference summary and figures from the full run
```

## Notes and limits

* `trustdfp/dfp_v8_lil.py` has its own `__main__` block that writes to a fixed path; it is not used by this repository.
* The results depend on the common-cost game described above; the agent-specific-cost game is not an exact potential
  game (`scripts/check_potential.py` shows a defect of about 1.1).
* Proposition 2's bound is conservative (largest observed ratio to the bound: 0.019 ring, 1e-5 star).

## Citation

Please cite the paper and this repository (see `CITATION.cff`). A BibTeX entry for the code is in `CITATION.bib`.

## License

MIT, see `LICENSE`.
