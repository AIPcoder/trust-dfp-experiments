# Changelog

## 1.1.0
- Added the experiment of the submitted paper: `scripts/run_m1.py`, `plot_m1.py`, `verify_submitted.py`
  (ring, equivocation and oscillating attacks, 100 seeds, T = 5000). Its output reproduces the paper's
  Nash-gap values (0.0047, 0.0136, 0.174, 0.463) and the estimation-error bound.
- Corrected the README of 1.0.0, which said the code reproduced every number of the paper. 1.0.0 contained only
  the extended study (ring and star, four attacks, 20 seeds); the README now separates the two studies.
- The pure-NE persistence round in the submitted-paper runner is computed correctly (it was one round too early).
- Added the Nash gap and estimation error definitions, a credits section and a statement on AI assistance.
- No change to the simulator, the game, or the extended-study scripts.

## 1.0.0
- First release: simulator, exact-potential game, extended study (ring and star, four attacks, 20 seeds).
