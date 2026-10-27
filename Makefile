.PHONY: install test quick full verify baseline submitted verify-submitted
install:        ; pip install -r requirements.txt
test:           ; python3 -m pytest -q
quick:          ; python3 scripts/run_experiments.py --runs 2 --T 1000 --out results_quick && python3 scripts/make_gaps.py --out results_quick && python3 scripts/summarize.py --out results_quick && python3 scripts/make_figures.py --out results_quick
full:           ; python3 scripts/run_experiments.py --runs 20 --T 5000 --out results && python3 scripts/make_gaps.py --out results && python3 scripts/summarize.py --out results && python3 scripts/make_figures.py --out results
verify:         ; python3 scripts/verify_paper_numbers.py --out results
baseline:       ; python3 scripts/baseline_no_attacker.py
# Submitted paper: ring, equivocation and oscillation, 100 seeds, T = 5000 (about 12 min on 2 cores)
submitted:        ; python3 scripts/run_m1.py --seeds 100 --steps 5000 --out results_m1 && python3 scripts/plot_m1.py --out results_m1
verify-submitted: ; python3 scripts/verify_submitted.py --out results_m1
