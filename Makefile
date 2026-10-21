.PHONY: install test quick full verify baseline
install:        ; pip install -r requirements.txt
test:           ; python3 -m pytest -q
quick:          ; python3 scripts/run_experiments.py --runs 2 --T 1000 --out results_quick && python3 scripts/make_gaps.py --out results_quick && python3 scripts/summarize.py --out results_quick && python3 scripts/make_figures.py --out results_quick
full:           ; python3 scripts/run_experiments.py --runs 20 --T 5000 --out results && python3 scripts/make_gaps.py --out results && python3 scripts/summarize.py --out results && python3 scripts/make_figures.py --out results
verify:         ; python3 scripts/verify_paper_numbers.py --out results
baseline:       ; python3 scripts/baseline_no_attacker.py
