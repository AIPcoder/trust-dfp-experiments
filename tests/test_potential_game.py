import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from itertools import product
from trustdfp import potential_game as p


def _game(seed=3):
    return p.build_potential_game(N=5, K=5, T=10, seed=seed)


def test_exact_potential_identity_all_profiles():
    _, d, _ = _game()
    assert p.potential_defect(5, 5, d) < 1e-12


def test_agent_specific_costs_are_not_exact_potential():
    _, d, d_ik = _game()
    assert p.potential_defect(5, 5, d_ik) > 1e-3


def test_pure_ne_iff_one_to_one_assignment():
    _, d, _ = _game()
    for a in product(range(5), repeat=5):
        assert p.is_ne(a, d) == (len(set(a)) == 5)


def test_potential_maximizer_is_ne():
    _, d, _ = _game()
    best = max(product(range(5), repeat=5), key=lambda a: p.phi(a, d))
    assert p.is_ne(best, d)
