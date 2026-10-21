"""Exhaustively verify the exact-potential property (Monderer-Shapley) on all K^N action profiles.
Prints the largest violation |du_i - dPhi| for the common-cost game (should be ~0) and, for contrast, for
agent-specific costs d_ik (not an exact potential game)."""
import _bootstrap  # noqa: F401
from trustdfp import potential_game as p

if __name__ == "__main__":
    N = K = 5
    _, d, d_ik = p.build_potential_game(N=N, K=K, T=10, seed=1)
    print(f"common cost d_k       : max potential defect = {p.potential_defect(N, K, d):.3e}   (5^5 = {K**N} profiles)")
    print(f"agent-specific d_ik   : max potential defect = {p.potential_defect(N, K, d_ik):.3e}   (not exact potential)")
