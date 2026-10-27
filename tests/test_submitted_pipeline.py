import os, sys, subprocess, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)


def test_persistence_start_index():
    from run_m1 import persistence_start

    assert persistence_start([0, 0, 1, 1, 1]) == 3
    assert persistence_start([1, 1, 1, 1, 1]) == 1
    assert persistence_start([1, 0, 1, 1, 0]) == -1
    assert persistence_start([0, 1, 0, 1, 1]) == 4


def test_submitted_pipeline_runs():
    with tempfile.TemporaryDirectory() as out:
        run = lambda *a: subprocess.run(
            [sys.executable, *a], cwd=ROOT, check=True, capture_output=True, text=True
        )
        run("scripts/run_m1.py", "--seeds", "2", "--steps", "300", "--workers", "2", "--out", out)
        run("scripts/plot_m1.py", "--out", out)
        assert os.path.exists(os.path.join(out, "fig_ring_100_horizontal.pdf"))
