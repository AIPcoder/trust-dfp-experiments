import os, sys, subprocess, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_quick_pipeline_runs_end_to_end():
    with tempfile.TemporaryDirectory() as out:
        run = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, check=True, capture_output=True, text=True)
        run("scripts/run_experiments.py", "--runs", "1", "--T", "300", "--jobs", "2", "--out", out)
        run("scripts/make_gaps.py", "--out", out)
        r = run("scripts/summarize.py", "--out", out)
        assert "prop2_violations: 0" in r.stdout
        run("scripts/make_figures.py", "--out", out)
        assert os.path.exists(os.path.join(out, "fig_nash.pdf")) and os.path.exists(os.path.join(out, "fig_est.pdf"))

def test_base_algorithm_file_is_unmodified():
    import hashlib
    h = hashlib.sha256(open(os.path.join(ROOT, "trustdfp", "dfp_v8_lil.py"), "rb").read()).hexdigest()
    assert h == open(os.path.join(ROOT, "trustdfp", "dfp_v8_lil.py.sha256")).read().split()[0]
