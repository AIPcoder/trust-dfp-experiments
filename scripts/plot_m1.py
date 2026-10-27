"""Two-panel figure of the submitted paper (Nash gap, estimation error) from run_m1.py output.

Mean over seeds (lines), 10th-90th percentile band for the trust gate; solid = with trust, dotted = without.
Figure style adapted from the version supplied by S. Aydin (see README, "Credits").
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results_m1"))
    opt = p.parse_args()
    data = np.load(opt.out / "m1_results.npz")
    gap, err = data["gap"], data["estimation_error"]
    t = np.arange(1, gap.shape[-1] + 1)
    names = ["Equivocation", "Oscillating"]
    colors = ["#09818c", "#b65b12"]
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif", "mathtext.fontset": "stix",
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "axes.linewidth": 0.6,
    })
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.42))
    for idx, arr in enumerate([gap, err]):
        for ai in range(2):
            for mi in range(2):
                curves = arr[ai, mi]
                ax[idx].plot(t, np.maximum(curves.mean(axis=0), 1e-10), color=colors[ai],
                             linestyle="-" if mi == 0 else ":", linewidth=1.4)
                if mi == 0:
                    lo, hi = np.percentile(curves, [10, 90], axis=0)
                    ax[idx].fill_between(t, np.maximum(lo, 1e-10), np.maximum(hi, 1e-10),
                                         color=colors[ai], alpha=0.13, linewidth=0)
        ax[idx].set_xscale("log")
        ax[idx].set_yscale("log")
        ax[idx].grid(True, which="both", alpha=0.20, linewidth=0.45)
        ax[idx].set_xlabel("$t$")
    ax[0].set_title("(a) Nash gap")
    ax[1].set_title("(b) Estimation error")
    ax[0].set_ylabel("Nash gap")
    ax[1].set_ylabel("Avg. estimation error")
    handles, labels = [], []
    for ai in range(2):
        for mi, mode in enumerate(("w/ trust", "w/o trust")):
            handles.append(Line2D([0], [0], color=colors[ai], linestyle="-" if mi == 0 else ":", linewidth=1.4))
            labels.append(f"{names[ai]}: {mode}")
    fig.legend(handles, labels, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.04), fontsize=7)
    fig.subplots_adjust(left=0.083, right=0.99, bottom=0.20, top=0.77, wspace=0.33)
    out = opt.out / "fig_ring_100_horizontal.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=220)
    print("Saved", out, "and", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
