"""Paper Figs. 1-2 (fig_nash.pdf, fig_est.pdf) from runs.pkl and gaps.npz."""

import numpy as np, pickle, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "font.family": "serif",
        "mathtext.fontset": "stix",
    }
)
import sys, os, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ap = argparse.ArgumentParser(description="Paper Figs. 1-2 from runs.pkl and gaps.npz")
_ap.add_argument("--out", default="results")
_ap.add_argument("--figdir", default=None)
_a = _ap.parse_args()
OUT = (_a.figdir or _a.out).rstrip("/") + "/"
os.makedirs(OUT, exist_ok=True)
_D = pickle.load(open(os.path.join(_a.out, "runs.pkl"), "rb"))
R = _D["res"]
G = np.load(os.path.join(_a.out, "gaps.npz"))
T = _D["meta"]["T"]
NRUNS = _D["meta"]["runs"]
t = np.arange(1, T + 1)
I1 = min(999, T - 1)
I2 = min(299, T - 1)
I0 = min(250, T - 2)  # anchors of the reference lines (paper: t=1000, 300)
ATS = ["collision", "byzantine", "oscillating", "equivocation"]
RUNS = range(1, NRUNS + 1)
cT = {"ring": "#028090", "star": "#B45309"}
CA = {
    "collision": "#B45309",
    "byzantine": "#7C3AED",
    "oscillating": "#0F766E",
    "equivocation": "#1E293B",
}
LS = {
    "collision": (0, ()),
    "byzantine": (0, (4, 1.5)),
    "oscillating": (0, (1, 1.2)),
    "equivocation": (0, (5, 1.5, 1, 1.5)),
}
LW = {"collision": 2.0, "byzantine": 1.2, "oscillating": 1.4, "equivocation": 1.2}
LAB = {
    "collision": "A collision",
    "byzantine": "B Byzantine",
    "oscillating": "C oscillating",
    "equivocation": "D equivocation",
}


def stack(topo, cfg, key, ats=ATS):
    return np.vstack(
        [np.array([R[(topo, a, cfg, r)][key] for r in RUNS], dtype=float) for a in ats]
    )


def band(a, x, y, c, al, ls="-", lw=1.2):
    m = y.mean(0)
    lo = np.percentile(y, 10, axis=0)
    hi = np.percentile(y, 90, axis=0)
    a.loglog(x, m, color=c, ls=ls, lw=lw)
    a.fill_between(x, np.maximum(lo, 1e-12), hi, color=c, alpha=al, lw=0)
    return m


# ---------- Fig 1
fig, ax = plt.subplots(
    2, 1, figsize=(3.45, 2.5), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]}
)
a = ax[0]
for tp in ("ring", "star"):
    gm = band(a, t, G[f"{tp}|trust_only"], cT[tp], 0.18)
    a.loglog(t, G[f"{tp}|none"].mean(0), color=cT[tp], ls=":", lw=1.3)
g = G["ring|trust_only"].mean(0)
a.loglog(t[I0:], (g[I1] * (I1 + 1) / t)[I0:], color="gray", ls="--", lw=0.8)
a.text(0.3 * T, g[I1] * (I1 + 1) / (0.3 * T) * 1.9, r"$\propto1/t$", fontsize=6, color="gray")
a.set_ylabel("Nash gap")
a.set_ylim(3e-3, 1.6)
a.grid(alpha=0.25, which="both")
h = [
    Line2D([], [], color=cT["ring"], lw=1.3),
    Line2D([], [], color=cT["star"], lw=1.3),
    Line2D([], [], color="k", lw=1.2),
    Line2D([], [], color="k", ls=":", lw=1.2),
]
a.legend(h, ["ring", "star", "trust gate", "no defense"], fontsize=5.5, loc="lower left", ncol=2)
a.set_title("(a) Nash gap of the frequencies (gate: mean and 10--90% band)", fontsize=6.5)
a = ax[1]


def ma(x, n=25):
    c = np.cumsum(np.insert(x, 0, 0))
    m = (c[n:] - c[:-n]) / n
    return np.concatenate([x[: n - 1], m])


for tp in ("ring", "star"):
    NE = stack(tp, "trust_only", "NE").mean(0)
    NEn = stack(tp, "none", "NE").mean(0)
    a.semilogx(t, NE, color=cT[tp], lw=1.3)
    a.semilogx(t, ma(NEn), color=cT[tp], ls=":", lw=1.3)
a.axhline(0.95, color=cT["star"], lw=0.5, alpha=0.5)
cnt = {
    tp: (int(stack(tp, "trust_only", "NE")[:, -1].sum()), stack(tp, "trust_only", "NE").shape[0])
    for tp in ("ring", "star")
}
a.text(1.2, 0.89, "star: %d/%d" % cnt["star"], fontsize=5.5, color=cT["star"])
a.text(1.2, 1.015, "ring: %d/%d" % cnt["ring"], fontsize=5.5, color=cT["ring"])
a.set_ylabel("runs at pure NE")
a.set_xlabel("$t$")
a.set_ylim(-0.03, 1.12)
a.grid(alpha=0.25, which="both")
a.set_title("(b) action profile is a pure NE (no defense: 25-round average)", fontsize=6.5)
fig.tight_layout(pad=0.4)
fig.savefig(OUT + "fig_nash.pdf")
fig.savefig(OUT + "fig_nash.png", dpi=220)
plt.close(fig)
# ---------- Fig 2
fig, ax = plt.subplots(2, 1, figsize=(3.45, 2.65), sharex=True, sharey=True)
for a, tp in zip(ax, ("ring", "star")):
    per_none = np.array([stack(tp, "none", "E", [x]).mean(0) for x in ATS])
    a.fill_between(t, per_none.min(0), per_none.max(0), color="gray", alpha=0.28, lw=0)
    a.loglog(t, per_none.mean(0), color="k", ls=":", lw=1.1)
    for x in ATS:
        a.loglog(t, stack(tp, "trust_only", "E", [x]).mean(0), color=CA[x], ls=LS[x], lw=LW[x])
    e = np.mean([stack(tp, "trust_only", "E", [x]).mean(0) for x in ATS], axis=0)
    y0 = 3 * e[I2]
    tt = t[I2:]
    a.loglog(tt, y0 * ((I2 + 1) / tt), color="gray", ls="--", lw=0.7)
    a.text(0.64 * T, y0 * (I2 + 1) / (0.64 * T) * 1.5, r"$\propto1/t$", fontsize=5.5, color="gray")
    a.loglog(tt, y0 * ((I2 + 1) / tt) ** 2, color="gray", ls="-.", lw=0.7)
    a.text(
        0.64 * T,
        y0 * ((I2 + 1) / (0.64 * T)) ** 2 * 1.5,
        r"$\propto1/t^2$",
        fontsize=5.5,
        color="gray",
    )
    a.grid(alpha=0.25, which="both")
    a.set_ylabel("avg. estimation error")
    a.set_title(f"({'a' if tp=='ring' else 'b'}) {tp}", fontsize=6.5)
ax[1].set_xlabel("$t$")
h = [Line2D([], [], color=CA[x], ls=LS[x], lw=LW[x]) for x in ATS] + [
    Line2D([], [], color="k", ls=":", lw=1.1),
    plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.28, lw=0),
]
ax[0].legend(
    h,
    [LAB[x] + " (gate)" for x in ATS]
    + ["no defense, mean over attacks", "no defense, range over attacks"],
    fontsize=5,
    loc="lower left",
    ncol=1,
)
fig.tight_layout(pad=0.4)
fig.savefig(OUT + "fig_est.pdf")
fig.savefig(OUT + "fig_est.png", dpi=220)
plt.close(fig)
print("ok")
