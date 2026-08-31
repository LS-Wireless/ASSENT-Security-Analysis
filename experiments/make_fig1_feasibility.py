"""Figure 1: feasibility of the learned scheduler (GATv2), two panels, from the committed JSONs.

  left   per-decision F1 stays near 1 while joint legality collapses  (f1_decouple.json)
  right  MILP / raw GNN / ILP projection / greedy repair objectives   (difficulty_gatv2.json)

No model, dataset, or solver needed. Titles are deliberately absent: the paper carries that
information in the LaTeX caption.

Run: python experiments/make_fig1_feasibility.py
"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import paths  # noqa: F401
from experiments import figstyle as fs
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=float, default=fs.FIG_W,
                help="canvas width in inches; must equal the width of the LaTeX box "
                     "(%(default)s = a two-column figure* at \\textwidth; fs.COL_W = one column)")
ap.add_argument("--height", type=float, default=fs.FIG_H, help="canvas height in inches")
ap.add_argument("--fontscale", type=float, default=None,
                help="override the type scale (default: width / one column). A wide canvas does "
                     "not need type scaled fully in proportion; keep labels at or below the 9 pt "
                     "body text.")
ap.add_argument("--out", default=None, help="output directory (default: figures/)")
ap.add_argument("--stem", default="fig1_feasibility", help="output file stem")
args = ap.parse_args()

(FIGSIZE, SCALE) = fs.size(args.width, args.height, args.fontscale)
fs.setup(SCALE)
R = paths.RESULTS
OUT = args.out or paths.FIGURES
os.makedirs(OUT, exist_ok=True)

fig, (axL, axR) = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True,
                               gridspec_kw={"width_ratios": [0.96, 1.04]})

# ---------------------------------------------------------------- left: F1 vs legality
dc = json.load(open(os.path.join(R, "f1_decouple.json")))["rows"]
ks = [r["k"] for r in dc]
axT = axL.twinx()
fs.tighten(fig)
fs.grid(axL)

l1, = axL.plot(ks, [r["mean_f1"] for r in dc], "-o", color=fs.GREEN,
               lw=fs.pt(2.2), ms=fs.pt(5.2), label="per-decision $F_1$", zorder=3)
l2, = axT.plot(ks, [r["illegal_pct"] for r in dc], "--s", color=fs.RED,
               lw=fs.pt(2.2), ms=fs.pt(5.2), label="plans illegal", zorder=3)

axL.set_xlabel("extra links added to the optimal plan", labelpad=2.0)
axL.set_ylabel("mean per-decision $F_1$", color=fs.GREEN, labelpad=2.0)
axL.set_ylim(0.90, 1.008)
axL.set_xticks(ks)
axL.set_yticks([0.90, 0.95, 1.00])
axL.tick_params(axis="y", labelcolor=fs.GREEN, pad=1.0)
axL.tick_params(axis="x", pad=1.5)
axT.set_ylabel("% of plans illegal", color=fs.RED, labelpad=2.0)
axT.set_ylim(-5, 108)
axT.set_yticks([0, 50, 100])
axT.tick_params(axis="y", labelcolor=fs.RED, pad=1.0)
axL.legend(handles=[l1, l2], frameon=False, loc="lower right",
           bbox_to_anchor=(0.995, 0.02), fontsize=fs.pt(5.9))

# ---------------------------------------------------------------- right: cost of legality
dif = json.load(open(os.path.join(R, "difficulty_gatv2.json")))
oa, gap = dif["obj_abs"], dif["obj_gap_pct"]
milp = oa["milp"]
bars = [("MILP\n(optimum)", oa["milp"], fs.GREEN, False, "100%"),
        ("raw GNN", oa["raw"], fs.RED, True, "INVALID"),
        ("ILP\nprojection", oa["ilp"], fs.BLUE, False, f"{100 + gap['ilp']:.2f}%"),
        ("greedy\nrepair", oa["greedy"], fs.CYAN, False, f"{100 + gap['greedy']:.2f}%")]

# No grid on the bar panel: the dashed MILP reference line is the only guide needed.
axR.axhline(milp, ls="--", color="0.4", lw=fs.pt(1.1), zorder=1)
for i, (lab, val, col, hatch, tag) in enumerate(bars):
    axR.bar(i, val, width=0.7, color=col, edgecolor="black", linewidth=fs.pt(0.7),
            hatch="///" if hatch else None, zorder=2)
    # White backing so the labels of the sub-optimum bars do not collide with the MILP line.
    axR.text(i, val + 0.0016, f"{val:.3f}\n{tag}", ha="center", va="bottom",
             fontsize=fs.pt(5.6), linespacing=1.2,
             color=fs.RED if hatch else "black",
             fontweight="bold" if hatch else "normal", zorder=4,
             bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9))

axR.set_xticks(range(len(bars)))
axR.set_xticklabels([b[0] for b in bars], fontsize=fs.pt(6.0))
axR.set_ylim(milp - 0.030, milp + 0.040)
axR.set_yticks([4.26, 4.27, 4.28, 4.29, 4.30])
axR.set_ylabel("mean objective value", labelpad=2.0)
axR.tick_params(axis="both", pad=1.5)
axR.set_axisbelow(True)

fs.save(fig, OUT, args.stem)
