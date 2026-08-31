"""Figure 2: the signal FDI attack (steal variant) across three attacker positions.

  left   users the cheater serves vs lie size k   (at most +0.2, even at 32x)
  right  radio chains the cheater uses vs lie size k, against the budget of 4

Reads the committed attack JSONs; no model, dataset, or solver needed. Titles are deliberately
absent: the paper carries that information in the LaTeX caption.

Run: python experiments/make_fig2_attack.py
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
ap.add_argument("--stem", default="fig2_attack", help="output file stem")
args = ap.parse_args()

(FIGSIZE, SCALE) = fs.size(args.width, args.height, args.fontscale)
fs.setup(SCALE)
R = paths.RESULTS
OUT = args.out or paths.FIGURES
os.makedirs(OUT, exist_ok=True)


def load(fn):
    return json.load(open(os.path.join(R, fn)))["rows"]


# Distinct in color, in marker, and in line style, so the panels survive grayscale printing.
SERIES = {0: (fs.BLUE, "o", "-"), 3: (fs.ORANGE, "s", "--"), 6: (fs.GREEN, "^", "-.")}

fig, (axU, axR) = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)
fs.tighten(fig)
fs.grid(axU)
fs.grid(axR)

for a, (col, mk, ls) in SERIES.items():
    r = load(f"attack_gatv2_repro_gain_steal_a{a}.json")
    k = [x["kappa"] for x in r]
    axU.plot(k, [x["atk_users"] for x in r], ls, marker=mk, color=col,
             lw=fs.pt(2.2), ms=fs.pt(5.2), label=f"attacker = AP {a}", zorder=3)
    axR.plot(k, [x["atk_rfload"] for x in r], ls, marker=mk, color=col,
             lw=fs.pt(2.2), ms=fs.pt(5.2), label=f"AP {a}", zorder=3)

KS = [1, 2, 4, 8, 16, 32]
TICKLAB = [f"$2^{{{i}}}$" for i in range(6)]
for ax in (axU, axR):
    ax.set_xscale("log", base=2)
    ax.set_xlabel("lie size $k$", labelpad=2.0)
    ax.set_xticks(KS)
    ax.set_xticklabels(TICKLAB)
    ax.tick_params(axis="both", pad=1.5)

axU.set_ylabel("users the attacker serves", labelpad=2.0)
axU.set_ylim(0, 4.2)
axU.set_yticks([0, 1, 2, 3, 4])
axU.legend(frameon=False, loc="lower right", bbox_to_anchor=(0.995, 0.02),
           fontsize=fs.pt(5.9), ncol=1)

axR.set_ylabel("radio chains the attacker uses", labelpad=2.0)
axR.axhline(4, ls=":", color=fs.GREY, lw=fs.pt(1.3), zorder=2)
axR.set_ylim(3.0, 4.22)
axR.set_yticks([3.0, 3.5, 4.0])
axR.text(1.05, 4.05, "RF-chain budget = 4", fontsize=fs.pt(4.6), color=fs.GREY,
         ha="left", va="bottom")

fs.save(fig, OUT, args.stem)
