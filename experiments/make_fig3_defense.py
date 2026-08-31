"""Figure 3: report lies degrade feasibility, and the cross-AP consistency check.

  left   % of plans broken vs lie size, for the interference lie, the sensing lie, and a
         two-AP interference collusion
  right  against the interference lie: residual infeasibility after the output-side rule fix
         alone, after adding the cross-AP input check, and the detection rate

Reads the committed attack and defense JSONs; no model, dataset, or solver needed. Titles are
deliberately absent: the paper carries that information in the LaTeX caption.

Run: python experiments/make_fig3_defense.py
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
ap.add_argument("--stem", default="fig3_defense", help="output file stem")
args = ap.parse_args()

(FIGSIZE, SCALE) = fs.size(args.width, args.height, args.fontscale)
fs.setup(SCALE)
R = paths.RESULTS
OUT = args.out or paths.FIGURES
os.makedirs(OUT, exist_ok=True)


def load(fn):
    return json.load(open(os.path.join(R, fn)))["rows"]


KS = [1, 2, 4, 8, 16, 32]
TICKLAB = [f"$2^{{{i}}}$" for i in range(6)]

fig, (axD, axR) = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)
fs.tighten(fig)
fs.grid(axD)
fs.grid(axR)

# ------------------------------------------------- left: how badly the lies break the system
corr = load("attack_gatv2_repro_corr_deflation_def.json")      # one-AP interference lie
sens = load("attack_gatv2_repro_sensing_inflation_a3.json")    # one-AP sensing lie, worst position
coll = load("collusion_corr_deflation.json")                   # two-AP interference collusion
k = [x["kappa"] for x in corr]

axD.plot(k, [x["infeas_pct"] for x in corr], "-o", color=fs.BLUE, lw=fs.pt(2.2), ms=fs.pt(5.2),
         label="interference lie (1 AP)", zorder=3)
axD.plot(k, [x["infeas_pct"] for x in sens], "--s", color=fs.ORANGE, lw=fs.pt(2.2), ms=fs.pt(5.2),
         label="sensing lie (1 AP)", zorder=3)
axD.plot(k, [x["system_invalid_pct"] for x in coll], "-.^", color=fs.PURPLE, lw=fs.pt(2.2), ms=fs.pt(5.4),
         label="interference lie (2 APs)", zorder=3)

axD.set_ylabel("% of plans broken (illegal)", labelpad=2.0)
# Extra headroom at the bottom keeps the legend clear of the lowest curve.
axD.set_ylim(67, 100)
axD.set_yticks([70, 80, 90, 100])
axD.legend(frameon=False, loc="lower right", bbox_to_anchor=(0.995, 0.02), fontsize=fs.pt(5.9))

# ------------------------------------------------- right: the cross-AP consistency check
ca = load("crossap_defense.json")
axR.plot(k, [x["repaironly_inf"] for x in ca], "-o", color=fs.RED, lw=fs.pt(2.2), ms=fs.pt(5.2),
         label="rule fix only", zorder=3)
axR.plot(k, [x["crossap_inf"] for x in ca], "--s", color=fs.GREEN, lw=fs.pt(2.2), ms=fs.pt(5.2),
         label="+ cross-AP check", zorder=3)
axR.plot(k, [x["detect_pct"] for x in ca], ":^", color=fs.GREY, lw=fs.pt(1.9), ms=fs.pt(5.0),
         label="cheater caught (%)", zorder=3)

axR.set_ylabel("% still illegal (or caught)", labelpad=2.0)
axR.set_ylim(-6, 108)
axR.set_yticks([0, 25, 50, 75, 100])
axR.legend(frameon=False, loc="center right", bbox_to_anchor=(0.995, 0.52), fontsize=fs.pt(5.9))

for ax in (axD, axR):
    ax.set_xscale("log", base=2)
    ax.set_xlabel("lie size $k$", labelpad=2.0)
    ax.set_xticks(KS)
    ax.set_xticklabels(TICKLAB)
    ax.tick_params(axis="both", pad=1.5)

fs.save(fig, OUT, args.stem)
