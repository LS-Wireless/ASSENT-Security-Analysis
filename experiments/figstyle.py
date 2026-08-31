"""Shared figure style for the paper figures.

Sizing rationale (this is the important part). In the submitted version each figure was drawn
on a 4.8 to 7.4 inch canvas and then scaled down to one ACM column by \\includegraphics, by a
factor of 0.34 to 0.45. An 11 pt axis label therefore reached the page at 3.7 to 5 pt, well
under the 9 pt body text. Enlarging the fonts on a big canvas does not fix that, because the
canvas is scaled, not the fonts.

So every figure here is drawn at its FINAL printed size: one column wide (COL_W), which is what
\\includegraphics[width=\\columnwidth]{...} asks for. The scale factor becomes 1.0, and a 7 pt
label is 7 pt on paper. Keep it that way: if you widen a canvas, widen the LaTeX box to match,
or the text shrinks again.

The paper's figures span both columns (figure*, FIG_W = 7.00 in), which is the only way to make
a two-panel figure genuinely wider: inside a single column the width is fixed at COL_W, and a
wider canvas is simply scaled back down. At full width the panels are about 3.1 in across and
the type sits at 8.4 pt labels / 7 pt ticks, under the 9 pt body text.

ACM compliance:
  - Vector PDF output, no rasterization.
  - pdf.fonttype 42 embeds TrueType, so no Type 3 fonts (ACM and IEEE both reject those).
  - The palette is Paul Tol's colorblind-safe qualitative set, and every series is also
    separated by marker and line style so the figures survive grayscale printing.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# One ACM sigconf column, measured from the typeset paper: 240.2 pt.
COL_W = 3.34
# Full text width, for a two-column figure* that spans the page (504 pt in acmart sigconf).
FULL_W = 7.00
# Default height for a two-panel, one-column figure.
COL_H = 1.62

# Committed geometry: the paper's figures are two-column figure* environments, included with
# \includegraphics[width=\textwidth]{...}. Type is NOT scaled in full proportion to the width;
# FIG_SCALE puts axis labels at 8.4 pt and ticks at 7 pt, comfortably under the 9 pt body text.
FIG_W = FULL_W
FIG_H = 2.20
FIG_SCALE = 1.40

# Set by setup(); every hard-coded point size in the figure scripts goes through pt().
SCALE = 1.0

# Paul Tol qualitative, colorblind-safe and distinguishable in grayscale.
BLUE = "#4477AA"
ORANGE = "#EE7733"
GREEN = "#117733"
RED = "#CC3311"
PURPLE = "#AA3377"
CYAN = "#88CCEE"
GREY = "#444444"

GRID_KW = dict(color="0.75", linestyle="--", linewidth=0.5, alpha=0.7)


def pt(x):
    """Scale a hard-coded point size by the current figure scale."""
    return x * SCALE


def setup(scale=1.0):
    """Apply the shared rcParams. Call once, before creating any figure.

    `scale` multiplies every type and line size. Pass width / COL_W when drawing on a wider
    canvas, so a two-column figure keeps the same visual density instead of looking sparse.
    """
    global SCALE
    SCALE = scale
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 5.5 * scale,
        "axes.labelsize": 6 * scale,
        "xtick.labelsize": 5 * scale,
        "ytick.labelsize": 5 * scale,
        "legend.fontsize": 4.8 * scale,
        "axes.linewidth": 0.7 * scale,
        "xtick.major.width": 0.7 * scale,
        "ytick.major.width": 0.7 * scale,
        "xtick.major.size": 1.6 * scale,
        "ytick.major.size": 1.6 * scale,
        "xtick.minor.size": 1.5 * scale,
        "ytick.minor.size": 1.5 * scale,
        "lines.linewidth": 2.0 * scale,
        "lines.markersize": 4.6 * scale,
        "lines.markeredgewidth": 0.0,
        "legend.handlelength": 1.3,
        "legend.handletextpad": 0.35,
        "legend.labelspacing": 0.25,
        "legend.borderpad": 0.3,
        "legend.borderaxespad": 0.3,
        "legend.columnspacing": 1.0,
        "pdf.fonttype": 42,   # TrueType, never Type 3
        "ps.fonttype": 42,
        "pdf.compression": 9,
        "figure.dpi": 300,
    })


def size(width=None, height=None, fontscale=None):
    """(width, height) for a two-panel figure, and the matching type scale.

    At the committed full-text width the scale is FIG_SCALE, tuned by eye. At any other width
    the type scales in proportion to one column, which keeps the visual density constant.
    """
    w = FIG_W if width is None else width
    h = FIG_H if height is None else height
    if fontscale is None:
        fontscale = FIG_SCALE if abs(w - FIG_W) < 1e-6 else w / COL_W
    return (w, h), fontscale


def tighten(fig):
    """Squeeze the constrained-layout margins.

    Take space from padding before taking it from type: the defaults reserve about 3 pt of
    whitespace on every edge and between panels, which is pure loss on a 3.34 in canvas."""
    eng = fig.get_layout_engine()
    if eng is not None:
        eng.set(w_pad=0.008, h_pad=0.008, wspace=0.015, hspace=0.015)


def grid(ax, axis="both"):
    """Readability grid, always drawn behind the data."""
    ax.grid(True, axis=axis, **GRID_KW)
    ax.set_axisbelow(True)


def save(fig, out_dir, stem):
    """Write the vector PDF the paper embeds, plus a PNG for quick viewing on GitHub."""
    import os
    pdf = os.path.join(out_dir, f"{stem}.pdf")
    png = os.path.join(out_dir, f"{stem}.png")
    # No bbox_inches="tight": it would trim the canvas and break the 1:1 size calibration.
    fig.savefig(pdf)
    fig.savefig(png, dpi=400)
    plt.close(fig)
    print(f"wrote {stem}.pdf / .png")
