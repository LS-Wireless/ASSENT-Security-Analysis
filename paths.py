"""Central path setup for the ASSENT security analysis.

Every script in `attacks/`, `defenses/`, `experiments/`, and `feasibility/` starts with

    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(HERE, ".."))
    import paths  # noqa: F401

Importing this module does three things:

  1. Puts the released ASSENT repository on `sys.path`, so `from src.learning_v4...`
     and the bare `assent_run_v4` style imports both resolve.
  2. Exposes `RESULTS` and `FIGURES`, the two output roots used across the codebase.
  3. Creates those output directories if they do not exist yet.

The upstream clone is NOT vendored here. Clone it yourself:

    git clone https://github.com/LS-Wireless/ASSENT-CellFree-ISAC upstream/ASSENT-CellFree-ISAC

or point the environment variable ASSENT_UPSTREAM at an existing clone.
"""
import os
import sys

# Repository root (this file lives at the root).
ROOT = os.path.dirname(os.path.abspath(__file__))

# Machine-readable outputs behind every number in the paper.
RESULTS = os.path.join(ROOT, "results")

# Generated plots. Not tracked in git; every figure is rebuilt from RESULTS.
FIGURES = os.path.join(ROOT, "figures")

# The released ASSENT implementation (never modified by anything in this repo).
UPSTREAM = os.environ.get("ASSENT_UPSTREAM") or os.path.join(ROOT, "upstream", "ASSENT-CellFree-ISAC")
UPSTREAM = os.path.abspath(UPSTREAM)

# Decode thresholds published with the ASSENT release (run_04 / GATv2). Kept here for
# reference only: the experiments read the per-architecture thresholds that were tuned on
# the validation split and stored in results/checkpoints*/<arch>/summary.json.
DECODE_THRESHOLDS = {"x": 0.60, "tau": 0.10, "s": 0.10, "ytx": 0.525, "yrx": 0.725}

for _d in (RESULTS, FIGURES):
    os.makedirs(_d, exist_ok=True)

# The upstream code is imported two ways: as `src.learning_v4.<module>` from the repo root,
# and as bare `assent_run_v4` / `assent_run_utils_v4` from inside src/learning_v4.
for _p in (UPSTREAM, os.path.join(UPSTREAM, "src", "learning_v4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if not os.path.isdir(UPSTREAM):
    sys.stderr.write(
        "\n[paths] The released ASSENT code was not found at:\n"
        f"        {UPSTREAM}\n"
        "        Scripts that only read results/*.json still work; anything that loads the\n"
        "        model or the dataset will fail. Clone it with:\n"
        "          git clone https://github.com/LS-Wireless/ASSENT-CellFree-ISAC "
        "upstream/ASSENT-CellFree-ISAC\n"
        "        or set ASSENT_UPSTREAM to an existing clone.\n\n"
    )
