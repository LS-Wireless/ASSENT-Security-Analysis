# RUNBOOK, Running the ASSENT Security Experiments

**Verified against the actual cloned repo on 2026-06-06** (`upstream/ASSENT-CellFree-ISAC`, commit as cloned). This corrects several assumptions from the earlier analysis.

---

## ❓ Do we need a GPU?, NO.

**Authoritative answer: no GPU at any stage.** The authors themselves trained ASSENT on CPU. Their own training config (`src/learning_v4/checkpoints/run_04/metadata.json`) reads:

```json
"TrainConfig": { "device": "cpu", "amp": false, "epochs": 100, "batch_size": 16, ... }
```

- **Why CPU is fine:** the graph is tiny (8 APs + 10 CUs + 4 targets ≈ 22 nodes), the model is small (hidden 128, 3 layers), and the dataset is 10k graphs. This trains on a laptop CPU.
- **A GPU is purely optional**, it would only shorten the one-time training wall-clock (and only if you iterate a lot). Nothing in the pipeline *requires* CUDA.
- **The real compute cost is Gurobi** (the MILP), which is **CPU-only** by nature, and most of our experiments don't even call it (see below).

---

## 🧱 Clean separation

```
ASSENT-Security-Analysis/
├── upstream/ASSENT-CellFree-ISAC/   ← authors' repo, PRISTINE, never edit (not tracked here)
├── paths.py                         ← puts upstream on sys.path (import, don't copy)
├── attacks/  defenses/  feasibility/  experiments/
└── results/  figures/               ← our outputs only
```

**Rule:** our code *imports* the authors' functions (`import src.utils.optimization_utils as opt`) but never modifies anything under `upstream/`. All our attacks, defenses, metrics, and generated results live in this repository. This guarantees our work is never conflated with the original implementation.

---

## ✅ What the clone confirmed / corrected

| Item | Earlier assumption | **Verified reality** |
|------|--------------------|----------------------|
| Trained weights (`.pt`) | shipped in repo | **NOT shipped**, `.gitignore` excludes `*.pt`. Only training logs (`history/summary/metadata.json`) are committed. → **we must retrain, OR get the `.pt` files from the authors** (you collaborate with them, easiest). |
| Decode threshold | 0.5 | **Per-output tuned thresholds**, swept over [0.2, 0.8]: `thr_x=0.60, thr_tau=0.10, thr_s=0.10, thr_ytx=0.525, thr_yrx=0.725` (from `run_04/…summary.json`). Use these in the feasibility harness, not 0.5. |
| Datasets | unclear | **Present & loadable.** Inputs = `exp1_baseline/run_01/2025-10-03_results_p1..4_of_4.pkl` (~102 MB). Labels = `exp2_data_gen/run_01/2025-10-13-13-25_results.pkl` (~16 MB). Loaded by `assent_data_utils_v4.py::data_loader()`. |
| Git LFS | needed? | **Not needed for v4.** Only `exp1_baseline/run_03` + `exp1_pareto/run_01` are LFS pointers; v4 uses `run_01`, which is a normal git object (already downloaded). No `git lfs pull` required. |
| MILP solver | `optimization_utils.py` | `src/optimization/milp_solver.py` + `src/utils/optimization_utils.py` (`ProblemParams`, `solve_problem`, `compute_milp_objective`). Caps **active**: `K_tx=2, K_rx=2, rf_cap=4`. |
| Versions | one | `learning_v1..v4` + `assent/`. **Use `learning_v4`** (current) and/or `src/assent/` (polished paper runs). |

---

## 🔧 Environment setup (one time, CPU)

Python 3.12 with PyTorch and PyTorch Geometric. Nothing here needs a GPU.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
git clone https://github.com/LS-Wireless/ASSENT-CellFree-ISAC upstream/ASSENT-CellFree-ISAC
pip install -r upstream/ASSENT-CellFree-ISAC/requirements.txt
```

If you keep the upstream clone somewhere else, point `ASSENT_UPSTREAM` at it instead of using the `upstream/` subdirectory.

Notes:
- PyG runs these conv layers (NNConv/GATv2Conv/TransformerConv) without the optional C++ extensions (`torch-scatter` etc.). If an import complains, `pip install torch-scatter` against your torch build.
- **Gurobi is optional.** The projection defense used for every number in the paper is `defenses/project_scipy.py`, which solves the same ILP with scipy HiGHS and needs no license. `defenses/project.py` is the Gurobi reference formulation, kept for cross-checking; a licensed solver is needed only for that and for generating new MILP labels (see matrix below).

---

## ▶️ How to run, staged

> Our scripts import `paths.py` at the repository root, which puts the upstream clone on `sys.path` and chdirs where the upstream loaders expect. So run them from this repository root, from anywhere, without editing the upstream tree.

### Stage 0, Reproduce (de-risk before building)
1. **Build the graph cache** (one-time; reads the .pkl datasets, writes `cache_graphs/`):
   ```bash
   cd upstream/ASSENT-CellFree-ISAC/src/learning_v4
   python assent_data_v4.py        # precompute_graph_dataset(nsamps=10000)
   ```
2. **Train** (CPU; reproduces F1 ≈ paper Table II). Either:
   - **(a) retrain ourselves:** `python assent_run_v4.py`  (100 epochs, early stopping), or
   - **(b) ask the authors for the `*_best_model_fully_trained.pt` files** referenced in `summary.json` → drop into `checkpoints/run_04/` → skip training entirely.
3. **Evaluate / confirm the headline anomaly:** `python assent_evaluation_v4.py` (or `assent_test_v4.py`) → check F1 and the objective vs MILP (the 4.311 > 4.286 story). Use the tuned thresholds above.

### Stage 1, Feasibility audit and repair cost
```bash
python experiments/run_repair.py --arch gatv2      # audit + projection, per backbone
python experiments/analyze_difficulty.py           # raw vs ILP vs greedy objective (Fig. 1, right)
python experiments/f1_decouple.py                  # F1 versus legality (Fig. 1, left)
python experiments/alpha_strata.py                 # cost across the comm/sensing trade-off
```
Binarize with the tuned thresholds, check constraints 7c-7k, report violation rate and infeasibility. The MILP labels are already in the dataset, so no new solve is needed.

### Stage 2, Report-channel attacks
```bash
python experiments/run_attack.py --attack gain_steal --a_star 3     # signal lie (Fig. 2)
python experiments/run_attack.py --attack corr_deflation --defense  # interference lie (Fig. 3, left)
python experiments/run_attack.py --attack sensing_inflation --a_star 3
python experiments/run_collusion.py --attack corr_deflation         # two colluding APs
```
Each attack perturbs only `a*`'s own reported slice before the forward pass, then measures impact on the true channel.

### Stage 3, Defenses
```bash
python experiments/run_crossap.py                  # cross-AP consistency check (Fig. 3, right)
```
The projection is `defenses/project_scipy.py` (HiGHS, license-free), which is what the paper uses; `defenses/project.py` is the Gurobi reference formulation.

### What needs Gurobi vs not
| Step | Gurobi? |
|------|---------|
| Build cache, train, evaluate, all attacks, the feasibility audit, and the projection via `project_scipy.py` | ❌ no |
| `defenses/project.py` (Gurobi reference formulation), generating NEW data, OOD MILP ground-truth | ✅ yes (CPU) |

---

## ⏱️ Rough time budget (CPU laptop)
- Build graph cache: minutes. Train one architecture: ~tens of minutes (authors used CPU). Three architectures: a few hours total, or **zero** if the authors share weights.
- Attack and audit sweeps over the 1k-graph test split: seconds to minutes each.
- Projection: one small ILP per graph; minutes for the test split.

**Bottom line:** a normal CPU is sufficient end-to-end. Getting the trained `.pt` files from the authors is the single biggest time-saver and removes the only nontrivial CPU job (training).

---

## Rebuilding the figures

The three paper figures come from the committed JSONs alone, with no model, dataset, or solver:

```bash
python experiments/make_fig1_feasibility.py
python experiments/make_fig2_attack.py
python experiments/make_fig3_defense.py
```

Shared styling is in `experiments/figstyle.py`. Each figure is drawn at the full text width
(7.00 in) so that a `figure*` with `\includegraphics[width=\textwidth]` applies no scaling and the
figure text reaches the page at its designed size: 8.4 pt labels, 7 pt ticks. Pass `--width`,
`--height`, and `--fontscale` to re-render at another size, but keep the canvas width equal to the
width of the LaTeX box, or the text shrinks with it.
