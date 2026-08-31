# Feasible but Not Safe: Constraint Violations and Report-Channel Attacks in Learned Cell-Free ISAC Association

Analysis code and machine-readable results for our paper at the **ACM Workshop on Wireless of the
Students, by the Students, and for the Students (ACM S3 '26)**, co-located with ACM MobiCom 2026,
Austin, TX, USA.

> Mehdi Zafari, Iman Mohammadi, and A. Lee Swindlehurst.
> *Feasible but Not Safe: Constraint Violations and Report-Channel Attacks in Learned Cell-Free ISAC
> Association.* ACM S3 '26.

A learned scheduler for distributed cell-free integrated sensing and communication (ISAC) maps
self-reported per-access-point (AP) statistics to an association plan in one forward pass. That
efficiency rests on two unstated assumptions: that hard constraints enforced only as soft training
penalties still hold at inference, and that the reported statistics are truthful. We measure both on
the released [ASSENT](https://github.com/LS-Wireless/ASSENT-CellFree-ISAC) system and show they fail
in a linked way.

* Despite per-decision F1 above 0.84, about 70% to 74% of inferred plans break a hard constraint,
  because per-decision accuracy is blind to joint feasibility.
* Projecting onto the nearest feasible plan restores legality at a cost of under 0.1% of the
  objective, and a trivial greedy repair does about as well.
* Feasibility is not safety. One AP running a false data injection (FDI) attack cannot steal users
  (under 0.2 gain) but can drive infeasibility to 96%.
* What a lie corrupts dictates the defense. Objective-faking lies are undone by projection;
  constraint-faking lies are not, and need a cheap cross-AP consistency check.

Every number and figure in the paper is computed from the JSON files committed under `results/`, so
you can inspect or re-plot all of it without re-running the pipeline.

## Layout

```
paths.py       repository root paths, and puts the upstream ASSENT clone on sys.path
feasibility/   the exact hard-constraint checker (7c to 7k) and the objective evaluation
attacks/       false data injection attacks by one compromised AP (falsify.py) plus invariant tests
defenses/      nearest-feasible projection (scipy HiGHS, plus the Gurobi reference) and the
               cross-AP consistency check
experiments/   the measurement drivers, the figure scripts, and the correctness gates
results/       machine-readable outputs behind every number in the paper, plus the checkpoint
               summaries (tuned decode thresholds and test split) for each backbone
figures/       the three figures as published in the paper, in vector PDF and PNG
docs/          the methodology and validation record, and the runbook
```

The repository is scoped to the camera-ready paper: every script and result file here backs a claim
in it. Exploratory work that did not make the paper lives in our private working repository.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

That is enough to regenerate every figure. Re-running the measurement pipeline additionally needs
the released ASSENT code and data:

```bash
git clone https://github.com/LS-Wireless/ASSENT-CellFree-ISAC upstream/ASSENT-CellFree-ISAC
pip install -r upstream/ASSENT-CellFree-ISAC/requirements.txt
```

`paths.py` looks for the clone at `upstream/ASSENT-CellFree-ISAC`. If you keep it elsewhere, set
`ASSENT_UPSTREAM` to that path instead. Nothing in this repository ever writes to the upstream tree.

Two more inputs are needed for a full re-run and are deliberately not in git:

1. **Trained checkpoints.** The ASSENT release ships training logs but not the `.pt` weights.
   Retrain with `experiments/train_model.py --arch gatv2 --tag repro`, or get the weights from the
   ASSENT authors and drop them into `results/checkpoints_repro/<arch>/best_model.pt`. The small
   `summary.json` files that carry the validation-tuned decode thresholds and the test split *are*
   committed, so the decode is fully specified.
2. **Gurobi**, only if you want the Gurobi reference formulation in `defenses/project.py` or want to
   regenerate MILP labels. Every number in the paper comes from `defenses/project_scipy.py`, which
   solves the same ILP with scipy HiGHS and needs no license.

Everything runs on CPU. `docs/RUNBOOK.md` has the staged, step-by-step notes.

## Figures

The three figures in the paper are committed under `figures/`, as both vector PDF (what the paper
embeds) and PNG. Each is a two-panel figure spanning the full text width:

| File | Contents |
|---|---|
| `figures/fig1_feasibility.pdf` | Fig. 1. Left: per-decision F1 stays near 1 while joint legality collapses. Right: MILP, raw GNN, ILP projection, and greedy repair objectives |
| `figures/fig2_attack.pdf` | Fig. 2. The signal FDI attack across three attacker positions: users served, and radio chains used |
| `figures/fig3_defense.pdf` | Fig. 3. Left: how report lies break feasibility. Right: the cross-AP consistency check |

Regenerating them needs no model, dataset, or solver. These read `results/*.json` only:

```bash
python experiments/make_fig1_feasibility.py
python experiments/make_fig2_attack.py
python experiments/make_fig3_defense.py
```

Each figure is drawn at its final printed size (7.00 in, the full `\textwidth`) rather than on a
canvas that LaTeX then shrinks, so include them in a `figure*` at `width=\textwidth` and the text
lands on the page at the size it was designed for. Axis labels are 8.4 pt and tick labels 7 pt,
under the 9 pt body text.

`--width`, `--height`, and `--fontscale` let you re-render at another size; keep the canvas width
equal to the width of the LaTeX box, or the type shrinks with it. Shared styling lives in
`experiments/figstyle.py`: colorblind-safe palette, series separated by marker and line style as
well as color so they survive grayscale, and TrueType embedding so no Type 3 fonts reach the
publisher.

## From the paper to the code

| Paper claim | Script | Result file |
|---|---|---|
| 73.5% / 69.2% / 72.8% of plans illegal (GATv2 / NNConv / TransformerConv), F1 at least 0.84 | `experiments/run_repair.py --arch <arch> --tag repro` | `results/repair_<arch>_repro.json` |
| Violation breakdown: RF budget 54% to 58%, co-serving correlated users 20% to 26%, mode errors 18% to 20% | same | `raw_viol_rates` in the same files |
| One extra link keeps F1 at 0.995 while 93% of plans turn illegal (Fig. 1, left) | `experiments/f1_decouple.py` | `results/f1_decouple.json` |
| Raw GNN 4.293 versus MILP optimum 4.286; ILP projection retains 99.95%, greedy 99.91%; 1.17 excess links per invalid plan (Fig. 1, right) | `experiments/analyze_difficulty.py --arch gatv2` | `results/difficulty_gatv2.json` |
| Objective loss near 0.06% across the whole communication and sensing trade-off, paired 95% CI within [0.04%, 0.11%] | `experiments/alpha_strata.py` | `results/alpha_strata.json` |
| Signal lie: attacker users move by at most 0.2 up to 32x, across three AP positions (Fig. 2) | `experiments/run_attack.py --attack gain_steal --a_star <0,3,6>` | `results/attack_gatv2_repro_gain_steal_a{0,3,6}.json` |
| The same holds on the other two backbones | same | `results/attack_{nnconv,transformer}_repro_gain_steal_a0.json` |
| A greedy user-targeting heuristic performs no better than the steal variant | `experiments/run_attack.py --attack gain_targeted --a_star <0,3,6>` | `results/attack_gatv2_repro_gain_targeted_a{0,3,6}.json` |
| Interference lie pushes broken plans from 73% to 84% (Fig. 3, left) | `experiments/run_attack.py --attack corr_deflation --defense` | `results/attack_gatv2_repro_corr_deflation_def.json` |
| Sensing lie drives infeasibility to 96% and hoards sensing jobs | `experiments/run_attack.py --attack sensing_inflation --a_star 3` | `results/attack_gatv2_repro_sensing_inflation_a3.json` |
| Two colluding APs reach 91% | `experiments/run_collusion.py --attack corr_deflation` | `results/collusion_corr_deflation.json` |
| Projection fully cancels the sensing lie (0% residual) but leaves 12% after the interference lie | `experiments/run_attack.py ... --defense` | `results/attack_gatv2_repro_sensing_inflation_a3_def.json`, `..._corr_deflation_def.json` |
| Cross-AP check catches the attacker about 99.5% of the time with about 1% false alarms, cutting the residual from 12% to 4% (Fig. 3, right) | `experiments/run_crossap.py` | `results/crossap_defense.json` |

## Correctness gates

The measurement engine is only as good as its checker, so the checker is tested against ground
truth rather than trusted:

```bash
python experiments/validate_core.py    # oracle: 1000 MILP-optimal plans must show 0 violations,
                                       # plus an objective cross-check against the upstream function
python experiments/verify_project.py   # brute-force enumeration on small instances: the projection
                                       # must return the provably optimal feasible plan, not just a
                                       # feasible one
python attacks/test_attacks.py         # attack invariants: k=1 is an exact no-op, and only the
                                       # attacker's own reported slice ever changes
python defenses/test_project.py        # projected plans are always feasible, on clean and attacked
                                       # inputs alike
python experiments/inspect_scenario.py --arch gatv2 --verify   # per-AP attribution of every violation
```

`docs/methodology-and-validation.md` records how each number was produced, how it was checked, and
which earlier claims we corrected along the way.

## Scope and limitations

These are the paper's own caveats, restated so nobody over-reads the numbers here.

* Everything is measured at the single released ASSENT configuration: 8 APs with 4 RF chains each,
  10 users, 4 targets, and that deployment geometry. Scale generality is untested. Labeling fresh
  scenarios at other sizes needs the exact MILP solver, which is the expensive step the learned
  scheduler exists to avoid. We attempted a synthetic scale audit and could not validate it, so the
  paper states this as an open limitation rather than shipping a shaky result. See
  `docs/methodology-and-validation.md` for what that attempt showed.
* Restoring feasibility for under 0.1% is a property of this distribution, where the network's
  violations are marginal (1.17 excess links on average). It is not a general claim about
  projection cost.
* The 0% infeasibility after projection is true by construction. The contribution is the measured
  cost of doing it, and the finding that a trivial greedy repair gets within 0.03% to 0.10% of the
  exact ILP. The ILP is not the contribution.
* The attacker is allowed to scale reported values up to 32x. That is a deliberate worst case.
  Unclipped extremes like these would plausibly be caught as out-of-distribution anomalies in a real
  deployment.
* We evaluate constraints (7c) through (7k) of the ASSENT MILP. The optional capacity caps (7l) to
  (7n) are excluded.

## Citation

```bibtex
@inproceedings{zafari2026feasible,
  author    = {Zafari, Mehdi and Mohammadi, Iman and Swindlehurst, A. Lee},
  title     = {Feasible but Not Safe: Constraint Violations and Report-Channel Attacks
               in Learned Cell-Free {ISAC} Association},
  booktitle = {Proceedings of the ACM Workshop on Wireless of the Students, by the Students,
               and for the Students (ACM S3 '26)},
  year      = {2026},
  address   = {Austin, TX, USA},
  publisher = {ACM}
}
```

The system under study:

```bibtex
@inproceedings{zafari2026assent,
  author    = {Zafari, Mehdi and Swindlehurst, A. Lee},
  title     = {{ASSENT}: Learning-Based Association Optimization for Distributed Cell-Free {ISAC}},
  booktitle = {Proc. IEEE International Conference on Communications (ICC)},
  year      = {2026},
  note      = {arXiv:2511.09992}
}
```

## License

MIT, see [LICENSE](LICENSE). The upstream ASSENT code is separately licensed by its authors and is
not redistributed here.
