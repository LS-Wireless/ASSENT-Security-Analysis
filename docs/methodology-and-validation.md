# Methodology and validation record

How every number in the paper was produced and how it was checked.

All results come from 100-epoch reproduction models (`results/checkpoints_repro/`, trained with the
ASSENT authors' own configs and seeds) evaluated on the held-out 10% test split of 1000 scenarios.
Setup: 8 APs with 4 RF chains each, 10 users, 4 targets. Three GNN backbones: NNConv,
TransformerConv, GATv2Conv.

---

## 1. Reproducing the released system

Before measuring anything, we confirm we are measuring the real ASSENT. All 15 per-decision F1
scores land within **0.015** of the published Table II, most within 0.01, and the GATv2 MILP
objective matches exactly at 4.286.

| backbone | tau | x | s | y_tx | y_rx | max abs delta vs paper |
|---|---|---|---|---|---|---|
| nnconv | .981/.985 | .910/.907 | .970/.975 | .836/.841 | .904/.919 | 0.015 |
| transformer | .984/.979 | .928/.928 | .975/.978 | .867/.868 | .922/.911 | 0.011 |
| gatv2 | .977/.983 | .911/.908 | .978/.981 | .841/.848 | .914/.918 | 0.007 |

Threshold tuning is done on the validation split, not the test split. The train/test split is a
clean disjoint hold-out. The F1 metric matches the authors' own implementation.

One decode detail worth recording: the released evaluation script reuses NNConv's thresholds for all
three models, which would drop GATv2's y_rx to .892 against the published .918. The paper's numbers
use proper per-model thresholds, and so do we. Those tuned thresholds are committed in
`results/checkpoints_repro/<arch>/summary.json`.

Scripts: `experiments/train_model.py`, `experiments/eval_repro.py`, `experiments/run_authors_eval.py`.
Result: `results/repro_compare.json`.

---

## 2. The feasibility audit

The MILP has hard rules every plan must obey. `feasibility/checker.py` implements constraints (7c)
through (7k) exactly as `solve_problem` states them, evaluated on a decoded binary plan. The
auxiliary variables (z, w, v) are pure linearizations of products, not decision constraints, so they
are excluded. The optional capacity caps (7l) to (7n) are also excluded, and the paper says so.

The authors' own evaluation never checks this. It reports F1 and objective, and leaves feasibility
as an unimplemented comment.

| backbone | plans breaking at least one rule | top broken rules |
|---|---|---|
| nnconv | **69.2%** | RF budget 54%, correlation 23%, mode coupling 19% |
| transformer | **72.8%** | RF budget 58%, target caps 24%, correlation 21% |
| gatv2 | **73.5%** | RF budget 58%, correlation 26%, mode coupling 20% |

**Why the checker can be trusted.** Three independent gates:

* **Oracle.** The MILP's own optimal plans are feasible by construction, so the checker must report
  zero violations on them. It reports **0 out of 1000**. It never false-flags.
* **Coverage.** Every invalid plan is attributed to a specific AP or target, with **0 unexplained**.
* **Scope.** All 5 decision variables and all 10 rule families are covered.

Scripts: `experiments/validate_core.py`, `experiments/inspect_scenario.py --verify`,
`experiments/run_repair.py`. Results: `results/repair_<arch>_repro.json`.

---

## 3. Why F1 is blind to legality

This is the general claim, and it is separated deliberately from the ASSENT-specific cost number.

Starting from the optimal plans (F1 = 1, all legal), adding one random excess link keeps per-head F1
at **0.995** while **93%** of plans turn illegal; two links reach 99%. With D = 156 binary choices
per plan, flipping one moves a per-choice score by about 1/D but can break legality outright. With D
in the hundreds, the two metrics pull apart.

Script: `experiments/f1_decouple.py`. Result: `results/f1_decouple.json`. Figure: Fig. 1 left.

---

## 4. The cost of restoring feasibility

`defenses/project_scipy.py` solves, after the GNN, a small ILP that snaps the plan to the nearest
legal one: maximum agreement with the GNN's confidences subject to the same hard rules. It is built
on scipy HiGHS and needs no license; the formulation is identical to the Gurobi reference in
`defenses/project.py`.

**Correctness gates:**

* **Brute force.** On small instances we enumerate every feasible plan. The solver returns the
  provably optimal one in **33 out of 33** instances, not merely a feasible one.
* **Feasibility.** **0 out of 1000** projected plans are invalid.
* **Formulation audit.** All constraint families and the objective sign match the Gurobi reference
  and the checker term for term.

**Results** (1000 scenarios): invalid drops to 0.0% for all three backbones, with the objective
within 0.05% of the MILP optimum. Accuracy is preserved or slightly improved after projection (for
GATv2: tau .977 to .982, x .911 to .914, s .978 to .982). Cost is about 21 ms per scenario.

**Why the cost is small, and why the ILP is not the contribution.** The data is not loose:
constraints genuinely bind, with mean AP radio load 3.2 out of 4, 70% of APs at full budget, and a
saturated AP in every scenario. But the network's violations are marginal, at **1.17 excess
associations** per invalid plan, with 56% of invalid plans breaking at most 2 rules. Those extras
are nearly worthless. Against the MILP optimum: raw GNN +0.16% (inflated by the rules it breaks),
ILP projection **-0.05%** (retains 99.95%), and a trivial drop-only greedy **-0.09%** (retains
99.91%, feasible on all 1000). The ILP beats the greedy by only 0.03% to 0.10% across the three
backbones.

Three framing traps we deliberately avoid, and the paper avoids them too:

1. "73% to 0%" is guaranteed by construction. Projecting onto the feasible set must give zero
   violations. It is not the contribution. The paper leads with the cost, not the 0%.
2. The fix is cheap because the GNN over-allocates marginal extras, not because the ILP is clever.
   The greedy is reported as the baseline for exactly this reason.
3. The cost is measured only on the released ASSENT distribution. See §7.

**Robustness across the trade-off.** The cost stays near 0.06% across the communication and sensing
trade-off: sensing-heavy 0.070% [0.038, 0.104], balanced 0.060%, communication-heavy 0.059%, overall
0.063% [0.039, 0.092], bootstrap 95% CI. It is not an artifact of one operating point.

**On the residual optimality gap.** In 78 of 1000 scenarios the projected feasible objective edges
above the MILP objective by at most +0.012. The distribution straddles zero (mean -0.002, median
-0.00005). The cause is that ASSENT's MILP maximizes a McCormick-linearized surrogate while we
evaluate the true non-linearized utility, so the MILP plan is optimal for the surrogate, not for the
evaluation metric, and a feasible plan can edge above it in near-ties. A code audit confirmed no
missing constraint. The correct statement, and the one the paper makes, is that repair is
statistically indistinguishable from the optimum while guaranteeing feasibility. Never that repair
beats the optimum.

Scripts: `experiments/run_repair.py`, `experiments/analyze_difficulty.py`,
`experiments/alpha_strata.py`, `experiments/verify_project.py`. Results:
`results/difficulty_<arch>.json`, `results/alpha_strata.json`. Figure: Fig. 1 right.

---

## 5. The attacks

The central server sees only the summary statistics each AP reports about itself, and cannot verify
them. A single compromised AP lies. The harness decodes on the **falsified** inputs the server sees
and evaluates impact on the **true** channel, because the physics is unchanged by the lie. Attacks
touch only the attacker's own reported slice, which is unit-tested in `attacks/test_attacks.py`.

All numbers below are n=1000, GATv2, with the attacker identity swept over a* in {0, 3, 6}, since
what an AP can acquire depends on network geometry.

**Signal lie (stealing).** Two designs: `gain_targeted` inflates the attacker's gain to its own
strongest user; `gain_steal` is the smarter design, inflating the gain to the most contested user it
does not already dominate. The GNN is robust to both. Across all attacker positions, all three
backbones, and inflation up to 32x, the attacker's distinct users move by at most **0.2** (a*=0:
1.57 to 1.60; a*=3: 2.76 to 2.92; a*=6: 2.49 to 2.58). The smart design is no better than the naive
one. Radio load does rise modestly.

Two corrections against our earlier write-ups, both folded into the paper:

* The absolute "about 1.5 users" figure was an a*=0 artifact. Baseline capture is geometry
  dependent (1.57 / 2.76 / 2.49 for a*=0/3/6), so the sweep is reported rather than one position.
* The mechanism is **not** a z-score invariance argument. That applies only to a uniform lie.
  `gain_targeted` and `gain_steal` do shift features; the GNN is simply insensitive to a single AP's
  reported gain to one user. The uniform variant and the invariance claim have been removed from
  this repository.

**Interference lie (`corr_deflation`).** The attacker hides its interference so the server co-serves
users it should not. This pushes system infeasibility from **73% to 84%** but does not increase the
attacker's distinct-user count, which is non-monotonic (1.57 to 1.64 at k=2, back to 1.51 at k=32).

**Sensing lie (`sensing_inflation`).** A fake sensing report pushes infeasibility to **96%** (GATv2,
a*=3; 88% for NNConv) and lets the attacker hijack sensing-transmit roles (0.67 to 1.14).

**Collusion.** Two colluding APs deflating interference reach **91.5%** infeasible with combined
load 5.6 to 7.2, and still cannot steal users (3.36 to 3.82). More cheaters means more breakage, not
more theft.

**The honest takeaway.** Single-AP report falsification, gain or interference or sensing, naive or
smart, is not a user-theft attack. Its real effect is degrading feasibility, which at 96% illegal is
an operational denial problem, plus a modest radio-load grab.

Scripts: `experiments/run_attack.py`, `experiments/run_collusion.py`. Results:
`results/attack_*.json`, `results/collusion_corr_deflation.json`. Figures: Fig. 2, Fig. 3 left.

---

## 6. The defense taxonomy

This is the link between the two halves of the paper, and it is decided by **what a lie corrupts**.

* **Objective-only corruption.** The sensing lie corrupts G_sens, which feeds the objective but no
  hard constraint. The feasible set is untouched, so the projection has all the true constraint
  parameters and **fully restores legality: 0% residual**, even against the 96% attack.
* **Constraint corruption.** The interference lie corrupts S_comm, which defines the no-clash rule
  itself. The projection then enforces the wrong feasible set, so about **12%** of the
  mathematically legal plans still violate a rule in physical reality. The attacker is pushed back
  only from 1.5 to 1.3 users.
* **Neither.** The signal lie cannot fake a constraint (the radio budget is not reportable) and does
  not meaningfully move the objective, so it fails on both paths.

So exactly one case needs an input-side check, and that is what `defenses/crossap.py` is for. Under
a symmetric AP topology, reports are redundant: every AP reports interference for the same user
pairs, and a genuinely clashing pair looks correlated to many APs. The server flags any AP whose
interference sits more than one median absolute deviation below the group median and lifts the
flagged values toward the group, using reported data only, with no raw CSI and no oracle.

It detects the cheater about **99.5%** of the time (99% to 100% for k >= 4) with about **1%** false
alarms, cutting the residual broken share from **12% to 4%**. It bounds the attacker rather than
eliminating it: a large lie is almost always caught, a small lie evades detection but barely
perturbs the system. The residual 4% is natural cross-AP variance, not undetected lies.

One earlier claim we corrected: the projection's "backfire" on the interference attacker, where the
harder the AP lies the more illegally co-served users get stripped, is a property of **any** repair
that uses the true correlations. A trivial greedy drop backfires identically. It is not an ILP
property and the paper does not credit the ILP for it.

Scripts: `experiments/run_crossap.py`, `experiments/run_attack.py --defense`. Results:
`results/crossap_defense.json`, `results/attack_*_def.json`. Figure: Fig. 3 right.

---

## 7. Scope and limitations

* **Single network size.** Everything is measured at the released 8 AP / 10 user / 4 target
  configuration and geometry. On denser or asymmetric instances, or with a different mix of
  violation types, the repair cost could be higher. We attempted a synthetic scale audit by feeding
  the size-agnostic GNN freshly generated scenarios at other sizes. It could not be validated: even
  at the released size, generated infeasibility swung between 74% and 100% depending on the lambda
  choice, and the violation structure did not match the real test set. The GNN is sensitive to its
  exact input distribution in ways we cannot replicate without re-running the full pipeline.
  Properly labeled new-scale data needs the exact MILP solver, which is the expensive step the
  learned scheduler exists to avoid. We therefore state scale generality as an open limitation in
  the paper rather than shipping a shaky synthetic result, and the exploratory script is not part of
  this repository.
* **Worst-case attacker.** We let the attacker scale reported values up to 32x. Unclipped extremes
  like these would plausibly be filtered as out-of-distribution anomalies in a real deployment. The
  paper says so.
* **Constraint scope.** We evaluate (7c) through (7k). The optional capacity caps (7l) to (7n) are
  excluded.
* **Objective reimplementation.** `feasibility/objective.py` reimplements the upstream objective
  rather than importing it, because the upstream version sums a generator with `np.sum`, which does
  not sum correctly under numpy 2.x. Everything else matches upstream exactly, and
  `validate_core.py` cross-checks ours against the upstream function on its bug-free path and
  against the dataset's stored objective values.

---

## 8. Verification summary

| Claim | How it was verified | Result |
|---|---|---|
| The code reproduces the published system | 100-epoch retrain, 3 backbones, against Table II | within 0.015 on all F1 |
| The checker is correct and complete | oracle on MILP-optimal plans, coverage, rule scope | 0 of 1000, 0 unexplained, 10 of 10 rules |
| The projection finds the optimal feasible plan | brute-force enumeration on small instances | 33 of 33 |
| The projection is feasible by construction | feasibility check on 1000 projected plans | 0 invalid |
| Attacks touch only the attacker's own reports | unit-tested invariants, k=1 is an exact no-op | pass |
| The residual optimality gap is benign | with and without reward diagnostic, plus code audit | MILP linearization, not a bug |

**Reproduce them:** `experiments/validate_core.py`, `experiments/verify_project.py`,
`attacks/test_attacks.py`, `defenses/test_project.py`,
`experiments/inspect_scenario.py --arch gatv2 --verify`.
