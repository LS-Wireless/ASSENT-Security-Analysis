"""Verify the E3 projection defense guarantees on real GNN outputs (clean + attacked):
  (1) projected solution is ALWAYS feasible (check_feasibility == 0),
  (2) attacker RF load is bounded by N_RF regardless of the lie.
Run: python defenses/test_project.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import paths  # noqa: F401
import numpy as np

from experiments import harness as H
from feasibility.checker import check_feasibility
from defenses.project import project
from attacks.falsify import gain_inflation_targeted

PASS = "PASS"; FAIL = "FAIL"; ok = True
N = 40
res = H.load_results(nsamps=10000)
model, thr, summ = H.load_our_model("nnconv")
idx = summ["test_idx"][:N]
N_RF_max = int(np.asarray(res["N_RF"]).max())

bad_feas = 0; bad_cap = 0; n = 0
for gid in idx:
    for kappa in [1.0, 16.0]:                       # clean and strongly attacked
        Gc2, Sc2, Gs2 = gain_inflation_targeted(res["G_comm"][gid], res["S_comm"][gid], res["G_sens"][gid], 0, kappa)
        g = H.make_graph(res, gid, Gc=Gc2, Sc=Sc2, Gs=Gs2)
        probs = H.decode_probs(model, g)
        prm = H.params_true(res, gid)
        dsol = project(probs, prm)
        n += 1
        if check_feasibility(dsol, prm)["infeasible"]:
            bad_feas += 1
        cap = dsol["x"][0].sum() + dsol["y_tx"][0].sum()
        if cap > N_RF_max:
            bad_cap += 1

print(f"projected {n} cases (clean+attacked).")
def chk(name, cond):
    global ok; print(f"  [{PASS if cond else FAIL}] {name}"); ok &= cond
chk(f"all projected solutions feasible (0/{n} infeasible)", bad_feas == 0)
chk(f"attacker RF load <= N_RF in all cases (0/{n} over)", bad_cap == 0)
print(f"\n=== DEFENSE GUARANTEES: {PASS if ok else FAIL} ===")
sys.exit(0 if ok else 1)
