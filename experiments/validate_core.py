"""Validate the whole stack before any training:
  (A) SMOKE: upstream data pipeline + all 3 model forwards run under our env (torch 2.11 / PyG 2.8).
  (B) ORACLE: our feasibility checker reports 0 violations on MILP-optimal solutions.
  (C) OBJECTIVE: our corrected objective matches upstream (add_reward=False) and tracks opt_objVal;
                 also probe the upstream tx_reward generator bug.

Run:  python experiments/validate_core.py   (from anywhere; paths handles imports)
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))        # so `import paths`, feasibility.*
import paths  # noqa: F401  (adds upstream + learning_v4 to sys.path, defines DECODE_THRESHOLDS)

import numpy as np
import torch

# our modules
from feasibility.checker import check_feasibility, audit_dataset
from feasibility.objective import compute_objective

# upstream (run data_loader from repo root)
os.chdir(paths.UPSTREAM)
from src.learning_v4.assent_data_utils_v4 import data_loader
from src.learning_v4.assent_data_v4 import GraphDataset
from src.learning_v4.assent_model_v4 import ASSENTGNN, ASSENTGNN_GAT, ASSENTGNN_Transformer
import src.utils.optimization_utils as opt

N = int(os.environ.get("VAL_N", "500"))
PASS = "\033[92mPASS\033[0m"; FAIL = "\033[91mFAIL\033[0m"
ok_all = True

print(f"\n=== Loading {N} samples ===")
res = data_loader(nsamps=N, CONSOLE_RUN=True)
A = np.asarray(res["G_comm"][0]).shape[0]
U = np.asarray(res["G_comm"][0]).shape[1]
T = np.asarray(res["G_sens"][0]).shape[-1]
N_RF = res["N_RF"]; K_tx = res["K_tx_milp"]; K_rx = res["K_rx_milp"]
print(f"A={A} U={U} T={T} | N_RF={N_RF} K_tx={K_tx} K_rx={K_rx} | nsamps={len(res['G_comm'])}")

# ---------- (A) SMOKE: model forwards ----------
print("\n=== (A) SMOKE: model forward on 1 graph (untrained) ===")
ds = GraphDataset(res["G_comm"], res["S_comm"], res["G_sens"], res["alpha"],
                  res["lambda_cu"], res["lambda_tg"], res["solution"], N_RF=N_RF,
                  device="cpu", K_tx_milp=K_tx, K_rx_milp=K_rx)
g0 = ds[0]
for name, Mod in [("NNConv", ASSENTGNN), ("GATv2", ASSENTGNN_GAT), ("Transformer", ASSENTGNN_Transformer)]:
    try:
        torch.manual_seed(0)
        m = Mod(hidden=128, layers=3, drop=0.2)
        m.eval()
        with torch.no_grad():
            out = m(g0)
        shapes = {k: tuple(v.shape) for k, v in out.items()}
        # expected: x_logit [A*U,1], tau [A,1], s [T,1], ytx/yrx [A*min(K,T),1]
        good = (out["tau_logit"].shape[0] == A and out["s_logit"].shape[0] == T
                and out["x_logit"].shape[0] == A * U)
        print(f"  {name:12s} {PASS if good else FAIL}  {shapes}")
        ok_all &= good
    except Exception as e:
        print(f"  {name:12s} {FAIL}  {type(e).__name__}: {e}")
        ok_all = False

# ---------- (B) ORACLE: feasibility on MILP solutions ----------
print(f"\n=== (B) ORACLE: feasibility of {N} MILP-optimal solutions (must be 0 violations) ===")
params_list = [{"N_RF": N_RF, "rho_thresh": 0.8, "S_comm": res["S_comm"][i],
                "K_tx": K_tx, "K_rx": K_rx, "C_rx": T} for i in range(len(res["solution"]))]
summary = audit_dataset(res["solution"], params_list)
print(f"  infeasible: {summary['infeasible_pct']:.3f}% of {summary['n_samples']}")
worst = {k: v for k, v in summary["violation_rate"].items() if v > 0}
if worst:
    print(f"  {FAIL}  per-constraint violation rate (%): {worst}")
    # show one offending sample for debugging
    for i, (sol, prm) in enumerate(zip(res["solution"], params_list)):
        r = check_feasibility(sol, prm)
        if r["infeasible"]:
            print(f"    e.g. sample {i}: {{k:v for k,v in r.items() if isinstance(v,int) and v>0}} -> "
                  f"{ {k: v for k, v in r.items() if isinstance(v, int) and v > 0} }")
            break
    ok_all = False
else:
    print(f"  {PASS}  all MILP solutions feasible (checker validated)")

# ---------- (C) OBJECTIVE cross-check ----------
print("\n=== (C) OBJECTIVE: ours vs upstream vs stored opt_objVal (first 5) ===")
maxerr_noreward = 0.0
gen_bug = False
for i in range(min(5, len(res["solution"]))):
    p = {"G_comm": res["G_comm"][i], "S_comm": res["S_comm"][i], "G_sens": res["G_sens"][i],
         "lambda_cu": res["lambda_cu"][i], "lambda_tg": res["lambda_tg"][i],
         "alpha": res["alpha"][i], "interf_penalty": 0.01}
    sol = res["solution"][i]
    ours = compute_objective(p, sol, add_reward=True)
    ours_nr = compute_objective(p, sol, add_reward=False)
    # upstream with add_reward=False avoids the buggy generator line
    up_nr = opt.compute_milp_objective(p, sol, add_reward=False)
    err = abs(ours_nr["obj_val"] - up_nr["obj_val"])
    maxerr_noreward = max(maxerr_noreward, err)
    # probe upstream add_reward=True (may hit the generator bug)
    up_full_str = ""
    try:
        up_full = opt.compute_milp_objective(p, sol, add_reward=True)
        up_full_str = f"upstream_full={up_full['obj_val']:.4f}"
    except Exception as e:
        gen_bug = True
        up_full_str = f"upstream_full ERROR({type(e).__name__})"
    print(f"  i={i} opt_objVal={res['opt_objVal'][i]:.4f} | ours_full={ours['obj_val']:.4f} | "
          f"ours_noreward={ours_nr['obj_val']:.4f} | {up_full_str}")

print(f"\n  max|ours_noreward - upstream_noreward| over 5 = {maxerr_noreward:.2e}  "
      f"{PASS if maxerr_noreward < 1e-6 else FAIL}")
ok_all &= (maxerr_noreward < 1e-6)
# Does ours_full reproduce opt_objVal? (it should if normalization matches data-gen)
errs_full = []
for i in range(min(50, len(res["solution"]))):
    p = {"G_comm": res["G_comm"][i], "S_comm": res["S_comm"][i], "G_sens": res["G_sens"][i],
         "lambda_cu": res["lambda_cu"][i], "lambda_tg": res["lambda_tg"][i],
         "alpha": res["alpha"][i], "interf_penalty": 0.01}
    o = compute_objective(p, res["solution"][i], add_reward=True)["obj_val"]
    errs_full.append(abs(o - res["opt_objVal"][i]))
print(f"  ours_full vs opt_objVal over 50: max={max(errs_full):.4f} mean={np.mean(errs_full):.4f} "
      f"(small => our objective reproduces the dataset's MILP objective)")
print(f"  upstream tx_reward generator bug present in this env: {gen_bug}")

print(f"\n=== OVERALL: {PASS if ok_all else FAIL} ===")
sys.exit(0 if ok_all else 1)
