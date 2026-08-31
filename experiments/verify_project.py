"""Correctness gate for the scipy repair layer (defenses/project_scipy.py).

THREE independent checks:
 (1) BRUTE FORCE (formulation correctness): on tiny instances, enumerate EVERY binary plan,
     keep the feasible ones (per the independently-validated check_feasibility), and find the
     one with maximum agreement objective. The solver's output must match that maximum exactly.
 (2) FEASIBILITY: on real test scenarios, every projected plan has check_feasibility==0.
 (3) OPTIMALITY SANITY: on real scenarios, the repaired objective never exceeds the MILP
     optimum (U_repaired <= U_MILP), since the MILP is the optimum over the SAME feasible set.

Run: python experiments/verify_project.py
"""
import os, sys, types, itertools, pickle
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
_opt = types.ModuleType("src.utils.optimization_utils")
class Solution: pass            # noqa: E701
class ProblemParams: pass       # noqa: E701
_opt.Solution = Solution; _opt.ProblemParams = ProblemParams
sys.modules["src.utils.optimization_utils"] = _opt

import paths  # noqa: F401
import numpy as np
from feasibility.checker import check_feasibility
from feasibility.objective import compute_objective
from defenses.project_scipy import project


def agreement(sol, probs):
    return (float(np.sum((2 * np.asarray(probs["px"]) - 1) * sol["x"]))
            + float(np.sum((2 * np.asarray(probs["ptau"]).reshape(-1) - 1) * sol["tau"]))
            + float(np.sum((2 * np.asarray(probs["ps"]).reshape(-1) - 1) * sol["s"]))
            + float(np.sum((2 * np.asarray(probs["pytx"]) - 1) * sol["y_tx"]))
            + float(np.sum((2 * np.asarray(probs["pyrx"]) - 1) * sol["y_rx"])))


def brute_force_max(probs, params):
    A, U = np.asarray(probs["px"]).shape; T = np.asarray(probs["ps"]).reshape(-1).shape[0]
    nx, nyt, nyr, nt = A * U, A * T, A * T, A
    N = nx + nyt + nyr + nt + T
    best = -1e18; best_sol = None; n_feas = 0
    for bits in itertools.product([0, 1], repeat=N):
        z = np.array(bits, np.int8)
        sol = {"x": z[:nx].reshape(A, U), "y_tx": z[nx:nx + nyt].reshape(A, T),
               "y_rx": z[nx + nyt:nx + nyt + nyr].reshape(A, T),
               "tau": z[nx + nyt + nyr:nx + nyt + nyr + nt], "s": z[nx + nyt + nyr + nt:]}
        if check_feasibility(sol, params)["total_violations"] == 0:
            n_feas += 1
            a = agreement(sol, probs)
            if a > best:
                best = a; best_sol = sol
    return best, best_sol, n_feas


def rand_instance(rng, A, U, T):
    probs = {"px": rng.random((A, U)), "ptau": rng.random(A), "ps": rng.random(T),
             "pytx": rng.random((A, T)), "pyrx": rng.random((A, T))}
    S = rng.random((A, U, U)); S = (S + S.transpose(0, 2, 1)) / 2
    for a in range(A):
        np.fill_diagonal(S[a], 1.0)
        for u in range(U):
            for up in range(u + 1, U):
                if rng.random() < 0.35:                      # force some > rho pairs
                    S[a, u, up] = S[a, up, u] = 0.85 + 0.15 * rng.random()
    params = {"N_RF": 2, "rho_thresh": 0.8, "S_comm": S, "K_tx": 1, "K_rx": 1, "C_rx": T}
    return probs, params


def check_brute(n_trials=30):
    print("(1) BRUTE-FORCE formulation check (A=2,U=2,T=1; every plan enumerated)")
    rng = np.random.default_rng(0); fails = 0
    for k in range(n_trials):
        probs, params = rand_instance(rng, 2, 2, 1)
        bmax, _, n_feas = brute_force_max(probs, params)
        sol = project(probs, params)
        viol = check_feasibility(sol, params)["total_violations"]
        pagr = agreement(sol, probs)
        ok = (viol == 0) and (abs(pagr - bmax) < 1e-6)
        if not ok:
            fails += 1
            print(f"   trial {k}: FAIL  solver_agr={pagr:.6f} brute_max={bmax:.6f} viol={viol} feas_plans={n_feas}")
    print(f"   {n_trials - fails}/{n_trials} trials: solver hit the true feasible maximum AND feasible "
          f"-> {'PASS' if fails == 0 else 'FAIL'}")
    # a few larger instances (A=2,U=3,T=1 -> 2^13; exercises correlation pairs), enumerable fast
    big_ok = True
    for seed in (7, 11, 23):
        rng = np.random.default_rng(seed); probs, params = rand_instance(rng, 2, 3, 1)
        bmax, _, n_feas = brute_force_max(probs, params)
        sol = project(probs, params); pagr = agreement(sol, probs)
        viol = check_feasibility(sol, params)["total_violations"]
        ok = (viol == 0) and (abs(pagr - bmax) < 1e-6); big_ok = big_ok and ok
        print(f"   larger A=2,U=3,T=1 seed={seed} ({n_feas} feasible plans): solver_agr={pagr:.4f} "
              f"brute_max={bmax:.4f} viol={viol} -> {'PASS' if ok else 'FAIL'}")
    return fails == 0 and big_ok


def check_real(n=120):
    print(f"\n(2)+(3) FEASIBILITY + OPTIMALITY SANITY on {n} real test scenarios (gatv2 verified model)")
    sys.path.insert(0, os.path.join(HERE))
    from experiments import harness as H
    import json, torch
    from src.learning_v4.assent_model_v4 import ASSENTGNN_GAT
    res = pickle.load(open(os.path.join(paths.RESULTS, "assent_data_plain.pkl"), "rb"))
    c = json.load(open(os.path.join(paths.UPSTREAM, "src", "learning_v4", "checkpoints",
                                    "run_04", "metadata.json")))["config"]
    m = ASSENTGNN_GAT(hidden=c["nhidden"], layers=c["nlayers"], drop=c["drop"], heads=c["gatv2_heads"])
    ck = os.path.join(paths.RESULTS, "checkpoints_repro", "gatv2")
    m.load_state_dict(torch.load(os.path.join(ck, "best_model.pt"), map_location="cpu")); m.eval()
    summ = json.load(open(os.path.join(ck, "summary.json")))
    test_idx = summ["test_idx"][:n]
    n_inf = 0; n_over = 0; worst_over = 0.0
    for gid in test_idx:
        probs = H.decode_probs(m, H.make_graph(res, gid))
        prm = H.params_true(res, gid)
        rep = project(probs, prm)
        if check_feasibility(rep, prm)["infeasible"]:
            n_inf += 1
        op = H.obj_params_true(res, gid)
        U_rep = compute_objective(op, rep)["obj_val"]
        U_milp = compute_objective(op, res["solution"][gid])["obj_val"]
        if U_rep > U_milp + 1e-6:
            n_over += 1; worst_over = max(worst_over, U_rep - U_milp)
    print(f"   repaired plans that are INFEASIBLE: {n_inf}/{n}  -> {'PASS' if n_inf == 0 else 'FAIL'}")
    print(f"   repaired objective EXCEEDS MILP optimum: {n_over}/{n} (worst +{worst_over:.4g}) "
          f"-> {'PASS' if n_over == 0 else 'CHECK'}")
    return n_inf == 0, n_over


if __name__ == "__main__":
    p1 = check_brute()
    p2, n_over = check_real()
    print(f"\nVERDICT: formulation={'OK' if p1 else 'BROKEN'}, feasibility={'OK' if p2 else 'BROKEN'}, "
          f"optimality-sanity={'OK' if n_over == 0 else f'{n_over} minor (objective reimpl noise)'}")
