"""Is the dataset easy, and why is the repair layer nearly free?  Four measurements:

(A) CONSTRAINT TIGHTNESS in the MILP optima -- how loaded are the APs? If the optimum rarely
    saturates the radio budget, there is slack and repair is cheap because the GNN's extra
    associations are things the optimum deliberately left off.
(B) VIOLATION STRUCTURE in the raw GNN plans -- how many rules each invalid plan breaks, and by
    how much (RF overflow). Mostly "+1 over budget" => trivially fixable.
(C) VALUE of the violations -- how much objective the extra (illegal) decisions actually add.
(D) ILP repair vs a TRIVIAL drop-only greedy repair -- if greedy ~ ILP ~ MILP, the data is
    forgiving (any feasible projection works) and the ILP is not doing the heavy lifting; if the
    ILP clearly beats greedy, the structure matters.

Run: python experiments/analyze_difficulty.py --arch gatv2 --n 1000
"""
import os, sys, types, json, pickle, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
_opt = types.ModuleType("src.utils.optimization_utils")
class Solution: pass            # noqa: E701
class ProblemParams: pass       # noqa: E701
_opt.Solution = Solution; _opt.ProblemParams = ProblemParams
sys.modules["src.utils.optimization_utils"] = _opt

import paths  # noqa: F401
import numpy as np
import torch
from experiments import harness as H
from feasibility.checker import check_feasibility
from feasibility.objective import compute_objective
from defenses.project_scipy import project
from src.learning_v4.assent_model_v4 import ASSENTGNN, ASSENTGNN_GAT, ASSENTGNN_Transformer

ARCH_RUN = {"nnconv": "run_03", "gatv2": "run_04", "transformer": "run_02"}
HEADS = ["tau", "x", "s", "ytx", "yrx"]


def load_model(arch):
    c = json.load(open(os.path.join(paths.UPSTREAM, "src", "learning_v4", "checkpoints",
                                    ARCH_RUN[arch], "metadata.json")))["config"]
    if arch == "gatv2":
        m = ASSENTGNN_GAT(hidden=c["nhidden"], layers=c["nlayers"], drop=c["drop"], heads=c["gatv2_heads"])
    elif arch == "transformer":
        m = ASSENTGNN_Transformer(hidden=c["nhidden"], layers=c["nlayers"], drop=c["drop"],
                                  heads=c["transformer_heads"], beta=c["transformer_beta"])
    else:
        m = ASSENTGNN(hidden=c["nhidden"], layers=c["nlayers"], drop=c["drop"])
    ck = os.path.join(paths.RESULTS, "checkpoints_repro", arch)
    m.load_state_dict(torch.load(os.path.join(ck, "best_model.pt"), map_location="cpu")); m.eval()
    s = json.load(open(os.path.join(ck, "summary.json")))
    return m, {k: s[f"thr_{k}"] for k in HEADS}, s["test_idx"]


def _bcast(v, n):
    a = np.asarray(v, float); return np.full(n, a) if a.ndim == 0 else a


def greedy_drop(raw, probs, params):
    """Trivial feasibility-by-DELETION: only ever remove offending decisions (lowest GNN prob
    first), never add. Iterate until feasible. A weak baseline -- if the ILP only ties this,
    the violations were pure over-allocation."""
    sol = {k: np.array(raw[k], copy=True) for k in ["x", "tau", "s", "y_tx", "y_rx"]}
    px, pytx, pyrx = probs["px"], probs["pytx"], probs["pyrx"]
    A, U = sol["x"].shape; T = sol["s"].shape[0]
    rho = float(params["rho_thresh"]); S = np.asarray(params["S_comm"])
    N_RF = _bcast(params["N_RF"], A); K_tx = _bcast(params["K_tx"], T)
    K_rx = _bcast(params["K_rx"], T); C_rx = _bcast(params["C_rx"], A)
    for _ in range(60):
        if not check_feasibility(sol, params)["infeasible"]:
            break
        for a in range(A):                                  # mode coupling
            if sol["tau"][a] == 1: sol["y_rx"][a, :] = 0
            else: sol["x"][a, :] = 0; sol["y_tx"][a, :] = 0
        for a in range(A):                                  # correlation
            served = np.where(sol["x"][a] == 1)[0]
            for ii in range(len(served)):
                for jj in range(ii + 1, len(served)):
                    u, v = served[ii], served[jj]
                    if S[a, u, v] > rho and sol["x"][a, u] and sol["x"][a, v]:
                        sol["x"][a, u if px[a, u] < px[a, v] else v] = 0
        for a in range(A):                                  # RF budget
            load = sol["x"][a].sum() + sol["y_tx"][a].sum()
            while load > N_RF[a]:
                cand = [("x", u, px[a, u]) for u in range(U) if sol["x"][a, u]] + \
                       [("t", t, pytx[a, t]) for t in range(T) if sol["y_tx"][a, t]]
                if not cand: break
                kind, idx, _ = min(cand, key=lambda c: c[2])
                sol["x"][a, idx] = 0 if kind == "x" else sol["x"][a, idx]
                if kind == "t": sol["y_tx"][a, idx] = 0
                load -= 1
            while sol["y_rx"][a].sum() > C_rx[a]:            # Rx capacity
                t = min([(t, pyrx[a, t]) for t in range(T) if sol["y_rx"][a, t]], key=lambda c: c[1])[0]
                sol["y_rx"][a, t] = 0
        for t in range(T):                                  # target scheduling
            ntx, nrx = sol["y_tx"][:, t].sum(), sol["y_rx"][:, t].sum()
            if sol["s"][t] == 1 and (ntx == 0 or nrx == 0): sol["s"][t] = 0
            if sol["s"][t] == 0:
                sol["y_tx"][:, t] = 0; sol["y_rx"][:, t] = 0
            else:
                while sol["y_tx"][:, t].sum() > K_tx[t]:
                    aa = min([(a, pytx[a, t]) for a in range(A) if sol["y_tx"][a, t]], key=lambda c: c[1])[0]
                    sol["y_tx"][aa, t] = 0
                while sol["y_rx"][:, t].sum() > K_rx[t]:
                    aa = min([(a, pyrx[a, t]) for a in range(A) if sol["y_rx"][a, t]], key=lambda c: c[1])[0]
                    sol["y_rx"][aa, t] = 0
    return sol


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arch", default="gatv2")
    ap.add_argument("--n", type=int, default=1000); args = ap.parse_args()
    res = pickle.load(open(os.path.join(paths.RESULTS, "assent_data_plain.pkl"), "rb"))
    model, thr, test_idx = load_model(args.arch); test_idx = test_idx[:args.n]
    N_RF = int(np.asarray(res["N_RF"]).max())

    ld_milp = []; sat_ap = []; sat_scn = []; users_served = []; alphas = []     # (A) tightness
    nviol = []; rf_over = []                                                     # (B) structure
    d_raw = []; d_rep = []; d_grd = []; uml = []; n_inf = 0                      # (C)/(D) value
    grd_inf = 0
    for gid in test_idx:
        lab = res["solution"][gid]
        xm = np.asarray(lab.x)
        load = xm.sum(1) + np.asarray(lab.y_tx).sum(1)            # per-AP load in MILP optimum
        ld_milp += list(load); sat_ap += list((load >= N_RF).astype(int))
        sat_scn.append(int((load >= N_RF).any())); users_served.append(int((xm.sum(0) > 0).sum()))
        alphas.append(float(res["alpha"][gid]))

        probs = H.decode_probs(model, H.make_graph(res, gid)); raw = H.sol_from_probs(probs, thr)
        prm = H.params_true(res, gid); op = H.obj_params_true(res, gid)
        r = check_feasibility(raw, prm)
        if r["infeasible"]:
            n_inf += 1
            nviol.append(r["total_violations"])
            rload = raw["x"].sum(1) + raw["y_tx"].sum(1)
            rf_over.append(int(np.clip(rload - N_RF, 0, None).sum()))
        um = compute_objective(op, lab)["obj_val"]
        ur = compute_objective(op, raw)["obj_val"]
        rep = project(probs, prm); up = compute_objective(op, rep)["obj_val"]
        grd = greedy_drop(raw, probs, prm)
        if check_feasibility(grd, prm)["infeasible"]: grd_inf += 1
        ug = compute_objective(op, grd)["obj_val"]
        d_raw.append(ur - um); d_rep.append(up - um); d_grd.append(ug - um); uml.append(um)

    n = len(test_idx)
    print(f"\n=== Difficulty analysis: {args.arch}, {n} scenarios ===")
    print("(A) CONSTRAINT TIGHTNESS in MILP optima:")
    print(f"    mean AP radio load = {np.mean(ld_milp):.2f} / {N_RF}   "
          f"APs at full budget = {100*np.mean(sat_ap):.1f}%   "
          f"scenarios with >=1 saturated AP = {100*np.mean(sat_scn):.1f}%")
    print(f"    mean users served = {np.mean(users_served):.2f} / 10     mean alpha = {np.mean(alphas):.2f}")
    print("(B) VIOLATION STRUCTURE in raw GNN plans (infeasible ones):")
    if nviol:
        nv = np.array(nviol)
        print(f"    violations per invalid plan: mean={nv.mean():.2f} median={np.median(nv):.0f} "
              f"max={nv.max()}  (==1: {100*np.mean(nv==1):.0f}%, <=2: {100*np.mean(nv<=2):.0f}%)")
        print(f"    mean total RF overflow per invalid plan = {np.mean(rf_over):.2f} extra associations")
    Um = float(np.mean(uml))    # true percentages = 100 * (mean objective gap) / (mean MILP objective)
    print(f"(C) VALUE of the violations / repair cost (% of the MILP optimum, mean U_milp={Um:.3f}):")
    print(f"    raw GNN (infeasible):  {100*np.mean(d_raw)/Um:+.3f}%   (extras inflate the score)")
    print(f"    ILP repair:            {100*np.mean(d_rep)/Um:+.3f}%   (retains {100+100*np.mean(d_rep)/Um:.3f}%)")
    print(f"    trivial greedy drop:   {100*np.mean(d_grd)/Um:+.3f}%   (retains {100+100*np.mean(d_grd)/Um:.3f}%; "
          f"greedy infeasible left: {grd_inf}/{n})")
    print("(D) VERDICT:")
    gap = 100*(np.mean(d_rep) - np.mean(d_grd))/Um
    print(f"    ILP repair beats trivial greedy by {gap:+.3f}% of objective.")
    print(f"    -> {'Data is forgiving: even drop-only is near-optimal; repair is cheap because the GNN'
          ' merely over-allocates low-value extras.' if abs(gap) < 0.5 else 'The ILP structure matters: it'
          ' meaningfully beats trivial deletion.'}")
    out = {"arch": args.arch, "n": n,
           "tightness": {"mean_ap_load": float(np.mean(ld_milp)), "pct_aps_saturated": float(100*np.mean(sat_ap)),
                         "pct_scn_with_sat": float(100*np.mean(sat_scn)),
                         "mean_users_served": float(np.mean(users_served)), "mean_alpha": float(np.mean(alphas))},
           "violations": {"mean_per_invalid": float(np.mean(nviol)) if nviol else 0.0,
                          "pct_eq1": float(100*np.mean(np.array(nviol) == 1)) if nviol else 0.0,
                          "pct_le2": float(100*np.mean(np.array(nviol) <= 2)) if nviol else 0.0,
                          "mean_rf_overflow": float(np.mean(rf_over)) if rf_over else 0.0,
                          "infeasible_pct": float(100*n_inf/n)},
           "U_milp": Um,
           "obj_gap_pct": {"raw": float(100*np.mean(d_raw)/Um), "ilp": float(100*np.mean(d_rep)/Um),
                           "greedy": float(100*np.mean(d_grd)/Um), "greedy_infeasible_left": grd_inf},
           "obj_abs": {"milp": Um, "raw": float(Um+np.mean(d_raw)), "ilp": float(Um+np.mean(d_rep)),
                       "greedy": float(Um+np.mean(d_grd))}}
    json.dump(out, open(os.path.join(paths.RESULTS, f"difficulty_{args.arch}.json"), "w"), indent=2)
    print(f"saved -> difficulty_{args.arch}.json")


if __name__ == "__main__":
    main()
