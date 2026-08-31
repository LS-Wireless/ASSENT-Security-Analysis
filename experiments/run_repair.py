"""Evaluate the repair layer on the test set, per architecture.

For every test scenario we compute, for the RAW GNN plan and the REPAIRED (projected) plan:
  - rule violations (% of plans invalid, and per-rule rates)
  - per-decision F1 vs the MILP optimal (tau, x, s, ytx, yrx)  [== paper Table II metric]
  - mean objective, alongside the MILP optimum
and we report the 'repair tax': how much objective the repaired GNN gives up versus the
(infeasible) raw GNN and versus the MILP optimum.

Runs gurobipy-free (scipy HiGHS projection + stubbed unpickling + pure-numpy objective/checker).
Run: python experiments/run_repair.py --arch gatv2 --tag repro --n 1000
"""
import os, sys, types, json, pickle, argparse, time
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
from sklearn.metrics import f1_score, confusion_matrix

from experiments import harness as H
from feasibility.checker import check_feasibility
from feasibility.objective import compute_objective
from defenses.project_scipy import project
from src.learning_v4.assent_model_v4 import ASSENTGNN, ASSENTGNN_GAT, ASSENTGNN_Transformer

ARCH_RUN = {"nnconv": "run_03", "gatv2": "run_04", "transformer": "run_02"}
HEADS = ["tau", "x", "s", "ytx", "yrx"]; LK = {"tau": "tau", "x": "x", "s": "s", "ytx": "y_tx", "yrx": "y_rx"}
PAPER_F1 = {"nnconv": [0.985, 0.907, 0.975, 0.841, 0.919],
            "transformer": [0.979, 0.928, 0.978, 0.868, 0.911],
            "gatv2": [0.983, 0.908, 0.981, 0.848, 0.918]}


def _ckroot(tag): return "checkpoints" if tag in ("paper", "none", "") else f"checkpoints_{tag}"


def load_model(arch, tag):
    c = json.load(open(os.path.join(paths.UPSTREAM, "src", "learning_v4", "checkpoints",
                                    ARCH_RUN[arch], "metadata.json")))["config"]
    if arch == "gatv2":
        m = ASSENTGNN_GAT(hidden=c["nhidden"], layers=c["nlayers"], drop=c["drop"], heads=c["gatv2_heads"])
    elif arch == "transformer":
        m = ASSENTGNN_Transformer(hidden=c["nhidden"], layers=c["nlayers"], drop=c["drop"],
                                  heads=c["transformer_heads"], beta=c["transformer_beta"])
    else:
        m = ASSENTGNN(hidden=c["nhidden"], layers=c["nlayers"], drop=c["drop"])
    ck = os.path.join(paths.RESULTS, _ckroot(tag), arch)
    m.load_state_dict(torch.load(os.path.join(ck, "best_model.pt"), map_location="cpu")); m.eval()
    s = json.load(open(os.path.join(ck, "summary.json")))
    thr = {k: s[f"thr_{k}"] for k in HEADS}
    return m, thr, s["test_idx"]


def f1_block(P, Y):
    return {h: f1_score(Y[h], P[h], zero_division=0) for h in HEADS}


def viol_rates(counts, n):
    return {k: 100.0 * v / n for k, v in sorted(counts.items(), key=lambda x: -x[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gatv2"); ap.add_argument("--tag", default="repro")
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    res = pickle.load(open(os.path.join(paths.RESULTS, "assent_data_plain.pkl"), "rb"))
    model, thr, test_idx = load_model(args.arch, args.tag)
    test_idx = test_idx[:args.n]

    Praw = {h: [] for h in HEADS}; Prep = {h: [] for h in HEADS}; Y = {h: [] for h in HEADS}
    raw_inf = rep_inf = 0; raw_vc = {}; rep_vc = {}
    U_raw, U_rep, U_milp = [], [], []
    n_over = 0
    t0 = time.time()
    for i, gid in enumerate(test_idx):
        probs = H.decode_probs(model, H.make_graph(res, gid))
        raw = H.sol_from_probs(probs, thr)
        prm = H.params_true(res, gid)
        rep = project(probs, prm)
        lab = res["solution"][gid]
        for h in HEADS:
            Y[h] += list(np.asarray(getattr(lab, LK[h])).ravel().astype(int))
            Praw[h] += list(np.asarray(raw[LK[h]]).ravel()); Prep[h] += list(np.asarray(rep[LK[h]]).ravel())
        rr = check_feasibility(raw, prm); rp = check_feasibility(rep, prm)
        raw_inf += int(rr["infeasible"]); rep_inf += int(rp["infeasible"])
        for k, v in rr.items():
            if isinstance(v, int) and k != "total_violations" and v > 0: raw_vc[k] = raw_vc.get(k, 0) + 1
        for k, v in rp.items():
            if isinstance(v, int) and k != "total_violations" and v > 0: rep_vc[k] = rep_vc.get(k, 0) + 1
        op = H.obj_params_true(res, gid)
        ur = compute_objective(op, raw)["obj_val"]; up = compute_objective(op, rep)["obj_val"]
        um = compute_objective(op, lab)["obj_val"]
        U_raw.append(ur); U_rep.append(up); U_milp.append(um)
        if up > um + 1e-6: n_over += 1
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(test_idx)} ({(time.time()-t0)/(i+1)*1000:.0f} ms/scenario)")

    n = len(test_idx)
    f1raw = f1_block(Praw, Y); f1rep = f1_block(Prep, Y)
    Um, Ur, Up = float(np.mean(U_milp)), float(np.mean(U_raw)), float(np.mean(U_rep))
    out = {"arch": args.arch, "tag": args.tag, "n_test": n,
           "raw_infeasible_pct": 100.0 * raw_inf / n, "rep_infeasible_pct": 100.0 * rep_inf / n,
           "raw_viol_rates": viol_rates(raw_vc, n), "rep_viol_rates": viol_rates(rep_vc, n),
           "f1_raw": {h: f1raw[h] for h in HEADS}, "f1_rep": {h: f1rep[h] for h in HEADS},
           "paper_f1": dict(zip(HEADS, PAPER_F1[args.arch])),
           "U_milp": Um, "U_raw": Ur, "U_rep": Up,
           "repaired_gap_vs_milp_pct": 100.0 * (Um - Up) / abs(Um),
           "raw_exceeds_milp_count": n_over}
    fn = os.path.join(paths.RESULTS, f"repair_{args.arch}_{args.tag}.json")
    json.dump(out, open(fn, "w"), indent=2)

    print(f"\n==== REPAIR LAYER on {args.arch} ({args.tag}), {n} test scenarios ====")
    print(f"INVALID PLANS:   raw GNN {out['raw_infeasible_pct']:.1f}%   ->   repaired {out['rep_infeasible_pct']:.1f}%")
    print(f"{'head':<5} {'paper':>6} {'rawF1':>6} {'repF1':>6}  (repaired vs optimal)")
    for i, h in enumerate(HEADS):
        print(f"{h:<5} {PAPER_F1[args.arch][i]:>6.3f} {f1raw[h]:>6.3f} {f1rep[h]:>6.3f}")
    print(f"OBJECTIVE:  MILP={Um:.3f}  rawGNN={Ur:.3f} (infeasible, inflated)  repaired={Up:.3f}")
    print(f"REPAIR TAX:  repaired is {out['repaired_gap_vs_milp_pct']:.2f}% below the MILP optimum; "
          f"raw exceeded MILP on {n_over}/{n} (infeasible inflation)")
    print(f"saved -> {fn}")


if __name__ == "__main__":
    main()
