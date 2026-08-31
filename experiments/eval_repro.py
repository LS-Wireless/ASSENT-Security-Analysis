"""Reproduce ASSENT's paper numbers from our re-trained models and compare:
  Table II  -> per-head test-set F1 (tau, x, s, ytx, yrx)
  Fig 2     -> mean objective: MILP (U_gt) and each GNN (U_pd)
Runs gurobipy-free (stub for unpickling + our pure-numpy objective/checker). Works in either venv.

Run: python experiments/eval_repro.py --tag repro
"""
import os, sys, types, json, pickle, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
# stub gurobipy-laden module so any incidental import is safe
_opt = types.ModuleType("src.utils.optimization_utils")
class Solution: pass            # noqa: E701
class ProblemParams: pass       # noqa: E701
_opt.Solution = Solution; _opt.ProblemParams = ProblemParams
sys.modules["src.utils.optimization_utils"] = _opt

import paths  # noqa: F401
import numpy as np
import torch
from sklearn.metrics import f1_score

from experiments import harness as H
from feasibility.objective import compute_objective
from src.learning_v4.assent_model_v4 import ASSENTGNN, ASSENTGNN_GAT, ASSENTGNN_Transformer

ARCH_RUN = {"nnconv": "run_03", "gatv2": "run_04", "transformer": "run_02"}
# paper Table II F1 (order: tau, x, s, ytx, yrx) and Fig 2 objective
PAPER_F1 = {"nnconv": [0.985, 0.907, 0.975, 0.841, 0.919],
            "transformer": [0.979, 0.928, 0.978, 0.868, 0.911],
            "gatv2": [0.983, 0.908, 0.981, 0.848, 0.918]}
PAPER_OBJ = {"MILP": 4.286, "nnconv": 4.168, "transformer": 4.201, "gatv2": 4.311}


def _ckroot(tag):
    return "checkpoints" if tag in ("paper", "none", "") else f"checkpoints_{tag}"


def load_model(arch, tag, res_dir):
    meta = json.load(open(os.path.join(paths.UPSTREAM, "src", "learning_v4", "checkpoints",
                                       ARCH_RUN[arch], "metadata.json")))
    c = meta["config"]
    if arch == "gatv2":
        m = ASSENTGNN_GAT(hidden=c["nhidden"], layers=c["nlayers"], drop=c["drop"], heads=c["gatv2_heads"])
    elif arch == "transformer":
        m = ASSENTGNN_Transformer(hidden=c["nhidden"], layers=c["nlayers"], drop=c["drop"],
                                  heads=c["transformer_heads"], beta=c["transformer_beta"])
    else:
        m = ASSENTGNN(hidden=c["nhidden"], layers=c["nlayers"], drop=c["drop"])
    ck = os.path.join(res_dir, _ckroot(tag), arch)
    m.load_state_dict(torch.load(os.path.join(ck, "best_model.pt"), map_location="cpu"))
    m.eval()
    summ = json.load(open(os.path.join(ck, "summary.json")))
    thr = {"x": summ["thr_x"], "tau": summ["thr_tau"], "s": summ["thr_s"],
           "ytx": summ["thr_ytx"], "yrx": summ["thr_yrx"]}
    return m, thr, summ["test_idx"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="repro")
    ap.add_argument("--data_pkl", default=os.path.join(paths.RESULTS, "assent_data_plain.pkl"))
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()
    res = pickle.load(open(args.data_pkl, "rb"))

    print(f"\n{'arch':12s} | {'head':>4s}  paper  ours   (F1, test set)")
    rows = {}
    for arch in ["nnconv", "transformer", "gatv2"]:
        ckdir = os.path.join(paths.RESULTS, _ckroot(args.tag), arch, "best_model.pt")
        if not os.path.exists(ckdir):
            print(f"{arch:12s} | (not trained yet)"); continue
        model, thr, test_idx = load_model(arch, args.tag, paths.RESULTS)
        if args.n: test_idx = test_idx[:args.n]
        heads = ["tau", "x", "s", "ytx", "yrx"]
        P = {h: [] for h in heads}; Y = {h: [] for h in heads}
        Ugt, Upd = [], []
        for gid in test_idx:
            g = H.make_graph(res, gid)
            sol = H.decode(model, g, thr)
            lab = res["solution"][gid]
            P["tau"] += list(sol["tau"]);              Y["tau"] += list(np.asarray(lab.tau).astype(int))
            P["x"]   += list(sol["x"].ravel());        Y["x"]   += list(np.asarray(lab.x).ravel().astype(int))
            P["s"]   += list(sol["s"]);                Y["s"]   += list(np.asarray(lab.s).astype(int))
            P["ytx"] += list(sol["y_tx"].ravel());     Y["ytx"] += list(np.asarray(lab.y_tx).ravel().astype(int))
            P["yrx"] += list(sol["y_rx"].ravel());     Y["yrx"] += list(np.asarray(lab.y_rx).ravel().astype(int))
            op = H.obj_params_true(res, gid)
            Ugt.append(compute_objective(op, lab)["obj_val"])
            Upd.append(compute_objective(op, sol)["obj_val"])
        f1 = {h: f1_score(Y[h], P[h], zero_division=0) for h in heads}
        rows[arch] = {"f1": [f1[h] for h in heads], "U_gt": float(np.mean(Ugt)), "U_pd": float(np.mean(Upd))}
        for i, h in enumerate(heads):
            print(f"{arch:12s} | {h:>4s}  {PAPER_F1[arch][i]:.3f}  {f1[h]:.3f}")
        print(f"{'':12s} |  obj  MILP {PAPER_OBJ['MILP']:.3f}/ours {np.mean(Ugt):.3f} ; "
              f"GNN paper {PAPER_OBJ[arch]:.3f}/ours {np.mean(Upd):.3f}\n")
    json.dump({"paper_f1": PAPER_F1, "paper_obj": PAPER_OBJ, "ours": rows},
              open(os.path.join(paths.RESULTS, "repro_compare.json"), "w"), indent=2)
    print("saved results/repro_compare.json")


if __name__ == "__main__":
    main()
