"""Run the AUTHORS' own evaluation code as-is on our retrained models, to (1) cross-check our
harness against their pipeline and (2) test their evaluation details.

Their evaluation (assent_evaluation_v4.py + assent_test_utils_v4.py):
  - builds the test set with their GraphDataset + PyG DataLoader,
  - run_inference() -> summarize_inference() -> print_metrics_table()  => Table II (F1 etc.),
  - milp_vs_assent() => per-scenario MILP utility U_gt vs ASSENT utility U_pd  => Fig 2.

Two faithful-execution wrinkles we handle:
  - their unpickled Solution labels need gurobipy -> we stub it (data is precomputed; no solving).
  - their milp_vs_assent calls opt.compute_milp_objective, which crashes on numpy>=2 (the upstream
    np.sum(generator) bug). We bind opt.compute_milp_objective to our fixed objective so THEIR
    code path runs unchanged otherwise.

We also test the threshold detail: their script reuses summ_nnconv['pareto'][0] thresholds for ALL
three models. We compare each model's OWN tuned thresholds vs NNConv's thresholds.

Run: python experiments/run_authors_eval.py --arch gatv2 --n 1000
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
from feasibility.objective import compute_objective
# bind the authors' objective entry point to our fixed implementation (numpy>=2 safe)
_opt.compute_milp_objective = lambda params, sol, *a, **k: compute_objective(params, sol)

from experiments import harness as H  # noqa: F401  (sets cwd to upstream, imports build_graph)
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from src.learning_v4.assent_data_v4 import GraphDataset
from src.learning_v4.assent_test_utils_v4 import run_inference, summarize_inference, print_metrics_table, milp_vs_assent
from src.learning_v4.assent_model_v4 import ASSENTGNN, ASSENTGNN_GAT, ASSENTGNN_Transformer

ARCH_RUN = {"nnconv": "run_03", "gatv2": "run_04", "transformer": "run_02"}
HEADS = ["x", "tau", "s", "ytx", "yrx"]
PAPER_F1 = {"nnconv": dict(zip(["tau", "x", "s", "ytx", "yrx"], [0.985, 0.907, 0.975, 0.841, 0.919])),
            "transformer": dict(zip(["tau", "x", "s", "ytx", "yrx"], [0.979, 0.928, 0.978, 0.868, 0.911])),
            "gatv2": dict(zip(["tau", "x", "s", "ytx", "yrx"], [0.983, 0.908, 0.981, 0.848, 0.918]))}


def load_model_thr(arch):
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
    thr = {f"thr_{k}": s[f"thr_{k}"] for k in HEADS}
    return m, thr, s["test_idx"]


def f1row(model, loader, thr):
    _, _, _, _ = run_inference(model, loader, "cpu", thr_dict=thr)  # warm
    pc, pb, yc, _ = run_inference(model, loader, "cpu", thr_dict=thr)
    rows = summarize_inference(pc, pb, yc, keys=HEADS)
    return {r["var"].lower(): r["F1"] for r in rows}, (pc, pb, yc)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arch", default="gatv2")
    ap.add_argument("--n", type=int, default=1000); args = ap.parse_args()
    res = pickle.load(open(os.path.join(paths.RESULTS, "assent_data_plain.pkl"), "rb"))
    model, own_thr, test_idx = load_model_thr(args.arch)
    nn_thr = load_model_thr("nnconv")[1]              # NNConv thresholds (what their script reuses)
    te = np.asarray(test_idx[:args.n])

    ds = GraphDataset(res["G_comm"], res["S_comm"], res["G_sens"], res["alpha"],
                      res["lambda_cu"], res["lambda_tg"], res["solution"], N_RF=res["N_RF"],
                      device="cpu", K_tx_milp=res["K_tx_milp"], K_rx_milp=res["K_rx_milp"])
    loader = DataLoader(Subset(ds, te.tolist()), batch_size=256, shuffle=False)

    print(f"\n=== AUTHORS' pipeline on our retrained {args.arch} ({len(te)} test scenarios) ===")
    own, _ = f1row(model, loader, own_thr)
    nnt, _ = f1row(model, loader, nn_thr)
    print(f"{'head':<5} {'paper':>6} {'own-thr':>8} {'NNConv-thr':>11}  (their script reuses NNConv-thr)")
    for h in ["tau", "x", "s", "ytx", "yrx"]:
        print(f"{h:<5} {PAPER_F1[args.arch][h]:>6.3f} {own[h]:>8.3f} {nnt[h]:>11.3f}")

    # their objective comparison (Fig 2), via their milp_vs_assent
    recs = milp_vs_assent(model, loader, te, "cpu", res, thr_dict=own_thr)
    Ugt = np.mean([r["U_gt"] for r in recs]); Upd = np.mean([r["U_pd"] for r in recs])
    print(f"\nTheir milp_vs_assent (own-thr):  MILP U_gt={Ugt:.3f}   ASSENT U_pd={Upd:.3f}")
    # cross-check vs our eval_repro JSON if present
    rp = os.path.join(paths.RESULTS, f"repair_{args.arch}_repro.json")
    if os.path.exists(rp):
        ours = json.load(open(rp))
        print(f"Our run_repair (same model):     MILP={ours['U_milp']:.3f}   rawGNN={ours['U_raw']:.3f}  "
              f"(cross-check: U_gt vs our MILP, U_pd vs our rawGNN)")


if __name__ == "__main__":
    main()
