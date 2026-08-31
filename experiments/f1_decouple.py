"""Why per-decision F1 is the wrong metric: it is structurally near-blind to feasibility.

Start from the OPTIMAL plans (F1 = 1.0, 0% illegal). Add k random extra links (off->on flips in x,
the over-allocation a GNN actually makes). Measure mean per-head F1 and the % of plans that break a
rule, as k grows. The point: a tiny accuracy drop (F1 stays ~0.99) coincides with a huge legality
collapse (toward 100% illegal), because legality is a JOINT property of the whole plan, while F1
scores each decision on its own. This is system-independent: it follows from how the metrics are
defined, not from ASSENT.

Run: python experiments/f1_decouple.py --n 1000
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
from sklearn.metrics import f1_score
from experiments import harness as H
from experiments.analyze_difficulty import load_model
from feasibility.checker import check_feasibility

HEADS = ["tau", "x", "s", "ytx", "yrx"]; LK = {"tau": "tau", "x": "x", "s": "s", "ytx": "y_tx", "yrx": "y_rx"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 3, 5, 8]); a = ap.parse_args()
    res = pickle.load(open(os.path.join(paths.RESULTS, "assent_data_plain.pkl"), "rb"))
    _, _, test_idx = load_model("gatv2"); test_idx = test_idx[:a.n]
    rng = np.random.default_rng(0)

    print(f"\n=== F1 is near-blind to legality (perturb the OPTIMAL plan, {len(test_idx)} cases) ===")
    print(f"{'extra links k':>13} | {'mean per-head F1':>16} | {'plans illegal %':>15}")
    rows = []
    for k in a.ks:
        P = {h: [] for h in HEADS}; Y = {h: [] for h in HEADS}; n_inf = 0
        for gid in test_idx:
            lab = res["solution"][gid]
            sol = {h2: np.array(getattr(lab, h2), copy=True) for h2 in ["x", "tau", "s", "y_tx", "y_rx"]}
            A, U = sol["x"].shape
            off = np.argwhere(sol["x"] == 0)                    # off (a,u) pairs we can turn on
            if k > 0 and len(off):
                pick = off[rng.choice(len(off), size=min(k, len(off)), replace=False)]
                sol["x"][pick[:, 0], pick[:, 1]] = 1            # add k extra served links
            for h in HEADS:
                Y[h] += list(np.asarray(getattr(lab, LK[h])).ravel().astype(int))
                P[h] += list(np.asarray(sol[LK[h]]).ravel().astype(int))
            prm = H.params_true(res, gid)
            n_inf += int(check_feasibility(sol, prm)["infeasible"])
        f1 = float(np.mean([f1_score(Y[h], P[h], zero_division=1) for h in HEADS]))
        inf = 100.0 * n_inf / len(test_idx)
        rows.append({"k": k, "mean_f1": f1, "illegal_pct": inf})
        print(f"{k:>13} | {f1:>16.4f} | {inf:>14.1f}%")
    json.dump({"n": len(test_idx), "rows": rows},
              open(os.path.join(paths.RESULTS, "f1_decouple.json"), "w"), indent=2)
    print("\nTakeaway: a near-perfect per-decision F1 can sit on top of an almost-always-illegal plan,\n"
          "because F1 grades each choice alone while legality is a joint property of the whole plan.")


if __name__ == "__main__":
    main()
