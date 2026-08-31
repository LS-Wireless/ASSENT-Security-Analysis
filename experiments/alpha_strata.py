"""Does the 'repair is almost free' result hold across the comm/sensing trade-off (alpha)?
Bins the per-scenario repair cost (gatv2 test set) by alpha, with a bootstrap 95% CI on the mean.
Run: python experiments/alpha_strata.py --n 1000
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
from experiments import harness as H
from experiments.analyze_difficulty import load_model
from feasibility.checker import check_feasibility
from feasibility.objective import compute_objective
from defenses.project_scipy import project


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1000); a = ap.parse_args()
    res = pickle.load(open(os.path.join(paths.RESULTS, "assent_data_plain.pkl"), "rb"))
    model, thr, test_idx = load_model("gatv2"); test_idx = test_idx[:a.n]
    rng = np.random.default_rng(0)

    alphas, gaps, infs = [], [], []
    for gid in test_idx:
        probs = H.decode_probs(model, H.make_graph(res, gid))
        raw = H.sol_from_probs(probs, thr); prm = H.params_true(res, gid); op = H.obj_params_true(res, gid)
        rep = project(probs, prm)
        um = compute_objective(op, res["solution"][gid])["obj_val"]
        up = compute_objective(op, rep)["obj_val"]
        alphas.append(float(res["alpha"][gid])); gaps.append(100.0 * (um - up) / abs(um))
        infs.append(int(check_feasibility(raw, prm)["infeasible"]))
    alphas, gaps, infs = np.array(alphas), np.array(gaps), np.array(infs)

    def ci(x):
        if len(x) < 2: return (float(np.mean(x)) if len(x) else 0.0, 0.0, 0.0)
        bs = [np.mean(rng.choice(x, len(x), replace=True)) for _ in range(2000)]
        return float(np.mean(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    print(f"\n=== Repair cost vs alpha (gatv2, {len(test_idx)} scenarios) ===")
    print(f"{'alpha bin':<16} {'n':>5} {'invalid%':>9} {'repair cost % (mean [95% CI])':>34}")
    bins = [(0.0, 1/3, 'sensing-heavy'), (1/3, 2/3, 'balanced'), (2/3, 1.01, 'comm-heavy')]
    rows = []
    for lo, hi, name in bins:
        m = (alphas >= lo) & (alphas < hi)
        mean, l, u = ci(gaps[m])
        print(f"{name+' ['+str(round(lo,2))+','+str(round(hi,2))+')':<16} {m.sum():>5} "
              f"{100*infs[m].mean():>8.1f}% {mean:>11.3f}  [{l:.3f}, {u:.3f}]")
        rows.append({"bin": name, "lo": lo, "hi": hi, "n": int(m.sum()),
                     "invalid_pct": float(100*infs[m].mean()), "cost_mean": mean, "ci": [l, u]})
    mean, l, u = ci(gaps)
    print(f"{'ALL':<16} {len(gaps):>5} {100*infs.mean():>8.1f}% {mean:>11.3f}  [{l:.3f}, {u:.3f}]")
    json.dump({"n": len(test_idx), "overall_cost": [mean, l, u], "bins": rows},
              open(os.path.join(paths.RESULTS, "alpha_strata.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
