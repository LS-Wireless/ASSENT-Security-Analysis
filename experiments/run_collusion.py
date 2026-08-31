"""Multi-AP collusion: do TWO cheating APs do more damage than one? (Paper 2, addresses 'single
attacker' limitation.) Both colluders run the same lie; we measure combined user capture and the
system-wide denial (infeasibility), for the gain and interference attacks.
Run: python experiments/run_collusion.py --attack corr_deflation --n 1000
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
from attacks.falsify import ATTACKS


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--attack", default="corr_deflation", choices=list(ATTACKS))
    ap.add_argument("--colluders", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--kappas", type=float, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--n", type=int, default=1000); a = ap.parse_args()
    res = pickle.load(open(os.path.join(paths.RESULTS, "assent_data_plain.pkl"), "rb"))
    model, thr, test_idx = load_model("gatv2"); test_idx = test_idx[:a.n]
    atk = ATTACKS[a.attack]

    print(f"\n=== Collusion: {len(a.colluders)} APs {a.colluders} run {a.attack} (gatv2, n={len(test_idx)}) ===")
    print(f"{'k':>4} | {'colluders users':>15} {'their RF load':>13} | {'system invalid%':>15}")
    rows = []
    for k in a.kappas:
        users, load = [], []; n_inf = 0
        for gid in test_idx:
            Gc, Sc, Gs = res["G_comm"][gid], res["S_comm"][gid], res["G_sens"][gid]
            for c in a.colluders:                         # each colluder applies the lie to its own slice
                if a.attack == "corr_deflation":
                    Gc, Sc, Gs = atk(Gc, Sc, Gs, c, min(1.0, 1.0 / k) if k > 0 else 0.0)
                else:
                    Gc, Sc, Gs = atk(Gc, Sc, Gs, c, k)
            sol = H.sol_from_probs(H.decode_probs(model, H.make_graph(res, gid, Gc=Gc, Sc=Sc, Gs=Gs)), thr)
            prm = H.params_true(res, gid)
            users.append(int(sum(sol["x"][c].sum() for c in a.colluders)))
            load.append(float(sum(sol["x"][c].sum() + sol["y_tx"][c].sum() for c in a.colluders)))
            n_inf += int(check_feasibility(sol, prm)["infeasible"])
        n = len(test_idx)
        row = {"kappa": k, "colluders_users": float(np.mean(users)), "colluders_load": float(np.mean(load)),
               "system_invalid_pct": 100.0 * n_inf / n}
        rows.append(row)
        print(f"{k:>4.0f} | {row['colluders_users']:>15.2f} {row['colluders_load']:>13.2f} | "
              f"{row['system_invalid_pct']:>14.1f}%")
    json.dump({"attack": a.attack, "colluders": a.colluders, "n": len(test_idx), "rows": rows},
              open(os.path.join(paths.RESULTS, f"collusion_{a.attack}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
