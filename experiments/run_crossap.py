"""Does a CROSS-AP consistency check on the reports recover the interference lie that the
output-side repair alone cannot? (Paper 2 constructive defense.)

For the correlation-deflation attack we compare three server-side pipelines, all using ONLY
reported data (no oracle), evaluated against the TRUE interference:
  (1) no defense:           GNN on faked S, no repair
  (2) output repair only:   GNN on faked S, then re-impose rules with the faked S  (the partial fix)
  (3) cross-AP + repair:    sanitize the faked S vs the other APs' consensus, then GNN + repair
We report physical infeasibility (vs TRUE S), the attacker's kept users, the detector's hit rate
(did it flag the real attacker?), and the false-alarm cost when there is NO attack (k=1).

Run: python experiments/run_crossap.py --n 500
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
from attacks.falsify import correlation_deflation
from defenses.project_scipy import project
from defenses.crossap import sanitize_scomm


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--a_star", type=int, default=0)
    ap.add_argument("--kappas", type=float, nargs="+", default=[1, 2, 4, 8, 16, 32]); a = ap.parse_args()
    res = pickle.load(open(os.path.join(paths.RESULTS, "assent_data_plain.pkl"), "rb"))
    model, thr, test_idx = load_model("gatv2"); test_idx = test_idx[:a.n]

    print(f"\n=== Cross-AP consistency defense vs the interference attack (gatv2, a*={a.a_star}, n={len(test_idx)}) ===")
    print(f"{'k':>4} | {'noDef inf%':>10} | {'repair-only inf%':>16} | {'crossAP+rep inf%':>16} | "
          f"{'detect%':>7} {'falseAlarm%':>11} | {'atk users':>9}")
    rows = []
    for k in a.kappas:
        nd_inf = ro_inf = ca_inf = hit = flag_any = 0; ca_users = []
        for gid in test_idx:
            Gc, Sc, Gs = res["G_comm"][gid], res["S_comm"][gid], res["G_sens"][gid]
            factor = min(1.0, 1.0 / k) if k > 0 else 0.0
            Gc2, Sc2, Gs2 = correlation_deflation(Gc, Sc, Gs, a.a_star, factor)
            p_true = H.params_true(res, gid)
            # (1) no defense
            raw = H.sol_from_probs(H.decode_probs(model, H.make_graph(res, gid, Gc=Gc2, Sc=Sc2, Gs=Gs2)), thr)
            nd_inf += int(check_feasibility(raw, p_true)["infeasible"])
            # (2) output repair only (repair with faked S)
            probs2 = H.decode_probs(model, H.make_graph(res, gid, Gc=Gc2, Sc=Sc2, Gs=Gs2))
            p_fake = dict(p_true); p_fake["S_comm"] = Sc2
            ro_inf += int(check_feasibility(project(probs2, p_fake), p_true)["infeasible"])
            # (3) cross-AP sanitize, then GNN + repair with the cleaned S
            Sc_clean, flagged = sanitize_scomm(Sc2)
            if flagged != -1: flag_any += 1
            if flagged == a.a_star: hit += 1
            probs3 = H.decode_probs(model, H.make_graph(res, gid, Gc=Gc2, Sc=Sc_clean, Gs=Gs2))
            p_clean = dict(p_true); p_clean["S_comm"] = Sc_clean
            d = project(probs3, p_clean)
            ca_inf += int(check_feasibility(d, p_true)["infeasible"])
            ca_users.append(int(d["x"][a.a_star].sum()))
        n = len(test_idx)
        row = {"kappa": k, "nodef_inf": 100.0*nd_inf/n, "repaironly_inf": 100.0*ro_inf/n,
               "crossap_inf": 100.0*ca_inf/n, "detect_pct": 100.0*hit/n,
               "falsealarm_pct": 100.0*flag_any/n, "crossap_atk_users": float(np.mean(ca_users))}
        rows.append(row)
        print(f"{k:>4.0f} | {row['nodef_inf']:>9.1f}% | {row['repaironly_inf']:>15.1f}% | "
              f"{row['crossap_inf']:>15.1f}% | {row['detect_pct']:>6.1f}% {row['falsealarm_pct']:>10.1f}% | "
              f"{row['crossap_atk_users']:>9.2f}")
    json.dump({"a_star": a.a_star, "n": len(test_idx), "rows": rows},
              open(os.path.join(paths.RESULTS, "crossap_defense.json"), "w"), indent=2)
    print("\nfalseAlarm% at k=1 (no attack) = how often the detector wrongly flags an honest AP.")


if __name__ == "__main__":
    main()
