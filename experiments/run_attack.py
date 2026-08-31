"""Attack phase: one compromised AP a* reports FALSIFIED statistics to the central server to
grab resources. We sweep attack strength and measure, on the verified models, how much the
attacker captures -- with and without the repair layer as a defense.

Threat model (harness semantics): decode on the FALSIFIED inputs the server sees; evaluate
impact (feasibility, who-got-served) on the TRUE channel (the physics is unchanged by the lie).

Per strength k we report:
  atk_users   mean # users assigned to the attacker AP        (the resource grab)
  atk_share   attacker's users / all assignments
  atk_rfload  attacker's radio-chain load (x + y_tx)           (capped at N_RF only if feasible)
  coverage    mean # distinct users served by ANY AP          (honest-utility proxy)
  infeas%     % of plans breaking >=1 hard rule
  with --defense: the same AFTER the repair layer (which bounds any AP's load by N_RF).

Run: python experiments/run_attack.py --arch gatv2 --tag repro --attack gain_targeted --n 300 --defense
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
from attacks.falsify import ATTACKS
from defenses.project_scipy import project
from src.learning_v4.assent_model_v4 import ASSENTGNN, ASSENTGNN_GAT, ASSENTGNN_Transformer

ARCH_RUN = {"nnconv": "run_03", "gatv2": "run_04", "transformer": "run_02"}
HEADS = ["tau", "x", "s", "ytx", "yrx"]


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


def metrics(sol, a_star):
    x, ytx, yrx, tau = sol["x"], sol["y_tx"], sol["y_rx"], sol["tau"]
    total = int(x.sum())
    return {"atk_users": float(x[a_star].sum()),
            "atk_share": float(x[a_star].sum() / total) if total else 0.0,
            "atk_rfload": float(x[a_star].sum() + ytx[a_star].sum()),
            "atk_tx": float(tau[a_star]),                       # attacker forced into transmit mode?
            "atk_sens_tx": float(ytx[a_star].sum()),            # sensing roles grabbed (transmit to targets)
            "atk_sens_rx": float(yrx[a_star].sum()),            # sensing roles grabbed (receive from targets)
            "coverage": float((x.sum(axis=0) > 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gatv2"); ap.add_argument("--tag", default="repro")
    ap.add_argument("--attack", default="gain_targeted", choices=list(ATTACKS))
    ap.add_argument("--a_star", type=int, default=0)
    ap.add_argument("--kappas", type=float, nargs="+", default=[1, 1.5, 2, 4, 8, 16, 32])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--defense", action="store_true")
    args = ap.parse_args()
    res = pickle.load(open(os.path.join(paths.RESULTS, "assent_data_plain.pkl"), "rb"))
    model, thr, test_idx = load_model(args.arch, args.tag)
    test_idx = test_idx[:args.n]
    atk = ATTACKS[args.attack]
    mkeys = ["atk_users", "atk_share", "atk_rfload", "atk_tx", "atk_sens_tx", "atk_sens_rx", "coverage"]

    rows = []
    for k in args.kappas:
        agg = {m: [] for m in mkeys}; dagg = {m: [] for m in mkeys}
        n_inf = n_inf_d = cap_viol_d = 0
        for gid in test_idx:
            Gc, Sc, Gs = res["G_comm"][gid], res["S_comm"][gid], res["G_sens"][gid]
            if args.attack == "corr_deflation":
                Gc2, Sc2, Gs2 = atk(Gc, Sc, Gs, args.a_star, min(1.0, 1.0 / k) if k > 0 else 0.0)
            else:
                Gc2, Sc2, Gs2 = atk(Gc, Sc, Gs, args.a_star, k)
            probs = H.decode_probs(model, H.make_graph(res, gid, Gc=Gc2, Sc=Sc2, Gs=Gs2))
            sol = H.sol_from_probs(probs, thr)
            prm = H.params_true(res, gid)
            for m, v in metrics(sol, args.a_star).items(): agg[m].append(v)
            n_inf += int(check_feasibility(sol, prm)["infeasible"])
            if args.defense:
                d = project(probs, prm)
                for m, v in metrics(d, args.a_star).items(): dagg[m].append(v)
                n_inf_d += int(check_feasibility(d, prm)["infeasible"])
                if (d["x"][args.a_star].sum() + d["y_tx"][args.a_star].sum()) > np.asarray(res["N_RF"]).max():
                    cap_viol_d += 1
        n = len(test_idx)
        row = {"kappa": k, "infeas_pct": 100.0 * n_inf / n}
        for m in mkeys: row[m] = float(np.mean(agg[m]))
        if args.defense:
            row["def_infeas_pct"] = 100.0 * n_inf_d / n; row["def_cap_viol"] = cap_viol_d
            for m in mkeys: row[f"def_{m}"] = float(np.mean(dagg[m]))
        rows.append(row)
        msg = (f"k={k:6.1f} | atk_users={row['atk_users']:.2f} tx={row['atk_tx']:.2f} "
               f"sensTx={row['atk_sens_tx']:.2f} sensRx={row['atk_sens_rx']:.2f} rfload={row['atk_rfload']:.2f} "
               f"infeas={row['infeas_pct']:.1f}%")
        if args.defense:
            msg += (f"  || DEF atk_users={row['def_atk_users']:.2f} rfload={row['def_atk_rfload']:.2f} "
                    f"infeas={row['def_infeas_pct']:.1f}% cap_viol={row['def_cap_viol']}")
        print(msg)

    out = {"arch": args.arch, "tag": args.tag, "attack": args.attack, "a_star": args.a_star,
           "n_test": len(test_idx), "defense": bool(args.defense), "rows": rows}
    fn = os.path.join(paths.RESULTS, f"attack_{args.arch}_{args.tag}_{args.attack}_a{args.a_star}{'_def' if args.defense else ''}.json")
    json.dump(out, open(fn, "w"), indent=2)
    print(f"saved -> {fn}")


if __name__ == "__main__":
    main()
