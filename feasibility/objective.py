"""Objective evaluation, faithful to upstream optimization_utils.compute_milp_objective.

We reimplement it (rather than import) for two reasons:
  1. The upstream version computes the Tx reward as `np.sum(mu[a]*tau[a] for a in range(A))`
     -- np.sum on a GENERATOR, which in numpy 2.x does NOT sum correctly. We fix that.
  2. Keeping our own copy in this repo guarantees we never edit upstream.

Everything else matches upstream EXACTLY (internal max_gain normalization, U_comm_ref /
U_sens_ref, interference term over u<up). validate_core.py cross-checks this against the
upstream function (with add_reward=False, where upstream is bug-free) and against the
dataset's stored opt_objVal.
"""
import numpy as np
from types import SimpleNamespace


def _as_ns(sol):
    if isinstance(sol, dict):
        return SimpleNamespace(**sol)
    return sol


def compute_objective(params: dict, solution, add_reward: bool = True) -> dict:
    """params: {'G_comm','S_comm','G_sens','lambda_cu','lambda_tg','alpha','interf_penalty'}
    solution: object/dict with x[A,U], tau[A], y_tx[A,T], y_rx[A,T], s[T]."""
    p = SimpleNamespace(**params)
    sol = _as_ns(solution)

    Gc = np.asarray(p.G_comm, dtype=np.float64)            # [A,U]
    Sc = np.asarray(p.S_comm, dtype=np.float64)            # [A,U,U]
    Gs = np.asarray(p.G_sens, dtype=np.float64)            # [A,A,T]
    A, U = Gc.shape
    T = Gs.shape[-1]

    x   = np.asarray(sol.x,    dtype=np.float64)
    tau = np.asarray(sol.tau,  dtype=np.float64).reshape(-1)
    ytx = np.asarray(sol.y_tx, dtype=np.float64)
    yrx = np.asarray(sol.y_rx, dtype=np.float64)
    s   = np.asarray(sol.s,    dtype=np.float64).reshape(-1)

    lcu = np.full(U, p.lambda_cu) if np.isscalar(p.lambda_cu) else np.asarray(p.lambda_cu, float)
    ltg = np.full(T, p.lambda_tg) if np.isscalar(p.lambda_tg) else np.asarray(p.lambda_tg, float)
    alpha = float(p.alpha)
    beta = float(p.interf_penalty)

    # ---- internal normalization (identical to upstream compute_milp_objective) ----
    max_gain = max(np.max(Gc), np.max(Gs))
    Gcn = Gc / max_gain
    Gsn = Gs / max_gain
    mu = np.sum(Gcn * lcu.reshape(1, -1), axis=1)          # [A]

    iu, iup = np.triu_indices(U, k=1)
    U_comm_ref = float(np.sum(Gcn) - beta * np.sum(Sc[:, iu, iup] * Gcn[:, iu]))
    U_sens_ref = float(np.sum(Gsn))

    # communication utility
    comm_p1 = float(np.sum(lcu[None, :] * Gcn * x))
    l_pair = 0.5 * (lcu[iu] + lcu[iup])                    # [P]
    V = x[:, iu] * x[:, iup]                               # [A,P]
    comm_p2 = float(np.sum(l_pair[None, :] * Sc[:, iu, iup] * Gcn[:, iu] * V))
    comm_util = (comm_p1 - beta * comm_p2) / U_comm_ref

    # sensing utility: sum_{a_t,a_r,t} ltg[t]*Gsn[a_t,a_r,t]*ytx[a_t,t]*yrx[a_r,t]*s[t]
    # Gsn indexed [a_t, a_r, t]; ytx[a_t,t], yrx[a_r,t].
    zt = (ytx[:, None, :] * yrx[None, :, :]) * s[None, None, :]   # [A_t, A_r, T]
    sens_util = float(np.sum(ltg[None, None, :] * Gsn * zt)) / U_sens_ref

    # Tx-mode reward (FIXED: vectorized sum, not a generator)
    tx_reward = float(np.sum(mu * tau)) / U_comm_ref if add_reward else 0.0

    obj_val = alpha * comm_util + (1 - alpha) * sens_util + tx_reward
    obj_val_noreward = alpha * comm_util + (1 - alpha) * sens_util
    return {"comm_util": comm_util, "sens_util": sens_util, "tx_reward": tx_reward,
            "obj_val": obj_val, "obj_val_noreward": obj_val_noreward}
