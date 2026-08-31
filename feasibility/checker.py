"""Hard-constraint feasibility checker for ASSENT solutions.

Implements EXACTLY the MILP constraints from upstream
src/utils/optimization_utils.py::solve_problem (constraints 1-7), evaluated on a
decoded binary solution. The aux variables (z, w, v) are pure linearizations of
products and are NOT decision constraints, so they are excluded.

Correctness oracle: any MILP-optimal solution is feasible by construction, so
check_feasibility(milp_solution, milp_params) MUST report 0 violations. See
experiments/validate_core.py.

Variable conventions (match optimization_utils.Solution):
  tau  : [A]      AP mode (1=Tx, 0=Rx)
  x    : [A, U]   AP-user association
  y_tx : [A, T]   Tx-AP -> target
  y_rx : [A, T]   Rx-AP -> target   (extracted as [A,T] in solve_problem)
  s    : [T]      target scheduled
"""
import numpy as np


def _get(sol, name):
    if isinstance(sol, dict):
        return sol[name]
    return getattr(sol, name)


def _bcast(val, n, dtype=float):
    arr = np.asarray(val, dtype=dtype)
    if arr.ndim == 0:
        return np.full(n, arr, dtype=dtype)
    assert arr.shape[0] == n, f"expected length {n}, got {arr.shape}"
    return arr


def check_feasibility(sol, params, tol=1e-6, return_detail=False):
    """Return per-constraint violation counts for one decoded solution.

    params keys: N_RF, rho_thresh, S_comm [A,U,U], K_tx, K_rx, C_rx
    (N_RF/C_rx may be scalar or [A]; K_tx/K_rx scalar or [T]).
    """
    x   = np.asarray(_get(sol, "x"),    dtype=float)
    tau = np.asarray(_get(sol, "tau"),  dtype=float).reshape(-1)
    ytx = np.asarray(_get(sol, "y_tx"), dtype=float)
    yrx = np.asarray(_get(sol, "y_rx"), dtype=float)
    s   = np.asarray(_get(sol, "s"),    dtype=float).reshape(-1)

    A, U = x.shape
    T = s.shape[0]
    assert tau.shape[0] == A, (tau.shape, A)
    assert ytx.shape == (A, T), (ytx.shape, (A, T))
    assert yrx.shape == (A, T), (yrx.shape, (A, T))

    N_RF = _bcast(params["N_RF"], A)
    rho  = float(params["rho_thresh"])
    S    = np.asarray(params["S_comm"], dtype=float)            # [A,U,U]
    K_tx = _bcast(params["K_tx"], T)
    K_rx = _bcast(params["K_rx"], T)
    C_rx = _bcast(params["C_rx"], A)

    v = {}
    # --- Constraint 1: AP Tx/Rx mode coupling (eq. 7c-7e) ---
    v["mode_x_le_tau"]   = int(np.sum((x   > 0.5) & (tau[:, None] < 0.5)))   # x[a,u] <= tau[a]
    v["mode_ytx_le_tau"] = int(np.sum((ytx > 0.5) & (tau[:, None] < 0.5)))   # y_tx <= tau
    v["mode_yrx_le_1mtau"] = int(np.sum((yrx > 0.5) & (tau[:, None] > 0.5))) # y_rx <= 1-tau

    # --- Constraint 2: RF-chain budget (eq. 7f) ---
    rf_load = x.sum(axis=1) + ytx.sum(axis=1)                                # [A]
    v["rf_cap"] = int(np.sum(rf_load > N_RF + tol))

    # --- Constraint 3: target scheduling (eq. 7h-7m) ---
    txsum = ytx.sum(axis=0)                                                  # [T]
    rxsum = yrx.sum(axis=0)                                                  # [T]
    v["tx_le_Ktx_s"] = int(np.sum(txsum > K_tx * s + tol))                   # sum_a y_tx <= K_tx*s
    v["rx_ge_s"]     = int(np.sum(rxsum < s - tol))                          # sum_a y_rx >= s
    v["tx_ge_s"]     = int(np.sum(txsum < s - tol))                          # sum_a y_tx >= s
    v["rx_le_Krx_s"] = int(np.sum(rxsum > K_rx * s + tol))                   # sum_a y_rx <= K_rx*s

    # --- Constraint 4: user-user correlation exclusion (eq. 7g), u < up ---
    iu, iup = np.triu_indices(U, k=1)
    pair_sum = x[:, iu] + x[:, iup]                                          # [A,P]
    rhs = 2.0 + rho - S[:, iu, iup]                                          # [A,P]
    v["corr_excl"] = int(np.sum(pair_sum > rhs + tol))

    # --- Constraint 6: Rx capacity per AP (eq. 7n) ---
    v["rx_capacity"] = int(np.sum(yrx.sum(axis=1) > C_rx + tol))

    total = int(sum(v.values()))
    out = {**v, "total_violations": total, "infeasible": bool(total > 0)}
    if return_detail:
        out["rf_overflow_amount"] = float(np.clip(rf_load - N_RF, 0, None).sum())
    return out


def audit_dataset(solutions, params_per_sample):
    """Aggregate feasibility over a list of (solution, params) -> summary dict.

    Returns per-constraint violation RATE (fraction of samples with >=1 violation
    of that constraint) and overall infeasibility %.
    """
    keys = ["mode_x_le_tau", "mode_ytx_le_tau", "mode_yrx_le_1mtau", "rf_cap",
            "tx_le_Ktx_s", "rx_ge_s", "tx_ge_s", "rx_le_Krx_s", "corr_excl", "rx_capacity"]
    n = len(solutions)
    counts = {k: 0 for k in keys}
    n_infeasible = 0
    for sol, prm in zip(solutions, params_per_sample):
        r = check_feasibility(sol, prm)
        if r["infeasible"]:
            n_infeasible += 1
        for k in keys:
            if r[k] > 0:
                counts[k] += 1
    return {
        "n_samples": n,
        "infeasible_pct": 100.0 * n_infeasible / max(1, n),
        "violation_rate": {k: 100.0 * counts[k] / max(1, n) for k in keys},
    }
