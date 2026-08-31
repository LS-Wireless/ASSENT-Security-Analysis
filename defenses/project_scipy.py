"""Repair layer (feasibility projection) on scipy HiGHS -- NO gurobipy.

Why scipy: gurobipy's DLL is blocked by Windows Application Control on this machine, so the
gurobi version in defenses/project.py cannot run. This is a drop-in replacement with the
IDENTICAL formulation, solved by scipy.optimize.milp (HiGHS, a bundled pure-Python-callable
solver -- no external DLL). Same signature, same return type.

Formulation (mirrors upstream solve_problem eq. 7c-7n, one-for-one with feasibility/checker.py):
  maximize   sum over all binary decision vars v of  (2*p_v - 1) * v          (agree with GNN)
  subject to x<=tau, y_tx<=tau, y_rx<=1-tau, RF budget, Rx capacity,
             correlation-exclusion, and per-target tx/rx scheduling caps.
Because the constraints ARE the checker's constraints, any returned plan is feasible by
construction (asserted in the runner: check_feasibility(...)['total_violations'] == 0).
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def _bcast(v, n):
    a = np.asarray(v, float)
    return np.full(n, a) if a.ndim == 0 else a


def project(probs, params, **_):
    """probs: {'px'[A,U], 'ptau'[A], 'ps'[T], 'pytx'[A,T], 'pyrx'[A,T]} in [0,1].
    params: {'N_RF','rho_thresh','S_comm'[A,U,U],'K_tx','K_rx','C_rx'}.
    Returns a feasible solution dict {x,tau,s,y_tx,y_rx} (numpy int8)."""
    px = np.asarray(probs["px"], float); ptau = np.asarray(probs["ptau"], float).reshape(-1)
    ps = np.asarray(probs["ps"], float).reshape(-1)
    pytx = np.asarray(probs["pytx"], float); pyrx = np.asarray(probs["pyrx"], float)
    A, U = px.shape; T = ps.shape[0]
    N_RF = _bcast(params["N_RF"], A); rho = float(params["rho_thresh"])
    S = np.asarray(params["S_comm"], float)
    K_tx = _bcast(params["K_tx"], T); K_rx = _bcast(params["K_rx"], T); C_rx = _bcast(params["C_rx"], A)

    nx, nyt, nyr, nt = A * U, A * T, A * T, A
    N = nx + nyt + nyr + nt + T
    XI = lambda a, u: a * U + u                      # noqa: E731
    TXI = lambda a, t: nx + a * T + t                # noqa: E731
    RXI = lambda a, t: nx + nyt + a * T + t          # noqa: E731
    TI = lambda a: nx + nyt + nyr + a                # noqa: E731
    SI = lambda t: nx + nyt + nyr + nt + t           # noqa: E731

    rows, ub = [], []

    def add(pairs, hi):
        r = np.zeros(N)
        for j, c in pairs:
            r[j] = c
        rows.append(r); ub.append(hi)

    for a in range(A):
        for u in range(U):
            add([(XI(a, u), 1), (TI(a), -1)], 0.0)                                  # x <= tau   (7c)
        for t in range(T):
            add([(TXI(a, t), 1), (TI(a), -1)], 0.0)                                 # y_tx <= tau (7d)
            add([(RXI(a, t), 1), (TI(a), 1)], 1.0)                                  # y_rx <= 1-tau (7e)
        add([(XI(a, u), 1) for u in range(U)] + [(TXI(a, t), 1) for t in range(T)],
            float(N_RF[a]))                                                          # RF budget  (7f)
        add([(RXI(a, t), 1) for t in range(T)], float(C_rx[a]))                      # Rx capacity (7n)
        for u in range(U):
            for up in range(u + 1, U):
                if S[a, u, up] > rho:
                    add([(XI(a, u), 1), (XI(a, up), 1)], 2.0 + rho - float(S[a, u, up]))  # corr (7g)
    for t in range(T):
        add([(TXI(a, t), 1) for a in range(A)] + [(SI(t), -float(K_tx[t]))], 0.0)    # sum y_tx <= Ktx*s (7h)
        add([(RXI(a, t), 1) for a in range(A)] + [(SI(t), -float(K_rx[t]))], 0.0)    # sum y_rx <= Krx*s (7m)
        add([(SI(t), 1)] + [(RXI(a, t), -1) for a in range(A)], 0.0)                 # s <= sum y_rx     (7i)
        add([(SI(t), 1)] + [(TXI(a, t), -1) for a in range(A)], 0.0)                 # s <= sum y_tx     (7j)

    Aub = np.asarray(rows); bub = np.asarray(ub)
    c = np.empty(N)
    c[:nx] = -(2 * px.ravel() - 1)
    c[nx:nx + nyt] = -(2 * pytx.ravel() - 1)
    c[nx + nyt:nx + nyt + nyr] = -(2 * pyrx.ravel() - 1)
    c[nx + nyt + nyr:nx + nyt + nyr + nt] = -(2 * ptau - 1)
    c[nx + nyt + nyr + nt:] = -(2 * ps - 1)

    r = milp(c=c, constraints=LinearConstraint(Aub, -np.inf, bub),
             integrality=np.ones(N), bounds=Bounds(0, 1))
    if r.x is None:
        raise RuntimeError(f"projection failed: status={r.status} {getattr(r, 'message', '')}")
    z = np.round(r.x).astype(np.int8)
    return {"x": z[:nx].reshape(A, U),
            "tau": z[nx + nyt + nyr:nx + nyt + nyr + nt],
            "s": z[nx + nyt + nyr + nt:],
            "y_tx": z[nx:nx + nyt].reshape(A, T),
            "y_rx": z[nx + nyt:nx + nyt + nyr].reshape(A, T)}
