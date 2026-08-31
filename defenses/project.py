"""E3 defense: project GNN soft outputs onto the hard MILP feasible set.

Given the GNN's per-decision probabilities, solve a small ILP that maximizes agreement
with those probabilities subject to the SAME hard constraints as upstream solve_problem
(eq. 7c-7n). Because the constraints mirror the checker exactly, any projected solution
is feasible by construction (validated: check_feasibility == 0). The RF-cap constraint
also bounds any single AP's captured resources by N_RF regardless of how it lied.

Objective weight per binary var = (2*prob - 1): rewards setting high-confidence (p>0.5)
decisions to 1 and low-confidence to 0 -- the maximum-likelihood feasible allocation.

Uses gurobipy's restricted license (problem is ~156 vars / ~600 constraints, well under
the 2000/2000 limit). No academic license needed.
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def _bcast(v, n):
    a = np.asarray(v, float)
    return np.full(n, a) if a.ndim == 0 else a


def project(probs, params, time_limit=None):
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

    m = gp.Model("project"); m.Params.OutputFlag = 0
    if time_limit: m.Params.TimeLimit = time_limit
    x = m.addVars(A, U, vtype=GRB.BINARY, name="x")
    ytx = m.addVars(A, T, vtype=GRB.BINARY, name="ytx")
    yrx = m.addVars(A, T, vtype=GRB.BINARY, name="yrx")
    tau = m.addVars(A, vtype=GRB.BINARY, name="tau")
    s = m.addVars(T, vtype=GRB.BINARY, name="s")

    # --- constraints mirror solve_problem exactly ---
    for a in range(A):
        for u in range(U):
            m.addConstr(x[a, u] <= tau[a])                                   # 7c
        for t in range(T):
            m.addConstr(ytx[a, t] <= tau[a])                                 # 7d
            m.addConstr(yrx[a, t] <= 1 - tau[a])                             # 7e
        m.addConstr(gp.quicksum(x[a, u] for u in range(U))
                    + gp.quicksum(ytx[a, t] for t in range(T)) <= N_RF[a])   # 7f
        m.addConstr(gp.quicksum(yrx[a, t] for t in range(T)) <= C_rx[a])     # 7n
    for t in range(T):
        m.addConstr(gp.quicksum(ytx[a, t] for a in range(A)) <= K_tx[t] * s[t])  # 7h
        m.addConstr(gp.quicksum(yrx[a, t] for a in range(A)) >= s[t])            # 7i (rx>=s)
        m.addConstr(gp.quicksum(ytx[a, t] for a in range(A)) >= s[t])            # 7j (tx>=s)
        m.addConstr(gp.quicksum(yrx[a, t] for a in range(A)) <= K_rx[t] * s[t])  # 7m
    for a in range(A):
        for u in range(U):
            for up in range(u + 1, U):
                m.addConstr(x[a, u] + x[a, up] <= 2 + rho - S[a, u, up])     # 7g

    # --- objective: maximize agreement with GNN confidences ---
    obj = (gp.quicksum((2*px[a, u] - 1) * x[a, u] for a in range(A) for u in range(U))
           + gp.quicksum((2*ptau[a] - 1) * tau[a] for a in range(A))
           + gp.quicksum((2*ps[t] - 1) * s[t] for t in range(T))
           + gp.quicksum((2*pytx[a, t] - 1) * ytx[a, t] for a in range(A) for t in range(T))
           + gp.quicksum((2*pyrx[a, t] - 1) * yrx[a, t] for a in range(A) for t in range(T)))
    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()
    if m.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or m.SolCount == 0:
        raise RuntimeError(f"projection failed, status={m.status}")

    X = np.array([[int(round(x[a, u].X)) for u in range(U)] for a in range(A)], np.int8)
    YTX = np.array([[int(round(ytx[a, t].X)) for t in range(T)] for a in range(A)], np.int8)
    YRX = np.array([[int(round(yrx[a, t].X)) for t in range(T)] for a in range(A)], np.int8)
    TAU = np.array([int(round(tau[a].X)) for a in range(A)], np.int8)
    S_ = np.array([int(round(s[t].X)) for t in range(T)], np.int8)
    return {"x": X, "tau": TAU, "s": S_, "y_tx": YTX, "y_rx": YRX}
