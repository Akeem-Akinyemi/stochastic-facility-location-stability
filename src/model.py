"""
We build the deterministic equivalent (the sample average approximation)
as explicit sparse matrices and solve it with HiGHS through scipy.optimize.milp.
We also provide the second-stage cost of a fixed plan, used to evaluate plans.

Notation follows docs/model_notes.md:
    i = site, j = zone, s = scenario
    y_i      binary, 1 if site i is opened
    x_ijs    units served from site i to zone j in scenario s
    u_js     units of unmet demand in zone j in scenario s
"""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix


def build_saa(f, K, c, pi, D):
    """
    Build the matrices of the SAA model.

    f : (I,) opening costs        K : (I,) capacities
    c : (I, J) service costs      pi : penalty per unit of unmet demand
    D : (N, J) demand scenarios, one per row

    Variable order: [ y (I) | x (N*I*J) | u (N*J) ].
    Returns the objective vector, the constraint matrix with its bounds,
    the integrality vector, and the variable bounds.
    """
    f, K, c, D = map(np.asarray, (f, K, c, D))
    I, J = c.shape
    N = D.shape[0]
    n_x, n_u = N * I * J, N * J
    n_var = I + n_x + n_u

    def x_idx(i, j, s):
        return I + s * I * J + i * J + j

    def u_idx(j, s):
        return I + n_x + s * J + j

    # Objective: opening costs, then the scenario-average serving and penalty costs.
    cost = np.zeros(n_var)
    cost[:I] = f
    s_g, i_g, j_g = np.meshgrid(np.arange(N), np.arange(I), np.arange(J), indexing="ij")
    cost[x_idx(i_g, j_g, s_g).ravel()] = np.tile(c.ravel(), N) / N
    cost[I + n_x:] = pi / N

    rows, cols, vals = [], [], []

    # Demand balance (1): sum_i x_ijs + u_js = d_js, one row per (s, j).
    s_b, j_b = np.meshgrid(np.arange(N), np.arange(J), indexing="ij")
    bal_row = (s_b * J + j_b).ravel()
    for i in range(I):
        rows.append(bal_row)
        cols.append(x_idx(i, j_b, s_b).ravel())
        vals.append(np.ones(N * J))
    rows.append(bal_row)
    cols.append(u_idx(j_b, s_b).ravel())
    vals.append(np.ones(N * J))
    lower = list(D.ravel())
    upper = list(D.ravel())

    # Capacity (2): sum_j x_ijs - K_i y_i <= 0, one row per (s, i).
    s_c, i_c = np.meshgrid(np.arange(N), np.arange(I), indexing="ij")
    cap_row = N * J + (s_c * I + i_c).ravel()
    for j in range(J):
        rows.append(cap_row)
        cols.append(x_idx(i_c, j, s_c).ravel())
        vals.append(np.ones(N * I))
    rows.append(cap_row)
    cols.append(i_c.ravel())
    vals.append(-K[i_c.ravel()])
    lower += [-np.inf] * (N * I)
    upper += [0.0] * (N * I)

    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(N * J + N * I, n_var)).tocsr()

    integrality = np.zeros(n_var)
    integrality[:I] = 1                      # only y is integer
    ub = np.full(n_var, np.inf)
    ub[:I] = 1
    return cost, A, np.array(lower), np.array(upper), integrality, Bounds(0, ub)


def solve_saa(f, K, c, pi, D, time_limit=None):
    """
    Solve the SAA model. Returns a dict with the chosen sites y, the objective
    value (the cost the sample promises), and the solver status.
    """
    I = len(f)
    cost, A, lo, hi, integ, bounds = build_saa(f, K, c, pi, D)
    options = {"mip_rel_gap": 1e-9}
    if time_limit is not None:
        options["time_limit"] = time_limit
    res = milp(cost, constraints=LinearConstraint(A, lo, hi),
               integrality=integ, bounds=bounds, options=options)
    if res.x is None:
        raise RuntimeError(f"solver failed: {res.message}")
    return {"y": np.round(res.x[:I]).astype(int), "objective": res.fun,
            "status": res.status, "message": res.message}


def second_stage_cost(y, K, c, pi, d):
    """
    Cost of serving demand vector d with the sites in y (serving cost plus penalty).
    This is Q(y, s) from the notes: one small linear program, no integers.
    Only open sites enter the LP; with no open site, all demand is unmet.
    """
    y, K, c, d = np.asarray(y), np.asarray(K), np.asarray(c), np.asarray(d)
    open_sites = np.flatnonzero(y)
    J = len(d)
    if open_sites.size == 0:
        return pi * d.sum()
    m = open_sites.size
    c_open, K_open = c[open_sites], K[open_sites]
    cost = np.concatenate([c_open.ravel(), np.full(J, pi)])   # [x (m*J) | u (J)]
    A_eq = np.zeros((J, m * J + J))
    for j in range(J):
        A_eq[j, np.arange(m) * J + j] = 1
        A_eq[j, m * J + j] = 1
    A_ub = np.zeros((m, m * J + J))
    for k in range(m):
        A_ub[k, k * J:(k + 1) * J] = 1
    res = linprog(cost, A_ub=A_ub, b_ub=K_open, A_eq=A_eq, b_eq=d,
                  bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"Second-stage LP failed: {res.message}")
    return float(res.fun)