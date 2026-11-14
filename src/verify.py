"""
Independent checks of a solved SAA model.

We recompute everything from the returned solution vector and from separate
linear programs, so a bug in the constraint matrix cannot hide itself.
"""

import numpy as np

from src.model import second_stage_cost


def verify_solution(f, K, c, pi, D, res):
    """
    Return a dict of residuals. All entries should be zero up to solver tolerance
    (min_variable should be non-negative).
    """
    f, K, c, D = map(np.asarray, (f, K, c, D))
    I, J = c.shape
    N = D.shape[0]
    z = res["x"]
    y = z[:I]
    x = z[I:I + N * I * J].reshape(N, I, J)       # x[s, i, j]
    u = z[I + N * I * J:].reshape(N, J)           # u[s, j]
    y_int = np.round(y)

    out = {}
    out["y_integrality"] = np.max(np.abs(y - y_int))
    out["min_variable"] = min(x.min(), u.min(), y.min())
    # Constraint (1): served + unmet equals demand.
    out["balance_residual"] = np.max(np.abs(x.sum(axis=1) + u - D))
    # Constraint (2): a site serves at most its capacity, and nothing if closed.
    out["capacity_violation"] = max(np.max(x.sum(axis=2) - K * y_int), 0.0)
    # Objective recomputed directly from the variables.
    obj_vars = f @ y_int + (np.einsum("sij,ij->", x, c) + pi * u.sum()) / N
    out["objective_from_variables"] = obj_vars - res["objective"]
    # Objective recomputed from separate small LPs for the chosen plan.
    q = [second_stage_cost(y_int, K, c, pi, d) for d in D]
    out["objective_independent"] = f @ y_int + np.mean(q) - res["objective"]
    return out