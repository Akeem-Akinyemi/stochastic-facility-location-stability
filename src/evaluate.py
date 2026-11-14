"""
evaluate.py: cost of a fixed plan on a set of scenarios.

For a plan (the set of open sites) the scenarios do not interact, so the cost on
a test set is the opening cost plus the average of one small linear program per
scenario. We cache the per-scenario costs by plan, because the same plan is
chosen many times across repetitions, and keep the per-scenario values so that
later we can compare plans on identical scenarios and compute risk measures.

When highspy is installed, each plan's linear program is built once and only the
right-hand side (the demand) changes between scenarios, which is about ten times
faster than calling linprog for every scenario. Without highspy we fall back to
second_stage_cost from model.py; both give the same values.
"""

import numpy as np

from src.model import second_stage_cost

try:
    import highspy
except ImportError:          # the fallback below still works
    highspy = None


class PlanEvaluator:
    """Evaluate plans on a fixed scenario set (rows of `scenarios` are demand vectors)."""

    def __init__(self, city, scenarios, use_highs=True):
        self.city = city
        self.scenarios = np.asarray(scenarios, dtype=float)
        self.use_highs = use_highs and highspy is not None
        self._cache = {}

    @staticmethod
    def _key(y):
        return tuple(np.flatnonzero(y).tolist())

    def scenario_costs(self, y):
        """Second-stage cost of plan y in every scenario, shape (M,)."""
        key = self._key(y)
        if key not in self._cache:
            self._cache[key] = self._compute(key)
        return self._cache[key]

    def expected_cost(self, y):
        """Opening cost plus the average second-stage cost over the scenario set."""
        key = self._key(y)
        return self.city.open_cost[list(key)].sum() + self.scenario_costs(y).mean()

    def _compute(self, key):
        city, D = self.city, self.scenarios
        if len(key) == 0:                              # nothing open: all demand is unmet
            return city.penalty * D.sum(axis=1)
        if not self.use_highs:
            y = np.zeros(len(city.open_cost), dtype=int)
            y[list(key)] = 1
            return np.array([second_stage_cost(y, city.capacity, city.service_cost,
                                               city.penalty, d) for d in D])
        return self._compute_highs(list(key))

    def _compute_highs(self, sites):
        """Build the plan's linear program once; only the demand changes per scenario."""
        city, D = self.city, self.scenarios
        m, J = len(sites), D.shape[1]
        n = m * J + J                                  # variables: x (m*J) then u (J)
        inf = highspy.kHighsInf

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        for _ in range(n):
            h.addVar(0.0, inf)
        cost = np.concatenate([city.service_cost[sites].ravel(), np.full(J, city.penalty)])
        h.changeColsCost(n, np.arange(n, dtype=np.int32), cost)
        for j in range(J):                             # rows 0..J-1: served + unmet = demand
            idx = np.concatenate([np.arange(m) * J + j, [m * J + j]]).astype(np.int32)
            h.addRow(0.0, 0.0, m + 1, idx, np.ones(m + 1))
        for k in range(m):                             # next m rows: site capacity
            h.addRow(-inf, float(city.capacity[sites[k]]), J,
                     np.arange(k * J, (k + 1) * J, dtype=np.int32), np.ones(J))

        out = np.empty(len(D))
        for s, d in enumerate(D):
            for j in range(J):
                h.changeRowBounds(j, d[j], d[j])
            h.run()
            assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
            out[s] = h.getInfo().objective_function_value
        return out