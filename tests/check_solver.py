from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

# minimize -x - y subject to x + y <= 1.5, x and y integer in [0, 1]
res = milp(c=[-1, -1],
           constraints=LinearConstraint([[1, 1]], ub=1.5),
           integrality=[1, 1],
           bounds=Bounds(0, 1))
print(res.x, res.fun)