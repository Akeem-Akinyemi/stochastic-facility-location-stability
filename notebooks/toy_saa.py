import numpy as np

c, p = 1.0, 3.0                      # cost per unit of capacity, penalty per unit unmet
cv, mean = 0.4, 100.0
s2 = np.log(1 + cv**2)
mu, sg = np.log(mean) - s2 / 2, np.sqrt(s2)

def cost(q, d):
    """Average cost of capacity q over demand sample d."""
    return c * q + p * np.maximum(d - q, 0).mean()

rng = np.random.default_rng(0)
big = np.exp(mu + sg * rng.standard_normal(1_000_000))   # stands in for the truth
q_star = np.quantile(big, 1 - c / p)                      # optimum is this quantile
v_star = cost(q_star, big)
print(f"best possible expected cost: {v_star:.2f}  (capacity {q_star:.1f})")

for n in (10, 50, 200):
    promised, realized = [], []
    for _ in range(500):
        d = np.exp(mu + sg * rng.standard_normal(n))
        q = np.quantile(d, 1 - c / p)                     # optimal capacity for this sample
        promised.append(cost(q, d))                       # what the sample claims
        realized.append(cost(q, big))                     # what it really costs
    print(f"N={n:4d}  promised {np.mean(promised):6.2f}   realized {np.mean(realized):6.2f}")