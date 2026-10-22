"""
demand.py: the true demand model.

Demand in each zone is lognormal with mean equal to the city's mean demand
and a fixed coefficient of variation. Zones are spatially correlated through
the underlying normal variables: corr(log D_j, log D_k) = exp(-distance / corr_length).
The correlation of the demands themselves is slightly lower, e.g. 0.351 instead
of 0.368 at 3 km with cv = 0.4. Experiments treat this model as the truth.
"""

from dataclasses import dataclass

import numpy as np

from src.city import City, make_city


@dataclass
class DemandModel:
    mu: np.ndarray      # (J,) mean of log-demand
    sigma: float        # std of log-demand
    chol: np.ndarray    # (J, J) Cholesky factor of the log-demand correlation matrix

    def sample(self, n, rng):
        """Draw n demand vectors; returns an (n, J) array."""
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("n must be a positive integer")
        # Correlated standard normals: each row has covariance chol @ chol.T.
        z = rng.standard_normal((n, self.mu.size)) @ self.chol.T
        return np.exp(self.mu + self.sigma * z)


def make_demand_model(city: City, cv=0.4, corr_length=3.0):
    """
    Build the true demand model for a city.

    cv          : coefficient of variation of demand in each zone
    corr_length : distance (km) over which log-demand correlation decays
    """
    if cv <= 0:
        raise ValueError("cv must be positive")
    if corr_length <= 0:
        raise ValueError("corr_length must be positive")

    # For a lognormal, mean = exp(mu + sigma^2 / 2) and cv^2 = exp(sigma^2) - 1.
    sigma2 = np.log(1 + cv**2)
    mu = np.log(city.mean_demand) - sigma2 / 2

    # Zone-to-zone distances and the exponential correlation matrix.
    d = np.linalg.norm(city.zone_xy[:, None, :] - city.zone_xy[None, :, :], axis=2)
    corr = np.exp(-d / corr_length)
    # A tiny jitter guards the factorization for other cities. It changes
    # variances by 1e-8, which is negligible.
    chol = np.linalg.cholesky(corr + 1e-8 * np.eye(corr.shape[0]))

    return DemandModel(mu=mu, sigma=np.sqrt(sigma2), chol=chol)


if __name__ == "__main__":
    # Check the model by simulation and against closed-form values.
    city = make_city()
    model = make_demand_model(city)
    rng = np.random.default_rng(1)
    x = model.sample(100_000, rng)

    rel_err = np.abs(x.mean(axis=0) - city.mean_demand) / city.mean_demand
    print("Largest relative error in zone means:", rel_err.max().round(3))
    print("Average zone cv:", (x.std(axis=0) / x.mean(axis=0)).mean().round(3))

    total = x.sum(axis=1)
    print("Total demand: mean", total.mean().round(1), " std (simulated)", total.std().round(1))

    # Exact total std: Cov(D_j, D_k) = m_j m_k (exp(sigma^2 R_jk) - 1) for a lognormal.
    corr = model.chol @ model.chol.T
    m = city.mean_demand
    exact_std = np.sqrt((np.outer(m, m) * (np.exp(model.sigma**2 * corr) - 1)).sum())
    print("Total demand std (exact):", exact_std.round(1))

    # With independent zones, the total std would be much smaller.
    indep_std = 0.4 * np.sqrt((m**2).sum())
    print("Total demand std if zones were independent:", indep_std.round(1))
    