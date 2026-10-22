from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class City:
    site_xy: np.ndarray       # (I, 2) site coordinates in km
    zone_xy: np.ndarray       # (J, 2) zone coordinates in km
    open_cost: np.ndarray     # (I,) fixed cost of opening each site
    capacity: np.ndarray      # (I,) maximum demand each site can serve
    mean_demand: np.ndarray   # (J,) expected demand in each zone
    dist: np.ndarray          # (I, J) distance from each site to each zone
    service_cost: np.ndarray  # (I, J) cost per unit served
    penalty: float            # cost per unit of unmet demand


def validate(c):
    assert np.all(c.capacity > 0), "capacities must be positive"
    assert np.all(c.mean_demand > 0), "mean demand must be positive"
    assert np.all(c.open_cost > 0), "opening costs must be positive"
    assert np.all(c.service_cost >= 0), "service costs must be non-negative"
    assert c.penalty > c.service_cost.max(), "penalty must exceed max service cost"


def make_city(n_sites=12, n_zones=40, size=10.0, seed=2024,
              cost_per_km=2.0, penalty=40.0):
    rng = np.random.default_rng(seed)
    centre = np.array([size / 2, size / 2])

    site_xy = rng.uniform(0, size, (n_sites, 2))
    zone_xy = rng.uniform(0, size, (n_zones, 2))

    # Demand is about 25 at the centre, about 7 at the corners,
    # and tends to 5 far away
    d_zone_centre = np.linalg.norm(zone_xy - centre, axis=1)
    mean_demand = 5 + 20 * np.exp(-d_zone_centre / (size / 3))

    # Capacity varies between sites
    capacity = rng.uniform(80, 160, n_sites).round()

    # Opening cost grows with capacity and with closeness to the centre
    d_site_centre = np.linalg.norm(site_xy - centre, axis=1)
    land_factor = 1 + 0.5 * np.exp(-d_site_centre / (size / 3))
    open_cost = ((300 + 3 * capacity) * land_factor
                 + rng.uniform(-50, 50, n_sites)).round()

    dist = np.linalg.norm(site_xy[:, None, :] - zone_xy[None, :, :], axis=2)
    service_cost = cost_per_km * dist

    city = City(site_xy, zone_xy, open_cost, capacity,
                mean_demand, dist, service_cost, penalty)
    validate(city)
    return city


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = make_city()
    print("Total mean demand:", c.mean_demand.sum().round(1))
    print("Mean demand range:", c.mean_demand.min().round(1), "to", c.mean_demand.max().round(1))
    print("Total capacity:", c.capacity.sum())
    print("Capacity / demand ratio:", (c.capacity.sum() / c.mean_demand.sum()).round(2))
    print("Opening cost range:", c.open_cost.min(), "to", c.open_cost.max())
    print("Max service cost per unit:", c.service_cost.max().round(1))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(*c.zone_xy.T, s=c.mean_demand * 6, alpha=0.4,
               label="Demand zones (size = mean demand)")
    ax.scatter(*c.site_xy.T, marker="s", s=c.capacity, color="red",
               label="Candidate sites (size = capacity)")
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Synthetic city")
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("results/figures/city.png", dpi=150, bbox_inches="tight")