# Two-stage stochastic facility location model

## Sets and data

- $I$: candidate sites, $i = 1,\dots,12$
- $J$: demand zones, $j = 1,\dots,40$
- $S$: demand scenarios, $s = 1,\dots,N$, each with probability $1/N$
- $f_i$: cost of opening site $i$
- $K_i$: capacity of site $i$
- $c_{ij}$: cost of serving one unit of demand from site $i$ to zone $j$
- $\pi$: penalty per unit of unmet demand
- $d_{js}$: demand of zone $j$ in scenario $s$

## Decision variables

- $y_i \in \{0,1\}$: equals 1 if we open site $i$ (first stage).
- $x_{ijs} \ge 0$: units served from site $i$ to zone $j$ in scenario $s$ (second stage).
- $u_{js} \ge 0$: units of unmet demand in zone $j$ in scenario $s$ (second stage).

The sites $y_i$ have no scenario index because we open them once, before demand is known. The service and unmet variables depend on the scenario because we react to the demand that occurs.

## Model

$$
\min \; \sum_{i \in I} f_i\, y_i \;+\; \frac{1}{N} \sum_{s \in S} \left[ \sum_{i \in I} \sum_{j \in J} c_{ij}\, x_{ijs} \;+\; \pi \sum_{j \in J} u_{js} \right]
$$

subject to

$$
\sum_{i \in I} x_{ijs} + u_{js} = d_{js} \qquad \forall j \in J,\; s \in S \tag{1}
$$

$$
\sum_{j \in J} x_{ijs} \le K_i\, y_i \qquad \forall i \in I,\; s \in S \tag{2}
$$

$$
y_i \in \{0,1\}, \quad x_{ijs} \ge 0, \quad u_{js} \ge 0 \tag{3}
$$

## What each part means

- **Objective, first term:** the fixed cost of the sites we open.
- **Objective, second term:** the average over scenarios of the serving cost plus the penalty for unmet demand. This is the sample average approximation of the expected second-stage cost.
- **Constraint (1):** in every scenario, each zone's demand is either served by some site or counted as unmet.
- **Constraint (2):** a site serves at most its capacity, and only if we opened it. This links the first stage to the second.
- **Constraint (3):** the sites are yes/no decisions, and quantities served and unmet cannot be negative.
- **Complete recourse:** even if we open no sites, setting $u_{js} = d_{js}$ satisfies constraint (1), so every scenario is always feasible.
- **Why $\pi > c_{ij}$:** otherwise the model would prefer leaving demand unmet even when an open site has spare capacity.