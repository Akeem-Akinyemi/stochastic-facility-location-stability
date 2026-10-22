# Two-stage stochastic facility location model

## Sets and data

- $I$: candidate sites, $i=1,\ldots,12$.
- $J$: demand zones, $j=1,\ldots,40$.
- $S$: sampled demand scenarios, $s=1,\ldots,N$, each assigned weight $1/N$.
- $f_i$: cost of opening site $i$.
- $K_i$: capacity of site $i$.
- $c_{ij}$: cost of serving one unit from site $i$ to zone $j$.
- $\pi$: penalty per unit of unmet demand (40 in our city).
- $d_{js}$: demand in zone $j$ under scenario $s$.

## Decision variables

- $y_i\in\{0,1\}$: 1 if site $i$ is opened, otherwise 0 (first stage).
- $x_{ijs}\geq0$: units delivered from site $i$ to zone $j$ in scenario $s$ (second stage).
- $u_{js}\geq0$: unmet demand in zone $j$ in scenario $s$ (second stage).

We choose the same sites before demand is known, so $y_i$ has no scenario index. Deliveries and unmet demand can change after each scenario is revealed.

## Objective

$$
\min_{y,x,u}\quad \sum_{i\in I}f_i y_i + \frac{1}{N}\sum_{s\in S}\left(\sum_{i\in I}\sum_{j\in J}c_{ij}x_{ijs}+\pi\sum_{j\in J}u_{js}\right).
$$

We minimize the cost of opening facilities plus the average cost of deliveries and unmet demand across the sampled scenarios.

## Constraints

**(1) Demand balance**

$$
\sum_{i\in I}x_{ijs}+u_{js}=d_{js}\qquad\forall j\in J,\ s\in S.
$$

In each scenario, every zone's demand must either be supplied by open sites or counted as unmet demand.

**(2) Facility capacity**

$$
\sum_{j\in J}x_{ijs}\leq K_i y_i\qquad\forall i\in I,\ s\in S.
$$

Each site can supply no more than its capacity, and a closed site cannot supply anything.

**(3) Variable domains**

$$
y_i\in\{0,1\},\qquad x_{ijs}\geq0,\qquad u_{js}\geq0.
$$

Opening a site is a yes-or-no decision, while delivered and unmet quantities cannot be negative.
## Full Model
$$
\min_{y,x,u}\quad \sum_{i\in I}f_i y_i + \frac{1}{N}\sum_{s\in S}\left(\sum_{i\in I}\sum_{j\in J}c_{ij}x_{ijs}+\pi\sum_{j\in J}u_{js}\right).
$$

Subject to :

$$
\sum_{i\in I}x_{ijs}+u_{js}=d_{js}\qquad\forall j\in J,\ s\in S.
$$

$$
\sum_{j\in J}x_{ijs}\leq K_i y_i\qquad\forall i\in I,\ s\in S.
$$

$$
y_i\in\{0,1\},\qquad x_{ijs}\geq0,\qquad u_{js}\geq0.
$$

## Complete recourse

Even if no site is opened, the model can set all deliveries to zero and unmet demand equal to demand, so the second-stage constraints remain feasible (assuming demand is nonnegative).

## Model size

There are 12 binary opening variables, $12\times40\times N=480N$ delivery variables, and $40N$ unmet-demand variables. Thus the number of variables is $12+520N$, and the number of demand-balance and capacity constraints is $(40+12)N=52N$.

| Number of scenarios ($N$) | Variables ($12+520N$) | Constraints ($52N$) |
|---:|---:|---:|
| 10 | 5,212 | 520 |
| 50 | 26,012 | 2,600 |
| 200 | 104,012 | 10,400 |

Variable-domain restrictions are not included in the constraint counts.

## What if the penalty is lower than some service costs?

If the penalty for unmet demand is lower than the cost of serving a customer from a particular site, the model may prefer leaving that demand unmet even when the site has available capacity. This would go against our intention of encouraging the model to serve customers whenever it is cheaper than paying the penalty.
