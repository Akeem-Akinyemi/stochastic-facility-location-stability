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


## Toy example for hand verification.

This small example checks our two-stage stochastic facility location formulation before we implement the larger model. We have three candidate sites, four demand zones, and three equally likely scenarios. The first-stage decision is which sites to open; the second-stage decision is how much each open site supplies to each zone, with any remaining demand charged a penalty.

### Data

**Site opening costs and capacities**

| Site $i$ | Opening cost $f_i$ | Capacity $K_i$ |
|---|---:|---:|
| 1 | 100 | 50 |
| 2 | 120 | 60 |
| 3 | 140 | 40 |

**Service cost per unit, $c_{ij}$** (rows are sites; columns are demand zones)

| Site | Zone 1 | Zone 2 | Zone 3 | Zone 4 |
|---|---:|---:|---:|---:|
| 1 | 2 | 3 | 6 | 7 |
| 2 | 6 | 2 | 3 | 4 |
| 3 | 7 | 5 | 2 | 1 |

The penalty for each unit of unmet demand is $\pi=12$, greater than every service cost in this example.

**Demand by scenario, $d_{js}$**

| Scenario | Zone 1 | Zone 2 | Zone 3 | Zone 4 | Total demand |
|---|---:|---:|---:|---:|---:|
| $s_1$ (low) | 10 | 15 | 10 | 5 | 40 |
| $s_2$ (medium) | 20 | 20 | 20 | 10 | 70 |
| $s_3$ (high) | 30 | 25 | 30 | 20 | 105 |

Every scenario has probability $1/3$.

### Cost calculation

For a chosen set of open sites $A$, the opening cost is

$$F(A)=\sum_{i\in A}f_i.$$

For scenario $s$, we minimize the **second-stage operating cost**

$$Q(A,s)=\min_{x,u}\left\{\sum_{i\in A}\sum_{j=1}^{4}c_{ij}x_{ijs}+12\sum_{j=1}^{4}u_{js}\right\}$$

subject to demand balance and the capacities of open sites. The total SAA objective for plan $A$ is

$$\boxed{V(A)=F(A)+\frac{Q(A,1)+Q(A,2)+Q(A,3)}{3}.}$$

We pay the opening cost **once**, not once per scenario.

### Answer key: brute force over all eight opening plans

The values were computed by solving each second-stage problem as a linear program with the HiGHS solver. We also solved the full model as a single mixed-integer program, which selected the same plan with the same cost. We verified five entries by hand (see below).

| Open sites | Opening cost | $Q(A,1)$ | $Q(A,2)$ | $Q(A,3)$ | $V(A)$ |
|---|---:|---:|---:|---:|---:|
| None | 0 | 480 | 840 | 1260 | 860.00 |
| {3} | 140 | 170 | 460 | 840 | 630.00 |
| {2} | 120 | 140 | 320 | 700 | 506.67 |
| {2,3} | 260 | 115 | 210 | 350 | 485.00 |
| {1} | 100 | 160 | 400 | 780 | 546.67 |
| {1,3} | 240 | 90 | 150 | 360 | 440.00 |
| **{1,2}** | **220** | **100** | **180** | **295** | **411.67** |
| {1,2,3} | 360 | 75 | 130 | 200 | 495.00 |

For the winning plan, the average operating cost is

$$\frac{100+180+295}{3}=\frac{575}{3}=191.666\ldots.$$

Adding the fixed opening cost $100+120=220$ gives

$$V(\{1,2\})=220+\frac{575}{3}=\boxed{411.67}.$$

Since $411.67$ is lower than the total for every other plan, **open sites 1 and 2**.

### Hand calculations

#### 1) Sites {1,2}, low-demand scenario $s_1$

Demand is $(10,15,10,5)$, totaling $40$. The two open sites have capacities $50$ and $60$, so we can serve all demand without exceeding capacity.

- Zone 1: supply 10 units from site 1 at 2 per unit: $10\times2=20$.
- Zone 2: supply 15 units from site 2 at 2 per unit: $15\times2=30$.
- Zone 3: supply 10 units from site 2 at 3 per unit: $10\times3=30$.
- Zone 4: supply 5 units from site 2 at 4 per unit: $5\times4=20$.

Site 1 supplies $10\le50$ units, and site 2 supplies $15+10+5=30\le60$ units. Unmet demand is zero. Each zone is served from its cheapest **open** site, so no cheaper feasible allocation exists.

$$Q(\{1,2\},1)=20+30+30+20+12(0)=\boxed{100}.$$

#### 2) Sites {1,2}, high-demand scenario $s_3$

Demand is $(30,25,30,20)$, totaling $105$. Available capacity is $50+60=110$, enough to serve everyone, but allocation matters.

**Start with the cheapest site for each zone:**

- Zone 1: 30 units from site 1, cost $30\times2=60$.
- Zone 2: 25 units from site 2, cost $25\times2=50$.
- Zone 3: 30 units from site 2, cost $30\times3=90$.
- Zone 4: 20 units from site 2, cost $20\times4=80$.

This would cost $60+50+90+80=280$, but site 2 would supply $25+30+20=75$ units, exceeding its capacity of $60$ by **15 units**.

**Finding the cheapest reassignment from site 2 to site 1:**

| Zone | Site 2 cost/unit | Site 1 cost/unit | Extra cost if moved |
|---|---:|---:|---:|
| 2 | 2 | 3 | $3-2=1$ |
| 3 | 3 | 6 | $6-3=3$ |
| 4 | 4 | 7 | $7-4=3$ |

Moving 15 units of zone 2 to site 1 costs only $15\times1=15$ extra, the smallest possible increase. Site 1 has $50-30=20$ spare units, so this reassignment is feasible.

**Final deliveries:**

- Site 1: zone 1 receives 30 units; zone 2 receives 15 units. Load $30+15=45\le50$. Cost $30\times2+15\times3=60+45=105$.
- Site 2: zone 2 receives 10 units; zone 3 receives 30 units; zone 4 receives 20 units. Load $10+30+20=60\le60$. Cost $10\times2+30\times3+20\times4=20+90+80=190$.
- All zones receive exactly their requested demand; unmet demand is zero.

$$Q(\{1,2\},3)=105+190=\boxed{295}.$$

**Why is this optimal?** The hypothetical cheapest-site allocation costs 280 before capacity restrictions. Any feasible all-served allocation must move at least 15 units away from site 2; every such move costs at least 1 extra per unit, so the cost is at least $280+15=295$. Our allocation attains this bound. Leaving demand unmet cannot improve it, since every available delivery costs less than the penalty of 12 per unit, and all demand can be served with these capacities.

#### 3) Site {1} only, high-demand scenario $s_3$

Only site 1 is open, with capacity 50. Total demand is $30+25+30+20=105$, so at least $105-50=55$ units must remain unmet.

Since the penalty is 12 per unit, serving one unit saves $12-c_{1j}$. The savings are largest for zone 1 (saving $12-2=10$ per unit) and next largest for zone 2 (saving $12-3=9$ per unit). Therefore, fill capacity using the cheapest service costs first:

- Supply all 30 units to zone 1: $30\times2=60$.
- Supply 20 of zone 2's 25 units: $20\times3=60$.
- Zone 2 has 5 unmet units; zone 3 has 30; zone 4 has 20. Total unmet $5+30+20=55$.
- Penalty: $55\times12=660$.

Site 1 supplies exactly $30+20=50$ units, its capacity.

$$Q(\{1\},3)=60+60+660=\boxed{780}.$$

#### 4) No sites, medium-demand scenario $s_2$

No facilities are open, so total available capacity is zero and no demand can be served. Scenario 2 demand is $20+20+20+10=70$ units, all unmet.

$$Q(\varnothing,2)=70\times12=\boxed{840}.$$

#### 5) Sites {2,3}, low-demand scenario $s_1$

Demand is $(10,15,10,5)$ and the open capacities are $60$ at site 2 and $40$ at site 3. The total demand is only 40, so there is ample capacity; choose the cheaper of the two open sites for each zone:

| Zone | Demand | Cheapest open site | Cost/unit | Cost |
|---|---:|---|---:|---:|
| 1 | 10 | Site 2 | 6 | $10\times6=60$ |
| 2 | 15 | Site 2 | 2 | $15\times2=30$ |
| 3 | 10 | Site 3 | 2 | $10\times2=20$ |
| 4 | 5 | Site 3 | 1 | $5\times1=5$ |

Site 2 serves $10+15=25\le60$ units, and site 3 serves $10+5=15\le40$ units. No demand is unmet. Because every zone uses its cheapest open site and capacities are respected, this allocation is optimal.

$$Q(\{2,3\},1)=60+30+20+5=\boxed{115}.$$

### Why the average-demand plan can be worse

The mean demand across the three scenarios is

$$\bar d=\left(\frac{10+20+30}{3},\frac{15+20+25}{3},\frac{10+20+30}{3},\frac{5+10+20}{3}\right)=\left(20,20,20,\frac{35}{3}\right).$$

If we optimize against **only this one mean-demand vector**, we evaluate each plan as its opening cost plus the second-stage cost for $\bar d$:

| Open sites | Cost under mean demand |
|---|---:|
| None | 860.00 |
| {3} | 613.33 |
| {2} | 456.67 |
| {2,3} | 471.67 |
| {1} | 520.00 |
| **{1,3}** | **391.67** |
| {1,2} | 406.67 |
| {1,2,3} | 491.67 |

The best plan under mean demand is therefore sites {1,3}. We can check this entry by hand. With sites 1 and 3 open, zones 1 and 2 are served from site 1 (load $40\le50$, cost $20\times2+20\times3=100$), and zones 3 and 4 from site 3 (load $20+\tfrac{35}{3}=\tfrac{95}{3}\le40$, cost $20\times2+\tfrac{35}{3}\times1=\tfrac{155}{3}$). The second-stage cost is $100+\tfrac{155}{3}=\tfrac{455}{3}\approx151.67$, and adding the opening cost $240$


## Implementation notes

We implemented the two-stage model in `src/model.py` by building the constraint matrix explicitly and solving it with HiGHS through SciPy.

**Variable order.** All variables sit in one vector: the site-opening variables $y$ first, then the service quantities $x$, then the unmet-demand quantities $u$. The $y$ block comes first because it is the first-stage decision and does not depend on the scenario. The $x$ and $u$ blocks are grouped by scenario, since their values depend on the demand observed.

**Constraint rows.** The matrix has two groups of rows:
- *Demand balance*, one row per scenario and zone: coefficient 1 on every $x_{ijs}$ serving that zone and on $u_{js}$, with right-hand side $d_{js}$.
- *Capacity*, one row per scenario and site: coefficient 1 on every $x_{ijs}$ leaving that site and $-K_i$ on $y_i$, with right-hand side 0.

We store the matrix in sparse format because almost all entries are zero. For $N=200$ scenarios on the full instance, a dense matrix would hold about 1.08 billion entries, while only about 200,000 are nonzero.

**Verification.** We check the code against the toy example from the Stage G section (`tests/test_toy.py`). The tests compare the second-stage costs of all eight opening plans with the hand-checked answer key. They also confirm that the full model selects the same sites as direct enumeration of all plans. Run them with `python -m pytest -q`.