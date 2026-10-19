# Stability of Scenario-Based Stochastic Facility Location:
# Scenario Generation and Stabilization

## Problem statement
Facility location under uncertain demand can be modelled as a two-stage stochastic mixed-integer program, in which site-opening decisions are made before demand is revealed and are approximated using a finite set of demand scenarios. In practice the chosen sites depend strongly on which scenarios are sampled, and plans optimized on one sample tend to perform worse on new demand. This project formulates and solves a scenario-based facility location model, and investigates three questions: (1) how unstable are the optimal site selection and cost as the number and sampling of scenarios change, and how large is the gap between promised and realized cost; (2) how does the method used to generate demand scenarios (independent, spatially correlated, resampled from data, and reduced by clustering) affect stability and out-of-sample cost; and (3) can stabilization techniques (consensus over resampled solutions, a risk-averse CVaR objective, and a regularized version) reduce instability and the out-of-sample gap, and at what cost in expected total cost? Results are compared against a mean-demand deterministic plan, a simple heuristic, and a large-sample reference solution.

## Status
Work in progress. Stage B complete: repository set up.

## Repository structure
src/ tests/ notebooks/ results/ docs/ data/

## How to run
(to be filled in later)