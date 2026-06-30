# GoalForge — Prediction Workflow & Design

> **Status: in progress.** Being written from a research pass over soccer data sources,
> statistical scoreline models, ML/DL approaches, and match-simulation methods. The full
> version (model trade-offs + end-to-end pipeline) is assembled in the next commit.

## Outline
1. Problem & prediction targets — final score, goal scorers, assisters from two XIs
2. Data sources — player (≈3 yr) / coach / lineup / training labels
3. Feature engineering — recency-weighted form, per-90 rates, action-value (VAEP/xT), coach effects
4. Team-level scoreline model (backbone) — Dixon-Coles / bivariate Poisson / Bayesian hierarchical / ML
5. Player-level goal & assist allocation — scoring weights, multinomial / Dirichlet
6. Monte-Carlo match simulation — team λ → goals → scorers → assists → aggregate
7. Training, evaluation & backtesting — RPS, log-loss, calibration, top-k
8. Where the GPU is used
9. Candidate models — mature vs. exploratory (pros / cons)
10. Roadmap
