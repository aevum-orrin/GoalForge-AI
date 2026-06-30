"""GoalForge AI — soccer match outcome and goal/assist prediction.

Given the starting lineups of two teams, GoalForge predicts the final score, the
scorer of each goal, and the assisting player, by combining player- and coach-level
historical performance with a probabilistic match-simulation engine.

Subpackages
-----------
data        Ingestion and loading of player / team / coach / match data.
features    Feature engineering (player form, action-value ratings, coach effects).
models      Team-level scoreline models and player-level goal/assist models.
simulation  Monte-Carlo match engine producing scorelines and goal/assist events.
prediction  End-to-end agent: two lineups -> structured prediction.
evaluation  Backtesting and metrics (calibration, RPS, log-loss, top-k accuracy).
utils       Shared helpers (config, IO, logging).
"""

__version__ = "0.0.1"
