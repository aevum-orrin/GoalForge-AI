# GoalForge AI ⚽

GoalForge is a research project that predicts the outcome of a soccer match from the
**starting lineups of the two teams**. Given the two XI's, it aims to predict:

- the **final score**,
- **who scores** each goal, and
- **who assists** each goal (when there is one).

The approach combines each player's recent performance (≈ last 3 seasons), the two
coaches' track records, and opponent strength into a probabilistic **match-simulation
engine**: a team-level model estimates how many goals each side is expected to score, and
a player-level model distributes those goals (and assists) across the lineup. Thousands of
Monte-Carlo simulations of the match then yield score, scorer, and assister probabilities.

The first target is the **FIFA World Cup**; the design generalizes to any match (club or
international) for which both starting lineups are available.

> **Status:** early scaffold. The full design — data sources, models, and the prediction
> pipeline — lives in [docs/workflow.md](docs/workflow.md). No production code yet.

## Why an "agent"?
The end goal is an automated agent: hand it two lineups, and it fetches the required
historical data, builds features, runs the simulation, and returns a structured prediction
— no manual steps in between.

## Repository layout
```
configs/         YAML run/experiment configs
data/            Local data cache (raw/interim/processed/external) — contents git-ignored
docs/            Design docs; start with docs/workflow.md
models/          Saved model artifacts — contents git-ignored
notebooks/       Exploratory analysis
reports/         Generated figures and prediction outputs
scripts/         Thin CLI entry points (download / train / predict / backtest)
slurm/           Great Lakes (Slurm) job templates
src/goalforge/   Main Python package
  data/          ingestion & loaders
  features/      feature engineering (player form, ratings, coach effects)
  models/        scoreline & player goal/assist models
  simulation/    Monte-Carlo match engine
  prediction/    end-to-end lineup -> prediction agent
  evaluation/    backtesting & metrics
  utils/         shared helpers
tests/           test suite
```

## Quickstart (UMich Great Lakes)
See [CLAUDE.md](CLAUDE.md) for the full command reference. In short:
```bash
# 1) one-time: install Miniforge + create the env (see CLAUDE.md)
conda env create -f environment.yml
conda activate goalforge
pip install -e .            # install the goalforge package (editable)

# 2) GPU work runs through Slurm — never on the login node
sbatch slurm/train_gpu.sbatch
```

## Hardware
Development happens on Great Lakes. GPUs are requested via Slurm (see [slurm/](slurm/)) and
are used where they genuinely help — large vectorized Monte-Carlo simulation, neural /
player-embedding models, and (optionally) GPU Bayesian inference. Classical statistical
models (Dixon–Coles, Poisson) run fine on CPU.

## License
TBD.
