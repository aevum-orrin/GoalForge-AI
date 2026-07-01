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
> pipeline — lives in [docs/workflow.md](docs/workflow.md). A working **Phase-0 pipeline**
> (Dixon–Coles scoreline + Monte-Carlo scorer/assist allocation) is implemented and runs on
> both synthetic data and real StatsBomb World Cup data — see Quickstart.

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
app/             Streamlit web UI
public/          web frontend (static HTML/CSS/JS; served by FastAPI locally & Vercel)
api/             self-contained Vercel serverless function (numpy + fastapi)
scripts/         CLI entry points (run_pipeline / train / run_worldcup)
slurm/           Great Lakes (Slurm) job templates
src/goalforge/   Main Python package
  data/          ingestion & loaders (synthetic, StatsBomb, martj42)
  features/      feature engineering (player form, ratings, coach effects)
  models/        scoreline (Dixon-Coles, hierarchical) & player models
  simulation/    Monte-Carlo match engine
  prediction/    end-to-end agent + checkpoint + likely-XI
  evaluation/    temporal split, baselines, metrics, backtest
  api/           FastAPI backend
  utils/         shared helpers
tests/           test suite
```

## Quickstart (UMich Great Lakes)
See [CLAUDE.md](CLAUDE.md) for the full command reference. In short:
```bash
# one-time: env = anaconda module (scientific stack) + project .venv, then install goalforge
module load python3.11-anaconda/2024.02
python -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -e . && pip install statsbombpy      # statsbombpy = real-data path
# (alternative, self-contained: conda env create -f environment.yml && conda activate goalforge && pip install -e .)

# per-session
source slurm/env_setup.sh

# --- run Phase 0 ---
python scripts/run_pipeline.py                   # synthetic, offline: fit -> backtest -> predict
pytest -q                                        # test suite (11 tests)
python scripts/run_worldcup.py Argentina France  # real StatsBomb WC2022 data

# train a checkpoint (team model on martj42 internationals + player rates on StatsBomb)
python scripts/train.py

# --- web app ---
python -m streamlit run app/streamlit_app.py                    # Streamlit UI
python -m uvicorn goalforge.api.app:app --host 127.0.0.1 --port 8000   # FastAPI + web/ frontend

# GPU work runs through Slurm — never on the login node
sbatch slurm/train_gpu.sbatch
```

**What Phase 0 does:** generate/load matches → fit a Dixon–Coles scoreline model → estimate
per-player scoring/assist rates (empirical-Bayes shrinkage) → Monte-Carlo simulate the match
→ output scoreline probabilities, per-player anytime-scorer and assist probabilities, with a
walk-forward RPS backtest. Real data flows through [statsbomb.py](src/goalforge/data/statsbomb.py);
synthetic data through [synthetic.py](src/goalforge/data/synthetic.py).

## Hardware
Development happens on Great Lakes. GPUs are requested via Slurm (see [slurm/](slurm/)) and
are used where they genuinely help — large vectorized Monte-Carlo simulation, neural /
player-embedding models, and (optionally) GPU Bayesian inference. Classical statistical
models (Dixon–Coles, Poisson) run fine on CPU.

## License
TBD.
