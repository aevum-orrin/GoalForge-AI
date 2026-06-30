# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this project is
GoalForge predicts a soccer match's **final score, goal scorers, and assisters** from the
two teams' **starting lineups**, using players' last-~3-years performance plus coach
history, fed into a probabilistic match-simulation engine. First target: FIFA World Cup;
designed to generalize to any match. Full design: [docs/workflow.md](docs/workflow.md).

## Conventions
- **Code and code comments: English.** Keep comments concise.
- **Chat / discussion with the user: mixed Chinese-English** is fine.
- **Workflow / design docs: mixed Chinese-English** is acceptable.
- Python package lives under `src/goalforge/` (src layout); install editable with
  `pip install -e .`.
- Large data and model artifacts stay out of git (see `.gitignore`); keep them under
  `data/` and `models/`, or on `/scratch` for big files.
- When adding dependencies, update **both** `environment.yml` and `requirements.txt`.
- **Isolated env only** — never `pip install` into the system Python (it's 3.6.8 here).
  Use the `goalforge` conda env (or a venv). Keep the footprint slim; add heavy/optional
  deps (torch, numpyro) only when actually needed.
- **Keep `README.md` and `docs/workflow.md` in sync** when code or structure changes.
- **Modeling honesty:** World-Cup-level data is small and squad turnover is high — treat
  in-sample scores as pipeline checks, prefer walk-forward / backtest evaluation, and
  never present leaky or overfit numbers as real predictive performance.

## Environment (UMich Great Lakes)
Developed on Great Lakes (`gl-login*.arc-ts.umich.edu`), a Slurm + Lmod HPC cluster.

### One-time setup
```bash
# Miniforge (conda) in your home dir — Great Lakes has no system conda
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
source $HOME/miniforge3/etc/profile.d/conda.sh

# create the project env
conda env create -f environment.yml
conda activate goalforge
pip install -e .
```

### Per-session
```bash
source slurm/env_setup.sh     # makes conda available + activates `goalforge`
```

## GPUs are on compute nodes, never the login node
The login node has no GPU. Request one through Slurm.

```bash
# interactive GPU session (1 GPU, 1 hour) — replace your_account
salloc --account=your_account --partition=gpu --gpus=1 --cpus-per-task=4 --mem=32G --time=01:00:00
# then, inside the allocation:
source slurm/env_setup.sh
nvidia-smi

# batch jobs
sbatch slurm/train_gpu.sbatch
sbatch slurm/predict.sbatch

# monitor / inspect
squeue -u $USER
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,ReqTRES%40
scancel <jobid>
```
Find your Slurm account(s): `my_accounts` (or `sacctmgr show assoc user=$USER format=account%30`).
Great Lakes GPU partitions include `gpu`, `spgpu` (A40), and `gpu_mig40`.

## Common commands
```bash
# tests / lint
pytest -q
ruff check src tests

# (planned) pipeline entry points — see docs/workflow.md
python -m goalforge.data.download       --config configs/default.yaml   # fetch & cache data
python -m goalforge.models.train        --config configs/default.yaml   # train models
python -m goalforge.prediction.predict_match --home <home.yaml> --away <away.yaml>
python -m goalforge.evaluation.backtest --config configs/default.yaml   # backtest
```

## Secrets
API keys (e.g. API-Football) go in a local `.env` (git-ignored). Never commit keys.

## Commits, attribution & pushing
- **Author:** `aevum-orrin` (`ctlang@umich.edu`); **co-author:** `Claude`. Every commit
  Claude helps with credits both via a trailer:
  ```
  Co-Authored-By: Claude <noreply@anthropic.com>
  ```
- Make **small, logical commits** so history stays reviewable; keep messages concise and
  imperative (state *why* when it isn't obvious).
- **Commit after a change, but do NOT `git push` unless the user explicitly asks.**
  (The initial scaffold push was explicitly requested.)

## Effort & token-limit windows
- Run at the **Effort level the user selected** (currently **Max**). Do **not** silently
  downgrade Effort to save tokens.
- If a long job exhausts the rolling **5-hour token-limit window**, **pause** the job
  (stop mid-task). When the window refreshes and budget returns, **resume** at the same
  Effort — don't restart from scratch and don't lower quality.
- Only reduce Effort when the user **explicitly** asks.

## Notes for Claude
- This repo is in the **design phase** — don't write large code dumps unprompted.
- GPU code must be runnable through the `slurm/` templates, not assumed to run on the
  login node.
- Keep `data/` and `models/` contents out of git.
