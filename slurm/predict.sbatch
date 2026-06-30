#!/bin/bash
# Great Lakes batch job: predict a single match from two starting lineups.
#   sbatch slurm/predict.sbatch
#SBATCH --job-name=goalforge-predict
#SBATCH --account=your_account          # <-- your Great Lakes Slurm account
#SBATCH --partition=standard            # CPU is enough for modest sims; use `gpu` for large vectorized sims
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
mkdir -p slurm/logs
source slurm/env_setup.sh

# Placeholder entry point (see docs/workflow.md for the lineup file format).
python -m goalforge.prediction.predict_match \
  --home configs/lineups/home.yaml \
  --away configs/lineups/away.yaml \
  --config configs/default.yaml
