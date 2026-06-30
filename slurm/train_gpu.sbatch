#!/bin/bash
# Great Lakes GPU batch job: train GoalForge models.
#   sbatch slurm/train_gpu.sbatch
#SBATCH --job-name=goalforge-train
#SBATCH --account=your_account          # <-- your Great Lakes Slurm account (see `my_accounts`)
#SBATCH --partition=gpu                 # GPU partition (also: spgpu, gpu_mig40)
#SBATCH --gpus=1                        # request 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out   # %x=job name, %j=job id
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ctlang@umich.edu

set -euo pipefail
mkdir -p slurm/logs
source slurm/env_setup.sh
nvidia-smi || true

# Train the scoreline / player models (see docs/workflow.md). Placeholder entry point.
python -m goalforge.models.train --config configs/default.yaml
