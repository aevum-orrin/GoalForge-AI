#!/bin/bash
# Great Lakes GPU job: leave-one-tournament-out eval of the lineup-aware neural Dixon-Coles
# (Phase 3). Owned NEDA L40S nodes = account nmasoud_owned1 + partition spgpu2 + qos arph.
#   sbatch slurm/train_neural.sbatch
#SBATCH --job-name=goalforge-neural
#SBATCH --account=nmasoud_owned1
#SBATCH --partition=spgpu2               # NEDA-owned L40S nodes (gl1700-1723)
#SBATCH --qos=arph                       # QOS granting the owned nodes
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/%x-%j.out    # %x=job name, %j=job id
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ctlang@umich.edu

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/ctlang/GoalForge_AI}"
mkdir -p slurm/logs

# Big/regenerable caches (the StatsBomb lineup dataset) live on NEDA scratch.
export GOALFORGE_DATA_DIR="${GOALFORGE_DATA_DIR:-/scratch/nmasoud_owned_root/nmasoud_owned1/ctlang/gf_cache/data}"

# torch env (built separately: torch 2.5.1+cu121; goalforge installed editable, --no-deps).
# The pip CUDA wheel bundles its own CUDA runtime, so no `module load cuda` is needed.
GPUPY=/home/ctlang/envs/goalforge-gpu/bin/python

nvidia-smi || true
"$GPUPY" -c "import torch; print('cuda available:', torch.cuda.is_available(),
             '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
"$GPUPY" scripts/train_neural.py --device cuda
