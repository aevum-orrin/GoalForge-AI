#!/usr/bin/env bash
# Source this on a Great Lakes login or compute node before running GoalForge:
#   source slurm/env_setup.sh
# Makes conda available and activates the `goalforge` env. (Sourced -> no `set -e`.)

# 1) Make conda available. Adjust CONDA_ROOT to your Miniforge/Miniconda install.
export CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
else
  echo "[env_setup] conda not found at $CONDA_ROOT — install Miniforge first (see CLAUDE.md)." >&2
  return 1 2>/dev/null || exit 1
fi

# 2) Activate the project environment.
conda activate goalforge || {
  echo "[env_setup] could not activate 'goalforge' env (create it: conda env create -f environment.yml)." >&2
  return 1 2>/dev/null || exit 1
}

# 3) (GPU jobs) load a CUDA toolkit matching your PyTorch build, e.g.:
# module load cuda/12.3.0 cudnn

echo "[env_setup] goalforge ready: $(python --version 2>&1)"
