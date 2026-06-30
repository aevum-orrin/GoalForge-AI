# Slurm job templates (UMich Great Lakes)

GPUs live on compute nodes, never the login node — submit work through Slurm.
Replace `your_account` with your allocation (find it with `my_accounts`).

| Script | Purpose | Partition |
|--------|---------|-----------|
| `env_setup.sh` | `source` to activate the `goalforge` conda env (+ optional CUDA module) | — |
| `train_gpu.sbatch` | Train scoreline / player models on 1 GPU | `gpu` |
| `predict.sbatch` | Predict one match (CPU sim by default) | `standard` |

```bash
# submit a batch job
sbatch slurm/train_gpu.sbatch

# interactive GPU session
salloc --account=your_account --partition=gpu --gpus=1 --cpus-per-task=4 --mem=32G --time=01:00:00

# monitor
squeue -u $USER
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS
```

Logs land in `slurm/logs/` (git-ignored). See [../CLAUDE.md](../CLAUDE.md) for full setup.
