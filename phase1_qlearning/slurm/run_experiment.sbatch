#!/bin/bash
#
# Full-scale experiment runner: one SLURM array task = one batch of sessions
# for one parameter cell of a (possibly swept) config.
#
# Usage (after filling in the TODOs below):
#   CONFIG=configs/sweep_sigma_u.yaml sbatch --array=0-$((N_CELLS*TASKS_PER_CELL-1)) \
#       phase1_qlearning/slurm/run_experiment.sbatch
#
# Array-index layout (row-major over the config's `sweep:` cartesian product):
#   cell_index = SLURM_ARRAY_TASK_ID / TASKS_PER_CELL
#   task_in_cell = SLURM_ARRAY_TASK_ID % TASKS_PER_CELL
# With the paper's Nsim = 1000 sessions per cell and run.batch = 10 sessions
# per task, TASKS_PER_CELL = 100. N_CELLS = product of the sweep list lengths
# (e.g. 11 for sweep_sigma_u.yaml, 16 for sweep_I.yaml / sweep_rho.yaml).
#
# Backend: BACKEND=numpy (default) runs the pure-NumPy reference path, which
# works on any Python with the requirements.txt wheels. BACKEND=numba runs
# the JIT-compiled hot loop (README "Numba backend" section) -- ~100x faster
# per session (benchmarked ~1-8M periods/sec/session vs ~12-16k/s for numpy,
# depending on run.batch; see README), which matters a lot at the paper's
# 20M-50B-period worst case. Numba needs a Python it supports (check with
# `module avail python`; as of this writing Numba does not yet support
# Python 3.14). If BACKEND=numba, set --cpus-per-task to match run.batch
# (each session runs on its own thread via numba.prange; oversubscribing
# threads past the allocated cpus tanks per-session throughput -- see the
# README benchmark table for the batch-vs-per-session-speed tradeoff).
#
#SBATCH --job-name=ai-collusion
#SBATCH --account=def-cbravo         # Alliance allocation (no --partition needed on Alliance clusters)
#SBATCH --time=2-00:00:00            # 2 days; raise if convergence needs more (max 7-00:00:00 on Graham)
#SBATCH --cpus-per-task=1            # bump to match run.batch (10) if BACKEND=numba
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

CONFIG="${CONFIG:?set CONFIG=<path to yaml config>}"
TASKS_PER_CELL="${TASKS_PER_CELL:-100}"   # Nsim / run.batch (1000 / 10)
OUTDIR="${OUTDIR:-results/$(basename "${CONFIG%.yaml}")}"
BACKEND="${BACKEND:-numpy}"               # numpy (default, always works) or numba (needs numba)

# Alliance cluster environment (venv built once on the login node per
# CLUSTER_READINESS.md step 3; submit this job from the repo root so the
# relative .venv path resolves). numba is installed in the same venv.
module load StdEnv/2023 python/3.11.5
source .venv/bin/activate

CELL=$(( SLURM_ARRAY_TASK_ID / TASKS_PER_CELL ))
TASK=$(( SLURM_ARRAY_TASK_ID % TASKS_PER_CELL ))
SEED=$(( 1000000 + CELL * 10000 + TASK ))   # unique, reproducible per task

mkdir -p "$OUTDIR" logs
srun python -m phase1_qlearning.run_session \
    --config "$CONFIG" \
    --cell-index "$CELL" \
    --seed "$SEED" \
    --backend "$BACKEND" \
    --quiet \
    --out "$OUTDIR/cell${CELL}_task${TASK}.npz"

# Afterwards, on a login node:
#   python -m phase1_qlearning.classify_mechanism "$OUTDIR"/*.npz
#   python -m phase1_qlearning.aggregate "$OUTDIR/*.npz" --csv "$OUTDIR/sessions.csv"
#   python -m phase1_qlearning.plots "$OUTDIR/*.npz" --out-dir "$OUTDIR/figures"
