#!/bin/bash
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 8
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=2
#SBATCH -t 2:00:00
#SBATCH -J resmill_cb_labyrinth
#SBATCH --licenses=cfs,SCRATCH
#SBATCH -A REPLACE_WITH_YOUR_ALLOCATION
#SBATCH -o logs/%x-%j.out

# 1M CB_LABYRINTH samples on config_full_cb_labyrinth.json.
#
# Measured (100-sample subset, sequential, no contention):
#   mean=22.54 s/sample  std=8.91 s  failures=0/100
#   -> 6261 core-h = 48.9 node-h for 1M samples.
#   -> on 8 nodes (1024 cores): 6.1 h wall, x1.2 margin = 7.3 h.
# 8 h walltime fits with ~30% safety buffer.

export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

module load conda
conda activate $WORK/conda_envs/resmill

CFG="$(dirname "$0")/config_full_cb_labyrinth.json"

srun --cpu-bind=cores python -m resmill.dataset.cli "$CFG"
