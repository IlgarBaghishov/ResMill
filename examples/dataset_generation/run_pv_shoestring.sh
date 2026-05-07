#!/bin/bash
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 4
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=2
#SBATCH -t 2:00:00
#SBATCH -J resmill_pv_shoestring
#SBATCH --licenses=cfs,SCRATCH
#SBATCH -A REPLACE_WITH_YOUR_ALLOCATION
#SBATCH -o logs/%x-%j.out

# 1M PV_SHOESTRING samples on config_full_pv_shoestring.json.
#
# Measured (200-sample sobol-subset, no contention):
#   mean=4.15 s/sample  median=3.25 s  std=3.11 s  failures=0/200
#   -> 1153 core-h = 9.0 node-h for 1M samples.
#   -> on 4 nodes (512 cores): 2.25 h wall, x1.2 margin = 2.7 h.
# 4 h walltime gives ~78% safety buffer; minimum-runtime job for premium
# queue's "near-instant" tier on a lightly loaded day.

export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

module load conda
conda activate $WORK/conda_envs/resmill

CFG="$(dirname "$0")/config_full_pv_shoestring.json"

srun --cpu-bind=cores python -m resmill.dataset.cli "$CFG"
