#!/bin/bash
#SBATCH -q premium
#SBATCH -C cpu
#SBATCH -N 16
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=2
#SBATCH -t 10:00:00
#SBATCH -J resmill_delta
#SBATCH --licenses=cfs,SCRATCH
#SBATCH -A REPLACE_WITH_YOUR_ALLOCATION
#SBATCH -o logs/%x-%j.out

# 1.5M delta samples on config_full_delta.json.
# Delta is the SLOWEST layer in the dataset — runs n_generations (4-12)
# independent fluvial simulations per sample and merges them.
#
# Measured (100-sample subset, sequential, no contention):
#   mean=46.85 s/sample  std=18.55 s  failures=0/100
#   -> 19519 core-h = 152.5 node-h for 1.5M samples.
#   -> on 12 nodes (1536 cores): 12.7 h wall, x1.2 margin = 15.3 h.
# 16 h walltime fits with ~26% safety buffer.
# (8-node alternative would need ~24 h wall — at the queue's max — so
# 12 nodes is the safer choice.)

export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

module load conda
conda activate $WORK/conda_envs/resmill

CFG="$(dirname "$0")/config_full_delta.json"

srun --cpu-bind=cores python -m resmill.dataset.cli "$CFG"
