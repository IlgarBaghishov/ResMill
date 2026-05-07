#!/bin/bash
#SBATCH -q premium
#SBATCH -C cpu
#SBATCH -N 8
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=2
#SBATCH -t 3:00:00
#SBATCH -J resmill_cb_jigsaw
#SBATCH --licenses=cfs,SCRATCH
#SBATCH -A REPLACE_WITH_YOUR_ALLOCATION
#SBATCH -o logs/%x-%j.out

# 1.5M CB_JIGSAW (braided) samples on config_full_cb_jigsaw.json.
#
# Measured (100-sample subset, sequential, no contention):
#   mean=25.88 s/sample  std=4.43 s  failures=0/100
#   -> 10784 core-h = 84.3 node-h for 1.5M samples.
#   -> on 8 nodes (1024 cores): 10.5 h wall, x1.2 margin = 12.6 h.
# 14 h walltime fits with ~33% safety buffer.

export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

module load conda
conda activate $WORK/conda_envs/resmill

CFG="$(dirname "$0")/config_full_cb_jigsaw.json"

srun --cpu-bind=cores python -m resmill.dataset.cli "$CFG"
