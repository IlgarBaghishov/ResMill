#!/bin/bash
#SBATCH -q premium
#SBATCH -C cpu
#SBATCH -N 8
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=2
#SBATCH -t 3:00:00
#SBATCH -J resmill_sh_distal
#SBATCH --licenses=cfs,SCRATCH
#SBATCH -A m1883
#SBATCH -o logs/%x-%j.out

# 1M SH_DISTAL samples on config_full_sh_distal.json.
#
# Measured (100-sample subset, sequential, no contention):
#   mean=34.53 s/sample  std=4.79 s  failures=0/100
#   -> 9591 core-h = 74.9 node-h for 1M samples.
#   -> on 8 nodes (1024 cores): 9.4 h wall, x1.2 margin = 11.2 h.
# 12 h walltime fits with ~28% safety buffer.

export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

module load conda
conda activate /global/cfs/cdirs/m1883/ilgar/conda_envs/resmill

CFG="/global/cfs/cdirs/m1883/ilgar/codes/ResMill/examples/dataset_generation/config_full_sh_distal.json"

srun --cpu-bind=cores python -m resmill.dataset.cli "$CFG"
