#!/bin/bash
#SBATCH -q premium
#SBATCH -C cpu
#SBATCH -N 8
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=2
#SBATCH -t 4:00:00
#SBATCH -J resmill_meander_oxbow
#SBATCH --licenses=cfs,SCRATCH
#SBATCH -A m1883
#SBATCH -o logs/%x-%j.out

# 1M MEANDER_OXBOW samples on config_full_meander_oxbow.json.
# Single-channel-per-level with neck-cutoff oxbow plugs.
#
# Measured (100-sample subset, sequential, no contention):
#   mean=31.08 s/sample  std=5.55 s  failures=0/100
#   -> 8632 core-h = 67.4 node-h for 1M samples.
#   -> on 8 nodes (1024 cores): 8.4 h wall, x1.2 margin = 10.1 h.
# 12 h walltime fits with ~42% safety buffer.

export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

module load conda
conda activate /global/cfs/cdirs/m1883/ilgar/conda_envs/resmill

CFG="/global/cfs/cdirs/m1883/ilgar/codes/ResMill/examples/dataset_generation/config_full_meander_oxbow.json"

srun --cpu-bind=cores python -m resmill.dataset.cli "$CFG"
