#!/bin/bash
#SBATCH -q premium
#SBATCH -C cpu
#SBATCH -N 4
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=2
#SBATCH -t 24:00:00
#SBATCH -J resmill_dataset
#SBATCH --licenses=cfs,SCRATCH
#SBATCH -A m1883
#SBATCH -o logs/%x-%j.out

# 4 Perlmutter CPU nodes x 128 physical cores = 512 ranks.
# Target: 10M samples (2.5M lobe + 2.5M meandering + 2.5M braided +
#         1.5M delta + 1.0M gaussian) per config_full.json.
# Measured throughput at full-node load (10K-sample interactive test):
#   ~46 samples/sec per 128-core node.
# Projections (with a comfortable margin):
#   10M samples on 4 nodes ~ 18.9 h wall  (-t 24:00:00 covers 1.3x margin)
#   10M samples on 2 nodes ~ 37.8 h wall  (fits 48 h budget)
#    1M samples on 2 nodes ~  3.8 h wall
# Expected sample-failure rate is ~0.04% (logged to failures_r*.jsonl).
# Expected on-disk size: ~6.4 TB at 64x64x32 (int8 facies + float16 poro + float16 perm).

# Pin Numba/BLAS to 1 thread per rank — each rank owns one physical core.
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

module load conda
conda activate /global/cfs/cdirs/m1883/ilgar/conda_envs/resmill

CFG="$(dirname "$0")/config_full.json"

srun --cpu-bind=cores python -m resmill.dataset.cli "$CFG"
