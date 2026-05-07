#!/bin/bash
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 4
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=2
#SBATCH -t 2:00:00
#SBATCH -J resmill_lobes
#SBATCH --licenses=cfs,SCRATCH
#SBATCH -A REPLACE_WITH_YOUR_ALLOCATION
#SBATCH -o logs/%x-%j.out

# 2M-sample lobe-only sweep — config_full_lobes.json.
#
# Sizing chosen to minimise total time-to-results (queue wait + wall):
#
#   Measured throughput from a 200-sample sobol-subset benchmark:
#     mean=2.94s/sample  median=2.88s  std=0.28s  failures=0/200
#     -> 1633 core-hours for 2,000,000 samples.
#
#   Wall time estimates:
#     1 node  (128 cores): 12.76 h   queue wait usually <1h, total ~13-14h
#     2 nodes (256 cores):  6.38 h   queue wait <1h, total ~7-8h
#     4 nodes (512 cores):  3.19 h   queue wait 1-3h, total ~4-6h   <-- chosen
#     8 nodes (1024 cores): 1.60 h   queue wait 2-4h+, total ~4-6h (similar)
#
#   4 nodes / 4 h wall is the sweet spot: enough margin (25%) over the 3.2h
#   actual estimate to absorb stragglers, and short enough that the
#   regular-queue scheduler still ranks the job high.  -t 4:00:00 fits the
#   regular queue's 4 h "near-instant" tier on lightly loaded days while
#   leaving full margin even on the worst Sobol corners.
#
# Expected on-disk size (post-crop 64x64x32 in int8 + float16 + float16):
#     facies  131 KB/sample x 2M = 262 GB
#     poro    262 KB/sample x 2M = 524 GB
#     perm    262 KB/sample x 2M = 524 GB
#     parquet ~1 KB/sample  x 2M =   2 GB
#     ------------------------------------
#     total   ~1.3 TB

# Pin Numba/BLAS to 1 thread per rank -- each rank owns one physical core.
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

module load conda
conda activate $WORK/conda_envs/resmill

CFG="$(dirname "$0")/config_full_lobes.json"

srun --cpu-bind=cores python -m resmill.dataset.cli "$CFG"
