# ResMill dataset generation on an HPC CPU partition

Generate a large 3D reservoir dataset by sweeping the parameter space of
every ResMill layer type (Lobe, Gaussian, Meandering, Braided, Delta).
Designed for 1–10M samples on HPC CPU nodes, with output staged
for direct upload to HuggingFace and consumption by GenFlows.

## What it produces

Each sample is a single-layer 3D reservoir at `(64, 64, 32)` resolution.
Outputs are grouped into shards under `output_dir/`:

```
output_dir/
  shard_r0000_s000000/
    facies.npy     (N, 64, 64, 32)  int8      binary 0/1
    poro.npy       (N, 64, 64, 32)  float16   porosity (unitless 0..1)
    perm.npy       (N, 64, 64, 32)  float16   permeability (mD)
    params.parquet                           N rows, one per sample
  shard_r0000_s000001/
  ...
  shard_r0001_s000000/
  ...
```

`params.parquet` columns:
- `layer_type` — one of `lobe`, `gaussian`, `meandering`, `braided`, `delta`
- `seed` — per-sample RNG seed
- `caption` — one-sentence natural-language description of the reservoir
- all sampled physics parameters (`poro_ave`, `perm_ave`, `ntg`, etc.);
  columns that don't apply to a given layer type are null in that row.

Each rank owns its own shard namespace, so there is no coordination
between parallel workers.

## Quick start

### 1. Create the env (one-time)
```bash
module load conda
conda create -p $WORK/conda_envs/resmill python=3.12 -y
conda activate $WORK/conda_envs/resmill
pip install -e $WORK/codes/ResMill
pip install -e '$WORK/codes/ResMill[dataset]'
```

### 2. Serial smoke test (10 samples, ~1 minute)
```bash
conda activate $WORK/conda_envs/resmill
python -m resmill.dataset.cli examples/dataset_generation/config_demo.json
```
Inspect one shard at
`$SCRATCH/resmill_dataset_demo/shard_r0000_s000000/`.

### 3. 128-way parallel test on an interactive node (1000 samples)
```bash
salloc -N 1 -C cpu -q interactive --ntasks-per-node 128 -t 00:30:00 -A REPLACE_WITH_YOUR_ALLOCATION
srun -n 128 --cpu-bind=cores \
     python -m resmill.dataset.cli examples/dataset_generation/config_parallel_test.json
```

### 4. Full run (10M samples, ~19 h wall on 4 nodes — default)
```bash
sbatch examples/dataset_generation/run.sh
```

The default `config_full.json` targets 10M samples with a non-uniform
layer mix:

| Layer | Count |
|---|---|
| Lobe | 2,500,000 |
| Meandering | 2,500,000 |
| Braided | 2,500,000 |
| Delta | 1,500,000 |
| Gaussian | 1,000,000 |
| **Total** | **10,000,000** |

On-disk size at 64×64×32: ~6.4 TB (facies int8 + poro float16 + perm
float16). Make sure `$SCRATCH` has enough quota.

For a smaller 1M run: reduce each `count` by 10× in `config_full.json`
and change `run.sh` to `-N 2 -t 05:00:00`. For different layer mixes,
just edit the `count` field per layer — the rank-stripe loop and shuffle
handle arbitrary proportions.

### Measured performance (10K-sample validation on 1 CPU node, 128 ranks)

| Metric | Value |
|---|---|
| Aggregate throughput | **~46 samples/sec/node** |
| Per-rank max RSS | 434-618 MB |
| Aggregate RAM / node | ~60-80 GB / 512 GB |
| Failure rate | 0.04% (4 out of 10,000) |
| Per-rank wall spread | ~1.7x (fast:slow) at 78 samples/rank |

The single-core benchmark (0.78 s/sample in isolation) does **not**
extrapolate to full-node throughput: under 128 concurrent Numba ranks,
memory-bandwidth and L3-cache contention slow each rank to ~2.8
core-seconds/sample. This is inherent to compute-bound Numba workloads
on EPYC; no knobs to turn further.

## Config schema

One JSON drives the whole run. See `config_demo.json` for a complete
example. Top-level keys:

- `output_dir`  — absolute path where shards are written
- `seed`        — master RNG seed (full pipeline is deterministic modulo
                   DeltaLayer's internal `default_rng`, which ignores it)
- `shard_size`  — samples per shard (`10000` is a good fit for HF)
- `grid`        — fixed grid geometry applied to every sample
- `layers`      — one section per layer type

Per-layer section:
- `count`    — number of samples to draw
- `sampling` — `sobol` / `lhs` / `grid` / `uniform`
- `params`   — per-parameter spec

Per-parameter spec:
- `{"range": [lo, hi]}`                    continuous float uniform
- `{"range": [lo, hi], "scale": "log"}`    log-uniform (for perms)
- `{"range": [lo, hi], "type": "int"}`     integer in [lo, hi] inclusive
- `{"choices": [...]}`                      categorical
- `{"value": v}`                            fixed scalar
- `{"levels": N}`                           (grid sampling only) number of
                                            grid steps on that axis;
                                            `count` must equal the product
                                            of all `levels` / `choices`.

## Consuming the dataset in GenFlows

The on-disk layout mirrors the mmap-based loader pattern already used in
`genflows/utils/data_lobes.py`. Minimal loader:

```python
import numpy as np, pyarrow.parquet as pq, torch
from pathlib import Path
from torch.utils.data import Dataset

class ResMillDataset(Dataset):
    def __init__(self, root):
        self.shards = sorted(Path(root).glob("shard_r*_s*"))
        self.lens = [np.load(s / "facies.npy", mmap_mode="r").shape[0]
                     for s in self.shards]
        self.cum = np.cumsum([0] + self.lens)

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, idx):
        s = int(np.searchsorted(self.cum[1:], idx, side="right"))
        local = idx - self.cum[s]
        shard = self.shards[s]
        facies = np.load(shard / "facies.npy", mmap_mode="r")[local]
        meta   = pq.read_table(shard / "params.parquet").slice(local, 1).to_pylist()[0]
        x = torch.from_numpy(facies.astype(np.float32)).unsqueeze(0) * 2 - 1
        return x, meta
```

## Uploading to HuggingFace

```bash
huggingface-cli upload <user>/resmill-reservoirs \
    $SCRATCH/resmill_dataset --repo-type dataset
```

`params.parquet` is natively readable by `datasets.load_dataset("parquet", ...)`.

## Sample failures

A small fraction (~0.04%) of parameter combinations trigger pre-existing
bugs in the underlying layer code — typically `BraidedChannelLayer` /
`MeanderingChannelLayer` hitting degenerate channel geometry that breaks
a scipy `UnivariateSpline` fit in `resmill/layers/_fluvial.py`. The CLI
catches these, skips the sample, and records the full context (layer
type, seed, parameter values, exception) to
`{output_dir}/failures_r{rank:04d}.jsonl`. Failures are not fatal; 1M
runs should yield ~999,600 good samples.

## Tuning notes

- **Hyperthreading is disabled** (`--cpus-per-task=2`, 128 ranks/node).
  Compute-bound Numba regresses under SMT, so we use 1 physical core per
  rank.
- **Load balance** is handled statically: the job list is pre-shuffled
  with a fixed seed before rank-striping, so the five layer types mix
  across ranks. At 1M samples / 256 ranks, end-tail imbalance is ~1–2%;
  at small smoke-test scales (<1000 samples) imbalance can be ~2× but
  doesn't matter since total wallclock is minutes.
- **DeltaLayer seeds**: Delta uses its own `np.random.default_rng()`
  internally, which ignores `np.random.seed()`. Delta samples remain
  stochastic and diverse, but cannot be exactly regenerated from the
  `seed` column alone. Fixing this requires a one-line edit to
  `resmill/layers/delta.py`.
- **Library bugs with inf/NaN in perm/poro**: `resmill/layers/lobe.py`
  and `resmill/layers/gaussian.py` occasionally produce `inf * 0 = NaN`
  on masked cells. The generator sanitizes these in
  `resmill/dataset/generate.py::generate_sample` (via `nan_to_num` +
  float16 range clip) so on-disk arrays are always finite.
