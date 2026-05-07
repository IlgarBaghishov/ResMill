"""Generate deterministic 90/5/5 train/validation/test splits stratified by
``layer_type``.

Output: three parquet files at SPLIT_DIR (= ``--out``) with one row per
sample, columns:

- ``layer_type`` (str)
- ``shard_dir`` (str, RELATIVE to the HF repo root, e.g. ``lobe/shard_0042``)
- ``sample_idx`` (int, row within the shard's npy / parquet files)
- ``split`` (str, one of ``train`` / ``validation`` / ``test``)

Determinism: each layer-type's sample IDs are sorted globally (by shard
+ row), then shuffled with a fixed seed (42), then 90/5/5 split. Same
seed → identical assignment. Reviewers / users get a fixed benchmark.

Usage:
    python build_splits.py \\
        --root $SCRATCH/SiliciclasticReservoirs \\
        --out $SCRATCH/SiliciclasticReservoirs/splits
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# layer_type → on-disk relative-path-prefix (under the HF repo root)
PRESET_DIRS = {
    "lobe":                    "lobe",
    "channel:PV_SHOESTRING":   "channel_pv_shoestring",
    "channel:CB_LABYRINTH":    "channel_cb_labyrinth",
    "channel:CB_JIGSAW":       "channel_cb_jigsaw",
    "channel:SH_DISTAL":       "channel_sh_distal",
    "channel:SH_PROXIMAL":     "channel_sh_proximal",
    "channel:MEANDER_OXBOW":   "channel_meander_oxbow",
    "delta":                   "delta",
}


def index_preset(root: Path, preset_dir: str):
    """Yield (layer_type, shard_dir_rel, sample_idx) for every sample under preset_dir.

    ``shard_dir`` is renamed from ``combined_shard_NNNN`` to ``shard_NNNN``
    in the staged HF layout (we'll create symlinks to match).
    """
    abs_dir = root / preset_dir
    shards = sorted(abs_dir.glob("shard_*"))
    rows = []
    for shard in shards:
        n = int(np.load(shard / "facies.npy", mmap_mode="r").shape[0])
        rel_shard = f"{preset_dir}/{shard.name}"
        for i in range(n):
            rows.append((rel_shard, i))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="HF staged dir; should contain the 8 preset subdirs as symlinks")
    ap.add_argument("--out", required=True,
                    help="output dir for {train,validation,test}.parquet")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.90)
    ap.add_argument("--val-frac",   type=float, default=0.05)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-layer-type split, then concat
    rng_master = np.random.default_rng(args.seed)
    train_rows, val_rows, test_rows = [], [], []
    for layer_type, preset_dir in PRESET_DIRS.items():
        if not (root / preset_dir).is_dir():
            print(f"  skip {layer_type}: {root / preset_dir} not found")
            continue
        rows = index_preset(root, preset_dir)
        n = len(rows)
        # Per-layer-type seed derived deterministically from the master seed
        sub_seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(sub_seed)
        perm = rng.permutation(n)
        n_train = int(round(n * args.train_frac))
        n_val   = int(round(n * args.val_frac))
        n_test  = n - n_train - n_val   # remainder to test (avoids rounding gaps)
        idx_train = perm[:n_train]
        idx_val   = perm[n_train:n_train + n_val]
        idx_test  = perm[n_train + n_val:]
        for i in idx_train:
            sd, si = rows[i]
            train_rows.append((layer_type, sd, si))
        for i in idx_val:
            sd, si = rows[i]
            val_rows.append((layer_type, sd, si))
        for i in idx_test:
            sd, si = rows[i]
            test_rows.append((layer_type, sd, si))
        print(f"  {layer_type:<25} n={n:>7}  train={n_train:>7} val={n_val:>5} test={n_test:>5}")

    # Shuffle the concatenated splits once with master seed so layer types
    # are interleaved (better DataLoader behaviour with sequential reads
    # since shards from many presets are touched per-batch-of-shards).
    for split_name, rows in [("train", train_rows),
                             ("validation", val_rows),
                             ("test", test_rows)]:
        rng = np.random.default_rng(args.seed + hash(split_name) % 1000)
        order = rng.permutation(len(rows))
        rows_shuf = [rows[i] for i in order]
        layer_types = [r[0] for r in rows_shuf]
        shard_dirs  = [r[1] for r in rows_shuf]
        sample_idxs = [r[2] for r in rows_shuf]
        tab = pa.table({
            "layer_type": layer_types,
            "shard_dir":  shard_dirs,
            "sample_idx": sample_idxs,
        })
        out = out_dir / f"{split_name}.parquet"
        pq.write_table(tab, out)
        print(f"  wrote {out}  ({len(rows_shuf)} rows)")


if __name__ == "__main__":
    main()
