"""Run the dataset generator on an interactive node using multiprocessing.

Thin parallel driver around the same ``build_jobs`` + ``generate_sample``
+ ``ShardWriter`` code path that the SLURM CLI uses
(``resmill.dataset.cli``); the only difference is we shard the joblist
across a ``multiprocessing.Pool`` instead of across MPI ranks, so it
runs without ``srun`` / ``salloc``.

Use this for QA / small-N test datasets, and for timing-extrapolation
runs that estimate the wall clock for a full production sweep. For an
actual 2M / 10M-sample production run prefer ``cli.py`` under SLURM
(one rank per core, shard namespaces don't collide).

Usage:
    python run_dataset.py CONFIG.json [--workers N]
                                      [--limit N]
                                      [--output-dir PATH]
                                      [--cores-per-node N]
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from resmill.dataset.generate import generate_sample
from resmill.dataset.io import ShardWriter
from resmill.dataset.sampling import build_jobs


def _gen(args):
    """Pool worker — single sample. Returns (idx, result, dt, err)."""
    idx, job, grid_cfg = args
    t0 = time.perf_counter()
    try:
        out = generate_sample(job, grid_cfg)
    except Exception as exc:
        return idx, None, time.perf_counter() - t0, repr(exc)[:300]
    return idx, out, time.perf_counter() - t0, None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--workers", type=int, default=32,
                   help="multiprocessing pool size (default 32)")
    p.add_argument("--limit", type=int, default=None,
                   help="take only the first N jobs from the joblist (the "
                        "joblist is already shuffled by build_jobs, so this "
                        "yields a representative random subset)")
    p.add_argument("--output-dir", default=None,
                   help="override the config's output_dir (useful so a test "
                        "run doesn't collide with a production output dir)")
    p.add_argument("--cores-per-node", type=int, default=128,
                   help="cores/node for the node-hour estimate "
                        "(Perlmutter CPU=128, default)")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    jobs = build_jobs(cfg["layers"], cfg["seed"])
    total_jobs = len(jobs)
    n_to_run = total_jobs if args.limit is None else min(int(args.limit), total_jobs)
    print(f"joblist size: {total_jobs}  |  running: {n_to_run}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(str(output_dir), rank=0, shard_size=cfg["shard_size"])
    print(f"output: {output_dir}")

    failures_path = output_dir / "failures_r0000.jsonl"
    n_failed = 0
    times: list[float] = []

    tasks = [(i, jobs[i], cfg["grid"]) for i in range(n_to_run)]
    print(f"running {len(tasks)} samples on {args.workers} workers...")
    t_start = time.perf_counter()
    with Pool(args.workers) as pool:
        for n, (idx, result, dt, err) in enumerate(
            pool.imap_unordered(_gen, tasks), 1
        ):
            if err is not None:
                n_failed += 1
                with open(failures_path, "a") as f:
                    f.write(json.dumps({
                        "global_index": idx,
                        "layer_type": jobs[idx]["layer_type"],
                        "seed": int(jobs[idx]["seed"]),
                        "params": jobs[idx]["params"],
                        "error": err,
                    }) + "\n")
            else:
                facies, poro, perm, facies_alluvsim, meta = result
                writer.add(facies, poro, perm, facies_alluvsim, meta)
                times.append(dt)
            if n % 16 == 0 or n == len(tasks):
                elapsed = time.perf_counter() - t_start
                eta = elapsed / n * (len(tasks) - n) if n < len(tasks) else 0
                print(f"  [{n:4d}/{len(tasks)}]  elapsed={elapsed:6.1f}s  "
                      f"eta={eta:6.1f}s  failed={n_failed}", flush=True)
    writer.close()

    wall = time.perf_counter() - t_start
    print(f"\n=== Done ===")
    print(f"  wall:    {wall:.1f}s   ({wall / 60:.2f} min)")
    print(f"  failed:  {n_failed} / {len(tasks)}")
    if times:
        a = np.array(times)
        print(f"  per-job: mean={a.mean():.2f}s  median={np.median(a):.2f}s  "
              f"min={a.min():.2f}s  max={a.max():.2f}s  std={a.std():.2f}s")
        ratio = a.max() / max(a.min(), 0.1)
        if ratio > 3:
            print(f"  high variability: max/min = {ratio:.1f}x")

        # Node-hour extrapolation: each sample's CPU cost is ~mean(per-job),
        # and a node has args.cores_per_node parallel workers.
        per_sample = float(a.mean())
        for target in (n_to_run, total_jobs):
            core_h = per_sample * target / 3600.0
            node_h = core_h / args.cores_per_node
            label = "this run " if target == n_to_run else "full joblist"
            print(f"  extrapolation  ({label}, target={target}):  "
                  f"{core_h:.1f} core-h  =  {node_h:.2f} node-h "
                  f"@ {args.cores_per_node} cores/node")
    if n_failed:
        print(f"  failure log: {failures_path}")


if __name__ == "__main__":
    main()
