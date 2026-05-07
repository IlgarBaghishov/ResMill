"""Plot facies / porosity / permeability slices for every sample in a
dataset shard directory.

Reads shards produced by ``resmill.dataset.cli`` / ``run_dataset.py``
(each shard is a directory with ``facies.npy`` + ``poro.npy`` +
``perm.npy`` + ``params.parquet``) and writes one PNG per sample with
4 slices along each of the 3 axes (X, Y, Z), the caption from the
parquet, and every parameter column dumped as a monospace footer.

Reuses ``resmill.plotting.plot_slices`` for the slice grid; the
function auto-detects binary (facies) vs continuous (poro/perm) mode
from the dtype. Permeability is plotted on a log10 scale because mD
spans several decades.

Output subdirectories (created if missing, default under <data_dir>/):
    facies → pictures/
    poro   → poro_pictures/
    perm   → perm_pictures/

Usage:
    python plot_dataset.py SHARD_DIR                      # all 3 props
    python plot_dataset.py SHARD_DIR --prop facies        # facies only
    python plot_dataset.py SHARD_DIR --prop poro --out X  # poro to X/
    python plot_dataset.py SHARD_DIR --workers 32 --limit 50
"""
from __future__ import annotations

import argparse
import textwrap
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from resmill.plotting import plot_slices


_PROP_DIR = {
    "facies":          "facies_binary_pictures",
    "poro":            "poro_pictures",
    "perm":            "perm_pictures",
    "facies_alluvsim": "facies_alluvsim_pictures",
}
_PROP_TITLE = {
    "facies":          None,
    "poro":            "Porosity",
    "perm":            "log10(Permeability, mD)",
    "facies_alluvsim": None,
}


def _format_param_block(meta: dict) -> str:
    """Multi-line monospace dump of every meta key (skipping noise / nulls)."""
    skip = {"layer_type", "seed", "caption", "preset"}
    items = [(k, v) for k, v in sorted(meta.items())
             if k not in skip and v is not None]
    lines, line, line_len = [], [], 0
    for k, v in items:
        if isinstance(v, float):
            cell = f"{k}={v:.4g}"
        elif isinstance(v, bool):
            cell = f"{k}={v}"
        else:
            cell = f"{k}={v}"
        if line_len + len(cell) + 2 > 85 and line:
            lines.append("  ".join(line))
            line, line_len = [], 0
        line.append(cell)
        line_len += len(cell) + 2
    if line:
        lines.append("  ".join(line))
    return "\n".join(lines)


def _render_one(task):
    """Worker — render one (facies/poro/perm) array + meta to a PNG."""
    arr, meta, out_path, axis, prop = task
    t0 = time.perf_counter()

    # Cast / transform per property:
    #   - facies: int8 binary, plot_slices uses categorical colours
    #   - poro:   float32 in [0, 1], continuous mode, mask zeros (inactive)
    #   - perm:   float32 in [0, 60000] mD; log10(eps + perm) so the wide
    #             dynamic range renders interpretably. Inactive cells with
    #             perm=0 become log10(0.1) = -1, which mask_zeros will
    #             filter via the "where >0" check, so we add eps inside
    #             the log only for safety and rely on the original==0
    #             mask afterwards.
    if prop == "facies":
        arr = np.asarray(arr, dtype=np.int8)
    elif prop == "facies_alluvsim":
        # int8 array with codes -1..4. plot_slices auto-detects this
        # range and uses the discrete Alluvsim colormap + legend.
        arr = np.asarray(arr, dtype=np.int8)
    elif prop == "poro":
        arr = np.asarray(arr, dtype=np.float32)
    elif prop == "perm":
        raw = np.asarray(arr, dtype=np.float32)
        log_arr = np.log10(np.maximum(raw, 0.1))
        log_arr[raw <= 0] = 0.0     # keep inactive cells at 0 so mask_zeros hides them
        arr = log_arr
    else:
        raise ValueError(f"unknown prop {prop!r}")

    # Re-render the caption from the saved meta + (job-time) preset so
    # any change to the caption template takes effect on the next plot
    # run without a data re-gen. We try the saved ``preset`` field
    # first (added by recent generate.py runs); if absent (older shards)
    # the channel/delta caption falls back to the generic head.
    from resmill.dataset.captions import caption_for
    layer_type = meta.get("layer_type", "channel")
    caption = caption_for(layer_type, meta)
    caption_wrapped = "\n".join(textwrap.wrap(caption, width=130)) or ""
    n_lines = max(1, caption_wrapped.count("\n") + 1)

    # plot_slices builds its own figure with a fixed-size gridspec; its
    # internal ``title_pad=0.5"`` only fits ~2 lines and our captions
    # routinely wrap to 2-3 lines, so the suptitle spills into the
    # first row of subplot titles (X=0, X=21 …). Workaround: ask
    # plot_slices for no title, then add a top band by physically
    # shifting every axis down to make room.
    plot_slices(arr, axis=axis, n_slices=4, title=None)
    fig = plt.gcf()

    # Compute how much vertical figure-fraction the caption needs at
    # 11pt with 1.3-line spacing. Saved fig is ~6-9 inches tall
    # depending on data shape; allow a generous fixed band.
    fig_h_in = float(fig.get_size_inches()[1])
    title_height_in = 0.18 * n_lines + 0.10   # 0.18" per line + 0.1" pad
    title_band_frac = title_height_in / fig_h_in
    new_top = 1.0 - title_band_frac

    # Shift each axis position down so the gridspec ends at ``new_top``.
    # Iterating axes is the only way to override an add_gridspec that
    # was instantiated with explicit positional kwargs (subplots_adjust
    # does not reliably propagate to it).
    for ax in fig.axes:
        bb = ax.get_position()
        # Old gridspec span from bb.y0..bb.y1 inside the old
        # [bottom_orig, ~1.0] band. Rescale linearly to fit
        # [bottom_orig, new_top].
        bottom_orig = 0.04 / fig_h_in     # plot_slices default
        old_span = (1.0 - 0.04 / fig_h_in) - bottom_orig
        new_span = new_top - bottom_orig
        scale = new_span / old_span if old_span > 0 else 1.0
        ny0 = bottom_orig + (bb.y0 - bottom_orig) * scale
        ny1 = bottom_orig + (bb.y1 - bottom_orig) * scale
        ax.set_position([bb.x0, ny0, bb.width, ny1 - ny0])

    fig.suptitle(caption_wrapped, fontsize=11, fontweight="bold",
                 y=0.99, va="top")
    fig.text(0.5, -0.06, _format_param_block(meta),
             fontsize=11, ha="center", family="monospace", va="top")
    fig.savefig(out_path, dpi=90, bbox_inches="tight",
                bbox_extra_artists=list(fig.texts))
    plt.close(fig)
    return out_path.name, time.perf_counter() - t0


def _load_dataset(data_dir: Path):
    """Yield ``(facies, poro, perm, facies_alluvsim, meta, sample_idx)``
    for every sample across every shard. ``facies_alluvsim`` may be
    None for old shards that pre-date the 6-class write.
    """
    shards = sorted(d for d in data_dir.iterdir()
                    if d.is_dir() and d.name.startswith("shard_"))
    if not shards:
        raise SystemExit(f"no shards found in {data_dir}")
    sample_idx = 0
    for shard in shards:
        facies = np.load(shard / "facies.npy", mmap_mode="r")
        poro = np.load(shard / "poro.npy", mmap_mode="r")
        perm = np.load(shard / "perm.npy", mmap_mode="r")
        alluv_path = shard / "facies_alluvsim.npy"
        facies_alluv = np.load(alluv_path, mmap_mode="r") if alluv_path.exists() else None
        params = pq.read_table(shard / "params.parquet").to_pylist()
        if facies.shape[0] != len(params):
            raise RuntimeError(
                f"{shard.name}: facies has {facies.shape[0]} samples but "
                f"params has {len(params)}"
            )
        for k in range(facies.shape[0]):
            alluv_k = np.array(facies_alluv[k]) if facies_alluv is not None else None
            yield (np.array(facies[k]), np.array(poro[k]), np.array(perm[k]),
                   alluv_k, params[k], sample_idx)
            sample_idx += 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("data_dir", help="output dir with shard_*/ subdirs")
    p.add_argument("--prop",
                   choices=["facies", "poro", "perm", "facies_alluvsim", "all"],
                   default="all",
                   help="which property to plot; 'all' (default) writes "
                        "facies -> pictures/, poro -> poro_pictures/, "
                        "perm -> perm_pictures/, facies_alluvsim -> "
                        "facies_alluvsim_pictures/ in one pass")
    p.add_argument("--out", default=None,
                   help="override the output dir (only meaningful when "
                        "--prop is one of facies/poro/perm; ignored when "
                        "--prop=all because three subdirs are written)")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit", type=int, default=None,
                   help="render only the first N samples")
    p.add_argument("--axis", type=lambda s: None if s == "all" else int(s),
                   default=None,
                   help="0|1|2 for a single-axis grid, or 'all' (default) "
                        "for 3 rows of slices")
    args = p.parse_args()

    data_dir = Path(args.data_dir).resolve()
    props = (["facies", "poro", "perm", "facies_alluvsim"]
             if args.prop == "all" else [args.prop])

    # Resolve per-prop output dirs (mkdir before launching workers)
    out_dirs: dict[str, Path] = {}
    for prop in props:
        if args.out is not None and len(props) == 1:
            out_dirs[prop] = Path(args.out).resolve()
        else:
            out_dirs[prop] = data_dir / _PROP_DIR[prop]
        out_dirs[prop].mkdir(parents=True, exist_ok=True)

    print(f"data: {data_dir}")
    for prop in props:
        print(f"  -> {prop:6s}: {out_dirs[prop]}")

    tasks = []
    for facies, poro, perm, facies_alluv, meta, idx in _load_dataset(data_dir):
        if args.limit is not None and idx >= args.limit:
            break
        for prop in props:
            if prop == "facies_alluvsim":
                if facies_alluv is None:
                    print(f"  skip facies_alluvsim for sample {idx} "
                          f"(shard predates 6-class write)")
                    continue
                arr = facies_alluv
            else:
                arr = {"facies": facies, "poro": poro, "perm": perm}[prop]
            out_path = out_dirs[prop] / f"sample_{idx:04d}.png"
            tasks.append((arr, meta, out_path, args.axis, prop))

    print(f"plotting {len(tasks)} PNGs ({len(tasks) // len(props)} samples × "
          f"{len(props)} props) on {args.workers} workers")
    t_start = time.perf_counter()
    times: list[float] = []
    with Pool(args.workers) as pool:
        for n, (name, dt) in enumerate(
            pool.imap_unordered(_render_one, tasks), 1
        ):
            times.append(dt)
            if n % 64 == 0 or n == len(tasks):
                elapsed = time.perf_counter() - t_start
                eta = elapsed / n * (len(tasks) - n) if n < len(tasks) else 0
                print(f"  [{n:4d}/{len(tasks)}]  elapsed={elapsed:6.1f}s  "
                      f"eta={eta:6.1f}s", flush=True)
    wall = time.perf_counter() - t_start
    a = np.array(times)
    print(f"\n=== Done ===")
    print(f"  wall: {wall:.1f}s")
    print(f"  per-plot: mean={a.mean():.2f}s  min={a.min():.2f}s  "
          f"max={a.max():.2f}s")


if __name__ == "__main__":
    main()
