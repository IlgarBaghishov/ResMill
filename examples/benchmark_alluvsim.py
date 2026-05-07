"""Benchmark + render Alluvsim presets — plain Python script (no notebook).

Run this on your machine, then send me the output. I'll run it on mine and
we can compare exactly what each step costs. This isolates the engine cost
from VS Code's notebook overhead.

Usage::

    cd $HOME/ResMill
    python examples/benchmark_alluvsim.py

Outputs:
* Per-cell timing breakdown printed to stdout
* PNG images written to ``/tmp/alluvsim_<preset>.png`` (one per preset,
  same layout as the tutorial: 4 XY slices + XZ + YZ)
* System info (Python / numpy / scipy / numba / CPU) printed at the top
"""
from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

# Headless-safe backend (won't pop up windows; just writes PNGs)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import resmill as rm
from resmill.layers.channel import (
    PV_SHOESTRING, CB_JIGSAW, CB_LABYRINTH, SH_DISTAL, SH_PROXIMAL,
)


# ---------------------------------------------------------------------------
# Configuration — same grid as the tutorial / dataset target
# ---------------------------------------------------------------------------
GRID = dict(nx=64, ny=64, nz=32, x_len=640, y_len=640, z_len=16, top_depth=0)
SEED = 69069
OUT_DIR = Path("/tmp")

# 6-class Alluvsim facies palette
FACIES_NAMES = {
    -1: "FF (floodplain)", 0: "FFCH (mud plug)", 1: "CS (splay)",
    2: "LV (levee)", 3: "LA (point bar)", 4: "CH (channel)",
}
FACIES_COLORS = {
    -1: "#b4b4b4", 0: "#5b4636", 1: "#f2d16b",
    2: "#e8a23a", 3: "#c89b5e", 4: "#7a3f14",
}


def alluvsim_cmap():
    codes = sorted(FACIES_COLORS.keys())
    cmap = mcolors.ListedColormap([FACIES_COLORS[c] for c in codes])
    bounds = [c - 0.5 for c in codes] + [codes[-1] + 0.5]
    return cmap, mcolors.BoundaryNorm(bounds, cmap.N)


def build_channel(preset, layer_cls, seed=SEED):
    layer = layer_cls(**GRID)
    np.random.seed(seed)
    kw = dict(preset)
    z_max = max(kw["level_z"])
    scale = GRID["z_len"] * 0.95 / z_max if z_max > 0 else 1.0
    kw["level_z"] = [z * scale for z in kw["level_z"]]
    kw["mCHsource"] = GRID["y_len"] / 2
    kw["seed"] = seed
    layer.create_geology(**kw)
    return layer


def render(layer, title, out_path, n_xy_slices=4):
    """Render a reservoir as 4 XY slices + 1 XZ + 1 YZ in a single PNG."""
    facies = layer.facies
    nx, ny, nz = facies.shape
    cmap, norm = alluvsim_cmap()
    iz_indices = np.linspace(0, nz - 1, n_xy_slices, dtype=int)
    iy_mid = ny // 2
    ix_mid = nx // 2

    n_panels = n_xy_slices + 2
    fig, axes = plt.subplots(1, n_panels, figsize=(2.0 * n_panels, 3.0), dpi=72)
    plot_kw = dict(cmap=cmap, norm=norm, interpolation="nearest",
                   origin="lower", aspect="auto")
    for j, iz in enumerate(iz_indices):
        axes[j].imshow(facies[:, :, iz].T, **plot_kw)
        axes[j].set_title(f"XY z={iz}", fontsize=9)
        if j != 0:
            axes[j].set_yticklabels([])
    axes[-2].imshow(facies[:, iy_mid, :].T, **plot_kw)
    axes[-2].set_title(f"XZ iy={iy_mid}", fontsize=9)
    axes[-1].imshow(facies[ix_mid, :, :].T, **plot_kw)
    axes[-1].set_title(f"YZ ix={ix_mid}", fontsize=9)
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=72, bbox_inches="tight")
    plt.close(fig)


def system_info():
    print("=" * 70)
    print("System info")
    print("=" * 70)
    print(f"  Python      : {sys.version.split()[0]}  ({platform.python_implementation()})")
    print(f"  Platform    : {platform.platform()}")
    print(f"  Machine     : {platform.machine()}")
    print(f"  Processor   : {platform.processor() or 'unknown'}")
    try:
        import multiprocessing
        print(f"  CPU count   : {multiprocessing.cpu_count()}")
    except Exception:
        pass
    # Try /proc/cpuinfo on Linux for model name
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    print(f"  CPU model   : {line.split(':', 1)[1].strip()}")
                    break
    except Exception:
        pass
    print(f"  NumPy       : {np.__version__}")
    try:
        import scipy
        print(f"  SciPy       : {scipy.__version__}")
    except ImportError:
        print(f"  SciPy       : NOT INSTALLED")
    try:
        import numba
        print(f"  Numba       : {numba.__version__}")
    except ImportError:
        print(f"  Numba       : NOT INSTALLED")
    print(f"  Matplotlib  : {matplotlib.__version__}  (backend={matplotlib.get_backend()})")
    print(f"  ResMill    : (loaded from {Path(rm.__file__).resolve()})")
    print()


def main():
    system_info()

    print("=" * 70)
    print(f"Grid: {GRID['nx']} x {GRID['ny']} x {GRID['nz']}  "
          f"({GRID['x_len']} x {GRID['y_len']} x {GRID['z_len']} m)")
    print("=" * 70)

    # ----- 1. Warm up numba JIT + scipy + matplotlib (one time) -----
    print("\n[1/3] Warming up numba JIT + scipy + matplotlib (~3-10s)...")
    t0 = time.perf_counter()
    w = rm.ChannelLayer(**GRID)
    np.random.seed(0)
    w.create_geology(
        seed=0, ntime=15,
        nlevel=2, level_z=[3.0, 8.0], NTGtarget=0.10,
        probAvulOutside=0.10, probAvulInside=0.30,
        mLVdepth=0.5, mLVwidth=20, mLVheight=0.4,
        mCSnum=1, mCSnumlobe=1,
        mFFCHprop=0.4, stdevFFCHprop=0.1,
    )
    # Warm matplotlib too
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.imshow(w.facies[:, :, 5].T, cmap=alluvsim_cmap()[0])
    plt.close(fig)
    del w
    t_warm = time.perf_counter() - t0
    print(f"      Warm-up complete in {t_warm:.2f}s.")

    # ----- 2. Per-preset generation + rendering -----
    print("\n[2/3] Generating + rendering each preset (one PNG per preset)...")
    presets = [
        ("pv_shoestring", PV_SHOESTRING, rm.ChannelLayer),
        ("cb_jigsaw",     CB_JIGSAW,     rm.ChannelLayer),
        ("cb_labyrinth",  CB_LABYRINTH,  rm.ChannelLayer),
        ("sh_distal",     SH_DISTAL,     rm.ChannelLayer),
        ("sh_proximal",   SH_PROXIMAL,   rm.ChannelLayer),
    ]
    print(f"      {'preset':14s}  {'gen':>6s}  {'render':>7s}  {'NTG':>6s}  {'png':<32s}")
    print(f"      {'-' * 14}  {'-' * 6}  {'-' * 7}  {'-' * 6}  {'-' * 32}")
    total_gen = 0.0
    total_render = 0.0
    for name, preset, cls in presets:
        out_path = OUT_DIR / f"alluvsim_{name}.png"

        t0 = time.perf_counter()
        layer = build_channel(preset, cls)
        t_gen = time.perf_counter() - t0
        total_gen += t_gen

        t0 = time.perf_counter()
        render(layer, f"{name}", out_path)
        t_render = time.perf_counter() - t0
        total_render += t_render

        ntg = layer.active.mean() * 100
        print(f"      {name:14s}  {t_gen:5.2f}s  {t_render:6.2f}s  {ntg:5.1f}%  {str(out_path):<32s}")

    # ----- 3. Summary -----
    print("\n[3/3] Summary")
    print(f"      Total generation time : {total_gen:6.2f}s")
    print(f"      Total render time     : {total_render:6.2f}s")
    print(f"      Total (after warmup)  : {total_gen + total_render:6.2f}s")
    print(f"      Warmup                : {t_warm:6.2f}s")
    print(f"      Wall clock total      : {total_gen + total_render + t_warm:6.2f}s")
    print()
    print("=" * 70)
    print(f"PNGs written to: {OUT_DIR}/alluvsim_*.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
