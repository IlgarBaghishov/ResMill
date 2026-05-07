# ResMill

Rule-based 3D geological reservoir modeling library.

## What This Project Does

ResMill generates synthetic 3D geological models using rule-based and stochastic methods. It targets subsurface reservoir modeling for oil & gas, groundwater, and carbon storage applications.

## Problem It Solves

Building realistic 3D geological models typically requires expensive commercial software and extensive manual input. ResMill provides a Python-native, pip-installable alternative that generates geologically plausible reservoir models programmatically. Users define layer types and parameters; the library handles the physics-based geometry and property modeling.

## Supported Geology Types

- **Lobe layers** — Turbidite lobe deposition with compensational stacking, Bouma sequences, and upthinning
- **Gaussian layers** — Sequential Gaussian simulation (SGS) with spatial correlation for heterogeneous sand/shale distributions
- **Channel layers** — Fluvial channel systems with meandering, migration, avulsion, neck cutoffs, and point bar geometry

## Architecture

- `Layer` base class defines grid geometry (nx, ny, nz, dimensions, depth, dip)
- Each layer type inherits from `Layer` and implements `create_geology()` to populate 3D property arrays
- `Reservoir` stacks multiple layers vertically, validating compatibility
- All outputs are numpy arrays shaped `(nx, ny, nz)`

## Key Commands

- Install: `pip install -e ".[dev]"`
- Test: `pytest tests/`
- Tutorial: `jupyter notebook notebooks/tutorial.ipynb`

## Conventions

- Array ordering: `(nx, ny, nz)` with `meshgrid(..., indexing='ij')`
- Properties: `poro_mat` (porosity, 0-1), `perm_mat` (permeability, mD), `active` (0/1 facies mask)
- Physics parameters go in `create_geology()`, not `__init__()` — init is grid-only
- Channel internals use Numba JIT and are prefixed with `_` (private)

## Parameter units — gotchas

- **Length is in meters, everywhere.** `x_len, y_len, z_len, top_depth, dx, dy, dz` are meters; the fluvial engine hardcodes `g = 9.8 m/s²`, so all `mCH*`, `mLV*`, `mCSLO*`, `mdistMigrate` (channel/delta) and `dhmin/dhmax/rmin/rmax` (lobe) and `facies_filter/sand_filter` (gaussian) are also meters. `LobeLayer` and `GaussianLayer` convert these physical inputs to cell-units internally via `self.dx/dy/dz`.
- **`perm_ave` and `perm_std` for `LobeLayer` and `GaussianLayer` are in log10(mD) space**, not linear mD. See the docstrings in `layers/lobe.py` and `layers/gaussian.py`. Typical sensible ranges:
  - `perm_ave`: `[0, 4]` → mean perm spans 1 to 10,000 mD
  - `perm_std`: `[0.1, 1.5]` → log10-std (factor-of-1.3 to factor-of-30 spread)
  - Passing linear-mD values (e.g. `perm_ave=500`) makes the internal `10**perm_mat` overflow float64 and the output cells saturate to the on-disk clip ceiling (60000 mD). `poro_ave` and `poro_std` stay in linear [0, 1] units.
- `poro_mat` is not zeroed outside `active` in GaussianLayer (the Gaussian porosity field fills all cells); if you want "reservoir-only" porosity at consumption time, do `poro * facies` yourself.
- `facies` semantics differ by layer: `LobeLayer.facies` is a multi-valued lobe index (1..N); `MeanderingChannelLayer`/`BraidedChannelLayer`/`DeltaLayer.facies` are 0/1; `GaussianLayer` has no `facies` attribute — use `active` for the 0/1 mask.
