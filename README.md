# ResMill

Rule-based 3D geological reservoir modeling.

## Install

```bash
pip install resmill
```

For development:

```bash
git clone https://anonymous.4open.science/r/ResMill-7377
cd ResMill
pip install -e ".[dev]"
```

## Quick Start

```python
import resmill as rm

# Create a turbidite lobe layer
lobe = rm.LobeLayer(nx=100, ny=100, nz=50, x_len=3000, y_len=3000, z_len=100, top_depth=5000)
lobe.create_geology(poro_ave=0.20, perm_ave=1.5, poro_std=0.03, perm_std=0.5, ntg=0.7)

# Create a meandering channel layer
channel = rm.MeanderingChannelLayer(nx=128, ny=64, nz=16, x_len=2048, y_len=1024, z_len=48, top_depth=5100)
channel.create_geology(channel_width=40, n_channels=5)

# Create a braided channel layer
braided = rm.BraidedChannelLayer(nx=128, ny=64, nz=16, x_len=2048, y_len=1024, z_len=48, top_depth=5148)
braided.create_geology(braidplain_width=600, n_channels=10, n_threads=3)

# Stack into a reservoir
reservoir = rm.Reservoir([lobe, channel, braided])

# Visualize
rm.plot_cube_slices(reservoir.poro_mat, title="Porosity")
```

## Layer Types

- **LobeLayer** — Turbidite lobe deposition with compensational stacking
- **GaussianLayer** — Sequential Gaussian simulation for heterogeneous facies
- **MeanderingChannelLayer** — Meandering fluvial channels with migration, avulsion, and point bars
- **BraidedChannelLayer** — Braided fluvial channels with multi-thread systems, mid-channel bars, and bifurcation-bar-confluence geometry

## License

MIT
