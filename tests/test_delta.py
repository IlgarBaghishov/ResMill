import numpy as np
import pytest

from georules.layers.delta import DeltaLayer


def _make_layer(**kwargs):
    defaults = dict(nx=64, ny=48, nz=24, x_len=64 * 16, y_len=48 * 16,
                    z_len=24 * 3, top_depth=1000)
    defaults.update(kwargs)
    return DeltaLayer(**defaults)


def test_delta_shapes():
    np.random.seed(0)
    layer = _make_layer()
    layer.create_geology(feeder_width=50)
    assert layer.poro_mat.shape == (64, 48, 24)
    assert layer.perm_mat.shape == (64, 48, 24)
    assert layer.active.shape == (64, 48, 24)
    assert layer.facies.shape == (64, 48, 24)


def test_delta_has_sand():
    np.random.seed(0)
    layer = _make_layer()
    layer.create_geology(feeder_width=50)
    assert layer.active.sum() > 50, (
        "Delta should produce a meaningful sand body")


def test_delta_poro_in_bounds():
    np.random.seed(0)
    layer = _make_layer()
    layer.create_geology(feeder_width=50)
    assert np.all(layer.poro_mat >= 0)
    assert np.all(layer.poro_mat <= 1)


def test_delta_perm_positive_where_active():
    np.random.seed(0)
    layer = _make_layer()
    layer.create_geology(feeder_width=50)
    active_perm = layer.perm_mat[layer.active == 1]
    assert np.all(active_perm > 0)


def test_delta_facies_is_binary():
    """Delta should emit binary facies for cross-layer `.active` consistency."""
    np.random.seed(0)
    layer = _make_layer()
    layer.create_geology(feeder_width=50)
    assert set(np.unique(layer.facies)).issubset({0, 1})


def test_delta_fan_spreads_in_y():
    """Fan architecture: the Y-spread of channels in the fan region
    (middle + distal X) should be larger than in the feeder region
    (proximal X) — a direct signature of the bifurcation tree.
    """
    np.random.seed(0)
    layer = _make_layer()
    layer.create_geology(feeder_width=50, n_generations=6)
    a = layer.active
    nx = a.shape[0]
    proximal = a[: nx // 4, :, :].sum(axis=(0, 2))
    distal = a[nx // 2 :, :, :].sum(axis=(0, 2))
    prox_y_span = float((proximal > 0).sum())
    dist_y_span = float((distal > 0).sum())
    assert dist_y_span > prox_y_span, (
        f"Expected Y-span of distal fan > proximal feeder; "
        f"got prox={prox_y_span}, dist={dist_y_span}")


def test_delta_in_reservoir():
    from georules.layers.gaussian import GaussianLayer
    from georules.reservoir import Reservoir
    np.random.seed(0)

    g = GaussianLayer(nx=64, ny=48, nz=8, x_len=64 * 16, y_len=48 * 16,
                      z_len=24, top_depth=1000)
    g.create_geology(poro_ave=0.2, perm_ave=1.5, poro_std=0.03,
                     perm_std=0.5, ntg=0.7)

    d = _make_layer(top_depth=1024)
    d.create_geology(feeder_width=50)

    res = Reservoir([g, d])
    assert res.poro_mat.shape == (64, 48, 32)
