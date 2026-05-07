import numpy as np
import pytest
from resmill.layers.gaussian import GaussianLayer


def test_gaussian_shapes():
    layer = GaussianLayer(nx=20, ny=20, nz=10, x_len=200, y_len=200, z_len=50, top_depth=1000)
    layer.create_geology(poro_ave=0.2, perm_ave=1.5, poro_std=0.03, perm_std=0.5, ntg=0.7)
    assert layer.poro_mat.shape == (20, 20, 10)
    assert layer.perm_mat.shape == (20, 20, 10)
    assert layer.active.shape == (20, 20, 10)


def test_gaussian_poro_non_negative():
    layer = GaussianLayer(nx=20, ny=20, nz=10, x_len=200, y_len=200, z_len=50, top_depth=1000)
    layer.create_geology(poro_ave=0.2, perm_ave=1.5, poro_std=0.03, perm_std=0.5, ntg=0.7)
    assert np.all(layer.perm_mat >= 0)


def test_gaussian_active_is_binary():
    layer = GaussianLayer(nx=20, ny=20, nz=10, x_len=200, y_len=200, z_len=50, top_depth=1000)
    layer.create_geology(poro_ave=0.2, perm_ave=1.5, poro_std=0.03, perm_std=0.5, ntg=0.7)
    assert set(np.unique(layer.active)).issubset({0, 1})


def test_gaussian_ntg_approximate():
    layer = GaussianLayer(nx=50, ny=50, nz=20, x_len=500, y_len=500, z_len=100, top_depth=1000)
    layer.create_geology(poro_ave=0.2, perm_ave=1.5, poro_std=0.03, perm_std=0.5, ntg=0.7)
    actual_ntg = layer.active.mean()
    assert abs(actual_ntg - 0.7) < 0.05


def test_gaussian_inactive_cells_zero_poro():
    layer = GaussianLayer(nx=20, ny=20, nz=10, x_len=200, y_len=200, z_len=50, top_depth=1000)
    layer.create_geology(poro_ave=0.2, perm_ave=1.5, poro_std=0.03, perm_std=0.5, ntg=0.7)
    inactive = layer.active == 0
    assert np.all(layer.poro_mat[inactive] == 0)
    assert np.all(layer.perm_mat[inactive] == 0)
