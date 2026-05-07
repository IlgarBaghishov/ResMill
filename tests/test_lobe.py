import numpy as np
import pytest
from resmill.layers.lobe import LobeLayer


def test_lobe_shapes():
    layer = LobeLayer(nx=10, ny=10, nz=5, x_len=100, y_len=100, z_len=10, top_depth=1000)
    layer.create_geology(poro_ave=0.2, perm_ave=1.0, poro_std=0.03, perm_std=0.3, ntg=0.7)
    assert layer.poro_mat.shape == (10, 10, 5)
    assert layer.perm_mat.shape == (10, 10, 5)
    assert layer.active.shape == (10, 10, 5)


def test_lobe_has_facies():
    layer = LobeLayer(nx=10, ny=10, nz=5, x_len=100, y_len=100, z_len=10, top_depth=1000)
    layer.create_geology(poro_ave=0.2, perm_ave=1.0, poro_std=0.03, perm_std=0.3, ntg=0.7)
    assert hasattr(layer, 'facies')
    assert layer.facies.shape == (10, 10, 5)


def test_lobe_poro_non_negative():
    layer = LobeLayer(nx=10, ny=10, nz=5, x_len=100, y_len=100, z_len=10, top_depth=1000)
    layer.create_geology(poro_ave=0.2, perm_ave=1.0, poro_std=0.03, perm_std=0.3, ntg=0.7)
    assert np.all(layer.poro_mat >= 0)
    assert np.all(layer.perm_mat >= 0)


def test_lobe_active_is_binary():
    layer = LobeLayer(nx=10, ny=10, nz=5, x_len=100, y_len=100, z_len=10, top_depth=1000)
    layer.create_geology(poro_ave=0.2, perm_ave=1.0, poro_std=0.03, perm_std=0.3, ntg=0.7)
    assert set(np.unique(layer.active)).issubset({0, 1})
