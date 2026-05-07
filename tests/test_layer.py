import numpy as np
import pytest
from resmill.layers.base import Layer


def test_grid_dimensions():
    layer = Layer(nx=4, ny=3, nz=2, x_len=100, y_len=60, z_len=10, top_depth=500)
    assert layer.dx == 25.0
    assert layer.dy == 20.0
    assert layer.dz == 5.0


def test_meshgrid_shape():
    layer = Layer(nx=4, ny=3, nz=2, x_len=100, y_len=60, z_len=10, top_depth=500)
    assert layer.X.shape == (5, 4)  # (nx+1, ny+1)
    assert layer.Y.shape == (5, 4)


def test_surface_depth_no_dip():
    layer = Layer(nx=4, ny=3, nz=2, x_len=100, y_len=60, z_len=10, top_depth=500, dip=0)
    np.testing.assert_allclose(layer.z1, 500.0)
    np.testing.assert_allclose(layer.z2, 510.0)


def test_surface_depth_with_dip():
    layer = Layer(nx=4, ny=3, nz=2, x_len=100, y_len=60, z_len=10, top_depth=500, dip=10)
    assert layer.z1[0, 0] == 500.0  # at y=0
    assert layer.z1[0, -1] > 500.0  # at y=y_len, dip adds depth


def test_active_default_all_ones():
    layer = Layer(nx=4, ny=3, nz=2, x_len=100, y_len=60, z_len=10, top_depth=500)
    assert layer.active.shape == (4, 3, 2)
    assert np.all(layer.active == 1)


def test_properties_none_before_geology():
    layer = Layer(nx=4, ny=3, nz=2, x_len=100, y_len=60, z_len=10, top_depth=500)
    assert layer.poro_mat is None
    assert layer.perm_mat is None


def test_create_geology_raises():
    layer = Layer(nx=4, ny=3, nz=2, x_len=100, y_len=60, z_len=10, top_depth=500)
    with pytest.raises(NotImplementedError):
        layer.create_geology()


def test_zz_has_two_surfaces():
    layer = Layer(nx=4, ny=3, nz=2, x_len=100, y_len=60, z_len=10, top_depth=500)
    assert len(layer.zz) == 2
    np.testing.assert_array_equal(layer.zz[0], layer.z1)
    np.testing.assert_array_equal(layer.zz[1], layer.z2)
