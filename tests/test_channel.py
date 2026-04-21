import numpy as np
import pytest
from georules.layers.channel import MeanderingChannelLayer, BraidedChannelLayer, ChannelLayerBase


# === Existing tests (unchanged logic) ===

def test_channel_shapes():
    layer = MeanderingChannelLayer(nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000)
    layer.create_geology(channel_width=40, n_channels=3)
    assert layer.poro_mat.shape == (64, 32, 16)
    assert layer.perm_mat.shape == (64, 32, 16)
    assert layer.active.shape == (64, 32, 16)


def test_channel_has_nonzero_facies():
    layer = MeanderingChannelLayer(nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000)
    layer.create_geology(channel_width=40, n_channels=5)
    assert layer.facies.max() > 0, "Channel model should produce at least some channel facies"


def test_channel_poro_in_bounds():
    layer = MeanderingChannelLayer(nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000)
    layer.create_geology(channel_width=40, n_channels=3)
    assert np.all(layer.poro_mat >= 0)
    assert np.all(layer.poro_mat <= 1)


def test_meandering_explicit_name():
    layer = MeanderingChannelLayer(
        nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000
    )
    layer.create_geology(channel_width=40, n_channels=3)
    assert layer.poro_mat.shape == (64, 32, 16)
    assert layer.facies.max() > 0


def test_base_class_not_directly_usable():
    """ChannelLayerBase.create_geology should raise NotImplementedError."""
    base = ChannelLayerBase(nx=4, ny=4, nz=4, x_len=100, y_len=100,
                            z_len=10, top_depth=500)
    with pytest.raises(NotImplementedError):
        base.create_geology()


# === Braided channel tests (fluvial engine + avulsion) ===

def test_braided_shapes():
    layer = BraidedChannelLayer(
        nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000
    )
    layer.create_geology(braidplain_width=80, n_channels=4)
    assert layer.poro_mat.shape == (64, 32, 16)
    assert layer.perm_mat.shape == (64, 32, 16)
    assert layer.active.shape == (64, 32, 16)
    assert layer.facies.shape == (64, 32, 16)


def test_braided_has_nonzero_facies():
    layer = BraidedChannelLayer(
        nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000
    )
    layer.create_geology(braidplain_width=80, n_channels=5)
    assert layer.facies.max() > 0


def test_braided_poro_in_bounds():
    layer = BraidedChannelLayer(
        nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000
    )
    layer.create_geology(braidplain_width=80, n_channels=3)
    assert np.all(layer.poro_mat >= 0)
    assert np.all(layer.poro_mat <= 1)


def test_braided_active_matches_facies():
    layer = BraidedChannelLayer(
        nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000
    )
    layer.create_geology(braidplain_width=80, n_channels=3)
    expected_active = (layer.facies > 0).astype(int)
    np.testing.assert_array_equal(layer.active, expected_active)


def test_braided_perm_positive_where_active():
    layer = BraidedChannelLayer(
        nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000
    )
    layer.create_geology(braidplain_width=80, n_channels=3)
    active_perm = layer.perm_mat[layer.active == 1]
    assert np.all(active_perm > 0)


def test_braided_in_reservoir():
    """BraidedChannelLayer should be stackable in a Reservoir."""
    from georules.layers.gaussian import GaussianLayer
    from georules.reservoir import Reservoir

    g = GaussianLayer(nx=64, ny=32, nz=8, x_len=1024, y_len=512,
                      z_len=24, top_depth=1000)
    g.create_geology(poro_ave=0.2, perm_ave=1.5, poro_std=0.03,
                     perm_std=0.5, ntg=0.7)

    b = BraidedChannelLayer(nx=64, ny=32, nz=8, x_len=1024, y_len=512,
                            z_len=24, top_depth=1024)
    b.create_geology(braidplain_width=80, n_channels=3)

    res = Reservoir([g, b])
    assert res.poro_mat.shape == (64, 32, 16)


def test_braided_deprecated_kwargs_are_ignored():
    """Old BBC-engine kwargs (n_threads, thread_width, bar_poro_factor)
    should still be accepted silently for backwards compatibility."""
    layer = BraidedChannelLayer(
        nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000
    )
    layer.create_geology(braidplain_width=80, n_channels=3,
                          n_threads=3, thread_width=30.0,
                          bar_poro_factor=0.7)
    assert layer.active.shape == (64, 32, 16)


def test_meandering_accepts_avulsion_kwargs():
    """In-model avulsion kwargs should flow through MeanderingChannelLayer."""
    np.random.seed(0)
    layer = MeanderingChannelLayer(
        nx=64, ny=32, nz=16, x_len=1024, y_len=512, z_len=48, top_depth=1000
    )
    layer.create_geology(channel_width=40, n_channels=3,
                          prob_avul_inside=0.3, prob_avul_outside=0.1)
    assert layer.facies.max() > 0
