import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import pytest
from resmill.plotting import plot_cube_slices, plot_slices, RESMILL_CMAP


def _figure_after(fn):
    """Run ``fn``; return whichever Figure was created (and clean up)."""
    plt.close('all')
    fn()
    figs = [plt.figure(n) for n in plt.get_fignums()]
    assert len(figs) == 1, f'expected exactly one figure, got {len(figs)}'
    fig = figs[0]
    plt.close('all')
    return fig


def test_plot_cube_slices_returns_fig():
    data = np.random.rand(10, 10, 5)
    fig, ax = plot_cube_slices(data)
    assert fig is not None
    plt.close(fig)


def test_plot_slices_creates_one_figure():
    data = np.random.rand(10, 10, 5)
    fig = _figure_after(lambda: plot_slices(data, axis=2))
    assert fig is not None


def test_plot_slices_returns_none():
    """plot_slices must return None so last-expr auto-display in Jupyter
    doesn't double up with flush_figures."""
    data = np.random.rand(10, 10, 5)
    plt.close('all')
    result = plot_slices(data, axis=2)
    assert result is None
    plt.close('all')


def test_plot_slices_all_axes_default():
    """No axis ⇒ 8 slices on each of X, Y, Z = 24 panels (+ a colorbar axis
    for continuous data)."""
    data = np.random.rand(16, 16, 16)
    fig = _figure_after(lambda: plot_slices(data, title='all axes'))
    # 24 slice panels + 1 horizontal colorbar axis (continuous mode)
    assert len(fig.axes) == 25


def test_plot_slices_alluvsim_auto_detect():
    """An int ndarray with values ⊂ {-1..4} → Alluvsim mode auto-detected."""
    arr = np.full((8, 8, 8), -1, dtype=np.int8)
    arr[3, 3, :] = 4   # one CH ribbon
    fig = _figure_after(lambda: plot_slices(arr, axis=2))
    assert fig is not None


def test_plot_slices_layer_auto_detect():
    """A Layer-like with .facies attr → Alluvsim mode + per-facies summary."""
    class _StubLayer:
        facies = np.full((8, 8, 8), -1, dtype=np.int8)
        facies[3, 3, :] = 4
        active = (facies >= 1).astype(np.int8)
    fig = _figure_after(lambda: plot_slices(_StubLayer()))
    assert fig is not None


def test_plot_cube_slices_with_zeros():
    """Handles data with many zeros (inactive cells)."""
    data = np.zeros((10, 10, 5))
    data[3:7, 3:7, 1:4] = np.random.rand(4, 4, 3)
    fig, ax = plot_cube_slices(data)
    assert fig is not None
    plt.close(fig)


def test_plot_slices_axis_0():
    data = np.random.rand(10, 10, 5)
    fig = _figure_after(lambda: plot_slices(data, axis=0, n_slices=3))
    assert fig is not None


def test_plot_mask_zeros_false():
    """mask_zeros=False keeps zero values visible (for facies data)."""
    data = np.zeros((10, 10, 5))
    data[2:8, 2:8, 1:4] = 1.0
    data[4:6, 4:6, 1:4] = 2.0

    fig1, ax1 = plot_cube_slices(data, mask_zeros=False)
    assert fig1 is not None
    plt.close(fig1)

    fig2 = _figure_after(lambda: plot_slices(data, axis=2, mask_zeros=False))
    assert fig2 is not None


def test_resmill_cmap_registered():
    """RESMILL_CMAP should be importable and registered with matplotlib."""
    assert RESMILL_CMAP is not None
    assert RESMILL_CMAP.name == 'resmill'
    # Should also be fetchable by name
    cmap = plt.get_cmap('resmill')
    assert cmap is not None


def test_default_cmap_is_resmill():
    """plot_cube_slices and plot_slices should default to resmill cmap."""
    data = np.random.rand(10, 10, 5)
    fig, ax = plot_cube_slices(data)
    assert fig is not None
    plt.close(fig)

    fig2 = _figure_after(lambda: plot_slices(data, axis=2))
    assert fig2 is not None
