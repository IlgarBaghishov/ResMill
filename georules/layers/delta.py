"""Delta layer — distributary-fan architecture.

A DeltaLayer is a prograding distributary fan.  In plan view (XY):

    feeder ──► apex ──┬──► branch ──┬──► branch ──► mouth bar
                      │             └──► branch ──► mouth bar
                      └──► branch ──┬──► branch ──► mouth bar
                                    └──► branch ──► mouth bar

A single feeder channel enters at (x=0, y=y_center) and reaches the
apex at ``x = apex_x_fraction * x_len``.  At the apex the channel
bifurcates recursively (binary tree, ``bifurcation_depth`` levels)
into distributaries that fan out within ``fan_angle_deg``.  Each leaf
distributary ends in a mouth bar painted with Alluvsim's
``calc_lobe.for`` envelope (parabolic proximal, elliptical distal).

In cross-section (XZ) the apex progrades: successive generations put
the apex progressively further in +x while stacking at higher
``chelev``, giving the classic clinoform / delta-front geometry.

Rasterisation reuses the meandering engine's ``genchannel`` (U-shape
channel cross-section with parabolic thalweg) so delta distributaries
look identical to a short MeanderingChannelLayer at their scale.
"""
from __future__ import annotations

import numpy as np

from .base import Layer
from .channel import ChannelLayerBase
from ._genchannel import genchannel


def _perturbed_centerline(x0, y0, heading, length, n_nodes,
                          meander_amp=0.04, rng=None):
    """Straight segment from (x0, y0) with heading + small sinusoidal wiggle.

    ``meander_amp`` is the peak perpendicular offset expressed as a
    fraction of segment length.  Wiggle tapers to zero at both ends so
    consecutive segments join smoothly.
    """
    if rng is None:
        rng = np.random.default_rng()
    ts = np.linspace(0.0, length, n_nodes)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    k = rng.uniform(0.8, 1.8)
    taper = np.sin(np.pi * ts / length)
    perp = meander_amp * length * taper * np.sin(k * np.pi * ts / length + phase)
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    xs = x0 + ts * cos_h - perp * sin_h
    ys = y0 + ts * sin_h + perp * cos_h
    return xs, ys


def _build_distributary_tree(apex_x, apex_y, fan_angle_rad,
                             max_depth, trunk_length,
                             feeder_width, step_len,
                             length_taper=0.85, width_taper=0.72,
                             meander_amp=0.04,
                             children_per_split=(3, 5), rng=None):
    """N-ary distributary tree anchored at ``(apex_x, apex_y)``.

    Returns a list of segment dicts:
        {'xs': np.ndarray, 'ys': np.ndarray,
         'width': float, 'heading_end': float, 'is_leaf': bool}

    Each node splits into ``k`` children (sampled uniformly from
    ``children_per_split``), with headings distributed evenly across
    ``subtree_angle`` and split points perturbed perpendicular to the
    parent heading so siblings emerge from distinct positions.  This
    produces a dense multi-channel net filling the fan rather than a
    sparse binary tree with visible gaps between branches.
    """
    if rng is None:
        rng = np.random.default_rng()
    kmin, kmax = int(children_per_split[0]), int(children_per_split[1])
    segments: list[dict] = []

    def recurse(x0, y0, heading, depth_remaining, subtree_angle, length, width):
        n_nodes = max(30, int(length / step_len))
        xs, ys = _perturbed_centerline(x0, y0, heading, length, n_nodes,
                                       meander_amp=meander_amp, rng=rng)
        is_leaf = depth_remaining == 0
        segments.append({
            'xs': xs, 'ys': ys, 'width': float(width),
            'heading_end': float(heading), 'is_leaf': is_leaf,
        })
        if is_leaf:
            return
        k = int(rng.integers(kmin, kmax + 1))
        perp_x = -np.sin(heading)
        perp_y = np.cos(heading)
        # Even angular distribution across subtree_angle.
        # frac_i ∈ (-0.5, +0.5), spread = 0.9 so leaves stay inside the
        # parent's subtree sector.
        for i in range(k):
            frac = (i + 0.5) / k - 0.5
            frac += rng.uniform(-0.15, 0.15) / k  # small jitter per child
            child_heading = heading + 0.9 * subtree_angle * frac
            split_jitter = 1.2 * width * frac
            sx = xs[-1] + split_jitter * perp_x
            sy = ys[-1] + split_jitter * perp_y
            recurse(
                sx, sy, child_heading,
                depth_remaining - 1,
                subtree_angle / max(k, 2),
                length * length_taper, width * width_taper,
            )

    recurse(apex_x, apex_y, 0.0, max_depth, fan_angle_rad,
            trunk_length, feeder_width)
    return segments


def _stamp_centerline(xs, ys, width, chelev, nx, ny, nz,
                      xsiz, ysiz, zsiz, x_grid, y_grid,
                      dwratio, poro0, facies, poro):
    """Rasterise a single centerline into ``facies``/``poro`` via genchannel.

    Computes tangents (vx, vy), curvature and thalweg asymmetry from
    the centerline; all other genchannel inputs carry no-op defaults.

    The centerline is passed through as-is, including nodes that fall
    outside the grid.  ``genchannel``'s ``find_near_grid`` already clips
    the stamp region to valid grid cells, and keeping the out-of-grid
    nodes makes them available as nearest-neighbour anchors for cells
    near the boundary — so channels extending past the grid still paint
    their in-grid portion cleanly up to the edge instead of being
    dropped (the old 20-node minimum) and leaving a visible gap.
    """
    if xs.size < 3:
        return
    cx = xs.astype(float)
    cy = ys.astype(float)
    dx = np.gradient(cx)
    dy = np.gradient(cy)
    speed = np.sqrt(dx * dx + dy * dy) + 1e-9
    vx = dx / speed
    vy = dy / speed
    ddx = np.gradient(vx)
    ddy = np.gradient(vy)
    curv = vx * ddy - vy * ddx
    max_abs = float(np.abs(curv).max()) + 1e-9
    thalweg = 0.5 + 0.25 * (curv / max_abs)
    thalweg = np.clip(thalweg, 0.05, 0.95)
    chwidth = width * np.ones_like(cx)
    genchannel(
        width, xsiz, ysiz, chelev, zsiz,
        nx, ny, nz, cx, cy, x_grid, y_grid,
        vx, vy, curv, 0.9, 5.0 * zsiz,
        1, 0, 0, 0, facies, poro, poro0,
        thalweg, chwidth, dwratio,
        [1_000_000_000], 8_000,
    )


def _paint_mouth_bar(facies, poro, ex, ey, heading,
                     LL, WW, hw_ratio, dw_ratio,
                     chelev, xsiz, ysiz, zsiz,
                     x_grid, y_grid, nz, poro0):
    """Paint Alluvsim-style mouth-bar envelope at a distributary terminus.

    Along-axis length ``LL``; half-width envelope ``y_fn(s)`` is
    parabolic from 0 to ``s_l = LL/3`` (grows from WW/4 to WW) and
    elliptical from ``s_l`` to ``LL`` (tapers to zero).  At each (s, d)
    inside the envelope, writes facies=1 across the vertical slab
    ``[chelev - y_fn*dw_ratio*f, chelev + y_fn*hw_ratio*f]`` where
    ``f = 1 - (d/y_fn)^2`` — direct port of calc_lobe.for lines 190-219.
    """
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    s_l = LL / 3.0
    pad = max(LL, WW)
    xmin = ex - pad; xmax = ex + pad
    ymin = ey - pad; ymax = ey + pad
    nx = facies.shape[0]; ny = facies.shape[1]
    ix0 = max(0, int((xmin - x_grid[0]) / xsiz))
    ix1 = min(nx, int((xmax - x_grid[0]) / xsiz) + 1)
    iy0 = max(0, int((ymin - y_grid[0]) / ysiz))
    iy1 = min(ny, int((ymax - y_grid[0]) / ysiz) + 1)

    for ix in range(ix0, ix1):
        for iy in range(iy0, iy1):
            dx = x_grid[ix] - ex
            dy = y_grid[iy] - ey
            s = dx * cos_h + dy * sin_h
            if s < 0.0 or s > LL:
                continue
            d = -dx * sin_h + dy * cos_h
            if s <= s_l:
                # Parabolic proximal ramp: WW/4 at apex → WW at s_l.
                y_fn = WW * (0.25 + 0.75 * (s / s_l) ** 2)
            else:
                u = (s - s_l) / (LL - s_l)
                y_fn = WW * np.sqrt(max(0.0, 1.0 - u * u))
            if y_fn < 1e-3 or abs(d) > y_fn:
                continue
            env = 1.0 - (d / y_fn) ** 2
            top_z = chelev + y_fn * hw_ratio * env
            bot_z = chelev - y_fn * dw_ratio * env
            iz_bot = max(0, int(bot_z / zsiz))
            iz_top = min(nz - 1, int(top_z / zsiz))
            if iz_top < iz_bot:
                continue
            facies[ix, iy, iz_bot:iz_top + 1] = 1
            depth = iz_top - iz_bot + 1
            if depth <= 0:
                continue
            zz = np.arange(depth)
            profile = 0.9 / (1.0 + np.exp(-4.0 * (zz / max(1, depth) - 0.2))) + 0.1
            poro_slab = (profile * poro0 * env + 0.05).clip(0.05, 0.99)
            existing = poro[ix, iy, iz_bot:iz_top + 1]
            poro[ix, iy, iz_bot:iz_top + 1] = np.maximum(existing, poro_slab)


class DeltaLayer(ChannelLayerBase):
    """Prograding distributary fan.

    Feeder + bifurcating distributary tree + mouth-bar lobes at leaf
    terminals, stacked across successive aggradation levels so the
    apex progrades in +x (clinoform architecture in XZ).

    Channel rasterisation reuses ``_genchannel.genchannel`` (same
    U-shape cross-section as ``MeanderingChannelLayer``); the mouth-bar
    envelope is a direct port of Alluvsim's ``calc_lobe.for``.
    """

    def create_geology(self, feeder_width=60.0, n_generations=8,
                       fan_angle_deg=95.0, bifurcation_depth=4,
                       children_per_split=(3, 5),
                       apex_x_fraction=0.25,
                       progradation_fraction=0.0,
                       trunk_length_factor=5.5,
                       length_taper=0.88, width_taper=0.80,
                       meander_amp=0.04,
                       mouth_bar_length_factor=2.2,
                       mouth_bar_width_factor=1.6,
                       mouth_bar_hw_ratio=0.06,
                       mouth_bar_dw_ratio=0.08,
                       dwratio=0.25, poro_ave=0.25):
        """Generate a prograding delta.

        Parameters
        ----------
        feeder_width : float
            Channel width of the trunk feeder (same units as x_len).
            Distributary widths taper from this by ``width_taper`` at
            each bifurcation level.
        n_generations : int
            Number of stacked aggradation levels (delta generations).
            Each generation places the apex at a progressively larger
            x while ``chelev`` advances by one channel depth.
        fan_angle_deg : float
            Total angular spread of the leaf distributaries (default
            55°).  The fan opens symmetrically about +x from the apex.
        bifurcation_depth : int
            Levels of n-ary bifurcation.  Leaf count ≈
            ``mean(children_per_split) ** bifurcation_depth``.
        children_per_split : (int, int)
            Inclusive range for the number of children at each
            bifurcation, sampled uniformly per node.  Default (3, 5)
            gives a dense multi-channel fan instead of a sparse binary
            tree.
        apex_x_fraction : float
            Fractional x position of the apex in the *first* generation,
            relative to ``x_len``.  Default 0.25 puts the apex a quarter
            of the way in so the feeder occupies the proximal fifth.
        progradation_fraction : float
            Total fractional x distance over which the apex migrates
            across all ``n_generations``.  Default 0.0 keeps the apex
            fixed so every generation produces the same-sized fan at
            the same location (no clinoform).  Set > 0 to enable
            prograding-apex clinoform architecture.
        trunk_length_factor : float
            Root segment length expressed as a multiple of
            ``feeder_width``.  Subsequent levels shrink by
            ``length_taper``.
        length_taper, width_taper : float
            Per-level taper factors for segment length and channel
            width.  Defaults from field-compilation delta-plain geometry.
        meander_amp : float
            Per-segment sinusoidal perpendicular wiggle amplitude as a
            fraction of segment length.
        mouth_bar_length_factor, mouth_bar_width_factor : float
            Mouth-bar along-axis length and peak half-width relative to
            the *leaf* distributary width.
        mouth_bar_hw_ratio, mouth_bar_dw_ratio : float
            Height above / depth below the channel datum at the bar
            axis, per unit half-width (dimensionless).  Defaults from
            Alluvsim splay presets.
        dwratio : float
            Channel depth-to-width ratio (default 0.25, intermediate
            between meandering 0.4 and braided 0.1).
        poro_ave : float
            Reference porosity of channel + mouth-bar sand.
        """
        rng = np.random.default_rng()
        facies = np.zeros((self.nx, self.ny, self.nz), dtype=float)
        poro = 0.05 * np.ones((self.nx, self.ny, self.nz), dtype=float)
        x_grid = np.linspace(self.dx / 2, self.nx * self.dx - self.dx / 2, self.nx)
        y_grid = np.linspace(self.dy / 2, self.ny * self.dy - self.dy / 2, self.ny)
        x_len = self.nx * self.dx
        y_mid = 0.5 * self.ny * self.dy

        fan_angle_rad = np.deg2rad(fan_angle_deg)
        trunk_length = trunk_length_factor * feeder_width
        channel_depth = dwratio * feeder_width
        base_chelev = channel_depth + self.dz

        apex_x0 = apex_x_fraction * x_len
        apex_xN = (apex_x_fraction + progradation_fraction) * x_len
        apex_xs = np.linspace(apex_x0, apex_xN, n_generations)
        chelev_span = max(self.nz * self.dz - 2 * channel_depth - 2 * self.dz,
                          channel_depth)
        chelev_list = base_chelev + np.linspace(0.0, chelev_span, n_generations)

        leaf_width_final = feeder_width * (width_taper ** bifurcation_depth)
        MB_L = mouth_bar_length_factor * leaf_width_final * 2.0
        MB_W = mouth_bar_width_factor * leaf_width_final

        step_len = 0.5 * (self.dx + self.dy)

        for gen in range(n_generations):
            apex_x = float(apex_xs[gen])
            chelev = float(chelev_list[gen])
            # Jitter the apex laterally by up to ±10% of the fan's
            # half-width at depth 0 so successive generations don't
            # stack onto the exact same Y.
            apex_y = y_mid + rng.uniform(-0.1, 0.1) * trunk_length

            feeder_xs, feeder_ys = _perturbed_centerline(
                x0=max(self.dx, 0.5 * self.dx),
                y0=y_mid + rng.uniform(-0.05, 0.05) * y_mid,
                heading=np.arctan2(apex_y - y_mid, apex_x - 0.5 * self.dx),
                length=np.hypot(apex_x - 0.5 * self.dx, apex_y - y_mid),
                n_nodes=max(40, int(apex_x / step_len)),
                meander_amp=meander_amp * 0.6, rng=rng,
            )
            _stamp_centerline(
                feeder_xs, feeder_ys, feeder_width, chelev,
                self.nx, self.ny, self.nz, self.dx, self.dy, self.dz,
                x_grid, y_grid, dwratio, poro_ave, facies, poro,
            )

            segments = _build_distributary_tree(
                apex_x=apex_x, apex_y=apex_y,
                fan_angle_rad=fan_angle_rad,
                max_depth=bifurcation_depth,
                trunk_length=trunk_length,
                feeder_width=feeder_width,
                step_len=step_len,
                length_taper=length_taper,
                width_taper=width_taper,
                meander_amp=meander_amp,
                children_per_split=children_per_split,
                rng=rng,
            )
            for seg in segments:
                _stamp_centerline(
                    seg['xs'], seg['ys'], seg['width'], chelev,
                    self.nx, self.ny, self.nz, self.dx, self.dy, self.dz,
                    x_grid, y_grid, dwratio, poro_ave, facies, poro,
                )
                if seg['is_leaf']:
                    _paint_mouth_bar(
                        facies, poro, float(seg['xs'][-1]),
                        float(seg['ys'][-1]), seg['heading_end'],
                        MB_L, MB_W, mouth_bar_hw_ratio, mouth_bar_dw_ratio,
                        chelev, self.dx, self.dy, self.dz,
                        x_grid, y_grid, self.nz, poro_ave,
                    )

        self._finalize_properties(poro, facies, poro_ave)
