import numpy as np

from .base import Layer

__all__ = ["ChannelLayerBase", "MeanderingChannelLayer", "BraidedChannelLayer"]


def _rotate_xy_to_azimuth(arr, azimuth_deg):
    """Rotate a (nx, ny, nz) array in the XY plane to a compass azimuth.

    The fluvial streamline engine always flows along +x.  To point the
    channel belt along ``azimuth_deg`` using the same compass
    convention as ``DeltaLayer`` (0°=+x, 45°=+x,-y, 90°=-y, 135°=-x,-y,
    180°=-x, 225°=-x,+y, 270°=+y, 315°=+x,+y — CW from +x per
    ``extra/azimuth.jpg``), we rotate the output's XY plane by
    ``-azimuth_deg`` (scipy uses CCW-positive convention).

    ``order=0`` (nearest-neighbour) preserves integer facies values
    and keeps facies/poro aligned after resampling.  Cells outside
    the rotated support are filled with 0 (inactive/shale).
    """
    if float(azimuth_deg) % 360.0 == 0.0:
        return arr
    from scipy.ndimage import rotate
    return rotate(arr, -float(azimuth_deg), axes=(0, 1),
                  reshape=False, order=0, mode='constant', cval=0.0)


class ChannelLayerBase(Layer):
    """Base class for channel-type geological layers."""

    def _finalize_properties(self, engine_poro, engine_facies, poro_ave,
                             azimuth=0.0):
        """Convert engine output arrays into standard Layer properties.

        Shared by all channel subclasses to ensure consistent
        poro_mat, facies, active, perm_mat derivation.  When
        ``azimuth != 0``, rotates the XY plane of the engine output so
        the channel belt points along the requested compass azimuth
        (same convention as ``DeltaLayer``).
        """
        poro_rot = _rotate_xy_to_azimuth(engine_poro, azimuth)
        facies_rot = _rotate_xy_to_azimuth(engine_facies, azimuth)
        self.poro_ave = poro_ave
        self.poro_mat = poro_rot
        self.facies = facies_rot.astype(int)
        self.active = (self.facies > 0).astype(int)
        self.perm_mat = 10.0 * np.exp(20.0 * self.poro_mat) * self.active


class MeanderingChannelLayer(ChannelLayerBase):
    """Layer with meandering fluvial channel geology."""

    def create_geology(self, channel_width, n_channels,
                       depth_width_ratio=0.4, friction_coeff=0.0009,
                       amplitude=10.0, slope=0.008, discharge=0.9,
                       meander_scale=1.2, avulsion_prob=0, poro_ave=0.3,
                       migration_distance_ratio=1.0, boundary_reflect=True,
                       aggradation_mode='discrete', nlevel=6,
                       level_reseed_prob=0.6, level_jump_ratio=1.0,
                       prob_avul_inside=0.0, prob_avul_outside=0.0,
                       azimuth=0.0):
        """Generate meandering channel geology.

        Parameters
        ----------
        channel_width : float
            Half-width of channel belt (same units as x_len/y_len).
        n_channels : int
            Number of channel generations.
        depth_width_ratio : float
            Channel depth-to-width ratio.
        friction_coeff : float
            Friction coefficient (Cf).
        amplitude : float
            Secondary flow amplitude (A). Controls lateral bank erosion
            that drives channel migration during the simulation.
        slope : float
            Channel slope (I).
        discharge : float
            Normalized discharge (Q).
        meander_scale : float
            Controls sinuosity of the initial channel path.  Internally
            mapped as ``s = 0.2 * meander_scale**2`` before being used
            as noise amplitude in the streamline generator.
            0 = perfectly straight, ~0.5 = mild meanders,
            ~1.0 = moderate meanders, ~2.0 = very high sinuosity
            (legacy hard-coded s=0.8).  Default 1.2.
        avulsion_prob : float
            Probability of avulsion (0 to 1).
        poro_ave : float
            Reference porosity for channel fill (default 0.3).
        migration_distance_ratio : float
            Per-step lateral migration distance expressed as a multiple of
            (dx + dy).  Analogous to Alluvsim's ``mdistMigrate``.  Small
            values (~0.3-0.7) give Z-stable, near-vertical channels.
            Larger values (~2.0-3.0) reproduce the legacy slithering
            behaviour.  Default 1.0.
        boundary_reflect : bool
            When the migrating streamline approaches a Y-boundary, reflect
            the next streamline's initial Y position across the grid centre
            instead of re-seeding at mid-Y.  Preserves Z-continuity of the
            channel belt across boundary-touch events.  Default True.
        aggradation_mode : {'discrete', 'continuous'}
            ``'discrete'`` (default) — Alluvsim-style aggradation: the
            streamline migrates for ``N/nlevel`` iterations at a fixed
            elevation, a single snapshot is stamped, ``chelev`` jumps by
            one channel depth, and the streamline is optionally re-seeded
            for the next level.  Produces crisp per-Z meanders matching
            Alluvsim's pv_shoestring reference.

            ``'continuous'`` — legacy GeoRules behaviour: ``chelev``
            advances by a tiny amount every iteration and snapshots are
            stamped every 10 iterations, causing ~80 overlaid migration
            snapshots to amalgamate into a single belt in ~12 Z cells.
        nlevel : int
            Number of discrete aggradation levels (``aggradation_mode=
            'discrete'`` only).  Default 6.  Cf. Alluvsim's
            pv_shoestring (nlevel=5) and cb_jigsaw (nlevel=4).
        level_reseed_prob : float
            Probability of drawing a fresh streamline at a fresh random Y
            between successive levels (``aggradation_mode='discrete'``
            only).  0 keeps the same streamline migrating continuously
            across level boundaries; 1 gives an independent meander per
            level.  Default 0.6.
        level_jump_ratio : float
            Fraction of one channel depth by which ``chelev`` jumps
            between levels (``aggradation_mode='discrete'`` only).
            ``1.0`` (default) stacks stamps edge-to-edge, matching
            Alluvsim's pv_shoestring architecture.  ``<1.0`` makes stamps
            overlap vertically; combined with ``level_reseed_prob=0`` and
            many levels this produces a single continuous channel belt
            that migrates slowly from z=0 to the top.
        prob_avul_inside : float
            Per-iteration probability of Alluvsim-style in-model
            avulsion (``avulsioninside.for``): pick a node weighted by
            ``|curvature|``, stamp the current path, then regrow a
            fresh tail from that node's local azimuth.  Default 0
            (pure meandering).  Larger values (~0.5) drive braided
            architectures — see ``BraidedChannelLayer``.
        prob_avul_outside : float
            Per-iteration probability of fully reseeding the streamline
            at a random Y.  Default 0.
        azimuth : float
            Compass-convention flow direction of the channel belt in
            degrees (CW from +x, per ``extra/azimuth.jpg`` — same
            convention as ``DeltaLayer``):
              0°   → +x (default, native engine direction)
              45°  → +x,-y    90°  → -y       135° → -x,-y
              180° → -x       225° → -x,+y    270° → +y
              315° → +x,+y
            Implemented as a nearest-neighbour XY rotation of the
            engine's 3D output, so channel physics (migration,
            avulsion, aggradation) are unchanged — only the belt
            heading changes.  Cells outside the rotated support are
            marked inactive.
        """
        from ._fluvial import fluvial

        engine = fluvial(
            b=channel_width,
            nx=self.nx, ny=self.ny, nz=self.nz,
            xmn=self.dx / 2, ymn=self.dy / 2,
            xsiz=self.dx, ysiz=self.dy, zsiz=self.dz,
            dwratio=depth_width_ratio,
            Cf=friction_coeff, A=amplitude, I=slope, Q=discharge,
            meander_scale=meander_scale, pavul=avulsion_prob,
            migration_distance_ratio=migration_distance_ratio,
            boundary_reflect=boundary_reflect,
            aggradation_mode=aggradation_mode, nlevel=nlevel,
            level_reseed_prob=level_reseed_prob,
            level_jump_ratio=level_jump_ratio,
            prob_avul_inside=prob_avul_inside,
            prob_avul_outside=prob_avul_outside,
        )
        engine.simulation(nchannel=n_channels)
        self._finalize_properties(engine.poro, engine.facies, poro_ave,
                                  azimuth=azimuth)


class BraidedChannelLayer(MeanderingChannelLayer):
    """Braided fluvial channels — ``MeanderingChannelLayer`` with braided presets.

    Same ``fluvial`` streamline engine as the parent class, just
    configured per Alluvsim's CB-jigsaw / SH-proximal braided presets:
    moderate sinuosity (meander_scale~0.9), shallow/wide cross-section
    (depth_width_ratio~0.1), aggressive avulsion
    (prob_avul_inside=0.55, prob_avul_outside=0.30).  Avulsions
    truncate the current streamline at a |curvature|-weighted node
    and regrow a fresh tail; out-of-model avulsions re-seed the
    streamline at a random Y.  Stamps accumulate abandoned fragments,
    producing the classic braided interwoven-threads appearance.

    Cf. Pyrcz (2004) avulsioninside.for + streamsim.for main loop.
    """

    def create_geology(self, braidplain_width, n_channels=24,
                       depth_width_ratio=0.1,
                       meander_scale=0.9,
                       migration_distance_ratio=0.3,
                       prob_avul_inside=0.55, prob_avul_outside=0.30,
                       nlevel=None, level_reseed_prob=0.4,
                       level_jump_ratio=0.5,
                       slope=0.008, discharge=0.9,
                       amplitude=10.0, friction_coeff=0.0009,
                       poro_ave=0.3, azimuth=0.0,
                       n_threads=None, thread_width=None,
                       bar_poro_factor=None):
        """Generate braided channel geology.

        All parameters are forwarded to
        ``MeanderingChannelLayer.create_geology``; defaults below are
        the braided presets.

        Parameters
        ----------
        braidplain_width : float
            Full width of a single thread (maps to parent's
            ``channel_width``).  Narrow gives wide interwoven fragments;
            wide gives a few wandering channels.
        n_channels : int
            Number of channel generations (default 24).
        depth_width_ratio : float
            Channel depth-to-width ratio (default 0.1, shallower than
            meandering 0.4 — Alluvsim braided presets use ~0.05-0.08).
        meander_scale : float
            Sinuosity of each streamline (default 0.9).  Alluvsim's
            literal braided preset is ~0.35, but at small grid
            resolutions that reads as nearly straight.
        migration_distance_ratio : float
            Per-step lateral migration (default 0.3 — braided channels
            avulse more than they migrate).
        prob_avul_inside : float
            Per-iter probability of in-model avulsion (default 0.55).
        prob_avul_outside : float
            Per-iter probability of full streamline reseed at a random
            Y (default 0.30 — higher than Alluvsim's CB-jigsaw preset
            of ~0.12 so multiple concurrent threads are visible per
            XY slice at typical grid resolutions).
        nlevel : int or None
            Discrete aggradation levels.  When None (default), chosen
            so that ``nlevel * level_jump_ratio * channel_depth``
            covers the full grid Z range.
        level_reseed_prob : float
            Probability of re-drawing at random Y between levels
            (default 0.4).
        level_jump_ratio : float
            Fraction of channel depth per level jump (default 0.5 —
            half-depth overlap between levels).
        slope, discharge, amplitude, friction_coeff : float
            Sun-1996 flow parameters.
        poro_ave : float
            Reference porosity for channel fill (default 0.3).
        azimuth : float
            Compass-convention flow direction in degrees (CW from +x,
            per ``extra/azimuth.jpg``).  Forwarded to
            ``MeanderingChannelLayer.create_geology``; see its
            docstring for the full azimuth → entry/flow mapping.
            Default 0 (native +x-flowing engine).
        n_threads, thread_width, bar_poro_factor :
            Deprecated — no-ops retained for API backwards compatibility
            with the old BBC engine.  Ignored.
        """
        if nlevel is None:
            cz_est = max(1, int(depth_width_ratio * braidplain_width
                                 / self.dz))
            step_cells = max(1.0, level_jump_ratio * cz_est)
            nlevel = max(6, int(np.ceil(self.nz / step_cells)))

        super().create_geology(
            channel_width=braidplain_width,
            n_channels=n_channels,
            depth_width_ratio=depth_width_ratio,
            friction_coeff=friction_coeff,
            amplitude=amplitude,
            slope=slope,
            discharge=discharge,
            meander_scale=meander_scale,
            poro_ave=poro_ave,
            migration_distance_ratio=migration_distance_ratio,
            boundary_reflect=True,
            aggradation_mode='discrete',
            nlevel=nlevel,
            level_reseed_prob=level_reseed_prob,
            level_jump_ratio=level_jump_ratio,
            prob_avul_inside=prob_avul_inside,
            prob_avul_outside=prob_avul_outside,
            azimuth=azimuth,
        )

