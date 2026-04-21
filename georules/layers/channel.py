import numpy as np

from .base import Layer


class ChannelLayerBase(Layer):
    """Base class for channel-type geological layers."""

    def _finalize_properties(self, engine_poro, engine_facies, poro_ave):
        """Convert engine output arrays into standard Layer properties.

        Shared by all channel subclasses to ensure consistent
        poro_mat, facies, active, perm_mat derivation.
        """
        self.poro_ave = poro_ave
        self.poro_mat = engine_poro
        self.facies = engine_facies.astype(int)
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
                       level_reseed_prob=0.6, level_jump_ratio=1.0):
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
        )
        engine.simulation(nchannel=n_channels)
        self._finalize_properties(engine.poro, engine.facies, poro_ave)


class BraidedChannelLayer(ChannelLayerBase):
    """Layer with braided fluvial channel geology."""

    def create_geology(self, braidplain_width, n_channels, n_threads=3,
                       thread_width=None, depth_width_ratio=0.15,
                       slope=0.008, discharge=0.9,
                       bar_poro_factor=0.7, poro_ave=0.3):
        """Generate braided channel geology.

        Parameters
        ----------
        braidplain_width : float
            Total width of the braided belt (same units as x_len/y_len).
        n_channels : int
            Number of channel generations (aggradation steps).
        n_threads : int
            Number of simultaneous channel threads per generation.
        thread_width : float or None
            Half-width of each individual thread. If None,
            auto-calculated as braidplain_width / (2 * n_threads).
        depth_width_ratio : float
            Individual thread depth-to-width ratio (default 0.15,
            shallower than meandering default of 0.4).
        slope : float
            Channel slope.
        discharge : float
            Normalized discharge.
        bar_poro_factor : float
            Bar porosity as fraction of poro_ave (default 0.7).
        poro_ave : float
            Reference porosity for channel fill (default 0.3).
        """
        from ._braided import braided

        engine = braided(
            braidplain_width=braidplain_width,
            n_threads=n_threads,
            thread_width=thread_width,
            nx=self.nx, ny=self.ny, nz=self.nz,
            xmn=self.dx / 2, ymn=self.dy / 2,
            xsiz=self.dx, ysiz=self.dy, zsiz=self.dz,
            dwratio=depth_width_ratio,
            I=slope, Q=discharge,
            bar_poro_factor=bar_poro_factor,
        )
        engine.simulation(n_channels=n_channels)
        self._finalize_properties(engine.poro, engine.facies, poro_ave)

