"""Channel layer classes — Alluvsim-faithful fluvial reservoir generation.

A single engine (``_fluvial.fluvial``) drives all reservoir architectures.
The same parameter space that produces Alluvsim's PV-shoestring,
CB-jigsaw, CB-labyrinth, SH-distal and SH-proximal architectures is exposed
as kwargs on :py:meth:`MeanderingChannelLayer.create_geology` —
``BraidedChannelLayer`` is a thin subclass that picks the CB-jigsaw
defaults.

The engine internally tracks all six Alluvsim facies
(FF=-1, FFCH=0, CS=1, LV=2, LA=3, CH=4); the public output can be either
the full 6-class array (``output_facies='alluvsim'``) or a collapsed
binary 0/1 sand mask (``output_facies='binary'``, default — back-compat
with the 10M-reservoir dataset pipeline). Per-facies porosity and
permeability are pulled from a small constant table (``FACIES_PROPS``).

Importable parameter presets ``PV_SHOESTRING``, ``CB_JIGSAW``,
``CB_LABYRINTH``, ``SH_DISTAL``, ``SH_PROXIMAL`` mirror Alluvsim's
``runs/run_presets.py`` so users can call::

    layer = MeanderingChannelLayer(nx=64, ny=64, nz=32, x_len=640,
                                   y_len=640, z_len=32, top_depth=0.0)
    layer.create_geology(**PV_SHOESTRING)
"""
from __future__ import annotations

import numpy as np

from .base import Layer

__all__ = [
    "ChannelLayerBase", "MeanderingChannelLayer", "BraidedChannelLayer",
    "FACIES_PROPS",
    "PV_SHOESTRING", "CB_JIGSAW", "CB_LABYRINTH", "SH_DISTAL", "SH_PROXIMAL",
]


# ---------------------------------------------------------------------------
# Per-facies poro / log10(perm in mD) lookup table.
#
# Values pick a monotonic CH > LA > LV > CS > FFCH > FF ordering with a
# ~3-decade perm range, typical of fluvial reservoir characterisations.
# Override per-call via the ``facies_props`` kwarg on create_geology.
# ---------------------------------------------------------------------------
FACIES_PROPS: dict[int, dict[str, float]] = {
    -1: {"poro": 0.05, "log10_perm": -1.0},  # FF      overbank fines
     0: {"poro": 0.08, "log10_perm":  0.0},  # FFCH   abandoned mud plug
     1: {"poro": 0.18, "log10_perm":  1.5},  # CS     crevasse splay
     2: {"poro": 0.20, "log10_perm":  2.0},  # LV     levee
     3: {"poro": 0.25, "log10_perm":  2.7},  # LA     lateral accretion
     4: {"poro": 0.30, "log10_perm":  3.3},  # CH     active channel
}


class ChannelLayerBase(Layer):
    """Base class for channel-type geological layers."""

    def _finalize_properties(self, engine_poro, engine_facies, poro_ave):
        """Legacy 3-arg signature retained for ``DeltaLayer`` back-compat.

        Channel-engine derived layers (``MeanderingChannelLayer``,
        ``BraidedChannelLayer``) use :py:meth:`_finalize_facies_table`
        instead, which assigns properties from a per-facies lookup
        rather than the engine's per-cell U-shape porosity field.
        """
        self.poro_ave = poro_ave
        self.poro_mat = engine_poro
        self.facies = engine_facies.astype(int)
        self.active = (self.facies > 0).astype(int)
        self.perm_mat = 10.0 * np.exp(20.0 * self.poro_mat) * self.active

    def _finalize_facies_table(self, engine_facies: np.ndarray,
                               output_facies: str = "binary",
                               facies_props: dict | None = None):
        """Convert engine 6-class facies into ``self.facies / poro_mat / perm_mat / active``.

        Engine always emits the full 6-class Alluvsim facies on the input
        ``engine_facies`` array. This method:

        * always populates ``self.facies_alluvsim`` with the 6-class array;
        * if ``output_facies='alluvsim'``, leaves ``self.facies`` as the
          full -1..4 codes;
        * if ``output_facies='binary'`` (default), collapses
          {-1, 0} → 0 (shale: FF + FFCH) and {1, 2, 3, 4} → 1 (sand:
          CS + LV + LA + CH), matching Alluvsim's NTG accounting;
        * computes per-cell ``poro_mat`` and ``perm_mat`` from
          ``FACIES_PROPS`` (so a binary-mode reservoir still has internally
          varying poro/perm reflecting which sand sub-element each sand
          cell actually was).
        """
        props = dict(FACIES_PROPS)
        if facies_props:
            for k, v in facies_props.items():
                props.setdefault(int(k), {}).update(v)

        self.facies_alluvsim = engine_facies.astype(np.int8)

        # Per-cell base poro / perm from the per-facies lookup table.
        poro_lut = np.zeros(self.facies_alluvsim.shape, dtype=np.float32)
        for code, vals in props.items():
            mask = (self.facies_alluvsim == code)
            poro_lut[mask] = vals["poro"]

        # Upward-fining porosity within each sand column. Real channels
        # have coarse high-poro sand at the thalweg/base grading to
        # finer lower-poro sand toward the top — the classic point-bar
        # vertical sequence (Walker 1992). Apply a depth-from-local-top
        # ramp on CS/LV/LA/CH cells so a single CH cell at the bottom of
        # the U-shape gets ~0.32 porosity while one at the top gets ~0.20.
        #
        # Algorithm: for each (ix, iy) column, mark each sand cell with
        # its "depth from sand top" (number of sand cells above it of any
        # sand type). Then poro = base * (0.7 + 0.6 * depth_norm), where
        # depth_norm = depth / max_depth in that column, clipped [0, 1].
        # Net effect: poro varies ±30% from base around its facies value,
        # bottom > top — geologically correct.
        sand_mask = (self.facies_alluvsim >= 1)
        nx_, ny_, nz_ = self.facies_alluvsim.shape
        # Per (ix, iy) column: depth-from-sand-top in z-cell units.
        # For each column, count sand cells from top down: if top sand at
        # iz_top, then a cell at iz has depth = iz_top - iz (downward
        # positive). Normalize by the column's sand-z-extent.
        depth_norm = np.zeros_like(poro_lut)
        for ix in range(nx_):
            for iy in range(ny_):
                col = sand_mask[ix, iy, :]
                if not col.any():
                    continue
                iz_top = int(np.where(col)[0].max())
                iz_bot = int(np.where(col)[0].min())
                ext = max(iz_top - iz_bot, 1)
                # depth from top, normalized 0..1 (top=0, bottom=1)
                for iz in range(iz_bot, iz_top + 1):
                    if col[iz]:
                        depth_norm[ix, iy, iz] = (iz_top - iz) / ext
        # Apply ramp: 0.7× at top, 1.3× at bottom of the sand column.
        ramp = 0.7 + 0.6 * depth_norm
        poro_lut = poro_lut * ramp.astype(np.float32)
        # FF cells unchanged (depth_norm=0 there → ramp=0.7 → 0.7*0.05=0.035;
        # restore to base for non-sand cells).
        poro_lut[~sand_mask] = 0.0
        for code in (-1, 0):
            mask = (self.facies_alluvsim == code)
            if mask.any() and code in props:
                poro_lut[mask] = props[code]["poro"]
        self.poro_mat = poro_lut

        # Permeability: log10(perm) varies linearly with porosity within
        # each facies (Kozeny-Carman style). Use the per-cell poro to
        # interpolate perm: at base poro → base perm, at base*1.3 → 10× perm.
        log_perm = np.zeros_like(poro_lut)
        for code, vals in props.items():
            mask = (self.facies_alluvsim == code)
            if not mask.any():
                continue
            base_poro = float(vals["poro"])
            base_log_perm = float(vals["log10_perm"])
            # log(perm) = base + 4 * (poro - base_poro) / base_poro
            # i.e. ±30% poro variation → ±1.2 in log10(perm) → ~16× variation
            ratio = (poro_lut[mask] - base_poro) / max(base_poro, 1e-6)
            log_perm[mask] = base_log_perm + 4.0 * ratio
        self.perm_mat = (10.0 ** log_perm).astype(np.float32)

        if output_facies == "alluvsim":
            self.facies = self.facies_alluvsim.copy()
            self.active = (self.facies >= 1).astype(np.int8)
        elif output_facies == "binary":
            self.facies = (self.facies_alluvsim >= 1).astype(np.int8)
            self.active = self.facies.astype(np.int8)
        else:
            raise ValueError(
                f"output_facies={output_facies!r}; expected 'binary' or 'alluvsim'")


class MeanderingChannelLayer(ChannelLayerBase):
    """Meandering fluvial channel layer — full Alluvsim parameter set.

    The defaults below produce a PV-shoestring-style reservoir on a
    typical 64-128 cell grid; pass ``**CB_JIGSAW`` or another preset for
    a different architecture, or override individual params.
    """

    def create_geology(
        self,
        # ---- aggradation -----------------------------------------------
        nlevel: int = 3,
        level_z: list[float] | None = None,
        NTGtarget: float = 0.10,
        ntime: int = 120,
        # ---- avulsion --------------------------------------------------
        probAvulOutside: float = 0.10, probAvulInside: float = 0.05,
        # ---- channel geometry ------------------------------------------
        mCHdepth: float = 2.5, stdevCHdepth: float = 0.3, stdevCHdepth2: float = 0.2,
        mCHwdratio: float = 10.0, stdevCHwdratio: float = 1.0,
        mCHsinu: float = 1.6, stdevCHsinu: float = 0.15,
        mCHazi: float = 90.0, stdevCHazi: float = 1.0,
        mCHsource: float | None = None, stdevCHsource: float = 80.0,
        # ---- migration --------------------------------------------------
        mdistMigrate: float = 35.0, stdevdistMigrate: float = 10.0,
        # ---- levee (LV) — Alluvsim makepar central values --------------
        mLVdepth: float = 1.0, stdevLVdepth: float = 0.2,
        mLVwidth: float = 40.0, stdevLVwidth: float = 5.0,
        mLVheight: float = 0.5, stdevLVheight: float = 0.1,
        mLVasym: float = 0.3, stdevLVasym: float = 0.1,
        mLVthin: float = 0.3, stdevLVthin: float = 0.1,
        # ---- crevasse splay (CS) — makepar defaults --------------------
        mCSnum: float = 2.0, stdevCSnum: float = 0.5,
        mCSnumlobe: float = 3.0, stdevCSnumlobe: float = 1.0,
        mCSsource: float = 50.0, stdevCSsource: float = 20.0,
        mCSLOLL: float = 200.0, stdevCSLOLL: float = 50.0,
        mCSLOWW: float = 30.0, stdevCSLOWW: float = 10.0,
        mCSLOl: float = 100.0, stdevCSLOl: float = 20.0,
        mCSLOw: float = 20.0, stdevCSLOw: float = 10.0,
        mCSLO_hwratio: float = 0.03, stdevCSLO_hwratio: float = 0.01,
        mCSLO_dwratio: float = 0.02, stdevCSLO_dwratio: float = 0.005,
        # ---- abandoned-channel fill (FFCH) -----------------------------
        mFFCHprop: float = 0.0, stdevFFCHprop: float = 0.0,
        # ---- hydraulic — Alluvsim makepar central values ---------------
        Cf: float = 0.0078, scour_factor: float = 2.0,
        gradient: float = 0.001, Q: float = 5.0,
        # ---- pool / discretisation -------------------------------------
        CHndraw: int = 50, ndiscr: int = 5, nCHcor: int = 10,
        # ---- presentation ----------------------------------------------
        azimuth: float = 0.0,
        output_facies: str = "binary",
        facies_props: dict | None = None,
        seed: int | None = None,
    ):
        """Generate channel geology with Alluvsim-faithful semantics.

        All ``mFoo`` / ``stdevFoo`` kwargs are direct ports of Alluvsim's
        per-event Gaussian-draw parameters (``streamsim.par`` field set);
        see ``/home/ilgar/Alluvsim/CLAUDE.md`` §4 for full docs.

        Notable mappings:

        * ``mCHwdratio`` is Alluvsim's full width / depth ratio (so 10
          means a 10:1 W:D channel). GeoRules converts internally to
          ``depth / half_width``.
        * Default kwargs reproduce a PV-shoestring-ish reservoir on a
          typical 64-cell grid; override individual params or pass
          ``**PV_SHOESTRING`` / ``**CB_JIGSAW`` / etc. for canonical
          architectures.
        * ``output_facies='binary'`` (default) → ``self.facies`` is 0/1
          (shale=FF+FFCH, sand=CS+LV+LA+CH). ``'alluvsim'`` → full
          -1..4 codes. ``self.facies_alluvsim`` always holds the full
          codes.
        """
        from ._fluvial import fluvial

        engine = fluvial(
            nx=self.nx, ny=self.ny, nz=self.nz,
            xsiz=self.dx, ysiz=self.dy, zsiz=self.dz,
            xmn=self.dx / 2, ymn=self.dy / 2,
            nlevel=nlevel, level_z=level_z,
            NTGtarget=NTGtarget, ntime=ntime,
            probAvulOutside=probAvulOutside, probAvulInside=probAvulInside,
            mCHdepth=mCHdepth, stdevCHdepth=stdevCHdepth, stdevCHdepth2=stdevCHdepth2,
            mCHwdratio=mCHwdratio, stdevCHwdratio=stdevCHwdratio,
            mCHsinu=mCHsinu, stdevCHsinu=stdevCHsinu,
            mCHazi=mCHazi, stdevCHazi=stdevCHazi,
            mCHsource=mCHsource, stdevCHsource=stdevCHsource,
            mdistMigrate=mdistMigrate, stdevdistMigrate=stdevdistMigrate,
            mLVdepth=mLVdepth, stdevLVdepth=stdevLVdepth,
            mLVwidth=mLVwidth, stdevLVwidth=stdevLVwidth,
            mLVheight=mLVheight, stdevLVheight=stdevLVheight,
            mLVasym=mLVasym, stdevLVasym=stdevLVasym,
            mLVthin=mLVthin, stdevLVthin=stdevLVthin,
            mCSnum=mCSnum, stdevCSnum=stdevCSnum,
            mCSnumlobe=mCSnumlobe, stdevCSnumlobe=stdevCSnumlobe,
            mCSsource=mCSsource, stdevCSsource=stdevCSsource,
            mCSLOLL=mCSLOLL, stdevCSLOLL=stdevCSLOLL,
            mCSLOWW=mCSLOWW, stdevCSLOWW=stdevCSLOWW,
            mCSLOl=mCSLOl, stdevCSLOl=stdevCSLOl,
            mCSLOw=mCSLOw, stdevCSLOw=stdevCSLOw,
            mCSLO_hwratio=mCSLO_hwratio, stdevCSLO_hwratio=stdevCSLO_hwratio,
            mCSLO_dwratio=mCSLO_dwratio, stdevCSLO_dwratio=stdevCSLO_dwratio,
            mFFCHprop=mFFCHprop, stdevFFCHprop=stdevFFCHprop,
            Cf=Cf, A=scour_factor, I=gradient, Q=Q,
            CHndraw=CHndraw, ndiscr=ndiscr, nCHcor=nCHcor,
            azimuth=azimuth, seed=seed,
        )
        engine.simulation()
        self._finalize_facies_table(
            engine.facies, output_facies=output_facies, facies_props=facies_props,
        )


class BraidedChannelLayer(MeanderingChannelLayer):
    """Braided fluvial channels — ``MeanderingChannelLayer`` with CB-jigsaw defaults.

    Same engine, just defaults that produce the dense interwoven
    multi-thread architecture: shallow wide channels, aggressive in-model
    avulsion, prominent FFCH abandonment.
    """

    def create_geology(self, **kwargs):
        defaults = dict(CB_JIGSAW)
        defaults.update(kwargs)
        super().create_geology(**defaults)


# ---------------------------------------------------------------------------
# Importable parameter presets (mirror /home/ilgar/Alluvsim/runs/run_presets.py).
#
# Use as `MeanderingChannelLayer.create_geology(**PV_SHOESTRING)` or pass
# individual overrides on top.
# ---------------------------------------------------------------------------
PV_SHOESTRING = dict(
    ntime=120, nlevel=5, level_z=[1.5, 3.0, 5.0, 7.0, 9.0],
    NTGtarget=0.10,
    probAvulOutside=0.10, probAvulInside=0.05,
    mCHsinu=1.6, stdevCHsinu=0.15,
    mCHwdratio=10.0, stdevCHwdratio=1.0,
    mCHdepth=2.5, stdevCHdepth=0.3,
    mLVdepth=0.6, stdevLVdepth=0.1,
    mLVwidth=40.0, stdevLVwidth=5.0,
    mLVheight=0.4, stdevLVheight=0.1,
    mLVasym=0.3, stdevLVasym=0.1,
    mFFCHprop=0.0, stdevFFCHprop=0.0,
    mdistMigrate=35.0, stdevdistMigrate=10.0,
    Cf=0.0036, scour_factor=10.0, gradient=0.001, Q=5.0,
)

CB_JIGSAW = dict(
    ntime=250, nlevel=4, level_z=[2.0, 4.0, 6.0, 8.0],
    NTGtarget=0.30,
    probAvulOutside=0.05, probAvulInside=0.40,
    mCHsinu=1.3, stdevCHsinu=0.1,
    mCHwdratio=16.0, stdevCHwdratio=2.0,
    mCHdepth=3.0, stdevCHdepth=0.3,
    mLVdepth=0.6, stdevLVdepth=0.1,
    mLVwidth=40.0, stdevLVwidth=10.0,
    mLVheight=0.4, stdevLVheight=0.1,
    mLVasym=0.3, stdevLVasym=0.1,
    mFFCHprop=0.5, stdevFFCHprop=0.15,
    mdistMigrate=25.0, stdevdistMigrate=8.0,
    Cf=0.0036, scour_factor=10.0, gradient=0.001, Q=5.0,
)

CB_LABYRINTH = dict(
    ntime=600, nlevel=6, level_z=[1.5, 3.0, 4.5, 6.0, 7.5, 9.0],
    NTGtarget=0.25,
    probAvulOutside=0.05, probAvulInside=0.05,
    mCHsinu=1.3, stdevCHsinu=0.1,
    mCHwdratio=16.0, stdevCHwdratio=2.0,
    mCHdepth=3.0, stdevCHdepth=0.3,
    mLVdepth=0.5, stdevLVdepth=0.1,
    mLVwidth=30.0, stdevLVwidth=5.0,
    mLVheight=0.3, stdevLVheight=0.1,
    mLVasym=0.3, stdevLVasym=0.1,
    mFFCHprop=0.4, stdevFFCHprop=0.15,
    mdistMigrate=25.0, stdevdistMigrate=8.0,
    Cf=0.0036, scour_factor=10.0, gradient=0.001, Q=5.0,
)

SH_DISTAL = dict(
    ntime=250, nlevel=5, level_z=[2.0, 4.0, 6.0, 8.0, 9.5],
    NTGtarget=0.50,
    probAvulOutside=0.02, probAvulInside=0.05,
    mCHsinu=1.3, stdevCHsinu=0.1,
    mCHwdratio=18.0, stdevCHwdratio=2.0,
    mCHdepth=5.0, stdevCHdepth=0.3,
    mLVdepth=1.2, stdevLVdepth=0.2,
    mLVwidth=150.0, stdevLVwidth=25.0,
    mLVheight=1.0, stdevLVheight=0.2,
    mLVasym=0.2, stdevLVasym=0.1,
    mFFCHprop=0.0, stdevFFCHprop=0.0,
    mdistMigrate=35.0, stdevdistMigrate=10.0,
    Cf=0.0036, scour_factor=10.0, gradient=0.001, Q=5.0,
)

SH_PROXIMAL = dict(
    ntime=300, nlevel=3, level_z=[2.5, 5.0, 7.5],
    NTGtarget=0.40,
    probAvulOutside=0.08, probAvulInside=0.35,
    mCHsinu=1.2, stdevCHsinu=0.1,
    mCHwdratio=22.0, stdevCHwdratio=3.0,
    mCHdepth=3.5, stdevCHdepth=0.3,
    mLVdepth=0.4, stdevLVdepth=0.1,
    mLVwidth=20.0, stdevLVwidth=5.0,
    mLVheight=0.2, stdevLVheight=0.1,
    mLVasym=0.0, stdevLVasym=0.0,
    mFFCHprop=0.15, stdevFFCHprop=0.05,
    mdistMigrate=25.0, stdevdistMigrate=8.0,
    Cf=0.0036, scour_factor=10.0, gradient=0.001, Q=5.0,
)
