"""Channel layer classes — Alluvsim-faithful fluvial reservoir generation.

A single engine (``_fluvial.fluvial``) drives all reservoir architectures.
The same parameter space that produces Alluvsim's PV-shoestring,
CB-jigsaw, CB-labyrinth, SH-distal and SH-proximal architectures is exposed
as kwargs on :py:meth:`MeanderingChannelLayer.create_geology` —
``BraidedChannelLayer`` is a thin subclass that picks the CB-jigsaw
defaults.

The engine internally tracks all six Alluvsim facies
(FF=-1, FFCH=0, CS=1, LV=2, LA=3, CH=4) and exposes them on
``self.facies`` (always the 6-class array). ``self.active`` is the
binary 0/1 sand mask (``self.facies >= 1``). Per-cell porosity and
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
    "MEANDER_OXBOW",
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

    def _finalize_facies_table(self, engine_facies: np.ndarray,
                               facies_props: dict | None = None):
        """Convert engine 6-class facies into ``self.facies / active / poro_mat / perm_mat``.

        ``self.facies`` is always the full Alluvsim 6-class array
        (codes -1..4). ``self.active`` is the binary 0/1 sand mask
        derived from it. Per-cell porosity and permeability come from
        the per-facies lookup table ``FACIES_PROPS`` plus a Walker-1992
        upward-fining ramp inside each sand column.
        """
        props = dict(FACIES_PROPS)
        if facies_props:
            for k, v in facies_props.items():
                props.setdefault(int(k), {}).update(v)

        self.facies = engine_facies.astype(np.int8)
        self.active = (self.facies >= 1).astype(np.int8)

        # Per-cell base poro / perm from the per-facies lookup table.
        poro_lut = np.zeros(self.facies.shape, dtype=np.float32)
        for code, vals in props.items():
            mask = (self.facies == code)
            poro_lut[mask] = vals["poro"]

        # Walker-1992 upward-fining ramp inside each sand column.
        # See git history of this file for the derivation; in short,
        # poro = base × (0.7 + 0.6 × depth_norm) so a CH cell at the
        # bottom of the U-shape gets ~0.32 and one at the top ~0.20.
        sand_mask = (self.facies >= 1)
        nx_, ny_, nz_ = self.facies.shape
        depth_norm = np.zeros_like(poro_lut)
        for ix in range(nx_):
            for iy in range(ny_):
                col = sand_mask[ix, iy, :]
                if not col.any():
                    continue
                iz_top = int(np.where(col)[0].max())
                iz_bot = int(np.where(col)[0].min())
                ext = max(iz_top - iz_bot, 1)
                for iz in range(iz_bot, iz_top + 1):
                    if col[iz]:
                        depth_norm[ix, iy, iz] = (iz_top - iz) / ext
        ramp = 0.7 + 0.6 * depth_norm
        poro_lut = poro_lut * ramp.astype(np.float32)
        poro_lut[~sand_mask] = 0.0
        for code in (-1, 0):
            mask = (self.facies == code)
            if mask.any() and code in props:
                poro_lut[mask] = props[code]["poro"]
        self.poro_mat = poro_lut

        # Permeability: log10(perm) varies linearly with porosity (Kozeny-
        # Carman-ish): ±30% poro variation → ±1.2 in log10(perm) ≈ 16×.
        log_perm = np.zeros_like(poro_lut)
        for code, vals in props.items():
            mask = (self.facies == code)
            if not mask.any():
                continue
            base_poro = float(vals["poro"])
            base_log_perm = float(vals["log10_perm"])
            ratio = (poro_lut[mask] - base_poro) / max(base_poro, 1e-6)
            log_perm[mask] = base_log_perm + 4.0 * ratio
        self.perm_mat = (10.0 ** log_perm).astype(np.float32)


class MeanderingChannelLayer(ChannelLayerBase):
    """Meandering fluvial channel layer — full Alluvsim parameter set.

    The defaults below produce a PV-shoestring-style reservoir on a
    typical 64-128 cell grid; pass ``**CB_JIGSAW`` or another preset for
    a different architecture, or override individual params.
    """

    def create_geology(
        self,
        # ---- aggradation -----------------------------------------------
        nlevel: int = 8,
        level_z: list[float] | None = None,
        NTGtarget: float = 0.10,
        ntime: int = 240,
        # ``True`` ⇒ ``ntime`` is interpreted per-level (counter resets
        # at every level); ``False`` ⇒ ``ntime`` is the total event cap
        # across all levels (Alluvsim default).
        ntime_per_level: bool = False,
        # ---- avulsion --------------------------------------------------
        probAvulOutside: float = 0.10, probAvulInside: float = 0.05,
        # ---- channel geometry ------------------------------------------
        mCHdepth: float = 4.0, stdevCHdepth: float = 0.4, stdevCHdepth2: float = 0.3,
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
        # ---- neck-cutoff oxbow → mud plug -----------------------------
        # 0.0 = pure Alluvsim (cutoff is geometric only, abandoned bend
        # keeps prior LA/CH stamps). >0 = each excised oxbow loop is
        # painted as an FFCH mud plug over the upper ``mNeckFFCHprop``
        # of the channel column at that node — gives the
        # neck-cutoff → oxbow lake → mud plug succession in cross-section.
        mNeckFFCHprop: float = 0.0,
        # ---- hydraulic — Alluvsim makepar central values ---------------
        Cf: float = 0.0078, scour_factor: float = 2.0,
        gradient: float = 0.001, Q: float = 5.0,
        # ---- pool / discretisation -------------------------------------
        CHndraw: int = 50, ndiscr: int = 5, nCHcor: int = 10,
        # ---- presentation ----------------------------------------------
        azimuth: float = 0.0,
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
        * Output: ``self.facies`` is the full Alluvsim 6-class array
          (-1..4); ``self.active`` is the binary 0/1 sand mask
          (``self.facies >= 1``).
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
            mNeckFFCHprop=mNeckFFCHprop,
            ntime_per_level=ntime_per_level,
            Cf=Cf, A=scour_factor, I=gradient, Q=Q,
            CHndraw=CHndraw, ndiscr=ndiscr, nCHcor=nCHcor,
            azimuth=azimuth, seed=seed,
        )
        engine.simulation()
        self._finalize_facies_table(engine.facies, facies_props=facies_props)


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
# NOTE on preset ``nlevel`` / ``level_z`` / ``mCHdepth`` choice
# -------------------------------------------------------------
# All presets ship **no explicit ``level_z``** — the engine spreads the
# requested ``nlevel`` levels evenly across the layer's ``z_len`` via
# ``np.linspace(zsiz, nz·zsiz, nlevel)``. ``mCHdepth`` is set to ~4 m
# so each channel is 4 cells thick on the standard ``dz=1 m`` grid
# (nicely visible in cross-section), and ``nlevel`` is chosen so chelev
# spacing ≈ channel depth — adjacent levels just touch, and the column
# fills continuously without hand-tuning ``level_z`` per grid.
PV_SHOESTRING = dict(
    ntime=240, nlevel=8,
    NTGtarget=0.10,
    probAvulOutside=0.10, probAvulInside=0.05,
    mCHsinu=1.6, stdevCHsinu=0.15,
    mCHwdratio=10.0, stdevCHwdratio=1.0,
    mCHdepth=4.0, stdevCHdepth=0.4,
    mLVdepth=0.6, stdevLVdepth=0.1,
    mLVwidth=40.0, stdevLVwidth=5.0,
    mLVheight=0.4, stdevLVheight=0.1,
    mLVasym=0.3, stdevLVasym=0.1,
    mFFCHprop=0.0, stdevFFCHprop=0.0,
    mdistMigrate=35.0, stdevdistMigrate=10.0,
    Cf=0.0036, scour_factor=10.0, gradient=0.001, Q=5.0,
)

CB_JIGSAW = dict(
    ntime=400, nlevel=8,
    NTGtarget=0.30,
    probAvulOutside=0.05, probAvulInside=0.40,
    mCHsinu=1.3, stdevCHsinu=0.1,
    mCHwdratio=16.0, stdevCHwdratio=2.0,
    mCHdepth=4.0, stdevCHdepth=0.4,
    mLVdepth=0.6, stdevLVdepth=0.1,
    mLVwidth=40.0, stdevLVwidth=10.0,
    mLVheight=0.4, stdevLVheight=0.1,
    mLVasym=0.3, stdevLVasym=0.1,
    mFFCHprop=0.5, stdevFFCHprop=0.15,
    mdistMigrate=25.0, stdevdistMigrate=8.0,
    Cf=0.0036, scour_factor=10.0, gradient=0.001, Q=5.0,
)

CB_LABYRINTH = dict(
    ntime=600, nlevel=8,
    NTGtarget=0.25,
    probAvulOutside=0.05, probAvulInside=0.05,
    mCHsinu=1.3, stdevCHsinu=0.1,
    mCHwdratio=16.0, stdevCHwdratio=2.0,
    mCHdepth=4.0, stdevCHdepth=0.4,
    mLVdepth=0.5, stdevLVdepth=0.1,
    mLVwidth=30.0, stdevLVwidth=5.0,
    mLVheight=0.3, stdevLVheight=0.1,
    mLVasym=0.3, stdevLVasym=0.1,
    mFFCHprop=0.4, stdevFFCHprop=0.15,
    mdistMigrate=25.0, stdevdistMigrate=8.0,
    Cf=0.0036, scour_factor=10.0, gradient=0.001, Q=5.0,
)

SH_DISTAL = dict(
    ntime=400, nlevel=6,
    NTGtarget=0.50,
    probAvulOutside=0.02, probAvulInside=0.05,
    mCHsinu=1.3, stdevCHsinu=0.1,
    mCHwdratio=18.0, stdevCHwdratio=2.0,
    mCHdepth=5.0, stdevCHdepth=0.4,
    mLVdepth=1.2, stdevLVdepth=0.2,
    mLVwidth=150.0, stdevLVwidth=25.0,
    mLVheight=1.0, stdevLVheight=0.2,
    mLVasym=0.2, stdevLVasym=0.1,
    mFFCHprop=0.0, stdevFFCHprop=0.0,
    mdistMigrate=35.0, stdevdistMigrate=10.0,
    Cf=0.0036, scour_factor=10.0, gradient=0.001, Q=5.0,
)

SH_PROXIMAL = dict(
    ntime=400, nlevel=8,
    NTGtarget=0.40,
    probAvulOutside=0.08, probAvulInside=0.35,
    mCHsinu=1.2, stdevCHsinu=0.1,
    mCHwdratio=22.0, stdevCHwdratio=3.0,
    mCHdepth=4.0, stdevCHdepth=0.4,
    mLVdepth=0.4, stdevLVdepth=0.1,
    mLVwidth=20.0, stdevLVwidth=5.0,
    mLVheight=0.2, stdevLVheight=0.1,
    mLVasym=0.0, stdevLVasym=0.0,
    mFFCHprop=0.15, stdevFFCHprop=0.05,
    mdistMigrate=25.0, stdevdistMigrate=8.0,
    Cf=0.0036, scour_factor=10.0, gradient=0.001, Q=5.0,
)

# Single sinuous meandering channel that grows tight bends, neck-cuts
# off the oxbow loops, and fills each abandoned bend with a mud plug —
# the classic neck-cutoff → oxbow lake → mud plug succession of a
# meandering river belt, stacked vertically as a multi-storey channel
# sandstone. Per-event geometry is identical to ``tutorial_alluvsim``
# Showcase B (the canonical highly-sinuous case): pure migration
# (``probAvul* = 0``), ``mCHsinu = 1.5``, ``mdistMigrate = 2.0``,
# Pyrcz-Table-1 hydraulics — those values are already known to grow
# tight bends and trigger multiple neck cutoffs per level. We stack
# ``nlevel`` such belts on top of each other (so ``ntime`` is set to
# ~200 events per level) and turn on ``mNeckFFCHprop = 0.5`` so each
# excised oxbow loop is painted as an FFCH mud plug over the upper
# half of its column.
MEANDER_OXBOW = dict(
    # ``nlevel * mCHdepth >= nz·zsiz`` keeps the column saturated:
    # adjacent channels overlap so every Z slice shows the meander belt
    # rather than barren floodplain. The default ``level_z`` spread
    # (engine: ``linspace(mCHdepth, nz·zsiz, nlevel)``) puts the top
    # channel at the grid top and the bottom channel one full depth up.
    nlevel=8, ntime=50, ntime_per_level=True,
    NTGtarget=0.99,
    probAvulOutside=0.0, probAvulInside=0.0,
    mCHsinu=1.5, stdevCHsinu=0.03,
    mCHwdratio=11.0, stdevCHwdratio=0.5,
    mCHdepth=5.0, stdevCHdepth=0.3,
    mLVdepth=0.8, stdevLVdepth=0.1,
    mLVwidth=40.0, stdevLVwidth=5.0,
    mLVheight=0.5, stdevLVheight=0.05,
    mLVasym=0.0, mLVthin=0.0,
    mFFCHprop=0.0, stdevFFCHprop=0.0,
    mNeckFFCHprop=0.5,
    mCSnum=0.0, mCSnumlobe=0.0, stdevCSnum=0.0, stdevCSnumlobe=0.0,
    mdistMigrate=3.0, stdevdistMigrate=0.6,
    Cf=0.0036, scour_factor=10.0, gradient=0.001, Q=3.256,
    stdevCHsource=1.0,
)
