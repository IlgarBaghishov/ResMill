"""Crevasse-splay (CS) painters — port of Alluvsim ``calc_lobe.for`` and
``gensplay.for``.

Two pieces, both invoked per CS event from ``_fluvial.py:_stamp_splays``:

1. ``paint_lobe`` — paints the lobe envelope along a sub-streamline (the
   gensplay walker), using AL's parabolic-then-elliptical width function
   plus the ``y_function * hwratio * (1 - (d/y_fn)^2)`` thickness.
2. ``paint_splay`` — paints the thin gensplay sheet (one cell below
   ``chelev``) with linear taper ``mxz = int(0.2*chdepth*(1-i/nst)/zsiz)``.
   Preserves CH(4) / LV(2) (item 1.9).
"""
import numpy as np
from numba import jit


CS = 1


@jit(nopython=True)
def _paint_lobe_kernel(
    nx, ny, nz, xsiz, ysiz, zsiz, xmn, ymn, x_grid, y_grid,
    cx_lobe, cy_lobe, s_arr, lobe_LL, lobe_WW, lobe_l, lobe_w,
    lobe_hw, lobe_dw, lobe_datum, facies, ntg_counter, lk_cs,
    depth_norm, poro_mult_field, log_perm_offset_field,
    ev_poro_mult, ev_log_perm_offset,
):
    n_lobe = cx_lobe.size
    if n_lobe < 2:
        return 0
    # Bbox
    pad = lobe_WW + max(xsiz, ysiz) * 2
    xmn_b = cx_lobe.min() - pad
    xmx_b = cx_lobe.max() + pad
    ymn_b = cy_lobe.min() - pad
    ymx_b = cy_lobe.max() + pad
    ix0 = max(0, int((xmn_b - xmn) / xsiz))
    ix1 = min(nx, int((xmx_b - xmn) / xsiz) + 1)
    iy0 = max(0, int((ymn_b - ymn) / ysiz))
    iy1 = min(ny, int((ymx_b - ymn) / ysiz) + 1)

    for ix in range(ix0, ix1):
        x_loc = x_grid[ix]
        for iy in range(iy0, iy1):
            y_loc = y_grid[iy]
            # Nearest sub-streamline node
            best = 1e18
            best_idx = -1
            for i in range(n_lobe):
                d2 = (cx_lobe[i] - x_loc) ** 2 + (cy_lobe[i] - y_loc) ** 2
                if d2 < best:
                    best = d2
                    best_idx = i
            if best_idx < 0:
                continue
            close_distance = np.sqrt(best)
            if close_distance > lobe_WW:
                continue
            s_close = s_arr[best_idx]

            # Lobe half-width envelope
            y_function = 0.0
            if s_close <= lobe_l and lobe_l > 1e-9:
                y_function = lobe_WW - (lobe_WW - lobe_w) * (1.0 - s_close / lobe_l) ** 2
            elif s_close <= lobe_LL and lobe_LL > lobe_l + 1e-9:
                u = (s_close - lobe_l) / (lobe_LL - lobe_l)
                if u < 1.0:
                    y_function = lobe_WW * np.sqrt(max(0.0, 1.0 - u * u))
            if y_function <= 1e-3:
                continue
            if close_distance > y_function:
                continue
            d_ratio = close_distance / y_function
            envelope = max(0.0, 1.0 - d_ratio * d_ratio)
            lobe_top = lobe_datum + y_function * lobe_hw * envelope
            lobe_bottom = lobe_datum - y_function * lobe_dw * envelope
            if lobe_top <= lobe_bottom:
                continue

            # Stamp CS — preserve CH(4) and LV(2) (AL ``gensplay.for:106``).
            # Splay deposits are thin sheets without within-deposit upward
            # fining → depth_norm = 0.5 (ramp = 1.0 in finalize).
            for iz in range(nz):
                z_face = (iz + 0.5) * zsiz
                if z_face >= lobe_bottom and z_face <= lobe_top:
                    prev = facies[ix, iy, iz]
                    if prev == 4 or prev == 2:
                        continue
                    if prev < 1:
                        ntg_counter[0] += 1
                    facies[ix, iy, iz] = lk_cs
                    depth_norm[ix, iy, iz] = np.float32(0.5)
                    poro_mult_field[ix, iy, iz] = np.float32(ev_poro_mult)
                    log_perm_offset_field[ix, iy, iz] = np.float32(ev_log_perm_offset)
    return 0


def paint_lobe(
    cx_lobe: np.ndarray, cy_lobe: np.ndarray,
    lobe_LL: float, lobe_WW: float, lobe_l: float, lobe_w: float,
    lobe_hw_ratio: float, lobe_dw_ratio: float,
    lobe_datum: float,
    x_grid: np.ndarray, y_grid: np.ndarray,
    nx: int, ny: int, nz: int,
    xsiz: float, ysiz: float, zsiz: float,
    facies: np.ndarray, ntg_counter: np.ndarray,
    *, xmn: float = 0.0, ymn: float = 0.0, lk_cs: int = CS,
    depth_norm: np.ndarray | None = None,
    poro_mult_field: np.ndarray | None = None,
    log_perm_offset_field: np.ndarray | None = None,
    ev_poro_mult: float = 1.0, ev_log_perm_offset: float = 0.0,
):
    """Paint the lobe envelope along the supplied sub-streamline (gensplay walker).

    Lobe widths are halved on entry (Alluvsim ``calc_lobe.for:123-125``):
    `lobe_w/2`, `lobe_WW/2` (these are full-width inputs).
    """
    if cx_lobe is None or cx_lobe.size < 3:
        return
    if lobe_LL <= 0 or lobe_WW <= 0:
        return
    if depth_norm is None:
        depth_norm = np.full((nx, ny, nz), 0.5, dtype=np.float32)
    if poro_mult_field is None:
        poro_mult_field = np.ones((nx, ny, nz), dtype=np.float32)
    if log_perm_offset_field is None:
        log_perm_offset_field = np.zeros((nx, ny, nz), dtype=np.float32)
    s_arr = np.zeros(cx_lobe.size, dtype=np.float64)
    s_arr[1:] = np.sqrt(np.diff(cx_lobe) ** 2 + np.diff(cy_lobe) ** 2)
    s_arr = np.cumsum(s_arr)

    _paint_lobe_kernel(
        nx, ny, nz, xsiz, ysiz, zsiz, xmn, ymn, x_grid, y_grid,
        cx_lobe, cy_lobe, s_arr,
        float(lobe_LL), float(lobe_WW * 0.5),
        float(lobe_l), float(max(lobe_w * 0.5, 1e-3)),
        float(lobe_hw_ratio), float(lobe_dw_ratio),
        float(lobe_datum),
        facies, ntg_counter, int(lk_cs),
        depth_norm, poro_mult_field, log_perm_offset_field,
        float(ev_poro_mult), float(ev_log_perm_offset),
    )


@jit(nopython=True)
def _paint_splay_kernel(
    nx, ny, nz, xsiz, ysiz, zsiz, xmn, ymn,
    cx_walk, cy_walk, chelev, chdepth, facies, ntg_counter, lk_cs,
    depth_norm, poro_mult_field, log_perm_offset_field,
    ev_poro_mult, ev_log_perm_offset,
):
    """Thin gensplay sheet at iz_chelev - 1 with linear taper (item 1.10/2.26).

    Per ``gensplay.for:103-112``:
        ``mxz = int(0.2*chdepth*(nst-ist)/nst / zsiz + 0.5)``
        ``do iiz = iz, iz - mxz, -1``: paint CS, preserving CH(3)/LV(2).
    """
    nst = cx_walk.size
    if nst < 2:
        return 0
    iz_top = int(chelev / zsiz) - 1
    if iz_top < 0:
        return 0
    for ist in range(nst):
        ix = int((cx_walk[ist] - xmn) / xsiz)
        iy = int((cy_walk[ist] - ymn) / ysiz)
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            continue
        taper = (nst - ist) / float(nst)
        mxz = int(0.2 * chdepth * taper / zsiz + 0.5)
        for iiz in range(iz_top, iz_top - mxz - 1, -1):
            if iiz < 0 or iiz >= nz:
                continue
            prev = facies[ix, iy, iiz]
            if prev == 4 or prev == 2:
                continue
            if prev < 1:
                ntg_counter[0] += 1
            facies[ix, iy, iiz] = lk_cs
            depth_norm[ix, iy, iiz] = np.float32(0.5)
            poro_mult_field[ix, iy, iiz] = np.float32(ev_poro_mult)
            log_perm_offset_field[ix, iy, iiz] = np.float32(ev_log_perm_offset)
    return 0


def paint_splay(
    cx_walk: np.ndarray, cy_walk: np.ndarray,
    chelev: float, chdepth: float,
    x_grid: np.ndarray, y_grid: np.ndarray,
    nx: int, ny: int, nz: int,
    xsiz: float, ysiz: float, zsiz: float,
    facies: np.ndarray, ntg_counter: np.ndarray,
    *, xmn: float = 0.0, ymn: float = 0.0, lk_cs: int = CS,
    depth_norm: np.ndarray | None = None,
    poro_mult_field: np.ndarray | None = None,
    log_perm_offset_field: np.ndarray | None = None,
    ev_poro_mult: float = 1.0, ev_log_perm_offset: float = 0.0,
):
    """Paint the gensplay thin sheet along ``cx_walk, cy_walk``."""
    if cx_walk is None or cx_walk.size < 2:
        return
    if depth_norm is None:
        depth_norm = np.full((nx, ny, nz), 0.5, dtype=np.float32)
    if poro_mult_field is None:
        poro_mult_field = np.ones((nx, ny, nz), dtype=np.float32)
    if log_perm_offset_field is None:
        log_perm_offset_field = np.zeros((nx, ny, nz), dtype=np.float32)
    _paint_splay_kernel(
        nx, ny, nz, xsiz, ysiz, zsiz, xmn, ymn,
        cx_walk, cy_walk, float(chelev), float(chdepth),
        facies, ntg_counter, int(lk_cs),
        depth_norm, poro_mult_field, log_perm_offset_field,
        float(ev_poro_mult), float(ev_log_perm_offset),
    )
