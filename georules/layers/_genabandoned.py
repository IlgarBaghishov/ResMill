"""Abandoned-channel mud plug (port of ``genabandonedchannel.for``).

Stamps the current streamline's U-shape footprint at each node's
``CHelev``, with FFCH (mud) in the top ``mud_prop`` fraction of the column
and CH (residual sand) below — exactly as ``genabandonedchannel.for:213-228``.
The split uses the continuous-z criterion ``z > FFCHbot`` (not cell-count
rounding — item 2.18); ``FFCHbot = CHelev - mud_prop*(CHelev - CHbot)``.
"""
import numpy as np
from numba import jit

from ._genchannel import find_near_grid, _refine_nearest


# Alluvsim facies codes
FF, FFCH, CS, LV, LA, CH = -1, 0, 1, 2, 3, 4


@jit(nopython=True)
def _paint_abandoned_kernel(
    nz, mynx, myny, localx, localy, x, y, vx, vy, cy, cx, thalweg,
    chelev_arr, zsiz, dd, facies, chwidth, idmat, dwratio,
    mud_prop, lk_ffch, lk_ch, ntg_counter, ffch_counter,
):
    for myid in range(localx.size):
        idx = mynx[myid]
        idy = myny[myid]
        idis = dd[myid]
        dist = idmat[idis, myid]
        # Local-node halfwidth gate (item 1.6 / 4.24)
        if dist > chwidth[idis]:
            continue

        t = dwratio * chwidth[idis]
        WW = chwidth[idis] * 2.0
        chelev = chelev_arr[idis]
        dx2 = x[idx] - cx[idis]
        dy2 = y[idy] - cy[idis]
        indicator = dx2 * vy[idis] - dy2 * vx[idis]
        if indicator > 0:
            wid = dist + chwidth[idis]
        else:
            wid = chwidth[idis] - dist
        if wid < 0.0 or wid > WW:
            continue
        a = min(0.999, max(0.001, thalweg[idis]))

        if a < 0.5:
            by = -np.log(2.0) / np.log(a)
            chbot = chelev - 4.0 * t * (wid / WW)**by * (1.0 - (wid / WW)**by)
        else:
            cy_ = -np.log(2.0) / np.log(1.0 - a)
            chbot = chelev - 4.0 * t * (1.0 - wid / WW)**cy_ * (1.0 - (1.0 - wid / WW)**cy_)

        if chbot >= chelev:
            continue
        ffch_bot = chelev - mud_prop * (chelev - chbot)

        # Continuous-z stamp (item 2.18 / 2.19): iterate every iz, gate via z_face.
        for iz in range(nz):
            z_face = (iz + 0.5) * zsiz
            if z_face > chbot and z_face < chelev:
                if z_face > ffch_bot:
                    # Top fraction → FFCH (abandoned mud)
                    prev = facies[idx, idy, iz]
                    if prev >= 1:
                        ntg_counter[0] -= 1
                    facies[idx, idy, iz] = lk_ffch
                    ffch_counter[0] += 1
                else:
                    # Bottom fraction → CH (residual sand)
                    prev = facies[idx, idy, iz]
                    if prev < 1:
                        ntg_counter[0] += 1
                    facies[idx, idy, iz] = lk_ch
    return 0


def paint_abandoned(
    chwidth_node: float,
    cx: np.ndarray, cy: np.ndarray,
    vx: np.ndarray, vy: np.ndarray,
    thalweg: np.ndarray, chwidth_arr: np.ndarray,
    chelev_arr, dwratio: float, mud_prop: float,
    x: np.ndarray, y: np.ndarray,
    xsiz: float, ysiz: float, zsiz: float,
    nx: int, ny: int, nz: int,
    facies: np.ndarray,
    ntg_counter: np.ndarray, ffch_counter: np.ndarray,
    *, xmn: float = 0.0, ymn: float = 0.0,
    lk_ffch: int = FFCH, lk_ch: int = CH,
):
    # Reset ffch_counter unconditionally (item 1.13)
    ffch_counter[0] = 0
    if cx is None or cx.size < 3 or mud_prop <= 0.0:
        return

    if np.isscalar(chelev_arr):
        chelev_arr = np.full(cx.size, float(chelev_arr), dtype=np.float64)
    elif chelev_arr.size != cx.size:
        chelev_arr = np.full(cx.size, float(chelev_arr.mean()), dtype=np.float64)

    ndis = cx.size
    good = np.zeros((nx, ny))
    find_near_grid(cx, cy, good, xsiz, ysiz, xmn, ymn, chwidth_node, nx, ny)
    mynx, myny = np.where(good == 1)
    if mynx.size == 0:
        return
    cx2 = cx.reshape(ndis, 1)
    cy2 = cy.reshape(ndis, 1)
    localx = x[mynx]
    localy = y[myny]
    idmat = np.sqrt((cx2 - localx) ** 2 + (cy2 - localy) ** 2)
    dd = idmat.argmin(axis=0)
    refined = _refine_nearest(cx, cy, localx, localy, dd, ndiscr=5)
    for myid in range(localx.size):
        idmat[dd[myid], myid] = refined[myid]

    _paint_abandoned_kernel(
        nz, mynx, myny, localx, localy, x, y, vx, vy, cy, cx, thalweg,
        chelev_arr, zsiz, dd, facies, chwidth_arr, idmat, dwratio,
        float(mud_prop), int(lk_ffch), int(lk_ch),
        ntg_counter, ffch_counter,
    )
