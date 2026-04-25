"""Active-channel U-shape stamp (port of Alluvsim ``genchannel.for``).

Paints one streamline's Deutsch-Wang asymmetric U-shape cross-section into
``facies`` for the **active channel** (CH=4) and **lateral-accretion** (LA=3)
elements. Mud-plug abandonment is in ``_genabandoned.py``.

Per-cell: find nearest streamline node, refine via 5-step ``splint`` walk to
sub-node accuracy (matches AL ``genchannel.for:146-167``), evaluate
``chelev``, ``chwidth`` and ``thalweg`` at the refined arc-length, decide
which side of the centreline the cell sits on (cross-product on tangent),
build the asymmetric U-shape ``CHbot``, and stamp facies for cells in
``(CHbot, CHelev]``.

When ``erode_above`` is True (active CH) and the cell sits *above*
``CHelev`` with an existing reservoir code, it is reset to FF — port of
``genchannel.for:213-218``.
"""
import numpy as np
from numba import jit


@jit(nopython=True)
def find_near_grid(cx, cy, good, xsiz, ysiz, xmn, ymn, b, nx, ny):
    """Mark cells within ~6b of any centerline node (search bbox).

    Uses xmn/ymn cell-origin offset (item 1.14 fix). Per-axis bbox radius:
    ``int(6*b / xsiz)`` for x, ``int(6*b / ysiz)`` for y — isotropic in
    physical units even on anisotropic grids.
    """
    ndx = max(int(6.0 * b / xsiz), 1)
    ndy = max(int(6.0 * b / ysiz), 1)
    for i in range(cx.size):
        mynx = int((cx[i] - xmn) / xsiz)
        myny = int((cy[i] - ymn) / ysiz)
        for j in range(max(0, mynx - ndx), min(nx, mynx + ndx + 1)):
            for k in range(max(0, myny - ndy), min(ny, myny + ndy + 1)):
                good[j, k] = 1
    return 0


@jit(nopython=True)
def mychannel(nz, mynx, myny, localx, localy, x, y, vx, vy, cy, cx,
              thalweg, chelev_arr, zsiz, dd, facies, poro, chwidth,
              idmat, dwratio, merge_overlap, facies_code, ntg_counter,
              compute_poro, erode_above, poro0):
    """Paint one streamline's U-shape into facies (and poro if asked)."""
    for myid in range(localx.size):
        idx = mynx[myid]
        idy = myny[myid]
        idis = dd[myid]
        dist = idmat[idis, myid]
        # Local-node halfwidth gate (item 1.6)
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
        # Out-of-channel skip (item 1.15)
        if wid < 0.0 or wid > WW:
            continue
        # AL clamp: a in [1e-3, 1-1e-3] (item 4.2)
        a = min(0.999, max(0.001, thalweg[idis]))

        if a < 0.5:
            by = -np.log(2.0) / np.log(a)
            chbot = chelev - 4.0 * t * (wid / WW)**by * (1.0 - (wid / WW)**by)
            wid2 = a * WW
            maxD = 4.0 * t * (wid2 / WW)**by * (1.0 - (wid2 / WW)**by)
        else:
            cy_ = -np.log(2.0) / np.log(1.0 - a)
            chbot = chelev - 4.0 * t * (1.0 - wid / WW)**cy_ * (1.0 - (1.0 - wid / WW)**cy_)
            wid2 = a * WW
            maxD = 4.0 * t * (1.0 - wid2 / WW)**cy_ * (1.0 - (1.0 - wid2 / WW)**cy_)

        # Erode residual reservoir code above the channel surface (AL:213-218)
        # only for active CH stamps. Iterate every iz; gate on z_face > chelev.
        if erode_above and facies_code >= 1:
            for iz in range(nz):
                z_face = (iz + 0.5) * zsiz  # cell centre
                if z_face > chelev and facies[idx, idy, iz] >= 1:
                    ntg_counter[0] -= 1
                    facies[idx, idy, iz] = -1  # FF

        # Stamp the U-shape body.  Per-cell continuous-z criterion
        # (AL:219 ``if z > chbot and z < chelev``).
        for iz in range(nz):
            z_face = (iz + 0.5) * zsiz
            if z_face > chbot and z_face < chelev:
                if facies[idx, idy, iz] < 1 and facies_code >= 1:
                    ntg_counter[0] += 1
                facies[idx, idy, iz] = facies_code

                if compute_poro:
                    if maxD > 1e-9:
                        depth_into_channel = chelev - z_face
                        depth_norm = depth_into_channel / max(maxD, 1e-9)
                        new_poro = (
                            (0.9 / (1.0 + np.exp(-4.0 * (depth_norm - 0.2))) + 0.1)
                            * poro0 * min((chelev - chbot) / max(maxD, 1e-9), 1.0)**2 + 0.1
                        )
                        if merge_overlap:
                            if new_poro > poro[idx, idy, iz]:
                                poro[idx, idy, iz] = new_poro
                        else:
                            poro[idx, idy, iz] = new_poro
    return 0


def _refine_nearest(cx, cy, x_loc, y_loc, dd_initial, ndiscr=5):
    """5-step ``splint``-equivalent walk to sub-node accuracy (item 2.10).

    For each cell, given the closest discrete node ``dd_initial[myid]``, walk
    ``ndiscr`` linearly-interpolated sub-positions between the neighbouring
    nodes and return the (refined node index, refined arc-length distance).

    Linear interpolation between adjacent nodes is sufficient here (the
    streamline has already been spline-resampled to ndis0 uniform spacing
    in ``cal_curv``); the spline curve passes through these nodes.
    """
    n_local = dd_initial.size
    n_nodes = cx.size
    refined_dist = np.empty(n_local, dtype=np.float64)
    for myid in range(n_local):
        idis = int(dd_initial[myid])
        lo = max(0, idis - 1)
        hi = min(n_nodes - 1, idis + 1)
        best = (cx[idis] - x_loc[myid])**2 + (cy[idis] - y_loc[myid])**2
        for sub in range(1, ndiscr):
            t = sub / float(ndiscr)
            xt = (1.0 - t) * cx[lo] + t * cx[hi]
            yt = (1.0 - t) * cy[lo] + t * cy[hi]
            d = (xt - x_loc[myid])**2 + (yt - y_loc[myid])**2
            if d < best:
                best = d
        refined_dist[myid] = float(np.sqrt(max(best, 0.0)))
    return refined_dist


def genchannel(b, xsiz, ysiz, chelev_arr, zsiz, nx, ny, nz, cx, cy, x, y,
               vx, vy, curv, LV_asym, lV_height, ps, pavul, out, totalid,
               facies, poro, poro0, thalweg, chwidth, dwratio, cutoff, NN=800,
               *, xmn=0.0, ymn=0.0,
               merge_overlap=False, facies_code=1, ntg_counter=None,
               compute_poro=True, erode_above=False):
    """Public wrapper around ``mychannel`` — Alluvsim ``genchannel.for`` entry.

    ``chelev_arr`` is the per-node CHelev array (item 2.11). For a flat
    channel it can be a constant array; for graded channels it varies.
    """
    if ntg_counter is None:
        ntg_counter = np.zeros(1, dtype=np.int64)
    if np.isscalar(chelev_arr):
        chelev_arr = np.full(cx.size, float(chelev_arr), dtype=np.float64)
    elif chelev_arr.size != cx.size:
        chelev_arr = np.full(cx.size, float(chelev_arr.mean()), dtype=np.float64)
    ndis = cx.size
    good = np.zeros((nx, ny))
    find_near_grid(cx, cy, good, xsiz, ysiz, xmn, ymn, b, nx, ny)
    mynx, myny = np.where(good == 1)
    if mynx.size == 0:
        return 0
    cx2 = cx.reshape(ndis, 1)
    cy2 = cy.reshape(ndis, 1)
    localx = x[mynx]
    localy = y[myny]
    idmat = np.sqrt((cx2 - localx)**2 + (cy2 - localy)**2)
    dd = idmat.argmin(axis=0)
    # Sub-node refinement (item 2.10)
    refined = _refine_nearest(cx, cy, localx, localy, dd, ndiscr=5)
    # Replace the per-cell distance row in idmat at the dd index with the refined value
    for myid in range(localx.size):
        idmat[dd[myid], myid] = refined[myid]

    mychannel(nz, mynx, myny, localx, localy, x, y, vx, vy, cy, cx, thalweg,
              chelev_arr, zsiz, dd, facies, poro, chwidth, idmat,
              dwratio, bool(merge_overlap), int(facies_code), ntg_counter,
              bool(compute_poro), bool(erode_above), float(poro0))
    return 0
