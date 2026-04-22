import numpy as np
from numba import jit


@jit(nopython=True)
def find_near_grid(cx, cy, good, xsiz, ysiz, b, nx, ny):
    nd = int(6 * b / min(xsiz, ysiz))
    for i in range(cx.size):
        mynx = int(cx[i] / xsiz)
        myny = int(cy[i] / ysiz)
        for j in range(max(0, mynx - nd), min(nx, mynx + nd)):
            for k in range(max(0, myny - nd), min(ny, myny + nd)):
                good[j, k] = 1
    return 0


@jit(nopython=True)
def mychannel(nz, lV_height, LV_asym, curv, mynx, myny, localx, localy,
              x, y, vx, vy, cy, cx, thalweg, chelev, idz, out, ps, zsiz, dd,
              totalid, cutoff, pavul, poro0, facies, poro, chwidth, idmat,
              nx, ny, dwratio, NN):
    for myid in range(localx.size):
        idx = mynx[myid]
        idy = myny[myid]
        dist = idmat[dd[myid], myid]
        idis = dd[myid]
        if dist > chwidth[0]:
            continue

        t = dwratio * chwidth[idis]
        WW = chwidth[idis] * 2
        dx2 = x[idx] - cx[idis]
        dy2 = y[idy] - cy[idis]
        indicator = dx2 * vy[idis] - dy2 * vx[idis]
        if indicator > 0:
            wid = dist + chwidth[idis]
            wid = max(wid, 0)
        else:
            wid = chwidth[idis] - dist
            wid = max(wid, 0)
        a = min(0.99999, max(0.00001, thalweg[idis]))

        if a < 0.5:
            by = -np.log(2.0) / np.log(a)
            chbot = chelev - 4.0 * t * (wid / WW)**by * (1.0 - (wid / WW)**by)
            wid2 = a * WW
            maxD = 4.0 * t * (wid2 / WW)**by * (1.0 - (wid2 / WW)**by)
        else:
            ddy = -np.log(2) / np.log(1 - a)
            chbot = chelev - 4.0 * t * (max(0, 1.0 - wid / WW))**ddy * (1.0 - (max(0, 1.0 - wid / WW))**ddy)
            wid2 = a * WW
            maxD = 4.0 * t * (max(0, 1.0 - wid2 / WW))**ddy * (1.0 - (max(0, 1.0 - wid2 / WW))**ddy)

        ddz = int(max(((chelev - chbot) / zsiz), 0))
        facies[idx, idy, max(idz - ddz, 0):min(idz, nz)] = 1
        maxDint = int(maxD / zsiz)
        for myz in np.arange(ddz):
            iz = idz - myz
            if iz < 0 or iz >= nz:
                continue
            poro[idx, idy, iz] = (
                (0.9 / (1 + np.exp(-4 * (myz / (maxDint) - 0.2))) + 0.1)
                * (poro0) * (min((chelev - chbot) / (maxD), 1)**2) + 0.1
            )

    return 0


def genchannel(b, xsiz, ysiz, chelev, zsiz, nx, ny, nz, cx, cy, x, y,
               vx, vy, curv, LV_asym, lV_height, ps, pavul, out, totalid,
               facies, poro, poro0, thalweg, chwidth, dwratio, cutoff, NN=800):
    idz = int(chelev / zsiz)
    ndis = cx.size
    good = np.zeros((nx, ny))
    find_near_grid(cx, cy, good, xsiz, ysiz, b, nx, ny)
    mynx, myny = np.where(good == 1)
    cx2 = cx.reshape(ndis, 1)
    cy2 = cy.reshape(ndis, 1)
    localx = x[mynx]
    localy = y[myny]
    idmat = np.sqrt((cx2 - localx)**2 + (cy2 - localy)**2)
    dd = idmat.argmin(axis=0)

    mychannel(nz, lV_height, LV_asym, curv, mynx, myny, localx, localy,
              x, y, vx, vy, cy, cx, thalweg, chelev, idz, out, ps, zsiz, dd,
              totalid, cutoff, pavul, poro0, facies, poro, chwidth, idmat,
              nx, ny, dwratio, NN)
    return 0
