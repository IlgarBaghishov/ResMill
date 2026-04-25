"""Geometric neck-cutoff — port of Alluvsim ``neckcutoff.for``.

Scans every (idis, jdis) pair where ``jdis - idis >= dis_thresh`` to find
two non-adjacent nodes whose Euclidean separation is below ``ctol``. When
found, deletes nodes (idis, jdis) (the oxbow loop) by shifting the tail
left and reducing ``ndis``. Restarts the scan from the top after every cut
(matches AL's ``goto 435``).

Returns the new (compacted) ``ndis`` so the caller can slice the cx/cy
arrays.
"""
import numpy as np
from numba import jit


@jit(nopython=True)
def make_cutoff(cx, cy, dlength, ctol):
    """Modify cx/cy in place; return new ndis."""
    ndis = cx.size
    if ndis < 4:
        return ndis
    thresh = ctol * ctol
    # dis_thresh = ctol / (avg_arc_length / ndis) + 2  (AL:92-93)
    if ndis > 1:
        s_avg = float(np.sum(dlength)) / max(ndis, 1)
        dis_thresh = int(ctol / max(s_avg, 1e-9)) + 2
    else:
        dis_thresh = 2
    if dis_thresh < 2:
        dis_thresh = 2

    while True:
        cut_found = False
        for idis in range(ndis):
            for jdis in range(idis + dis_thresh, ndis):
                xi = cx[idis]
                yi = cy[idis]
                xj = cx[jdis]
                yj = cy[jdis]
                cdist = (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj)
                if cdist < thresh:
                    # Shift-left compaction: keep [0..idis], drop (idis..jdis], keep [jdis+1..]
                    count = 1
                    for j in range(jdis + 1, ndis):
                        cx[idis + count] = cx[j]
                        cy[idis + count] = cy[j]
                        count += 1
                    ndis = ndis - (jdis - idis)
                    cut_found = True
                    break
            if cut_found:
                break
        if not cut_found:
            break
    return ndis
