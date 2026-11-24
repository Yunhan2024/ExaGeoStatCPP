"""2D min-bounding-box recursive bisection, a line-by-line port of the C++ generator used for the QSR baseline
(same cut rule as --ordering=minbbox: cut across the longer side at the tile-aligned position within two tiles of
the middle that minimizes the sum of the two halves' bbox diagonals; halves in Hilbert order of their centroids).
QSR uses it only to set the scale of its rank surrogate (the largest predicted rank over the min-bbox tiles)."""
import sys
import numpy as np

ORDER = 1 << 16


def _hilbert(x, y):
    x = int(x * (ORDER - 1)); y = int(y * (ORDER - 1)); d = 0; s = ORDER // 2
    while s > 0:
        rx = 1 if x & s else 0; ry = 1 if y & s else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1: x, y = s - 1 - x, s - 1 - y
            x, y = y, x
        s //= 2
    return d


def minbbox_tiles(p, tile):
    """List of index arrays, one per tile (the last one may be short)."""
    sys.setrecursionlimit(10000); out = []

    def rec(sub):
        n = len(sub); nt = -(-n // tile)
        if nt <= 1: out.append(sub); return
        q = p[sub]; ext = q.max(0) - q.min(0); ax = 0 if ext[0] >= ext[1] else 1
        s = sub[np.argsort(q[:, ax], kind='stable')]; q = p[s]
        pmin = np.minimum.accumulate(q); pmax = np.maximum.accumulate(q)
        smin = np.minimum.accumulate(q[::-1])[::-1]; smax = np.maximum.accumulate(q[::-1])[::-1]
        mid = nt // 2; best = min(max(mid, 1), nt - 1); bo = np.inf
        for dt in range(-2, 3):
            lt = mid + dt
            if lt < 1 or lt >= nt: continue
            ls = lt * tile
            if ls < 1 or ls >= n: continue
            o = np.hypot(*(pmax[ls - 1] - pmin[ls - 1])) + np.hypot(*(smax[ls] - smin[ls]))
            if o < bo: bo, best = o, lt
        A, B = s[:best * tile], s[best * tile:]
        if _hilbert(*p[A].mean(0)) > _hilbert(*p[B].mean(0)): A, B = B, A
        rec(A); rec(B)

    rec(np.arange(len(p)))
    return out
