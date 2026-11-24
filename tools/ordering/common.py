"""Shared helpers for the ordering tools: reading locations, writing permutation files, Hilbert keys and the
spectral tile order.

Permutation file (read by ExaGeoStatCPP with --ordering=file --ordering_file=PATH):
    '#' header lines, then one zero-based location index per line; output position i holds input row order[i].
    Consecutive groups of b rows are the tiles. Headers checked by ExaGeoStatCPP: '# n <N>',
    '# coordsum <sum x> <sum y> [<sum z>]' (rejects another point set) and '# coordwsum ...' (sums with weight
    (i + 1) / n for input row i; rejects the same points in another row order, e.g. Morton-sorted)."""
import numpy as np


def read_locations(path):
    """CSV with a header line and one location per row (x,y or x,y,z); returns an (n, d) array in file order."""
    with open(path) as f:
        f.readline()
        p = np.loadtxt(f, delimiter=',', ndmin=2)
    if p.shape[1] not in (2, 3) or not np.isfinite(p).all():
        raise ValueError(f'{path}: expected 2 or 3 finite columns per row')
    return p


def write_order(path, order, points, tile, method, extra=()):
    order = np.asarray(order, dtype=np.int64); n = len(points)
    if not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError('not a permutation of the input rows')
    with open(path, 'w') as f:
        f.write(f'# method {method}\n# n {n}\n# tile {tile}\n')
        f.write('# coordsum ' + ' '.join(f'{s:.17g}' for s in points.sum(0)) + '\n')
        w = np.arange(1, n + 1) / n
        f.write('# coordwsum ' + ' '.join(f'{s:.17g}' for s in w @ points) + '\n')
        for line in extra: f.write(f'# {line}\n')
        f.write('\n'.join(map(str, order.tolist())) + '\n')


def read_order(path):
    vals = [int(l) for l in open(path) if l.strip() and not l.startswith('#')]
    return np.array(vals, dtype=np.int64)


def hilbert_key(p, lo, hi, order=16):
    """Hilbert index on a 2^order grid over the box [lo, hi] (same curve as HilbertIndex2D in ExaGeoStatCPP)."""
    n = 1 << order
    u = np.clip(((p - lo) / np.maximum(hi - lo, 1e-300) * (n - 1)).astype(np.int64), 0, n - 1)
    x, y = u[:, 0].copy(), u[:, 1].copy(); d = np.zeros(len(p), dtype=np.int64); s = n // 2
    while s > 0:
        rx = (x & s) > 0; ry = (y & s) > 0
        d += s * s * ((3 * rx) ^ ry)
        m = ~ry; f = m & rx
        x[f] = s - 1 - x[f]; y[f] = s - 1 - y[f]
        x[m], y[m] = y[m].copy(), x[m].copy()
        s //= 2
    return d


def centroids(p, lab, nt):
    cnt = np.bincount(lab, minlength=nt)
    return np.stack([np.bincount(lab, p[:, k], nt) for k in range(p.shape[1])], 1) / cnt[:, None]


def hilbert_tile_order(p, lab, nt):
    lo, hi = p.min(0), p.max(0)
    return np.argsort(hilbert_key(centroids(p, lab, nt)[:, :2], lo[:2], hi[:2]), kind='stable')


def spectral_tile_order(p, lab, nt):
    """Tiles sorted by the Fiedler vector of the tile adjacency graph (edge weight = number of Delaunay edges between
    the two tiles). Tiles are first numbered in Hilbert order of their centroids, which fixes the sign of the vector
    (the first Hilbert tile is on the negative side) and breaks ties."""
    from scipy.spatial import Delaunay
    base = hilbert_tile_order(p, lab, nt); rank = np.empty(nt, int); rank[base] = np.arange(nt)
    h = rank[lab]                                                          # tile ids in Hilbert numbering
    tri = Delaunay(p[:, :2]).simplices
    E = np.unique(np.sort(np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]], 1), axis=0)
    a, b = h[E[:, 0]], h[E[:, 1]]; x = a != b
    W = np.zeros((nt, nt)); np.add.at(W, (a[x], b[x]), 1); W = W + W.T
    L = np.diag(W.sum(1)) - W
    v = np.linalg.eigh(L)[1][:, 1]
    if v[0] > 0: v = -v
    return base[np.argsort(v, kind='stable')]


def order_from_tiles(p, lab, tile_order):
    """Final permutation: tiles in the given order, locations inside a tile in Hilbert order."""
    nt = len(tile_order); pos = np.empty(nt, int); pos[tile_order] = np.arange(nt)
    lo, hi = p.min(0), p.max(0)
    return np.lexsort((hilbert_key(p[:, :2], lo[:2], hi[:2]), pos[lab]))
