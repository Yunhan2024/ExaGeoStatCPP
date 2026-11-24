"""Capacity-constrained centroidal power diagram (each tile exactly B points, near-hexagonal cells) with an anisotropic
boundary ring. Lloyd iterations on a subsample with Newton-balanced power weights, then Newton weights + a narrow-band
transportation LP on all points for exact capacity; tiles placed by centroid Hilbert order, points inside a tile in Hilbert order."""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS'): os.environ.setdefault(_v, '8')
import numpy as np
from scipy.spatial import cKDTree

B = 4096


def metric(c, lo, hi, side, alpha, sides, delta):
    """per-cell diagonal metric: cells whose centre is within delta ideal sides of a domain edge measure distance across
    that edge x alpha and along it / alpha (cells thin across, long along the edge); sides 'lr', 'tb' or 'all'"""
    mx, my = np.ones(len(c)), np.ones(len(c))
    if alpha == 1: return mx, my
    if sides in ('lr', 'all'):
        f = np.minimum(c[:, 0] - lo[0], hi[0] - c[:, 0]) < delta * side; mx[f] *= alpha; my[f] /= alpha
    if sides in ('tb', 'all'):
        f = np.minimum(c[:, 1] - lo[1], hi[1] - c[:, 1]) < delta * side; my[f] *= alpha; mx[f] /= alpha
    return mx, my


def dist2(p, c, cand, m):
    d = p[:, None, :] - c[cand]
    return m[0][cand] * d[..., 0] ** 2 + m[1][cand] * d[..., 1] ** 2


def hilbert_key(p, lo, hi, order=16):
    n = 1 << order; u = np.clip(((p - lo) / np.maximum(hi - lo, 1e-300) * (n - 1)).astype(np.int64), 0, n - 1)
    x, y = u[:, 0].copy(), u[:, 1].copy(); d = np.zeros(len(p), dtype=np.int64); s = n // 2
    while s > 0:
        rx = (x & s) > 0; ry = (y & s) > 0
        d += s * s * ((3 * rx) ^ ry)
        m = ~ry; f = m & rx
        x[f] = s - 1 - x[f]; y[f] = s - 1 - y[f]
        x[m], y[m] = y[m].copy(), x[m].copy()
        s //= 2
    return d


def init_centres(p, nt, kind):
    """hexagonal lattice ('hex': rows along x; 'hexT': rows along y) with nt sites, rows x per-row count closest to sqrt(3)/2"""
    if kind == 'hexT':
        return init_centres(p[:, ::-1], nt, 'hex')[:, ::-1]
    lo, hi = p.min(0), p.max(0); W, H = hi - lo; best = None
    for R in range(1, nt + 1):
        if nt % R: continue
        k = nt // R; sc = abs(np.log((H / R) / (W / k) / (np.sqrt(3) / 2)))
        if best is None or sc < best[0]: best = (sc, R, k)
    _, R, k = best
    return np.array([[lo[0] + (i + (0.25 if r % 2 == 0 else 0.75)) * W / k, lo[1] + (r + 0.5) * H / R] for r in range(R) for i in range(k)])


def newton_weights(p, c, w, cap, m=None, K=4, iters=3, eps_q=0.02, ret_assign=False):
    """Newton steps on the power weights over all points: Jacobian = boundary-density Laplacian (points whose two best
    cells are i,j and whose margin is below eps, per unit weight), gauge fixed by least squares. Returns w, final count error."""
    nt = len(c); _, cand = cKDTree(c).query(p, k=K); d2 = dist2(p, c, cand, m) if m is not None else ((p[:, None, :] - c[cand]) ** 2).sum(-1)
    rows = np.arange(len(p)); w = w.copy()
    for it in range(iters + 1):
        cost = d2 - w[cand]; o = np.argpartition(cost, 1, axis=1)[:, :2]
        c1 = cost[rows, o[:, 0]]; c2 = cost[rows, o[:, 1]]; sw = c2 < c1
        b1 = np.where(sw, cand[rows, o[:, 1]], cand[rows, o[:, 0]]); b2 = np.where(sw, cand[rows, o[:, 0]], cand[rows, o[:, 1]])
        marg = np.abs(c2 - c1); err = np.bincount(b1, minlength=nt) - cap
        if it == iters or np.abs(err).max() <= 2: break
        eps = np.quantile(marg, eps_q); f = marg < eps
        G = np.zeros((nt, nt)); np.add.at(G, (b1[f], b2[f]), 1.0 / eps); G = (G + G.T) / 2     # only the far side of a boundary moves
        Lap = np.diag(G.sum(1)) - G                                    # d count_i / d w = Lap @ dw  (raising w_i grows cell i)
        dw = np.linalg.lstsq(Lap, -err, rcond=None)[0]; dw -= dw.mean()
        lim = 0.5 * np.median(d2[rows, o[:, 0]]) * 4                     # damp: weight step <= ~half a cell's squared size
        if np.abs(dw).max() > lim: dw *= lim / np.abs(dw).max()
        w += dw
    return (w, int(np.abs(err).max()), b1) if ret_assign else (w, int(np.abs(err).max()))


def exact_assign(p, c, w, cap, K=4, J=3, fracs=(0.1, 0.2, 0.4), m=None):
    """capacity-exact assignment: points whose power-cost margin is large keep their cell; the boundary band
    (smallest margins) is re-assigned by a transportation LP (min sum of squared distances, integral vertex)"""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    n, nt = len(p), len(c)
    _, cand = cKDTree(c).query(p, k=K); d2 = dist2(p, c, cand, m) if m is not None else ((p[:, None, :] - c[cand]) ** 2).sum(-1)
    cost = d2 - w[cand]
    o = np.argsort(cost, 1); a = cand[np.arange(n), o[:, 0]]
    marg = np.take_along_axis(cost, o[:, 1:2], 1)[:, 0] - np.take_along_axis(cost, o[:, :1], 1)[:, 0]
    for frac in fracs:
        free = marg < np.quantile(marg, frac); rem = cap - np.bincount(a[~free], minlength=nt)
        fi = np.where(free)[0]; nf = len(fi)
        jj = np.take_along_axis(cand[fi], o[fi][:, :J], 1); cc = np.take_along_axis(d2[fi], o[fi][:, :J], 1)
        rows = np.r_[np.repeat(np.arange(nf), J), nf + jj.ravel()]; cols = np.r_[np.arange(nf * J), np.arange(nf * J)]
        A = coo_matrix((np.ones(2 * nf * J), (rows, cols)), shape=(nf + nt, nf * J)).tocsr()
        r = linprog(cc.ravel(), A_eq=A, b_eq=np.r_[np.ones(nf), rem], bounds=(0, 1), method='highs-ds')
        if r.status == 0:
            x = r.x.reshape(nf, J); assert ((x > 1e-9) & (x < 1 - 1e-9)).sum() == 0
            a = a.copy(); a[fi] = jj[np.arange(nf), np.argmax(x, 1)]
            assert (np.bincount(a, minlength=nt) == cap).all()
            return a
    raise RuntimeError('band LP infeasible')


def region_assign(p, idx, cells, c, w, m, cap, J=3):
    """capacity-exact re-assignment of points idx among the given cells only (local LP; used by the centre-nudge searches)"""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    d = p[idx][:, None, :] - c[cells][None]
    cost = m[0][cells] * d[..., 0] ** 2 + m[1][cells] * d[..., 1] ** 2 - w[cells]
    o = np.argsort(cost, 1)[:, :J]; cc = np.take_along_axis(cost, o, 1); nf, nc = len(idx), len(cells)
    rows = np.r_[np.repeat(np.arange(nf), J), nf + o.ravel()]; cols = np.r_[np.arange(nf * J), np.arange(nf * J)]
    A = coo_matrix((np.ones(2 * nf * J), (rows, cols)), shape=(nf + nc, nf * J)).tocsr()
    r = linprog(cc.ravel(), A_eq=A, b_eq=np.r_[np.ones(nf), np.full(nc, cap)], bounds=(0, 1), method='highs-ds')
    if r.status != 0: return None
    x = r.x.reshape(nf, J); return np.asarray(cells)[o[np.arange(nf), np.argmax(x, 1)]]


def labels_fast(p, init='hex', iters=20, sub=8, seed=0, alpha=1.0, sides='lr', delta=0.6, newton=2, state=False):
    """tile label of every point (exactly B per tile); state=True also returns centres, weights, metric"""
    n = len(p); nt = n // B; assert nt * B == n
    rng = np.random.default_rng(seed); s = p[rng.permutation(n)[:n // sub]]
    c = init_centres(p, nt, init); w = np.zeros(nt)
    lo, hi = p.min(0), p.max(0); side = np.sqrt((hi - lo).prod() / nt)
    mk = lambda c: metric(c, lo, hi, side, alpha, sides, delta)
    for it in range(iters):
        w, _, a = newton_weights(s, c, w, len(s) / nt, m=mk(c), iters=6 if it == 0 else newton, ret_assign=True)
        cn = centroids(s, a, nt); ok = np.isfinite(cn).all(1); c = np.where(ok[:, None], cn, c)   # an emptied cell keeps its centre
    m = mk(c); w, _ = newton_weights(p, c, w, B, m=m, iters=2)
    a = exact_assign(p, c, w, B, m=m, fracs=(0.01, 0.02, 0.05, 0.1, 0.2, 0.4))
    return (a, c, w, m) if state else a


def centroids(p, a, nt):
    cnt = np.bincount(a, minlength=nt)
    return np.c_[np.bincount(a, p[:, 0], nt), np.bincount(a, p[:, 1], nt)] / cnt[:, None]


def order_from_labels(p, a):
    nt = len(p) // B; lo, hi = p.min(0), p.max(0)
    cen = centroids(p, a, nt)
    tord = np.argsort(hilbert_key(cen, lo, hi), kind='stable'); hk = hilbert_key(p, lo, hi)
    srt = np.lexsort((hk, np.argsort(tord)[a]))       # by tile placement, then Hilbert inside the tile
    return srt
