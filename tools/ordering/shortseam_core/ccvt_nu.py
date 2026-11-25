"""Equal-count centroidal power diagram computed on the locations themselves (no area, no density model), for any point
density: tiles = power cells of nt sites, every tile exactly `tile` locations, sites at the centroids of their tiles
(capacity-constrained Lloyd). Same construction as the research HexCVT with two changes that remove its
uniform-density assumptions: the start is the centroids of the min-box tiles (follows the density) instead of a
hexagonal lattice of the box, and the weight balancing measures everything relative to the local site spacing
(band of points near a power line = within DELTA site spacings; weight step limited to THETA spacing^2 per neighbour
pair) instead of using one global length. Deterministic.
usage (module): labels = ccvt_nu.tiles(p, tile)"""
import numpy as np
from scipy.spatial import cKDTree

DELTA, THETA = 0.08, 0.25


def _two_best(d2, w, cand):
    cost = d2 - w[cand]; o = np.argpartition(cost, 1, axis=1)[:, :2]; rows = np.arange(len(cost))
    c1, c2 = cost[rows, o[:, 0]], cost[rows, o[:, 1]]; sw = c2 < c1
    b1 = np.where(sw, cand[rows, o[:, 1]], cand[rows, o[:, 0]]); b2 = np.where(sw, cand[rows, o[:, 0]], cand[rows, o[:, 1]])
    return b1, b2, np.abs(c2 - c1)


def balance(p, c, w, cap, K=16, iters=20, tol=2):
    """power weights for which every cell of the sites c holds cap points (damped Newton; returns w, largest count error,
    assignment)"""
    nt = len(c); K = min(K, nt); _, cand = cKDTree(c).query(p, k=K); d2 = ((p[:, None, :] - c[cand]) ** 2).sum(-1); w = w.copy()
    for it in range(iters + 1):
        b1, b2, marg = _two_best(d2, w, cand); err = np.bincount(b1, minlength=nt) - cap
        if np.abs(err).max() <= tol or it == iters: break
        dij = np.linalg.norm(c[b1] - c[b2], axis=1); g = marg / (2 * dij); f = g < DELTA * dij    # distance to the power line
        G = np.zeros((nt, nt)); np.add.at(G, (b1[f], b2[f]), 1.0 / (2 * DELTA * dij[f] ** 2)); G = (G + G.T) / 2
        Lap = np.diag(G.sum(1)) - G                                   # d count / d w
        dw = np.linalg.lstsq(Lap, -err.astype(float), rcond=None)[0]; dw -= dw.mean()
        r = (np.abs(dw[b1] - dw[b2]) / (THETA * dij ** 2))[f].max() if f.any() else 0.0
        if r > 1: dw /= r
        w += dw
    return w, int(np.abs(err).max()), b1


def exact(p, c, w, cap, K=16, J=4, fracs=(0.02, 0.05, 0.1, 0.2, 0.4, 1.0)):
    """exactly cap points per cell: the points closest to a power line (relative to the site spacing) are re-assigned by a
    transportation LP on power costs"""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    n, nt = len(p), len(c); K = min(K, nt); J = min(J, K)
    _, cand = cKDTree(c).query(p, k=K); d2 = ((p[:, None, :] - c[cand]) ** 2).sum(-1); cost = d2 - w[cand]
    o = np.argsort(cost, 1)[:, :J]; rows = np.arange(n); a = cand[rows, o[:, 0]]
    marg = (cost[rows, o[:, 1]] - cost[rows, o[:, 0]]) / (2 * ((c[cand[rows, o[:, 1]]] - c[a]) ** 2).sum(1))
    for frac in fracs:
        free = marg <= np.quantile(marg, frac); rem = cap - np.bincount(a[~free], minlength=nt)
        if rem.min() < 0: continue
        fi = np.where(free)[0]; nf = len(fi); jj = np.take_along_axis(cand[fi], o[fi], 1); cc = np.take_along_axis(cost[fi], o[fi], 1)
        A = coo_matrix((np.ones(2 * nf * J), (np.r_[np.repeat(np.arange(nf), J), nf + jj.ravel()], np.r_[np.arange(nf * J), np.arange(nf * J)])),
                       shape=(nf + nt, nf * J)).tocsr()
        r = linprog(cc.ravel(), A_eq=A, b_eq=np.r_[np.ones(nf), rem], bounds=(0, 1), method='highs-ds')
        if r.status == 0:
            x = r.x.reshape(nf, J); a = a.copy(); a[fi] = jj[np.arange(nf), np.argmax(x, 1)]
            if (np.bincount(a, minlength=nt) == cap).all(): return a
    raise RuntimeError('assignment LP infeasible')


def tiles(p, tile, c0=None, lloyd=40, sub=None, seed=0, info=None):
    """tile label of every location; c0 = start sites (default: centroids of the min-box tiles)"""
    n = len(p); nt = n // tile; assert nt * tile == n
    if c0 is None:
        import minbbox
        c0 = np.array([p[t].mean(0) for t in minbbox.minbbox_tiles(p, tile)])
    sub = sub or max(1, n // 120000)                                   # Lloyd on a subsample of ~120k points
    s = p[np.random.default_rng(seed).permutation(n)[:n // sub]]; c = c0.copy(); w = np.zeros(nt)
    for it in range(lloyd):
        w, err, a = balance(s, c, w, len(s) / nt, iters=30 if it == 0 else 4, tol=max(2, 0.01 * len(s) / nt))
        cnt = np.bincount(a, minlength=nt); ok = cnt > 0
        cn = np.c_[np.bincount(a, s[:, 0], nt), np.bincount(a, s[:, 1], nt)] / np.maximum(cnt, 1)[:, None]
        c = np.where(ok[:, None], cn, c)
    w, err, _ = balance(p, c, w, tile, iters=40, tol=2)
    if info is not None: info.update(count_error_before_lp=err, lloyd=lloyd, sub=sub)
    return exact(p, c, w, tile)
