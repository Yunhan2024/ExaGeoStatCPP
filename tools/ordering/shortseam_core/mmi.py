"""MMI ordering (2D, tile 4096): minimax-interface power diagram.
Objective = the longest shared interface between two tiles (exact max rank ~ l_max^0.70 across minbbox/QSR/HexCVT).
Site level (nt sites; no points, no SVD): power diagram of the sites in the data box (mirror sites give the box sides),
regular triangulation from one 3D hull of the lifted sites. Variables = site centres + power weights; minimise the
soft-max of all interface lengths + LAM * sum (cell area - ideal)^2 with analytic gradients (L-BFGS-B, soft-max
temperature continuation), then Newton on the weights for exactly equal areas (uniform density).
Starts: hexagonal row lattice inside a boundary ring of thin cells along the four sides (wall cells long along the wall,
touching 2-3 interior cells; corner cells) over a grid of ring depths / wall-cell lengths / row counts, plain lattices,
jittered copies and random sites; stage 1 = short optimisation of all starts, stage 2 = full schedule on the best TOPK,
in parallel; keep the smallest hard l_max. Default search (MMI_SEARCH=U, 2026-11-20): only the distinct structured starts,
one orientation for a near-square box, full schedule on the best 2 (7.6-17.5% of the CPU of MMI_SEARCH=full). Point level, once: Newton power weights on the real point counts (analytic Jacobian) +
transportation LP on power costs over the smallest-margin band -> exactly B points per tile. Tiles placed by centroid Hilbert order, points inside a tile in Hilbert order.
Deterministic.  usage: mmi.py XY_CSV OUT_PREFIX [procs]      writes OUT_PREFIX_order.npy and OUT_PREFIX_run.json"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS'): os.environ[_v] = '1'
import sys, time, json
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccvt

B = ccvt.B
TAUS = tuple(float(x) for x in os.environ.get('MMI_TAUS', '0.05,0.02,0.01,0.005').split(','))
ITERS = int(os.environ.get('MMI_ITERS', '300'))                     # L-BFGS iterations per temperature
RING_T = tuple(float(x) for x in os.environ.get('MMI_RING_T', '0.5,0.6,0.7,0.8,0.9').split(','))     # ring depth (ideal sides)
RING_K = tuple(float(x) for x in os.environ.get('MMI_RING_K', '0.9,1.0,1.1,1.2').split(','))     # ring cell length = K / depth
NSTART = int(os.environ.get('MMI_NSTART', '400'))                   # stage-1 starts (short optimisation)
TOPK = int(os.environ.get('MMI_TOPK', '48'))                       # stage-2 (full temperature schedule) on the best TOPK
SEARCH = os.environ.get('MMI_SEARCH', 'U')                         # 'U': unique structured starts, one orientation if W~H, top 2 (2026-11-20 default)
                                                                    # 'full': all NSTART starts, top TOPK (MMI results of 2026-11-18)
TOPK_U = 2
STAGE1 = ((0.05, 0.02), 100)
MD = 3.0                                                            # mirror only sites within MD ideal sides of a side


def _bc(idx, val, n):
    """scatter-add (rows of val) by idx into n slots"""
    if val.ndim == 1: return np.bincount(idx, val, n)
    return np.stack([np.bincount(idx, val[:, k], n) for k in range(val.shape[1])], 1)


def _solve2(M, r):
    """batched 2x2 solve M x = r; returns x and det"""
    det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
    x0 = (M[:, 1, 1] * r[:, 0] - M[:, 0, 1] * r[:, 1]) / det; x1 = (M[:, 0, 0] * r[:, 1] - M[:, 1, 0] * r[:, 0]) / det
    return np.c_[x0, x1], det


def diagram(c, w, W, H):
    """power diagram of sites c (nt x 2, inside [0,W]x[0,H]) with weights w: real-real edges (i, j, l), cell areas A,
    hidden-site count; internal arrays kept for backward()."""
    nt = len(c); x, y = c[:, 0], c[:, 1]
    nx = [(np.ones(nt, bool), 1., 0.), (x < MD, -1., 0.), (x > W - MD, -1., 2 * W)]
    ny = [(np.ones(nt, bool), 1., 0.), (y < MD, -1., 0.), (y > H - MD, -1., 2 * H)]
    S, own, ax, ay = [], [], [], []
    for fx, axx, bx in nx:                                                        # identity first: real index = position
        for fy, ayy, by in ny:
            k = np.where(fx & fy)[0]
            S.append(np.c_[axx * x[k] + bx, ayy * y[k] + by]); own.append(k); ax.append(np.full(len(k), axx)); ay.append(np.full(len(k), ayy))
    S = np.concatenate(S); own = np.concatenate(own); ax = np.concatenate(ax); ay = np.concatenate(ay); ws = w[own]
    P3 = np.c_[S, (S ** 2).sum(1) - ws]
    for opt in ('Qt', 'QJ'):                                                      # QJ (joggle) only if Qt left a degenerate facet
        hull = ConvexHull(P3, qhull_options=opt)
        F = hull.simplices[hull.equations[:, 2] < -1e-9]                          # lower hull (no vertical facets) = regular triangulation
        a, b, cc = F[:, 0], F[:, 1], F[:, 2]
        M = 2 * np.stack([S[b] - S[a], S[cc] - S[a]], 1)
        sq = (S ** 2).sum(1); r = np.c_[sq[b] - sq[a] - ws[b] + ws[a], sq[cc] - sq[a] - ws[cc] + ws[a]]
        V, det = _solve2(M, r)
        if np.abs(det).min() > 1e-12: break
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]]), 1); fid = np.tile(np.arange(len(F)), 3)
    o = np.lexsort((E[:, 1], E[:, 0])); E, fid = E[o], fid[o]
    k = np.where((E[1:] == E[:-1]).all(1))[0]                                     # edges shared by two lower facets
    e0, f1, f2 = E[k], fid[k], fid[k + 1]
    keep = e0[:, 0] < nt                                                          # touches a real site (i < j, real index < nt)
    e0, f1, f2 = e0[keep], f1[keep], f2[keep]
    i, j = e0[:, 0], e0[:, 1]
    dv = V[f1] - V[f2]; L = np.sqrt((dv ** 2).sum(1))
    rr = j < nt; ownm = (~rr) & (own[j] == i)                                     # real-real; real-own-mirror (box side)
    use = rr | ownm
    d = np.sqrt(((S[j] - S[i]) ** 2).sum(1)); hi = (d ** 2 + ws[i] - ws[j]) / (2 * d); hj = (d ** 2 + ws[j] - ws[i]) / (2 * d)
    A = _bc(i[use], 0.5 * L[use] * hi[use], nt) + _bc(j[rr], 0.5 * L[rr] * hj[rr], nt)
    return dict(i=i[rr], j=j[rr], l=L[rr], d=d[rr], A=A, hidden=nt - len(np.unique(F[F < nt])),
                _=(nt, own, ax, ay, S, ws, F, M, det, V, f1, f2, i, j, dv, L, rr, use, d, hi, hj))


def backward(D, gl_rr, gA):
    """gradient wrt (c, w) of Phi given dPhi/dl (real-real edges, in D order) and dPhi/dA"""
    nt, own, ax, ay, S, ws, F, M, det, V, f1, f2, i, j, dv, L, rr, use, d, hi, hj = D['_']
    a, b, cc = F[:, 0], F[:, 1], F[:, 2]; ns = len(S)
    gL = np.zeros(len(L)); gL[rr] += gl_rr
    gS = np.zeros_like(S); gW = np.zeros(ns)
    n = (S[j] - S[i]) / d[:, None]                                                # dd/dS_j = n, dd/dS_i = -n
    for side, other, m_, h_ in ((i, j, use, hi), (j, i, rr, hj)):                 # A_side += 1/2 L h_side
        g = gA[own[side[m_]]]
        gL[m_] += g * 0.5 * h_[m_]
        coef = g * 0.5 * L[m_]                                                    # dPhi/dh_side
        dhdd = 0.5 - (ws[side[m_]] - ws[other[m_]]) / (2 * d[m_] ** 2)
        q = (coef * dhdd)[:, None] * n[m_]
        gS += _bc(j[m_], q, ns) - _bc(i[m_], q, ns)
        gW += _bc(side[m_], coef / (2 * d[m_]), ns) - _bc(other[m_], coef / (2 * d[m_]), ns)
    u = np.where(L[:, None] > 1e-12, dv / np.maximum(L, 1e-12)[:, None], 0.0) * gL[:, None]
    GV = _bc(f1, u, len(F)) - _bc(f2, u, len(F))
    Mt = np.transpose(M, (0, 2, 1)); U, _ = _solve2(Mt, GV)                       # u = M^-T G_v
    gS += _bc(a, 2 * (V - S[a]) * (U[:, :1] + U[:, 1:]), ns) + _bc(b, 2 * (S[b] - V) * U[:, :1], ns) + _bc(cc, 2 * (S[cc] - V) * U[:, 1:], ns)
    gW += _bc(a, U[:, 0] + U[:, 1], ns) - _bc(b, U[:, 0], ns) - _bc(cc, U[:, 1], ns)
    gc = _bc(own, gS * np.c_[ax, ay], nt); gw = _bc(own, gW, nt)
    return gc, gw


def lap(D, nt):
    """dA/dw (symmetric Laplacian; raising w_i grows cell i)"""
    g = D['l'] / (2 * D['d']); Lp = np.zeros((nt, nt))
    np.add.at(Lp, (D['i'], D['j']), -g); np.add.at(Lp, (D['j'], D['i']), -g); Lp[np.diag_indices(nt)] = -Lp.sum(1)
    return Lp


def gsolve(Lp, rhs):
    """solve Lp x = rhs on the zero-mean subspace (Lp has the constant null vector)"""
    n = len(rhs); A = Lp + np.ones((n, n)) / n
    x = np.linalg.solve(A, rhs - rhs.mean()); return x - x.mean()


def equalise(c, w, W, H, tol=1e-9, iters=12):
    """Newton on the weights: every cell area -> 1 (site units), centres fixed. Returns w, diagram (None on failure)"""
    nt = len(c)
    for _ in range(iters):
        D = diagram(c, w, W, H); err = D['A'] - 1.0
        if D['hidden'] or len(D['l']) == 0 or not np.isfinite(err).all(): return w, None
        if np.abs(err).max() < tol: return w, D
        dw = gsolve(lap(D, nt), -err)
        lim = 0.25                                                                # damp (site units^2)
        if np.abs(dw).max() > lim: dw *= lim / np.abs(dw).max()
        w = w + dw
    D = diagram(c, w, W, H)
    return w, (D if not D['hidden'] and np.abs(D['A'] - 1).max() < 1e-6 else None)


LAM = float(os.environ.get('MMI_LAM', '30'))


def penalty(x, nt, W, H, tau):
    """soft-max of interface lengths + LAM * sum (A - 1)^2 over (c, w)"""
    c, w = x[:2 * nt].reshape(nt, 2), x[2 * nt:]
    D = diagram(c, w, W, H)
    l = D['l']
    if len(l) == 0 or not (np.isfinite(l).all() and np.isfinite(D['A']).all()): return 1e3, np.zeros_like(x)   # degenerate trial point
    z = l / tau; mx = z.max(); e = np.exp(z - mx); s = e.sum()
    ea = D['A'] - 1.0; f = tau * (mx + np.log(s)) + LAM * (ea ** 2).sum() + 50.0 * D['hidden']
    gc, gw = backward(D, e / s, 2 * LAM * ea)
    return f, np.r_[gc.ravel(), gw]


def optimise(args):
    c0, W, H = args[:3]; taus = args[3] if len(args) > 3 else TAUS; its = args[4] if len(args) > 4 else ITERS
    nt = len(c0); x = np.r_[c0.ravel(), np.zeros(nt)]
    eps = 1e-3; bnds = [(eps, W - eps), (eps, H - eps)] * nt + [(None, None)] * nt
    for tau in taus:
        x = minimize(penalty, x, args=(nt, W, H, tau), jac=True, method='L-BFGS-B', bounds=bnds,
                     options=dict(maxiter=its, maxcor=30, ftol=1e-13, gtol=1e-10)).x
    c = x[:2 * nt].reshape(nt, 2); w, D = equalise(c, x[2 * nt:], W, H)
    if D is None: w, D = equalise(c, np.zeros(nt), W, H, iters=30)
    if D is None: return dict(c=c, w=w, lmax=9.9, top10=9.9, ok=False)
    return dict(c=c, w=w, lmax=float(D['l'].max()), top10=float(np.sort(D['l'])[-10:].mean()), ok=True)


# ---------------- starts ----------------
def rows_lattice(n, x0, x1, y0, y1, R):
    """n sites in R rows along x (near-even counts, the longer rows spread out), alternate rows half-offset"""
    ks = [n // R + (1 if i < n % R else 0) for i in range(R)]; ks = [ks[(i * 7) % R] for i in range(R)]
    out = []
    for r, k in enumerate(ks):
        yy = y0 + (r + 0.5) * (y1 - y0) / R; off = 0.25 if r % 2 == 0 else 0.75
        out += [(x0 + (i + off) * (x1 - x0) / k, yy) for i in range(k)]
    return np.array(out)


def ring_start(nt, W, H, t, kap, R, transpose):
    """thin wall cells (depth t, length ~kap/t) along the four sides, a corner cell in each corner, row lattice inside"""
    Wx, Hy = (H, W) if transpose else (W, H)
    g = kap / t; mx = max(1, int(round((Wx - 2 * t) / g))); my = max(1, int(round((Hy - 2 * t) / g)))
    ring = []
    for i in range(mx):
        xx = t + (i + 0.5) * (Wx - 2 * t) / mx; ring += [(xx, t / 2), (xx, Hy - t / 2)]
    for i in range(my):
        yy = t + (i + 0.5) * (Hy - 2 * t) / my; ring += [(t / 2, yy), (Wx - t / 2, yy)]
    ring += [(t / 2, t / 2), (Wx - t / 2, t / 2), (t / 2, Hy - t / 2), (Wx - t / 2, Hy - t / 2)]
    nin = nt - len(ring)
    if nin < 1 or R > nin: return None
    c = np.r_[np.array(ring), rows_lattice(nin, t, Wx - t, t, Hy - t, R)]
    return np.clip(c[:, ::-1] if transpose else c, 0.01, [W - .01, H - .01])


def starts(nt, W, H):
    """plain row lattices and boundary-ring lattices over a grid of ring depths / wall-cell lengths / row counts, both
    orientations, then jittered copies and uniform random sites up to NSTART"""
    out = []
    for tr in (False, True):
        Wx, Hy = (H, W) if tr else (W, H)
        R0 = max(1, int(round(np.sqrt(nt * Hy / Wx * 2 / np.sqrt(3)))))
        for R in range(max(1, R0 - 2), R0 + 3):                                  # plain lattice (no ring)
            c = rows_lattice(nt, 0, Wx, 0, Hy, R); out.append((f'plain R{R}{" T" if tr else ""}', np.clip(c[:, ::-1] if tr else c, 0.01, [W - .01, H - .01])))
        for t in RING_T:
            for kap in RING_K:
                Wi, Hi = Wx - 2 * t, Hy - 2 * t; g = kap / t
                nin = nt - 2 * max(1, int(round(Wi / g))) - 2 * max(1, int(round(Hi / g))) - 4
                if nin < 1: continue
                R0 = max(1, int(round(np.sqrt(nin * Hi / Wi * 2 / np.sqrt(3)))))
                for R in range(max(1, R0 - 2), R0 + 3):
                    c = ring_start(nt, W, H, t, kap, R, tr)
                    if c is not None: out.append((f'ring t{t} k{kap} R{R}{" T" if tr else ""}', c))
    base = list(out)
    for k in range(max(0, NSTART - len(base))):                                   # fill up: jittered copies and uniform random sites
        rng = np.random.default_rng(k)
        if k % 2 == 0:
            nm, c = base[(k // 2) % len(base)]
            out.append((f'{nm} jit{k}', np.clip(c + 0.15 * rng.standard_normal(c.shape), 0.01, [W - .01, H - .01])))
        else:
            out.append((f'rand{k}', np.c_[rng.uniform(0.05, W - 0.05, nt), rng.uniform(0.05, H - 0.05, nt)]))
    return out


def starts_u(nt, W, H):
    """rule U: the structured starts only (plain + ring; no jittered/random fill), byte-identical configurations removed,
    and only one orientation when the box is near square (|W/H - 1| < 0.05)"""
    sq = abs(W / H - 1) < 0.05; seen = set(); out = []
    for nm, c in starts(nt, W, H):
        if ' jit' in nm or nm.startswith('rand') or (sq and ' T' in nm): continue
        key = np.round(c, 9).tobytes()
        if key not in seen: seen.add(key); out.append((nm, c))
    return out


def run(csv, prefix, procs=None):
    from multiprocessing import Pool
    t0 = time.perf_counter(); p = read_xy(csv); n = len(p); nt = n // B; assert nt * B == n
    lo, hi = p.min(0), p.max(0); s0 = np.sqrt((hi - lo).prod() / nt); W, H = (hi - lo) / s0
    st = starts(nt, W, H) if SEARCH == 'full' else starts_u(nt, W, H); topk = TOPK if SEARCH == 'full' else TOPK_U
    t1 = time.perf_counter(); procs = procs or min(len(st), os.cpu_count() - 1)
    with Pool(procs) as pool:
        r1 = pool.map(optimise, [(c, W, H) + STAGE1 for _, c in st], chunksize=4)
        top = sorted(range(len(st)), key=lambda k: (not r1[k]['ok'], round(r1[k]['lmax'], 6), round(r1[k]['top10'], 6), k))[:topk]
        r2 = pool.map(optimise, [(r1[k]['c'], W, H) for k in top], chunksize=1)
    res = [dict(r, start=k) for k, r in zip(top, r2)]
    t2 = time.perf_counter()
    best = min(range(len(res)), key=lambda k: (not res[k]['ok'], round(res[k]['lmax'], 6), round(res[k]['top10'], 6), k))
    lab, w, newton_err = assign_points(p, res[best]['c'], res[best]['w'], lo, s0, W, H)
    t3 = time.perf_counter()
    o = ccvt.order_from_labels(p, lab); t4 = time.perf_counter()
    assert np.array_equal(np.sort(o), np.arange(n))
    np.save(f'{prefix}_order.npy', o)
    rec = dict(csv=str(csv), n=n, tile=B, procs=procs, chosen_start=st[res[best]['start']][0], taus=TAUS, iters=ITERS, stage1_cfg=STAGE1, search=SEARCH, nstart=len(st), topk=topk,
               newton_count_err=newton_err,
               seconds=dict(read=t1 - t0, site_opt=t2 - t1, assign=t3 - t2, order=t4 - t3, total=t4 - t0),
               site_lmax=res[best]['lmax'], site_top10=res[best]['top10'], sites=(res[best]['c'] * s0 + lo).tolist(),
               stage1=[dict(name=s[0], lmax=r['lmax'], top10=r['top10'], ok=r['ok']) for s, r in zip(st, r1)],
               stage2=[dict(name=st[r['start']][0], lmax=r['lmax'], top10=r['top10'], ok=r['ok']) for r in res])
    json.dump(rec, open(f'{prefix}_run.json', 'w'), indent=1)
    print(f'total {t4 - t0:.2f} s (read {t1 - t0:.2f}, site opt {t2 - t1:.2f} [{len(st)} starts -> top {len(top)}], assign {t3 - t2:.2f}, order {t4 - t3:.2f}); '
          f'chosen {st[res[best]["start"]][0]} site l_max {res[best]["lmax"]:.3f} top10 {res[best]["top10"]:.3f}', flush=True)
    return o


def assign_points(p, cs, ws, lo, s0, W, H, K=12, J=3, tol=2):
    """exactly B points per tile: Newton on the power weights with the analytic count Jacobian (B x dA/dw of the site
    diagram), then a transportation LP on power costs over the smallest-margin band (power diagram of the LP duals)"""
    from scipy.spatial import cKDTree
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    n, nt = len(p), len(cs); c = lo + cs * s0; w = ws.copy(); K = min(nt, K)
    _, cand = cKDTree(c).query(p, k=K); d2 = ((p[:, None, :] - c[cand]) ** 2).sum(-1); rows = np.arange(n)
    for it in range(8):
        cost = d2 - w[cand] * s0 ** 2; a = cand[rows, cost.argmin(1)]; err = np.bincount(a, minlength=nt) - B
        if np.abs(err).max() <= tol: break
        D = diagram(cs, w, W, H); w = w + gsolve(B * lap(D, nt), -err.astype(float))
    newton_err = int(np.abs(err).max())
    cost = d2 - w[cand] * s0 ** 2; o = np.argsort(cost, 1); a = cand[rows, o[:, 0]]
    marg = np.take_along_axis(cost, o[:, 1:2], 1)[:, 0] - np.take_along_axis(cost, o[:, :1], 1)[:, 0]
    marg = marg / (2 * np.linalg.norm(c[cand[rows, o[:, 1]]] - c[a], axis=1))      # geometric distance to the nearest power line
    for frac in (0.005, 0.01, 0.02, 0.05, 0.1):
        free = marg < np.quantile(marg, frac); rem = B - np.bincount(a[~free], minlength=nt)
        fi = np.where(free)[0]; nf = len(fi)
        jj = np.take_along_axis(cand[fi], o[fi][:, :J], 1); cc = np.take_along_axis(cost[fi], o[fi][:, :J], 1)
        A = coo_matrix((np.ones(2 * nf * J), (np.r_[np.repeat(np.arange(nf), J), nf + jj.ravel()], np.r_[np.arange(nf * J), np.arange(nf * J)])),
                       shape=(nf + nt, nf * J)).tocsr()
        r = linprog(cc.ravel(), A_eq=A, b_eq=np.r_[np.ones(nf), rem], bounds=(0, 1), method='highs-ds')
        if r.status == 0:
            x = r.x.reshape(nf, J); assert ((x > 1e-9) & (x < 1 - 1e-9)).sum() == 0
            a = a.copy(); a[fi] = jj[np.arange(nf), np.argmax(x, 1)]
            assert (np.bincount(a, minlength=nt) == B).all()
            return a, w, newton_err
    raise RuntimeError('band LP infeasible')


def read_xy(path):
    txt = open(path).read(); txt = txt[txt.index('\n') + 1:]
    p = np.fromstring(txt.replace('\n', ','), sep=',').reshape(-1, 2)
    assert np.isfinite(p).all(); return p


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
