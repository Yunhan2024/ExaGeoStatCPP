"""ShortSeam's site level with the point density in it (shortseam.py --density).
Same model as mmi.py (power diagram of nt sites in the data box, variables = sites + power weights, soft-max
of the seam lengths + LAM * sum of squared mass errors, L-BFGS-B with analytic gradients through a temperature ladder,
Newton on the weights for exactly equal masses), with two substitutions:
    cell area A_i         ->  mass  M_i  = integral of rho over the cell            (rho: locations per tile and unit area)
    seam length l_ij      ->  lam_ij     = integral of sqrt(rho) along the shared edge
For constant rho both are the original quantities. rho = histogram of the locations on a G x G grid of the box (G =
sqrt(n / 8)), Gaussian-smoothed over 1.5 cells, floored at 2% of the mean, bilinear in between. Integrals by quadrature
(25 points per fan triangle site-edge, 12 per edge). Start: the sites of a given tiling (its tile centroids), weights 0.
Point level: ccvt_nu.balance + ccvt_nu.exact (counts on the locations themselves).
usage (module): labels, info = mmi_rho.tiles(p, tile, labels0)"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
import mmi, ccvt_nu

LAM, TAUS, ITERS = 30.0, (0.05, 0.02, 0.01, 0.005), 200
_m = 5                                                           # fan triangle split into _m^2 sub-triangles, their centroids
_bary = np.array([((i + (1 + k) / 3) / _m, (j + (1 + k) / 3) / _m) for i in range(_m) for j in range(_m - i) for k in (0, 1) if i + j + k < _m])
_TS, _TT = _bary[:, 0], _bary[:, 1]; _TW = np.full(len(_bary), 1.0 / len(_bary))
_EU = (np.arange(12) + 0.5) / 12; _EW = np.full(12, 1.0 / 12)


class Density:
    def __init__(self, p, tile, W, H, lo, s0):
        n = len(p); G = max(16, int(np.sqrt(n / 8))); u = (p - lo) / s0
        Hc, xe, ye = np.histogram2d(u[:, 0], u[:, 1], bins=G, range=[[0, W], [0, H]])
        rho = gaussian_filter(Hc / tile / ((W / G) * (H / G)), 1.5, mode='nearest'); rho = np.maximum(rho, 0.02 * rho.mean())
        self.r, self.G, self.W, self.H = rho, G, W, H

    def __call__(self, x):
        """rho and its gradient at points x (..., 2): bilinear between the cell centres, constant beyond the outer centres"""
        G, r = self.G, self.r
        gx = np.clip(x[..., 0] / self.W * G - 0.5, 0, G - 1 - 1e-9); gy = np.clip(x[..., 1] / self.H * G - 0.5, 0, G - 1 - 1e-9)
        i = gx.astype(int); j = gy.astype(int); fx = gx - i; fy = gy - j
        r00, r10, r01, r11 = r[i, j], r[i + 1, j], r[i, j + 1], r[i + 1, j + 1]
        v = (1 - fx) * (1 - fy) * r00 + fx * (1 - fy) * r10 + (1 - fx) * fy * r01 + fx * fy * r11
        dx = ((1 - fy) * (r10 - r00) + fy * (r11 - r01)) * (G / self.W); dy = ((1 - fx) * (r01 - r00) + fx * (r11 - r10)) * (G / self.H)
        return v, np.stack([dx, dy], -1)


def evaluate(c, w, W, H, rho):
    """masses, weighted seam lengths (real-real edges) and everything the gradient needs"""
    D = mmi.diagram(c, w, W, H)
    nt, own, ax, ay, S, ws, F, M, det, V, f1, f2, i, j, dv, L, rr, use, d, hi, hj = D['_']
    P, Q = V[f1], V[f2]
    xe = P[:, None, :] + _EU[None, :, None] * (Q - P)[:, None, :]; r, gr = rho(xe); sq = np.sqrt(r)
    lam = L * (sq * _EW).sum(1); rbar_e = (r * _EW).sum(1)
    mass = np.zeros(nt); tri = []
    for side, m_, h_ in ((i, use, hi), (j, rr, hj)):
        A0 = S[side[m_]]; x = A0[:, None, :] + _TS[None, :, None] * (P[m_] - A0)[:, None, :] + _TT[None, :, None] * (Q[m_] - A0)[:, None, :]
        rt, gt = rho(x); a = 0.5 * L[m_] * h_[m_]; rb = (rt * _TW).sum(1)
        mass += np.bincount(own[side[m_]], a * rb, nt); tri.append((a, rb, gt))
    return dict(D=D, lam=lam, mass=mass, sq=sq, gr=gr, rbar_e=rbar_e, tri=tri)


def gradient(Ev, g_lam, g_mass):
    """gradient wrt (c, w) given dPhi/d lam (all edges; zero outside real-real) and dPhi/d mass (nt)"""
    D = Ev['D']; nt, own, ax, ay, S, ws, F, M, det, V, f1, f2, i, j, dv, L, rr, use, d, hi, hj = D['_']
    a_, b_, cc = F[:, 0], F[:, 1], F[:, 2]; ns = len(S); nF = len(F); bc = mmi._bc
    gL = g_lam * (Ev['sq'] * _EW).sum(1)                                       # d lam = dL * mean sqrt(rho) + ...
    gsq = Ev['gr'] / (2 * Ev['sq'][..., None])                                 # grad sqrt(rho) at the edge points
    k = (g_lam * L)[:, None]
    GV = bc(f1, k * (gsq * (_EW * (1 - _EU))[None, :, None]).sum(1), nF) + bc(f2, k * (gsq * (_EW * _EU)[None, :, None]).sum(1), nF)
    gS = np.zeros_like(S); gW = np.zeros(ns)
    n = (S[j] - S[i]) / d[:, None]
    for (side, other, m_, h_), (a, rb, gt) in zip(((i, j, use, hi), (j, i, rr, hj)), Ev['tri']):
        gm = g_mass[own[side[m_]]]; g = gm * rb                                 # coefficient on d(signed fan area)
        gL[m_] += g * 0.5 * h_[m_]
        coef = g * 0.5 * L[m_]
        dhdd = 0.5 - (ws[side[m_]] - ws[other[m_]]) / (2 * d[m_] ** 2)
        q = (coef * dhdd)[:, None] * n[m_]
        gS += bc(j[m_], q, ns) - bc(i[m_], q, ns)
        gW += bc(side[m_], coef / (2 * d[m_]), ns) - bc(other[m_], coef / (2 * d[m_]), ns)
        k2 = (gm * a)[:, None]                                                  # density varies inside the fan triangle
        gS += bc(side[m_], k2 * (gt * (_TW * (1 - _TS - _TT))[None, :, None]).sum(1), ns)
        GV += bc(f1[m_], k2 * (gt * (_TW * _TS)[None, :, None]).sum(1), nF) + bc(f2[m_], k2 * (gt * (_TW * _TT)[None, :, None]).sum(1), nF)
    u = np.where(L[:, None] > 1e-12, dv / np.maximum(L, 1e-12)[:, None], 0.0) * gL[:, None]
    GV += bc(f1, u, nF) - bc(f2, u, nF)
    Mt = np.transpose(M, (0, 2, 1)); U, _ = mmi._solve2(Mt, GV)
    gS += bc(a_, 2 * (V - S[a_]) * (U[:, :1] + U[:, 1:]), ns) + bc(b_, 2 * (S[b_] - V) * U[:, :1], ns) + bc(cc, 2 * (S[cc] - V) * U[:, 1:], ns)
    gW += bc(a_, U[:, 0] + U[:, 1], ns) - bc(b_, U[:, 0], ns) - bc(cc, U[:, 1], ns)
    return bc(own, gS * np.c_[ax, ay], nt), bc(own, gW, nt)


def penalty(x, nt, W, H, tau, rho):
    c, w = x[:2 * nt].reshape(nt, 2), x[2 * nt:]
    Ev = evaluate(c, w, W, H, rho); rr = Ev['D']['_'][16]; lam = Ev['lam'][rr]
    if len(lam) == 0 or not (np.isfinite(lam).all() and np.isfinite(Ev['mass']).all()): return 1e3, np.zeros_like(x)
    z = lam / tau; mx = z.max(); e = np.exp(z - mx); s = e.sum(); em = Ev['mass'] - 1.0
    f = tau * (mx + np.log(s)) + LAM * (em ** 2).sum() + 50.0 * Ev['D']['hidden']
    g_lam = np.zeros(len(Ev['lam'])); g_lam[rr] = e / s
    gc, gw = gradient(Ev, g_lam, 2 * LAM * em)
    return f, np.r_[gc.ravel(), gw]


def equalise(c, w, W, H, rho, tol=1e-6, iters=30):
    """Newton on the weights: every cell mass -> 1"""
    nt = len(c)
    for _ in range(iters):
        Ev = evaluate(c, w, W, H, rho); D = Ev['D']; err = Ev['mass'] - 1.0
        if D['hidden'] or not np.isfinite(err).all(): return w, None
        if np.abs(err).max() < tol: return w, Ev
        _, own, _, _, _, _, _, _, _, _, _, _, i, j, _, L, rr, _, d, _, _ = D['_']
        g = (L * Ev['rbar_e'])[rr] / (2 * d[rr]); Lp = np.zeros((nt, nt))
        np.add.at(Lp, (i[rr], j[rr]), -g); np.add.at(Lp, (j[rr], i[rr]), -g); Lp[np.diag_indices(nt)] = -Lp.sum(1)
        dw = mmi.gsolve(Lp, -err); lim = 0.25 * (d[rr] ** 2).min()
        if np.abs(dw).max() > lim: dw *= lim / np.abs(dw).max()
        w = w + dw
    Ev = evaluate(c, w, W, H, rho)
    return w, (Ev if not Ev['D']['hidden'] and np.abs(Ev['mass'] - 1).max() < 1e-3 else None)


def tiles(p, tile, lab0, info=None):
    n = len(p); nt = n // tile; lo, hi = p.min(0), p.max(0); s0 = np.sqrt((hi - lo).prod() / nt); W, H = (hi - lo) / s0
    rho = Density(p, tile, W, H, lo, s0)
    c = np.array([p[lab0 == k].mean(0) for k in range(nt)]); c = np.clip((c - lo) / s0, 1e-3, [W - 1e-3, H - 1e-3])
    w, Ev = equalise(c, np.zeros(nt), W, H, rho)
    rr = lambda Ev: Ev['lam'][Ev['D']['_'][16]]
    start = float(rr(Ev).max()) if Ev is not None else float('nan')
    x = np.r_[c.ravel(), w]; eps = 1e-3; bnds = [(eps, W - eps), (eps, H - eps)] * nt + [(None, None)] * nt
    for tau in TAUS:
        x = minimize(penalty, x, args=(nt, W, H, tau, rho), jac=True, method='L-BFGS-B', bounds=bnds,
                     options=dict(maxiter=ITERS, maxcor=30, ftol=1e-13, gtol=1e-10)).x
    c = x[:2 * nt].reshape(nt, 2); w, Ev = equalise(c, x[2 * nt:], W, H, rho)
    if Ev is None: w = x[2 * nt:]; Ev = evaluate(c, w, W, H, rho)     # masses only approximately equal; the point level fixes the counts
    if info is not None: info.update(site_lam_max_start=start, site_lam_max=float(rr(Ev).max()), site_mass_error=float(np.abs(Ev['mass'] - 1).max()))
    cp = lo + c * s0; wp, err, _ = ccvt_nu.balance(p, cp, w * s0 ** 2, tile, iters=40, tol=2)
    return ccvt_nu.exact(p, cp, wp, tile)
