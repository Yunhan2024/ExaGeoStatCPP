"""Density-equalizing map of 2D locations (diffusion cartogram, Gastner & Newman 2004): one preprocessing step shared by
the ordering tools, so that tools built for uniform density run unchanged on non-uniform locations.
The point density on a G x G grid of the bounding box (mean 1, blended with LAM of uniform so that empty regions stay
finite, Gaussian-smoothed over SIG cells) evolves by the heat equation with reflecting walls until it is uniform; every
location is carried by the velocity field v = -grad(rho) / rho (midpoint rule, step limited to CFL grid cells). The map
is continuous, keeps the box and its sides, and is close to the identity for uniform locations. The orderings are
computed on the mapped locations; the permutation is then used for the original ones.
G = sqrt(n / PPC). Deterministic.    usage: warp.py XY.csv OUT.csv"""
import os, sys
import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import map_coordinates

LAM = float(os.environ.get('WARP_LAM', '0.02'))      # share of uniform density blended in
PPC = float(os.environ.get('WARP_PPC', '8'))         # mean locations per grid cell
SIG = float(os.environ.get('WARP_SIG', '1.5'))       # initial smoothing, grid cells
CFL = 0.5                                            # largest move of a location per step, grid cells
TOL = 1e-3                                           # stop when |rho - 1| < TOL everywhere


def _velocity(a, lam2, t, G):
    """(vx, vy) on the cell-centre grid, padded by one mirrored cell (normal component odd), and max |rho - 1|"""
    rho = idctn(a * np.exp(-lam2 * t), type=2, norm='ortho')
    r = np.pad(rho, 1, mode='edge')                              # zero normal gradient at the walls
    gx = (r[2:, 1:-1] - r[:-2, 1:-1]) * (G / 2); gy = (r[1:-1, 2:] - r[1:-1, :-2]) * (G / 2)
    vx, vy = -gx / rho, -gy / rho
    px = np.pad(vx, 1, mode='edge'); px[0, :] = -px[1, :]; px[-1, :] = -px[-2, :]
    py = np.pad(vy, 1, mode='edge'); py[:, 0] = -py[:, 1]; py[:, -1] = -py[:, -2]
    return px, py, float(np.abs(rho - 1).max())


def _at(px, py, u, G):
    c = u.T * G + 0.5                                            # padded index of cell centre i is i + 1
    return np.stack([map_coordinates(px, c, order=1, mode='nearest'), map_coordinates(py, c, order=1, mode='nearest')], 1)


def transform(p, lam=LAM, ppc=PPC, sig=SIG, info=None):
    """locations (n, 2) -> mapped locations in the unit square (same row order)"""
    lo, hi = p.min(0), p.max(0); u = (p - lo) / (hi - lo); n = len(u)
    G = max(16, int(np.sqrt(n / ppc)))
    H, _, _ = np.histogram2d(u[:, 0], u[:, 1], bins=G, range=[[0, 1], [0, 1]])
    rho = (1 - lam) * H / H.mean() + lam
    k = np.arange(G); lam2 = np.pi ** 2 * (k[:, None] ** 2 + k[None, :] ** 2)      # heat decay rates of the cosine modes
    a = dctn(rho, type=2, norm='ortho')
    t = 0.5 * (sig / G) ** 2; steps = 0                           # Gaussian of width sig cells = heat flow up to this time
    while True:
        px, py, dev = _velocity(a, lam2, t, G)
        if dev < TOL: break
        v1 = _at(px, py, u, G); vmax = float(np.abs(v1).max())
        dt = min(CFL / G / max(vmax, 1e-12), 0.05)
        qx, qy, _ = _velocity(a, lam2, t + dt / 2, G)
        u = np.clip(u + dt * _at(qx, qy, np.clip(u + 0.5 * dt * v1, 0, 1), G), 0, 1)
        t += dt; steps += 1
    if info is not None: info.update(grid=G, steps=steps, t_end=t, lam=lam, ppc=ppc, sig=sig)
    return u


def diagnostics(p, u, tile, seed=0, m=2000, k=40):
    """count spread over cells holding `tile` locations on average (before / after), and the local anisotropy of the map:
    ratio of the singular values of the best linear fit over the k nearest neighbours of m sample locations"""
    from scipy.spatial import cKDTree
    n = len(p); g = max(2, int(round(np.sqrt(n / tile))))
    def cv(x):
        lo, hi = x.min(0), x.max(0)
        H, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=g, range=[[lo[0], hi[0]], [lo[1], hi[1]]]); return float(H.std() / H.mean())
    rng = np.random.default_rng(seed); s = rng.choice(n, min(m, n), replace=False)
    _, nb = cKDTree(p).query(p[s], k=k); ratio = []
    for i, row in zip(s, nb):
        A = p[row] - p[row].mean(0); Bm = u[row] - u[row].mean(0)
        J = np.linalg.lstsq(A, Bm, rcond=None)[0]; sv = np.linalg.svd(J, compute_uv=False); ratio.append(sv[0] / sv[1])
    ratio = np.array(ratio)
    return dict(count_cv_before=cv(p), count_cv_after=cv(u), poisson_cv=float(1 / np.sqrt(tile)),
                aniso_median=float(np.median(ratio)), aniso_p90=float(np.percentile(ratio, 90)), aniso_max=float(ratio.max()))


if __name__ == '__main__':
    p = np.loadtxt(sys.argv[1], delimiter=',', skiprows=1, ndmin=2); info = {}
    u = transform(p, info=info)
    np.savetxt(sys.argv[2], u, delimiter=',', header='x,y', comments='', fmt='%.17g'); print(info)
