#!/usr/bin/env python3
"""ShortSeam ordering for 2D locations: equal-count tiles whose longest shared boundary (seam) is as short as possible.

Tiles are the cells of a power diagram (convex polygons). Site level (no locations involved): at most twelve
integer start designs (a ring of thin wall tiles plus hexagonal rows inside) are each optimized over sites and
weights, minimizing a soft-max of all seam lengths with equal tile areas, through a temperature ladder; tiles left
with too few neighbours get one edge swap between temperatures; the design with the shortest longest seam wins.
Point level: Newton steps on the power weights with the real counts, then a transportation LP on a narrow band of
boundary locations gives exactly b locations per tile. Deterministic.

Tile order (only the order of whole tiles changes; the tiles and all initial tile ranks stay the same):
  spectral (default)  tiles sorted by the Fiedler vector of the tile adjacency graph (Delaunay edges between tiles)
  hilbert             Hilbert order of tile centroids
Locations inside a tile follow Hilbert order.

The algorithm is in shortseam_core/ (research name MMI: mmi_v2.py = start designs and edge swaps, mmi.py = power
diagram, objective, optimizer and point assignment, ccvt.py = helpers), unchanged.

--density (for locations whose density varies; off by default, so the default output is unchanged): the site level
measures every tile by the integral of the point density over it (mass 1 per tile) instead of its area, and every seam
by the integral of sqrt(density) along it instead of its length (mmi_rho.py; same power diagram, soft-max objective,
temperature ladder and optimizer; equal to the default quantities for constant density). The integer start designs
assume uniform density, so the start sites are the tile centroids of two tilings computed first: an equal-count
centroidal power diagram on the locations (ccvt_nu.py) and the default ShortSeam run on density-equalized coordinates
(warp.py); both are optimized and the one with the shorter longest seam at site level is kept. A start that fails is
skipped. Point level: weights balanced relative to the local site spacing, then the transportation LP (ccvt_nu.py).

usage: shortseam.py LOCATIONS.csv OUT.order [--tile 4096] [--density] [--tile-order spectral|hilbert] [--procs N]"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS'): os.environ[_v] = '1'
import sys, json, time, argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'shortseam_core'))
import common
import mmi, mmi_v2, ccvt


def shortseam(p, tile, tile_order='spectral', procs=None):
    n = len(p)
    if p.shape[1] != 2: raise SystemExit('shortseam.py handles 2D locations only')
    if n % tile: raise SystemExit(f'{n} locations is not a multiple of the tile size {tile}')
    ccvt.B = mmi.B = tile
    t0 = time.perf_counter(); nt = n // tile
    lo, hi = p.min(0), p.max(0); unit = np.sqrt((hi - lo).prod() / nt); W, H = (hi - lo) / unit
    best, cand, results = mmi_v2.site_search(nt, W, H, procs); win = results[best]; t1 = time.perf_counter()
    labels, weights, count_error = mmi.assign_points(p, win['c'], win['w'], lo, unit, W, H); t2 = time.perf_counter()
    if tile_order == 'spectral': tord = common.spectral_tile_order(p, labels, nt)
    else: tord = common.hilbert_tile_order(p, labels, nt)
    order = common.order_from_tiles(p, labels, tord); t3 = time.perf_counter()
    assert np.array_equal(np.sort(order), np.arange(n)) and (np.bincount(labels, minlength=nt) == tile).all()
    rec = dict(n=n, tile=tile, tiles=nt, design=cand[best][0], site_lmax=win['lmax'], bound=mmi_v2.BOUND,
               gap_to_bound=win['lmax'] / mmi_v2.BOUND - 1, tile_order=tile_order,
               seconds=dict(sites=t1 - t0, points=t2 - t1, order=t3 - t2, total=t3 - t0))
    return order, labels, rec


def shortseam_density(p, tile, tile_order='spectral', procs=None):
    import mmi_rho, ccvt_nu, warp
    n = len(p); nt = n // tile; t0 = time.perf_counter(); starts = {}; skipped = {}

    def labels_of(order): lab = np.empty(n, np.int64); lab[order] = np.arange(n) // tile; return lab

    for name, make in (('count-cvt', lambda: (lambda lab: common.order_from_tiles(p, lab, common.hilbert_tile_order(p, lab, nt)))(ccvt_nu.tiles(p, tile))),
                       ('warped', lambda: shortseam(warp.transform(p), tile, 'hilbert', procs)[0])):
        try: starts[name] = labels_of(make())
        except Exception as e: skipped[name] = f'{type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ""}'[:120]
    runs = []
    for name, lab0 in starts.items():
        try: st = {}; runs.append((mmi_rho.tiles(p, tile, lab0, info=st), name, st))
        except RuntimeError as e: skipped[name] = f'site level: {e}'[:120]
    if not runs: raise SystemExit(f'shortseam.py --density: no start could be completed ({skipped})')
    labels, name, st = min(runs, key=lambda r: r[2]['site_lam_max']); t1 = time.perf_counter()
    tord = common.spectral_tile_order(p, labels, nt) if tile_order == 'spectral' else common.hilbert_tile_order(p, labels, nt)
    order = common.order_from_tiles(p, labels, tord)
    assert np.array_equal(np.sort(order), np.arange(n)) and (np.bincount(labels, minlength=nt) == tile).all()
    rec = dict(n=n, tile=tile, tiles=nt, density=True, design=f'density start={name}', site_lmax=st['site_lam_max'],
               site_lmax_per_start={r[1]: r[2]['site_lam_max'] for r in runs}, skipped_starts=skipped, tile_order=tile_order,
               seconds=dict(tiles=t1 - t0, order=time.perf_counter() - t1, total=time.perf_counter() - t0))
    return order, labels, rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('locations'); ap.add_argument('out')
    ap.add_argument('--tile', type=int, default=4096); ap.add_argument('--density', action='store_true')
    ap.add_argument('--tile-order', choices=('spectral', 'hilbert'), default='spectral')
    ap.add_argument('--procs', type=int, default=None)
    a = ap.parse_args()
    p = common.read_locations(a.locations)
    order, _, rec = (shortseam_density if a.density else shortseam)(p, a.tile, a.tile_order, a.procs)
    common.write_order(a.out, order, p, a.tile, f'shortseam tile-order={a.tile_order}' + (' density' if a.density else ''),
                       extra=[f'design {rec["design"]}', f'site_lmax {rec["site_lmax"]:.6f}'])
    print(json.dumps(rec, default=float))


if __name__ == '__main__':
    main()
