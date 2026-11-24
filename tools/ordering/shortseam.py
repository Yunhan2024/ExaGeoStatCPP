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

usage: shortseam.py LOCATIONS.csv OUT.order [--tile 4096] [--tile-order spectral|hilbert] [--procs N]"""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('locations'); ap.add_argument('out')
    ap.add_argument('--tile', type=int, default=4096)
    ap.add_argument('--tile-order', choices=('spectral', 'hilbert'), default='spectral')
    ap.add_argument('--procs', type=int, default=None)
    a = ap.parse_args()
    p = common.read_locations(a.locations)
    order, _, rec = shortseam(p, a.tile, a.tile_order, a.procs)
    common.write_order(a.out, order, p, a.tile, f'shortseam tile-order={a.tile_order}',
                       extra=[f'design {rec["design"]}', f'site_lmax {rec["site_lmax"]:.6f}'])
    print(json.dumps(rec, default=float))


if __name__ == '__main__':
    main()
