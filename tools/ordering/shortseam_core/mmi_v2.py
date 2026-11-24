"""MMI v2 (2D, tile 4096): same model and same optimiser as mmi.py, with a tidier way of choosing the start.

What stays the same (imported from mmi.py)
    Every tile has a site (x, y) and a weight w. A location belongs to the site with the smallest
    |location - site|^2 - w, so the tiles are convex polygons that fill the box (a power diagram).
    The optimiser moves sites and weights so that the longest edge shared by two tiles becomes as short as possible
    while every tile keeps area 1; 'temperature' controls how sharply it focuses on the longest edge.
    The points are then assigned to the tiles (exactly 4096 each) and ordered along a Hilbert curve.

What is new: the start
    mmi.py lays out 35-105 starts from a grid of continuous knobs, pushes each a little, keeps the best 2 and pushes
    those again. Here:

    1. Candidates are integer designs. A design is: how many wall tiles on each side of the box (m), and how many
       rows inside (R). The depth of the ring of wall tiles is not a knob: it follows from 'a wall tile has area 1'.
       m and R each take the integer nearest to their ideal value and its two neighbours. Row lattices without a
       ring are included the same way. At most 12 candidates.
    2. Every candidate is pushed once, down the whole temperature ladder. After each temperature the tiles'
       neighbours are counted. A tile with too few neighbours must have an edge longer than the bound 0.7624
       (see MIN_NEIGHBOURS), and the optimiser cannot change who is next to whom. So such a tile is given one more
       neighbour by swapping one edge, the areas are made equal again, and the push continues.
    3. The candidate with the shortest longest edge wins. Its distance from 0.7624 says how far it can be from the
       best straight-edge layout.

    There is no ranking of half-pushed candidates, no 'keep 2', no second push and no penalty constant.

Where each number comes from
    derived      IDEAL_WALL_TILE_LENGTH, the hexagonal row count, MIN_NEIGHBOURS, BOUND
    reasoned     WINDOW = 1: the ideal values are approximate, so one integer either side is tried as well
    budget       the temperature ladder and the steps per temperature (mmi.TAUS, mmi.ITERS): more steps give
                 slightly better layouts and cost proportionally more; the ladder values themselves matter little
    resolution   SWAP_FRACTIONS, SWAP_BISECTIONS: how finely an edge swap is located

Validation (site level, SUMMARY.md 2026-11-23): within 1% of the best of 400 mmi.py starts in 22 of 25 tile counts
(mmi.py's default rule: 17 of 25); it misses at very few tiles (10 tiles: 10.8% longer; 15 and 192 tiles: 1.2-1.3%).

usage: mmi_v2.py XY_CSV OUT_PREFIX [procs]      writes OUT_PREFIX_order.npy and OUT_PREFIX_run.json
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_v] = '1'                       # one BLAS thread per process: the candidates run in parallel
import sys, time, json, math
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mmi, ccvt

# Lengths are in units where one tile has area 1.
BOUND = 0.7624                  # no straight-edge equal-area layout of a rectangle has a shorter longest edge:
                                # side of the regular pentagon of area 1 (SUMMARY.md, section 2)
IDEAL_WALL_TILE_LENGTH = 1.4    # a wall tile with 4 neighbours has its shortest longest edge when it is sqrt(2) long
                                # and 1/sqrt(2) deep; 1.4 is that value as used in the validation runs
HEX_ROWS = 2 / math.sqrt(3)     # rows of regular hexagons in a box w x h holding n tiles: sqrt(n * h / w * HEX_ROWS)
WINDOW = 1                      # integers tried on either side of the nearest one
# Fewest neighbours a tile may have, by the number of box sides it touches (0 interior, 1 wall, 2 corner).
# With one neighbour fewer, an area-1 tile must have an edge of at least 1.00, 0.877 and 0.910 respectively.
MIN_NEIGHBOURS = {0: 5, 1: 4, 2: 3}
SWAP_FRACTIONS = np.arange(0.1, 0.85, 0.1)      # how far two tiles are moved towards each other when an edge is swapped
SWAP_BISECTIONS = 8


# ---------------------------------------------------------------------------------------------------------------
# 1. candidates: integer designs
# ---------------------------------------------------------------------------------------------------------------
def row_lattice(n, x0, x1, y0, y1, rows):
    """n sites in `rows` staggered rows (like bricks) inside the box [x0, x1] x [y0, y1]"""
    per_row = [n // rows + (1 if r < n % rows else 0) for r in range(rows)]
    if math.gcd(7, rows) == 1:                                  # spread the longer rows over the height
        per_row = [per_row[(r * 7) % rows] for r in range(rows)]
    sites = []
    for r, k in enumerate(per_row):
        y = y0 + (r + 0.5) * (y1 - y0) / rows
        shift = 0.25 if r % 2 == 0 else 0.75                    # neighbouring rows are offset by half a tile
        sites += [(x0 + (i + shift) * (x1 - x0) / k, y) for i in range(k)]
    return np.array(sites)


def ring_depths(W, H, mx, my):
    """depth of the wall tiles on the left/right (tx) and bottom/top (ty) so that each has area 1, or None"""
    tx = ty = 0.7
    for _ in range(200):
        if W - 2 * tx <= 0 or H - 2 * ty <= 0:
            return None
        ty_new, tx_new = mx / (W - 2 * tx), my / (H - 2 * ty)
        if abs(tx_new - tx) + abs(ty_new - ty) < 1e-13:
            break
        tx, ty = tx_new, ty_new
    else:
        return None
    return (tx, ty) if 0 < tx < W / 4 and 0 < ty < H / 4 else None


def ideal_rows(nt, W, H, mx, my):
    """row count that makes the interior tiles regular hexagons, for a ring with (mx, my) wall tiles (0, 0 = no ring)"""
    if mx == 0 and my == 0:
        return max(1, int(round(math.sqrt(nt * H / W * HEX_ROWS))))
    depths = ring_depths(W, H, mx, my)
    inside = nt - 4 - 2 * mx - 2 * my
    if depths is None or inside < 1:
        return None
    tx, ty = depths
    return max(1, int(round(math.sqrt(inside * (H - 2 * ty) / (W - 2 * tx) * HEX_ROWS))))


def design_sites(nt, W, H, mx, my, rows):
    """sites of one design: a tile in each corner, mx wall tiles on the bottom and on the top, my on the left and on
    the right, and `rows` staggered rows inside; mx = my = 0 means staggered rows in the whole box. None if impossible."""
    if mx == 0 and my == 0:
        if not 1 <= rows <= nt:
            return None
        sites = row_lattice(nt, 0, W, 0, H, rows)
    else:
        depths = ring_depths(W, H, mx, my)
        inside = nt - 4 - 2 * mx - 2 * my
        if depths is None or inside < 1 or not 1 <= rows <= inside:
            return None
        tx, ty = depths
        ring = [(tx / 2, ty / 2), (W - tx / 2, ty / 2), (tx / 2, H - ty / 2), (W - tx / 2, H - ty / 2)]
        for i in range(mx):
            x = tx + (i + 0.5) * (W - 2 * tx) / mx
            ring += [(x, ty / 2), (x, H - ty / 2)]
        for i in range(my):
            y = ty + (i + 0.5) * (H - 2 * ty) / my
            ring += [(tx / 2, y), (W - tx / 2, y)]
        sites = np.r_[np.array(ring), row_lattice(inside, tx, W - tx, ty, H - ty, rows)]
    if len(sites) != nt:
        return None
    sites = np.clip(sites, 0.01, [W - 0.01, H - 0.01])
    return sites if len(np.unique(np.round(sites, 9), axis=0)) == nt else None


def candidates(nt, W, H):
    """the integer designs that are pushed: [(label, mx, my, rows, sites), ...]"""
    out = []
    m0x = int(round((W - IDEAL_WALL_TILE_LENGTH) / IDEAL_WALL_TILE_LENGTH))
    m0y = int(round((H - IDEAL_WALL_TILE_LENGTH) / IDEAL_WALL_TILE_LENGTH))
    offsets = range(-WINDOW, WINDOW + 1)
    for d in offsets:                                           # ring designs: the same offset on both axes
        mx, my = m0x + d, m0y + d
        if mx < 1 or my < 1:
            continue
        r0 = ideal_rows(nt, W, H, mx, my)
        if r0 is None:
            continue
        for dr in offsets:
            sites = design_sites(nt, W, H, mx, my, r0 + dr)
            if sites is not None:
                out.append((f'ring m({mx},{my}) R{r0 + dr}', mx, my, r0 + dr, sites))
    r0 = ideal_rows(nt, W, H, 0, 0)
    for dr in offsets:                                          # row lattices without a ring
        sites = design_sites(nt, W, H, 0, 0, r0 + dr)
        if sites is not None:
            out.append((f'rows R{r0 + dr}', 0, 0, r0 + dr, sites))
    return out


# ---------------------------------------------------------------------------------------------------------------
# 2. pushing one candidate, with edge swaps for tiles that have too few neighbours
# ---------------------------------------------------------------------------------------------------------------
def optimise_at(x, nt, W, H, tau):
    """mmi.py's optimiser at one temperature; x = all site coordinates followed by all weights"""
    eps = 1e-3
    bounds = [(eps, W - eps), (eps, H - eps)] * nt + [(None, None)] * nt
    options = dict(maxiter=mmi.ITERS, maxcor=30, ftol=1e-13, gtol=1e-10)
    return minimize(mmi.penalty, x, args=(nt, W, H, tau), jac=True, method='L-BFGS-B', bounds=bounds, options=options).x


def equal_area_layout(x, nt, W, H):
    """adjust the weights (sites fixed) until every tile has area exactly 1; returns (sites, weights, diagram) or None"""
    c = x[:2 * nt].reshape(nt, 2)
    w, D = mmi.equalise(c, x[2 * nt:], W, H)
    if D is None:
        w, D = mmi.equalise(c, np.zeros(nt), W, H, iters=30)
    return None if D is None else (c, w, D)


def neighbour_counts(D, W, H):
    """per tile: number of box sides it touches, and number of tiles it shares an edge with"""
    nt, _, _, _, S = D['_'][:5]
    i, j, real_real, used = D['_'][12], D['_'][13], D['_'][16], D['_'][17]
    sides = [set() for _ in range(nt)]
    wall_edges = used & ~real_real                              # edges between a tile and its mirror image = box sides
    for tile, mirror in zip(i[wall_edges], j[wall_edges]):
        x, y = S[mirror]
        sides[tile].add('L' if x < 0 else 'R' if x > W else 'B' if y < 0 else 'T')
    walls = np.array([len(s) for s in sides])
    neighbours = np.bincount(np.r_[D['i'], D['j']], minlength=nt)
    return walls, neighbours


def minimum_for(walls_touched):
    return MIN_NEIGHBOURS[min(int(walls_touched), 2)]


def tiles_with_too_few_neighbours(walls, neighbours):
    return [a for a in range(len(walls)) if neighbours[a] < minimum_for(walls[a])]


def find_swaps(D, walls, neighbours):
    """For each tile A with too few neighbours, one edge swap that gives it a new neighbour.
    Around A, two consecutive neighbours B and C share an edge; behind that edge sits a tile D. Removing the edge B-C
    makes A and D neighbours. B and C must be able to lose a neighbour, and of all such edges the shortest is taken.
    Returns [(A, D, length of the edge B-C, midpoint of the edge B-C), ...]."""
    nt = len(walls)
    triangles, corners = D['_'][6], D['_'][9]                   # each triangle = three mutually adjacent sites; its
    by_pair = {}                                                # corner = the point where their three tiles meet
    for t, (a, b, c) in enumerate(triangles):
        for u, v in ((a, b), (b, c), (a, c)):
            by_pair.setdefault((min(u, v), max(u, v)), []).append(t)
    swaps = []
    for A in tiles_with_too_few_neighbours(walls, neighbours):
        best = None
        for t in np.where((triangles == A).any(1))[0]:
            B, C = [int(s) for s in triangles[t] if s != A]
            if B >= nt or C >= nt:                              # mirror images: that edge lies on the box side
                continue
            if neighbours[B] - 1 < minimum_for(walls[B]) or neighbours[C] - 1 < minimum_for(walls[C]):
                continue
            behind = [u for u in by_pair[(min(B, C), max(B, C))] if u != t]
            if len(behind) != 1:
                continue
            Dd = [int(s) for s in triangles[behind[0]] if s != B and s != C][0]
            if Dd >= nt or Dd == A:
                continue
            length = float(np.linalg.norm(corners[t] - corners[behind[0]]))
            if best is None or length < best[2]:
                best = (A, Dd, length, (corners[t] + corners[behind[0]]) / 2)
        if best is not None:
            swaps.append(best)
    return swaps


def shared_edge(D, a, b):
    hit = ((D['i'] == a) & (D['j'] == b)) | ((D['i'] == b) & (D['j'] == a))
    return float(D['l'][hit].sum()) if hit.any() else 0.0


def apply_swaps(c, w, W, H, swaps):
    """Carry out the swaps: move A and D towards the midpoint of the edge between them until they share an edge as
    long as the one it replaces, then make all areas equal again. Returns the new x, or None if that fails."""
    c = c.copy(); nt = len(c)
    for A, Dd, length, mid in swaps:
        target = max(length, 1e-3)

        def moved(fraction):
            c2 = c.copy()
            c2[A] += fraction * (mid - c[A]); c2[Dd] += fraction * (mid - c[Dd])
            D2 = mmi.diagram(c2, w, W, H)
            return (-1.0 if D2['hidden'] else shared_edge(D2, A, Dd)), c2

        low, high, fallback = 0.0, None, None
        for fraction in SWAP_FRACTIONS:
            edge, c2 = moved(fraction)
            if edge >= target:
                high = fraction
                break
            if edge > 0 and (fallback is None or edge > fallback[0]):
                fallback = (edge, c2)
            low = fraction
        if high is None:                                        # never as long as the old edge: keep the longest seen
            if fallback is not None:
                c = fallback[1]
            continue
        for _ in range(SWAP_BISECTIONS):                        # smallest move that reaches the target length
            middle = (low + high) / 2
            edge, _ = moved(middle)
            if edge >= target:
                high = middle
            else:
                low = middle
        c = moved(high)[1]
    w2, D2 = mmi.equalise(c, w, W, H)
    if D2 is None:
        w2, D2 = mmi.equalise(c, np.zeros(nt), W, H, iters=30)
    return None if D2 is None else np.r_[c.ravel(), w2]


def push(args):
    """one candidate down the temperature ladder; returns the final layout and what happened after each temperature"""
    sites, W, H = args
    nt = len(sites); started = time.perf_counter()
    x = np.r_[sites.ravel(), np.zeros(nt)]                      # all weights start at zero
    trace = []
    for k, tau in enumerate(mmi.TAUS):
        x = optimise_at(x, nt, W, H, tau)
        layout = equal_area_layout(x, nt, W, H)
        if layout is None:
            return dict(ok=False, lmax=9.9, trace=trace, seconds=time.perf_counter() - started)
        c, w, D = layout
        walls, neighbours = neighbour_counts(D, W, H)
        too_few = tiles_with_too_few_neighbours(walls, neighbours)
        is_last = k == len(mmi.TAUS) - 1
        swaps = [] if is_last else find_swaps(D, walls, neighbours)
        trace.append(dict(tau=tau, lmax=float(D['l'].max()), too_few=len(too_few), swaps=len(swaps)))
        x = np.r_[c.ravel(), w]
        if swaps:
            swapped = apply_swaps(c, w, W, H, swaps)
            if swapped is not None:
                x = swapped
    lengths = np.sort(D['l'])
    return dict(ok=True, c=c, w=w, lmax=float(lengths[-1]), top10=float(lengths[-10:].mean()), too_few=len(too_few),
                trace=trace, seconds=time.perf_counter() - started)


def site_search(nt, W, H, procs=None):
    """push every candidate (in parallel) and return (index of the best, candidates, results)"""
    from multiprocessing import Pool
    cand = candidates(nt, W, H)
    assert cand, 'no integer design fits this box'
    procs = procs or min(len(cand), os.cpu_count() - 1)
    with Pool(procs) as pool:
        results = pool.map(push, [(sites, W, H) for *_, sites in cand], chunksize=1)
    best = min(range(len(results)), key=lambda k: results[k]['lmax'])
    assert results[best]['ok'], 'every candidate failed'
    return best, cand, results


# ---------------------------------------------------------------------------------------------------------------
# 3. the whole pipeline: read points -> sites -> exactly B points per tile -> order
# ---------------------------------------------------------------------------------------------------------------
def run(csv, prefix, procs=None):
    t0 = time.perf_counter()
    p = mmi.read_xy(csv); n = len(p); nt = n // mmi.B
    assert nt * mmi.B == n, f'{n} points is not a multiple of the tile size {mmi.B}'
    lo, hi = p.min(0), p.max(0)
    unit = np.sqrt((hi - lo).prod() / nt)                       # length unit: one tile has area 1
    W, H = (hi - lo) / unit
    t1 = time.perf_counter()
    best, cand, results = site_search(nt, W, H, procs)
    win = results[best]
    t2 = time.perf_counter()
    labels, weights, count_error = mmi.assign_points(p, win['c'], win['w'], lo, unit, W, H)
    t3 = time.perf_counter()
    order = ccvt.order_from_labels(p, labels)
    t4 = time.perf_counter()
    assert np.array_equal(np.sort(order), np.arange(n))
    np.save(f'{prefix}_order.npy', order)
    record = dict(
        csv=str(csv), n=n, tile=mmi.B, tiles=nt, box=[float(W), float(H)], taus=mmi.TAUS, iters=mmi.ITERS,
        chosen=cand[best][0], site_lmax=win['lmax'], site_top10=win['top10'],
        gap_to_bound=win['lmax'] / BOUND - 1, newton_count_err=count_error,
        seconds=dict(read=t1 - t0, site_search=t2 - t1, assign=t3 - t2, order=t4 - t3, total=t4 - t0),
        sites=(win['c'] * unit + lo).tolist(),
        candidates=[dict(design=label, lmax=r['lmax'], seconds=r['seconds'], trace=r['trace'])
                    for (label, *_), r in zip(cand, results)])
    json.dump(record, open(f'{prefix}_run.json', 'w'), indent=1)
    print(f'{nt} tiles, {len(cand)} candidates; chosen {cand[best][0]}: longest edge {win["lmax"]:.4f} '
          f'({100 * (win["lmax"] / BOUND - 1):.1f}% above the bound {BOUND})')
    for (label, *_), r in zip(cand, results):
        steps = '  '.join(f'{s["too_few"]} too few, {s["swaps"]} swapped' for s in r['trace'])
        print(f'  {label:<20} {r["lmax"]:.4f}   after each temperature: {steps}')
    print(f'total {t4 - t0:.2f} s (read {t1 - t0:.2f}, sites {t2 - t1:.2f}, assign {t3 - t2:.2f}, order {t4 - t3:.2f})', flush=True)
    return order


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
