#!/usr/bin/env python3
"""QSR (Quantum Slab-Row) ordering for 2D locations.

The domain is cut into slabs along its longest axis (about two tiles thick). Every slab picks a row layout (row
axis, number of rows r0-1..r0+1, every near-even split of its tiles over the rows); rows and tiles are cut by
point count, so every tile holds exactly b locations. The layouts of all slabs are chosen jointly by minimizing a
sum over tile pairs of a penalty on a geometric rank surrogate (seam length and tile thickness, fitted once on
exact ranks; priority: largest rank, then sum of squared ranks, then mean). The penalty is averaged over the
Matern ranges beta = 0.03, 0.1, 0.3, so one ordering serves all three.

Solvers (the problem is the same in every case):
  exact (default)  exhaustive search after exact dominance pruning; no quantum hardware or extra package needed
  qpu              domain-wall QUBO on a D-Wave Advantage2 QPU (needs dwave-ocean-sdk and a configured token);
                   the best valid sample is compared with the exact optimum
  sa               the same QUBO with D-Wave's classical simulated annealer (dwave-samplers), for trying the QUBO
                   path without hardware
  --qubo FILE      also write the QUBO (linear, quadratic, offset, variable meaning) as JSON

--nonuniform (for locations whose density varies; off by default, so the default output is unchanged): every slab also
gets layouts whose rows hold different numbers of tiles (tiles per row from a small DP that makes each row's tiles close
to a given aspect ratio in that row's own extent), and the problem is solved for two slab cuts, equal tile counts and
slabs about two tile sides thick in their own density; the cut with the lower energy is kept.

Output: a permutation file for ExaGeoStatCPP (--ordering=file --ordering_file=OUT); tiles in slab/row order,
locations inside a tile in cut order (the convention of the QSR orderings measured on Shaheen III).

usage: qsr.py LOCATIONS.csv OUT.order [--tile 4096] [--nonuniform] [--solver exact|qpu|sa] [--qubo QUBO.json] [--reads 2000]"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys, json, math, time, argparse, itertools
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common, minbbox

M = json.load(open(HERE / 'qsr_model.json'))
GAMMAS, EVENCAP = (0.7, 1.0, 1.4), 5000         # --nonuniform: row aspect targets; cap on the near-even splits per row count
BETAS = tuple(M['betas'])


# ---------------- geometric rank surrogate ----------------
def features(lo1, hi1, lo2, hi2):
    """shared-edge length, eta = max bbox diagonal / centre distance, both boxes' thickness across the separating axis"""
    ov = np.minimum(hi1, hi2) - np.maximum(lo1, lo2)
    gap = np.linalg.norm(np.maximum(0, -ov), axis=-1); ovp = np.maximum(0, ov)
    sep = np.argmin(ov, axis=-1); on = np.arange(ov.shape[-1]) == sep[..., None]
    contact = np.where(gap < M['GAPT'], np.where(on, 1.0, ovp).prod(axis=-1), 0.0)
    e1 = np.broadcast_to(hi1 - lo1, ov.shape); e2 = np.broadcast_to(hi2 - lo2, ov.shape)
    t1 = np.where(on, e1, 0).sum(-1); t2 = np.where(on, e2, 0).sum(-1)
    d1 = np.linalg.norm(hi1 - lo1, axis=-1); d2 = np.linalg.norm(hi2 - lo2, axis=-1)
    cd = np.linalg.norm((lo1 + hi1) / 2 - (lo2 + hi2) / 2, axis=-1)
    return contact, np.maximum(d1, d2) / np.maximum(cd, 1e-12), np.minimum(t1, t2), np.maximum(t1, t2)


def boxes(p, tiles):
    return np.array([p[t].min(0) for t in tiles]), np.array([p[t].max(0) for t in tiles])


class Surrogate:
    """log r = w0 + w1 log(edge) + w2 log(thin) + w3 log(thick) for pairs sharing an edge, v0 + v1 log(eta) otherwise.
    Scale Mp: the largest predicted rank over the min-bbox tiles of the same point set."""
    def __init__(self, p, tile):
        T = minbbox.minbbox_tiles(p, tile); lo, hi = boxes(p, T); ii, jj = np.tril_indices(len(T), -1)
        pr = self.predict(*features(lo[ii], hi[ii], lo[jj], hi[jj]))
        self.Mp = {b: float(pr[b].max()) for b in BETAS}

    @staticmethod
    def predict(contact, eta, thin, thick):
        face = contact > M['CT']; out = {}
        lc, ltn, ltk, le = (np.log(np.maximum(x, 1e-12)) for x in (contact, thin, thick, eta))
        for b in BETAS:
            w, v = M['face'][str(b)], M['eta'][str(b)]
            out[b] = np.where(face, np.exp(w[0] + w[1] * lc + w[2] * ltn + w[3] * ltk), np.exp(v[0] + v[1] * le))
        return out

    def cost(self, lo1, hi1, lo2, hi2):
        pr = self.predict(*features(lo1, hi1, lo2, hi2)); tot = 0
        for b in BETAS:
            x = pr[b] / self.Mp[b]; e = np.maximum(0, x - M['THR'])
            tot = tot + M['WMEAN'] * x + M['W2'] * x ** 2 + M['WMAX'] * (
                M['HD'] * (np.minimum(e, M['HCAP']) / M['HD']) ** M['HQ'] + np.maximum(0, e - M['HCAP']))
        return tot / len(BETAS)


# ---------------- slab-row layouts ----------------
def chunks(p, idx, axis, counts, tile):
    s = idx[np.argsort(p[idx, axis], kind='stable')]; out = []; k = 0
    for c in counts: out.append(s[k:k + c * tile]); k += c * tile
    return out


def even(t, r):
    a, m = divmod(t, r)
    return sorted(tuple(a + (i in S) for i in range(r)) for S in itertools.combinations(range(r), m))


def even_capped(t, r):
    """even(t, r); beyond EVENCAP splits only those whose longer rows are contiguous (cyclically) or evenly spread"""
    a, m = divmod(t, r)
    if math.comb(r, m) <= EVENCAP: return even(t, r)
    S = {frozenset((s + i) % r for i in range(m)) for s in range(r)} | {frozenset((s + (i * r) // m) % r for i in range(m)) for s in range(r)}
    return sorted(tuple(a + (i in s) for i in range(r)) for s in S)


def _partition(cost, t):
    """cost[s, e] of one part covering items s..e-1; cheapest split of 0..t into consecutive parts"""
    f = np.full(t + 1, np.inf); f[0] = 0; back = np.zeros(t + 1, int)
    for e in range(1, t + 1): v = f[:e] + cost[:e, e]; back[e] = int(v.argmin()); f[e] = v[back[e]]
    out = []; e = t
    while e > 0: out.append(int(e - back[e])); e = int(back[e])
    return out[::-1]


def _spans(a, b, t, tile):
    """for points sorted along a: extent along a and across (b) of the tiles s..s+k-1, as arrays [s][k-1]"""
    out = []
    for s in range(t):
        bb = b[s * tile:]; k = np.arange(1, t - s + 1)
        out.append((a[(s + k) * tile - 1] - a[s * tile], np.maximum.accumulate(bb)[tile - 1::tile] - np.minimum.accumulate(bb)[tile - 1::tile], k))
    return out


def adaptive_split(p, srt, ra, t, tile, gamma):
    """tiles per row (rows = chunks along ra of the sorted indices srt): k * (row extent along ra) / (row extent across) -> gamma"""
    cost = np.full((t + 1, t + 1), np.inf)
    for s, (h, W, k) in enumerate(_spans(p[srt, ra], p[srt, 1 - ra], t, tile)):
        cost[s, s + k] = k * np.log(np.maximum(k * h, 1e-300) / (gamma * np.maximum(W, 1e-300))) ** 2
    return tuple(_partition(cost, t))


def thick_counts(p, sa, tile):
    """tiles per slab so that thickness / local tile side = sqrt(thickness * tiles / length) is close to K for every slab"""
    nt = len(p) // tile; s = np.argsort(p[:, sa], kind='stable'); cost = np.full((nt + 1, nt + 1), np.inf)
    for st, (w, L, k) in enumerate(_spans(p[s, sa], p[s, 1 - sa], nt, tile)):
        cost[st, st + k] = k * np.log(np.sqrt(np.maximum(w * k, 1e-300) / np.maximum(L, 1e-300)) / M['K']) ** 2
    return _partition(cost, nt)


def slab_options(p, slab, t, c, tile, split=False):
    reg, U, rows, out, seen = {}, [], {}, [], set(); ext = np.ptp(p[slab], axis=0)
    srt = {ra: slab[np.argsort(p[slab, ra], kind='stable')] for ra in (0, 1)}

    def row_tiles(ra, start, k):
        if (ra, start, k) not in rows:
            ids = []
            for tl in chunks(p, srt[ra][start * tile:(start + k) * tile], 1 - ra, [1] * k, tile):
                kk = np.sort(tl).tobytes()
                if kk not in reg: reg[kk] = len(U); U.append(tl)
                ids.append(reg[kk])
            rows[(ra, start, k)] = ids
        return rows[(ra, start, k)]

    for ra in (0, 1):
        r0 = max(1, int(round(ext[ra] / c)))
        for r in sorted({r0 - 1, r0, r0 + 1}):
            if r < 1 or r > t: continue
            for rc in (even_capped if split else even)(t, r):
                ids, st = [], 0
                for k in rc: ids += row_tiles(ra, st, k); st += k
                if frozenset(ids) in seen: continue
                seen.add(frozenset(ids)); out.append(dict(name=f'rows{"xy"[ra]}{list(rc)}', ids=ids))
    if split:
        for ra in (0, 1):
            for g in GAMMAS:
                rc = adaptive_split(p, srt[ra], ra, t, tile, g); ids, st = [], 0
                for k in rc: ids += row_tiles(ra, st, k); st += k
                if frozenset(ids) in seen: continue
                seen.add(frozenset(ids)); out.append(dict(name=f'split{"xy"[ra]}{g}{list(rc)}', ids=ids))
    lo, hi = boxes(p, U); A = np.zeros((len(out), len(U)))
    for k, o in enumerate(out): o['tiles'] = [U[i] for i in o['ids']]; A[k, o['ids']] = 1
    return dict(opts=out, lo=lo, hi=hi, A=A)


def within(model, bg):
    C = model.cost(bg['lo'][:, None], bg['hi'][:, None], bg['lo'][None], bg['hi'][None]); np.fill_diagonal(C, 0)
    return 0.5 * ((bg['A'] @ C) * bg['A']).sum(1)


def cross_matrix(model, bg, bh):
    return bg['A'] @ model.cost(bg['lo'][:, None], bg['hi'][:, None], bh['lo'][None], bh['hi'][None]) @ bh['A'].T


def prune(lin, cross, cap=None):
    """Exact dominance pruning: drop option o of slab g if another option is better for every choice of the other
    slabs (keeps every optimum). With cap, stops once the domain-wall QUBO has <= cap bits."""
    G = len(lin); alive = [np.arange(len(l)) for l in lin]; changed = True
    fits = lambda: cap is not None and sum(len(a) - 1 for a in alive) <= cap
    while changed and not fits():
        changed = False
        for g in range(G):
            a = alive[g]
            if len(a) < 2: continue
            nb = [M_[np.ix_(a, alive[y])] if x == g else M_[np.ix_(alive[x], a)].T for (x, y), M_ in cross.items() if g in (x, y)]
            for i in np.argsort(-(lin[g][a] + sum(S.min(1) for S in nb)), kind='stable'):
                d = lin[g][a[i]] - lin[g][a]
                for S in nb: d = d - (S - S[i]).max(1)
                d[i] = -np.inf
                if (d > 0).any():
                    alive[g] = np.delete(a, i); changed = True; break
            if fits(): break
    return [list(map(int, a)) for a in alive]


def exhaustive(lin, cross, alive):
    G = len(alive); E = np.zeros([len(a) for a in alive])
    for g in range(G):
        sh = [1] * G; sh[g] = len(alive[g]); E = E + lin[g][alive[g]].reshape(sh)
    for (g, h), M_ in cross.items():
        sh = [1] * G; sh[g] = len(alive[g]); sh[h] = len(alive[h]); E = E + M_[np.ix_(alive[g], alive[h])].reshape(sh)
    i = np.unravel_index(np.argmin(E), E.shape)
    return tuple(alive[g][i[g]] for g in range(G))


def energy(lin, cross, ch):
    return float(sum(lin[g][o] for g, o in enumerate(ch)) + sum(M_[ch[g], ch[h]] for (g, h), M_ in cross.items()))


# ---------------- domain-wall QUBO ----------------
def wall_inversion(lin, cross, alive):
    """Per slab, the largest energy gain a broken domain wall can make against valid choices of the other slabs."""
    out = []
    for g, a in enumerate(alive):
        D = lin[g][a][:, None] - lin[g][a][None, :]
        for (x, y), M_ in cross.items():
            if g not in (x, y): continue
            S = M_[np.ix_(a, alive[y])] if x == g else M_[np.ix_(alive[x], a)].T
            D = D + (S[:, None, :] - S[None, :, :]).max(-1)
        out.append(max(float(np.triu(D, 1).max()), 1e-3) if len(a) > 1 else 0.0)
    return out


def dw_qubo(lin, cross, alive, pen):
    """Slab with options a_0..a_{k-1} uses bits d_1..d_{k-1}, d_i = [choice >= i]; penalty P_g d_{i+1} (1 - d_i)."""
    G = len(alive); lin = [np.array(l, dtype=float) for l in lin]; J = {}
    for (g, h), M_ in cross.items():
        Ms = M_[np.ix_(alive[g], alive[h])]; r, c, m = Ms.mean(1), Ms.mean(0), Ms.mean()
        J[(g, h)] = Ms - r[:, None] - c[None, :] + m
        lin[g][alive[g]] += r; lin[h][alive[h]] += c
    bits, pos = [], {}
    for g in range(G):
        for i in range(1, len(alive[g])): pos[(g, i)] = len(bits); bits.append((g, i))

    def affine(g, i):
        k = len(alive[g])
        if k == 1: return 1.0, {}
        if i == 0: return 1.0, {pos[(g, 1)]: -1.0}
        if i == k - 1: return 0.0, {pos[(g, i)]: 1.0}
        return 0.0, {pos[(g, i)]: 1.0, pos[(g, i + 1)]: -1.0}

    L = {b: 0.0 for b in range(len(bits))}; Q = {}; const = 0.0

    def addq(a, b, w):
        if a == b: L[a] += w
        else: k = (min(a, b), max(a, b)); Q[k] = Q.get(k, 0.0) + w

    for g in range(G):
        base = lin[g][alive[g]].min()
        for i, o in enumerate(alive[g]):
            c0, cv = affine(g, i); w = lin[g][o] - base; const += w * c0
            for b, a in cv.items(): L[b] += w * a
    for (g, h), R in J.items():
        for i in range(len(alive[g])):
            c0, cv = affine(g, i)
            for j in range(len(alive[h])):
                w = R[i, j]
                if w == 0: continue
                d0, dv = affine(h, j); const += w * c0 * d0
                for b, a in cv.items(): L[b] += w * a * d0
                for b, a in dv.items(): L[b] += w * c0 * a
                for b, a in cv.items():
                    for b2, a2 in dv.items(): addq(b, b2, w * a * a2)
    for g in range(G):
        for i in range(1, len(alive[g]) - 1):
            a, b = pos[(g, i)], pos[(g, i + 1)]; L[b] += pen[g]; addq(a, b, -pen[g])
    return L, Q, bits, const


def decode(sample, bits, alive):
    ch = []
    for g in range(len(alive)):
        d = [sample[b] for b, (g2, _) in enumerate(bits) if g2 == g]
        if any(d[i + 1] > d[i] for i in range(len(d) - 1)): return None
        ch.append(alive[g][sum(d)])
    return tuple(ch)


def sample_qubo(L, Q, solver, reads, label, big):
    import dimod
    bqm = dimod.BinaryQuadraticModel(L, Q, 0.0, dimod.BINARY); info = dict(variables=len(L), interactions=len(Q))
    if solver == 'sa':
        from dwave.samplers import SimulatedAnnealingSampler
        ss = SimulatedAnnealingSampler().sample(bqm, num_reads=reads, seed=0)
    else:
        from dwave.system import DWaveSampler, FixedEmbeddingComposite
        import minorminer
        from minorminer import busclique
        sampler = DWaveSampler(solver=dict(name__regex=r'Advantage2.*')); info['solver'] = sampler.solver.name
        g = sampler.to_networkx_graph()
        emb = minorminer.find_embedding(list(Q), g, random_seed=0, tries=10) if big else \
            busclique.find_clique_embedding(sorted(L), g, use_cache=True)
        if not emb or set(emb) != set(L): raise RuntimeError('embedding failed')
        info['physical_qubits'] = sum(map(len, emb.values()))
        cs = (2.0 if big else 1.0) * max(abs(v) for v in bqm.to_ising()[1].values()) if Q else None
        kw = {} if cs is None else dict(chain_strength=cs)
        ss = FixedEmbeddingComposite(sampler, emb).sample(bqm, num_reads=reads, annealing_time=100, label=label, **kw)
        info['problem_id'] = ss.info.get('problem_id'); info['qpu_access_s'] = ss.info.get('timing', {}).get('qpu_access_time', 0) / 1e6
    rows = [({int(k): int(v) for k, v in r.sample.items()}, int(r.num_occurrences)) for r in ss.data(['sample', 'num_occurrences'])]
    return rows, info


# ---------------- driver ----------------
def qsr(p, tile, solver='exact', qubo_path=None, reads=2000, log=print, nonuniform=False):
    n = len(p)
    if p.shape[1] != 2: raise SystemExit('qsr.py handles 2D locations only')
    if n % tile: raise SystemExit(f'{n} locations is not a multiple of the tile size {tile}')
    if tile != 4096: log(f'note: the rank surrogate was calibrated at tile 4096; tile {tile} uses it unchanged')
    if p.min() < -1e-9 or p.max() > 1 + 1e-9: log('note: the surrogate was calibrated on locations in the unit square')
    t0 = time.perf_counter(); model = Surrogate(p, tile)
    nt = n // tile; ext = np.ptp(p, axis=0); c = np.sqrt(ext.prod() / nt)
    sa = int(np.argmax(ext)); ns = max(1, int(round(ext[sa] / (M['K'] * c))))
    alts = []
    for cut in (('equal-count', 'density-thick') if nonuniform else ('equal-count',)):
        counts = thick_counts(p, sa, tile) if cut == 'density-thick' else [nt // ns + (1 if i < nt % ns else 0) for i in range(ns)]
        parts = chunks(p, np.arange(n), sa, counts, tile)    # density-thick slabs centre the row counts with their own tile side
        slabs = [slab_options(p, s, t, np.sqrt(np.ptp(p[s], axis=0).prod() / t) if cut == 'density-thick' else c, tile, nonuniform)
                 for s, t in zip(parts, counts)]
        lin = [within(model, s) for s in slabs]
        cross = {(g, h): cross_matrix(model, slabs[g], slabs[h]) for g, h in itertools.combinations(range(len(slabs)), 2)}
        full = prune(lin, cross); best = exhaustive(lin, cross, full)
        alts.append((energy(lin, cross, best), cut, counts, slabs, lin, cross, full, best))
    e_best, cut, counts, slabs, lin, cross, full, best = min(alts, key=lambda a: a[0]); ns = len(slabs); opts = [s['opts'] for s in slabs]
    rec = dict(n=n, tile=tile, nonuniform=nonuniform, slab_cut=cut, energy_per_slab_cut={a[1]: a[0] for a in alts},
               slabs=ns, slab_axis='xy'[sa], tiles_per_slab=counts, options_per_slab=[len(o) for o in opts],
               options_after_pruning=[len(a) for a in full], exact_choice=[opts[g][o]['name'] for g, o in enumerate(best)],
               exact_energy=e_best, solver=solver)
    choice = best
    if solver != 'exact' or qubo_path:
        alive = prune(lin, cross, cap=M['BITCAP'])
        key = lambda g, o: lin[g][o] + sum(M_[o, alive[y]].mean() if x == g else M_[alive[x], o].mean()
                                           for (x, y), M_ in cross.items() if g in (x, y))
        alive = [sorted(a, key=lambda o, g=g: key(g, o)) for g, a in enumerate(alive)]
        big = sum(len(a) - 1 for a in alive) > 64
        pen = [(3.0 if big else 2.0) * v for v in wall_inversion(lin, cross, alive)]
        L, Q, bits, const = dw_qubo(lin, cross, alive, pen)
        if big: Q = {k: v for k, v in Q.items() if abs(bits[k[0]][0] - bits[k[1]][0]) <= 1}
        rec.update(qubo_bits=len(L), qubo_terms=len(Q))
        if qubo_path:
            json.dump(dict(linear={str(k): v for k, v in L.items()}, quadratic=[[a, b, v] for (a, b), v in Q.items()],
                           offset=const, bits=[dict(slab=g, wall=i, meaning=f'slab {g} chooses option index >= {i} of '
                                                    f'{[opts[g][o]["name"] for o in alive[g]]}') for g, i in bits],
                           note='domain-wall encoding; valid strings per slab are 1..10..0'),
                      open(qubo_path, 'w'), indent=1)
            rec['qubo_file'] = str(qubo_path)
        if solver != 'exact':
            try:
                rows, info = sample_qubo(L, Q, solver, reads, f'qsr_N{n}_B{tile}', big)
            except ImportError as e:
                raise SystemExit(f'--solver {solver} needs the D-Wave Ocean SDK ({e}); the default --solver exact needs nothing extra')
            dec = [(decode(s, bits, alive), occ) for s, occ in rows]; val = [(ch, occ) for ch, occ in dec if ch is not None]
            rec.update(info, reads=sum(o for _, o in rows), valid_reads=sum(o for _, o in val))
            if val:
                choice = min((ch for ch, _ in val), key=lambda ch: energy(lin, cross, ch))
                rec['sampler_gap_to_exact'] = energy(lin, cross, choice) - e_best
            else:
                log('no valid sample; using the exact optimum'); rec['sampler_gap_to_exact'] = None
    rec['choice'] = [opts[g][o]['name'] for g, o in enumerate(choice)]
    tiles = [t for g, o in enumerate(choice) for t in opts[g][o]['tiles']]
    order = np.concatenate(tiles); assert np.array_equal(np.sort(order), np.arange(n)) and all(len(t) == tile for t in tiles)
    rec['seconds'] = time.perf_counter() - t0
    return order, rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('locations'); ap.add_argument('out')
    ap.add_argument('--nonuniform', action='store_true')
    ap.add_argument('--tile', type=int, default=4096); ap.add_argument('--solver', choices=('exact', 'qpu', 'sa'), default='exact')
    ap.add_argument('--qubo', default=None); ap.add_argument('--reads', type=int, default=2000)
    a = ap.parse_args()
    p = common.read_locations(a.locations)
    order, rec = qsr(p, a.tile, a.solver, a.qubo, a.reads, log=lambda s: print(s, file=sys.stderr), nonuniform=a.nonuniform)
    common.write_order(a.out, order, p, a.tile, f'qsr solver={a.solver}' + (' nonuniform' if a.nonuniform else ''), extra=[f'choice {" ".join(rec["choice"])}'])
    print(json.dumps(rec, default=float))


if __name__ == '__main__':
    main()
