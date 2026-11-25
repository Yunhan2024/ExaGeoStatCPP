# Tile-aware orderings for TLR: QSR and ShortSeam

Two spatial orderings for tile low-rank (TLR) geostatistics, computed in Python and applied by ExaGeoStatCPP with
`--ordering=file`. Both choose which locations share a tile (exactly `b` per tile) so that the longest boundary
shared by two tiles, and with it the largest off-diagonal tile rank, is small.

| Tool | Tiles | Solver | Typical time (2D, tile 4096) |
|---|---|---|---|
| `qsr.py` | axis-aligned rectangles: slabs along the longest axis, each with one of an enumerated set of row layouts | exact search by default; optional domain-wall QUBO on a D-Wave QPU or simulated annealing | 0.2-6.4 s for 61,440-983,040 locations |
| `shortseam.py` | convex polygons of a power diagram, longest shared edge minimized | continuous optimization over sites and weights, then an exact-count assignment | 3-11 s for 61,440-983,040 locations (12 processes) |

Both are 2D and need the number of locations to be a multiple of the tile size.

## Requirements

`numpy` and `scipy` (see `requirements.txt`). Only `qsr.py --solver qpu|sa` needs the D-Wave Ocean SDK
(`dwave-ocean-sdk`; `qpu` also needs a configured D-Wave API token).

## Usage

```bash
# locations: CSV with a header line and one location per row (x,y), in the order ExaGeoStat holds them
python3 tools/ordering/qsr.py       locations.csv qsr.order       --tile 4096
python3 tools/ordering/shortseam.py locations.csv shortseam.order --tile 4096

# locations whose density varies (clusters, gaps, gradients): see "Non-uniform locations" below
python3 tools/ordering/qsr.py       locations.csv qsr.order       --tile 4096 --nonuniform
python3 tools/ordering/shortseam.py locations.csv shortseam.order --tile 4096 --density

# QSR through the QUBO path
python3 tools/ordering/qsr.py locations.csv qsr.order --qubo qsr_qubo.json   # exact solve, also write the QUBO
python3 tools/ordering/qsr.py locations.csv qsr.order --solver sa            # simulated annealing on the QUBO
python3 tools/ordering/qsr.py locations.csv qsr.order --solver qpu           # D-Wave Advantage2 QPU

# ExaGeoStatCPP: synthetic locations (same --N and --seed as the CSV) or loaded data (same row order)
bin/examples/end-to-end/Example_Data_Generation_Modeling_and_Prediction --N=61440 --seed=42 ... \
    --ordering=file --ordering_file=shortseam.order
```

Each tool prints a JSON record as the last line of its output (SciPy's optimizer may print warnings before it).

### Which row order the permutation refers to

Line `i` of the permutation (after the `#` headers) is the input row that goes to position `i`; consecutive groups
of `b` positions are the tiles. With `--ordering=file`, synthetic locations are used in generation order (they
are not Morton-sorted first) and loaded data in file order. ExaGeoStatCPP checks the headers and stops with an
error if the permutation was made for another number of locations (`# n`), another point set (`# coordsum`) or
the same points in another row order (`# coordwsum`). The seed-42 2D sets distributed for the experiments
(`exageostat-N<n>-seed42-xy.csv`) are in generation order: a permutation computed from such a file is accepted for
`--N=<n> --seed=42` (checked for 61,440).

## QSR (Quantum Slab-Row)

1. Cut the domain into slabs along its longest axis, about two tiles thick, each holding a whole number of tiles.
2. For every slab enumerate its row layouts: rows along x or y, r0-1..r0+1 rows (r0 = slab extent / tile side), every
   near-even split of its tiles over the rows. Rows and tiles are cut by point count, so every tile has `b` locations.
3. Score a choice of layouts by a sum over tile pairs of a penalty on a geometric rank surrogate (shared-edge length
   and tile thickness, fitted once on exact ranks; `qsr_model.json`). The penalty puts the largest predicted ranks
   first, then the sum of squared ranks, then the mean, and is averaged over the Matern ranges 0.03, 0.1, 0.3, so
   one ordering serves all three. Its scale is set by the min-bbox tiles of the same locations (`minbbox.py`).
4. The cost has one term per slab and one per pair of slabs. Exact dominance pruning removes options that are worse
   in every context; the rest is searched exhaustively (`--solver exact`), or written as a domain-wall QUBO
   (`d_i = [choice >= i]` per slab) and sampled (`--solver qpu|sa`). Sampled solutions are compared with the exact
   optimum (`sampler_gap_to_exact` in the record).

Output: tiles in slab and row order, locations inside a tile in cut order, i.e. the convention of the QSR orderings
measured on Shaheen III.

## ShortSeam

1. Tiles are power-diagram cells (convex polygons). At most twelve integer start designs (a ring of thin tiles along
   the domain sides plus hexagonal rows inside) are each optimized over sites and weights: a soft-max of all shared
   edge lengths plus an equal-area penalty, through a temperature ladder (L-BFGS-B), with exactly equal areas after
   every temperature. Tiles with too few neighbours get one edge swap between temperatures. The design with the
   shortest longest edge wins; its distance to the straight-edge lower bound 0.7624 (tile side units) is reported.
2. Locations are assigned to tiles by Newton steps on the power weights with the real counts, then a
   transportation LP on a narrow band of boundary locations gives exactly `b` locations per tile.
3. Tile order (`--tile-order`): `spectral` (default) sorts tiles by the Fiedler vector of the tile adjacency graph
   (Delaunay edges between tiles); `hilbert` uses the Hilbert order of the tile centroids. Only the order of whole
   tiles changes, so every initial tile rank is the same for both. Locations inside a tile follow Hilbert order.

The algorithm code is in `shortseam_core/` (research name MMI: `mmi_v2.py` start designs and edge swaps, `mmi.py`
power diagram, objective, optimizer and point assignment, `ccvt.py` helpers), unchanged from the research version.

## Non-uniform locations (`qsr.py --nonuniform`, `shortseam.py --density`)

Both options are off by default; without them the tools give exactly the orderings described above. They keep each
method's structure, cost and solver and make the geometric quantities follow the local point density.

**`qsr.py --nonuniform`** (in `qsr.py`)
- Every slab gets six more layouts whose rows hold different numbers of tiles: rows along x or y, tiles per row from a
  small dynamic program that brings each row's tiles close to a target aspect ratio (0.7, 1, 1.4) in that row's own
  extent. With near-even rows, a slab that crosses a density change has to cut thin tiles on its dense side.
- The problem is solved for two slab cuts: equal tile counts (as before) and slabs about two tile sides thick in their
  own density (boundaries from a dynamic program; the row counts of such a slab are centred with its own tile side).
  The energies are comparable (same surrogate); the cut with the lower energy is kept (`slab_cut` in the record).
- Surrogate, penalty, pruning, exact search and the QUBO encoding are unchanged. When a slab would have more than 5000
  near-even splits for one row count, only the splits with contiguous or evenly spread longer rows are enumerated.

**`shortseam.py --density`** (`shortseam_core/mmi_rho.py`, `ccvt_nu.py`, `warp.py`)
- Site level: a tile is measured by the integral of the point density over it (mass 1 per tile) instead of its area,
  a seam by the integral of sqrt(density) along it instead of its length. Density = smoothed histogram of the
  locations. Power diagram, soft-max objective, temperature ladder and optimizer are the same; for constant density
  the two quantities equal area and length.
- Start sites: the integer start designs assume uniform density, so the starts are the tile centroids of two tilings
  computed first, an equal-count centroidal power diagram on the locations and the default ShortSeam run on
  density-equalized coordinates (diffusion cartogram). Both are optimized; the one with the shorter longest seam at
  site level is kept. A start that fails is skipped (`skipped_starts` in the record).
- Point level: power weights balanced with band and step sizes relative to the local site spacing, then the
  transportation LP.

Measured on the five location designs of the 2023 KAUST competition on spatial statistics for large datasets
(Chessboard, Left-bottom, Satellite, Clusters, Regular), first 897,024 of the 900,000 training locations, tile 4096,
219 tiles. Score: longest seam between two tiles, counted as Delaunay edges of the locations crossing between them,
in units of the count for one side of a square tile at uniform density (shorter is better; no rank is computed):

| | Chessboard | Left-bottom | Satellite | Clusters | Regular |
|---|---|---|---|---|---|
| min-bbox | 2.86 | 1.79 | 1.68 | 2.52 | 1.69 |
| `qsr.py` | 1.92 | 3.75 | 1.43 | 3.04 | 1.24 |
| `qsr.py --nonuniform` | 1.39 | 2.52 | 1.48 | 1.77 | 1.24 |
| `shortseam.py` | fails | fails | 1.14 | fails | 0.94 |
| `shortseam.py --density` | 1.06 | 1.16 | 0.98 | 1.65 | 0.94 |

On the 90K sets of the same designs (first 89,280 locations, tile 960) the largest rank of adjacent tile pairs
(numerical rank of exp(-d/0.03), relative Frobenius tolerance 1e-7; an estimator, not HiCMA's ranks) is 89 / 100 /
88 / 96 / 85 for `shortseam.py --density` and 169 / 144 / 137 / 148 / 120 for min-bbox; on these sets the seam score
ranks orderings like this estimator in 238 of the 245 comparisons where the scores differ by more than 10%.
`qsr.py --nonuniform` was scored with the seam length only.

Limits of the two options:
- No HiCMA ranks and no timings were measured for them; the 900K numbers are seam scores only.
- `qsr.py --nonuniform` on Left-bottom is still longer than min-bbox (a slab has one thickness along its whole length,
  and the density changes by a factor of 100 along it). It does not change Satellite or Regular. The QUBO is written
  for the kept slab cut; `--solver qpu|sa` was not run with the new layouts.
- `shortseam.py --density` on Clusters at 900K: the equal-count power diagram start fails (assignment LP infeasible),
  so only the other start is used. On Chessboard at 900K one run took 2757 s (others 70-160 s plus 25-60 s per start);
  the cause was not investigated. The integer start designs were not tried with the density-aware site level.
- Times, 900K: `qsr.py --nonuniform` 67-172 s, Clusters 1140 s (default `qsr.py`: about 31 s, Clusters 1005 s).

## Checks done for this branch

- `qsr.py` (default exact solver) reproduces, bit for bit, the QSR orderings measured on Shaheen III for the
  seed-42 2D sets with 61,440, 122,880, 245,760, 491,520 and 983,040 locations (tile 4096). `minbbox.py` gives the
  same tiles as the C++ generator for all five.
- `qsr.py --solver sa` (500 reads) returns the exact optimum for 61,440-245,760 locations; QUBOs have 40-48 bits.
  `--solver qpu` follows the code used for the published QPU runs (clique embedding on Advantage2, chain strength
  relative to the largest coupling) but was not run for this branch; it uses no spin-reversal transforms.
- `shortseam.py --tile-order hilbert` reproduces, bit for bit, the research orderings for the same five sets;
  `spectral_tile_order` reproduces the spectral tile orders of the tile-order study for 122,880 and 245,760.
- `--nonuniform` / `--density`: without the options, `qsr.py` (five 90K competition sets, tile 960) and `shortseam.py`
  (Satellite and Regular, the two it can run) give the same permutations as before these options were added. With the
  options, the permutations equal those of the research prototypes that produced the table above: `qsr.py
  --nonuniform` on the five 900K sets, `shortseam.py --density --tile-order hilbert` on the five 90K sets and on four
  900K sets (Chessboard at 900K was not rerun). `qsr.py --nonuniform --qubo` writes a 48-bit QUBO for the 90K
  Chessboard set; no sampler was run on it.
- End to end (StarPU build, 16,384 locations, tile 1024): ExaGeoStat's generated locations after
  `--ordering=file` equal the CSV rows in permutation order for both tools; wrong `n`, wrong point set and wrong
  row order are rejected. The 61,440-location QSR permutation is accepted for `--N=61440 --seed=42`.

## Limits

- 2D only for both tools (QSR in 3D exists in the research code but is not included here).
- The number of locations must be a multiple of the tile size.
- The QSR surrogate was calibrated at tile 4096 on locations in the unit square; other tile sizes use it unchanged.
- The spectral tile order was evaluated with a TLR Cholesky flop model (shorter critical path than Hilbert), not
  with measured HiCMA runs.
- `--ordering=file` is wired into the StarPU synthetic-data path and the CSV loader; the PaRSEC generator is
  unchanged (apply the permutation to the locations there, or reorder the input file).
