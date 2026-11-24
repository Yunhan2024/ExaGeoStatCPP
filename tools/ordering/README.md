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

## Checks done for this branch

- `qsr.py` (default exact solver) reproduces, bit for bit, the QSR orderings measured on Shaheen III for the
  seed-42 2D sets with 61,440, 122,880, 245,760, 491,520 and 983,040 locations (tile 4096). `minbbox.py` gives the
  same tiles as the C++ generator for all five.
- `qsr.py --solver sa` (500 reads) returns the exact optimum for 61,440-245,760 locations; QUBOs have 40-48 bits.
  `--solver qpu` follows the code used for the published QPU runs (clique embedding on Advantage2, chain strength
  relative to the largest coupling) but was not run for this branch; it uses no spin-reversal transforms.
- `shortseam.py --tile-order hilbert` reproduces, bit for bit, the research orderings for the same five sets;
  `spectral_tile_order` reproduces the spectral tile orders of the tile-order study for 122,880 and 245,760.
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
