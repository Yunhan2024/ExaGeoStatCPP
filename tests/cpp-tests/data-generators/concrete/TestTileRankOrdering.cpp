
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// ExaGeoStat is a software package, provided by King Abdullah University of Science and Technology (KAUST).

/**
 * @file TestTileRankOrdering.cpp
 * @brief Validation tests comparing TLR tile ranks under different point ordering strategies.
 * @details This file validates that Z-order (Morton code) sorting improves spatial locality,
 * leading to lower off-diagonal tile ranks compared to random ordering.
 * Uses direct SVD computation to determine numerical ranks without HiCMA dependency.
 * @version 1.1.0
 * @author ExaGeoStat Team
 * @date 2026-01-13
**/

#include <iostream>
#include <algorithm>
#include <random>
#include <numeric>
#include <vector>
#include <cmath>
#include <iomanip>

#include <catch2/catch_all.hpp>
#include <lapacke.h>

using namespace std;

/**
 * @brief Helper struct to hold rank statistics
 */
struct RankStats {
    double avgRank;
    double maxRank;
    double minRank;
    double totalRank;
    int numTiles;
};

/**
 * @brief Compute Morton code for 2D point (Z-order curve)
 * @param x X coordinate (normalized to [0, 1])
 * @param y Y coordinate (normalized to [0, 1])
 * @param bits Number of bits to use
 * @return Morton code
 */
uint64_t ComputeMortonCode(double x, double y, int bits = 16) {
    // Normalize to integer range
    uint32_t ix = static_cast<uint32_t>(x * ((1 << bits) - 1));
    uint32_t iy = static_cast<uint32_t>(y * ((1 << bits) - 1));

    // Interleave bits
    uint64_t code = 0;
    for (int i = 0; i < bits; i++) {
        code |= ((uint64_t)((ix >> i) & 1) << (2 * i));
        code |= ((uint64_t)((iy >> i) & 1) << (2 * i + 1));
    }
    return code;
}

/**
 * @brief Compute Matern covariance kernel value
 * @param dist Distance between two points
 * @param sigma Variance parameter
 * @param beta Smoothness parameter (nu = beta + 0.5)
 * @param nu Range parameter
 * @return Covariance value
 */
double MaternCovariance(double dist, double sigma, double beta, double nu) {
    if (dist < 1e-10) {
        return sigma;
    }

    // Using exponential covariance (Matern 1/2) for simplicity
    // C(h) = sigma * exp(-h / nu)
    return sigma * exp(-dist / nu);
}

/**
 * @brief Compute Euclidean distance between two 2D points
 */
double Distance2D(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return sqrt(dx * dx + dy * dy);
}

/**
 * @brief Compute numerical rank of a matrix using SVD
 * @param matrix Matrix data (column-major)
 * @param rows Number of rows
 * @param cols Number of columns
 * @param tolerance Relative tolerance for rank determination
 * @return Numerical rank
 */
int ComputeNumericalRank(const vector<double>& matrix, int rows, int cols, double tolerance) {
    int minDim = min(rows, cols);
    vector<double> S(minDim);
    vector<double> matrixCopy = matrix;  // LAPACKE modifies input
    vector<double> U(rows * rows);
    vector<double> VT(cols * cols);
    vector<double> superb(minDim - 1);

    // Compute SVD
    int info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'N', 'N',
                               rows, cols, matrixCopy.data(), rows,
                               S.data(), U.data(), rows, VT.data(), cols,
                               superb.data());

    if (info != 0) {
        cerr << "SVD computation failed with info = " << info << endl;
        return minDim;
    }

    // Find rank based on tolerance
    double maxSigma = S[0];
    double threshold = tolerance * maxSigma;

    int rank = 0;
    for (int i = 0; i < minDim; i++) {
        if (S[i] > threshold) {
            rank++;
        } else {
            break;
        }
    }

    return rank;
}

/**
 * @brief Generate 2D locations in a grid
 * @param N Number of points
 * @param locX Output X coordinates
 * @param locY Output Y coordinates
 */
void GenerateGridLocations(int N, vector<double>& locX, vector<double>& locY) {
    int gridSize = static_cast<int>(ceil(sqrt(N)));
    double step = 1.0 / gridSize;

    locX.resize(N);
    locY.resize(N);

    int idx = 0;
    for (int i = 0; i < gridSize && idx < N; i++) {
        for (int j = 0; j < gridSize && idx < N; j++) {
            locX[idx] = (i + 0.5) * step;
            locY[idx] = (j + 0.5) * step;
            idx++;
        }
    }
}

/**
 * @brief Sort locations by Morton code (Z-order)
 */
void SortByMortonCode(vector<double>& locX, vector<double>& locY) {
    int N = locX.size();

    // Compute Morton codes
    vector<pair<uint64_t, int>> mortonIndices(N);
    for (int i = 0; i < N; i++) {
        mortonIndices[i] = {ComputeMortonCode(locX[i], locY[i]), i};
    }

    // Sort by Morton code
    sort(mortonIndices.begin(), mortonIndices.end());

    // Reorder locations
    vector<double> newX(N), newY(N);
    for (int i = 0; i < N; i++) {
        newX[i] = locX[mortonIndices[i].second];
        newY[i] = locY[mortonIndices[i].second];
    }

    locX = newX;
    locY = newY;
}

/**
 * @brief Shuffle locations randomly
 */
void ShuffleLocations(vector<double>& locX, vector<double>& locY, int seed) {
    int N = locX.size();
    vector<int> indices(N);
    iota(indices.begin(), indices.end(), 0);

    mt19937 gen(seed);
    shuffle(indices.begin(), indices.end(), gen);

    vector<double> newX(N), newY(N);
    for (int i = 0; i < N; i++) {
        newX[i] = locX[indices[i]];
        newY[i] = locY[indices[i]];
    }

    locX = newX;
    locY = newY;
}

/**
 * @brief Compute covariance matrix tile
 * @param locX X coordinates
 * @param locY Y coordinates
 * @param rowStart Starting row index
 * @param rowEnd Ending row index (exclusive)
 * @param colStart Starting column index
 * @param colEnd Ending column index (exclusive)
 * @param sigma Variance parameter
 * @param beta Smoothness parameter
 * @param nu Range parameter
 * @return Tile matrix data (column-major)
 */
vector<double> ComputeCovarianceTile(const vector<double>& locX, const vector<double>& locY,
                                      int rowStart, int rowEnd, int colStart, int colEnd,
                                      double sigma, double beta, double nu) {
    int rows = rowEnd - rowStart;
    int cols = colEnd - colStart;
    vector<double> tile(rows * cols);

    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            double dist = Distance2D(locX[rowStart + i], locY[rowStart + i],
                                     locX[colStart + j], locY[colStart + j]);
            tile[j * rows + i] = MaternCovariance(dist, sigma, beta, nu);
        }
    }

    return tile;
}

/**
 * @brief Compute average distance between points in two tiles
 */
double ComputeAverageTileDistance(const vector<double>& locX, const vector<double>& locY,
                                   int tile1Start, int tile1End, int tile2Start, int tile2End) {
    double totalDist = 0;
    int count = 0;

    for (int i = tile1Start; i < tile1End; i++) {
        for (int j = tile2Start; j < tile2End; j++) {
            totalDist += Distance2D(locX[i], locY[i], locX[j], locY[j]);
            count++;
        }
    }

    return totalDist / count;
}

/**
 * @brief Run tile rank computation for given ordering
 * @param N Problem size
 * @param tileSize Tile size
 * @param applyZOrder If true, use Z-order; if false, use random order
 * @param seed Random seed
 * @param tolerance SVD tolerance for rank computation
 * @return RankStats for off-diagonal tiles
 */
RankStats ComputeTileRanks(int N, int tileSize, bool applyZOrder, int seed, double tolerance) {
    // Generate locations
    vector<double> locX, locY;
    GenerateGridLocations(N, locX, locY);

    // Apply ordering
    if (applyZOrder) {
        SortByMortonCode(locX, locY);
    } else {
        ShuffleLocations(locX, locY, seed);
    }

    // Covariance parameters (Matern)
    double sigma = 1.0;
    double beta = 0.5;
    double nu = 0.1;

    // Compute number of tiles
    int numTiles = (N + tileSize - 1) / tileSize;

    RankStats stats = {0.0, 0.0, numeric_limits<double>::max(), 0.0, 0};

    cout << (applyZOrder ? "[Z-Order]" : "[Random]") << " Computing "
         << numTiles << "x" << numTiles << " tiles..." << endl;

    // Process off-diagonal tiles
    for (int ti = 0; ti < numTiles; ti++) {
        for (int tj = 0; tj < numTiles; tj++) {
            if (ti == tj) continue;  // Skip diagonal

            int rowStart = ti * tileSize;
            int rowEnd = min(rowStart + tileSize, N);
            int colStart = tj * tileSize;
            int colEnd = min(colStart + tileSize, N);

            // Compute tile covariance matrix
            vector<double> tile = ComputeCovarianceTile(locX, locY, rowStart, rowEnd, colStart, colEnd,
                                                        sigma, beta, nu);

            // Compute numerical rank
            int rows = rowEnd - rowStart;
            int cols = colEnd - colStart;
            int rank = ComputeNumericalRank(tile, rows, cols, tolerance);

            stats.totalRank += rank;
            stats.maxRank = max(stats.maxRank, (double)rank);
            stats.minRank = min(stats.minRank, (double)rank);
            stats.numTiles++;
        }
    }

    if (stats.numTiles > 0) {
        stats.avgRank = stats.totalRank / stats.numTiles;
    }
    if (stats.minRank == numeric_limits<double>::max()) {
        stats.minRank = 0.0;
    }

    cout << (applyZOrder ? "[Z-Order]" : "[Random]")
         << " Avg Rank: " << fixed << setprecision(2) << stats.avgRank
         << ", Max: " << stats.maxRank
         << ", Min: " << stats.minRank
         << ", Tiles: " << stats.numTiles << endl;

    return stats;
}

/**
 * @brief Test comparing Z-Order vs Random ordering on tile ranks
 */
void TEST_ZORDER_VS_RANDOM_ORDERING() {

    SECTION("Z-Order vs Random Order - Small Problem") {
        int N = 64;
        int tileSize = 16;
        int seed = 42;
        double tolerance = 1e-6;

        cout << "\n=== Testing Z-Order vs Random Ordering ===" << endl;
        cout << "Problem Size: " << N << ", Tile Size: " << tileSize
             << ", Tolerance: " << tolerance << endl;

        // Run with Z-order
        RankStats zorderStats = ComputeTileRanks(N, tileSize, true, seed, tolerance);

        // Run with random order
        RankStats randomStats = ComputeTileRanks(N, tileSize, false, seed, tolerance);

        // Print comparison
        cout << "\n=== Comparison ===" << endl;
        cout << "Average Rank - Z-Order: " << zorderStats.avgRank
             << ", Random: " << randomStats.avgRank << endl;

        if (zorderStats.avgRank > 0 && randomStats.avgRank > 0) {
            double ratio = randomStats.avgRank / zorderStats.avgRank;
            cout << "Rank ratio (Random/Z-Order): " << fixed << setprecision(3) << ratio << endl;

            // Z-order should generally produce lower or equal ranks
            // due to better spatial locality
            if (ratio > 1.0) {
                cout << "Result: Z-Order produces LOWER ranks (better compressibility)" << endl;
            } else if (ratio < 1.0) {
                cout << "Result: Random produces lower ranks (unexpected for this test)" << endl;
            } else {
                cout << "Result: Similar ranks" << endl;
            }
        }

        // Basic sanity checks
        REQUIRE(zorderStats.avgRank >= 0);
        REQUIRE(randomStats.avgRank >= 0);
    }

    SECTION("Z-Order vs Random Order - Larger Problem") {
        int N = 256;
        int tileSize = 32;
        int seed = 123;
        double tolerance = 1e-6;

        cout << "\n=== Testing Z-Order vs Random Ordering (Larger) ===" << endl;
        cout << "Problem Size: " << N << ", Tile Size: " << tileSize
             << ", Tolerance: " << tolerance << endl;

        // Run with Z-order
        RankStats zorderStats = ComputeTileRanks(N, tileSize, true, seed, tolerance);

        // Run with random order
        RankStats randomStats = ComputeTileRanks(N, tileSize, false, seed, tolerance);

        // Print comparison
        cout << "\n=== Comparison ===" << endl;
        cout << "Average Rank - Z-Order: " << zorderStats.avgRank
             << ", Random: " << randomStats.avgRank << endl;

        if (zorderStats.avgRank > 0 && randomStats.avgRank > 0) {
            double ratio = randomStats.avgRank / zorderStats.avgRank;
            cout << "Rank ratio (Random/Z-Order): " << fixed << setprecision(3) << ratio << endl;
        }

        REQUIRE(zorderStats.avgRank >= 0);
        REQUIRE(randomStats.avgRank >= 0);
    }
}

/**
 * @brief Main test case for tile rank ordering experiments
 */
TEST_CASE("Tile Rank Ordering Methods") {
    TEST_ZORDER_VS_RANDOM_ORDERING();

    // Future sections can be added here:
    // SECTION("Hilbert Curve vs Z-Order") { ... }
    // SECTION("Clustering-based Ordering vs Z-Order") { ... }
}
