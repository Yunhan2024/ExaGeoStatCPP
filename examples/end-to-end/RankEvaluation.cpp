
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// ExaGeoStat is a software package, provided by King Abdullah University of Science and Technology (KAUST).

/**
 * @file RankEvaluation.cpp
 * @brief Standalone tool to evaluate TLR tile ranks without HiCMA dependency.
 * @details Uses SVD to compute numerical rank for each off-diagonal tile,
 *          allowing comparison of different clustering/ordering methods.
 * @version 1.0.0
 * @author ExaGeoStat Team
 * @date 2024
 **/

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>

#include <api/ExaGeoStat.hpp>
#include <kernels/Kernel.hpp>
#include <common/PluginRegistry.hpp>

// LAPACK SVD declaration
extern "C" {
    void dgesvd_(const char* jobu, const char* jobvt, const int* m, const int* n,
                 double* a, const int* lda, double* s, double* u, const int* ldu,
                 double* vt, const int* ldvt, double* work, const int* lwork, int* info);
}

using namespace exageostat::configurations;
using namespace exageostat::dataunits;
using namespace exageostat::kernels;
using namespace exageostat::common;

/**
 * @brief Compute numerical rank of a matrix using SVD
 * @param[in] matrix The matrix data (column-major)
 * @param[in] rows Number of rows
 * @param[in] cols Number of columns
 * @param[in] tolerance Relative tolerance for rank determination
 * @return The numerical rank
 */
int ComputeNumericalRank(double* matrix, int rows, int cols, double tolerance) {
    int m = rows;
    int n = cols;
    int min_mn = std::min(m, n);
    int lda = m;
    int ldu = 1;
    int ldvt = 1;

    // Allocate arrays for SVD
    std::vector<double> s(min_mn);
    std::vector<double> work_query(1);
    int lwork = -1;
    int info = 0;

    // Query optimal work size
    dgesvd_("N", "N", &m, &n, matrix, &lda, s.data(), nullptr, &ldu, nullptr, &ldvt,
            work_query.data(), &lwork, &info);

    lwork = static_cast<int>(work_query[0]);
    std::vector<double> work(lwork);

    // Make a copy of matrix since SVD destroys it
    std::vector<double> matrix_copy(matrix, matrix + rows * cols);

    // Compute SVD (singular values only)
    dgesvd_("N", "N", &m, &n, matrix_copy.data(), &lda, s.data(), nullptr, &ldu,
            nullptr, &ldvt, work.data(), &lwork, &info);

    if (info != 0) {
        std::cerr << "SVD failed with info = " << info << std::endl;
        return min_mn;
    }

    // Compute numerical rank based on tolerance
    double threshold = tolerance * s[0];
    int rank = 0;
    for (int i = 0; i < min_mn; i++) {
        if (s[i] > threshold) {
            rank++;
        } else {
            break;
        }
    }

    return rank;
}

/**
 * @brief Main entry point for the RankEvaluation tool.
 */
int main(int argc, char **argv) {

    std::cout << "=============================================" << std::endl;
    std::cout << "TLR Rank Evaluation Tool (No HiCMA Required)" << std::endl;
    std::cout << "=============================================" << std::endl;

    // Create configurations and parse arguments
    Configurations configurations;
    configurations.InitializeArguments(argc, argv);

    // Get parameters
    int N = configurations.GetProblemSize();
    int lts = configurations.GetLowTileSize();
    int accuracy = 7;  // Default
    try {
        accuracy = configurations.GetAccuracy();
    } catch (...) {}

    double tolerance = std::pow(10.0, -accuracy);

    // Initialize hardware (minimal, just for data loading)
#if DEFAULT_RUNTIME
    auto hardware = ExaGeoStatHardware(EXACT_DENSE, configurations.GetCoresNumber(),
                                       configurations.GetGPUsNumbers(),
                                       configurations.GetPGrid(),
                                       configurations.GetQGrid());
#else
    auto hardware = ExaGeoStatHardware(configurations);
#endif

    // Load data (locations)
    std::unique_ptr<ExaGeoStatData<double>> data;
    exageostat::api::ExaGeoStat<double>::ExaGeoStatLoadData(configurations, data);

    // Get locations
    Locations<double>* locations = data->GetLocations();
    int n_points = locations->GetSize();
    Dimension dimension = locations->GetDimension();

    // Create kernel
    Kernel<double>* kernel = exageostat::plugins::PluginRegistry<Kernel<double>>::Create(
            configurations.GetKernelName(), configurations.GetTimeSlot());

    // Get theta parameters
    std::vector<double> theta = configurations.GetInitialTheta();

    // Calculate number of tiles
    int n_tiles = (n_points + lts - 1) / lts;

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  N = " << n_points << std::endl;
    std::cout << "  Tile size (lts) = " << lts << std::endl;
    std::cout << "  Number of tiles = " << n_tiles << std::endl;
    std::cout << "  Accuracy = 10^-" << accuracy << std::endl;
    std::cout << "  Kernel = " << configurations.GetKernelName() << std::endl;

    std::string clustering_mode = "none";
    try {
        clustering_mode = configurations.GetClusteringMode();
    } catch (...) {}
    std::cout << "  Clustering mode = " << clustering_mode << std::endl;

    // Prepare location subset structures for tile generation
    Locations<double> loc_row(lts, dimension);
    Locations<double> loc_col(lts, dimension);

    // Allocate tile buffer
    std::vector<double> tile_data(lts * lts);

    // Statistics
    std::vector<int> ranks;
    int min_rank = INT_MAX;
    int max_rank = 0;
    double sum_rank = 0.0;
    int tile_count = 0;

    std::cout << "\nComputing tile ranks..." << std::endl;

    // Distance metric
    int distance_metric = 0;  // Euclidean
    try {
        distance_metric = static_cast<int>(configurations.GetDistanceMetric());
    } catch (...) {}

    // Iterate over lower triangular off-diagonal tiles
    for (int ti = 1; ti < n_tiles; ti++) {
        int row_start = ti * lts;
        int row_end = std::min(row_start + lts, n_points);
        int actual_rows = row_end - row_start;

        for (int tj = 0; tj < ti; tj++) {
            int col_start = tj * lts;
            int col_end = std::min(col_start + lts, n_points);
            int actual_cols = col_end - col_start;

            // Skip if tile is too small
            if (actual_rows < 2 || actual_cols < 2) continue;

            // Setup row locations
            loc_row.SetSize(actual_rows);
            for (int i = 0; i < actual_rows; i++) {
                loc_row.GetLocationX()[i] = locations->GetLocationX()[row_start + i];
                loc_row.GetLocationY()[i] = locations->GetLocationY()[row_start + i];
                if (dimension == Dimension3D && locations->GetLocationZ() != nullptr) {
                    loc_row.GetLocationZ()[i] = locations->GetLocationZ()[row_start + i];
                }
            }

            // Setup column locations
            loc_col.SetSize(actual_cols);
            for (int i = 0; i < actual_cols; i++) {
                loc_col.GetLocationX()[i] = locations->GetLocationX()[col_start + i];
                loc_col.GetLocationY()[i] = locations->GetLocationY()[col_start + i];
                if (dimension == Dimension3D && locations->GetLocationZ() != nullptr) {
                    loc_col.GetLocationZ()[i] = locations->GetLocationZ()[col_start + i];
                }
            }

            // Generate covariance matrix for this tile
            tile_data.assign(actual_rows * actual_cols, 0.0);
            kernel->GenerateCovarianceMatrix(tile_data.data(), actual_rows, actual_cols,
                                             0, 0, loc_row, loc_col, loc_row,
                                             theta.data(), distance_metric);

            // Compute numerical rank
            int rank = ComputeNumericalRank(tile_data.data(), actual_rows, actual_cols, tolerance);

            ranks.push_back(rank);
            if (rank < min_rank) min_rank = rank;
            if (rank > max_rank) max_rank = rank;
            sum_rank += rank;
            tile_count++;
        }

        // Progress indicator
        if (ti % 10 == 0 || ti == n_tiles - 1) {
            std::cout << "  Processed tile row " << ti << "/" << (n_tiles - 1) << "\r" << std::flush;
        }
    }
    std::cout << std::endl;

    // Calculate statistics
    double avg_rank = (tile_count > 0) ? (sum_rank / tile_count) : 0.0;

    // Calculate variance
    double variance = 0.0;
    for (int r : ranks) {
        variance += (r - avg_rank) * (r - avg_rank);
    }
    variance = (tile_count > 1) ? (variance / (tile_count - 1)) : 0.0;
    double std_dev = std::sqrt(variance);

    // Calculate median
    std::sort(ranks.begin(), ranks.end());
    double median_rank = 0.0;
    if (tile_count > 0) {
        if (tile_count % 2 == 0) {
            median_rank = (ranks[tile_count/2 - 1] + ranks[tile_count/2]) / 2.0;
        } else {
            median_rank = ranks[tile_count/2];
        }
    }

    // Output results
    std::cout << "\n=============================================" << std::endl;
    std::cout << "RANK STATISTICS" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "Clustering Mode: " << clustering_mode << std::endl;
    std::cout << "Number of Tiles: " << tile_count << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Min Rank:    " << min_rank << std::endl;
    std::cout << "Max Rank:    " << max_rank << std::endl;
    std::cout << "Avg Rank:    " << avg_rank << std::endl;
    std::cout << "Median Rank: " << median_rank << std::endl;
    std::cout << "Std Dev:     " << std_dev << std::endl;
    std::cout << "=============================================" << std::endl;

    // Compression ratio estimate
    double dense_storage = static_cast<double>(tile_count) * lts * lts;
    double tlr_storage = sum_rank * 2.0 * lts;  // U and V factors
    double compression = (tlr_storage > 0) ? (dense_storage / tlr_storage) : 0.0;
    std::cout << "Est. Compression: " << compression << "x" << std::endl;
    std::cout << "=============================================" << std::endl;

    delete kernel;

    return 0;
}
