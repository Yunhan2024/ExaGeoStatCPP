
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// ExaGeoStat is a software package, provided by King Abdullah University of Science and Technology (KAUST).

/**
 * @file HicmaImplementation.cpp
 * @brief Sets up the HiCMA descriptors needed for the tile low rank computations in ExaGeoStat.
 * @version 1.1.0
 * @author Omar Marzouk
 * @author Mahmoud ElKarargy
 * @author Sameh Abdulah
 * @date 2024-02-04
 **/

#ifdef USE_MKL
#include <mkl_service.h>
#endif
#include <climits>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <linear-algebra-solvers/concrete/hicma/tlr/HicmaImplementation.hpp>

using namespace std;
using namespace exageostat::linearAlgebra::tileLowRank;
using namespace exageostat::common;
using namespace exageostat::runtime;
using namespace exageostat::dataunits;
using namespace exageostat::kernels;
using namespace exageostat::helpers;
using namespace exageostat::results;

int store_only_diagonal_tiles = 1;
int use_scratch = 1;
int global_check = 0;  // used to create dense matrix for accuracy check

template<typename T>
void HicmaImplementation<T>::SetModelingDescriptors(std::unique_ptr<ExaGeoStatData<T>> &aData,
                                                    configurations::Configurations &aConfigurations, 
                                                    const int &aP) {

    int full_problem_size = aConfigurations.GetProblemSize() * aP;
    int lts = aConfigurations.GetLowTileSize();
    int p_grid = ExaGeoStatHardware::GetPGrid();
    int q_grid = ExaGeoStatHardware::GetQGrid();
    bool is_OOC = aConfigurations.GetIsOOC();
    int max_rank = aConfigurations.GetMaxRank();

    // Set the floating point precision based on the template type
    FloatPoint float_point;
    if (sizeof(T) == SIZE_OF_FLOAT) {
        float_point = EXAGEOSTAT_REAL_FLOAT;
    } else if (sizeof(T) == SIZE_OF_DOUBLE) {
        float_point = EXAGEOSTAT_REAL_DOUBLE;
    } else {
        throw runtime_error("Unsupported for now!");
    }

    int MBD = lts;
    int NBD = lts;
    int MD = full_problem_size;
    int ND = MBD;

#ifdef USE_MPI
    // Due to a bug in HiCMA with MPI, the first created descriptor is not seen.
    aData->GetDescriptorData()->SetDescriptor(common::HICMA_DESCRIPTOR, DESCRIPTOR_CD, is_OOC, nullptr, 
                                              float_point, lts, lts, lts * lts, full_problem_size, 1, 
                                              0, 0, full_problem_size, 1, p_grid, q_grid);
#endif

    aData->GetDescriptorData()->SetDescriptor(common::HICMA_DESCRIPTOR, DESCRIPTOR_CD, is_OOC, nullptr,
                                              float_point, MBD, NBD, MBD * NBD, MD, ND,
                                              0, 0, MD, ND, p_grid, q_grid);

    // Create HICMA descriptor for C (Dense) with lts tile size.
    // The CHAMELEON DESCRIPTOR_C uses dts, but HICMA_dgytlr_Tile needs lts-sized tiles.
    aData->GetDescriptorData()->SetDescriptor(common::HICMA_DESCRIPTOR, DESCRIPTOR_C, is_OOC, nullptr,
                                              float_point, lts, lts, lts * lts, full_problem_size, full_problem_size,
                                              0, 0, full_problem_size, full_problem_size, p_grid, q_grid);

    int MBUV = lts;
    int NBUV = 2 * max_rank;
    int MUV;
    int N_over_lts_times_lts = full_problem_size / lts * lts;
    
    if (N_over_lts_times_lts < full_problem_size) {
        MUV = N_over_lts_times_lts + lts;
    } else if (N_over_lts_times_lts == full_problem_size) {
        MUV = N_over_lts_times_lts;
    } else {
        throw runtime_error("This case can't happens, N need to be >= lts*lts");
    }
    
    int expr = MUV / lts;
    int NUV = 2 * expr * max_rank;
    
    aData->GetDescriptorData()->SetDescriptor(common::HICMA_DESCRIPTOR, DESCRIPTOR_CUV, is_OOC, nullptr, 
                                              float_point, MBUV, NBUV, MBUV * NBUV, MUV, NUV, 
                                              0, 0, MUV, NUV, p_grid, q_grid);

    auto *HICMA_descCUV = aData->GetDescriptorData()->GetDescriptor(DescriptorType::HICMA_DESCRIPTOR,
                                                                    DescriptorName::DESCRIPTOR_CUV).hicma_desc;

    int MBrk = 1;
    int NBrk = 1;
    int Mrk = HICMA_descCUV->mt;
    int Nrk = HICMA_descCUV->mt;
    
    aData->GetDescriptorData()->SetDescriptor(common::HICMA_DESCRIPTOR, DESCRIPTOR_CRK, is_OOC, nullptr, 
                                              float_point, MBrk, NBrk, MBrk * NBrk, Mrk, Nrk, 
                                              0, 0, Mrk, Nrk, p_grid, q_grid);

    aData->GetDescriptorData()->SetDescriptor(common::HICMA_DESCRIPTOR, DESCRIPTOR_Z, is_OOC, nullptr, 
                                              float_point, lts, lts, lts * lts, full_problem_size, 1, 
                                              0, 0, full_problem_size, 1, p_grid, q_grid);

    aData->GetDescriptorData()->SetDescriptor(common::HICMA_DESCRIPTOR, DESCRIPTOR_Z_COPY, is_OOC, nullptr, 
                                              float_point, lts, lts, lts * lts, full_problem_size, 1, 
                                              0, 0, full_problem_size, 1, p_grid, q_grid);

    aData->GetDescriptorData()->SetDescriptor(common::HICMA_DESCRIPTOR, DESCRIPTOR_DETERMINANT, is_OOC, nullptr,
                                              float_point, lts, lts, lts * lts, 1, 1, 
                                              0, 0, 1, 1, p_grid, q_grid);

    if (aConfigurations.GetIsNonGaussian()) {
        aData->GetDescriptorData()->SetDescriptor(common::HICMA_DESCRIPTOR, DESCRIPTOR_PRODUCT, is_OOC, nullptr,
                                                  float_point, lts, lts, lts * lts, 1, 1, 
                                                  0, 0, 1, 1, p_grid, q_grid);
        aData->GetDescriptorData()->SetDescriptor(common::HICMA_DESCRIPTOR, DESCRIPTOR_SUM, is_OOC, nullptr,
                                                  float_point, lts, lts, lts * lts, 1, 1, 
                                                  0, 0, 1, 1, p_grid, q_grid);
    }
}

template<typename T>
T HicmaImplementation<T>::ExaGeoStatMLETile(std::unique_ptr<ExaGeoStatData<T>> &aData,
                                            configurations::Configurations &aConfigurations, 
                                            const double *theta,
                                            T *apMeasurementsMatrix, 
                                            const Kernel<T> &aKernel) {

    if (!aData->GetDescriptorData()->GetIsDescriptorInitiated()) {
        this->InitiateDescriptors(aConfigurations, *aData->GetDescriptorData(), aKernel.GetVariablesNumber(),
                                  apMeasurementsMatrix);
    }

    // Create a Hicma sequence, if not initialized before through the same descriptors
    RUNTIME_request_t request_array[2] = {HICMA_REQUEST_INITIALIZER, HICMA_REQUEST_INITIALIZER};
    if (!aData->GetDescriptorData()->GetSequence()) {
        HICMA_sequence_t *sequence;
        this->ExaGeoStatCreateSequence(&sequence);
        aData->GetDescriptorData()->SetSequence(sequence);
        aData->GetDescriptorData()->SetRequest(request_array);
    }
    auto pSequence = (HICMA_sequence_t *) aData->GetDescriptorData()->GetSequence();

    // Initialization
    T loglik, logdet, test_time, variance, variance1 = 1, variance2 = 1, variance3;
    T dot_product, dot_product1, dot_product2, dot_product3;
    T dzcpy_time, time_facto, time_solve, logdet_calculate, matrix_gen_time;
    double accumulated_executed_time, accumulated_flops;

    int NRHS, i;
    T flops = 0.0;

    int N;
    int lts;
    int max_rank = aConfigurations.GetMaxRank();
    int iter_count = aData->GetMleIterations();
    auto kernel_name = aConfigurations.GetKernelName();
    int num_params = aKernel.GetParametersNumbers();
    int acc = aConfigurations.GetAccuracy();

    if (iter_count == 0) {
        this->SetModelingDescriptors(aData, aConfigurations, aKernel.GetVariablesNumber());
    }

    // Get descriptor pointers
    auto *HICMA_descCUV = aData->GetDescriptorData()->GetDescriptor(DescriptorType::HICMA_DESCRIPTOR,
                                                                    DescriptorName::DESCRIPTOR_CUV).hicma_desc;
    auto *HICMA_descC = aData->GetDescriptorData()->GetDescriptor(DescriptorType::HICMA_DESCRIPTOR,
                                                                  DescriptorName::DESCRIPTOR_C).hicma_desc;
    auto *HICMA_descCD = aData->GetDescriptorData()->GetDescriptor(DescriptorType::HICMA_DESCRIPTOR,
                                                                   DescriptorName::DESCRIPTOR_CD).hicma_desc;
    auto *HICMA_descCrk = aData->GetDescriptorData()->GetDescriptor(DescriptorType::HICMA_DESCRIPTOR,
                                                                    DescriptorName::DESCRIPTOR_CRK).hicma_desc;
    auto *HICMA_descZ = aData->GetDescriptorData()->GetDescriptor(DescriptorType::HICMA_DESCRIPTOR,
                                                                  DescriptorName::DESCRIPTOR_Z).hicma_desc;
    auto *CHAM_descZ = aData->GetDescriptorData()->GetDescriptor(DescriptorType::CHAMELEON_DESCRIPTOR,
                                                                 DescriptorName::DESCRIPTOR_Z).chameleon_desc;
    auto *CHAM_descZcpy = aData->GetDescriptorData()->GetDescriptor(DescriptorType::CHAMELEON_DESCRIPTOR,
                                                                    DescriptorName::DESCRIPTOR_Z_COPY).chameleon_desc;
    auto *HICMA_descZcpy = aData->GetDescriptorData()->GetDescriptor(DescriptorType::HICMA_DESCRIPTOR,
                                                                     DescriptorName::DESCRIPTOR_Z_COPY).hicma_desc;
    auto *HICMA_desc_det = aData->GetDescriptorData()->GetDescriptor(DescriptorType::HICMA_DESCRIPTOR,
                                                                     DescriptorName::DESCRIPTOR_DETERMINANT).hicma_desc;
    auto *CHAM_desc_product = aData->GetDescriptorData()->GetDescriptor(DescriptorType::CHAMELEON_DESCRIPTOR,
                                                                        DescriptorName::DESCRIPTOR_PRODUCT).chameleon_desc;
    auto *HICMA_desc_sum = aData->GetDescriptorData()->GetDescriptor(DescriptorType::HICMA_DESCRIPTOR,
                                                                     DescriptorName::DESCRIPTOR_SUM).chameleon_desc;

    N = HICMA_descCUV->m;
    NRHS = HICMA_descZ->n;
    lts = HICMA_descZ->mb;

    T *determinant = aData->GetDescriptorData()->GetDescriptorMatrix(HICMA_DESCRIPTOR, DESCRIPTOR_DETERMINANT);
    *determinant = 0;
    T *product = aData->GetDescriptorData()->GetDescriptorMatrix(CHAMELEON_DESCRIPTOR, DESCRIPTOR_PRODUCT);
    *product = 0;
    T *sum;

    string recovery_file = aConfigurations.GetRecoveryFile();
    if (recovery_file.empty() ||
        !(this->Recover((char *) (recovery_file.c_str()), iter_count, (T *) theta, &loglik, num_params))) {
        if (iter_count == 0) {
            // Copy dense matrix Z into tile low rank Z.
            this->CopyDescriptors(CHAM_descZ, HICMA_descZ, N, CHAMELEON_TO_HICMA);
            // Save a copy of descZ into descZcpy for restoring each iteration
            this->ExaGeoStatLapackCopyTile(EXAGEOSTAT_UPPER_LOWER, HICMA_descZ, HICMA_descZcpy);
            // Save another copy into descZcpy for chameleon, This is in case of other operations after Modeling. ex: Prediction.
            CHAMELEON_dlacpy_Tile(ChamUpperLower, CHAM_descZ, CHAM_descZcpy);
        }
    }

    // Matrix generation part.
    VERBOSE("LR:Generate New Covariance Matrix...")
    START_TIMING(matrix_gen_time);

    HICMA_problem_t hicma_problem;
    hicma_problem.theta = (double *) theta;
    hicma_problem.noise = 1e-4;
    hicma_problem.ndim = 2;

    hicma_problem.kernel_type = aConfigurations.GetDistanceMetric() == common::GREAT_CIRCLE_DISTANCE
                                ? STARSH_SPATIAL_MATERN2_GCD
                                : STARSH_SPATIAL_MATERN2_SIMD;

    int hicma_data_type;
    if (aConfigurations.GetIsNonGaussian()) {
        hicma_data_type = HICMA_STARSH_PROB_GEOSTAT_NON_GAUSSIAN;
    } else {
        // Use GEOSTAT_POINT to pass external coordinates from LocationGenerator.
        // This ensures STARSH uses ExaGeoStat's ordered coordinates (e.g. graph_part)
        // instead of internally regenerating coordinates with Morton ordering.
        hicma_data_type = HICMA_STARSH_PROB_GEOSTAT_POINT;
    }

    // Flatten locations into dimension-major layout: [x1..xN, y1..yN]
    // as required by starsh_ssdata_init()
    auto *locations = aData->GetLocations();
    double *point = new double[N * hicma_problem.ndim];
    for (int idx = 0; idx < N; idx++) {
        point[idx] = static_cast<double>(locations->GetLocationX()[idx]);
        point[N + idx] = static_cast<double>(locations->GetLocationY()[idx]);
    }
    hicma_problem.point = point;

    HICMA_zgenerate_problem(hicma_data_type, 'S', 0, N, lts, HICMA_descCUV->mt, HICMA_descCUV->nt,
                            &hicma_problem);
    int compress_diag = 0;
    HICMA_dgytlr_Tile(EXAGEOSTAT_LOWER, HICMA_descCUV, HICMA_descCD, HICMA_descCrk, 0, max_rank,
                      pow(10, -1.0 * acc), compress_diag, HICMA_descC);

    STOP_TIMING(matrix_gen_time);
    VERBOSE("Done.")

    // Extract and print TLR rank statistics
    {
        int mt = HICMA_descCrk->mt;
        int nt = HICMA_descCrk->nt;
        T *rank_matrix = (T *) HICMA_descCrk->mat;

        int min_rank = INT_MAX;
        int max_rank_val = 0;
        double sum_rank = 0.0;
        int count = 0;

        // Iterate over lower triangular tiles (excluding diagonal)
        for (int j = 0; j < nt; j++) {
            for (int i = j + 1; i < mt; i++) {
                int rank_val = (int) rank_matrix[j * mt + i];
                if (rank_val > 0) {  // Only count non-zero ranks
                    if (rank_val < min_rank) min_rank = rank_val;
                    if (rank_val > max_rank_val) max_rank_val = rank_val;
                    sum_rank += rank_val;
                    count++;
                }
            }
        }

        double avg_rank = (count > 0) ? (sum_rank / count) : 0.0;
        if (min_rank == INT_MAX) min_rank = 0;

        VERBOSE("TLR Rank Statistics after compression:")
        VERBOSE("  Min Rank: " << min_rank)
        VERBOSE("  Max Rank: " << max_rank_val)
        VERBOSE("  Avg Rank: " << avg_rank)
        VERBOSE("  Number of TLR tiles: " << count)

        // Store in Results for summary output
        Results::GetInstance()->SetTLRRankStatistics(min_rank, max_rank_val, avg_rank);

        // Per-distance rank distribution for analysis
        {
            int max_dist = mt;
            std::vector<double> dist_sum(max_dist, 0.0);
            std::vector<int> dist_count(max_dist, 0);
            std::vector<int> dist_max(max_dist, 0);
            for (int j = 0; j < nt; j++) {
                for (int i = j + 1; i < mt; i++) {
                    int rv = (int) rank_matrix[j * mt + i];
                    if (rv <= 0) continue;
                    int d = i - j;
                    dist_sum[d] += rv;
                    dist_count[d]++;
                    if (rv > dist_max[d]) dist_max[d] = rv;
                }
            }
            fprintf(stderr, "RANK_BY_DIST dist avg_rank max_rank count\n");
            for (int d = 1; d < max_dist && dist_count[d] > 0; d++) {
                fprintf(stderr, "RANK_BY_DIST %d %.2f %d %d\n",
                        d, dist_sum[d] / dist_count[d], dist_max[d], dist_count[d]);
            }
        }

        // η vs rank analysis: compute per-tile-pair admissibility ratio and correlate with rank
        {
            // Compute per-tile centroid and diameter
            std::vector<double> tile_cx(mt, 0), tile_cy(mt, 0), tile_diam(mt, 0);
            for (int t = 0; t < mt; t++) {
                int tlo = t * lts, thi = std::min(tlo + lts, N);
                double sx = 0, sy = 0;
                for (int p = tlo; p < thi; p++) { sx += point[p]; sy += point[N + p]; }
                int cnt = thi - tlo;
                tile_cx[t] = sx / cnt;
                tile_cy[t] = sy / cnt;
                double md2 = 0;
                for (int p = tlo; p < thi; p++)
                    for (int q = p + 1; q < thi; q++) {
                        double dx = point[p] - point[q], dy = point[N+p] - point[N+q];
                        double d2 = dx*dx + dy*dy;
                        if (d2 > md2) md2 = d2;
                    }
                tile_diam[t] = std::sqrt(md2);
            }

            // Output η vs rank for each tile pair
            fprintf(stderr, "ETA_VS_RANK eta rank tile_i tile_j centroid_dist max_diam\n");
            // Bin η into buckets for summary
            const int nbins = 20;
            double eta_max = 10.0;
            std::vector<double> bin_rank_sum(nbins, 0);
            std::vector<int> bin_count(nbins, 0);
            int pair_count = 0;
            for (int j = 0; j < nt; j++) {
                for (int i = j + 1; i < mt; i++) {
                    int rv = (int) rank_matrix[j * mt + i];
                    if (rv <= 0) continue;
                    double dx = tile_cx[i] - tile_cx[j], dy = tile_cy[i] - tile_cy[j];
                    double cdist = std::sqrt(dx*dx + dy*dy);
                    double max_d = std::max(tile_diam[i], tile_diam[j]);
                    double eta = (cdist > 1e-15) ? max_d / cdist : 999.0;
                    int bin = std::min((int)(eta / eta_max * nbins), nbins - 1);
                    bin_rank_sum[bin] += rv;
                    bin_count[bin]++;
                    // Output all pairs
                    if (pair_count < 99999) {
                        fprintf(stderr, "ETA_VS_RANK %.4f %d %d %d %.4f %.4f\n",
                                eta, rv, i, j, cdist, max_d);
                    }
                    pair_count++;
                }
            }
            fprintf(stderr, "ETA_BINNED eta_lo eta_hi avg_rank count\n");
            for (int b = 0; b < nbins; b++) {
                if (bin_count[b] == 0) continue;
                fprintf(stderr, "ETA_BINNED %.2f %.2f %.2f %d\n",
                        b * eta_max / nbins, (b+1) * eta_max / nbins,
                        bin_rank_sum[b] / bin_count[b], bin_count[b]);
            }
            // Fine-grained η bins for low-η region [0, 0.5]
            {
                const int fbins = 10;
                double fmax = 0.5;
                std::vector<double> fsum(fbins, 0);
                std::vector<int> fcount(fbins, 0);
                std::vector<int> fmin_r(fbins, INT_MAX), fmax_r(fbins, 0);
                for (int j = 0; j < nt; j++) {
                    for (int i = j + 1; i < mt; i++) {
                        int rv = (int) rank_matrix[j * mt + i];
                        if (rv <= 0) continue;
                        double dx = tile_cx[i] - tile_cx[j], dy = tile_cy[i] - tile_cy[j];
                        double cdist = std::sqrt(dx*dx + dy*dy);
                        double max_d = std::max(tile_diam[i], tile_diam[j]);
                        double eta = (cdist > 1e-15) ? max_d / cdist : 999.0;
                        if (eta >= fmax) continue;
                        int fb = std::min((int)(eta / fmax * fbins), fbins - 1);
                        fsum[fb] += rv; fcount[fb]++;
                        if (rv < fmin_r[fb]) fmin_r[fb] = rv;
                        if (rv > fmax_r[fb]) fmax_r[fb] = rv;
                    }
                }
                fprintf(stderr, "ETA_FINE eta_lo eta_hi avg_rank min_rank max_rank count\n");
                for (int b = 0; b < fbins; b++) {
                    if (fcount[b] == 0) continue;
                    fprintf(stderr, "ETA_FINE %.3f %.3f %.2f %d %d %d\n",
                            b * fmax / fbins, (b+1) * fmax / fbins,
                            fsum[b] / fcount[b], fmin_r[b], fmax_r[b], fcount[b]);
                }
            }
        // Graph cut analysis: build k-NN graph, count cross-tile edges per tile pair
        {
            const int knn = 8;
            // For each point, find k nearest neighbors, record their tile IDs
            // cross_cut[i*mt+j] = number of k-NN edges between tile i and tile j
            std::vector<int> cross_cut(mt * mt, 0);
            int total_cross = 0;

            for (int p = 0; p < N; p++) {
                int tile_p = p / lts;
                if (tile_p >= mt) tile_p = mt - 1;
                double px = point[p], py = point[N + p];

                // Find k nearest neighbors (brute force, fine for diagnostics)
                // Use a partial sort approach: keep top-k closest
                struct NN { double d2; int idx; };
                std::vector<NN> heap;
                heap.reserve(knn + 1);

                for (int q = 0; q < N; q++) {
                    if (q == p) continue;
                    double dx = px - point[q], dy = py - point[N + q];
                    double d2 = dx*dx + dy*dy;
                    if ((int)heap.size() < knn) {
                        heap.push_back({d2, q});
                        if ((int)heap.size() == knn)
                            std::make_heap(heap.begin(), heap.end(), [](const NN& a, const NN& b){ return a.d2 < b.d2; });
                    } else if (d2 < heap[0].d2) {
                        std::pop_heap(heap.begin(), heap.end(), [](const NN& a, const NN& b){ return a.d2 < b.d2; });
                        heap.back() = {d2, q};
                        std::push_heap(heap.begin(), heap.end(), [](const NN& a, const NN& b){ return a.d2 < b.d2; });
                    }
                }

                for (auto& nn : heap) {
                    int tile_q = nn.idx / lts;
                    if (tile_q >= mt) tile_q = mt - 1;
                    if (tile_p != tile_q) {
                        cross_cut[tile_p * mt + tile_q]++;
                        total_cross++;
                    }
                }
            }

            // Output: per tile pair, cross_cut count vs rank
            fprintf(stderr, "GRAPH_CUT_VS_RANK cut rank eta tile_i tile_j\n");
            int pair_out = 0;
            for (int j = 0; j < nt; j++) {
                for (int i = j + 1; i < mt; i++) {
                    int rv = (int) rank_matrix[j * mt + i];
                    if (rv <= 0) continue;
                    int cut = cross_cut[i * mt + j] + cross_cut[j * mt + i];
                    double dx = tile_cx[i] - tile_cx[j], dy = tile_cy[i] - tile_cy[j];
                    double cdist = std::sqrt(dx*dx + dy*dy);
                    double max_d = std::max(tile_diam[i], tile_diam[j]);
                    double eta = (cdist > 1e-15) ? max_d / cdist : 999.0;
                    if (pair_out < 99999)
                        fprintf(stderr, "GRAPH_CUT_VS_RANK %d %d %.4f %d %d\n", cut, rv, eta, i, j);
                    pair_out++;
                }
            }
            fprintf(stderr, "GRAPH_CUT_TOTAL %d\n", total_cross);
        }
        }
    }

    // ******************************
    VERBOSE("LR: re-Copy z...")
    START_TIMING(test_time);
    // re-store old Z
    this->ExaGeoStatLapackCopyTile(EXAGEOSTAT_UPPER_LOWER, HICMA_descZcpy, HICMA_descZ);
    STOP_TIMING(test_time);
    VERBOSE("Done.")

    // Calculate Cholesky Factorization (C=LL-1)
    VERBOSE("LR: Cholesky factorization of Sigma...")
    START_TIMING(time_facto);

    this->ExaGeoStatPotrfTile(EXAGEOSTAT_LOWER, HICMA_descCUV, 0, HICMA_descCD, HICMA_descCrk, max_rank, acc);

    STOP_TIMING(time_facto);
    flops = flops + flops_dpotrf(N);
    VERBOSE("Done.")

    // Calculate log(|C|) --> log(square(|L|))
    VERBOSE("LR:Calculating the log determinant ...")
    START_TIMING(logdet_calculate);
    RuntimeFunctions<T>::ExaGeoStatMeasureDetTileAsync(aConfigurations.GetComputation(), HICMA_descCD, pSequence,
                                                       &request_array[0], HICMA_desc_det);
    ExaGeoStatSequenceWait(pSequence);
    logdet = 2 * (*determinant);
    STOP_TIMING(logdet_calculate);
    VERBOSE("Done.")

    if (aConfigurations.GetIsNonGaussian()) {
        VERBOSE("Transform Z vector to Gaussian field ...")
        RuntimeFunctions<T>::ExaGeoStatNonGaussianTransformTileAsync(aConfigurations.GetComputation(), HICMA_descZ,
                                                                     (T *) theta, pSequence, &request_array[0]);
        ExaGeoStatSequenceWait(pSequence);

        VERBOSE(" Done.")
        sum = aData->GetDescriptorData()->GetDescriptorMatrix(HICMA_DESCRIPTOR, DESCRIPTOR_SUM);
        *sum = 0;

        VERBOSE("LR:Calculating dot product...")
        CHAMELEON_dgemm_Tile(ChamTrans, ChamNoTrans, 1, CHAM_descZ, CHAM_descZ, 0, CHAM_desc_product);
        VERBOSE(" Done.")

#ifdef USE_MPI
        double global_sum;
        double local_sum = *product;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
        RuntimeFunctions<T>::ExaGeoStatNonGaussianLogLikeTileAsync(aConfigurations.GetComputation(), HICMA_descZ,
                                                                   HICMA_desc_sum, (T *) theta, pSequence,
                                                                   &request_array[0]);
        ExaGeoStatSequenceWait(pSequence);
        VERBOSE(" Done.")
    }

    // Solving Linear System (L*X=Z)--->inv(L)*Z
    VERBOSE("LR:Solving the linear system ...")
    START_TIMING(time_solve);
    // Compute triangular solve LC*X = Z
    this->ExaGeoStatTrsmTile(EXAGEOSTAT_LEFT, EXAGEOSTAT_LOWER, EXAGEOSTAT_NO_TRANS, EXAGEOSTAT_NON_UNIT, 1,
                             HICMA_descCUV, HICMA_descCD, HICMA_descCrk, HICMA_descZ, max_rank);
    STOP_TIMING(time_solve);
    flops = flops + flops_dtrsm(ChamLeft, N, NRHS);
    VERBOSE("Done.")

    VERBOSE("Copy to chameleon")
    this->CopyDescriptors(HICMA_descZ, CHAM_descZ, N, HICMA_TO_CHAMELEON);

    VERBOSE("LR:Calculating dot product...")
    CHAMELEON_dgemm_Tile(ChamTrans, ChamNoTrans, 1, CHAM_descZ, CHAM_descZ, 0, CHAM_desc_product);
    dot_product = *product;
    loglik = -0.5 * dot_product - 0.5 * logdet;
    
    if (aConfigurations.GetIsNonGaussian()) {
        loglik = loglik - *sum - N * log(theta[3]) - (double) (N / 2.0) * log(2.0 * PI);
    } else {
        loglik = loglik - (double) (N / 2.0) * log(2.0 * PI);
    }
    VERBOSE("Done.")

    LOGGER("\t" << iter_count + 1 << " - Model Parameters (", true)
    if (aConfigurations.GetLogger()) {
        fprintf(aConfigurations.GetFileLogPath(), "\t %d- Model Parameters (", iter_count + 1);
    }

    if ((aConfigurations.GetKernelName() == "bivariate_matern_parsimonious_profile") ||
        (aConfigurations.GetKernelName() == "bivariate_matern_parsimonious2_profile")) {
        LOGGER(setprecision(8) << variance1 << setprecision(8) << variance2)
        if (aConfigurations.GetLogger()) {
            fprintf(aConfigurations.GetFileLogPath(), "%.8f, %.8f,", variance1, variance2);
        }
        i = 2;
    } else {
        i = 0;
    }

    for (; i < num_params; i++) {
        LOGGER_PRECISION(theta[i])
        if (i < num_params - 1) {
            LOGGER_PRECISION(", ")
        }
        if (aConfigurations.GetLogger()) {
            fprintf(aConfigurations.GetFileLogPath(), "%.8f, ", theta[i]);
        }
    }
    LOGGER_PRECISION(")----> LogLi: " << loglik << "\n", 18)

    if (aConfigurations.GetLogger()) {
        fprintf(aConfigurations.GetFileLogPath(), ")----> LogLi: %.18f\n", loglik);
    }

    VERBOSE(" ---- Facto Time: " << time_facto)
    VERBOSE(" ---- Log Determent Time: " << logdet_calculate)
    VERBOSE(" ---- dtrsm Time: " << time_solve)
    VERBOSE(" ---- Matrix Generation Time: " << matrix_gen_time)
    VERBOSE(" ---- Total Time: " << time_facto + logdet_calculate + time_solve)
    VERBOSE(" ---- Gflop/s: " << flops / 1e9 / (time_facto + time_solve))

    aData->SetMleIterations(aData->GetMleIterations() + 1);

    // for experiments and benchmarking
    accumulated_executed_time = Results::GetInstance()->GetTotalModelingExecutionTime() + time_facto + 
                                logdet_calculate + time_solve;
    Results::GetInstance()->SetTotalModelingExecutionTime(accumulated_executed_time);
    accumulated_flops = Results::GetInstance()->GetTotalModelingFlops() + (flops / 1e9 / (time_facto + time_solve));
    Results::GetInstance()->SetTotalModelingFlops(accumulated_flops);

    Results::GetInstance()->SetMLEIterations(iter_count + 1);
    Results::GetInstance()->SetMaximumTheta(vector<double>(theta, theta + num_params));
    Results::GetInstance()->SetLogLikValue(loglik);

    aConfigurations.SetEstimatedTheta(aConfigurations.GetStartingTheta());
    // Due to a leak in HiCMA in ZGEMM, We had to free the buffer manually.
#ifdef USE_MKL
    mkl_free_buffers();
#endif

    // Free coordinate array passed to STARSH via hicma_problem.point.
    // Must outlive all STARSH operations (starsh_ssdata_init stores pointer, not copy).
    delete[] point;

    return loglik;
}

template<typename T>
void HicmaImplementation<T>::ExaGeoStatLapackCopyTile(const UpperLower &aUpperLower, void *apA, void *apB) {
    int status = HICMA_dlacpy_Tile(aUpperLower, (HICMA_desc_t *) apA, (HICMA_desc_t *) apB);
    if (status != HICMA_SUCCESS) {
        throw std::runtime_error("CHAMELEON_dlacpy_Tile Failed!");
    }
}

template<typename T>
void HicmaImplementation<T>::ExaGeoStatSequenceWait(void *apSequence) {
    HICMA_Sequence_Wait((HICMA_sequence_t *) apSequence);
}

template<typename T>
void HicmaImplementation<T>::ExaGeoStatCreateSequence(void *apSequence) {
    int status = HICMA_Sequence_Create((HICMA_sequence_t **) apSequence);
    if (status != HICMA_SUCCESS) {
        throw std::runtime_error("HICMA_Sequence_Create Failed!");
    }
}

template<typename T>
void HicmaImplementation<T>::ExaGeoStatPotrfTile(const common::UpperLower &aUpperLower, void *apA, int aBand,
                                                 void *apCD, void *apCrk, const int &aMaxRank, const int &aAcc) {
    int status = HICMA_dpotrf_Tile(EXAGEOSTAT_LOWER, (HICMA_desc_t *) apA, (HICMA_desc_t *) apCD,
                                   (HICMA_desc_t *) apCrk, aBand, aMaxRank, pow(10, -1.0 * aAcc));
    if (status != HICMA_SUCCESS) {
        throw std::runtime_error("HICMA_dpotrf_Tile Failed, Matrix is not positive definite");
    }
}

template<typename T>
void HicmaImplementation<T>::ExaGeoStatTrsmTile(const common::Side &aSide, const common::UpperLower &aUpperLower,
                                                const common::Trans &aTrans, const common::Diag &aDiag, const T &aAlpha,
                                                void *apA, void *apCD, void *apCrk, void *apZ, const int &aMaxRank) {

    int status = HICMA_dtrsmd_Tile(aSide, aUpperLower, aTrans, aDiag, aAlpha, (HICMA_desc_t *) apA,
                                   (HICMA_desc_t *) apCD, (HICMA_desc_t *) apCrk, (HICMA_desc_t *) apZ, aMaxRank);
    if (status != HICMA_SUCCESS) {
        throw std::runtime_error("HICMA_dtrsmd_Tile Failed!");
    }
}
