
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// ExaGeoStat is a software package, provided by King Abdullah University of Science and Technology (KAUST).

/**
 * @file LocationGenerator.cpp
 * @brief Generates and manages spatial locations for ExaGeoStat.
 * @version 1.1.0
 * @author Mahmoud ElKarargy
 * @date 2024-02-04
**/

#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <algorithm>
#include <vector>
#include <string>
#include <queue>
#include <unordered_map>
#include <utility>

#include <data-generators/LocationGenerator.hpp>
#include <helpers/ByteHandler.hpp>

// ============================================================
// Graph Partition helpers (spectral bisection ordering)
// ============================================================
namespace {

    // Hilbert curve helpers for leaf-level and centroid ordering
    uint64_t XYToHilbert(uint32_t x, uint32_t y, int order) {
        uint64_t d = 0;
        for (int s = order / 2; s > 0; s /= 2) {
            int rx = (x & s) > 0 ? 1 : 0, ry = (y & s) > 0 ? 1 : 0;
            d += (uint64_t) s * s * ((3 * rx) ^ ry);
            if (ry == 0) {
                if (rx == 1) { x = s - 1 - x; y = s - 1 - y; }
                std::swap(x, y);
            }
        }
        return d;
    }

    uint64_t HilbertCode(double fx, double fy) {
        const int ORDER = 1 << 16;
        return XYToHilbert((uint32_t)(fx * (ORDER - 1)), (uint32_t)(fy * (ORDER - 1)), ORDER);
    }

    // 2D point type for graph partition
    struct GP_Point { double x, y; };

    // KD-tree for k-NN queries
    struct GP_KDNode { int point_idx, axis, left, right; };
    struct GP_KDTree {
        std::vector<GP_KDNode> nodes;
        const std::vector<GP_Point> &pts;
        std::vector<int> sorted_idx;

        GP_KDTree(const std::vector<GP_Point> &points, const std::vector<int> &indices)
                : pts(points), sorted_idx(indices) {
            nodes.reserve(indices.size());
            build(0, indices.size(), 0);
        }

        int build(int start, int end, int depth) {
            if (start >= end) return -1;
            int axis = depth % 2, mid = (start + end) / 2;
            std::nth_element(sorted_idx.begin() + start, sorted_idx.begin() + mid, sorted_idx.begin() + end,
                             [&](int a, int b) { return axis == 0 ? pts[a].x < pts[b].x : pts[a].y < pts[b].y; });
            int id = nodes.size();
            nodes.push_back({sorted_idx[mid], axis, -1, -1});
            nodes[id].left = build(start, mid, depth + 1);
            nodes[id].right = build(mid + 1, end, depth + 1);
            return id;
        }

        double dist_sq(int i, int j) const {
            double dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y;
            return dx * dx + dy * dy;
        }

        void knn(int nid, int qi, int k, std::vector<std::pair<double, int>> &heap) const {
            if (nid < 0) return;
            const auto &nd = nodes[nid];
            double d = dist_sq(qi, nd.point_idx);
            if (nd.point_idx != qi) {
                if ((int) heap.size() < k) {
                    heap.push_back({d, nd.point_idx});
                    std::push_heap(heap.begin(), heap.end());
                } else if (d < heap[0].first) {
                    std::pop_heap(heap.begin(), heap.end());
                    heap.back() = {d, nd.point_idx};
                    std::push_heap(heap.begin(), heap.end());
                }
            }
            double diff = nd.axis == 0 ? pts[qi].x - pts[nd.point_idx].x : pts[qi].y - pts[nd.point_idx].y;
            int first = diff < 0 ? nd.left : nd.right, second = diff < 0 ? nd.right : nd.left;
            knn(first, qi, k, heap);
            if (diff * diff < ((int) heap.size() < k ? 1e30 : heap[0].first))
                knn(second, qi, k, heap);
        }

        std::vector<int> query(int qi, int k) const {
            std::vector<std::pair<double, int>> heap;
            heap.reserve(k + 1);
            knn(0, qi, k, heap);
            std::vector<int> r;
            for (auto &p: heap) r.push_back(p.second);
            return r;
        }
    };

    // k-NN graph
    struct GP_Graph {
        int n;
        std::vector<std::vector<int>> adj;

        void build_knn(const std::vector<GP_Point> &points, const std::vector<int> &subset, int k) {
            n = subset.size();
            adj.resize(n);
            std::unordered_map<int, int> g2l;
            for (int i = 0; i < n; i++) g2l[subset[i]] = i;
            GP_KDTree tree(points, subset);
            for (int i = 0; i < n; i++)
                for (int nb: tree.query(subset[i], k)) {
                    auto it = g2l.find(nb);
                    if (it != g2l.end()) {
                        adj[i].push_back(it->second);
                        adj[it->second].push_back(i);
                    }
                }
            for (int i = 0; i < n; i++) {
                std::sort(adj[i].begin(), adj[i].end());
                adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
            }
        }
    };

    // Matern covariance kernel (sigma^2 omitted — cancels in normalized Laplacian)
    // Handles half-integer nu exactly; falls back to exponential otherwise.
    inline double MaternWeight(double d2, double beta, double nu) {
        if (d2 < 1e-15) return 1e15;
        double d = std::sqrt(d2);
        if (nu < 0.75) {                          // nu ~ 0.5: exponential
            return std::exp(-d / beta);
        } else if (nu < 1.25) {                   // nu ~ 1.0: approx
            double r = std::sqrt(2.0) * d / beta;
            return (1.0 + r) * std::exp(-r);
        } else if (nu < 1.75) {                   // nu ~ 1.5
            double r = std::sqrt(3.0) * d / beta;
            return (1.0 + r) * std::exp(-r);
        } else if (nu < 2.25) {                   // nu ~ 2.0: approx
            double r = 2.0 * d / beta;
            return std::exp(-r);
        } else if (nu < 2.75) {                   // nu ~ 2.5
            double r = std::sqrt(5.0) * d / beta;
            return (1.0 + r + r * r / 3.0) * std::exp(-r);
        } else {                                  // large nu: Gaussian limit
            return std::exp(-d2 / (2.0 * beta * beta));
        }
    }

    // Recursive spectral bisection with tile-aligned splits and KL boundary refinement
    void RecursiveGraphBisect(const std::vector<GP_Point> &points, const std::vector<int> &subset,
                              int tile_size, int knn_k, double beta, double nu,
                              std::vector<int> &result_order) {
        int n = subset.size(), num_tiles = (n + tile_size - 1) / tile_size;
        if (num_tiles <= 1) {
            std::vector<int> local = subset;
            std::sort(local.begin(), local.end(), [&](int a, int b) {
                return HilbertCode(points[a].x, points[a].y) < HilbertCode(points[b].x, points[b].y);
            });
            result_order.insert(result_order.end(), local.begin(), local.end());
            return;
        }
        int left_tiles = num_tiles / 2, left_size = left_tiles * tile_size;

        // Build k-NN graph on current subset
        GP_Graph g;
        g.build_knn(points, subset, knn_k);

        // Fiedler vector via weighted Laplacian power iteration
        std::vector<double> fiedler(n);
        std::vector<std::vector<std::pair<int, double>>> wadj(n);
        {
            std::vector<double> degree(n, 0.0);
            for (int i = 0; i < n; i++)
                for (int j: g.adj[i]) {
                    double dx = points[subset[i]].x - points[subset[j]].x;
                    double dy = points[subset[i]].y - points[subset[j]].y;
                    double d2 = dx * dx + dy * dy, w = MaternWeight(d2, beta, nu);
                    wadj[i].push_back({j, w});
                    degree[i] += w;
                }
            double cx = 0, cy = 0;
            for (int i = 0; i < n; i++) {
                cx += points[subset[i]].x;
                cy += points[subset[i]].y;
            }
            cx /= n;
            cy /= n;
            double cxx = 0, cxy = 0, cyy = 0;
            for (int i = 0; i < n; i++) {
                double dx = points[subset[i]].x - cx, dy = points[subset[i]].y - cy;
                cxx += dx * dx;
                cxy += dx * dy;
                cyy += dy * dy;
            }
            double trace = cxx + cyy, det = cxx * cyy - cxy * cxy;
            double lambda1 = trace / 2 + std::sqrt(std::max(0.0, trace * trace / 4 - det));
            double vx = cxy, vy = lambda1 - cxx, vn = std::sqrt(vx * vx + vy * vy);
            if (vn < 1e-15) { vx = 1; vy = 0; } else { vx /= vn; vy /= vn; }
            for (int i = 0; i < n; i++)
                fiedler[i] = (points[subset[i]].x - cx) * vx + (points[subset[i]].y - cy) * vy;
            double mean = std::accumulate(fiedler.begin(), fiedler.end(), 0.0) / n;
            for (auto &x: fiedler) x -= mean;
            double norm = 0;
            for (auto x: fiedler) norm += x * x;
            norm = std::sqrt(norm);
            if (norm > 1e-15) for (auto &x: fiedler) x /= norm;

            // Power iteration: 50 iterations with shifted inverse Laplacian
            double d_max = *std::max_element(degree.begin(), degree.end()) + 1.0;
            std::vector<double> v_new(n);
            for (int iter = 0; iter < 50; iter++) {
                for (int i = 0; i < n; i++) {
                    double sum = (d_max - degree[i]) * fiedler[i];
                    for (auto &[j, w]: wadj[i]) sum += w * fiedler[j];
                    v_new[i] = sum;
                }
                mean = std::accumulate(v_new.begin(), v_new.end(), 0.0) / n;
                for (auto &x: v_new) x -= mean;
                norm = 0;
                for (auto x: v_new) norm += x * x;
                norm = std::sqrt(norm);
                if (norm < 1e-15) break;
                for (auto &x: v_new) x /= norm;
                fiedler = v_new;
            }
        }
        std::vector<int> lo(n);
        std::iota(lo.begin(), lo.end(), 0);
        std::nth_element(lo.begin(), lo.begin() + left_size, lo.end(),
                         [&](int a, int b) { return fiedler[a] < fiedler[b]; });

        // KL boundary refinement: greedy pair-swaps to reduce cross-partition cut cost.
        {
            std::vector<int> part(n);
            for (int i = 0; i < left_size; i++) part[lo[i]] = 0;
            for (int i = left_size; i < n; i++) part[lo[i]] = 1;
            std::vector<double> D(n, 0.0);
            for (int v = 0; v < n; v++)
                for (auto &[u, w]: wadj[v])
                    D[v] += (part[u] != part[v]) ? w : -w;
            for (int pass = 0; pass < 10; pass++) {
                std::vector<int> bndA, bndB;
                for (int v = 0; v < n; v++) {
                    bool cross = false;
                    for (auto &[u, w]: wadj[v]) if (part[u] != part[v]) { cross = true; break; }
                    if (!cross) continue;
                    (part[v] == 0 ? bndA : bndB).push_back(v);
                }
                double best_gain = 0.0;
                int best_a = -1, best_b = -1;
                for (int a: bndA) {
                    for (int b: bndB) {
                        double wab = 0.0;
                        for (auto &[u, w]: wadj[a]) if (u == b) { wab = w; break; }
                        double gain = D[a] + D[b] - 2.0 * wab;
                        if (gain > best_gain) { best_gain = gain; best_a = a; best_b = b; }
                    }
                }
                if (best_a < 0) break;
                part[best_a] = 1; part[best_b] = 0;
                for (int v = 0; v < n; v++) {
                    D[v] = 0.0;
                    for (auto &[u, w]: wadj[v])
                        D[v] += (part[u] != part[v]) ? w : -w;
                }
            }
            int ia = 0, ib = left_size;
            for (int v = 0; v < n; v++) lo[part[v] == 0 ? ia++ : ib++] = v;
        }
        std::vector<int> subA, subB;
        subA.reserve(left_size);
        subB.reserve(n - left_size);
        for (int i = 0; i < left_size; i++) subA.push_back(subset[lo[i]]);
        for (int i = left_size; i < n; i++) subB.push_back(subset[lo[i]]);
        auto centroid = [&](const std::vector<int> &s) -> std::pair<double, double> {
            double ccx = 0, ccy = 0;
            for (int i: s) { ccx += points[i].x; ccy += points[i].y; }
            return {ccx / s.size(), ccy / s.size()};
        };
        auto[ax, ay] = centroid(subA);
        auto[bx, by] = centroid(subB);
        if (HilbertCode(ax, ay) > HilbertCode(bx, by)) std::swap(subA, subB);
        RecursiveGraphBisect(points, subA, tile_size, knn_k, beta, nu, result_order);
        RecursiveGraphBisect(points, subB, tile_size, knn_k, beta, nu, result_order);
    }

} // anonymous namespace

using namespace exageostat::generators;
using namespace exageostat::common;
using namespace exageostat::dataunits;
using namespace exageostat::helpers;

template<typename T>
void LocationGenerator<T>::GenerateLocations(const int &aN, const int &aTimeSlot, const Dimension &aDimension,
                                             Locations<T> &aLocations,
                                             configurations::Configurations &aConfigurations) {

    aLocations.SetSize(aN);
    int index = 0;
    aLocations.SetDimension(aDimension);

    int rootN;
    if (aDimension == Dimension3D) {
        //Cubic root.
        rootN = ceil(cbrt(aN));
    } else {
        //Square root.
        rootN = ceil(sqrt(aN));
    }

    int *grid = new int[rootN]();
    for (auto i = 0; i < rootN; i++) {
        grid[i] = i + 1;
    }
    T range_low = -0.4, range_high = 0.4;

    for (auto i = 0; i < rootN && index < aN; i++) {
        for (auto j = 0; j < rootN && index < aN; j++) {
            if (aDimension == Dimension3D) {
                for (auto k = 0; k < rootN && index < aN; k++) {
                    aLocations.GetLocationX()[index] =
                            (grid[i] - 0.5 + UniformDistribution(range_low, range_high)) / rootN;
                    aLocations.GetLocationY()[index] =
                            (grid[j] - 0.5 + UniformDistribution(range_low, range_high)) / rootN;
                    aLocations.GetLocationZ()[index] =
                            (grid[k] - 0.5 + UniformDistribution(range_low, range_high)) / rootN;
                    index++;
                }
            } else {
                aLocations.GetLocationX()[index] =
                        (grid[i] - 0.5 + UniformDistribution(range_low, range_high)) / rootN;
                aLocations.GetLocationY()[index] =
                        (grid[j] - 0.5 + UniformDistribution(range_low, range_high)) / rootN;
                if (aDimension == DimensionST) {
                    aLocations.GetLocationZ()[index] = 1.0;
                }
                index++;
            }
        }
    }
    delete[] grid;
    if (aDimension != DimensionST) {
        // Apply optional clustering-aware ordering; falls back to Morton-only.
        ApplyOrdering(aN, aDimension, aLocations, aConfigurations);
    } else {
        for (auto i = 0; i < aN; i++) {
            aLocations.GetLocationX()[i] = aLocations.GetLocationX()[i];
            aLocations.GetLocationY()[i] = aLocations.GetLocationY()[i];
            aLocations.GetLocationZ()[i] = (T) (i / aTimeSlot + 1);
        }
    }
}

template<typename T>
void LocationGenerator<T>::GenerateLocations(const int &aN, const int &aTimeSlot, const Dimension &aDimension,
                                             Locations<T> &aLocations) {

    aLocations.SetSize(aN);
    int index = 0;
    aLocations.SetDimension(aDimension);

    int rootN;
    if (aDimension == Dimension3D) {
        rootN = ceil(cbrt(aN));
    } else {
        rootN = ceil(sqrt(aN));
    }

    auto *grid = new int[rootN]();
    for (auto i = 0; i < rootN; i++) {
        grid[i] = i + 1;
    }

    if (aDimension == Dimension3D) {
        for (auto i = 0; i < rootN && index < aN; i++) {
            for (auto j = 0; j < rootN && index < aN; j++) {
                for (auto k = 0; k < rootN && index < aN; k++) {
                    aLocations.GetLocationX()[index] = (T) grid[i] / rootN + UniformDistribution(-0.4 / rootN, 0.4 / rootN);
                    aLocations.GetLocationY()[index] = (T) grid[j] / rootN + UniformDistribution(-0.4 / rootN, 0.4 / rootN);
                    aLocations.GetLocationZ()[index] = (T) grid[k] / rootN + UniformDistribution(-0.4 / rootN, 0.4 / rootN);
                    index++;
                }
            }
        }
    } else {
        for (auto i = 0; i < rootN && index < aN; i++) {
            for (auto j = 0; j < rootN && index < aN; j++) {
                aLocations.GetLocationX()[index] = (T) grid[i] / rootN + UniformDistribution(-0.4 / rootN, 0.4 / rootN);
                aLocations.GetLocationY()[index] = (T) grid[j] / rootN + UniformDistribution(-0.4 / rootN, 0.4 / rootN);
                index++;
            }
        }
    }
    delete[] grid;
    if (aDimension != DimensionST) {
        SortLocations(aN, aDimension, aLocations);
    }
}

template<typename T>
T LocationGenerator<T>::UniformDistribution(const T &aRangeLow, const T &aRangeHigh) {
    T myRand = (T) rand() / (T) (1.0 + RAND_MAX);
    T range = aRangeHigh - aRangeLow;
    return (myRand * range) + aRangeLow;
}

template<typename T>
void LocationGenerator<T>::ApplyOrdering(const int &aN, const Dimension &aDimension, Locations<T> &aLocations,
                                         configurations::Configurations &aConfigurations) {

    std::string mode;
    try {
        mode = aConfigurations.GetClusteringMode();
    } catch (...) {
        mode = "none";
    }

    std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);

    // Graph partition: separate path — produces a direct permutation, not cluster IDs
    if (mode == "graph_part") {
        int tile_size = aConfigurations.GetLowTileSize();
        if (tile_size <= 0) {
            tile_size = aConfigurations.GetDenseTileSize();
        }
        if (tile_size <= 0) {
            tile_size = aN;
        }
        int knn_k = 12;

        // Extract beta and nu from initial_theta for covariance-aware edge weights.
        // initial_theta = [sigma2, beta, nu] for UnivariateMaternStationary.
        // Fall back to geometric weights (1/d^2 equivalent: beta=1, nu>>1) if unavailable.
        double beta = 1.0, nu = 100.0;
        const auto &theta = aConfigurations.GetInitialTheta();
        if (theta.size() >= 3 && theta[1] > 0.0 && theta[2] > 0.0) {
            beta = theta[1];
            nu   = theta[2];
        }

        // Build GP_Point array from current locations
        std::vector<GP_Point> gp_points(aN);
        for (int i = 0; i < aN; i++) {
            gp_points[i].x = static_cast<double>(aLocations.GetLocationX()[i]);
            gp_points[i].y = static_cast<double>(aLocations.GetLocationY()[i]);
        }

        // Run recursive spectral bisection with covariance-aware Matern edge weights
        std::vector<int> all_indices(aN);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        std::vector<int> result_order;
        result_order.reserve(aN);
        RecursiveGraphBisect(gp_points, all_indices, tile_size, knn_k, beta, nu, result_order);
        // Apply the permutation: copy reordered coordinates back
        std::vector<T> tmp_x(aN), tmp_y(aN), tmp_z(aN);
        for (int i = 0; i < aN; i++) {
            tmp_x[i] = aLocations.GetLocationX()[result_order[i]];
            tmp_y[i] = aLocations.GetLocationY()[result_order[i]];
            if (aDimension == Dimension3D) {
                tmp_z[i] = aLocations.GetLocationZ()[result_order[i]];
            }
        }
        for (int i = 0; i < aN; i++) {
            aLocations.GetLocationX()[i] = tmp_x[i];
            aLocations.GetLocationY()[i] = tmp_y[i];
            if (aDimension == Dimension3D) {
                aLocations.GetLocationZ()[i] = tmp_z[i];
            }
        }

        // Tile compactness diagnostics
        {
            int num_tiles = (aN + tile_size - 1) / tile_size;
            double sum_ratio = 0, max_ratio = 0, min_ratio = 1e9;
            double sum_diam = 0;
            double ideal_area = 1.0 / num_tiles; // assumes [0,1]^2 domain
            double ideal_diam = 2.0 * std::sqrt(ideal_area / M_PI); // circle diameter
            for (int t = 0; t < num_tiles; t++) {
                int lo = t * tile_size;
                int hi = std::min(lo + tile_size, aN);
                double max_d2 = 0;
                // Sample-based diameter: check all pairs (exact for tile_size ≤ 2000)
                for (int i = lo; i < hi; i++) {
                    for (int j = i + 1; j < hi; j++) {
                        double dx = tmp_x[i] - tmp_x[j];
                        double dy = tmp_y[i] - tmp_y[j];
                        double d2 = dx * dx + dy * dy;
                        if (d2 > max_d2) max_d2 = d2;
                    }
                }
                double diam = std::sqrt(max_d2);
                double ratio = diam / ideal_diam;
                sum_diam += diam;
                sum_ratio += ratio;
                if (ratio > max_ratio) max_ratio = ratio;
                if (ratio < min_ratio) min_ratio = ratio;
            }
            // Per-tile diameter list
            fprintf(stderr, "TILE_DIAM_LIST");
            for (int t = 0; t < num_tiles; t++) {
                int tlo = t * tile_size;
                int thi = std::min(tlo + tile_size, aN);
                double td2 = 0;
                for (int i = tlo; i < thi; i++)
                    for (int j = i + 1; j < thi; j++) {
                        double dx2 = tmp_x[i] - tmp_x[j], dy2 = tmp_y[i] - tmp_y[j];
                        double d2t = dx2*dx2 + dy2*dy2;
                        if (d2t > td2) td2 = d2t;
                    }
                fprintf(stderr, " %.4f", std::sqrt(td2));
            }
            fprintf(stderr, "\n");

            // Inter-tile centroid distance for near-diagonal pairs
            std::vector<std::array<double,2>> tile_centroids(num_tiles);
            for (int t = 0; t < num_tiles; t++) {
                int tlo = t * tile_size;
                int thi = std::min(tlo + tile_size, aN);
                double sx=0, sy=0;
                for (int i = tlo; i < thi; i++) { sx += tmp_x[i]; sy += tmp_y[i]; }
                tile_centroids[t] = {sx/(thi-tlo), sy/(thi-tlo)};
            }
            fprintf(stderr, "TILE_CENT_DIST_D1");
            for (int t = 0; t + 1 < num_tiles; t++) {
                double dx = tile_centroids[t][0]-tile_centroids[t+1][0];
                double dy = tile_centroids[t][1]-tile_centroids[t+1][1];
                fprintf(stderr, " %.4f", std::sqrt(dx*dx+dy*dy));
            }
            fprintf(stderr, "\n");

            fprintf(stderr, "TILE_DIAG [graph_part] tiles=%d tile_size=%d ideal_diam=%.4f\n",
                    num_tiles, tile_size, ideal_diam);
            fprintf(stderr, "TILE_DIAG avg_diam=%.4f avg_ratio=%.2f min_ratio=%.2f max_ratio=%.2f\n",
                    sum_diam / num_tiles, sum_ratio / num_tiles, min_ratio, max_ratio);
        }

        return;
    }

    SortLocations(aN, aDimension, aLocations);
}

template<typename T>
void
LocationGenerator<T>::SortLocations(const int &aN, const Dimension &aDimension, Locations<T> &aLocations) {

    // Sort by Morton code but preserve original coordinate precision.
    // The old implementation decoded coordinates back from uint16_t Morton codes,
    // which lost precision and could create duplicate coordinates when two
    // distinct points quantized to the same (uint16_x, uint16_y).
    uint16_t x, y, z;
    std::vector<uint64_t> morton_codes(aN);
    std::vector<int> order(aN);

    // Encode Morton codes
    for (auto i = 0; i < aN; i++) {
        x = (uint16_t) (aLocations.GetLocationX()[i] * (double) UINT16_MAX + .5);
        y = (uint16_t) (aLocations.GetLocationY()[i] * (double) UINT16_MAX + .5);
        if (aDimension != Dimension2D) {
            z = (uint16_t) (aLocations.GetLocationZ()[i] * (double) UINT16_MAX + .5);
        } else {
            z = (uint16_t) 0.0;
        }
        morton_codes[i] = (SpreadBits(z) << 2) + (SpreadBits(y) << 1) + SpreadBits(x);
        order[i] = i;
    }

    // Sort indices by Morton code
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return morton_codes[a] < morton_codes[b];
    });

    // Permute original coordinates (preserving double precision)
    std::vector<T> tmp_x(aN), tmp_y(aN), tmp_z(aN);
    for (auto i = 0; i < aN; i++) {
        tmp_x[i] = aLocations.GetLocationX()[order[i]];
        tmp_y[i] = aLocations.GetLocationY()[order[i]];
        if (aDimension == Dimension3D) {
            tmp_z[i] = aLocations.GetLocationZ()[order[i]];
        }
    }
    for (auto i = 0; i < aN; i++) {
        aLocations.GetLocationX()[i] = tmp_x[i];
        aLocations.GetLocationY()[i] = tmp_y[i];
        if (aDimension == Dimension3D) {
            aLocations.GetLocationZ()[i] = tmp_z[i];
        }
    }
}
