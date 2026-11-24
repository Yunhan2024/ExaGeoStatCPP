
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

#include <cmath>
#include <algorithm>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <data-generators/LocationGenerator.hpp>
#include <helpers/ByteHandler.hpp>

using namespace exageostat::generators;
using namespace exageostat::common;
using namespace exageostat::dataunits;
using namespace exageostat::helpers;

template<typename T>
void LocationGenerator<T>::GenerateLocations(const int &aN, const int &aTimeSlot, const Dimension &aDimension,
                                             Locations<T> &aLocations, const bool &aSort) {

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
        if (aSort) {
            SortLocations(aN, aDimension, aLocations);
        }
    } else {
        for (auto i = 0; i < aN; i++) {
            aLocations.GetLocationX()[i] = aLocations.GetLocationX()[i];
            aLocations.GetLocationY()[i] = aLocations.GetLocationY()[i];
            aLocations.GetLocationZ()[i] = (T) (i / aTimeSlot + 1);
        }
    }
}

template<typename T>
T LocationGenerator<T>::UniformDistribution(const T &aRangeLow, const T &aRangeHigh) {
    T myRand = (T) rand() / (T) (1.0 + RAND_MAX);
    T range = aRangeHigh - aRangeLow;
    return (myRand * range) + aRangeLow;
}

template<typename T>
void
LocationGenerator<T>::SortLocations(const int &aN, const Dimension &aDimension, Locations<T> &aLocations) {

    // Some sorting, required by spatial statistics code
    uint16_t x, y, z;
    uint64_t vectorZ[aN];

    // Encode data into vector z
    for (auto i = 0; i < aN; i++) {
        x = (uint16_t) (aLocations.GetLocationX()[i] * (double) UINT16_MAX + .5);
        y = (uint16_t) (aLocations.GetLocationY()[i] * (double) UINT16_MAX + .5);
        if (aDimension != Dimension2D) {
            z = (uint16_t) (aLocations.GetLocationZ()[i] * (double) UINT16_MAX + .5);
        } else {
            z = (uint16_t) 0.0;
        }
        vectorZ[i] = (SpreadBits(z) << 2) + (SpreadBits(y) << 1) + SpreadBits(x);
    }
    // Sort vector z
    std::sort(vectorZ, vectorZ + aN, CompareUint64);

    // Decode data from vector z
    for (auto i = 0; i < aN; i++) {
        x = ReverseSpreadBits(vectorZ[i] >> 0);
        y = ReverseSpreadBits(vectorZ[i] >> 1);
        z = ReverseSpreadBits(vectorZ[i] >> 2);
        aLocations.GetLocationX()[i] = (double) x / (double) UINT16_MAX;
        aLocations.GetLocationY()[i] = (double) y / (double) UINT16_MAX;
        if (aDimension == Dimension3D) {
            aLocations.GetLocationZ()[i] = (double) z / (double) UINT16_MAX;
        }
    }
}

namespace {

    // Hilbert index of a point of the unit square on a 2^16 x 2^16 grid.
    uint64_t HilbertIndex2D(double aX, double aY) {
        const uint32_t order = 1u << 16;
        auto x = static_cast<uint32_t>(std::clamp(aX, 0.0, 1.0) * (order - 1));
        auto y = static_cast<uint32_t>(std::clamp(aY, 0.0, 1.0) * (order - 1));
        uint64_t index = 0;
        for (uint32_t s = order / 2; s > 0; s /= 2) {
            uint32_t rx = (x & s) > 0 ? 1 : 0;
            uint32_t ry = (y & s) > 0 ? 1 : 0;
            index += static_cast<uint64_t>(s) * s * ((3 * rx) ^ ry);
            if (ry == 0) {
                if (rx == 1) {
                    x = s - 1 - x;
                    y = s - 1 - y;
                }
                std::swap(x, y);
            }
        }
        return index;
    }

    // Morton index of a point of the unit cube, encoded like LocationGenerator::SortLocations.
    uint64_t MortonIndex3D(double aX, double aY, double aZ) {
        auto encode = [](double v) {
            return static_cast<uint16_t>(std::clamp(v, 0.0, 1.0) * (double) UINT16_MAX + .5);
        };
        return (exageostat::helpers::SpreadBits(encode(aZ)) << 2) +
               (exageostat::helpers::SpreadBits(encode(aY)) << 1) + exageostat::helpers::SpreadBits(encode(aX));
    }

    struct MinBBoxPoints {
        int dims;                         // 2 or 3
        std::vector<double> coords[3];    // raw coordinates, used for the geometry
        std::vector<uint64_t> curve;      // space-filling-curve index of every point
        double origin[3];
        double scale;                     // common scale mapping the bounding cube to [0, 1]

        uint64_t CurveIndex(const double *apPoint) const {
            double unit[3] = {0, 0, 0};
            for (int d = 0; d < dims; d++) {
                unit[d] = (apPoint[d] - origin[d]) / scale;
            }
            return dims == 2 ? HilbertIndex2D(unit[0], unit[1]) : MortonIndex3D(unit[0], unit[1], unit[2]);
        }
    };

    void MinBBoxBisect(const MinBBoxPoints &aPoints, const int aTileSize, std::vector<int> aSubset,
                       std::vector<int> &aOrder) {

        const int n = static_cast<int>(aSubset.size());
        const int tiles = (n + aTileSize - 1) / aTileSize;
        if (tiles <= 1) {
            std::stable_sort(aSubset.begin(), aSubset.end(),
                             [&](int a, int b) { return aPoints.curve[a] < aPoints.curve[b]; });
            aOrder.insert(aOrder.end(), aSubset.begin(), aSubset.end());
            return;
        }

        // Cut across the longer side of the bounding box.
        int axis = 0;
        double longest = -1;
        for (int d = 0; d < aPoints.dims; d++) {
            const auto &c = aPoints.coords[d];
            auto [lo, hi] = std::minmax_element(aSubset.begin(), aSubset.end(),
                                                [&](int a, int b) { return c[a] < c[b]; });
            if (c[*hi] - c[*lo] > longest) {
                longest = c[*hi] - c[*lo];
                axis = d;
            }
        }
        const auto &key = aPoints.coords[axis];
        std::stable_sort(aSubset.begin(), aSubset.end(), [&](int a, int b) { return key[a] < key[b]; });

        // Prefix and suffix bounding boxes give the diagonal of each candidate half in O(1).
        const int dims = aPoints.dims;
        std::vector<double> prefix_lo(n * dims), prefix_hi(n * dims), suffix_lo(n * dims), suffix_hi(n * dims);
        for (int d = 0; d < dims; d++) {
            const auto &c = aPoints.coords[d];
            for (int i = 0; i < n; i++) {
                double v = c[aSubset[i]];
                prefix_lo[i * dims + d] = i ? std::min(prefix_lo[(i - 1) * dims + d], v) : v;
                prefix_hi[i * dims + d] = i ? std::max(prefix_hi[(i - 1) * dims + d], v) : v;
            }
            for (int i = n - 1; i >= 0; i--) {
                double v = c[aSubset[i]];
                suffix_lo[i * dims + d] = i < n - 1 ? std::min(suffix_lo[(i + 1) * dims + d], v) : v;
                suffix_hi[i * dims + d] = i < n - 1 ? std::max(suffix_hi[(i + 1) * dims + d], v) : v;
            }
        }
        auto diagonal = [&](const std::vector<double> &lo, const std::vector<double> &hi, int i) {
            double sum = 0;
            for (int d = 0; d < dims; d++) {
                double extent = hi[i * dims + d] - lo[i * dims + d];
                sum += extent * extent;
            }
            return std::sqrt(sum);
        };

        // Tile-aligned cut near the middle that makes both halves most compact.
        const int middle = tiles / 2;
        int best_tiles = std::min(std::max(middle, 1), tiles - 1);
        double best = std::numeric_limits<double>::max();
        for (int shift = -2; shift <= 2; shift++) {
            int left_tiles = middle + shift;
            if (left_tiles < 1 || left_tiles >= tiles || left_tiles * aTileSize >= n) {
                continue;
            }
            int cut = left_tiles * aTileSize;
            double cost = diagonal(prefix_lo, prefix_hi, cut - 1) + diagonal(suffix_lo, suffix_hi, cut);
            if (cost < best) {
                best = cost;
                best_tiles = left_tiles;
            }
        }
        const int cut = best_tiles * aTileSize;
        std::vector<int> first(aSubset.begin(), aSubset.begin() + cut);
        std::vector<int> second(aSubset.begin() + cut, aSubset.end());

        // Visit the halves in space-filling-curve order of their centroids.
        auto centroid_index = [&](const std::vector<int> &part) {
            double centroid[3] = {0, 0, 0};
            for (int idx: part) {
                for (int d = 0; d < dims; d++) {
                    centroid[d] += aPoints.coords[d][idx];
                }
            }
            for (int d = 0; d < dims; d++) {
                centroid[d] /= static_cast<double>(part.size());
            }
            return aPoints.CurveIndex(centroid);
        };
        if (centroid_index(first) > centroid_index(second)) {
            std::swap(first, second);
        }
        MinBBoxBisect(aPoints, aTileSize, std::move(first), aOrder);
        MinBBoxBisect(aPoints, aTileSize, std::move(second), aOrder);
    }
}

namespace {

    // Copies the coordinates and computes the common scale mapping them to the unit square/cube.
    template<typename T>
    MinBBoxPoints ReadPoints(const int aN, const Dimension aDimension, Locations<T> &aLocations) {
        MinBBoxPoints points;
        points.dims = aDimension == Dimension3D ? 3 : 2;
        T *axes[3] = {aLocations.GetLocationX(), aLocations.GetLocationY(), aLocations.GetLocationZ()};
        points.scale = 0;
        for (int d = 0; d < points.dims; d++) {
            points.coords[d].assign(axes[d], axes[d] + aN);
            auto [lo, hi] = std::minmax_element(points.coords[d].begin(), points.coords[d].end());
            points.origin[d] = *lo;
            points.scale = std::max(points.scale, *hi - *lo);
        }
        if (points.scale <= 0) {
            points.scale = 1;
        }
        return points;
    }

    template<typename T>
    void WritePoints(const MinBBoxPoints &aPoints, const std::vector<int> &aOrder, Locations<T> &aLocations) {
        T *axes[3] = {aLocations.GetLocationX(), aLocations.GetLocationY(), aLocations.GetLocationZ()};
        for (int d = 0; d < aPoints.dims; d++) {
            for (size_t i = 0; i < aOrder.size(); i++) {
                axes[d][i] = static_cast<T>(aPoints.coords[d][aOrder[i]]);
            }
        }
    }
}

template<typename T>
std::vector<int>
LocationGenerator<T>::ApplyMinBBoxOrdering(const int &aN, const Dimension &aDimension, Locations<T> &aLocations,
                                           const int &aTileSize) {

    std::vector<int> order(aN);
    std::iota(order.begin(), order.end(), 0);
    if (aDimension == DimensionST || aTileSize <= 0 || aN <= 1) {
        return order;
    }

    MinBBoxPoints points = ReadPoints(aN, aDimension, aLocations);
    points.curve.resize(aN);
    for (int i = 0; i < aN; i++) {
        double point[3] = {0, 0, 0};
        for (int d = 0; d < points.dims; d++) {
            point[d] = points.coords[d][i];
        }
        points.curve[i] = points.CurveIndex(point);
    }

    std::vector<int> result;
    result.reserve(aN);
    MinBBoxBisect(points, aTileSize, std::move(order), result);
    WritePoints(points, result, aLocations);
    return result;
}

template<typename T>
std::vector<int>
LocationGenerator<T>::ApplyMortonOrdering(const int &aN, const Dimension &aDimension, Locations<T> &aLocations) {

    std::vector<int> order(aN);
    std::iota(order.begin(), order.end(), 0);
    if (aDimension == DimensionST || aN <= 1) {
        return order;
    }

    MinBBoxPoints points = ReadPoints(aN, aDimension, aLocations);
    std::vector<uint64_t> codes(aN);
    for (int i = 0; i < aN; i++) {
        double unit[3] = {0, 0, 0};
        for (int d = 0; d < points.dims; d++) {
            unit[d] = (points.coords[d][i] - points.origin[d]) / points.scale;
        }
        codes[i] = MortonIndex3D(unit[0], unit[1], unit[2]);
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return codes[a] < codes[b]; });
    WritePoints(points, order, aLocations);
    return order;
}

template<typename T>
std::vector<int>
LocationGenerator<T>::ApplyFileOrdering(const int &aN, const Dimension &aDimension, Locations<T> &aLocations,
                                        const std::string &aPath) {

    std::vector<int> order(aN);
    std::iota(order.begin(), order.end(), 0);
    if (aDimension == DimensionST) {
        return order;
    }

    std::ifstream file(aPath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open ordering file " + aPath);
    }
    const int dims = aDimension == Dimension3D ? 3 : 2;
    T *axes[3] = {aLocations.GetLocationX(), aLocations.GetLocationY(), aLocations.GetLocationZ()};
    std::vector<long long> values;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream fields(line);
        std::string head;
        if (!(fields >> head)) {
            continue;
        }
        if (head[0] != '#') {
            values.push_back(std::stoll(head));
            continue;
        }
        std::string key;
        fields >> key;
        if (key == "n") {
            long long n;
            if (fields >> n && n != aN) {
                throw std::runtime_error("Ordering file " + aPath + " is for " + std::to_string(n) +
                                         " locations, but there are " + std::to_string(aN));
            }
        } else if (key == "coordsum" || key == "coordwsum") {
            // The permutation refers to the locations in generation (or file) order. coordsum rejects another
            // point set (seed, size); coordwsum weights row i by (i + 1) / n and also rejects the same points in
            // another row order (e.g. Morton-sorted).
            const bool weighted = key == "coordwsum";
            for (int d = 0; d < dims; d++) {
                double expected;
                if (!(fields >> expected)) {
                    break;
                }
                double sum = 0;
                for (int i = 0; i < aN; i++) {
                    sum += (weighted ? (i + 1.0) / aN : 1.0) * static_cast<double>(axes[d][i]);
                }
                const double tol = sizeof(T) == sizeof(float) ? 1e-5 : 1e-9;
                if (std::abs(sum - expected) > tol * std::max(1.0, std::abs(expected))) {
                    throw std::runtime_error("Ordering file " + aPath + (weighted
                                             ? " was made for these locations in a different row order"
                                             : " was made for a different point set (coordinate sums differ)"));
                }
            }
        }
    }
    if (static_cast<long long>(values.size()) != aN) {
        throw std::runtime_error("Ordering file " + aPath + " has " + std::to_string(values.size()) +
                                 " indices, expected " + std::to_string(aN));
    }
    std::vector<char> seen(aN, 0);
    for (int i = 0; i < aN; i++) {
        if (values[i] < 0 || values[i] >= aN || seen[values[i]]) {
            throw std::runtime_error("Ordering file " + aPath + " is not a permutation of 0.." + std::to_string(aN - 1));
        }
        seen[values[i]] = 1;
        order[i] = static_cast<int>(values[i]);
    }

    std::vector<T> copy(aN);
    for (int d = 0; d < dims; d++) {
        for (int i = 0; i < aN; i++) {
            copy[i] = axes[d][order[i]];
        }
        std::copy(copy.begin(), copy.end(), axes[d]);
    }
    return order;
}