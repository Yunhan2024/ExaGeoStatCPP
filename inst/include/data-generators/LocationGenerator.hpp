
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// ExaGeoStat is a software package, provided by King Abdullah University of Science and Technology (KAUST).

/**
 * @file LocationGenerator.hpp
 * @brief Generates and manages spatial locations for ExaGeoStat.
 * @version 1.1.0
 * @author Mahmoud ElKarargy
 * @date 2024-02-04
**/

#ifndef EXAGEOSTATCPP_LOCATIONGENERATOR_HPP
#define EXAGEOSTATCPP_LOCATIONGENERATOR_HPP

#include <string>
#include <vector>

#include <data-units/Locations.hpp>

namespace exageostat::generators {

    /**
     * @class LocationGenerator
     * @brief Generates spatial locations based on given parameters.
     * @tparam T Data Type: float or double
     *
     */
    template<typename T>
    class LocationGenerator {

    public:

        /**
         * @brief Generates the data locations.
         * @details This method generates the X, Y, and Z variables used to define the locations of the data points.
         * @param[in] aN The number of data points.
         * @param[in] aTimeSlot The time slot.
         * @param[in] aDimension The dimension of the locations.
         * @param[out] aLocations Reference to the Locations object where the generated data will be stored.
         * @param[in] aSort Sort 2D/3D locations along the Morton curve (default). When false, the locations keep
         * their generation order and full precision, which is the row order that --ordering=file refers to.
         * @return void
         *
         */
        static void GenerateLocations(const int &aN, const int &aTimeSlot, const common::Dimension &aDimension,
                                      dataunits::Locations<T> &aLocations, const bool &aSort = true);

        /**
         * @brief Generate uniform distribution between rangeLow , rangeHigh.
         * @param[in] aRangeLow The Lower range.
         * @param[in] aRangeHigh The Higher range.
         * @return The scaled uniform distribution between the two bounds.
         *
         */
        static T UniformDistribution(const T &aRangeLow, const T &aRangeHigh);

        /**
         * @brief Sort locations in Morton order (input points must be in [0;1]x[0;1] square]).
         * @param[in] aN The problem size divided by P-Grid.
         * @param[in] aDimension Dimension of locations.
         * @param[in,out] aLocations Locations to be sorted.
         * @return void
         *
         */
        static void
        SortLocations(const int &aN, const common::Dimension &aDimension, dataunits::Locations<T> &aLocations);

        /**
         * @brief Reorders locations with min-bounding-box recursive bisection (min-bbox).
         * @details The locations are split recursively into tile-aligned halves. Each split is taken across the
         * longer side of the bounding box, at the tile-aligned position near the middle that minimizes the sum of
         * the bounding-box diagonals of the two halves. Halves are visited in space-filling-curve order of their
         * centroids, and the points inside a tile follow the same curve. Compact, well-separated tiles reduce
         * the ranks of off-diagonal tiles in tile low-rank computations.
         * @param[in] aN The number of locations.
         * @param[in] aDimension Dimension of locations (2D or 3D; space-time locations are not reordered).
         * @param[in,out] aLocations Locations to be reordered in place.
         * @param[in] aTileSize Number of locations per tile.
         * @return The permutation applied: new position i holds the location previously at position result[i].
         *
         */
        static std::vector<int>
        ApplyMinBBoxOrdering(const int &aN, const common::Dimension &aDimension, dataunits::Locations<T> &aLocations,
                             const int &aTileSize);

        /**
         * @brief Reorders locations along the Morton (Z-order) curve without altering coordinate values.
         * @details Unlike SortLocations, which re-encodes coordinates on a 16-bit grid, this keeps the original
         * values, which is required for loaded data whose measurements must follow their locations.
         * @param[in] aN The number of locations.
         * @param[in] aDimension Dimension of locations (2D or 3D; space-time locations are not reordered).
         * @param[in,out] aLocations Locations to be reordered in place.
         * @return The permutation applied: new position i holds the location previously at position result[i].
         *
         */
        static std::vector<int>
        ApplyMortonOrdering(const int &aN, const common::Dimension &aDimension, dataunits::Locations<T> &aLocations);

        /**
         * @brief Reorders locations with a permutation read from a file (e.g. QSR or ShortSeam from tools/ordering).
         * @details One zero-based location index per line; new position i holds the location previously at
         * position result[i]. Lines starting with '#' are comments; "# n <N>", "# coordsum <sx> <sy> [<sz>]" and
         * "# coordwsum ..." (sums weighted by (i + 1) / n) headers, when present, are checked against the
         * locations, so that a permutation made for another point set or another row order is rejected.
         * @param[in] aN The number of locations.
         * @param[in] aDimension Dimension of locations (2D or 3D; space-time locations are not reordered).
         * @param[in,out] aLocations Locations to be reordered in place.
         * @param[in] aPath Path of the permutation file.
         * @return The permutation applied: new position i holds the location previously at position result[i].
         *
         */
        static std::vector<int>
        ApplyFileOrdering(const int &aN, const common::Dimension &aDimension, dataunits::Locations<T> &aLocations,
                          const std::string &aPath);
    };

    /**
     * @brief Instantiates the Data Generator class for float and double types.
     * @tparam T Data Type: float or double
     *
     */
    EXAGEOSTAT_INSTANTIATE_CLASS(LocationGenerator)
}//namespace exageostat

#endif //EXAGEOSTATCPP_LOCATIONGENERATOR_HPP
