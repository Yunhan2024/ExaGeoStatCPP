
# Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
# All rights reserved.
# ExaGeoStat is a software package, provided by King Abdullah University of Science and Technology (KAUST).

# @file CMakeLists.txt
# @version 1.1.0
# @author Mahmoud ElKarargy
# @date 2023-03-08

set(EXAGEOSTAT_TESTFILES

        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/TestSyntheticGenerator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/TestCSVDataGenerator.cpp

        ${EXAGEOSTAT_TESTFILES}
        )

# Tile rank ordering test requires HiCMA for TLR compression
if (USE_HICMA)
    list(APPEND EXAGEOSTAT_TESTFILES
            ${CMAKE_CURRENT_SOURCE_DIR}/concrete/TestTileRankOrdering.cpp
            )
endif ()

set(EXAGEOSTAT_TESTFILES ${EXAGEOSTAT_TESTFILES} PARENT_SCOPE)
