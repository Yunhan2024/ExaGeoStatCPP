
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// ExaGeoStat is a software package, provided by King Abdullah University of Science and Technology (KAUST).

/**
 * @file StarpuGenerator.cpp
 * @brief Implementation of the StarpuGenerator class
 * @version 1.1.0
 * @author Mahmoud ElKarargy
 * @author Sameh Abdulah
 * @date 2025-02-17
**/

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <data-generators/concrete/runtime/StarpuGenerator.hpp>
#include <data-generators/LocationGenerator.hpp>
#include <data-loader/concrete/CSVLoader.hpp>

using namespace exageostat::common;
using namespace exageostat::results;
using namespace exageostat::configurations;
using namespace exageostat::generators::synthetic::starpu;

namespace {
    template<typename T>
    void LoadLocationsFromCSV(const std::string &aFilePath, int aN, Dimension aDimension,
                              exageostat::dataunits::Locations<T> &aLocations) {
        std::ifstream file(aFilePath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open locations file: " + aFilePath);
        }
        std::string line;
        int count = 0;
        while (std::getline(file, line) && count < aN) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::string token;
            // Read x
            if (!std::getline(ss, token, ',')) continue;
            T x = static_cast<T>(std::stod(token));
            // Read y
            if (!std::getline(ss, token, ',')) continue;
            T y = static_cast<T>(std::stod(token));

            aLocations.GetLocationX()[count] = x;
            aLocations.GetLocationY()[count] = y;
            if (aDimension == Dimension3D) {
                // Read z if available
                if (std::getline(ss, token, ',')) {
                    aLocations.GetLocationZ()[count] = static_cast<T>(std::stod(token));
                } else {
                    aLocations.GetLocationZ()[count] = 0;
                }
            }
            count++;
        }
        if (count < aN) {
            throw std::runtime_error("Locations file has only " + std::to_string(count) +
                                     " rows, but N=" + std::to_string(aN) + " requested.");
        }
    }
} // anonymous namespace

template<typename T>
StarpuGenerator<T> *StarpuGenerator<T>::GetInstance() {

    if (mpInstance == nullptr) {
        mpInstance = new StarpuGenerator<T>();
    }
    return mpInstance;
}

template<typename T>
std::unique_ptr<ExaGeoStatData<T>>
StarpuGenerator<T>::CreateSyntheticData(Configurations &aConfigurations, exageostat::kernels::Kernel<T> &aKernel) {

    // Initialize parameters
    int n = aConfigurations.GetProblemSize() * aConfigurations.GetTimeSlot();
    auto data = std::make_unique<ExaGeoStatData<T>>(n, aConfigurations.GetDimension());

    // Allocated new Locations object.
    auto *locations = new dataunits::Locations<T>(n, aConfigurations.GetDimension());
    int parameters_number = aKernel.GetParametersNumbers();

    // Set initial theta values.
    Configurations::InitTheta(aConfigurations.GetInitialTheta(), parameters_number);
    aConfigurations.SetInitialTheta(aConfigurations.GetInitialTheta());

    // Generate Locations phase
    std::string locations_file;
    try { locations_file = aConfigurations.GetLocationsFile(); } catch (...) {}
    if (!locations_file.empty()) {
        // Load external coordinates from CSV, then apply ordering
        LoadLocationsFromCSV(locations_file, n, aConfigurations.GetDimension(), *locations);
        if (aConfigurations.GetDimension() != DimensionST) {
            LocationGenerator<T>::ApplyOrdering(n, aConfigurations.GetDimension(), *locations, aConfigurations);
        }
    } else {
        LocationGenerator<T>::GenerateLocations(n, aConfigurations.GetTimeSlot(), aConfigurations.GetDimension(),
                                                *locations, aConfigurations);
    }
    data->SetLocations(*locations);

    // Generate Descriptors phase
    auto linear_algebra_solver = linearAlgebra::LinearAlgebraFactory<T>::CreateLinearAlgebraSolver(EXACT_DENSE);
    linear_algebra_solver->GenerateSyntheticData(aConfigurations, data, aKernel);

    if (aConfigurations.GetLogger()) {
        VERBOSE("Writing generated data to the disk (Synthetic Dataset Generation Phase) .....")
#ifdef USE_MPI
        auto pMatrix = new T[aConfigurations.GetProblemSize()];
        std::string path = aConfigurations.GetLoggerPath();

        CHAMELEON_Desc2Lap(ChamUpperLower, data->GetDescriptorData()->GetDescriptor(CHAMELEON_DESCRIPTOR,
                                                                                    DESCRIPTOR_Z).chameleon_desc,
                           pMatrix, aConfigurations.GetProblemSize());
        if (helpers::CommunicatorMPI::GetInstance()->GetRank() == 0) {
            dataLoader::csv::CSVLoader<T>::GetInstance()->WriteData(
                    *((T *) data->GetDescriptorData()->GetDescriptor(CHAMELEON_DESCRIPTOR,
                                                                     DESCRIPTOR_Z).chameleon_desc->mat),
                    aConfigurations.GetProblemSize(), parameters_number, path,
                    *data->GetLocations());
        }
        delete[] pMatrix;
#else
        std::string path = aConfigurations.GetLoggerPath();
        dataLoader::csv::CSVLoader<T>::GetInstance()->WriteData(*((T *) data->GetDescriptorData()->GetDescriptor(CHAMELEON_DESCRIPTOR,
                                                                                           DESCRIPTOR_Z).chameleon_desc->mat),
                                          aConfigurations.GetProblemSize(), aKernel.GetVariablesNumber(), path,
                                          *data->GetLocations());
#endif
    }

    Results::GetInstance()->SetGeneratedLocationsNumber(aConfigurations.GetProblemSize());
    Results::GetInstance()->SetIsLogger(aConfigurations.GetLogger());
    Results::GetInstance()->SetLoggerPath(aConfigurations.GetLoggerPath());

    return data;
}

template<typename T>
void StarpuGenerator<T>::ReleaseInstance() {
    if (mpInstance != nullptr) {
        mpInstance = nullptr;
    }
}

template<typename T> StarpuGenerator<T> *StarpuGenerator<T>::mpInstance = nullptr;
