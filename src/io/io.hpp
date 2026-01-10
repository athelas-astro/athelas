/**
 * @file io.hpp
 * --------------
 *
 * @brief HDF5 and std out IO routines
 *
 * @details Collection of functions for IO
 */

#pragma once

#include "basis/polynomial_basis.hpp"
#include "geometry/grid.hpp"
#include "limiters/slope_limiter.hpp"
#include "pgen/problem_in.hpp"
#include "state/state.hpp"

#include "H5Cpp.h"

namespace athelas::io {

struct GridType {
  double r{};
};

struct DataType {
  double x{};
};

// ---------------------------------------------------------------------------
// map a C++ scalar type to an HDF5 PredType
// ---------------------------------------------------------------------------
template <typename T>
auto h5_predtype() -> H5::PredType {
  if constexpr (std::is_same_v<T, float>) {
    return H5::PredType::NATIVE_FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    return H5::PredType::NATIVE_DOUBLE;
  } else if constexpr (std::is_same_v<T, int>) {
    return H5::PredType::NATIVE_INT;
  } else if constexpr (std::is_same_v<T, long>) {
    return H5::PredType::NATIVE_LONG;
  } else {
    static_assert(std::is_arithmetic_v<T>, "Unsupported scalar type for HDF5");
  }
}
void write_state(const StageData &stage_data, GridStructure &grid,
                 SlopeLimiter *SL, ProblemIn *pin, double time, int order,
                 int i_write, bool do_rad);

void print_simulation_parameters(GridStructure &grid, ProblemIn *pin);

void write_basis(const basis::ModalBasis &basis,
                 const std::string &problem_name);

} // namespace athelas::io
