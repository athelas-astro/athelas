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

/**
 * @brief Helper struct for organizing HDF5 output
 */
struct HDF5FieldInfo {
  std::string field_name;
  DataPolicy policy;
  std::vector<std::string> var_names;
  int nvars;

  // For building metadata
  struct VariableInfo {
    std::string name;
    std::string location;
    int index;
  };
  std::vector<VariableInfo> variables;
};

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

void write_output(const MeshState &mesh_state, GridStructure &mesh,
                  ProblemIn *pin, const std::string &filename, int cycle,
                  double time);

void write_output(const MeshState &mesh_state, GridStructure &grid,
                  SlopeLimiter *SL, ProblemIn *pin, double time, int i_write);

void print_simulation_parameters(GridStructure &grid, ProblemIn *pin);
} // namespace athelas::io
