/**
 * @file io.hpp
 * --------------
 *
 * @brief HDF5 and std out IO routines
 *
 * @details Collection of functions for IO
 */

#pragma once

#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "limiters/slope_limiter.hpp"
#include "pgen/problem_in.hpp"

#include "H5Cpp.h"

namespace athelas {
class PackageManager;
namespace io {

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

struct MeshType {
  double r{};
};

struct DataType {
  double x{};
};

/**
 * @brief Snapshot of evolution state written to (and read from) /info.
 *
 * **Counter semantics: "last completed" on disk; restart adds 1 to resume.**
 *
 *   - `last_cycle`     — index of the cycle whose step produced this dump
 *                        (0 for the pre-loop initial dump; loop cycles are
 *                        1-indexed). On restart, the next cycle to run is
 *                        `last_cycle + 1`.
 *   - `last_out_h5`    — highest .ath file index already on disk (excluding
 *                        the unnumbered `_final`). On restart, the next
 *                        in-loop dump uses index `last_out_h5 + 1`.
 *   - `last_out_hist`  — index of the most-recently-written history entry
 *                        (0 after the initial entry; incremented per fire).
 *                        On restart, the next history entry uses index
 *                        `last_out_hist + 1`.
 *
 * Both the in-loop and post-loop writers normalize their live "next-pending"
 * loop variables to these "last completed" values before constructing a
 * SimInfo, so the restart side can use a single `+1` rule uniformly.
 */
struct SimInfo {
  double time;
  double dt;
  int last_cycle;
  int last_out_h5;
  int last_out_hist;
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

void write_output(const MeshState &mesh_state, Mesh &mesh,
                  const PackageManager *packages,
                  const PackageManager *split_packages, ProblemIn *pin,
                  const std::string &filename, const SimInfo &info);

void write_output(const MeshState &mesh_state, Mesh &mesh,
                  const PackageManager *packages,
                  const PackageManager *split_packages, SlopeLimiter *SL,
                  ProblemIn *pin, const SimInfo &info, int i_write);

void print_simulation_parameters(Mesh &mesh, ProblemIn *pin);
} // namespace io
} // namespace athelas
