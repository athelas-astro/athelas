/**
 * @file boundary_conditions_base.hpp
 * --------------
 *
 * @brief Boundary conditions base structures
 *
 * TODO(astrobarker):
 *  - add bc guards in pgen: marshak only for rad, etc
 *  - Move anything possible to .cpp..
 */

#pragma once

#include <array>
#include <cassert>

#include "pgen/problem_in.hpp"

namespace athelas::bc {

enum class BcType : int {
  Outflow,
  Dirichlet,
  Reflecting,
  Periodic,
  Marshak,
  Null // don't go here
};

auto parse_bc_type(const std::string &name) -> BcType;

template <int N>
struct BoundaryConditionsData {
  BcType type;
  Kokkos::Array<double, N> dirichlet_values;
  //  double time; // placeholder for now

  // necessary
  BoundaryConditionsData() : type(BcType::Outflow) {}

  explicit BoundaryConditionsData(BcType type_) : type(type_) {}

  BoundaryConditionsData(BcType type_, const std::array<double, N> vals)
      : type(type_) {
    assert((type_ == BcType::Dirichlet || type_ == BcType::Marshak) &&
           "This constructor is for Dirichlet and Marshak boundary "
           "conditions!\n");
    for (int i = 0; i < N; ++i) {
      dirichlet_values[i] = vals[i];
    }
  }

  // TODO(astrobarker) overload ()?
  [[nodiscard]]
  auto get_dirichlet_value(const int i) const -> double {
    return dirichlet_values[i];
  }
};

constexpr static int NUM_HYDRO_VARS = 3;
constexpr static int NUM_RAD_VARS = 2;
struct BoundaryConditions {
  // in the below arrays, 0 is inner boundary, 1 is outer
  Kokkos::Array<BoundaryConditionsData<NUM_HYDRO_VARS>, 2> fluid_bc;
  Kokkos::Array<BoundaryConditionsData<NUM_RAD_VARS>, 2> rad_bc;
  bool do_rad = false;
};

// --- helper functions to pull out bc ---
template <int N>
auto get_bc_data(BoundaryConditions *bc)
    -> Kokkos::Array<BoundaryConditionsData<N>, 2>;

template <>
auto get_bc_data<3>(BoundaryConditions *bc)
    -> Kokkos::Array<BoundaryConditionsData<3>, 2> {
  return bc->fluid_bc;
}

template <>
auto get_bc_data<2>(BoundaryConditions *bc)
    -> Kokkos::Array<BoundaryConditionsData<2>, 2> {
  assert(bc->do_rad && "Need radiation enabled to get radiation bcs!\n");
  return bc->rad_bc;
}

auto make_boundary_conditions(const ProblemIn *pin) -> BoundaryConditions;
} // namespace athelas::bc
