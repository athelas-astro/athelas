/**
 * @file boundary_conditions_base.hpp
 * --------------
 *
 * @brief Boundary conditions base structures
 *
 * TODO(astrobarker): Move anything possible to .cpp..
 */

#pragma once

#include <cassert>

#include "Kokkos_Macros.hpp"
#include "pgen/problem_in.hpp"

namespace athelas {
enum class Boundary;
}

namespace athelas::bc {

enum class BcType : int {
  Outflow,
  Reflecting,
  Periodic,
  Marshak,
  Surface,
  // Radiation-only: like fluid Outflow, this uses the interior state to compute
  // the physical boundary flux. It is named by data source rather than
  // "outflow" because it does not enforce no incoming radiation
  // characteristics.
  InteriorFlux,
  FreeStreaming,
  Null // don't go here
};

auto parse_bc_type(const std::string &name) -> BcType;
void validate_fluid_bc(BcType type);
void validate_radiation_bc(Boundary side, BcType type);

/**
 * @struct BoundaryConditionData
 * @brief Catch all holder for boundary conditions data
 */
struct BoundaryConditionData {
  BcType type;
  double marshak_incoming_energy = 0.0;
  double surface_pressure = 0.0;
  //  double time; // placeholder for now

  // necessary
  BoundaryConditionData() : type(BcType::Outflow) {}

  explicit BoundaryConditionData(BcType type_) : type(type_) {}

  BoundaryConditionData(BcType type_, const double marshak_incoming_energy_)
      : type(type_), marshak_incoming_energy(marshak_incoming_energy_) {
    assert(type_ == BcType::Marshak &&
           "This constructor is for Marshak boundary conditions!\n");
  }
};

inline auto make_bc(BcType type) -> BoundaryConditionData {
  return BoundaryConditionData(type);
}

inline auto make_marshak_bc(const double incoming_energy)
    -> BoundaryConditionData {
  return BoundaryConditionData(BcType::Marshak, incoming_energy);
}

inline auto make_surface_bc(const double pressure) -> BoundaryConditionData {
  auto bc = BoundaryConditionData(BcType::Surface);
  bc.surface_pressure = pressure;
  return bc;
}

struct BoundaryConditions {
  // in the below arrays, 0 is inner boundary, 1 is outer
  Kokkos::Array<BoundaryConditionData, 2> fluid_bc;
  Kokkos::Array<BoundaryConditionData, 2> rad_bc;
  bool do_rad = false;
};

KOKKOS_INLINE_FUNCTION auto fluid_bc_data(BoundaryConditions *bc)
    -> Kokkos::Array<BoundaryConditionData, 2> {
  return bc->fluid_bc;
}

KOKKOS_INLINE_FUNCTION auto radiation_bc_data(BoundaryConditions *bc)
    -> Kokkos::Array<BoundaryConditionData, 2> {
  assert(bc->do_rad && "Need radiation enabled to get radiation bcs!\n");
  return bc->rad_bc;
}

auto make_boundary_conditions(const ProblemIn *pin) -> BoundaryConditions;
} // namespace athelas::bc
