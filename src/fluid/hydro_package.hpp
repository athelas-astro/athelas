/**
 * @file hydro_package.hpp
 * --------------
 *
 * @brief Pure hydrodynamics package
 */

#pragma once

#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "pgen/problem_in.hpp"
#include "state/state.hpp"

namespace athelas::fluid {

using bc::BoundaryConditions;

class HydroPackage {
 public:
  HydroPackage(const ProblemIn * /*pin*/, int n_stages, eos::EOS *eos,
               basis::ModalBasis *basis, BoundaryConditions *bcs, double cfl,
               int nx, bool active = true);

  KOKKOS_FUNCTION
  void update_explicit(const State *const state, AthelasArray3D<double> dU,
                       const GridStructure &grid,
                       const TimeStepInfo &dt_info) const;

  KOKKOS_FUNCTION
  void fluid_divergence(const State *const state, AthelasArray3D<double> dU,
                        const GridStructure &grid, int stage) const;

  KOKKOS_FUNCTION
  void fluid_geometry(const AthelasArray3D<double> ucf,
                      const AthelasArray3D<double> uaf,
                      AthelasArray3D<double> dU,
                      const GridStructure &grid) const;

  [[nodiscard]] KOKKOS_FUNCTION auto
  min_timestep(const State *const state, const GridStructure &grid,
               const TimeStepInfo & /*dt_info*/) const -> double;

  [[nodiscard]] KOKKOS_FUNCTION auto name() const noexcept -> std::string_view;

  [[nodiscard]] KOKKOS_FUNCTION auto is_active() const noexcept -> bool;

  void fill_derived(State *state, const GridStructure &grid,
                    const TimeStepInfo &dt_info) const;

  KOKKOS_FUNCTION
  void set_active(bool active);

  [[nodiscard]] KOKKOS_FUNCTION auto get_flux_u(int stage, int i) const
      -> double;
  [[nodiscard]] KOKKOS_FUNCTION auto basis() const -> const basis::ModalBasis *;

  [[nodiscard]] static constexpr auto num_vars() noexcept -> int {
    return NUM_VARS_;
  }

 private:
  bool active_;

  int nx_;
  double cfl_;

  eos::EOS *eos_;
  basis::ModalBasis *basis_;
  BoundaryConditions *bcs_;

  // package storage
  AthelasArray2D<double> dFlux_num_; // stores Riemann solutions
  AthelasArray2D<double> u_f_l_; // left faces
  AthelasArray2D<double> u_f_r_; // right faces
  AthelasArray2D<double> flux_u_; // Riemann velocities

  // constants
  static constexpr int NUM_VARS_ = 3;
};

} // namespace athelas::fluid
