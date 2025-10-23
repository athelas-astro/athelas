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

  void update_explicit(const State *const state, const GridStructure &grid,
                       const TimeStepInfo &dt_info) const;

  void apply_delta(AthelasArray3D<double> lhs,
                   const TimeStepInfo &dt_info) const;

  void fluid_divergence(const State *const state, const GridStructure &grid,
                        int stage) const;

  void fluid_geometry(const AthelasArray3D<double> ucf,
                      const AthelasArray3D<double> uaf,
                      const GridStructure &grid) const;

  [[nodiscard]] auto min_timestep(const State *const state,
                                  const GridStructure &grid,
                                  const TimeStepInfo & /*dt_info*/) const
      -> double;

  [[nodiscard]] auto name() const noexcept -> std::string_view;

  [[nodiscard]] auto is_active() const noexcept -> bool;

  void fill_derived(State *state, const GridStructure &grid,
                    const TimeStepInfo &dt_info) const;

  void set_active(bool active);

  [[nodiscard]] auto get_flux_u(int stage, int i) const -> double;
  [[nodiscard]] auto basis() const -> const basis::ModalBasis *;

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

  AthelasArray3D<double> delta_; // rhs delta

  // constants
  static constexpr int NUM_VARS_ = 3;
};

} // namespace athelas::fluid
