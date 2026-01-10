/**
 * @file hydro_package.hpp
 * --------------
 *
 * @brief Pure hydrodynamics package
 */

#pragma once

#include "basic_types.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "geometry/grid.hpp"
#include "pgen/problem_in.hpp"
#include "state/state.hpp"

namespace athelas::fluid {

using bc::BoundaryConditions;

class HydroPackage {
 public:
  HydroPackage(const ProblemIn * /*pin*/, int n_stages, int order,
               BoundaryConditions *bcs, double cfl, int nx, bool active = true);

  void update_explicit(const StageData &stage_data, const GridStructure &grid,
                       const TimeStepInfo &dt_info) const;

  void apply_delta(AthelasArray3D<double> lhs,
                   const TimeStepInfo &dt_info) const;

  void zero_delta() const noexcept;

  void fluid_divergence(const StageData &stage_data, const GridStructure &grid,
                        int stage) const;

  [[nodiscard]] auto min_timestep(const StageData &stage_data,
                                  const GridStructure &grid,
                                  const TimeStepInfo & /*dt_info*/) const
      -> double;

  [[nodiscard]] auto name() const noexcept -> std::string_view;

  [[nodiscard]] auto is_active() const noexcept -> bool;

  void fill_derived(StageData &stage_data, const GridStructure &grid,
                    const TimeStepInfo & /*dt_info*/) const;

  void set_active(bool active);

  [[nodiscard]] auto get_flux_u(int stage, int i) const -> double;

  [[nodiscard]] static constexpr auto num_vars() noexcept -> int {
    return NUM_VARS_;
  }

 private:
  bool active_;

  int nx_;
  double cfl_;

  BoundaryConditions *bcs_;

  // package storage
  AthelasArray2D<double> dFlux_num_; // stores Riemann solutions
  AthelasArray2D<double> u_f_l_; // left faces
  AthelasArray2D<double> u_f_r_; // right faces
  AthelasArray2D<double> flux_u_; // Riemann velocities

  AthelasArray4D<double> delta_; // rhs delta [nstages, nx, order, vars]

  // constants
  static constexpr int NUM_VARS_ = 3;
};

} // namespace athelas::fluid
