/**
 * @file gravity_package.hpp
 * --------------
 *
 * @brief Gravitational source package
 **/

#pragma once

#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "geometry/grid.hpp"
#include "pgen/problem_in.hpp"
#include "state/state.hpp"

namespace athelas::gravity {
namespace pkg_vars {
constexpr int Velocity = 0;
constexpr int Energy = 1;
} // namespace pkg_vars

using bc::BoundaryConditions;

class GravityPackage {
 public:
  GravityPackage(const ProblemIn *pin, GravityModel model, double gval,
                 double cfl, int n_stages, bool active = true);

  void update_explicit(const StageData &stage_data, const GridStructure &grid,
                       const TimeStepInfo &dt_info) const;

  template <GravityModel Model>
  void gravity_update(AthelasArray3D<double> state, const GridStructure &grid,
                      int stage, const basis::NodalBasis &basis) const;

  void apply_delta(AthelasArray3D<double> lhs,
                   const TimeStepInfo &dt_info) const;

  void zero_delta() const noexcept;

  [[nodiscard]] auto min_timestep(const StageData & /*state*/,
                                  const GridStructure & /*grid*/,
                                  const TimeStepInfo & /*dt_info*/) const
      -> double;

  [[nodiscard]] auto name() const noexcept -> std::string_view;

  [[nodiscard]] auto is_active() const noexcept -> bool;

  void fill_derived(StageData &stage_data, const GridStructure &grid,
                    const TimeStepInfo &dt_info) const;

  void set_active(bool active);

 private:
  bool active_;
  GravityModel model_;

  double gval_; // constant gravity

  double cfl_;

  AthelasArray4D<double> delta_; // rhs delta [nstages, nx, order, nvars]

  static constexpr int NUM_VARS_ = 2;
};

} // namespace athelas::gravity
