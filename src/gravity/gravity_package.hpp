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
  GravityPackage(const ProblemIn * /*pin*/, GravityModel model, double gval,
                 basis::ModalBasis *basis, double cfl, bool active = true);

  void update_explicit(const State *const state, const GridStructure &grid,
                       const TimeStepInfo &dt_info) const;

  template <GravityModel Model>
  void gravity_update(const AthelasArray3D<double> state,
                      const GridStructure &grid) const;

  void apply_delta(AthelasArray3D<double> lhs,
                   const TimeStepInfo &dt_info) const;

  [[nodiscard]] auto min_timestep(const State *const /*state*/,
                                  const GridStructure & /*grid*/,
                                  const TimeStepInfo & /*dt_info*/) const
      -> double;

  [[nodiscard]] auto name() const noexcept -> std::string_view;

  [[nodiscard]] auto is_active() const noexcept -> bool;

  void fill_derived(State *state, const GridStructure &grid,
                    const TimeStepInfo &dt_info) const;

  void set_active(bool active);

 private:
  bool active_;
  GravityModel model_;

  double gval_; // constant gravity

  basis::ModalBasis *basis_;

  double cfl_;

  AthelasArray3D<double> delta_; // rhs delta

  static constexpr int NUM_VARS_ = 2;
};

} // namespace athelas::gravity
