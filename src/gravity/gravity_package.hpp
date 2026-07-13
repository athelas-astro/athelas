/**
 * @file gravity_package.hpp
 * --------------
 *
 * @brief Gravitational source package
 **/

#pragma once

#include "basic_types.hpp"
#include "basis/nodal_basis.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "pgen/problem_in.hpp"

namespace athelas::gravity {
namespace pkg_vars {
constexpr int Velocity = 0;
constexpr int Energy = 1;
} // namespace pkg_vars

using bc::BoundaryConditions;

class GravityPackage {
 public:
  GravityPackage(const ProblemIn *pin, const std::string &model, double gval,
                 double cfl, int n_stages, bool active = true);

  [[nodiscard]] auto update_explicit(const StageData &stage_data,
                                     const TimeStepInfo &dt_info) const
      -> UpdateStatus;

  template <GravityModel Model>
  void gravity_update(AthelasArray3D<double> evolved, const Mesh &mesh,
                      const basis::NodalBasis &basis, int stage,
                      int idx_vel) const;

  void apply_delta(AthelasArray3D<double> lhs,
                   const TimeStepInfo &dt_info) const;

  void zero_delta() const noexcept;

  [[nodiscard]] auto min_timestep(const StageData &stage_data,
                                  const TimeStepInfo & /*dt_info*/) const
      -> double;

  [[nodiscard]] auto name() const noexcept -> std::string_view;

  [[nodiscard]] auto is_active() const noexcept -> bool;

  void fill_derived(StageData &stage_data, const TimeStepInfo &dt_info) const;

  void set_active(bool active);

 private:
  bool active_;
  GravityModel model_;
  double gval_; // constant gravity
  double cfl_;

  AthelasArray4D<double> delta_; // rhs delta [nstages, nx, nq, nvars]
  AthelasArray2D<double> gravity_pressure_;

  static constexpr int NUM_VARS_ = 2;
};

} // namespace athelas::gravity
