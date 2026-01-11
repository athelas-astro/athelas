#pragma once

#include "Kokkos_Macros.hpp"

#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "compdata.hpp"
#include "geometry/grid.hpp"
#include "pgen/problem_in.hpp"
#include "state/state.hpp"
#include "utils/constants.hpp"

namespace athelas::thermal_engine {

namespace pkg_vars {
constexpr int Energy = 0;
} // namespace pkg_vars

class ThermalEnginePackage {
 public:
  ThermalEnginePackage(const ProblemIn *pin, const StageData &stage_data,
                       const GridStructure *grid, int n_stages,
                       bool active = true);

  void update_explicit(const StageData &stage_data, const GridStructure &grid,
                       const TimeStepInfo &dt_info);

  void apply_delta(AthelasArray3D<double> lhs,
                   const TimeStepInfo &dt_info) const;

  void zero_delta() const noexcept;

  [[nodiscard]] auto min_timestep(const StageData & /*stage_data*/,
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

  AthelasArray4D<double> delta_; // [nstages, nx, order, nvars]

  double energy_target_;
  double energy_dep_; // actual energy to be deposited
  std::string mode_;
  double tend_;
  int mstart_;
  double mend_;
  int mend_idx_; // index of mass spread

  // TODO(astrobarker): Should this be runtime?
  static constexpr double RATIO_TIME_ = 100.0;
  static constexpr double RATIO_MASS_ = 100.0;

  double c_coeff_;
  double d_coeff_;
  double a_coeff_;
  double b_int_;
};

} // namespace athelas::thermal_engine
