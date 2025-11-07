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
  ThermalEnginePackage(const ProblemIn *pin, basis::ModalBasis *basis,
                       bool active = true);

  void update_explicit(const State *const state, const GridStructure &grid,
                       const TimeStepInfo &dt_info);

  void apply_delta(AthelasArray3D<double> lhs,
                   const TimeStepInfo &dt_info) const;

  [[nodiscard]] auto min_timestep(const State * /*state*/,
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

  basis::ModalBasis *basis_;

  AthelasArray3D<double> delta_;

  double energy_;
  std::string mode_;
  double tend_;
  int mstart_;
  double mend_;

  // TODO(astrobarker): Should this be runtime?
  static constexpr double RATIO_TIME_ = 100.0;
  static constexpr double RATIO_MASS_ = 100.0;
};

} // namespace athelas::thermal_engine
