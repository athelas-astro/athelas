#include "timestepper/operator_split_stepper.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"

namespace athelas {

using eos::EOS;

/**
 * @brief Operator split timestep
 */
void OperatorSplitStepper::step(PackageManager *pkgs, MeshState &mesh_state,
                                const GridStructure &grid, const double t,
                                const double dt) {

  static constexpr int stage = 0; // op split
  auto sd0 = mesh_state(stage);
  auto U = sd0.get_field("u_cf");

  const TimeStepInfo dt_info{
      .t = t, .dt = dt, .dt_coef_implicit = dt, .dt_coef = dt, .stage = 0};

  pkgs->fill_derived(sd0, grid, dt_info);
  pkgs->update_explicit(sd0, grid, dt_info);

  // TODO(astrobarker): need to think about what goes into this for opsplit.
  pkgs->update_implicit(sd0, U, grid, dt_info);
  pkgs->apply_delta(U, dt_info);
  pkgs->zero_delta();
}

} // namespace athelas
