#include "timestepper/operator_split_stepper.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"

namespace athelas {

using eos::EOS;

/**
 * @brief Operator split timestep
 */
void OperatorSplitStepper::step(PackageManager *pkgs, MeshState &mesh_state,
                                const GridStructure &grid,
                                TimeStepInfo &dt_info) {

  static constexpr int stage = 0; // op split
  auto sd0 = mesh_state(stage);
  auto U = sd0.get_field("u_cf");

  dt_info.stage = stage;
  dt_info.dt_coef_implicit = dt_info.dt;
  dt_info.dt_coef = dt_info.dt;

  pkgs->fill_derived(sd0, grid, dt_info);
  pkgs->update_explicit(sd0, grid, dt_info);

  // TODO(astrobarker): need to think about what goes into this for opsplit.
  pkgs->update_implicit(sd0, U, grid, dt_info);
  pkgs->apply_delta(U, dt_info);
  pkgs->zero_delta();
}

} // namespace athelas
