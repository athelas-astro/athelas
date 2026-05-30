#include "timestepper/operator_split_stepper.hpp"
#include "basic_types.hpp"
#include "geometry/mesh.hpp"
#include "interface/packages_base.hpp"
#include "interface/state.hpp"

namespace athelas {

/**
 * @brief Operator split timestep
 */
void OperatorSplitStepper::step(PackageManager *pkgs, MeshState &mesh_state,
                                const Mesh &mesh, TimeStepInfo &dt_info) {

  static constexpr int stage = 0; // op split
  auto sd0 = mesh_state(stage);
  auto U = sd0.get_field("u_cf");

  dt_info.stage = stage;
  dt_info.dt_coef_implicit = dt_info.dt;
  dt_info.dt_coef = dt_info.dt;

  pkgs->fill_derived(sd0, mesh, dt_info);
  pkgs->update_explicit(sd0, mesh, dt_info);

  // TODO(astrobarker): need to think about what goes into this for opsplit.
  pkgs->update_implicit(sd0, U, mesh, dt_info);
  pkgs->apply_delta(U, dt_info);
  pkgs->zero_delta();
}

} // namespace athelas
