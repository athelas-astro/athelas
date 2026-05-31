#include "pgen/rad_equilibrium.hpp"

#include <cmath>

#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"

namespace athelas::pgen::rad_equilibrium {

/**
 * Initialize equilibrium rad test
 **/
void init(MeshState &mesh_state, Mesh *mesh, ProblemIn *pin) {
  const bool rad_active = pin->param()->get<bool>("physics.radiation.enabled");
  athelas_requires(rad_active,
                   "Radiation equilibriation requires radiation enabled!");
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Radiation equilibriation requires ideal gas eos!");

  auto evolved = mesh_state(0).get_field("evolved");
  auto derived = mesh_state(0).get_field("derived");

  const int idx_tau = mesh_state(0).var_index("evolved", "specific_volume");
  const int idx_vel = mesh_state(0).var_index("evolved", "velocity");
  const int idx_ener =
      mesh_state(0).var_index("evolved", "specific_total_fluid_energy");
  const int idx_rad_energy =
      mesh_state(0).var_index("evolved", "specific_radiation_energy");
  const int idx_density = mesh_state(0).var_index("derived", "density");

  static const int nNodes = mesh->n_nodes();
  static const IndexRange ib(mesh->domain<Domain::Interior>());
  const IndexRange qb(nNodes);

  const auto V0 = pin->param()->get<double>("problem.params.v0", 0.0);
  const auto logD = pin->param()->get<double>("problem.params.logrho", -7.0);
  const auto logE_gas =
      pin->param()->get<double>("problem.params.logE_gas", 10.0);
  const auto logE_rad =
      pin->param()->get<double>("problem.params.logE_rad", 12.0);

  const double D = std::pow(10.0, logD);
  const double Ev_gas = std::pow(10.0, logE_gas);
  const double Ev_rad = std::pow(10.0, logE_rad);

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: RadEquilibrium", DevExecSpace(), ib.s,
      ib.e, qb.s, qb.e, KOKKOS_LAMBDA(const int i, const int q) {
        evolved(i, q, idx_tau) = 1.0 / D;
        evolved(i, q, idx_vel) = V0;
        evolved(i, q, idx_ener) = Ev_gas / D;
        evolved(i, q, idx_rad_energy) = Ev_rad / D;

        derived(i, q, idx_density) = D;
      });
}

} // namespace athelas::pgen::rad_equilibrium
