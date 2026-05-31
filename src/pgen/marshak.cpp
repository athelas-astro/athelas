#include "pgen/marshak.hpp"

#include <cmath>

#include "basic_types.hpp"
#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"
#include "utils/constants.hpp"

namespace athelas::pgen::marshak {

/**
 * @brief Initialize radiating shock
 **/
void init(MeshState &mesh_state, Mesh *mesh, ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "marshak",
                   "Marshak requires marshak eos!");

  const bool rad_active = pin->param()->get<bool>("physics.radiation.enabled");
  athelas_requires(rad_active, "Marshak requires radiation enabled!");

  auto evolved = mesh_state(0).get_field("evolved");
  auto derived = mesh_state(0).get_field("derived");

  const int idx_tau = mesh_state(0).var_index("evolved", "specific_volume");
  const int idx_vel = mesh_state(0).var_index("evolved", "velocity");
  const int idx_ener =
      mesh_state(0).var_index("evolved", "specific_total_fluid_energy");
  const int idx_rad_energy =
      mesh_state(0).var_index("evolved", "specific_radiation_energy");
  const int idx_rad_flux =
      mesh_state(0).var_index("evolved", "specific_radiation_flux");
  const int idx_density = mesh_state(0).var_index("derived", "density");

  const int nNodes = mesh->n_nodes();
  static const IndexRange ib(mesh->domain<Domain::Interior>());
  const IndexRange qb(nNodes);

  auto su_olson_energy = [&](const double alpha, const double T) {
    return (alpha / 4.0) * std::pow(T, 4.0);
  };

  const auto V0 = pin->param()->get<double>("problem.params.v0", 0.0);
  const auto rho0 = pin->param()->get<double>("problem.params.rho0", 10.0);
  const auto epsilon = pin->param()->get<double>("problem.params.epsilon", 1.0);
  const auto T0 = pin->param()->get<double>("problem.params.T0", 1.0e4); // K

  const double alpha = 4.0 * constants::a / epsilon;
  const double em_gas = su_olson_energy(alpha, T0) / rho0;

  // TODO(astrobarker): thread through
  const double e_rad = constants::a * std::pow(T0, 4.0);

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: Marshak", DevExecSpace(), ib.s, ib.e, qb.s,
      qb.e, KOKKOS_LAMBDA(const int i, const int q) {
        evolved(i, q, idx_tau) = 1.0 / rho0;
        evolved(i, q, idx_vel) = V0;
        evolved(i, q, idx_ener) = em_gas + 0.5 * V0 * V0;
        evolved(i, q, idx_rad_energy) = e_rad / rho0;
        evolved(i, q, idx_rad_flux) = 0.0;

        derived(i, q, idx_density) = rho0;
      });
}

} // namespace athelas::pgen::marshak
