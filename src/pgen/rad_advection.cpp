#include "pgen/rad_advection.hpp"

#include <cmath>
#include <string>

#include "eos/eos_variant.hpp"
#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"
#include "utils/constants.hpp"

namespace athelas::pgen::rad_advection {

/**
 * @brief Initialize radiation advection test
 * @note EXPERIMENTAL
 **/
void init(MeshState &mesh_state, Mesh *mesh, ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Radiation advection requires ideal gas eos!");

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

  static const int nNodes = mesh->n_nodes();
  static const IndexRange ib(mesh->domain<Domain::Interior>());
  const IndexRange qb(nNodes);

  const auto V0 = pin->param()->get<double>("problem.params.v0", 1.0);
  const auto velocity_profile = pin->param()->get<std::string>(
      "problem.params.velocity_profile", "constant");
  const auto D = pin->param()->get<double>("problem.params.rho", 1.0);
  const auto amp = pin->param()->get<double>("problem.params.amp", 1.0);
  const auto x0 = pin->param()->get<double>("problem.params.x0", 0.5);
  const auto width = pin->param()->get<double>("problem.params.width", 0.05);
  const auto floor = pin->param()->get<double>("problem.params.floor", 1.0e-8);
  const auto flux_factor =
      pin->param()->get<double>("problem.params.flux_factor", 0.5);
  const auto T_gas = pin->param()->get<double>("problem.params.T_gas", 1.0e4);
  athelas_requires(flux_factor >= 0.0 && flux_factor < 1.0,
                   "Radiation advection requires 0 <= flux_factor < 1.");
  athelas_requires(velocity_profile == "constant" ||
                       velocity_profile == "homologous",
                   "Radiation advection velocity_profile must be 'constant' "
                   "or 'homologous'.");
  const double xR = pin->param()->get<double>("problem.xr");
  const double mu = 1.0 + constants::m_e / constants::m_p;
  auto &eos = mesh_state.eos();
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: RadAdvection", DevExecSpace(), ib.s, ib.e,
      qb.s, qb.e, KOKKOS_LAMBDA(const int i, const int q) {
        const double X1 = mesh->node_coordinate(i, q);
        const double radiation_energy =
            amp *
            std::max(std::exp(-std::pow((X1 - x0) / width, 2.0) / 2.0), floor);

        evolved(i, q, idx_rad_energy) = radiation_energy / D;
        evolved(i, q, idx_rad_flux) =
            flux_factor * constants::c_cgs * evolved(i, q, idx_rad_energy);

        const double sie_fluid =
            constants::k_B * T_gas / (gm1 * mu * constants::m_p);
        const double velocity =
            velocity_profile == "homologous" ? V0 * X1 / xR : V0;
        evolved(i, q, idx_tau) = 1.0 / D;
        evolved(i, q, idx_vel) = velocity;
        evolved(i, q, idx_ener) =
            sie_fluid +
            0.5 * velocity * velocity; // p0 / (gamma - 1.0) / D + 0.5 * v * v;

        derived(i, q, idx_density) = D;
      });
}

} // namespace athelas::pgen::rad_advection
