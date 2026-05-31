#include "pgen/rad_shock.hpp"

#include <cmath>

#include "eos/eos_variant.hpp"
#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"
#include "utils/constants.hpp"

namespace athelas::pgen::rad_shock {

/**
 * @brief Initialize radiating shock
 **/
void init(MeshState &mesh_state, Mesh *mesh, ProblemIn *pin) {
  const bool rad_active = pin->param()->get<bool>("physics.radiation.enabled");
  athelas_requires(rad_active, "Radiative shock requires radiation enabled!");
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Radiative requires ideal gas eos!");

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

  const auto V_L = pin->param()->get<double>("problem.params.vL", 5.19e7);
  const auto V_R = pin->param()->get<double>("problem.params.vR", 1.73e7);
  const auto rhoL = pin->param()->get<double>("problem.params.rhoL", 5.69);
  const auto rhoR = pin->param()->get<double>("problem.params.rhoR", 17.1);
  const auto T_L = pin->param()->get<double>("problem.params.T_L", 2.18e6); // K
  const auto T_R = pin->param()->get<double>("problem.params.T_R", 7.98e6); // K
  const auto x_d = pin->param()->get<double>("problem.params.x_d", 0.013);

  // TODO(astrobarker): thread through
  const double mu = 1.0 + constants::m_e / constants::m_p;
  auto &eos = mesh_state.eos();
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;
  const double em_gas_L = constants::k_B * T_L / (gm1 * mu * constants::m_p);
  const double em_gas_R = constants::k_B * T_R / (gm1 * mu * constants::m_p);
  const double e_rad_L = constants::a * std::pow(T_L, 4.0);
  const double e_rad_R = constants::a * std::pow(T_R, 4.0);

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: RadShock", DevExecSpace(), ib.s, ib.e,
      qb.s, qb.e, KOKKOS_LAMBDA(const int i, const int q) {
        const double X1 = mesh->centers(i);

        if (X1 <= x_d) {
          evolved(i, q, idx_tau) = 1.0 / rhoL;
          evolved(i, q, idx_vel) = V_L;
          evolved(i, q, idx_ener) = em_gas_L + 0.5 * V_L * V_L;
          evolved(i, q, idx_rad_energy) = e_rad_L / rhoL;

          derived(i, q, idx_density) = rhoL;
        } else {
          evolved(i, q, idx_tau) = 1.0 / rhoR;
          evolved(i, q, idx_vel) = V_R;
          evolved(i, q, idx_ener) = em_gas_R + 0.5 * V_R * V_R;
          evolved(i, q, idx_rad_energy) = e_rad_R / rhoR;

          derived(i, q, idx_density) = rhoR;
        }
      });
}

} // namespace athelas::pgen::rad_shock
