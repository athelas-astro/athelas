#include "pgen/rad_shock_steady.hpp"

#include <cmath>

#include "eos/eos_variant.hpp"
#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"
#include "utils/constants.hpp"

namespace athelas::pgen::rad_shock_steady {

/**
 * @brief Initialize steady radiating shock
 *
 * Two different cases: Mach 2 and Mach 5.
 *
 * Mach 2 Case:
 * - Left side (pre-shock):
 *   - Density: 1.0 g/cm^3
 *   - Temperature: 1.16045181e6 K (100eV)
 * - Right side (post-shock):
 *   - Density: 2.286 g/cm^3
 *   - Temperature: 2.4109e6 K (207.756 eV)
 *
 * Mach 5 Case:
 * - Left side (pre-shock):
 *   - Density: 1.0 g/cm^3
 *   - Temperature: 1.16045181e6 K (100 eV)
 * - Right side (post-shock):
 *   - Density: 3.598 g/cm^3
 *   - Temperature: 9.9302e6 K (855.720 eV)
 **/
void init(MeshState &mesh_state, Mesh *mesh, ProblemIn *pin) {
  const bool rad_active = pin->param()->get<bool>("physics.radiation.enabled");
  athelas_requires(rad_active,
                   "Steady radiative shock requires radiation enabled!");
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Steady radiative requires ideal gas eos!");

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
  const auto rhoL = pin->param()->get<double>("problem.params.rhoL", 1.0);
  const auto rhoR = pin->param()->get<double>("problem.params.rhoR", 2.286);
  const auto T_L =
      pin->param()->get<double>("problem.params.T_L", 1.16045181e6); // K
  const auto T_R =
      pin->param()->get<double>("problem.params.T_R", 2.4109e6); // K

  // TODO(astrobarker): thread through
  const double Abar = 1.0;
  const auto &eos = mesh_state.eos();
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;
  const double em_gas_L =
      (T_L * constants::N_A * constants::k_B) / (gm1 * Abar);
  const double em_gas_R =
      (T_R * constants::N_A * constants::k_B) / (gm1 * Abar);
  const double e_rad_L = constants::a * std::pow(T_L, 4.0);
  const double e_rad_R = constants::a * std::pow(T_R, 4.0);

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: RadShockSteady", DevExecSpace(), ib.s,
      ib.e, qb.s, qb.e, KOKKOS_LAMBDA(const int i, const int q) {
        const double X1 = mesh->centers(i);
        const double x_d = 0.0;

        if (X1 <= x_d) {
          evolved(i, q, idx_tau) = 1.0 / rhoL;
          evolved(i, q, idx_vel) = V0;
          evolved(i, q, idx_ener) = em_gas_L + 0.5 * V0 * V0;
          evolved(i, q, idx_rad_energy) = e_rad_L / rhoL;

          derived(i, q, idx_density) = rhoL;
        } else {
          evolved(i, q, idx_tau) = 1.0 / rhoR;
          evolved(i, q, idx_vel) = V0;
          evolved(i, q, idx_ener) = em_gas_R + 0.5 * V0 * V0;
          evolved(i, q, idx_rad_energy) = e_rad_R / rhoR;

          derived(i, q, idx_density) = rhoR;
        }
      });
}

} // namespace athelas::pgen::rad_shock_steady
