/**
 * @file rad_equilibrium.hpp
 * --------------
 *
 * @brief Radiation fluid equilibriation test
 */

#pragma once

#include <cmath> /* sin */

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"
#include "utils/constants.hpp"

namespace athelas {

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
void rad_shock_steady_init(MeshState &mesh_state, GridStructure *grid,
                           ProblemIn *pin, bool /*first_init*/) {
  const bool rad_active = pin->param()->get<bool>("physics.rad_active");
  athelas_requires(rad_active,
                   "Steady radiative shock requires radiation enabled!");
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Steady radiative requires ideal gas eos!");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const int nNodes = grid->n_nodes();

  const int q_Tau = 0;
  const int q_V = 1;
  const int q_E = 2;

  const int iPF_D = 0;

  const int iCR_E = 3;

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
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: RadShockSteady (1)", DevExecSpace(),
      ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
        const int k = 0;
        const double X1 = grid->centers(i);

        if (X1 <= 0.0) {
          uCF(i, k, q_Tau) = 1.0 / rhoL;
          uCF(i, k, q_V) = V0;
          uCF(i, k, q_E) = em_gas_L;
          uCF(i, k, iCR_E) = e_rad_L;

          for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
            uPF(i, iNodeX, iPF_D) = rhoL;
          }
        } else {
          uCF(i, k, q_Tau) = 1.0 / rhoR;
          uCF(i, k, q_V) = V0;
          uCF(i, k, q_E) = em_gas_R;
          uCF(i, k, iCR_E) = e_rad_R;

          for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
            uPF(i, iNodeX, iPF_D) = rhoR;
          }
        }
      });

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: RadShockSteady (ghost)",
      DevExecSpace(), 0, ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          uPF(ib.s - 1 - i, iN, 0) = uPF(ib.s + i, (nNodes + 2) - iN - 1, 0);
          uPF(ib.s + 1 + i, iN, 0) = uPF(ib.s - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas
