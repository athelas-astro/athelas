/**
 * @file rad_shock.hpp
 * --------------
 *
 * @brief Radiation shock test
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
 * @brief Initialize radiating shock
 **/
void rad_shock_init(MeshState &mesh_state, GridStructure *grid, ProblemIn *pin,
                    bool /*first_init*/) {
  const bool rad_active = pin->param()->get<bool>("physics.rad_active");
  athelas_requires(rad_active, "Radiative shock requires radiation enabled!");
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Radiative requires ideal gas eos!");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const int nNodes = grid->n_nodes();

  constexpr static int q_Tau = 0;
  constexpr static int q_V = 1;
  constexpr static int q_E = 2;

  constexpr static int iPF_D = 0;

  constexpr static int iCR_E = 3;

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
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: RadShock (1)", DevExecSpace(), ib.s,
      ib.e, KOKKOS_LAMBDA(const int i) {
        const int k = 0;
        const double X1 = grid->centers(i);

        if (X1 <= x_d) {
          uCF(i, k, q_Tau) = 1.0 / rhoL;
          uCF(i, k, q_V) = V_L;
          uCF(i, k, q_E) = em_gas_L + 0.5 * V_L * V_L;
          uCF(i, k, iCR_E) = e_rad_L;

          for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
            uPF(i, iNodeX, iPF_D) = rhoL;
          }
        } else {
          uCF(i, k, q_Tau) = 1.0 / rhoR;
          uCF(i, k, q_V) = V_R;
          uCF(i, k, q_E) = em_gas_R + 0.5 * V_R * V_R;
          uCF(i, k, iCR_E) = e_rad_R;

          for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
            uPF(i, iNodeX, iPF_D) = rhoR;
          }
        }
      });

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: RadShock (ghost)", DevExecSpace(), 0,
      ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          uPF(ib.s - 1 - i, iN, 0) = uPF(ib.s + i, (nNodes + 2) - iN - 1, 0);
          uPF(ib.s + 1 + i, iN, 0) = uPF(ib.s - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas
