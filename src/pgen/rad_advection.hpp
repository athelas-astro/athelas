/**
 * @file rad_advection.hpp
 * --------------
 *
 * @brief Radiation advection test
 */

#pragma once

#include <cmath>

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"
#include "utils/constants.hpp"

namespace athelas {

/**
 * @brief Initialize radiation advection test
 * @note EXPERIMENTAL
 **/
void rad_advection_init(MeshState &mesh_state, GridStructure *grid,
                        ProblemIn *pin, bool /*first_init*/) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Radiation advection requires ideal gas eos!");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const int nNodes = grid->n_nodes();

  const int q_Tau = 0;
  const int q_V = 1;
  const int q_E = 2;

  const int iPF_D = 0;

  const int iCR_E = 3;
  const int iCR_F = 4;

  const auto V0 = pin->param()->get<double>("problem.params.v0", 1.0);
  const auto D = pin->param()->get<double>("problem.params.rho", 1.0);
  const auto amp = pin->param()->get<double>("problem.params.amp", 1.0);
  const auto width = pin->param()->get<double>("problem.params.width", 0.05);
  const double mu = 1.0 + constants::m_e / constants::m_p;
  auto &eos = mesh_state.eos();
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: RadAdvection (1)", DevExecSpace(),
      ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
        const int k = 0;
        const double X1 = grid->centers(i);

        uCF(i, k, iCR_E) =
            amp * std::max(std::exp(-std::pow((X1 - 0.5) / width, 2.0) / 2.0),
                           1.0e-8);
        uCF(i, k, iCR_F) = 1.0 * constants::c_cgs * uCF(i, k, iCR_E);

        const double Trad = std::pow(uCF(i, k, iCR_E) / constants::a, 0.25);
        const double sie_fluid =
            constants::k_B * Trad / (gm1 * mu * constants::m_p);
        uCF(i, k, q_Tau) = 1.0 / D;
        uCF(i, k, q_V) = V0;
        uCF(i, k, q_E) =
            sie_fluid +
            0.5 * V0 * V0; // p0 / (gamma - 1.0) / D + 0.5 * V0 * V0;

        for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
          uPF(i, iNodeX, iPF_D) = D;
        }
      });

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: RadAdvection (ghost)", DevExecSpace(),
      0, ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          uPF(ib.s - 1 - i, iN, 0) = uPF(ib.s + i, (nNodes + 2) - iN - 1, 0);
          uPF(ib.s + 1 + i, iN, 0) = uPF(ib.s - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas
