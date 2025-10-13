/**
 * @file ejecta_csm.hpp
 * --------------
 *
 * @brief Ejecta - CSM interaction test.
 * See Duffell 2016 (doi:10.3847/0004-637X/821/2/76)
 */

#pragma once

#include <cmath>

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * @brief Initialize ejecta csm test
 **/
void ejecta_csm_init(State *state, GridStructure *grid, ProblemIn *pin,
                     const eos::EOS *eos,
                     basis::ModalBasis *fluid_basis = nullptr) {
  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Shu Osher requires ideal gas eos!");
  }

  AthelasArray3D<double> uCF = state->u_cf();
  AthelasArray3D<double> uPF = state->u_pf();

  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const int nNodes = grid->n_nodes();

  constexpr static int q_Tau = 0;
  constexpr static int q_V = 1;
  constexpr static int q_E = 2;

  constexpr static int iPF_D = 0;

  const auto rstar = pin->param()->get<double>("problem.params.rstar", 0.01);
  const auto vmax =
      pin->param()->get<double>("problem.params.vmax", std::sqrt(10.0 / 3.0));

  const double rstar3 = rstar * rstar * rstar;

  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  // Phase 1: Initialize nodal values (always done)
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: EjectaCSM (1)", DevExecSpace(), ib.s,
      ib.e, KOKKOS_LAMBDA(const int i) {
        for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
          const double x = grid->node_coordinate(i, iNodeX);
          if (x <= rstar) {
            uPF(i, iNodeX + 1, iPF_D) =
                1.0 / (constants::FOURPI * rstar3 / 3.0);
          } else {
            uPF(i, iNodeX + 1, iPF_D) = 1.0;
          }
        }
      });

  // Phase 2: Initialize modal coefficients
  if (fluid_basis != nullptr) {
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: EjectaCSM (2)", DevExecSpace(),
        ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          const int k = 0;
          const double X1 = grid->centers(i);

          if (X1 <= rstar) {
            const double rho = 1.0 / (constants::FOURPI * rstar3 / 3.0);
            const double pressure = (1.0e-5) * rho * vmax * vmax;
            const double vel = vmax * (X1 / rstar);
            uCF(i, k, q_Tau) = 1.0 / rho;
            uCF(i, k, q_V) = vel;
            uCF(i, k, q_E) = (pressure / gm1 / rho) + 0.5 * vel * vel;
          } else {
            const double rho = 1.0;
            const double pressure = (1.0e-5) * rho * vmax * vmax;
            uCF(i, k, q_Tau) = 1.0 / rho;
            uCF(i, k, q_V) = 0.0;
            uCF(i, k, q_E) = (pressure / gm1 / rho);
          }
        });
  }

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: EjectaCSM (ghost)", DevExecSpace(), 0,
      ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          uPF(ib.s - 1 - i, iN, 0) = uPF(ib.s + i, (nNodes + 2) - iN - 1, 0);
          uPF(ib.e + 1 + i, iN, 0) = uPF(ib.e - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas
