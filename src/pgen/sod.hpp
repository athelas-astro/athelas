/**
 * @file sod.hpp
 * --------------
 *
 * @brief Sod shock tube
 */

#pragma once

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "loop_layout.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * @brief Initialize Sod shock tube
 **/
void sod_init(State *state, GridStructure *grid, ProblemIn *pin,
              const eos::EOS *eos,
              basis::ModalBasis * /*fluid_basis = nullptr*/) {
  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Sod requires ideal gas eos!");
  }

  AthelasArray3D<double> uCF = state->u_cf();
  AthelasArray3D<double> uPF = state->u_pf();

  static const int ilo = 1;
  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const int nNodes = grid->n_nodes();

  constexpr static int q_Tau = 0;
  constexpr static int q_V = 1;
  constexpr static int q_E = 2;

  constexpr static int iPF_D = 0;

  const auto V_L = pin->param()->get<double>("problem.params.vL", 0.0);
  const auto V_R = pin->param()->get<double>("problem.params.vR", 0.0);
  const auto D_L = pin->param()->get<double>("problem.params.rhoL", 1.0);
  const auto D_R = pin->param()->get<double>("problem.params.rhoR", 0.125);
  const auto P_L = pin->param()->get<double>("problem.params.pL", 1.0);
  const auto P_R = pin->param()->get<double>("problem.params.pR", 0.1);
  const auto x_d = pin->param()->get<double>("problem.params.x_d", 0.5);

  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  // Phase 1: Initialize nodal values (always done)
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Sod (1)", DevExecSpace(), ib.s, ib.e,
      KOKKOS_LAMBDA(const int i) {
        const double X1 = grid->centers(i);

        if (X1 <= x_d) {
          for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
            uPF(i, iNodeX, iPF_D) = D_L;
          }
        } else {
          for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
            uPF(i, iNodeX, iPF_D) = D_R;
          }
        }
      });

  // Phase 2: Initialize modal coefficients
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Sod (2)", DevExecSpace(), ib.s, ib.e,
      KOKKOS_LAMBDA(const int i) {
        const int k = 0;
        const double X1 = grid->centers(i);

        if (X1 <= x_d) {
          uCF(i, k, q_Tau) = 1.0 / D_L;
          uCF(i, k, q_V) = V_L;
          uCF(i, k, q_E) = (P_L / gm1) * uCF(i, k, q_Tau) + 0.5 * V_L * V_L;
        } else {
          uCF(i, k, q_Tau) = 1.0 / D_R;
          uCF(i, k, q_V) = V_R;
          uCF(i, k, q_E) = (P_R / gm1) * uCF(i, k, q_Tau) + 0.5 * V_R * V_R;
        }
      });

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Sod (ghost)", DevExecSpace(), 0,
      ilo - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          uPF(ib.s - 1 - i, iN, 0) = uPF(ib.s + i, (nNodes + 2) - iN - 1, 0);
          uPF(ib.e + 1 + i, iN, 0) = uPF(ib.e - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas
