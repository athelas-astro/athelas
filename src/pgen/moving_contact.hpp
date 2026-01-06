/**
 * @file moving_contact.hpp
 * --------------
 *
 * @brief Moving contact wave test
 */

#pragma once

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * @brief Initialize moving contact discontinuity test
 **/
void moving_contact_init(MeshState &mesh_state, GridStructure *grid,
                         ProblemIn *pin, const eos::EOS *eos,
                         basis::ModalBasis * /*fluid_basis = nullptr*/) {
  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Moving contact requires ideal gas eos!");
  }

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  static const IndexRange ib(grid->domain<Domain::Interior>());
  const int nNodes = grid->n_nodes();

  constexpr static int q_Tau = 0;
  constexpr static int q_V = 1;
  constexpr static int q_E = 2;

  constexpr static int iPF_D = 0;

  const auto V0 = pin->param()->get<double>("problem.params.v0", 0.1);
  const auto D_L = pin->param()->get<double>("problem.params.rhoL", 1.4);
  const auto D_R = pin->param()->get<double>("problem.params.rhoR", 1.0);
  const auto P_L = pin->param()->get<double>("problem.params.pL", 1.0);
  const auto P_R = pin->param()->get<double>("problem.params.pR", 1.0);

  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: MovingContact (1)", DevExecSpace(),
      ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
        const int k = 0;
        const double X1 = grid->centers(i);

        if (X1 <= 0.5) {
          uCF(i, k, q_Tau) = 1.0 / D_L;
          uCF(i, k, q_V) = V0;
          uCF(i, k, q_E) = (P_L / gm1) * uCF(i, k, q_Tau) + 0.5 * V0 * V0;

          for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
            uPF(i, iNodeX, iPF_D) = D_L;
          }
        } else {
          uCF(i, k, q_Tau) = 1.0 / D_R;
          uCF(i, k, q_V) = V0;
          uCF(i, k, q_E) = (P_R / gm1) * uCF(i, k, q_Tau) + 0.5 * V0 * V0;

          for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
            uPF(i, iNodeX, iPF_D) = D_R;
          }
        }
      });

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: MovingContact (ghost)",
      DevExecSpace(), 0, ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          uPF(ib.s - 1 - i, iN, 0) = uPF(ib.s + i, (nNodes + 2) - iN - 1, 0);
          uPF(ib.s + 1 + i, iN, 0) = uPF(ib.s - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas
