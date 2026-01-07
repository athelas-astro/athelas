/**
 * @file shockless_noh.hpp
 * --------------
 *
 * @brief Shockless Noh collapse
 */

#pragma once

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * @brief Initialize shockless Noh problem
 **/
void shockless_noh_init(MeshState &mesh_state, GridStructure *grid,
                        ProblemIn *pin, const eos::EOS * /*eos*/,
                        basis::ModalBasis * /*fluid_basis = nullptr*/) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Shockless Noh requires ideal gas eos!");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const int nNodes = grid->n_nodes();

  constexpr static int q_Tau = 0;
  constexpr static int q_V = 1;
  constexpr static int q_E = 2;

  constexpr static int iPF_D = 0;

  const auto D = pin->param()->get<double>("problem.params.rho0", 1.0);
  const auto E_M =
      pin->param()->get<double>("problem.params.specific_energy", 1.0);

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: ShocklessNoh (1)", DevExecSpace(),
      ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
        const int k = 0;
        const double X1 = grid->centers(i);

        uCF(i, k, q_Tau) = 1.0 / D;
        uCF(i, k, q_V) = -X1;
        uCF(i, k, q_E) = E_M + 0.5 * uCF(i, k, q_V) * uCF(i, k, q_V);

        for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
          uPF(i, iNodeX, iPF_D) = D;
        }
      });

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: ShocklessNoh (ghost)", DevExecSpace(),
      0, ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          uPF(ib.s - 1 - i, iN, 0) = uPF(ib.s + i, (nNodes + 2) - iN - 1, 0);
          uPF(ib.s + 1 + i, iN, 0) = uPF(ib.s - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas
