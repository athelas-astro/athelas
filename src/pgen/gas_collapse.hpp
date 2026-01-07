/**
 * @file gas_collapse.hpp
 * --------------
 *
 * @brief Collapsing gas cloud
 */

#pragma once

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * @brief Initialize gas collapse
 **/
void gas_collapse_init(MeshState &mesh_state, GridStructure *grid,
                       ProblemIn *pin, const eos::EOS *eos,
                       basis::ModalBasis * /*fluid_basis = nullptr*/) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Gas collapse requires ideal gas eos!");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  static const IndexRange ib(grid->domain<Domain::Interior>());
  const int nNodes = grid->n_nodes();

  constexpr static int q_Tau = 0;
  constexpr static int q_V = 1;
  constexpr static int q_E = 2;

  constexpr static int iPF_D = 0;

  const auto V0 = pin->param()->get<double>("problem.params.v0", 0.0);
  const auto rho0 = pin->param()->get<double>("problem.params.rho0", 1.0);
  const auto p0 = pin->param()->get<double>("problem.params.p0", 10.0);

  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: GasCollapse (1)", DevExecSpace(),
      ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
        const int k = 0;

        uCF(i, k, q_Tau) = rho0; // / rho0 * (1.0 / std::cosh(x / H));
        uCF(i, k, q_V) = V0;
        uCF(i, k, q_E) = (p0 / gm1) * uCF(i, k, q_Tau) + 0.5 * V0 * V0;

        for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
          uPF(i, iNodeX, iPF_D) = rho0;
        }
      });

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: GasCollapse (ghost)", DevExecSpace(),
      0, ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          uPF(ib.s - 1 - i, iN, 0) = uPF(ib.s + i, (nNodes + 2) - iN - 1, 0);
          uPF(ib.s + 1 + i, iN, 0) = uPF(ib.s - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas
