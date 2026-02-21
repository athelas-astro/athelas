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

  constexpr static int vars::cons::SpecificVolume = 0;
  constexpr static int vars::cons::Velocity = 1;
  constexpr static int vars::cons::Energy = 2;

  constexpr static int vars::prim::Rho = 0;

  const auto V0 = pin->param()->get<double>("problem.params.v0", 0.0);
  const auto rho0 = pin->param()->get<double>("problem.params.rho0", 1.0);
  const auto p0 = pin->param()->get<double>("problem.params.p0", 10.0);

  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: GasCollapse (1)", DevExecSpace(),
      ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
        const int k = 0;

        uCF(i, k, vars::cons::SpecificVolume) =
            rho0; // / rho0 * (1.0 / std::cosh(x / H));
        uCF(i, k, vars::cons::Velocity) = V0;
        uCF(i, k, vars::cons::Energy) =
            (p0 / gm1) * uCF(i, k, vars::cons::SpecificVolume) + 0.5 * V0 * V0;

        for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
          uPF(i, iNodeX, vars::prim::Rho) = rho0;
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
