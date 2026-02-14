#pragma once

#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * @brief Initialize shockless Noh problem
 **/
void shockless_noh_init(MeshState &mesh_state, GridStructure *grid,
                        ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Shockless Noh requires ideal gas eos!");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  static const int nNodes = grid->n_nodes();
  static const IndexRange ib(grid->domain<Domain::Interior>());
  const IndexRange qb(nNodes);

  const auto D = pin->param()->get<double>("problem.params.rho0", 1.0);
  const auto E_M =
      pin->param()->get<double>("problem.params.specific_energy", 1.0);

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: ShocklessNoh", DevExecSpace(),
      ib.s, ib.e, qb.s, qb.e, KOKKOS_LAMBDA(const int i, const int q) {
        const double X1 = grid->centers(i);

        uCF(i, q, vars::cons::SpecificVolume) = 1.0 / D;
        uCF(i, q, vars::cons::Velocity) = -X1;
        uCF(i, q, vars::cons::Energy) = E_M + 0.5 * uCF(i, q, vars::cons::Velocity) * uCF(i, q, vars::cons::Velocity);

        uPF(i, q, vars::prim::Rho) = D;
      });
}

} // namespace athelas
