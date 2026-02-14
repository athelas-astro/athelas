#pragma once

#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * @brief Initialize Noh problem
 **/
void noh_init(MeshState &mesh_state, GridStructure *grid, ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Noh requires ideal gas eos!");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  static const int nNodes = grid->n_nodes();
  static const IndexRange ib(grid->domain<Domain::Interior>());
  const IndexRange qb(nNodes);

  const auto P0 = pin->param()->get<double>("problem.params.p0", 0.000001);
  const auto V0 = pin->param()->get<double>("problem.params.v0", -1.0);
  const auto D0 = pin->param()->get<double>("problem.params.rho0", 1.0);

  const auto &eos = mesh_state.eos();
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: Noh", DevExecSpace(), ib.s, ib.e,
      qb.s, qb.e, KOKKOS_LAMBDA(const int i, const int q) {

        uCF(i, q, vars::cons::SpecificVolume) = 1.0 / D0;
        uCF(i, q, vars::cons::Velocity) = V0;
        uCF(i, q, vars::cons::Energy) = (P0 / gm1) * uCF(i, q, vars::cons::SpecificVolume) + 0.5 * V0 * V0;

        uPF(i, q, vars::prim::Rho) = D0;
      });
}

} // namespace athelas
