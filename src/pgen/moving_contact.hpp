#pragma once

#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * @brief Initialize moving contact discontinuity test
 **/
void moving_contact_init(MeshState &mesh_state, GridStructure *grid,
                         ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Moving contact requires ideal gas eos!");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  const int nNodes = grid->n_nodes();
  static const IndexRange ib(grid->domain<Domain::Interior>());
  const IndexRange qb(nNodes);

  const auto V0 = pin->param()->get<double>("problem.params.v0", 0.1);
  const auto D_L = pin->param()->get<double>("problem.params.rhoL", 1.4);
  const auto D_R = pin->param()->get<double>("problem.params.rhoR", 1.0);
  const auto P_L = pin->param()->get<double>("problem.params.pL", 1.0);
  const auto P_R = pin->param()->get<double>("problem.params.pR", 1.0);

  const auto &eos = mesh_state.eos();
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: MovingContact (1)", DevExecSpace(), ib.s,
      ib.e, qb.s, qb.e, KOKKOS_LAMBDA(const int i, const int q) {
        const double X1 = grid->centers(i);

        if (X1 <= 0.5) {
          uCF(i, q, vars::cons::SpecificVolume) = 1.0 / D_L;
          uCF(i, q, vars::cons::Velocity) = V0;
          uCF(i, q, vars::cons::Energy) =
              (P_L / gm1) * uCF(i, q, vars::cons::SpecificVolume) +
              0.5 * V0 * V0;

          uPF(i, q, vars::prim::Rho) = D_L;
        } else {
          uCF(i, q, vars::cons::SpecificVolume) = 1.0 / D_R;
          uCF(i, q, vars::cons::Velocity) = V0;
          uCF(i, q, vars::cons::Energy) =
              (P_R / gm1) * uCF(i, q, vars::cons::SpecificVolume) +
              0.5 * V0 * V0;

          uPF(i, q, vars::prim::Rho) = D_R;
        }
      });
}

} // namespace athelas
