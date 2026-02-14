#pragma once

#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "loop_layout.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * @brief Initialize Sod shock tube
 **/
void sod_init(MeshState &mesh_state, GridStructure *grid, ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Sod requires ideal gas eos!");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const int nNodes = grid->n_nodes();

  const auto V_L = pin->param()->get<double>("problem.params.vL", 0.0);
  const auto V_R = pin->param()->get<double>("problem.params.vR", 0.0);
  const auto D_L = pin->param()->get<double>("problem.params.rhoL", 1.0);
  const auto D_R = pin->param()->get<double>("problem.params.rhoR", 0.125);
  const auto P_L = pin->param()->get<double>("problem.params.pL", 1.0);
  const auto P_R = pin->param()->get<double>("problem.params.pR", 0.1);
  const auto x_d = pin->param()->get<double>("problem.params.x_d", 0.5);

  auto &eos = mesh_state.eos();
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Sod (1)", DevExecSpace(), ib.s, ib.e,
      KOKKOS_LAMBDA(const int i) {
        const double X1 = grid->centers(i);

        if (X1 <= x_d) {
          for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
            uPF(i, iNodeX, vars::prim::Rho) = D_L;
          }
        } else {
          for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
            uPF(i, iNodeX, vars::prim::Rho) = D_R;
          }
        }
      });

    static const IndexRange nb(nNodes);
    auto r = grid->nodal_grid();
    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "Pgen :: Sod", DevExecSpace(), ib.s, ib.e,
        nb.s, nb.e,
        KOKKOS_LAMBDA(const int i, const int q) {
          const double x = r(i, q+1);
          if (x <= x_d) {
            uCF(i, q, vars::cons::SpecificVolume) = 1.0 / D_L;
            uCF(i, q, vars::cons::Velocity) = V_L;
            uCF(i, q, vars::cons::Energy) =
                (P_L / gm1) * uCF(i, q, vars::cons::SpecificVolume) + 0.5 * V_L * V_L;
          } else {
            uCF(i, q, vars::cons::SpecificVolume) = 1.0 / D_R;
            uCF(i, q, vars::cons::Velocity) = V_R;
            uCF(i, q, vars::cons::Energy) =
                (P_R / gm1) * uCF(i, q, vars::cons::SpecificVolume) + 0.5 * V_R * V_R;
          }
        });
}

} // namespace athelas
