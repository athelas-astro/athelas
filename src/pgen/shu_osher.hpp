#pragma once

#include <cmath>

#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * @brief Initialize Shu Osher hydro test
 **/
void shu_osher_init(MeshState &mesh_state, GridStructure *grid, ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Shu Osher requires ideal gas eos!");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const int nNodes = grid->n_nodes();

  const auto V0 = pin->param()->get<double>("problem.params.v0", 2.629369);
  const auto D_L = pin->param()->get<double>("problem.params.rhoL", 3.857143);
  const auto P_L =
      pin->param()->get<double>("problem.params.pL", 10.333333333333);
  const auto P_R = pin->param()->get<double>("problem.params.pR", 1.0);

  auto &eos = mesh_state.eos();
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: ShuOsher (1)", DevExecSpace(), ib.s,
      ib.e, KOKKOS_LAMBDA(const int i) {
        const double X1 = grid->centers(i);

        if (X1 <= -4.0) {
          // Left state: constant values
          for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
            uPF(i, iNodeX, vars::prim::Rho) = D_L;
          }
        } else {
          // Right state: sinusoidal density
          for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
            const double x = grid->node_coordinate(i, iNodeX);
            uPF(i, iNodeX, vars::prim::Rho) = (1.0 + 0.2 * sin(5.0 * x));
          }
        }
      });

    auto tau_func = [&D_L](double x, int /*ix*/, int /*iN*/) -> double {
      if (x <= -4.0) {
        return 1.0 / D_L;
      }
      return 1.0 / (1.0 + 0.2 * sin(5.0 * x));
    };
    auto velocity_func = [&V0](double x, int /*ix*/, int /*iN*/) -> double {
      if (x <= -4.0) {
        return V0;
      }
      return 0.0;
    };
    auto energy_func = [&P_L, &P_R, &V0, &D_L, &gm1](double x, int /*ix*/,
                                                     int /*iN*/) -> double {
      if (x <= -4.0) {
        return (P_L / gm1) / D_L + 0.5 * V0 * V0;
      }
      const double rho = 1.0 + 0.2 * sin(5.0 * x);
      return (P_R / gm1) / rho;
    };

      static const IndexRange qb(nNodes);
      auto r = grid->nodal_grid();
      athelas::par_for(
          DEFAULT_LOOP_PATTERN, "Pgen :: ShuOsher", DevExecSpace(),
          ib.s, ib.e, qb.s, qb.e,
          KOKKOS_LAMBDA(const int i, const int q) {
            const double x = r(i, q+1);
            uCF(i, q, vars::cons::SpecificVolume) = tau_func(x, i, q);
            uCF(i, q, vars::cons::Velocity) = velocity_func(x, i, q);
            uCF(i, q, vars::cons::Energy) = energy_func(x, i, q);
          });
}

} // namespace athelas
