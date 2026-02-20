#pragma once

#include <cmath>

#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"
#include "utils/constants.hpp"

namespace athelas {

/**
 * @brief Initialize smooth flow test problem
 **/
void smooth_flow_init(MeshState &mesh_state, GridStructure *grid,
                      ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Smooth flow requires ideal gas eos!");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  static const int nNodes = grid->n_nodes();
  static const IndexRange ib(grid->domain<Domain::Interior>());
  const IndexRange qb(nNodes);

  const auto amp =
      pin->param()->get<double>("problem.params.amp", 0.9999999999999999);

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: SmoothFlow (1)", DevExecSpace(), ib.s,
      ib.e, KOKKOS_LAMBDA(const int i) {
        for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
          const double x = grid->node_coordinate(i, iNodeX);
          uPF(i, iNodeX, vars::prim::Rho) =
              (1.0 + amp * sin(constants::PI * x));
        }
      });

  auto density_func = [&amp](double x, int /*ix*/, int /*iN*/) -> double {
    return 1.0 + amp * sin(constants::PI * x);
  };
  auto velocity_func = [](double /*x*/, int /*ix*/, int /*iN*/) -> double {
    return 0.0;
  };
  auto energy_func = [&amp](double x, int /*ix*/, int /*iN*/) -> double {
    const double D = 1.0 + amp * sin(constants::PI * x);
    return (D * D * D / 2.0) / D;
  };

  static const IndexRange nb(nNodes);
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: SmoothFlow (2)", DevExecSpace(), ib.s,
      ib.e, nb.s, nb.e, KOKKOS_LAMBDA(const int i, const int node) {
        const double x = grid->node_coordinate(i, node);
        uCF(i, node, vars::cons::SpecificVolume) =
            1.0 / density_func(x, i, node);
        uCF(i, node, vars::cons::Velocity) = velocity_func(x, i, node);
        uCF(i, node, vars::cons::Energy) = energy_func(x, i, node);
      });
}

} // namespace athelas
