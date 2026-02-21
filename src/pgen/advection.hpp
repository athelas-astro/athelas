/**
 * @file advection.hpp
 * --------------
 *
 * @brief Fluid advection test
 */

#pragma once

#include <cmath> /* sin */

#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"
#include "utils/constants.hpp"

namespace athelas {

/**
 * @brief Initialize advection test
 **/
void advection_init(MeshState &mesh_state, GridStructure *grid,
                    ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Advection requires ideal gas eos!");

  // Smooth advection problem
  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const int nNodes = grid->n_nodes();

  const auto V0 = pin->param()->get<double>("problem.params.v0", -1.0);
  const auto P0 = pin->param()->get<double>("problem.params.p0", 0.01);
  const auto Amp = pin->param()->get<double>("problem.params.amp", 1.0);

  const auto &eos = mesh_state.eos();
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Advection (1)", DevExecSpace(), ib.s,
      ib.e, KOKKOS_LAMBDA(const int i) {
        for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
          const double x = grid->node_coordinate(i, iNodeX);
          uPF(i, iNodeX, vars::prim::Rho) =
              (2.0 + Amp * sin(2.0 * constants::PI * x));
        }
      });

  auto density_func = [&Amp](double x, int /*ix*/, int /*iN*/) -> double {
    return 2.0 + Amp * sin(2.0 * constants::PI * x);
  };
  auto velocity_func = [&V0](double /*x*/, int /*ix*/, int /*iN*/) -> double {
    return V0;
  };
  auto energy_func = [&P0, &V0, &Amp, &gm1](double x, int /*ix*/,
                                            int /*iN*/) -> double {
    const double rho = 2.0 + Amp * sin(2.0 * constants::PI * x);
    return (P0 / gm1) / rho + 0.5 * V0 * V0;
  };

  static const IndexRange nb(nNodes);
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: Advection (nodal)", DevExecSpace(), ib.s,
      ib.e, nb.s, nb.e, KOKKOS_LAMBDA(const int i, const int node) {
        const double x = grid->node_coordinate(i, node);
        uCF(i, node, vars::cons::SpecificVolume) =
            1.0 / density_func(x, i, node);
        uCF(i, node, vars::cons::Velocity) = velocity_func(x, i, node);
        uCF(i, node, vars::cons::Energy) = energy_func(x, i, node);
      });
}

} // namespace athelas
