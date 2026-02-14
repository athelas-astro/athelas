#pragma once

#include <cmath>

#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "solvers/hydrostatic_equilibrium.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * @brief Initialize hydrostatic balance self gravity test
 **/
void hydrostatic_balance_init(MeshState &mesh_state, GridStructure *grid,
                              ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "polytropic",
                   "Hydrostatic balance requires polytropic eos!");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");
  auto uAF = mesh_state(0).get_field("u_af");

  static const IndexRange ib(grid->domain<Domain::Interior>());
  const int nNodes = grid->n_nodes();

  const auto rho_c = pin->param()->get<double>("problem.params.rho_c", 1.0e8);
  const auto p_thresh =
      pin->param()->get<double>("problem.params.p_threshold", 1.0e-10);

  const auto polytropic_k = pin->param()->get<double>("eos.k");
  const auto polytropic_n = pin->param()->get<double>("eos.n");

  const auto &eos = mesh_state.eos();
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  auto rho_from_p = [&polytropic_k, &polytropic_n](const double p) -> double {
    return std::pow(p / polytropic_k, polytropic_n / (polytropic_n + 1.0));
  };

    auto solver = HydrostaticEquilibrium(rho_c, p_thresh,
                                         pin->param()->get<double>("eos.k"),
                                         pin->param()->get<double>("eos.n"));
    solver.solve(mesh_state, grid, pin);

    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: HydrostaticBalance (1)",
        DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
            uPF(i, iNodeX, vars::prim::Rho) =
                rho_from_p(uAF(i, iNodeX, vars::prim::Rho));
          }
        });

    auto tau_func = [&](double /*x*/, int ix, int iN) -> double {
      return 1.0 / rho_from_p(uAF(ix, iN, 0));
    };
    auto velocity_func = [](double /*x*/, int /*ix*/, int /*iN*/) -> double {
      return 0.0;
    };
    auto energy_func = [&](double /*x*/, int ix, int iN) -> double {
      const double rho = rho_from_p(uAF(ix, iN, 0));
      return (uAF(ix, iN, 0) / gm1) / rho;
    };

      static const IndexRange qb(nNodes);
      athelas::par_for(
          DEFAULT_LOOP_PATTERN, "Pgen :: HydrostaticBalance (2)",
          DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
          KOKKOS_LAMBDA(const int i, const int q) {
            const int iN = q + 1; // uAF interior index
            uCF(i, q, vars::cons::SpecificVolume) = tau_func(0.0, i, iN);
            uCF(i, q, vars::cons::Velocity) = velocity_func(0.0, i, iN);
            uCF(i, q, vars::cons::Energy) = energy_func(0.0, i, iN);
          });
}

} // namespace athelas
