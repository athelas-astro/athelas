#pragma once

#include <cmath>

#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * @file ejecta_csm.hpp
 * --------------
 *
 * @brief Ejecta - CSM interaction test.
 * See Duffell 2016 (doi:10.3847/0004-637X/821/2/76)
 */
void ejecta_csm_init(MeshState &mesh_state, GridStructure *grid, ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Shu-Osher requires ideal gas eos!");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  static const int nNodes = grid->n_nodes();
  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const IndexRange qb(nNodes);

  const auto rstar = pin->param()->get<double>("problem.params.rstar", 0.01);
  const auto vmax =
      pin->param()->get<double>("problem.params.vmax", std::sqrt(10.0 / 3.0));

  const double rstar3 = rstar * rstar * rstar;

  const auto &eos = mesh_state.eos();
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: EjectaCSM (1)", DevExecSpace(), ib.s,
      ib.e, KOKKOS_LAMBDA(const int i) {
        for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
          const double x = grid->node_coordinate(i, iNodeX);
          if (x <= rstar) {
            uPF(i, iNodeX + 1, vars::prim::Rho) =
                1.0 / (constants::FOURPI * rstar3 / 3.0);
          } else {
            uPF(i, iNodeX + 1, vars::prim::Rho) = 1.0;
          }
        }
      });

  auto r = grid->nodal_grid();
    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "Pgen :: EjectaCSM (2)", DevExecSpace(),
        ib.s, ib.e, qb.s, qb.e, KOKKOS_LAMBDA(const int i, const int q) {
          const double X1 = r(i, q + 1);

          if (X1 <= rstar) {
            const double rho = 1.0 / (constants::FOURPI * rstar3 / 3.0);
            const double pressure = (1.0e-5) * rho * vmax * vmax;
            const double vel = vmax * (X1 / rstar);
            uCF(i, q, vars::cons::SpecificVolume) = 1.0 / rho;
            uCF(i, q, vars::cons::Velocity) = vel;
            uCF(i, q, vars::cons::Energy) = (pressure / gm1 / rho) + 0.5 * vel * vel;
          } else {
            const double rho = 1.0;
            const double pressure = (1.0e-5) * rho * vmax * vmax;
            uCF(i, q, vars::cons::SpecificVolume) = 1.0 / rho;
            uCF(i, q, vars::cons::Velocity) = 0.0;
            uCF(i, q, vars::cons::Energy) = (pressure / gm1 / rho);
          }
        });
}

} // namespace athelas
