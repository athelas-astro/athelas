#pragma once

#include <cmath>

#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * Initialize equilibrium rad test
 **/
void rad_equilibrium_init(MeshState &mesh_state, GridStructure *grid,
                          ProblemIn *pin) {
  const bool rad_active = pin->param()->get<bool>("physics.rad_active");
  athelas_requires(rad_active,
                   "Radiation equilibriation requires radiation enabled!");
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Radiation equilibriation requires ideal gas eos!");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  static const int nNodes = grid->n_nodes();
  static const IndexRange ib(grid->domain<Domain::Interior>());
  const IndexRange qb(nNodes);

  const auto V0 = pin->param()->get<double>("problem.params.v0", 0.0);
  const auto logD = pin->param()->get<double>("problem.params.logrho", -7.0);
  const auto logE_gas =
      pin->param()->get<double>("problem.params.logE_gas", 10.0);
  const auto logE_rad =
      pin->param()->get<double>("problem.params.logE_rad", 12.0);

  const double D = std::pow(10.0, logD);
  const double Ev_gas = std::pow(10.0, logE_gas);
  const double Ev_rad = std::pow(10.0, logE_rad);

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: RadEquilibrium", DevExecSpace(), ib.s,
      ib.e, qb.s, qb.e, KOKKOS_LAMBDA(const int i, const int q) {
        uCF(i, q, vars::cons::SpecificVolume) = 1.0 / D;
        uCF(i, q, vars::cons::Velocity) = V0;
        uCF(i, q, vars::cons::Energy) = Ev_gas / D;
        uCF(i, q, vars::cons::RadEnergy) = Ev_rad / D;

        uPF(i, q, vars::prim::Rho) = D;
      });
}

} // namespace athelas
