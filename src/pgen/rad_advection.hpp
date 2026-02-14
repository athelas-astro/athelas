#pragma once

#include <cmath>

#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"
#include "utils/constants.hpp"

namespace athelas {

/**
 * @brief Initialize radiation advection test
 * @note EXPERIMENTAL
 **/
void rad_advection_init(MeshState &mesh_state, GridStructure *grid,
                        ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Radiation advection requires ideal gas eos!");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  static const int nNodes = grid->n_nodes();
  static const IndexRange ib(grid->domain<Domain::Interior>());
  const IndexRange qb(nNodes);

  const auto V0 = pin->param()->get<double>("problem.params.v0", 1.0);
  const auto D = pin->param()->get<double>("problem.params.rho", 1.0);
  const auto amp = pin->param()->get<double>("problem.params.amp", 1.0);
  const auto width = pin->param()->get<double>("problem.params.width", 0.05);
  const double mu = 1.0 + constants::m_e / constants::m_p;
  auto &eos = mesh_state.eos();
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: RadAdvection", DevExecSpace(),
      ib.s, ib.e, qb.s, qb.e, KOKKOS_LAMBDA(const int i, const int q) {
        const double X1 = grid->centers(i);

        uCF(i, q, vars::cons::RadEnergy) =
            amp * std::max(std::exp(-std::pow((X1 - 0.5) / width, 2.0) / 2.0),
                           1.0e-8);
        uCF(i, q, vars::cons::RadFlux) = 1.0 * constants::c_cgs * uCF(i, q, vars::cons::RadEnergy);

        const double Trad = std::pow(uCF(i, q, vars::cons::RadEnergy) / constants::a, 0.25);
        const double sie_fluid =
            constants::k_B * Trad / (gm1 * mu * constants::m_p);
        uCF(i, q, vars::cons::SpecificVolume) = 1.0 / D;
        uCF(i, q, vars::cons::Velocity) = V0;
        uCF(i, q, vars::cons::Energy) =
            sie_fluid +
            0.5 * V0 * V0; // p0 / (gamma - 1.0) / D + 0.5 * V0 * V0;

        uPF(i, q, vars::prim::Rho) = D;
      });
}

} // namespace athelas
