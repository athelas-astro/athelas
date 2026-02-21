/**
 * @file marshak.hpp
 * --------------
 *
 * @brief Radiation marshak wave test
 */

#pragma once

#include <cmath>

#include "basic_types.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"
#include "utils/constants.hpp"

namespace athelas {

/**
 * @brief Initialize radiating shock
 **/
void marshak_init(MeshState &mesh_state, GridStructure *grid, ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "marshak",
                   "Marshak requires marshak eos!");

  const bool rad_active = pin->param()->get<bool>("physics.rad_active");
  athelas_requires(rad_active, "Marshak requires radiation enabled!");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  const int nNodes = grid->n_nodes();
  static const IndexRange ib(grid->domain<Domain::Interior>());
  const IndexRange qb(nNodes);

  auto su_olson_energy = [&](const double alpha, const double T) {
    return (alpha / 4.0) * std::pow(T, 4.0);
  };

  const auto V0 = pin->param()->get<double>("problem.params.v0", 0.0);
  const auto rho0 = pin->param()->get<double>("problem.params.rho0", 10.0);
  const auto epsilon = pin->param()->get<double>("problem.params.epsilon", 1.0);
  const auto T0 = pin->param()->get<double>("problem.params.T0", 1.0e4); // K

  const double alpha = 4.0 * constants::a / epsilon;
  const double em_gas = su_olson_energy(alpha, T0) / rho0;

  // TODO(astrobarker): thread through
  const double e_rad = constants::a * std::pow(T0, 4.0);

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: Marshak", DevExecSpace(), ib.s, ib.e, qb.s,
      qb.e, KOKKOS_LAMBDA(const int i, const int q) {
        uCF(i, q, vars::cons::SpecificVolume) = 1.0 / rho0;
        uCF(i, q, vars::cons::Velocity) = V0;
        uCF(i, q, vars::cons::Energy) = em_gas + 0.5 * V0 * V0;
        uCF(i, q, vars::cons::RadEnergy) = e_rad / rho0;
        uCF(i, q, vars::cons::RadFlux) = 0.0;

        uPF(i, q, vars::prim::Rho) = rho0;
      });
}

} // namespace athelas
