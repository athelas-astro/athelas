/**
 * @file marshak.hpp
 * --------------
 *
 * @brief Radiation marshak wave test
 */

#pragma once

#include <cmath>

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"
#include "utils/constants.hpp"

namespace athelas {

/**
 * @brief Initialize radiating shock
 **/
void marshak_init(MeshState &mesh_state, GridStructure *grid, ProblemIn *pin,
                  bool /*first_init*/) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "marshak",
                   "Marshak requires marshak eos!");

  const bool rad_active = pin->param()->get<bool>("physics.rad_active");
  athelas_requires(rad_active, "Marshak requires radiation enabled!");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  static const IndexRange ib(grid->domain<Domain::Interior>());
  const int nNodes = grid->n_nodes();

  // TODO(astrobarker) move these to a namespace like constants
  constexpr static int q_Tau = 0;
  constexpr static int q_V = 1;
  constexpr static int q_E = 2;

  constexpr static int iPF_D = 0;

  constexpr static int iCR_E = 3;
  constexpr static int iCR_F = 4;

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
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Marshak (1)", DevExecSpace(), ib.s,
      ib.e, KOKKOS_LAMBDA(const int i) {
        const int k = 0;

        uCF(i, k, q_Tau) = 1.0 / rho0;
        uCF(i, k, q_V) = V0;
        uCF(i, k, q_E) = em_gas + 0.5 * V0 * V0;
        uCF(i, k, iCR_E) = e_rad;
        uCF(i, k, iCR_F) = 0.0;

        for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
          uPF(i, iNodeX, iPF_D) = rho0;
        }
      });

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Marshak (ghost)", DevExecSpace(), 0,
      ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          uPF(ib.s - 1 - i, iN, 0) = uPF(ib.s + i, (nNodes + 2) - iN - 1, 0);
          uPF(ib.s + 1 + i, iN, 0) = uPF(ib.s - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas
