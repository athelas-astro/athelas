/**
 * @file rad_equilibrium.hpp
 * --------------
 *
 * @brief Radiation fluid equilibriation test
 */

#pragma once

#include <cmath>

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * Initialize equilibrium rad test
 **/
void rad_equilibrium_init(MeshState &mesh_state, GridStructure *grid,
                          ProblemIn *pin, const eos::EOS * /*eos*/,
                          basis::ModalBasis * /*fluid_basis = nullptr*/,
                          basis::ModalBasis * /*radiation_basis = nullptr*/) {
  const bool rad_active = pin->param()->get<bool>("physics.rad_active");
  if (!rad_active) {
    THROW_ATHELAS_ERROR("Radiation equilibriation requires radiation enabled!");
  }
  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Radiation equilibriation requires ideal gas eos!");
  }

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const int nNodes = grid->n_nodes();

  const int q_Tau = 0;
  const int q_V = 1;
  const int q_E = 2;

  const int iPF_D = 0;

  const int iCR_E = 3;

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
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: RadEquilibrium (1)", DevExecSpace(),
      ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
        const int k = 0;

        uCF(i, k, q_Tau) = 1.0 / D;
        uCF(i, k, q_V) = V0;
        uCF(i, k, q_E) = Ev_gas / D;
        uCF(i, k, iCR_E) = Ev_rad;

        for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
          uPF(i, iNodeX, iPF_D) = D;
        }
      });

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: RadEquilibrium (ghost)",
      DevExecSpace(), 0, ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          uPF(ib.s - 1 - i, iN, 0) = uPF(ib.s + i, (nNodes + 2) - iN - 1, 0);
          uPF(ib.s + 1 + i, iN, 0) = uPF(ib.s - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas
