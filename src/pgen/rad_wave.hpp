/**
 * @file rad_wave.hpp
 * --------------
 *
 * @brief Radiation wave test
 */

#pragma once

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * @brief Initialize radiation wave test
 **/
void rad_wave_init(State *state, GridStructure *grid, ProblemIn *pin,
                   const eos::EOS *eos,
                   basis::ModalBasis * /*fluid_basis = nullptr*/,
                   basis::ModalBasis * /*radiation_basis = nullptr*/) {
  const bool rad_active = pin->param()->get<bool>("physics.rad_active");
  if (!rad_active) {
    THROW_ATHELAS_ERROR("Radiation wave requires radiation enabled!");
  }

  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Radiation wave requires ideal gas eos!");
  }

  AthelasArray3D<double> uCF = state->u_cf();
  AthelasArray3D<double> uPF = state->u_pf();

  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const int nNodes = grid->n_nodes();

  constexpr static int q_Tau = 0;
  constexpr static int q_V = 1;
  constexpr static int q_E = 2;

  constexpr static int iPF_D = 0;

  constexpr static int iCR_E = 3;

  const auto lambda = pin->param()->get<double>("problem.params.lambda", 0.1);
  const auto kappa = pin->param()->get<double>("problem.params.kappa", 1.0);
  const auto epsilon =
      pin->param()->get<double>("problem.params.epsilon", 1.0e-6);
  const auto rho0 = pin->param()->get<double>("problem.params.rho0", 1.0);
  const auto P0 = pin->param()->get<double>("problem.params.p0", 1.0e-6);

  // TODO(astrobarker): thread through
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: RadWave (1)", DevExecSpace(), ib.s,
      ib.e, KOKKOS_LAMBDA(const int i) {
        const int k = 0;
        const double X1 = grid->centers(i);

        uCF(i, k, q_Tau) = 1.0 / rho0;
        uCF(i, k, q_V) = 0.0;
        uCF(i, k, q_E) = (P0 / gm1) / rho0;
        uCF(i, k, iCR_E) = epsilon;

        for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
          uPF(i, iNodeX, iPF_D) = rho0;
        }
      });

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: RadWave (ghost)", DevExecSpace(), 0,
      ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          uPF(ib.s - 1 - i, iN, 0) = uPF(ib.s + i, (nNodes + 2) - iN - 1, 0);
          uPF(ib.s + 1 + i, iN, 0) = uPF(ib.s - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas
