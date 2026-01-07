/**
 * @file shu_osher.hpp
 * --------------
 *
 * @brief Shu Osher shock tube
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
 * @brief Initialize Shu Osher hydro test
 **/
void shu_osher_init(MeshState &mesh_state, GridStructure *grid, ProblemIn *pin,
                    const eos::EOS *eos,
                    basis::ModalBasis *fluid_basis = nullptr) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Shu Osher requires ideal gas eos!");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const int nNodes = grid->n_nodes();

  constexpr static int q_Tau = 0;
  constexpr static int q_V = 1;
  constexpr static int q_E = 2;

  constexpr static int iPF_D = 0;

  const auto V0 = pin->param()->get<double>("problem.params.v0", 2.629369);
  const auto D_L = pin->param()->get<double>("problem.params.rhoL", 3.857143);
  const auto P_L =
      pin->param()->get<double>("problem.params.pL", 10.333333333333);
  const auto P_R = pin->param()->get<double>("problem.params.pR", 1.0);

  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  // Phase 1: Initialize nodal values (always done)
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: ShuOsher (1)", DevExecSpace(), ib.s,
      ib.e, KOKKOS_LAMBDA(const int i) {
        const double X1 = grid->centers(i);

        if (X1 <= -4.0) {
          // Left state: constant values
          for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
            uPF(i, iNodeX, iPF_D) = D_L;
          }
        } else {
          // Right state: sinusoidal density
          for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
            const double x = grid->node_coordinate(i, iNodeX);
            uPF(i, iNodeX, iPF_D) = (1.0 + 0.2 * sin(5.0 * x));
          }
        }
      });

  // Phase 2: Initialize modal coefficients
  if (fluid_basis != nullptr) {
    // Use L2 projection for accurate modal coefficients
    auto tau_func = [&D_L](double x, int /*ix*/, int /*iN*/) -> double {
      if (x <= -4.0) {
        return 1.0 / D_L;
      }
      return 1.0 / (1.0 + 0.2 * sin(5.0 * x));
    };

    auto velocity_func = [&V0](double x, int /*ix*/, int /*iN*/) -> double {
      if (x <= -4.0) {
        return V0;
      }
      return 0.0;
    };

    auto energy_func = [&P_L, &P_R, &V0, &D_L, &gm1](double x, int /*ix*/,
                                                     int /*iN*/) -> double {
      if (x <= -4.0) {
        return (P_L / gm1) / D_L + 0.5 * V0 * V0;
      }
      const double rho = 1.0 + 0.2 * sin(5.0 * x);
      return (P_R / gm1) / rho;
    };

    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: ShuOsher (2)", DevExecSpace(), ib.s,
        ib.e, KOKKOS_LAMBDA(const int i) {
          const int k = 0;
          const double X1 = grid->centers(i);

          if (X1 <= -4.0) {
            uCF(i, k, q_Tau) = 1.0 / D_L;
            uCF(i, k, q_V) = V0;
            uCF(i, k, q_E) = (P_L / gm1) * uCF(i, k, q_Tau) + 0.5 * V0 * V0;
          } else {
            // Project each conserved variable
            fluid_basis->project_nodal_to_modal(uCF, uPF, grid, q_Tau, i,
                                                tau_func);
            fluid_basis->project_nodal_to_modal(uCF, uPF, grid, q_V, i,
                                                velocity_func);
            fluid_basis->project_nodal_to_modal(uCF, uPF, grid, q_E, i,
                                                energy_func);
          }
        });

  } else {
    // Fallback: set cell averages only (k=0)
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: ShuOsher (2)", DevExecSpace(), ib.s,
        ib.e, KOKKOS_LAMBDA(const int i) {
          const int k = 0;
          const double X1 = grid->centers(i);

          if (X1 <= -4.0) {
            uCF(i, k, q_Tau) = 1.0 / D_L;
            uCF(i, k, q_V) = V0;
            uCF(i, k, q_E) = (P_L / gm1) * uCF(i, k, q_Tau) + 0.5 * V0 * V0;
          } else {
            uCF(i, k, q_Tau) = 1.0 / (1.0 + 0.2 * sin(5.0 * X1));
            uCF(i, k, q_V) = 0.0;
            uCF(i, k, q_E) = (P_R / gm1) * uCF(i, k, q_Tau);
          }
        });
  }

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: ShuOsher (ghost)", DevExecSpace(), 0,
      ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          uPF(ib.s - 1 - i, iN, 0) = uPF(ib.s + i, (nNodes + 2) - iN - 1, 0);
          uPF(ib.s + 1 + i, iN, 0) = uPF(ib.s - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas
