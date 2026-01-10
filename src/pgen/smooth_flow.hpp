/**
 * @file smooth_flow.hpp
 * --------------
 *
 * @brief Smooth flow test
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
 * @brief Initialize smooth flow test problem
 **/
void smooth_flow_init(MeshState &mesh_state, GridStructure *grid,
                      ProblemIn *pin, bool first_init) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Smooth flow requires ideal gas eos!");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const int nNodes = grid->n_nodes();

  constexpr static int q_Tau = 0;
  constexpr static int q_V = 1;
  constexpr static int q_E = 2;

  constexpr static int iPF_D = 0;

  const auto amp =
      pin->param()->get<double>("problem.params.amp", 0.9999999999999999);

  // Phase 1: Initialize nodal values (always done)
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: SmoothFlow (1)", DevExecSpace(), ib.s,
      ib.e, KOKKOS_LAMBDA(const int i) {
        for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
          const double x = grid->node_coordinate(i, iNodeX);
          uPF(i, iNodeX, iPF_D) = (1.0 + amp * sin(constants::PI * x));
        }
      });

  // Phase 2: Initialize modal coefficients
  if (!first_init) {
    const auto &fluid_basis = mesh_state.fluid_basis();
    // Use L2 projection for accurate modal coefficients
    auto density_func = [&amp](double x, int /*ix*/, int /*iN*/) -> double {
      return 1.0 + amp * sin(constants::PI * x);
    };

    auto velocity_func = [](double /*x*/, int /*ix*/, int /*iN*/) -> double {
      return 0.0;
    };

    auto energy_func = [&amp](double x, int /*ix*/, int /*iN*/) -> double {
      const double D = 1.0 + amp * sin(constants::PI * x);
      return (D * D * D / 2.0) / D;
    };

    // Project each conserved variable using Kokkos parallel for
    fluid_basis.project_nodal_to_modal_all_cells(uCF, uPF, grid, q_Tau,
                                                 density_func);
    fluid_basis.project_nodal_to_modal_all_cells(uCF, uPF, grid, q_V,
                                                 velocity_func);
    fluid_basis.project_nodal_to_modal_all_cells(uCF, uPF, grid, q_E,
                                                 energy_func);
  } else {
    // Fallback: set cell averages only (k=0)
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: SmoothFlow (2)", DevExecSpace(),
        ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          const int k = 0;
          const double X1 = grid->centers(i);

          const double D = (1.0 + (amp * sin(constants::PI * X1)));
          uCF(i, k, q_Tau) = 1.0 / D;
          uCF(i, k, q_V) = 0.0;
          uCF(i, k, q_E) = (D * D * D / 2.0) * uCF(i, k, q_Tau);
        });
  }

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: SmoothFlow (ghost)", DevExecSpace(),
      0, ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          uPF(ib.s - 1 - i, iN, 0) = uPF(ib.s + i, (nNodes + 2) - iN - 1, 0);
          uPF(ib.s + 1 + i, iN, 0) = uPF(ib.s - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas
