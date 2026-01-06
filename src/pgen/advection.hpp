/**
 * @file advection.hpp
 * --------------
 *
 * @brief Fluid advection test
 */

#pragma once

#include <cmath> /* sin */

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"
#include "utils/constants.hpp"

namespace athelas {

/**
 * @brief Initialize advection test
 **/
void advection_init(MeshState &mesh_state, GridStructure *grid, ProblemIn *pin,
                    const eos::EOS *eos,
                    basis::ModalBasis *fluid_basis = nullptr) {
  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("Advection requires ideal gas eos!");
  }

  // Smooth advection problem
  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const int nNodes = grid->n_nodes();

  const int q_Tau = 0;
  const int q_V = 1;
  const int q_E = 2;

  const int iPF_D = 0;

  const auto V0 = pin->param()->get<double>("problem.params.v0", -1.0);
  const auto P0 = pin->param()->get<double>("problem.params.p0", 0.01);
  const auto Amp = pin->param()->get<double>("problem.params.amp", 1.0);

  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  // Phase 1: Initialize nodal values (always done)
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Advection (1)", DevExecSpace(), ib.s,
      ib.e, KOKKOS_LAMBDA(const int i) {
        for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
          const double x = grid->node_coordinate(i, iNodeX);
          uPF(i, iNodeX, iPF_D) = (2.0 + Amp * sin(2.0 * constants::PI * x));
        }
      });

  // Phase 2: Initialize modal coefficients
  if (fluid_basis != nullptr) {
    // Use L2 projection for accurate modal coefficients
    auto density_func = [&Amp](double x, int /*ix*/, int /*iN*/) -> double {
      return 2.0 + Amp * sin(2.0 * constants::PI * x);
    };

    auto velocity_func = [&V0](double /*x*/, int /*ix*/, int /*iN*/) -> double {
      return V0;
    };

    auto energy_func = [&P0, &V0, &Amp, &gm1](double x, int /*ix*/,
                                              int /*iN*/) -> double {
      const double rho = 2.0 + Amp * sin(2.0 * constants::PI * x);
      return (P0 / gm1) / rho + 0.5 * V0 * V0;
    };

    // L2 projection onto modal basis
    fluid_basis->project_nodal_to_modal_all_cells(uCF, uPF, grid, q_Tau,
                                                  density_func);
    fluid_basis->project_nodal_to_modal_all_cells(uCF, uPF, grid, q_V,
                                                  velocity_func);
    fluid_basis->project_nodal_to_modal_all_cells(uCF, uPF, grid, q_E,
                                                  energy_func);
  } else {
    // Fallback: set cell averages only (k=0)
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Advection (2)", DevExecSpace(),
        ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          const int k = 0;
          const double X1 = grid->centers(i);

          uCF(i, k, q_Tau) = 1.0 / (2.0 + Amp * sin(2.0 * constants::PI * X1));
          uCF(i, k, q_V) = V0;
          uCF(i, k, q_E) = (P0 / gm1) * uCF(i, k, q_Tau) + 0.5 * V0 * V0;
        });
  }

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Advection (ghost)", DevExecSpace(), 0,
      ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          uPF(ib.s - 1 - i, iN, 0) = uPF(ib.s + i, (nNodes + 2) - iN - 1, 0);
          uPF(ib.e + 1 + i, iN, 0) = uPF(ib.e - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas
