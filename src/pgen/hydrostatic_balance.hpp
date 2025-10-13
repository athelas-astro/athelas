/**
 * @file hydrostatic_balance.hpp
 * --------------
 *
 * @brief Hydrostatic balance test.
 */

#pragma once

#include <cmath>

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "solvers/hydrostatic_equilibrium.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * @brief Initialize Shu Osher hydro test
 **/
void hydrostatic_balance_init(State *state, GridStructure *grid, ProblemIn *pin,
                              const eos::EOS *eos,
                              basis::ModalBasis *fluid_basis = nullptr) {
  if (pin->param()->get<std::string>("eos.type") != "polytropic") {
    THROW_ATHELAS_ERROR("Hydrostatic balance requires polytropic eos!");
  }

  AthelasArray3D<double> uCF = state->u_cf();
  AthelasArray3D<double> uPF = state->u_pf();
  AthelasArray3D<double> uAF = state->u_af();

  static const IndexRange ib(grid->domain<Domain::Interior>());
  const int nNodes = grid->n_nodes();

  constexpr static int q_Tau = 0;
  constexpr static int q_V = 1;
  constexpr static int q_E = 2;

  constexpr static int iPF_D = 0;

  const auto rho_c = pin->param()->get<double>("problem.params.rho_c", 1.0e8);
  const auto p_thresh =
      pin->param()->get<double>("problem.params.p_threshold", 1.0e-10);

  const auto polytropic_k = pin->param()->get<double>("eos.k");
  const auto polytropic_n = pin->param()->get<double>("eos.n");

  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  auto rho_from_p = [&polytropic_k, &polytropic_n](const double p) -> double {
    return std::pow(p / polytropic_k, polytropic_n / (polytropic_n + 1.0));
  };

  if (fluid_basis == nullptr) {
    auto solver = HydrostaticEquilibrium(rho_c, p_thresh, eos,
                                         pin->param()->get<double>("eos.k"),
                                         pin->param()->get<double>("eos.n"));
    solver.solve(state, grid, pin);

    // Phase 1: Initialize nodal values (always done)
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: HydrostaticBalance (1)",
        DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
            uPF(i, iNodeX, iPF_D) = rho_from_p(uAF(i, iNodeX, 0));
          }
        });
  }

  // Phase 2: Initialize modal coefficients
  if (fluid_basis != nullptr) {
    // Use L2 projection for accurate modal coefficients
    auto tau_func = [&](double /*x*/, int ix, int iN) -> double {
      return 1.0 / rho_from_p(uAF(ix, iN, 0));
    };

    auto velocity_func = [](double /*x*/, int /*ix*/, int /*iN*/) -> double {
      return 0.0;
    };

    auto energy_func = [&](double /*x*/, int ix, int iN) -> double {
      const double rho = rho_from_p(uAF(ix, iN, 0));
      return (uAF(ix, iN, 0) / gm1) / rho;
    };

    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: HydrostaticBalance (2)",
        DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          // Project each conserved variable
          fluid_basis->project_nodal_to_modal(uCF, uPF, grid, q_Tau, i,
                                              tau_func);
          fluid_basis->project_nodal_to_modal(uCF, uPF, grid, q_V, i,
                                              velocity_func);
          fluid_basis->project_nodal_to_modal(uCF, uPF, grid, q_E, i,
                                              energy_func);
        });
  }

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: HydrostaticBalance (ghost)",
      DevExecSpace(), 0, ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          uPF(ib.s - 1 - i, iN, 0) = uPF(ib.s + i, (nNodes + 2) - iN - 1, 0);
          uPF(ib.s + 1 + i, iN, 0) = uPF(ib.s - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas
