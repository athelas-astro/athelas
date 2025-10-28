/**
 * @file progenitor.hpp
 * --------------
 *
 * @brief Supernova progenitor initialization
 */

#pragma once

#include <cmath>

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
// #include "io/parser.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * @brief Initialize supernova progenitor
 **/
void progenitor_init(State *state, GridStructure *grid, ProblemIn *pin,
                     const eos::EOS *eos,
                     basis::ModalBasis *fluid_basis = nullptr) {
  if (pin->param()->get<std::string>("eos.type") != "paczynski") {
    THROW_ATHELAS_ERROR("Problem 'supernova' requires paczynski eos!");
  }

  if (!pin->param()->get<bool>("physics.composition_enabled")) {
    THROW_ATHELAS_ERROR("Problem 'supernova' requires composition enabled!!");
  }

  if (!pin->param()->get<bool>("physics.gravity_active")) {
    THROW_ATHELAS_ERROR("Problem 'supernova' requires gravity enabled!!");
  }

  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const int nNodes = grid->n_nodes();
  static const int order = nNodes;
  const auto ncomps =
      pin->param()->get<int>("composition.ncomps"); // mass fractions

  auto uCF = state->u_cf();
  auto uPF = state->u_pf();
  auto uAF = state->u_af();

  std::shared_ptr<atom::CompositionData> comps =
      std::make_shared<atom::CompositionData>(grid->n_elements() + 2, order,
                                              ncomps);

  auto mass_fractions = state->mass_fractions();
  auto charges = comps->charge();
  auto neutrons = comps->neutron_number();
  auto ye = comps->ye();
  auto *species_indexer = comps->species_indexer();

  const auto fn_hydro =
      pin->param()->get<std::string>("problem.params.fn_hydro");

  // --- read in hydro data ---
  auto hydro_data = io::Parser::parse_file(fn_hydro, ' ');

  auto [radius, density, velocity, pressure] =
      get_columns_by_indices<double, double, double, double>(*hydro_data, 0, 1,
                                                             2, 3);

  const size_t n_zones_prog = radius.size();

  std::println("nzones mesa {}", n_zones_prog);

  for (int i = 0; i < n_zones_prog; ++i) {
    std::println("rho({}) = {}", i, density[i]);
  }

  if (fluid_basis == nullptr) {
    // Phase 1: Initialize nodal values (always done)
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: HydrostaticBalance (1)",
        DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
            // todo: get x1, x2, y1, y2, x, for LINTERP
            uPF(i, iNodeX, vars::prim::Rho) = density[0];
          }
        });
  }

  //  // Phase 2: Initialize modal coefficients
  //  if (fluid_basis != nullptr) {
  //    // Use L2 projection for accurate modal coefficients
  //    auto tau_func = [&](double /*x*/, int ix, int iN) -> double {
  //      return 1.0 / rho_from_p(uAF(ix, iN, 0));
  //    };
  //
  //    auto velocity_func = [](double /*x*/, int /*ix*/, int /*iN*/) -> double
  //    {
  //      return 0.0;
  //    };
  //
  //    auto energy_func = [&](double /*x*/, int ix, int iN) -> double {
  //      const double rho = rho_from_p(uAF(ix, iN, 0));
  //      return (uAF(ix, iN, 0) / gm1) / rho;
  //    };
  //
  //    athelas::par_for(
  //        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: HydrostaticBalance (2)",
  //        DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
  //          // Project each conserved variable
  //          fluid_basis->project_nodal_to_modal(uCF, uPF, grid, q_Tau, i,
  //                                              tau_func);
  //          fluid_basis->project_nodal_to_modal(uCF, uPF, grid, q_V, i,
  //                                              velocity_func);
  //          fluid_basis->project_nodal_to_modal(uCF, uPF, grid, q_E, i,
  //                                              energy_func);
  //        });
  //  }
  //
  //  // Fill density in guard cells
  //  athelas::par_for(
  //      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: HydrostaticBalance (ghost)",
  //      DevExecSpace(), 0, ib.s - 1, KOKKOS_LAMBDA(const int i) {
  //        for (int iN = 0; iN < nNodes + 2; iN++) {
  //          uPF(ib.s - 1 - i, iN, 0) = uPF(ib.s + i, (nNodes + 2) - iN - 1,
  //          0); uPF(ib.s + 1 + i, iN, 0) = uPF(ib.s - i, (nNodes + 2) - iN -
  //          1, 0);
  //        }
  //      });
}

} // namespace athelas
