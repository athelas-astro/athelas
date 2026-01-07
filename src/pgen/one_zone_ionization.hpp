/**
 * @file one_zone_ionization.hpp
 * --------------
 *
 * @brief One zone ionization test
 */

#pragma once

#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions.hpp"
#include "composition/composition.hpp"
#include "composition/saha.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * Initialize one_zone_ionization test
 **/
void one_zone_ionization_init(MeshState &mesh_state, GridStructure *grid,
                              ProblemIn *pin, const eos::EOS *eos,
                              basis::ModalBasis *fluid_basis = nullptr) {
  const bool ionization_active =
      pin->param()->get<bool>("physics.ionization_enabled");
  const int saha_ncomps =
      pin->param()->get<int>("ionization.ncomps"); // for ionization
  const auto ncomps =
      pin->param()->get<int>("composition.ncomps", 3); // mass fractions
  if (ncomps != 3) {
    throw_athelas_error("One zone ionization requires ncomps = 3");
  }
  if (!ionization_active) {
    throw_athelas_error("One zone ionization requires ionization enabled!");
  }
  const auto eos_type = pin->param()->get<std::string>("eos.type");
  //  if (pin->param()->get<std::string>("eos.type") != "ideal") {
  //    throw_athelas_error("One zone ionization requires ideal gas eos!");
  //  }
  // Don't try to track ionization for more species than we use.
  // We will track ionization for the first saha_ncomps species
  if (saha_ncomps != 3) {
    throw_athelas_error("One zone ionization requires [ionization.ncomps] = "
                        "[composition.ncomps] = 3!");
  }

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");
  auto uAF = mesh_state(0).get_field("u_af");
  auto sd0 = mesh_state(0);

  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const int nNodes = grid->n_nodes();
  static const int order = mesh_state.p_order();

  const auto temperature =
      pin->param()->get<double>("problem.params.temperature", 5800); // K
  const auto rho =
      pin->param()->get<double>("problem.params.rho", 1000.0); // g/cc
  const double vel = 0.0;
  const double tau = 1.0 / rho;

  const auto fn_ionization =
      pin->param()->get<std::string>("ionization.fn_ionization");
  const auto fn_deg =
      pin->param()->get<std::string>("ionization.fn_degeneracy");

  if (temperature <= 0.0 || rho <= 0.0) {
    throw_athelas_error("Temperature and denisty must be positive definite!");
  }

  std::shared_ptr<atom::CompositionData> comps =
      std::make_shared<atom::CompositionData>(grid->n_elements() + 2, order,
                                              ncomps);
  std::shared_ptr<atom::IonizationState> ionization_state =
      std::make_shared<atom::IonizationState>(grid->n_elements() + 2, nNodes,
                                              ncomps, 7, saha_ncomps,
                                              fn_ionization, fn_deg);
  auto mass_fractions = mesh_state.mass_fractions("u_cf");
  auto charges = comps->charge();
  auto neutrons = comps->neutron_number();
  auto inv_atomic_mass = comps->inverse_atomic_mass();
  auto ionization_states = ionization_state->ionization_fractions();
  auto zbar = ionization_state->zbar();

  // Set up ball of hydrogen, helium, carbon.
  constexpr int i_H = 0;
  constexpr int i_He = 1;
  constexpr int i_C = 2;
  constexpr double X_H = 0.75;
  constexpr double X_He = 0.23;
  constexpr double X_C = 0.02;
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: OneZoneIonization :: nodal",
      DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
        uCF(i, vars::modes::CellAverage, vars::cons::SpecificVolume) =
            1.0 / rho;
        for (int q = 0; q < nNodes + 2; q++) {
          uPF(i, q, vars::prim::Rho) = rho;
          uAF(i, q, vars::aux::Tgas) = temperature;
          // Set Zbar assuming full ionization -- used as guess in Saha below.
          zbar(i, q, i_H) = 1;
          zbar(i, q, i_He) = 2;
          zbar(i, q, i_C) = 6;
        }

        mass_fractions(i, vars::modes::CellAverage, i_H) = X_H;
        charges(i_H) = 1;
        neutrons(i_H) = 0;
        inv_atomic_mass(i_H) = 1.0;

        mass_fractions(i, vars::modes::CellAverage, i_He) = X_He;
        charges(i_He) = 2;
        neutrons(i_He) = 2;
        inv_atomic_mass(i_He) = 1.0 / 4.0;

        mass_fractions(i, vars::modes::CellAverage, i_C) = X_C;
        charges(i_C) = 6;
        neutrons(i_C) = 6;
        inv_atomic_mass(i_C) = 1.0 / 12.0;
      });
  mesh_state.setup_composition(comps);
  mesh_state.setup_ionization(ionization_state);

  if (fluid_basis != nullptr) {
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: OneZoneIonization (1)",
        DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          double lambda[8];
          const int k = vars::modes::CellAverage;

          uCF(i, k, vars::cons::SpecificVolume) = tau;
          uCF(i, k, vars::cons::Velocity) = vel;

          for (int q = 0; q < nNodes + 2; q++) {
            uPF(i, q, vars::prim::Rho) = rho;
            uAF(i, q, vars::aux::Tgas) = temperature;
            if (eos_type == "paczynski") {
              atom::paczynski_terms(sd0, i, q, lambda);
            }
            uAF(i, q, vars::aux::Pressure) = pressure_from_density_temperature(
                eos, rho, temperature, lambda);
            uPF(i, q, vars::prim::Sie) = sie_from_density_pressure(
                eos, rho, uAF(i, q, vars::aux::Pressure), lambda);
          }
        });

    auto mkk = fluid_basis->mass_matrix();
    auto phi = fluid_basis->phi();
    auto weights = grid->weights();
    auto dr = grid->widths();
    auto sqrt_gm = grid->sqrt_gm();
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: OneZoneIonization :: Project sie",
        DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          // Project the nodal representation to a modal one
          // Compute L2 projection: <f_q, phi_k> / <phi_k, phi_k>
          for (int k = 0; k < order; k++) {
            double numerator = 0.0;
            const double denominator = mkk(i, k);

            // Compute <f_q, phi_k>
            for (int q = 0; q < nNodes; q++) {
              const double nodal_val = uPF(i, q, vars::prim::Sie);
              const double rho = uPF(i, q + 1, vars::prim::Rho);

              numerator += nodal_val * phi(i, q + 1, k) * weights(q) * dr(i) *
                           sqrt_gm(i, q + 1) * rho;
            }
            uCF(i, k, vars::cons::Energy) = numerator / denominator;
            // We apply a simple exponential filter to modes
            if (k > 0) {
              uCF(i, k, vars::cons::Energy) *= std::exp(-k);
            }
          }
        });

    atom::fill_derived_comps<Domain::Interior>(sd0, uCF, grid, fluid_basis);
    atom::solve_saha_ionization<Domain::Interior>(sd0, uCF, *grid, *eos,
                                                  *fluid_basis);
    atom::fill_derived_ionization<Domain::Interior>(sd0, uCF, grid,
                                                    fluid_basis);
    // composition boundary condition
    static const IndexRange vb_comps(std::make_pair(3, 3 + ncomps - 1));
    bc::fill_ghost_zones_composition(uCF, vb_comps);
  }

  // Fill density and temperature in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: OneZoneIonization (ghost)",
      DevExecSpace(), 0, ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          uPF(ib.s - 1 - i, iN, vars::prim::Rho) =
              uPF(ib.s + i, (nNodes + 2) - iN - 1, vars::prim::Rho);
          uPF(ib.e + 1 + i, iN, vars::prim::Rho) =
              uPF(ib.e - i, (nNodes + 2) - iN - 1, vars::prim::Rho);
          uAF(ib.s - 1 - i, iN, vars::aux::Tgas) =
              uAF(ib.s + i, (nNodes + 2) - iN - 1, vars::aux::Tgas);
          uAF(ib.e + 1 + i, iN, vars::aux::Tgas) =
              uAF(ib.e - i, (nNodes + 2) - iN - 1, vars::aux::Tgas);
          zbar(ib.s - 1 - i, iN, i_H) =
              zbar(ib.s + i, (nNodes + 2) - iN - 1, i_H);
          zbar(ib.e + 1 + i, iN, i_H) =
              zbar(ib.e - i, (nNodes + 2) - iN - 1, i_H);
          zbar(ib.s - 1 - i, iN, i_He) =
              zbar(ib.s + i, (nNodes + 2) - iN - 1, i_He);
          zbar(ib.e + 1 + i, iN, i_He) =
              zbar(ib.e - i, (nNodes + 2) - iN - 1, i_He);
          zbar(ib.s - 1 - i, iN, i_C) =
              zbar(ib.s + i, (nNodes + 2) - iN - 1, i_C);
          zbar(ib.e + 1 + i, iN, i_C) =
              zbar(ib.e - i, (nNodes + 2) - iN - 1, i_C);
        }
      });
}

} // namespace athelas
