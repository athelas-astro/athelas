#pragma once

#include "basic_types.hpp"
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
                              ProblemIn *pin) {
  const bool ionization_active =
      pin->param()->get<bool>("physics.ionization_enabled");
  const int saha_ncomps =
      pin->param()->get<int>("ionization.ncomps"); // for ionization
  const auto ncomps =
      pin->param()->get<int>("composition.ncomps", 3); // mass fractions
  athelas_requires(ncomps == 3, "One zone ionization requires ncomps = 3.");
  athelas_requires(ionization_active,
                   "One zone ionization requires ionization enabled!");
  const auto eos_type = pin->param()->get<std::string>("eos.type");
  // athelas_requires(eos_type == "ideal", "One zone ionization requires ideal
  // gas eos!");
  //  Don't try to track ionization for more species than we use.
  //  We will track ionization for the first saha_ncomps species
  athelas_requires(saha_ncomps == 3,
                   "One zone ionization requires [ionization.ncomps] = 3.");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");
  auto uAF = mesh_state(0).get_field("u_af");
  auto sd0 = mesh_state(0);

  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const int nNodes = grid->n_nodes();

  const auto temperature =
      pin->param()->get<double>("problem.params.temperature", 5800); // K
  const auto rho =
      pin->param()->get<double>("problem.params.rho", 1000.0); // g/cc

  const auto fn_ionization =
      pin->param()->get<std::string>("ionization.fn_ionization");
  const auto fn_deg =
      pin->param()->get<std::string>("ionization.fn_degeneracy");

  if (temperature <= 0.0 || rho <= 0.0) {
    throw_athelas_error("Temperature and denisty must be positive definite!");
  }

  std::shared_ptr<atom::CompositionData> comps =
      std::make_shared<atom::CompositionData>(grid->n_elements() + 2, nNodes,
                                              ncomps);
  std::shared_ptr<atom::IonizationState> ionization_state =
      std::make_shared<atom::IonizationState>(
          grid->n_elements() + 2, nNodes, ncomps, 7, saha_ncomps, fn_ionization,
          fn_deg, pin->param()->get<std::string>("ionization.solver"));
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
        for (int q = 0; q < nNodes + 2; q++) {
        uCF(i, q, vars::cons::SpecificVolume) =
            1.0 / rho;
        mass_fractions(i, q, i_H) = X_H;
        mass_fractions(i, q, i_He) = X_He;
        mass_fractions(i, q, i_C) = X_C;

        }
        for (int q = 0; q < nNodes + 2; q++) {
          uPF(i, q, vars::prim::Rho) = rho;
          uAF(i, q, vars::aux::Tgas) = temperature;

          // Set Zbar assuming full ionization -- used as guess in Saha below.
          zbar(i, q, i_H) = 1;
          zbar(i, q, i_He) = 2;
          zbar(i, q, i_C) = 6;
        }

        charges(i_H) = 1;
        neutrons(i_H) = 0;
        inv_atomic_mass(i_H) = 1.0;

        charges(i_He) = 2;
        neutrons(i_He) = 2;
        inv_atomic_mass(i_He) = 1.0 / 4.0;

        charges(i_C) = 6;
        neutrons(i_C) = 6;
        inv_atomic_mass(i_C) = 1.0 / 12.0;
      });
  mesh_state.setup_composition(comps);
  mesh_state.setup_ionization(ionization_state);

  const auto &eos = mesh_state.eos();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: OneZoneIonization :: Energy",
      DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
        eos::EOSLambda lambda;
        for (int q = 0; q < nNodes + 2; q++) {
          uPF(i, q, vars::prim::Rho) = rho;
          uAF(i, q, vars::aux::Tgas) = temperature;
          if (eos_type == "paczynski") {
            atom::paczynski_terms(sd0, i, q, lambda.ptr());
          }
          uAF(i, q, vars::aux::Pressure) = pressure_from_density_temperature(
              eos, rho, temperature, lambda.ptr());
          uPF(i, q, vars::prim::Sie) = sie_from_density_pressure(
              eos, rho, uAF(i, q, vars::aux::Pressure), lambda.ptr());
        }
        for (int q = 0; q < nNodes; ++q) {
          uCF(i, q, vars::cons::Energy) = uPF(i, q + 1, vars::prim::Sie);
        }
      });

  // atom::fill_derived_comps<Domain::Interior>(sd0, grid);
  // atom::solve_saha_ionization<Domain::Interior, atom::SahaSolver::Linear>(
  //     sd0, *grid);
  // atom::fill_derived_ionization<Domain::Interior>(sd0, grid);
  //  composition boundary condition
  static const IndexRange vb_comps(std::make_pair(3, 3 + ncomps - 1));
  bc::fill_ghost_zones_composition(uCF, vb_comps);

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
