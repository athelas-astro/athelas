/**
 * @file progenitor.hpp
 * --------------
 *
 * @brief Supernova progenitor initialization
 *
 * This one is the most involved of all pgens.
 * We have to:
 *  - read the profiles
 *  - apply any mass cut
 *  - reform our mesh to conform to the progenitor domain
 *  - interpolate data to our mesh
 *  - project nodal fields to a modal basis.
 *
 *  Due to the complexity (and low cost, relatively), much of this
 *  is designed to be done on the host.
 *
 * As elsewhere, because the pgen generally needs the basis,
 * and the basis needs the density field for its inner product, have to
 * call this twice. The first call is all in the if (fluid_basis == nullptr)
 * and constructs some nodal primitives.
 *
 * The second call goes into the other branch and is more involved.
 * Here we load, interpolate, and project the mass fractions and then
 * the rest of the conserved quantities.
 *
 * Mass fractions are renormalized at the end. This helps to maintain
 * conservation. Errors can be accumulated 1) in the construction of the
 * profile, 2) in the interpolation from the progenitor grid to Athelas's grid,
 * and 3) when Ni56 is placed by hand.
 */

#pragma once

#include <cmath>

#include "Kokkos_Core.hpp"
#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions.hpp"
#include "composition/composition.hpp"
#include "composition/saha.hpp"
#include "constants.hpp"
#include "eos/eos_variant.hpp"
#include "error.hpp"
#include "geometry/grid.hpp"
#include "io/parser.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"
#include "utilities.hpp"

namespace athelas {

/**
 * @brief Initialize supernova progenitor
 **/
void progenitor_init(State *state, GridStructure *grid, ProblemIn *pin,
                     const eos::EOS *eos,
                     basis::ModalBasis *fluid_basis = nullptr) {
  static constexpr int NUM_COLS_HYDRO = 5;

  // Perform a number of sanity checks
  if (pin->param()->get<std::string>("eos.type") != "paczynski") {
    THROW_ATHELAS_ERROR("Problem 'supernova' requires paczynski eos!");
  }

  if (!pin->param()->get<bool>("physics.composition_enabled")) {
    THROW_ATHELAS_ERROR("Problem 'supernova' requires composition enabled!!");
  }

  if (!pin->param()->get<bool>("physics.ionization_enabled")) {
    THROW_ATHELAS_ERROR("Problem 'supernova' requires ionization enabled!!");
  }

  if (!pin->param()->get<bool>("physics.gravity_active")) {
    THROW_ATHELAS_ERROR("Problem 'supernova' requires gravity enabled!!");
  }

  if (pin->param()->get<std::string>("problem.geometry") != "spherical") {
    THROW_ATHELAS_ERROR("Problem 'supernova' requires spherical geometry!");
  }

  const auto ncomps = pin->param()->get<int>("composition.ncomps");

  const auto fn_ionization =
      pin->param()->get<std::string>("ionization.fn_ionization");
  const auto fn_deg =
      pin->param()->get<std::string>("ionization.fn_degeneracy");
  const int saha_ncomps = pin->param()->get<int>("ionization.ncomps");

  if (saha_ncomps > ncomps) {
    THROW_ATHELAS_ERROR("One zone ionization requires [ionization.ncomps] <= "
                        "[problem.params.ncomps]!");
  }

  // check if we want to do a mass cut
  double mass_cut = 0.0;
  if (pin->param()->contains("problem.params.mass_cut")) {
    mass_cut = pin->param()->get<double>("problem.params.mass_cut");
    if (mass_cut < 0.0) {
      THROW_ATHELAS_ERROR("The mass cut cannot be negative!");
    }
  } else {
    WARNING_ATHELAS(
        "You are running a supernova problem but have not specified a mass "
        "cut! It is assumed then that the mass cut is already present in the "
        "input profile. If not, bad things may happen.");
  }

  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const int nNodes = grid->n_nodes();
  static const int order = nNodes;

  auto uCF = state->u_cf();
  auto uPF = state->u_pf();
  auto uAF = state->u_af();

  std::shared_ptr<atom::CompositionData> comps =
      std::make_shared<atom::CompositionData>(grid->n_elements() + 2, order,
                                              ncomps);

  std::shared_ptr<atom::IonizationState> ionization_state =
      std::make_shared<atom::IonizationState>(grid->n_elements() + 2, nNodes,
                                              ncomps, ncomps + 1, saha_ncomps,
                                              fn_ionization, fn_deg);

  auto mass_fractions = state->mass_fractions();
  auto charges = comps->charge();
  auto neutrons = comps->neutron_number();
  auto ye = comps->ye();
  auto *species_indexer = comps->species_indexer();

  const auto fn_hydro =
      pin->param()->get<std::string>("problem.params.fn_hydro");

  // --- read in hydro data ---
  auto hydro_data = io::Parser::parse_file(fn_hydro, ' ');

  if (!hydro_data) {
    THROW_ATHELAS_ERROR("Error reading hydro profile!");
  }

  // Safety check on the structure of the profile.
  const size_t num_columns_hydro = hydro_data->rows[0].size();
  if (num_columns_hydro != NUM_COLS_HYDRO) {
    THROW_ATHELAS_ERROR("Wrong number of columns in hydro profile!");
  }

  auto [radius_star, density_star, velocity_star, pressure_star,
        temperature_star] =
      io::get_columns_by_indices<double, double, double, double, double>(
          *hydro_data, 0, 1, 2, 3, 4);

  const double rstar = radius_star.back();
  const int n_zones_prog = static_cast<int>(radius_star.size());

  // Create Kokkos views
  AthelasArray1D<double> radius_view("progenitor radius", n_zones_prog);
  AthelasArray1D<double> density_view("progenitor density", n_zones_prog);
  AthelasArray1D<double> velocity_view("progenitor velocity", n_zones_prog);
  AthelasArray1D<double> pressure_view("progenitor pressure", n_zones_prog);
  AthelasArray1D<double> temperature_view("progenitor temperature",
                                          n_zones_prog);

  // Create host mirrors for data transfer
  auto radius_host = Kokkos::create_mirror_view(radius_view);
  auto density_host = Kokkos::create_mirror_view(density_view);
  auto velocity_host = Kokkos::create_mirror_view(velocity_view);
  auto pressure_host = Kokkos::create_mirror_view(pressure_view);
  auto temperature_host = Kokkos::create_mirror_view(temperature_view);

  // Copy data from vectors to host mirrors
  for (int i = 0; i < n_zones_prog; ++i) {
    radius_host(i) = radius_star[i];
    density_host(i) = density_star[i];
    velocity_host(i) = velocity_star[i];
    pressure_host(i) = pressure_star[i];
    temperature_host(i) = temperature_star[i];
  }

  // Deep copy from host to device
  Kokkos::deep_copy(radius_view, radius_host);
  Kokkos::deep_copy(density_view, density_host);
  Kokkos::deep_copy(velocity_view, velocity_host);
  Kokkos::deep_copy(pressure_view, pressure_host);
  Kokkos::deep_copy(temperature_view, temperature_host);

  if (fluid_basis == nullptr) {
    // Phase 1: Initialize nodal values
    // Here we construct nodal density and temperature.
    // This is where we deal with the mass cut.
    // We also need to re-build the mesh

    // We're going to do this on host for simplicity.
    int idx_cut = 0;
    if (mass_cut != 0.0) {
      double mass_enc = 0.0;
      for (int i = 0; i < n_zones_prog - 1; ++i) {
        mass_enc += constants::FOURPI * density_host(i) * radius_host(i) *
                    radius_host(i) * (radius_host(i + 1) - radius_host(i)) /
                    constants::M_sun;
        if (mass_enc >= mass_cut) {
          idx_cut = i;
          break;
        }
      }
    }

    auto &xl = pin->param()->get_mutable_ref<double>("problem.xl");
    auto &xr = pin->param()->get_mutable_ref<double>("problem.xr");
    xl = radius_host[idx_cut];
    xr = rstar;
    auto newgrid = GridStructure(pin);
    *grid = newgrid;

    auto centers = grid->centers();
    auto x_l = grid->x_l();
    auto r = grid->nodal_grid();
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Supernova (1)", DevExecSpace(),
        ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          for (int q = 0; q < nNodes + 2; q++) {
            // TODO(astrobarker): I really need to just expand the nodal grid to
            // include the interfaces Could be annoying in the output though.

            const double r_athelas =
                (q == 0) ? x_l(i)
                         : ((q == nNodes + 1) ? x_l(i + 1) : r(i, q - 1));
            const int index_left =
                std::min(utilities::find_closest_cell(radius_view, r_athelas,
                                                      n_zones_prog),
                         n_zones_prog - 2);
            uPF(i, q, vars::prim::Rho) = utilities::LINTERP(
                radius_view(index_left), radius_view(index_left + 1),
                density_view(index_left), density_view(index_left + 1),
                r_athelas);
            uAF(i, q, vars::aux::Pressure) = utilities::LINTERP(
                radius_view(index_left), radius_view(index_left + 1),
                pressure_view(index_left), pressure_view(index_left + 1),
                r_athelas);
            uAF(i, q, vars::aux::Tgas) = utilities::LINTERP(
                radius_view(index_left), radius_view(index_left + 1),
                temperature_view(index_left), temperature_view(index_left + 1),
                r_athelas);
          }
        });
  }

  // Phase 2: Initialize modal coefficients
  if (fluid_basis != nullptr) {
    auto species = comps->charge();
    auto neutron_number = comps->neutron_number();
    auto species_h = Kokkos::create_mirror_view(species);
    auto neutron_number_h = Kokkos::create_mirror_view(neutron_number);
    auto upf_h = Kokkos::create_mirror_view(uPF);

    // --- read in composition data ---
    const auto fn_comps =
        pin->param()->get<std::string>("problem.params.fn_comps");
    auto comps_data = io::Parser::parse_file(fn_comps, ' ');
    if (!comps_data) {
      THROW_ATHELAS_ERROR("Error reading composition profile!");
    }

    // check that the number of species in the profile matches the input deck
    const int num_columns_comps = static_cast<int>(comps_data->rows[0].size());
    if (num_columns_comps != ncomps) {
      THROW_ATHELAS_ERROR(
          "The number of species in the compositional profile '" + fn_comps +
          "' does not match the number of species list in in the input deck "
          "([composition.ncomps])! There are " +
          std::to_string(num_columns_comps) + " in the profile and " +
          std::to_string(ncomps) + " in the input deck!");
    }

    // First, load the leading two rows containing isotope info.
    // We also check for and store the species indices of Ni/Co/Fe 56
    bool ni56_present = false;
    bool co56_present = false;
    bool fe56_present = false;
    for (int e = 0; e < ncomps; ++e) {
      auto data = io::get_column_by_index<int>(*comps_data, e);
      species_h(e) = data[0];
      neutron_number_h(e) = data[1];
      // We need to store the element index of Ni56, Co56, Fe56
      // in the species indexer. If you need to track other specific
      // species, this might be a good place to do it.
      if (species_h(e) == 28 && neutron_number_h(e) == 28) {
        species_indexer->add("ni56", e);
        ni56_present = true;
      }
      if (species_h(e) == 27 && neutron_number_h(e) == 29) {
        species_indexer->add("co56", e);
        co56_present = true;
      }
      if (species_h(e) == 26 && neutron_number_h(e) == 30) {
        species_indexer->add("fe56", e);
        fe56_present = true;
      }
    }
    if (!ni56_present || !co56_present || !fe56_present) {
      THROW_ATHELAS_ERROR("All of Ni/Co/Fe 56 are required to be present in "
                          "the composition profile!");
    }

    // TODO(astrobarker): should probably just make a host view directly.
    AthelasArray2D<double> comps_star("progenitor composition", n_zones_prog,
                                      ncomps);
    auto comps_star_host = Kokkos::create_mirror_view(comps_star);

    // backwards loop ordering, but reduces loads
    for (int e = 0; e < ncomps; ++e) {
      auto data = io::get_column_by_index<double>(*comps_data, e);
      // loop over zones offset by 2 -- first two rows are isotope info.
      for (int i = 2; i < n_zones_prog + 2; ++i) {
        const size_t i_cell = i - 2;
        comps_star_host(i_cell, e) = data[i];
      }
    }

    // Now we need to do a nodal to modal projection of the mass fractions.
    // Before we can do that we need to interpolate them to our grid.
    auto mass_fractions_h = Kokkos::create_mirror_view(state->mass_fractions());
    auto r = grid->nodal_grid();
    auto r_h = Kokkos::create_mirror_view(r);
    auto mass_matrix_h = Kokkos::create_mirror_view(fluid_basis->mass_matrix());
    auto phi_h = Kokkos::create_mirror_view(fluid_basis->phi());
    auto sqrt_gm_h = Kokkos::create_mirror_view(grid->sqrt_gm());
    auto dr_h = Kokkos::create_mirror_view(grid->widths());
    auto weights_h = Kokkos::create_mirror_view(grid->weights());
    std::vector<double> x_cell(nNodes); // holds per element nodal data
    for (int i = ib.s; i <= ib.e; ++i) {
      for (int e = 0; e < ncomps; ++e) {
        // loop over nodes on an element, interpolate to nodal positions
        for (int q = 0; q < nNodes; ++q) {
          const double rq = r_h(i, q);
          const int idx =
              utilities::find_closest_cell(radius_host, rq, n_zones_prog);
          x_cell[q] = utilities::LINTERP(radius_host(idx), radius_host(idx + 1),
                                         comps_star_host(idx, e),
                                         comps_star_host(idx + 1, e), rq);
        }
        // Project the nodal representation to a modal one
        // Compute L2 projection: <f_q, phi_k> / <phi_k, phi_k>
        // This should probably be a function.
        for (int k = 0; k < order; k++) {
          double numerator = 0.0;
          const double denominator = mass_matrix_h(i, k);

          // Compute <f_q, phi_k>
          for (int q = 0; q < nNodes; q++) {
            const double nodal_val = x_cell[q];
            const double rho = upf_h(i, q + 1, vars::prim::Rho);

            numerator += nodal_val * phi_h(i, q + 1, k) * weights_h(q) *
                         dr_h(i) * sqrt_gm_h(i, q + 1) * rho;
          }
          mass_fractions_h(i, k, e) = numerator / denominator;
          if (k > 0) {
            mass_fractions(i, k, e) *= std::exp(-k);
          }
        }
      }
    }

    Kokkos::deep_copy(state->mass_fractions(), mass_fractions_h);
    Kokkos::deep_copy(species, species_h);
    Kokkos::deep_copy(neutron_number, neutron_number_h);

    // If we want to default the ionization fractions to anything other
    // than 0, we can do it below. i.e., for species that we are not doing
    // Saha solves, what is their default ionization state?
    auto mass_fractions = state->mass_fractions();
    auto charges = comps->charge();
    auto neutrons = comps->neutron_number();
    auto ionization_states = ionization_state->ionization_fractions();
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Supernova (Ionization)",
        DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          for (int q = 0; q < nNodes + 2; ++q) {
            for (int elem = 0; elem < saha_ncomps; ++elem) {
              const int Z = charges(elem);

              ionization_states(i, q, elem, Z) = 1.0;
            }
          }
        });

    state->setup_composition(comps);
    state->setup_ionization(ionization_state);

    // We need to do the nodal to modal projection for specific volume first
    // and separate, as a full modal basis representation of tau is needed
    // for doing the Saha solve.
    auto sqrt_gm = grid->sqrt_gm();
    auto dr = grid->widths();
    AthelasArray1D<double> tau_cell("supernova :: tau cell", nNodes);
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Supernova :: Project tau",
        DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          // loop over nodes on an element, interpolate to nodal positions
          for (int q = 0; q < nNodes; ++q) {
            tau_cell(q) = 1.0 / uPF(i, q, vars::prim::Rho);
          }
          // Project the nodal representation to a modal one
          // Compute L2 projection: <f_q, phi_k> / <phi_k, phi_k>
          // This should probably be a function.
          for (int k = 0; k < order; k++) {
            double numerator = 0.0;
            const double denominator = mass_matrix_h(i, k);

            // Compute <f_q, phi_k>
            for (int q = 0; q < nNodes; q++) {
              const double nodal_val = tau_cell(q);
              const double rho = uPF(i, q + 1, vars::prim::Rho);

              numerator += nodal_val * phi_h(i, q + 1, k) * weights_h(q) *
                           dr(i) * sqrt_gm(i, q + 1) * rho;
            }
            uCF(i, k, vars::cons::SpecificVolume) = numerator / denominator;
            // We apply a simple exponential filter to modes
            // This is critical as any discontinuities or non-smooth features
            // in the nodal density (specific volume) profile can result in
            // artificially high slopes etc. This smooths those higher modes
            // so that we don't experience that. We need to do this here as
            // the following fill_derived_ calls evaluate tau at nodal locations
            // and it can be unphysical. We will also apply this to the
            // other fields.
            if (k > 0) {
              uCF(i, k, vars::cons::SpecificVolume) *= std::exp(-k);
            }
          }
        });

    // Compute necessary terms for using the Paczynski eos
    atom::fill_derived_comps<Domain::Interior>(state, grid, fluid_basis);

    // Get the initial Saha ionization state.
    // Also computes the electron number density
    atom::solve_saha_ionization<Domain::Interior>(*state, *grid, *eos,
                                                  *fluid_basis);
    atom::fill_derived_ionization<Domain::Interior>(state, grid, fluid_basis);

    // Finally, compute the radhydro variables. This requires, as before,
    // interpolation of the progenitor data to nodal collocation points
    // and projection from nodal to modal representation.
    // For the fluid energy we use the progenitor pressure and invert
    // the eos to get a specific internal energy. This helps maintain
    // any pressure based structure.

    // Per cell nodal storage
    AthelasArray1D<double> vel_cell("supernova :: vel cell", nNodes);
    AthelasArray1D<double> energy_cell("supernova :: energy cell", nNodes);
    auto phi_fluid = fluid_basis->phi();
    auto mkk_fluid = fluid_basis->mass_matrix();
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Supernova (2)", DevExecSpace(),
        ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          double lambda[8];
          // Project each conserved variable
          // loop over nodes on an element, interpolate to nodal positions
          for (int q = 0; q < nNodes; ++q) {
            const double rq = r(i, q);
            const int idx =
                utilities::find_closest_cell(radius_view, rq, n_zones_prog);
            tau_cell(q) = basis::basis_eval(phi_fluid, uCF, i,
                                            vars::cons::SpecificVolume, q);
            vel_cell(q) = utilities::LINTERP(
                radius_view(idx), radius_view(idx + 1), velocity_view(idx),
                velocity_view(idx + 1), rq);
            const double pressure_q = uAF(i, q + 1, vars::aux::Pressure);
            atom::paczynski_terms(state, i, q, lambda);
            energy_cell(q) = eos::sie_from_density_pressure(
                                 eos, 1.0 / tau_cell(q), pressure_q, lambda) +
                             0.5 * vel_cell(q) * vel_cell(q);
            std::println("SIE :: {:.5e} | {} {:.5e} {:.5e} {:.5e}",
                         energy_cell(q), q, tau_cell(q), vel_cell(q),
                         pressure_q);
            // Recompute temperature given the rest of the state
          }
          // Project the nodal representation to a modal one
          // Compute L2 projection: <f_q, phi_k> / <phi_k, phi_k>
          // This should probably be a function.
          for (int k = 0; k < order; k++) {
            double numerator_vel = 0.0;
            double numerator_energy = 0.0;
            const double denominator = mkk_fluid(i, k);

            // Compute <f_q, phi_k>
            for (int q = 0; q < nNodes; q++) {
              const double nodal_vel = vel_cell(q);
              const double nodal_energy = energy_cell(q);
              const double rho = uPF(i, q + 1, vars::prim::Rho);

              numerator_vel += nodal_vel * phi_h(i, q + 1, k) * weights_h(q) *
                               dr(i) * sqrt_gm(i, q + 1) * rho;
              numerator_energy += nodal_energy * phi_h(i, q + 1, k) *
                                  weights_h(q) * dr(i) * sqrt_gm(i, q + 1) *
                                  rho;
            }
            uCF(i, k, vars::cons::Velocity) = numerator_vel / denominator;
            uCF(i, k, vars::cons::Energy) = numerator_energy / denominator;
            // We apply a simple exponential filter to modes
            // This is critical as any discontinuities or non-smooth features
            // in the nodal density (specific volume) profile can result in
            // artificially high slopes etc. This smooths those higher modes
            // so that we don't experience that. We need to do this here as
            // the following fill_derived_ calls evaluate tau at nodal locations
            // and it can be unphysical. We will also apply this to the
            // other fields.
            if (k > 0) {
              uCF(i, k, vars::cons::Velocity) *= std::exp(-k);
              uCF(i, k, vars::cons::Energy) *= std::exp(-k);
            }
          }
        });

    // Now recompute the temperature so that everything is happy and consistent
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Supernova :: Recompute T",
        DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          for (int q = 0; q < nNodes + 2; q++) {
            double lambda[8];
            atom::paczynski_terms(state, i, q, lambda);
            const double tau = basis::basis_eval(phi_fluid, uCF, i,
                                                 vars::cons::SpecificVolume, q);
            const double vel =
                basis::basis_eval(phi_fluid, uCF, i, vars::cons::Velocity, q);
            const double ener =
                basis::basis_eval(phi_fluid, uCF, i, vars::cons::Energy, q);
            uAF(i, q, vars::aux::Tgas) = eos::temperature_from_conserved(
                eos,
                basis::basis_eval(phi_fluid, uCF, i, vars::cons::SpecificVolume,
                                  q),
                basis::basis_eval(phi_fluid, uCF, i, vars::cons::Velocity, q),
                basis::basis_eval(phi_fluid, uCF, i, vars::cons::Energy, q),
                lambda);
            std::println("new T :: {} {} {:.5e} | {:.5e} {:.5e} {:.5e}", i, q,
                         uAF(i, q, vars::aux::Tgas), tau, vel, ener);
          }
        });

    atom::fill_derived_comps<Domain::Interior>(state, grid, fluid_basis);
    atom::solve_saha_ionization<Domain::Interior>(*state, *grid, *eos,
                                                  *fluid_basis);
    atom::fill_derived_ionization<Domain::Interior>(state, grid, fluid_basis);
    // composition boundary condition
    static const IndexRange vb_comps(std::make_pair(3, 3 + ncomps - 1));
    bc::fill_ghost_zones_composition(uCF, vb_comps);
  }

  // Fill density and temperature in guard cells.
  // Temperature must be filled in when ionization is active.
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Supernova (ghost)", DevExecSpace(), 0,
      ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int q = 0; q < nNodes + 2; q++) {
          uPF(ib.s - 1 - i, q, vars::prim::Rho) =
              uPF(ib.s + i, (nNodes + 2) - q - 1, vars::prim::Rho);
          uPF(ib.e + 1 + i, q, vars::prim::Rho) =
              uPF(ib.e - i, (nNodes + 2) - q - 1, vars::prim::Rho);
          uAF(ib.s - 1 - i, q, vars::aux::Tgas) =
              uAF(ib.s + i, (nNodes + 2) - q - 1, vars::aux::Tgas);
          uAF(ib.e + 1 + i, q, vars::aux::Tgas) =
              uAF(ib.e - i, (nNodes + 2) - q - 1, vars::aux::Tgas);
        }
      });
}

} // namespace athelas
