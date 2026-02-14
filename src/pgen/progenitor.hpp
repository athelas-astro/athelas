/**
 * @file progenitor.hpp
 * --------------
 *
 * @brief Supernova progenitor initialization
 *
 * This one is the most involved of all pgens.
 *
 * Due to the complexity (and low cost, relatively), much of this
 * is designed to be done on the host.
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
 * Mass fractions are renormalized. This helps to maintain
 * conservation. Errors can be accumulated 1) in the construction of the
 * profile, 2) in the interpolation from the progenitor grid to Athelas's grid,
 * and 3) when Ni56 is placed by hand.
 *
 * NOTE: Currently injected ni56 is distributed uniformly out to a
 * prescribed mass.
 *
 * The radiation energy density is set by assuming that the (new) gas and
 * radiation temperatures are equal. The radiation flux comes from the
 * luminosity in the stellar profile.
 *
 * All conserved/evolved fields have their higher modes (slopes / k > 0)
 * decayed with an exponential filter: U_K *= exp(-k).
 * This is because the nodal -> modal projection can lead to artificially
 * large modes if the element contains a sharp feature, such as at
 * compositional interfaces. The cell average is not modified.
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
#include "radiation/rad_utilities.hpp"
#include "state/state.hpp"
#include "utilities.hpp"

namespace athelas {

/**
 * @brief Initialize supernova progenitor
 **/
void progenitor_init(MeshState &mesh_state, GridStructure *grid, ProblemIn *pin) {
  // If we ever add columns to the hydro profile, change this.
  constexpr int NUM_COLS_HYDRO = 6;

  // Perform a number of sanity checks
  athelas_requires(pin->param()->get<std::string>("eos.type") == "paczynski",
                   "Problem 'supernoa' requires Paczynski eos!");
  athelas_requires(pin->param()->get<bool>("physics.composition_enabled"),
                   "Problem 'supernova' requires composition enabled!");
  athelas_requires(pin->param()->get<bool>("physics.ionization_enabled"),
                   "Problem 'supernova' requires ionization enabled!");
  athelas_requires(pin->param()->get<bool>("physics.gravity_active"),
                   "Problem 'supernova' requires gravity enabled!");
  athelas_requires(pin->param()->get<std::string>("problem.geometry") ==
                       "spherical",
                   "Problem 'supernova' requires spherical geometry!");

  const auto ncomps = pin->param()->get<int>("composition.ncomps");
  const auto fn_ionization =
      pin->param()->get<std::string>("ionization.fn_ionization");
  const auto fn_deg =
      pin->param()->get<std::string>("ionization.fn_degeneracy");
  const auto saha_ncomps = pin->param()->get<int>("ionization.ncomps");
  const auto ni_injection_mass =
      pin->param()->get<double>("problem.params.ni_mass", 0.0);
  const auto ni_injection_bndry =
      pin->param()->get<double>("problem.params.ni_boundary", 0.0);
  const auto rad_enabled = pin->param()->get<bool>("physics.rad_active");

  // sanity checks
  athelas_requires(ni_injection_mass >= 0.0 && ni_injection_bndry >= 0.0,
                   "The nickel injection mass and boundary must be >= 0.0!");

  bool inject_ni = false;
  if (ni_injection_mass > 0.0 && ni_injection_bndry > 0.0) {
    inject_ni = true;
  }

  athelas_requires(saha_ncomps <= ncomps,
                   "The 'supernova' problem requires [ionization.ncomps] <= "
                   "[composition.ncomps]!");

  // check if we want to do a mass cut
  double mass_cut = 0.0;
  if (pin->param()->contains("problem.params.mass_cut")) {
    mass_cut = pin->param()->get<double>("problem.params.mass_cut");
    athelas_requires(mass_cut >= 0.0, "The mass cut cannot be negative!");
    if (mass_cut > ni_injection_bndry && inject_ni) {
      throw_athelas_error(
          "The mass cut should be smaller than the nickel injection boundary!");
    }
  } else {
    athelas_warning(
        "You are running a supernova problem but have not specified a mass "
        "cut! It is assumed then that the mass cut is already present in the "
        "input profile. If not, bad things may happen.");
  }

  static const int nNodes = grid->n_nodes();
  static const int order = mesh_state.p_order();
  static const int nx = grid->n_elements();
  static const IndexRange ib(grid->domain<Domain::Interior>());

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uAF = mesh_state(0).get_field("u_af");
  auto uPF = mesh_state(0).get_field("u_pf");

  std::shared_ptr<atom::CompositionData> comps =
      std::make_shared<atom::CompositionData>(nx + 2, nNodes, ncomps);

  auto mass_fractions = mesh_state.mass_fractions("u_cf");
  auto charges = comps->charge();
  auto neutrons = comps->neutron_number();
  auto ye = comps->ye();
  auto *species_indexer = comps->species_indexer();

  const auto fn_hydro =
      pin->param()->get<std::string>("problem.params.fn_hydro");

  // --- read in hydro data ---
  auto hydro_data = io::Parser::parse_file(fn_hydro);

  if (!hydro_data) {
    throw_athelas_error("Error reading hydro profile!");
  }

  // Safety check on the structure of the profile.
  const size_t num_columns_hydro = hydro_data->rows[0].size();
  if (num_columns_hydro != NUM_COLS_HYDRO) {
    throw_athelas_error("Wrong number of columns in hydro profile!");
  }

  auto [radius_star, density_star, velocity_star, pressure_star,
        temperature_star, luminosity_star] =
      io::get_columns_by_indices<double, double, double, double, double,
                                 double>(*hydro_data, 0, 1, 2, 3, 4, 5);

  const double rstar = radius_star.back();
  const int n_zones_prog = static_cast<int>(radius_star.size());

  // Create Kokkos views
  AthelasArray1D<double> radius_view("progenitor radius", n_zones_prog);
  AthelasArray1D<double> density_view("progenitor density", n_zones_prog);
  AthelasArray1D<double> velocity_view("progenitor velocity", n_zones_prog);
  AthelasArray1D<double> pressure_view("progenitor pressure", n_zones_prog);
  AthelasArray1D<double> temperature_view("progenitor temperature",
                                          n_zones_prog);
  AthelasArray1D<double> luminosity_view("progenitor luminosity", n_zones_prog);

  // Create host mirrors for data transfer
  auto radius_host = Kokkos::create_mirror_view(radius_view);
  auto density_host = Kokkos::create_mirror_view(density_view);
  auto velocity_host = Kokkos::create_mirror_view(velocity_view);
  auto pressure_host = Kokkos::create_mirror_view(pressure_view);
  auto temperature_host = Kokkos::create_mirror_view(temperature_view);
  auto luminosity_host = Kokkos::create_mirror_view(luminosity_view);

  // Copy data from vectors to host mirrors
  for (int i = 0; i < n_zones_prog; ++i) {
    radius_host(i) = radius_star[i];
    density_host(i) = density_star[i];
    velocity_host(i) = velocity_star[i];
    pressure_host(i) = pressure_star[i];
    temperature_host(i) = temperature_star[i];
    luminosity_host(i) = luminosity_star[i];
  }

  // Deep copy from host to device
  Kokkos::deep_copy(radius_view, radius_host);
  Kokkos::deep_copy(density_view, density_host);
  Kokkos::deep_copy(velocity_view, velocity_host);
  Kokkos::deep_copy(pressure_view, pressure_host);
  Kokkos::deep_copy(temperature_view, temperature_host);
  Kokkos::deep_copy(luminosity_view, luminosity_host);

    // Phase 1: Initialize nodal values
    // Here we construct nodal density and temperature.
    // This is where we deal with the mass cut.
    // We also need to re-build the mesh

    // We're going to do this on host for simplicity.
    int idx_cut = 0;
    double r_cut = radius_host[0];
    if (mass_cut != 0.0) {
      auto &mass_cut_ref =
          pin->param()->get_mutable_ref<double>("problem.params.mass_cut");
      double mass_enc = 0.0;
      double dm = 0.0;
      for (int i = 0; i < n_zones_prog - 1; ++i) {
        dm = constants::FOUR_THIRDS_PI * density_host(i) *
             (std::pow(radius_host(i + 1), 3.0) -
              std::pow(radius_host(i), 3.0)) /
             constants::M_sun;
        mass_enc += dm;
        if (mass_enc >= mass_cut) {
          idx_cut = i - 1;
          r_cut =
              utilities::LINTERP(mass_enc - dm, mass_enc, radius_host(idx_cut),
                                 radius_host(idx_cut + 1), mass_cut);
          mass_cut = mass_enc;
          mass_cut_ref = mass_cut;
          break;
        }
      }
    }

    auto &xl = pin->param()->get_mutable_ref<double>("problem.xl");
    auto &xr = pin->param()->get_mutable_ref<double>("problem.xr");
    xl = r_cut;
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

            const double r_athelas = r(i, q);
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

  // Phase 2: Everything else
    const auto &fluid_basis = mesh_state.fluid_basis();
    const auto &eos = mesh_state.eos();
    auto species = comps->charge();
    auto neutron_number = comps->neutron_number();
    auto inv_atomic_mass = comps->inverse_atomic_mass();
    auto species_h = Kokkos::create_mirror_view(species);
    auto neutron_number_h = Kokkos::create_mirror_view(neutron_number);
    auto inv_atomic_mass_h = Kokkos::create_mirror_view(inv_atomic_mass);
    auto upf_h = Kokkos::create_mirror_view(uPF);

    // --- read in composition data ---
    const auto fn_comps =
        pin->param()->get<std::string>("problem.params.fn_comps");
    auto comps_data = io::Parser::parse_file(fn_comps);
    if (!comps_data) {
      throw_athelas_error("Error reading composition profile!");
    }

    // check that the number of species in the profile matches the input deck
    const int num_columns_comps = static_cast<int>(comps_data->rows[0].size());
    if (num_columns_comps != ncomps) {
      throw_athelas_error(
          "The number of species in the compositional profile '" + fn_comps +
          "' does not match the number of species list in in the input deck "
          "([composition.ncomps])! There are " +
          std::to_string(num_columns_comps) + " in the profile and " +
          std::to_string(ncomps) + " in the input deck!");
    }

    // First, load the leading two rows containing isotope info.
    // We also check for and store the species indices of Ni/Co/Fe 56
    // Also grab, if present, index for neutrons. Grab max charge/
    int max_charge = 0;
    bool neut_present = false;
    bool ni56_present = false;
    bool co56_present = false;
    bool fe56_present = false;
    for (int e = 0; e < ncomps; ++e) {
      auto data = io::get_column_by_index<int>(*comps_data, e);
      species_h(e) = data[0];
      neutron_number_h(e) = data[1];
      inv_atomic_mass_h(e) = 1.0 / (data[0] + data[1]);

      // We need to store the element index of Ni56, Co56, Fe56
      // in the species indexer. If you need to track other specific
      // species, this might be a good place to do it.
      if (species_h(e) == 0 && neutron_number_h(e) == 1) {
        species_indexer->add("neut", e);
        neut_present = true;
      }
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
      if (e == ncomps - 1) {
        max_charge = data[0];
      }
    }
    if (!ni56_present || !co56_present || !fe56_present) {
      throw_athelas_error("All of Ni/Co/Fe 56 are required to be present in "
                          "the composition profile!");
    }
    if (max_charge == 0) {
      throw_athelas_error(
          "Something weird possibly happening in supernova pgen -- max charge "
          "of species thought to be 0. Crashing out.");
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

    // Now we make sure that is neutrons are in our composition data
    // then they are the first species.
    if (neut_present) {
      const int ind_neut = species_indexer->get<int>("neut");
      if (ind_neut != 0) {
        species_h(ind_neut) = species_h(0);
        species_h(0) = 0; // Z
        neutron_number_h(ind_neut) = neutron_number_h(0);
        neutron_number_h(0) = 1;
        inv_atomic_mass_h(ind_neut) = inv_atomic_mass_h(0);
        inv_atomic_mass_h(0) = 1.0;

        for (int i = 2; i < n_zones_prog + 2; ++i) {
          const size_t i_cell = i - 2;
          double x_0 = comps_star_host(i_cell, 0);
          comps_star_host(i_cell, 0) = comps_star_host(i_cell, ind_neut);
          comps_star_host(i_cell, ind_neut) = x_0;
        }
      }
    }

    // Handle Ni56 injection
    if (inject_ni) {
      const int ind_ni = species_indexer->get<int>("ni56");
      int ncells_spread = 0;
      double mass_enc = 0.0;
      for (int i = 0; i < n_zones_prog - 1; ++i) {
        mass_enc += constants::FOURPI * density_host(i) * radius_host(i) *
                    radius_host(i) * (radius_host(i + 1) - radius_host(i)) /
                    constants::M_sun;
        if (mass_enc > ni_injection_bndry) {
          break;
        }
        ncells_spread++;
      }

      for (int i = 2; i < n_zones_prog + 2; ++i) {
        const int i_cell = i - 2;
        if (i_cell <= ncells_spread) {
          comps_star_host(i_cell, ind_ni) =
              ni_injection_mass / (ni_injection_bndry - mass_cut);
        } else {
          // Should we just set it to 0.0? Will cause problems if ever we
          // do ionization of ni56.
          comps_star_host(i_cell, ind_ni) = utilities::SMALL();
        }
      }
    }

    // Now we need to do a nodal to modal projection of the mass fractions.
    // Before we can do that we need to interpolate them to our grid.
    // TODO(astrobarker): this can be device side.
    auto mass_fractions_h =
        Kokkos::create_mirror_view(mesh_state.mass_fractions("u_cf"));
    auto r_h = Kokkos::create_mirror_view(r);
    auto mass_matrix_h = Kokkos::create_mirror_view(fluid_basis.mass_matrix());
    auto phi_h = Kokkos::create_mirror_view(fluid_basis.phi());
    auto sqrt_gm_h = Kokkos::create_mirror_view(grid->sqrt_gm());
    auto dr_h = Kokkos::create_mirror_view(grid->widths());
    auto weights = grid->weights();
    auto weights_h = Kokkos::create_mirror_view(weights);
    std::vector<double> x_cell(nNodes); // holds per element nodal data
    for (int i = ib.s; i <= ib.e; ++i) {
      for (int e = 0; e < ncomps; ++e) {
        // loop over nodes on an element, interpolate to nodal positions
        for (int q = 0; q < nNodes; ++q) {
          const double rq = r_h(i, q + 1);
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
            mass_fractions_h(i, k, e) *= std::exp(-k);
          }
        }
      }
    }

    Kokkos::deep_copy(mesh_state.mass_fractions("u_cf"), mass_fractions_h);
    Kokkos::deep_copy(species, species_h);
    Kokkos::deep_copy(neutron_number, neutron_number_h);
    Kokkos::deep_copy(inv_atomic_mass, inv_atomic_mass_h);

    std::shared_ptr<atom::IonizationState> ionization_state =
        std::make_shared<atom::IonizationState>(
            grid->n_elements() + 2, nNodes, ncomps, max_charge + 1, saha_ncomps,
            fn_ionization, fn_deg,
            pin->param()->get<std::string>("ionization.solver"));

    // If we want to default the ionization fractions to anything other
    // than 0, we can do it below. i.e., for species that we are not doing
    // Saha solves, what is their default ionization state?
    // This can probably be removed. (Zbar must be initialized)
    auto ionization_states = ionization_state->ionization_fractions();
    auto zbar = ionization_state->zbar();
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Supernova :: Default Ionization",
        DevExecSpace(), ib.s - 1, ib.e + 1, KOKKOS_LAMBDA(const int i) {
          for (int q = 0; q < nNodes + 2; ++q) {
            for (int elem = 0; elem < ncomps; ++elem) {
              const int Z = charges(elem);
              if (Z == 0) {
                continue;
              }

              ionization_states(i, q, elem, Z) = 1.0;
              zbar(i, q, elem) = Z;
            }
          }
        });

    // Renormalize mass fractions. We try to enforce nodal conservation.
    // I compute the sum of all mass fractions on every nodal location
    // and then use it to renormalize the cell average of each species.
    // It is unclear if this is the most DG way to do it, but it seems to work.
    // We demand that the mass fractions of ni56 remain unchanged. The
    // normalization factor is (1 - X_Ni) / (Sum_X - X_Ni) and applied
    // to all species except Ni56.
    auto phi_fluid = fluid_basis.phi();
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN,
        "Pgen :: Supernova :: Renormalize mass fractions", DevExecSpace(), ib.s,
        ib.e, 0, nNodes - 1, KOKKOS_LAMBDA(const int i, const int q) {
          double sum_x = 0.0;
          for (int e = 0; e < ncomps; ++e) {
            const double x_q =
                basis::basis_eval(phi_fluid, mass_fractions, i, e, q);
            sum_x += x_q;
          }

          // Normalize all species except Ni56
          const int ind_ni = species_indexer->get<int>("ni56");
          const double x_ni =
              mass_fractions(i, vars::modes::CellAverage, ind_ni);
          for (int e = 0; e < ncomps; ++e) {
            if (e != ind_ni) {
              mass_fractions(i, vars::modes::CellAverage, e) *=
                  (1.0 - x_ni) / (sum_x - x_ni);
            }
          }
        });

    mesh_state.setup_composition(comps);
    mesh_state.setup_ionization(ionization_state);

    // We need to do the nodal to modal projection for specific volume first
    // and separate, as a full modal basis representation of tau is needed
    // for doing the Saha solve.
    // NOTE: Here we also project the pressure onto a modal basis
    // and then _back_ onto nodes. This is to ensure that the pressure
    // representation on an elemenent is consistent with the new density one,
    // i.e., if density is constant over an element so too will pressure.
    auto sqrt_gm = grid->sqrt_gm();
    auto dr = grid->widths();
    auto mkk_fluid = fluid_basis.mass_matrix();
    AthelasArray2D<double> tau_cell("supernova :: tau cell", nx + 2, nNodes);
    AthelasArray2D<double> pressure_cell("supernova :: pressure cell (modal)",
                                         nx + 2, order);
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Supernova :: Project tau",
        DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          // loop over nodes on an element, interpolate to nodal positions
          for (int q = 0; q < nNodes; ++q) {
            tau_cell(i, q) = 1.0 / uPF(i, q, vars::prim::Rho);
          }
          // Project the nodal representation to a modal one
          // Compute L2 projection: <f_q, phi_k> / <phi_k, phi_k>
          // This should probably be a function.
          for (int k = 0; k < order; k++) {
            double numerator_tau = 0.0;
            double numerator_pre = 0.0;
            const double denominator = mkk_fluid(i, k);

            // Compute <f_q, phi_k>
            for (int q = 0; q < nNodes; q++) {
              const double nodal_val_tau = tau_cell(i, q);
              const double nodal_val_pre = uAF(i, q, vars::aux::Pressure);
              // Curious.. is it okay to allow rho to vary over the cell when
              // doing this?
              const double rho = uPF(i, q + 1, vars::prim::Rho);

              const double int_factor = phi_fluid(i, q + 1, k) * weights(q) *
                                        dr(i) * sqrt_gm(i, q + 1) * rho;
              numerator_tau += nodal_val_tau * int_factor;
              numerator_pre += nodal_val_pre * int_factor;
            }
            uCF(i, k, vars::cons::SpecificVolume) = numerator_tau / denominator;
            pressure_cell(i, k) = numerator_pre / denominator;
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
              pressure_cell(i, k) *= std::exp(-k);
            }
          }
          // Now project the pressure _back_ onto nodes
          for (int q = 0; q < nNodes + 2; ++q) {
            uAF(i, q, vars::aux::Pressure) =
                basis::basis_eval(phi_fluid, pressure_cell, i, q);
          }
        });

    // Compute necessary terms for using the Paczynski eos
    auto sd0 = mesh_state(0);
    atom::fill_derived_comps<Domain::Interior>(sd0, grid);

    // Finally, compute the radhydro variables. This requires, as before,
    // interpolation of the progenitor data to nodal collocation points
    // and projection from nodal to modal representation.
    // For the fluid energy we use the progenitor pressure and invert
    // the eos to get a specific internal energy. This helps maintain
    // any pressure based structure. We do the radiation energy at the end,
    // once we have recomputed the gas temperature

    // Per cell nodal storage
    AthelasArray2D<double> vel_cell("supernova :: vel cell", nx + 2, nNodes);
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Supernova :: Project RadHydro vars",
        DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          // Project each conserved variable
          // loop over nodes on an element, interpolate to nodal positions
          for (int q = 0; q < nNodes; ++q) {
            const double rq = r(i, q + 1);
            const int idx =
                utilities::find_closest_cell(radius_view, rq, n_zones_prog);
            tau_cell(i, q) = basis::basis_eval(phi_fluid, uCF, i,
                                               vars::cons::SpecificVolume, q);
            vel_cell(i, q) = utilities::LINTERP(
                radius_view(idx), radius_view(idx + 1), velocity_view(idx),
                velocity_view(idx + 1), rq);
          }

          // Project the nodal representation to a modal one
          // Compute L2 projection: <f_q, phi_k> / <phi_k, phi_k>
          // This should probably be a function.
          for (int k = 0; k < order; k++) {
            double numerator_vel = 0.0;
            const double denominator = mkk_fluid(i, k);

            // Compute <f_q, phi_k>
            for (int q = 0; q < nNodes; q++) {
              const double nodal_vel = vel_cell(i, q);
              const double rho = uPF(i, q + 1, vars::prim::Rho);

              numerator_vel += nodal_vel * phi_fluid(i, q + 1, k) * weights(q) *
                               dr(i) * sqrt_gm(i, q + 1) * rho;
            }
            uCF(i, k, vars::cons::Velocity) = numerator_vel / denominator;
            // Exponential filter
            if (k > 0) {
              uCF(i, k, vars::cons::Velocity) *= std::exp(-k);
            }
          }
        });

    if (ionization_state->solver() == atom::SahaSolver::Linear) {
      atom::compute_temperature_with_saha<Domain::Interior,
                                          eos::EOSInversion::Pressure,
                                          atom::SahaSolver::Linear>(sd0, *grid);
    }
    if (ionization_state->solver() == atom::SahaSolver::Log) {
      atom::compute_temperature_with_saha<
          Domain::Interior, eos::EOSInversion::Pressure, atom::SahaSolver::Log>(
          sd0, *grid);
    }

    AthelasArray2D<double> energy_cell("supernova :: energy cell", nx + 2,
                                       nNodes);
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Supernova :: Project sie",
        DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          double lambda[8];
          // Project each conserved variable
          // loop over nodes on an element, interpolate to nodal positions
          for (int q = 0; q < nNodes; ++q) {
            tau_cell(i, q) = basis::basis_eval(phi_fluid, uCF, i,
                                               vars::cons::SpecificVolume, q);
            const double temperature_q = uAF(i, q + 1, vars::aux::Tgas);
            atom::paczynski_terms(sd0, i, q, lambda);
            energy_cell(i, q) =
                eos::sie_from_density_temperature(eos, 1.0 / tau_cell(i, q),
                                                  temperature_q, lambda) +
                0.5 * vel_cell(i, q) * vel_cell(i, q);
          }

          // Project the nodal representation to a modal one
          // Compute L2 projection: <f_q, phi_k> / <phi_k, phi_k>
          // This should probably be a function.
          for (int k = 0; k < order; k++) {
            double numerator_energy = 0.0;
            const double denominator = mkk_fluid(i, k);

            // Compute <f_q, phi_k>
            for (int q = 0; q < nNodes; q++) {
              const double nodal_energy = energy_cell(i, q);
              const double rho = uPF(i, q + 1, vars::prim::Rho);

              numerator_energy += nodal_energy * phi_fluid(i, q + 1, k) *
                                  weights(q) * dr(i) * sqrt_gm(i, q + 1) * rho;
            }
            uCF(i, k, vars::cons::Energy) = numerator_energy / denominator;
            // Exponential filter
            if (k > 0) {
              uCF(i, k, vars::cons::Energy) *= std::exp(-k);
            }
          }
        });

    // Setup radiation variables as needed
    // Note: the inner product used here is different than for the fluid:
    // we don't use density.
    if (rad_enabled) {
      const auto &rad_basis = mesh_state.rad_basis();
      AthelasArray2D<double> rad_energy_cell("supernova :: rad energy cell",
                                             nx + 2, nNodes);
      AthelasArray2D<double> rad_flux_cell("supernova :: rad flux cell", nx + 2,
                                           nNodes);
      auto phi_rad = rad_basis.phi();
      auto mkk_rad = rad_basis.mass_matrix();
      athelas::par_for(
          DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Supernova :: Project Rad energy",
          DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
            // Project each conserved variable
            // loop over nodes on an element, interpolate to nodal positions
            for (int q = 0; q < nNodes; ++q) {
              const double rq = r(i, q + 1);
              const int idx =
                  utilities::find_closest_cell(radius_view, rq, n_zones_prog);
              rad_energy_cell(i, q) =
                  radiation::rad_energy(uAF(i, q, vars::aux::Tgas));
              rad_flux_cell(i, q) =
                  utilities::LINTERP(radius_view(idx), radius_view(idx + 1),
                                     luminosity_view(idx),
                                     luminosity_view(idx + 1), rq) /
                  (constants::FOURPI * rq * rq);
            }

            // Project the nodal representation to a modal one
            for (int k = 0; k < order; k++) {
              double numerator_energy = 0.0;
              double numerator_flux = 0.0;
              const double denominator = mkk_rad(i, k);

              // Compute <f_q, phi_k>
              for (int q = 0; q < nNodes; q++) {
                numerator_energy += rad_energy_cell(i, q) *
                                    phi_rad(i, q + 1, k) * weights(q) * dr(i) *
                                    sqrt_gm(i, q + 1);
                numerator_flux += rad_flux_cell(i, q) * phi_rad(i, q + 1, k) *
                                  weights(q) * dr(i) * sqrt_gm(i, q + 1);
              }
              uCF(i, k, vars::cons::RadEnergy) = numerator_energy / denominator;
              uCF(i, k, vars::cons::RadFlux) = numerator_flux / denominator;
              // Exponential filter
              if (k > 0) {
                uCF(i, k, vars::cons::RadEnergy) *= std::exp(-k);
                uCF(i, k, vars::cons::RadFlux) *= 0.0; // std::exp(-k);
              }
            }
          });
    }

    int nvars = rad_enabled ? 5 : 3;
    // composition boundary condition
    static const IndexRange vb_comps(std::make_pair(nvars, nvars + ncomps - 1));
    bc::fill_ghost_zones_composition(uCF, vb_comps);

    // now let us offset the enclosed mass by the mass cut
    if (mass_cut != 0.0) {
      auto menc = grid->enclosed_mass();
      athelas::par_for(
          DEFAULT_FLAT_LOOP_PATTERN,
          "Pgen :: Supernova :: Adjust enclosed mass", DevExecSpace(), ib.s,
          ib.e, KOKKOS_LAMBDA(const int i) {
            for (int q = 0; q < nNodes; q++) {
              menc(i, q) += mass_cut * constants::M_sun;
            }
          });
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
