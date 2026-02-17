/**
 * @file progenitor.hpp
 * --------------
 *
 * @brief Supernova progenitor initialization
 *
 * This one is the most involved of all pgens.
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
 */

#pragma once

#include <cmath>

#include "Kokkos_Core.hpp"
#include "basic_types.hpp"
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
void progenitor_init(MeshState &mesh_state, GridStructure *grid,
                     ProblemIn *pin) {
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

  // Initialize nodal values
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
           (std::pow(radius_host(i + 1), 3.0) - std::pow(radius_host(i), 3.0)) /
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

  // --- Rebuild the grid after making the mass cut. ---
  auto &xl = pin->param()->get_mutable_ref<double>("problem.xl");
  auto &xr = pin->param()->get_mutable_ref<double>("problem.xr");
  xl = r_cut;
  xr = rstar;
  auto newgrid = GridStructure(pin);
  *grid = newgrid;


  auto r = grid->nodal_grid();

  // --- Interpolate density, pressure, temperature at nodes & interfaces. ---
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Supernova (1)", DevExecSpace(), ib.s,
      ib.e, KOKKOS_LAMBDA(const int i) {
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

  // We go ahead and form the basis here now that the grid is constructed.
  // I don't particularly like this pattern, but as it stands the basis is 
  // needed in the Saha solves to come.
  auto fluid_basis_tmp = std::make_unique<basis::NodalBasis>(
      uPF, grid,
      pin->param()->get<int>("fluid.nnodes"),
      pin->param()->get<int>("problem.nx"), false);
  mesh_state.setup_fluid_basis(std::move(fluid_basis_tmp));

  // Phase 2: Everything else
  const auto &eos = mesh_state.eos();
  auto species = comps->charge();
  auto neutron_number = comps->neutron_number();
  auto inv_atomic_mass = comps->inverse_atomic_mass();
  auto species_h = Kokkos::create_mirror_view(species);
  auto neutron_number_h = Kokkos::create_mirror_view(neutron_number);
  auto inv_atomic_mass_h = Kokkos::create_mirror_view(inv_atomic_mass);

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
  // Also grab, if present, index for neutrons. Grab max charge.
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

  // Now we make sure that if neutrons are in our composition data
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

  // --- Interpolate mass fractions onto our grid. ---
  // TODO(astrobarker): this can be device side.
  auto mass_fractions_h =
      Kokkos::create_mirror_view(mesh_state.mass_fractions("u_cf"));
  auto r_h = Kokkos::create_mirror_view(r);
  for (int i = ib.s; i <= ib.e; ++i) {
    for (int e = 0; e < ncomps; ++e) {
      // loop over nodes on an element, interpolate to nodal positions
      for (int q = 0; q < nNodes; ++q) {
        const double rq = r_h(i, q + 1);
        const int idx =
            utilities::find_closest_cell(radius_host, rq, n_zones_prog);
        mass_fractions_h(i, q, e) = utilities::LINTERP(radius_host(idx), radius_host(idx + 1),
                                       comps_star_host(idx, e),
                                       comps_star_host(idx + 1, e), rq);
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
  // We demand that the mass fractions of ni56 remain unchanged. The
  // normalization factor is (1 - X_Ni) / (Sum_X - X_Ni) and applied
  // to all species except Ni56.
  const int ind_ni = species_indexer->get<int>("ni56");
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN,
      "Pgen :: Supernova :: Renormalize mass fractions", DevExecSpace(), ib.s,
      ib.e, 0, nNodes - 1, KOKKOS_LAMBDA(const int i, const int q) {
        double sum_x = 0.0;
        for (int e = 0; e < ncomps; ++e) {
          const double x_q = mass_fractions(i, q, e);
          sum_x += x_q;
        }

        // Normalize all species except Ni56
        const double x_ni = mass_fractions(i, q, ind_ni);
        for (int e = 0; e < ncomps; ++e) {
          if (e != ind_ni) {
            mass_fractions(i, q, e) *=
                (1.0 - x_ni) / (sum_x - x_ni);
          }
        }
      });

  mesh_state.setup_composition(comps);
  mesh_state.setup_ionization(ionization_state);

  // Set the specific volume
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Supernova :: Tau",
      DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
        for (int q = 0; q < nNodes; ++q) {
          uCF(i, q, vars::cons::SpecificVolume) = 1.0 / uPF(i, q+1, vars::prim::Rho);
        }
      });

  // Finally, interpolate the remaining radhydro variables.
  // For the fluid energy we use the progenitor pressure and invert
  // the eos to get a specific internal energy. This helps maintain
  // any pressure based structure. We do the radiation energy at the end,
  // once we have recomputed the gas temperature
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Supernova :: Velocity",
      DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
        for (int q = 0; q < nNodes; ++q) {
          const double rq = r(i, q + 1);
          const int idx =
              utilities::find_closest_cell(radius_view, rq, n_zones_prog);
          uCF(i, q, vars::cons::Velocity) = utilities::LINTERP(
              radius_view(idx), radius_view(idx + 1), velocity_view(idx),
              velocity_view(idx + 1), rq);
        }
      });

  // There is one subtelty that must be taken care of:
  // We have interpolated pressure, density, temperature etc onto element
  // interfaces. Those values are not guaranteed to be consistent with the 
  // basis representation. Imagine a first order element: the interface values
  // on the element must equal the cell center value. This must be enforced 
  // via the basis before the interface values are used. It needs to be done 
  // before the first fill derived call. This could be a good candidate for a 
  // post_init kernel..
  const auto &basis = mesh_state.fluid_basis(); 
  auto phi = basis.phi();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Supernova :: Consistent interfaces",
      DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          uAF(i, 0, vars::aux::Pressure) = basis::basis_eval<Interface::Left>(phi, uAF, i, vars::aux::Pressure);
          uAF(i, nNodes + 1, vars::aux::Pressure) = basis::basis_eval<Interface::Right>(phi, uAF, i, vars::aux::Pressure);
          uAF(i, 0, vars::aux::Tgas) = basis::basis_eval<Interface::Left>(phi, uAF, i, vars::aux::Tgas);
          uAF(i, nNodes + 1, vars::aux::Tgas) = basis::basis_eval<Interface::Right>(phi, uAF, i, vars::aux::Tgas);
          uPF(i, 0, vars::prim::Rho) = basis::basis_eval<Interface::Left>(phi, uPF, i, vars::prim::Rho);
          uPF(i, nNodes + 1, vars::prim::Rho) = basis::basis_eval<Interface::Right>(phi, uPF, i, vars::prim::Rho);
      });

  // Compute necessary terms for using the Paczynski eos
  auto sd0 = mesh_state(0);
  atom::fill_derived_comps<Domain::Interior>(sd0, grid);


  // Perform an eos inversion using the progenitor pressure for 
  // the updated temperature. It is coupled to a Saha solver for the 
  // ionization state.
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

  auto number_density = comps->number_density();
  auto ybar = ionization_state->ybar();
  auto e_ion_corr = ionization_state->e_ion_corr();
  auto sigma1 = ionization_state->sigma1();
  auto sigma2 = ionization_state->sigma2();
  auto sigma3 = ionization_state->sigma3();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Supernova :: Project sie",
      DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
        for (int q = 0; q < nNodes; ++q) {
          const double temperature_q = uAF(i, q + 1, vars::aux::Tgas);
          const double vel = uCF(i, q, vars::cons::Velocity);
            eos::EOSLambda lambda;
            lambda.data[0] = number_density(i, q + 1);
            lambda.data[1] = ye(i, q + 1);
            lambda.data[2] = ybar(i, q + 1);
            lambda.data[3] = sigma1(i, q + 1);
            lambda.data[4] = sigma2(i, q + 1);
            lambda.data[5] = sigma3(i, q + 1);
            lambda.data[6] = e_ion_corr(i, q + 1);
            lambda.data[7] = temperature_q;
          uCF(i, q, vars::cons::Energy) =
              eos::sie_from_density_temperature(eos, 1.0 / uCF(i, q, vars::cons::SpecificVolume),
                                                temperature_q, lambda.ptr()) +
              0.5 * vel * vel;
        }
      });

  // Setup radiation variables as needed
  // Radiation energy is set in equilibrium aT_g^4
  // Radiaiton flux is currently set using the progenitor luminosity
  // TODO(astrobarker): Support initializing with 0 flux.
  if (rad_enabled) {
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Supernova :: Radiation",
        DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          for (int q = 0; q < nNodes; ++q) {
            const double rq = r(i, q + 1);
            const int idx =
                utilities::find_closest_cell(radius_view, rq, n_zones_prog);
            uCF(i, q, vars::cons::RadEnergy) =
                radiation::rad_energy(uAF(i, q, vars::aux::Tgas));
            uCF(i, q, vars::cons::RadFlux) =
                utilities::LINTERP(radius_view(idx), radius_view(idx + 1),
                                   luminosity_view(idx),
                                   luminosity_view(idx + 1), rq) /
                (constants::FOURPI * rq * rq);
          }
        });
  }

  int nvars = rad_enabled ? 5 : 3;
  // composition boundary condition
  static const IndexRange vb_comps(std::make_pair(nvars, nvars + ncomps - 1));
  bc::fill_ghost_zones_composition(uCF, vb_comps);

  // Now let us offset the enclosed mass by the mass cut
  if (mass_cut != 0.0) {
    auto menc = grid->enclosed_mass();
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Supernova :: Adjust enclosed mass",
        DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
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
