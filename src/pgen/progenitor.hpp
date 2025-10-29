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
 */

#pragma once

#include <cmath>

#include "Kokkos_Core.hpp"
#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
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
  static constexpr int NUM_COLS_HYDRO = 4;
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

  if (!hydro_data) {
    THROW_ATHELAS_ERROR("Error reading hydro profile!");
  }

  // Safety check on the structure of the profile.
  const size_t num_columns_hydro = hydro_data->rows[0].size();
  if (num_columns_hydro != NUM_COLS_HYDRO) {
    THROW_ATHELAS_ERROR("Wrong number of columns in hydro profile!");
  }

  auto [radius_star, density_star, velocity_star, pressure_star] =
      io::get_columns_by_indices<double, double, double, double>(*hydro_data, 0,
                                                                 1, 2, 3);

  const double rstar = radius_star.back();
  const int n_zones_prog = static_cast<int>(radius_star.size());

  // Create Kokkos views
  AthelasArray1D<double> radius_view("progenitor radius", n_zones_prog);
  AthelasArray1D<double> density_view("progenitor density", n_zones_prog);
  AthelasArray1D<double> velocity_view("progenitor velocity", n_zones_prog);
  AthelasArray1D<double> pressure_view("progenitor pressure", n_zones_prog);

  // Create host mirrors for data transfer
  auto radius_host = Kokkos::create_mirror_view(radius_view);
  auto density_host = Kokkos::create_mirror_view(density_view);
  auto velocity_host = Kokkos::create_mirror_view(velocity_view);
  auto pressure_host = Kokkos::create_mirror_view(pressure_view);

  // Copy data from vectors to host mirrors
  for (int i = 0; i < n_zones_prog; ++i) {
    radius_host(i) = radius_star[i];
    density_host(i) = density_star[i];
    velocity_host(i) = velocity_star[i];
    pressure_host(i) = pressure_star[i];
  }

  // Deep copy from host to device
  Kokkos::deep_copy(radius_view, radius_host);
  Kokkos::deep_copy(density_view, density_host);
  Kokkos::deep_copy(velocity_view, velocity_host);
  Kokkos::deep_copy(pressure_view, pressure_host);

  auto r = grid->nodal_grid();
  if (fluid_basis == nullptr) {
    // Phase 1: Initialize nodal values
    // We also need to re-build the mesh
    auto &xl = pin->param()->get_mutable_ref<double>("problem.xl");
    auto &xr = pin->param()->get_mutable_ref<double>("problem.xr");
    xl = radius_host[0];
    xr = rstar;
    auto newgrid = GridStructure(pin);
    *grid = newgrid;

    auto centers = grid->centers();
    auto x_l = grid->x_l();
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
        }
      }
    }

    Kokkos::deep_copy(state->mass_fractions(), mass_fractions_h);
    Kokkos::deep_copy(species, species_h);
    Kokkos::deep_copy(neutron_number, neutron_number_h);

    // Use L2 projection for accurate modal coefficients
    auto tau_func = [&](double x, int /*i*/, int /*q*/) -> double {
      const int idx =
          utilities::find_closest_cell(radius_view, x, n_zones_prog);
      const double rho =
          utilities::LINTERP(radius_view(idx), radius_view(idx + 1),
                             density_view(idx), density_view(idx + 1), x);
      return 1.0 / rho;
    };

    auto velocity_func = [&](double x, int /*i*/, int /*q*/) -> double {
      const int idx =
          utilities::find_closest_cell(radius_view, x, n_zones_prog);
      return utilities::LINTERP(radius_view(idx), radius_view(idx + 1),
                                velocity_view(idx), velocity_view(idx + 1), x);
    };

    auto energy_func = [&](double x, int /*i*/, int /*q*/) -> double {
      double lambda[8];
      const int idx =
          utilities::find_closest_cell(radius_view, x, n_zones_prog);
      const double rho =
          utilities::LINTERP(radius_view(idx), radius_view(idx + 1),
                             density_view(idx), density_view(idx + 1), x);
      const double pressure =
          utilities::LINTERP(radius_view(idx), radius_view(idx + 1),
                             pressure_view(idx), pressure_view(idx + 1), x);
      return eos::sie_from_density_pressure(eos, rho, pressure, lambda);
    };

    /*
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Supernova (2)",
        DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          // Project each conserved variable
          fluid_basis->project_nodal_to_modal(uCF, uPF, grid,
    vars::cons::SpecificVolume, i, tau_func);
          fluid_basis->project_nodal_to_modal(uCF, uPF, grid,
    vars::cons::Velocity, i, velocity_func);
          fluid_basis->project_nodal_to_modal(uCF, uPF, grid,
    vars::cons::Energy, i, energy_func);
        });
  */
  }

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Supernova (ghost)", DevExecSpace(), 0,
      ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int q = 0; q < nNodes + 2; q++) {
          uPF(ib.s - 1 - i, q, vars::prim::Rho) =
              uPF(ib.s + i, (nNodes + 2) - q - 1, vars::prim::Rho);
          uPF(ib.s + 1 + i, q, vars::prim::Rho) =
              uPF(ib.s - i, (nNodes + 2) - q - 1, vars::prim::Rho);
        }
      });
}

} // namespace athelas
