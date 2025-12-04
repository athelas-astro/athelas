#pragma once

#include "Kokkos_Macros.hpp"
#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "constants.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "loop_layout.hpp"
#include "state/state.hpp"

namespace athelas::atom {

void paczynski_terms(const State *state, int ix, int node, double *lambda);

/**
 * @brief Number density of atomic species in particles/cm^3
 */
KOKKOS_INLINE_FUNCTION
auto element_number_density(const double mass_frac, const double atomic_mass,
                            const double rho) -> double {
  return (mass_frac * rho) / (atomic_mass * constants::amu_to_g);
}

// Compute electron number density (derived quantity)
auto electron_density(const AthelasArray3D<double> mass_fractions,
                      const AthelasArray4D<double> ion_fractions,
                      const AthelasArray1D<int> charges, int i, int q,
                      double rho) -> double;

/**
 * @brief Fill derived composition quantities
 *
 * Currently, fills number densities and electron fraction.
 *
 * TODO(astrobarker): Explore hierarchical parallelism for inner loops
 */
template <Domain MeshDomain>
void fill_derived_comps(State *const state, const GridStructure *const grid,
                        const basis::ModalBasis *const basis) {
  static const auto &nnodes = grid->n_nodes();
  static const IndexRange ib(grid->domain<MeshDomain>());
  static const IndexRange nb(nnodes + 2);

  auto phi = basis->phi();

  auto ucf = state->u_cf();

  auto *const comps = state->comps();
  const auto mass_fractions = state->mass_fractions();
  const auto species = comps->charge();
  const auto neutron_number = comps->neutron_number();
  auto ye = comps->ye();
  auto abar = comps->abar();
  auto number_density = comps->number_density();
  const size_t num_species = comps->n_species();

  static constexpr double inv_m_p = 1.0 / constants::m_p;
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Composition :: fill derived", DevExecSpace(), ib.s,
      ib.e, nb.s, nb.e, KOKKOS_LAMBDA(const int i, const int q) {
        double n = 0.0;
        double ye_q = 0.0;
        double sum_y = 0.0;
        for (size_t e = 0; e < num_species; ++e) {
          const double Z = species(e);
          const double A = Z + neutron_number(e);
          const double rho =
              1.0 /
              basis::basis_eval(phi, ucf, i, vars::cons::SpecificVolume, q);
          const double xk = basis::basis_eval(phi, mass_fractions, i, e, q);
          n += xk / A;
          ye_q += Z * xk / A;
          sum_y += element_number_density(xk, A, rho) / (rho * constants::N_A);
        }
        number_density(i, q) = n * inv_m_p;
        ye(i, q) = ye_q;
        abar(i, q) = 1.0 / sum_y;
      });
}

// per-point version called by the below
KOKKOS_INLINE_FUNCTION
auto fill_derived_ionization(const basis::ModalBasis *basis,
                             const AthelasArray3D<double> mass_fractions,
                             const CompositionData *comps,
                             const IonizationState *ionization_state,
                             const IndexRange &eb, const int &num_species,
                             const int &i, const int &q)
    -> std::tuple<double, double, double> {
  const auto ionization_fractions = ionization_state->ionization_fractions();
  const auto *const atomic_data = ionization_state->atomic_data();

  const auto ion_data = atomic_data->ion_data();
  const auto species_offsets = atomic_data->offsets();
  const auto species = comps->charge();
  const auto neutron_number = comps->neutron_number();
  const auto number_density = comps->number_density();
  // const auto electron_number_density = comps->electron_number_density();
  const auto abar = comps->abar();

  double sum1 = 0.0;
  double sum2 = 0.0;
  double sum3 = 0.0;
  for (int e = eb.s; e <= eb.e; ++e) {
    // pull out element info
    const int z = species(e);
    const auto species_atomic_data = species_data(ion_data, species_offsets, z);
    const auto ionization_fractions_e =
        Kokkos::subview(ionization_fractions, i, q, e, Kokkos::ALL);
    const int nstates = z + 1;

    // max ionization fraction
    double y_r = 0;
    double chi_r = 0.0;
    for (int s = 0; s < nstates; ++s) {
      const double y = ionization_fractions_e(s);
      // chi_r_new = (y > ymax) * species_atomic_data(s).chi;
      // ymax = std::max(y, ymax);
      if (y > y_r) {
        y_r = y;
        chi_r = species_atomic_data(s).chi;
      }
    }

    // 4. The good stuff -- integrate the various sigma terms
    // and the internal energy term from partial ionization.
    // Start with constructing the abundance n_k

    const double atomic_mass = z + neutron_number(e);
    const double xk = basis->basis_eval(mass_fractions, i, e, q);
    const double nu_k = abar(i, q) * xk / atomic_mass;
    const double term = nu_k * y_r * (1.0 - y_r);
    sum1 += term;
    sum2 += chi_r * term;
    sum3 += chi_r * chi_r * term;
  } // loop species
  return {sum1, sum2, sum3};
}

/**
 * @brief Fill derived ionization quantities
 *
 * These are quantities needed for the Paczynski eos.
 *
 * TODO(astrobarker): Explore hierarchical parallelism for inner loops
 */
template <Domain MeshDomain>
void fill_derived_ionization(State *const state,
                             const GridStructure *const grid,
                             const basis::ModalBasis *const basis) {
  static const auto &nnodes = grid->n_nodes();
  static const IndexRange ib(grid->domain<MeshDomain>());
  static const IndexRange nb(nnodes + 2);

  const auto *const comps = state->comps();
  const auto mass_fractions = state->mass_fractions();
  const auto number_density = comps->number_density();
  const auto electron_number_density = comps->electron_number_density();
  const size_t num_species = comps->n_species();
  auto *species_indexer = comps->species_indexer();

  auto *const ionization_state = state->ionization_state();
  static const auto &ncomps_saha = ionization_state->ncomps();
  auto ybar = ionization_state->ybar();
  auto sigma1 = ionization_state->sigma1();
  auto sigma2 = ionization_state->sigma2();
  auto sigma3 = ionization_state->sigma3();

  static const bool has_neuts = species_indexer->contains("neut");
  static const int start_elem = (has_neuts) ? 1 : 0;

  static const int end_elem = (has_neuts) ? ncomps_saha : ncomps_saha - 1;
  static const IndexRange eb_saha(std::make_pair(start_elem, end_elem));

  static const IndexRange eb(std::make_pair(start_elem, num_species - 1));

  const auto ucf = state->u_cf();

  // NOTE: check index ranges inside here when saha ncomps =/= num_species
  // Should we be skipping neutrons?
  auto phi = basis->phi();
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Ionization :: fill derived", DevExecSpace(), ib.s,
      ib.e, nb.s, nb.e, KOKKOS_LAMBDA(const int i, const int q) {
        const double rho =
            1.0 / basis::basis_eval(phi, ucf, i, vars::cons::SpecificVolume, q);
        // This kernel is horrible.
        // Reduce the ionization based quantities sigma1-3, e_ion_corr
        ybar(i, q) = electron_number_density(i, q) / number_density(i, q) / rho;
        const auto [s1, s2, s3] = fill_derived_ionization(
            basis, mass_fractions, comps, ionization_state, eb_saha,
            num_species, i, q);
        sigma1(i, q) = s1;
        sigma2(i, q) = s2;
        sigma3(i, q) = s3;
      });
}

} // namespace athelas::atom
