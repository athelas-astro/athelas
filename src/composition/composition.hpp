#pragma once

#include "Kokkos_Macros.hpp"
#include "basis/polynomial_basis.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "loop_layout.hpp"
#include "state/state.hpp"

namespace athelas::atom {

void paczynski_terms(const State *state, int ix, int node, double *lambda);

// Compute total element number density
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

  auto *const comps = state->comps();
  const auto mass_fractions = state->mass_fractions();
  const auto species = comps->charge();
  const auto neutron_number = comps->neutron_number();
  auto ye = comps->ye();
  auto number_density = comps->number_density();
  const size_t num_species = comps->n_species();

  static constexpr double inv_m_p = 1.0 / constants::m_p;
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Composition :: fill derived", DevExecSpace(), ib.s,
      ib.e, nb.s, nb.e, KOKKOS_LAMBDA(const int i, const int q) {
        double n = 0.0;
        double ye_q = 0.0;
        for (size_t e = 0; e < num_species; ++e) {
          const double Z = species(e);
          const double A = Z + neutron_number(e);
          const double xk = basis::basis_eval(phi, mass_fractions, i, e, q);
          n += xk / A;
          ye_q += Z * xk / A;
        }
        number_density(i, q) = n * inv_m_p;
        ye(i, q) = ye_q;
      });
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
  const auto species = comps->charge();
  const auto neutron_number = comps->neutron_number();
  const auto number_density = comps->number_density();
  const auto electron_number_density = comps->electron_number_density();
  const size_t num_species = comps->n_species();

  auto *const ionization_states = state->ionization_state();
  const auto ionization_fractions = ionization_states->ionization_fractions();
  auto ybar = ionization_states->ybar();
  auto e_ion_corr = ionization_states->e_ion_corr();
  auto sigma1 = ionization_states->sigma1();
  auto sigma2 = ionization_states->sigma2();
  auto sigma3 = ionization_states->sigma3();

  // pull out atomic data containers
  const auto *const atomic_data = ionization_states->atomic_data();
  const auto ion_data = atomic_data->ion_data();
  const auto species_offsets = atomic_data->offsets();

  const auto ucf = state->u_cf();

  // NOTE: check index ranges inside here when saha ncomps =/= num_species
  auto phi = basis->phi();
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Ionization :: fill derived", DevExecSpace(), ib.s,
      ib.e, nb.s, nb.e, KOKKOS_LAMBDA(const int i, const int q) {
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;
        double sum_e_ion_corr = 0.0;
        const double rho =
            1.0 / basis::basis_eval(phi, ucf, i, vars::cons::SpecificVolume, q);
        // This kernel is horrible.
        // Reduce the ionization based quantities sigma1-3, e_ion_corr
        ybar(i, q) = electron_number_density(i, q) / number_density(i, q);
        for (size_t e = 0; e < num_species; ++e) {
          // pull out element info
          const auto species_atomic_data =
              species_data(ion_data, species_offsets, e);
          const auto ionization_fractions_e =
              Kokkos::subview(ionization_fractions, i, q, e, Kokkos::ALL);
          const size_t nstates = e + 1;

          // 1. Get lmax -- index associated with max ionization per species
          size_t lmax = 0;
          double ymax = 0;
          for (size_t i = 0; i < nstates; ++i) {
            const double y = ionization_fractions_e(i);
            if (y > ymax) {
              ymax = y;
              lmax = i;
            }
          }

          // 2. Sum ionization fractions * ionization potentials for e_ion_corr
          double sum_ion_pot = 0.0;
          for (size_t i = 0; i < nstates; ++i) {
            // I think that this pattern is not optimal.
            double sum_pot = 0.0;
            for (size_t m = 0; m < i; ++m) {
              sum_pot += species_atomic_data(i).chi;
            }
            sum_ion_pot += ionization_fractions_e(i) * sum_pot;
          }

          // 3. Find two most populated states and store the higher as y_r.
          // chi_r is the ionization potential between these states.
          // Check index logic.
          // Wish I could avoid branching logic...
          double y_r = 0;
          double chi_r = 0.0;
          if (lmax == 0) {
            y_r = ionization_fractions_e(lmax);
            chi_r = species_atomic_data(lmax).chi;
          } else if (lmax == (e + 0)) {
            y_r = ionization_fractions_e(lmax);
            chi_r = species_atomic_data(lmax - 1).chi;
          } else {
            // Comparison between lmax+1 and lmax-1 indices
            if (ionization_fractions_e(lmax + 1) >
                ionization_fractions_e(lmax - 1)) {
              y_r = ionization_fractions_e(lmax + 1);
              chi_r = species_atomic_data(lmax).chi;
            } else {
              y_r = ionization_fractions_e(lmax);
              chi_r = species_atomic_data(lmax - 1).chi;
            }
          }

          // 4. The good stuff -- integrate the various sigma terms
          // and the internal energy term from partial ionization.
          // Start with constructing the abundance n_k

          const double atomic_mass = species(e) + neutron_number(e);
          const double xk = basis->basis_eval(mass_fractions, i, e, q);
          const double nk = element_number_density(xk, atomic_mass, rho);
          sum1 += nk * y_r * (1 - y_r); // sigma1
          sum2 += chi_r * sigma1(i, q); // sigma2
          sum3 += chi_r * sigma2(i, q); // sigma3
          sum_e_ion_corr +=
              number_density(i, q) * nk * sum_ion_pot; // e_ion_corr
          std::println(
              "e_ion_corr :: i N nu_j sumpot sumcorr {} {} {} {} {:.5e}", i,
              number_density(i, q), nk, sum_ion_pot, sum_e_ion_corr);
        }
        sigma1(i, q) = sum1;
        sigma2(i, q) = sum2;
        sigma3(i, q) = sum3;
        e_ion_corr(i, q) = sum_e_ion_corr;
      });
}

} // namespace athelas::atom
