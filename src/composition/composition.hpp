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

void paczynski_terms(const StageData &stage_data, int ix, int node,
                     double *lambda);

/**
 * @brief Number density of atomic species in particles/cm^3
 */
KOKKOS_INLINE_FUNCTION
auto element_number_density(const double mass_frac,
                            const double inv_atomic_mass, const double rho)
    -> double {
  return mass_frac * rho * inv_atomic_mass * constants::N_A;
}

/**
 * @brief Fill derived composition quantities
 *
 * Currently, fills number densities and electron fraction.
 *
 * TODO(astrobarker): Explore hierarchical parallelism for inner loops
 */
template <Domain MeshDomain>
void fill_derived_comps(StageData &stage_data,
                        const GridStructure *const grid) {
  static const auto &nnodes = grid->n_nodes();
  static const IndexRange ib(grid->domain<MeshDomain>());
  static const IndexRange nb(nnodes + 2);

  const auto &basis = stage_data.fluid_basis();
  auto phi = basis.phi();

  auto ucf = stage_data.get_field("u_cf");
  auto *const comps = stage_data.comps();
  auto mass_fractions = stage_data.mass_fractions("u_cf");
  auto mass_fractions_nodal = stage_data.get_field("x_q");
  auto species = comps->charge();
  auto inv_atomic_mass = comps->inverse_atomic_mass();
  auto ye = comps->ye();
  auto abar = comps->abar();
  auto number_density = comps->number_density();
  static const int num_species = static_cast<int>(comps->n_species());

  static constexpr double inv_m_p = 1.0 / constants::m_p;
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Composition :: fill derived", DevExecSpace(), ib.s,
      ib.e, nb.s, nb.e, KOKKOS_LAMBDA(const int i, const int q) {
        double ye_q = 0.0;
        double sum_y = 0.0;
        for (int e = 0; e < num_species; ++e) {
          const double Z = species(e);
          const double inv_A = inv_atomic_mass(e);
          const double xk = basis::basis_eval(phi, mass_fractions, i, e, q);
          const double xk_invA = xk * inv_A;
          ye_q += Z * xk_invA;
          sum_y += xk_invA;
          mass_fractions_nodal(i, q, e) = xk;
        }
        number_density(i, q) = sum_y * inv_m_p;
        ye(i, q) = ye_q;
        abar(i, q) = 1.0 / (sum_y);
      });
}

// per-point version called by the below
KOKKOS_INLINE_FUNCTION
auto fill_derived_ionization(const AthelasArray3D<double> phi,
                             const AthelasArray3D<double> mass_fractions,
                             const CompositionData *comps,
                             const IonizationState *ionization_state,
                             const IndexRange &eb, const int &num_species,
                             const int &i, const int &q)
    -> std::tuple<double, double, double, double> {
  const auto ionization_fractions = ionization_state->ionization_fractions();
  const auto *const atomic_data = ionization_state->atomic_data();

  const auto ion_data = atomic_data->ion_data();
  auto sum_pots = atomic_data->sum_pots();
  const auto species_offsets = atomic_data->offsets();
  const auto species = comps->charge();
  const auto neutron_number = comps->neutron_number();
  const auto number_density = comps->number_density();
  const auto abar = comps->abar();
  const auto inv_atomic_mass = comps->inverse_atomic_mass();

  double sum1 = 0.0;
  double sum2 = 0.0;
  double sum3 = 0.0;
  double sum_e_ion_corr = 0.0;
  // Loop over Saha species – solve ionization for the Saha subset
  for (int e = eb.s; e <= eb.e; ++e) {
    const int z = species(e);
    const double x_e = basis::basis_eval(phi, mass_fractions, i, e, q);

    const double inv_A = inv_atomic_mass(e);

    // species atomic data
    const auto species_atomic_data = species_data(ion_data, species_offsets, z);
    const auto species_sum_pot = species_data(sum_pots, species_offsets, z);

    auto ionization_fractions_e =
        Kokkos::subview(ionization_fractions, i, q, e, Kokkos::ALL);

    // reset ionization fractions, setup saha factors

    const int nstates = z + 1;

    // Paczynski sigma terms: same logic as per-point fill_derived_ionization
    // y_r = max populated ionization fraction, chi_r corresponding potential
    double y_r = ionization_fractions_e(0);
    double chi_r = 0.0;
    double sum_ion_pot = 0.0;
    for (int s = 1; s < nstates; ++s) {
      sum_ion_pot += ionization_fractions_e(s) * species_sum_pot(s - 1);
      const double y = ionization_fractions_e(s);
      if (y > y_r) {
        y_r = y;
        chi_r = species_atomic_data(s).chi;
      }
    }

    const double nu_k = abar(i, q) * x_e * inv_A;
    const double term = nu_k * y_r * (1.0 - y_r);
    sum1 += term;
    sum2 += chi_r * term;
    sum3 += chi_r * chi_r * term;
    sum_e_ion_corr += number_density(i, q) * nu_k * sum_ion_pot;
  }

  return {sum1, sum2, sum3, sum_e_ion_corr};
}

/**
 * @brief Fill derived ionization quantities
 *
 * These are quantities needed for the Paczynski eos.
 *
 * NOTE: This is currently unused in performance critical sections.
 * If this changes then the inner looping needs to be optimized.
 */
template <Domain MeshDomain>
void fill_derived_ionization(StageData &stage_data,
                             const GridStructure *const grid) {
  static const auto &nnodes = grid->n_nodes();
  static const IndexRange ib(grid->domain<MeshDomain>());
  static const IndexRange nb(nnodes + 2);

  auto ucf = stage_data.get_field("u_cf");
  const auto *const comps = stage_data.comps();
  const auto mass_fractions = stage_data.mass_fractions("u_cf");
  const auto number_density = comps->number_density();
  const auto electron_number_density = comps->electron_number_density();
  const size_t num_species = comps->n_species();
  auto *species_indexer = comps->species_indexer();

  auto *const ionization_state = stage_data.ionization_state();
  static const auto &ncomps_saha = ionization_state->ncomps();
  auto ybar = ionization_state->ybar();
  auto sigma1 = ionization_state->sigma1();
  auto sigma2 = ionization_state->sigma2();
  auto sigma3 = ionization_state->sigma3();
  auto e_ion_corr = ionization_state->e_ion_corr();

  static const bool has_neuts = species_indexer->contains("neut");
  static const int start_elem = (has_neuts) ? 1 : 0;

  static const int end_elem = (has_neuts) ? ncomps_saha : ncomps_saha - 1;
  static const IndexRange eb_saha(std::make_pair(start_elem, end_elem));

  static const IndexRange eb(std::make_pair(start_elem, num_species - 1));

  // NOTE: check index ranges inside here when saha ncomps =/= num_species
  // Should we be skipping neutrons?
  auto phi = stage_data.fluid_basis().phi();
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Ionization :: fill derived", DevExecSpace(), ib.s,
      ib.e, nb.s, nb.e, KOKKOS_LAMBDA(const int i, const int q) {
        const double rho =
            1.0 / basis::basis_eval(phi, ucf, i, vars::cons::SpecificVolume, q);
        // This kernel is horrible.
        // Reduce the ionization based quantities sigma1-3, e_ion_corr
        ybar(i, q) = electron_number_density(i, q) / number_density(i, q) / rho;
        const auto [s1, s2, s3, eion] =
            fill_derived_ionization(phi, mass_fractions, comps,
                                   ionization_state, eb, num_species, i, q);
        sigma1(i, q) = s1;
        sigma2(i, q) = s2;
        sigma3(i, q) = s3;
        e_ion_corr(i, q) = eion;
      });
}

} // namespace athelas::atom
