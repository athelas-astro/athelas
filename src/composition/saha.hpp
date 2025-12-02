#pragma once

#include "atom/atom.hpp"
#include "composition/compdata.hpp"
#include "composition/composition.hpp"
#include "constants.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "loop_layout.hpp"
#include "polynomial_basis.hpp"
#include "root_finders.hpp"
#include "solvers/root_finders.hpp"
#include "state/state.hpp"
#include "utils/error.hpp"

namespace athelas::atom {

// NOTE: a lot of logic in here is 1-indexed.
// I need to change it.

KOKKOS_INLINE_FUNCTION
auto saha_f(const double T, const IonLevel &ion_data) -> double {
  const double prefix = 2.0 * (ion_data.g_upper / ion_data.g_lower) *
                        constants::k_saha * std::pow(T, 1.5);
  const double suffix = std::exp(-ion_data.chi / (constants::k_B * T));

  return prefix * suffix;
}

/**
 * @brief Compute neutral ionization fraction. Eq 8 of Zaghloul et al 2000.
 */
KOKKOS_INLINE_FUNCTION
auto ion_frac0(const double Zbar, const AthelasArray1D<double> saha_factors,
               const double nk, const int min_state, const int max_state)
    -> double {
  const double inv_zbar_nk = 1.0 / (Zbar * nk);

  double denominator = 0.0;
  double prod = 1.0;
  for (int i = min_state; i < max_state; ++i) {
    prod *= inv_zbar_nk * saha_factors(i - 1);
    denominator += i * prod;
  }
  denominator += (min_state - 1.0);
  return Zbar / denominator;
}

KOKKOS_INLINE_FUNCTION
auto saha_target(const double Zbar, const AthelasArray1D<double> saha_factors,
                 const double nk, const int min_state, const int max_state)
    -> double {
  const double inv_zbar_nk = 1.0 / (Zbar * nk);
  double result = Zbar;

  double prod = 1.0;
  double sum0 = 0.0;
  double sum1 = 0.0;
  for (int i = min_state; i < max_state; ++i) {
    prod *= saha_factors(i - 1) * inv_zbar_nk;
    sum0 += prod;
    sum1 += i * prod;
  }
  const double denominator = (min_state - 1.0 + sum1);
  const double numerator = 1.0 + sum0;

  result *= (numerator / denominator);
  result = 1.0 - result;
  return result;
}

KOKKOS_INLINE_FUNCTION
auto saha_d_target(const double Zbar, const AthelasArray1D<double> saha_factors,
                   const double nk, const int min_state, const int max_state)
    -> double {

  double product = 1.0;
  double sigma0 = 0.0;
  double sigma1 = 0.0;
  double sigma2 = 0.0;
  double sigma3 = 0.0;

  const double inv_zbar_nk = 1.0 / (Zbar * nk);
  for (int i = min_state; i < max_state; ++i) {
    product *= saha_factors(i - 1) * inv_zbar_nk;
    sigma0 += product;
    sigma1 += i * product;
    sigma2 += (i - min_state + 1.0) * product;
    sigma3 += i * (i - min_state + 1.0) * product;
  }

  const double denom = 1.0 / (min_state - 1.0 + sigma1);
  return (sigma2 - (1.0 + sigma0) * (1.0 + sigma3 * denom)) * denom;
}

/**
 * @brief Saha solve on a given cell
 * @return zbar
 */
KOKKOS_INLINE_FUNCTION
void saha_solve(AthelasArray1D<double> ionization_states, const int Z,
                const AthelasArray1D<double> saha_factors, const double rho,
                const double nk, double &zbar_old) {

  using root_finders::RootFinder, root_finders::NewtonAlgorithm,
      root_finders::AANewtonAlgorithm, root_finders::RegulaFalsiAlgorithm,
      root_finders::RelativeError;
  // Set up static root finder for Saha ionization
  // We keep tight tolerances here.
  // TODO(astrobarker): make tolerances runtime
  static RootFinder<double, NewtonAlgorithm<double>> solver(
      {.abs_tol = 1.0e-12, .rel_tol = 1.0e-12, .max_iterations = 100});
  static constexpr double ZBARTOL = 1.0e-15;
  static constexpr double ZBARTOLINV = 1.0e15;

  const int num_states = Z + 1;
  int min_state = 1;
  int max_state = num_states;

  const double Zbar_nk_inv = 1.0 / (Z * nk);

  for (int i = 1; i < num_states; ++i) {
    const double f_saha = saha_factors(i - 1);
    // const double f_saha = saha_f(temperature, ion_datas(i - 1));

    if (f_saha * Zbar_nk_inv > ZBARTOLINV) {
      min_state = i + 1;
    }
    if (f_saha * Zbar_nk_inv < ZBARTOL) {
      max_state = i;
      break;
    }
  }

  if (max_state == 1) {
    ionization_states(0) = 1.0; // neutral
    zbar_old = 1.0e-16; // uncharged (but don't want division by 0)
  } else if (min_state == num_states) {
    ionization_states(Z) = 1.0; // full ionization
    zbar_old = Z;
  } else if (min_state == max_state) {
    zbar_old = min_state - 1.0;
    ionization_states(min_state - 1) = 1.0; // only one state possible
  } else { // iterative solve
    const double guess = zbar_old;

    // we use an Anderson acclerated Newton Raphson iteration
    zbar_old = solver.solve(saha_target, saha_d_target, guess, saha_factors, nk,
                            min_state, max_state);

    const double inv_zbar_nk = 1.0 / (zbar_old * nk);

    ionization_states(min_state - 1) =
        ion_frac0(zbar_old, saha_factors, nk, min_state, max_state);
    for (int i = min_state; i <= max_state - 1; ++i) {
      ionization_states(i) =
          ionization_states(i - 1) * saha_factors(i - 1) * inv_zbar_nk;
    }
  }
}

/**
 * @brief Functionality for saha ionization
 *
 * Word of warning: the code here is a gold medalist in index gymnastics.
 */
template <Domain MeshDomain>
void solve_saha_ionization(State &state, const GridStructure &grid,
                           const eos::EOS &eos,
                           const basis::ModalBasis &fluid_basis) {
  using basis::basis_eval;

  const auto uCF = state.u_cf();
  const auto uaf = state.u_af();
  const auto *const comps = state.comps();
  auto *const ionization_states = state.ionization_state();
  const auto *const atomic_data = ionization_states->atomic_data();
  const auto mass_fractions = state.mass_fractions();
  const auto species = comps->charge();
  const auto neutron_number = comps->neutron_number();
  auto n_e = comps->electron_number_density();
  auto *species_indexer = comps->species_indexer();
  auto ionization_fractions = ionization_states->ionization_fractions();
  auto zbars = ionization_states->zbar();
  auto saha_factors = ionization_states->saha_factor();

  const auto phi = fluid_basis.phi();

  // pull out atomic data containers
  const auto ion_data = atomic_data->ion_data();
  const auto species_offsets = atomic_data->offsets();

  const auto &nNodes = grid.n_nodes();
  assert(ionization_fractions.extent(2) <=
         static_cast<std::size_t>(std::numeric_limits<int>::max()));
  const auto &ncomps_saha = ionization_states->ncomps();
  const auto &ncomps_all = comps->n_species();

  static const bool has_neuts = species_indexer->contains("neut");
  static const int start_elem = (has_neuts) ? 1 : 0;
  static const int end_elem = (has_neuts) ? ncomps_saha : ncomps_saha - 1;
  static const IndexRange ib(grid.domain<MeshDomain>());
  static const IndexRange nb(nNodes + 2);
  static const IndexRange eb(std::make_pair(start_elem, end_elem));

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Saha :: Zero ioniation fractions and n_e",
      DevExecSpace(), ib.s, ib.e, nb.s, nb.e,
      KOKKOS_LAMBDA(const int i, const int q) {
        n_e(i, q) = 0.0;
        for (int e = eb.s; e <= eb.e; ++e) {
          for (int s = 0; s <= species(e); ++s) {
            ionization_fractions(i, q, e, s) = 0.0;
          }
        }
      });

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Saha :: Solve ionization all", DevExecSpace(),
      ib.s, ib.e, nb.s, nb.e, KOKKOS_LAMBDA(const int i, const int q) {
        const double rho =
            1.0 / basis_eval(phi, uCF, i, vars::cons::SpecificVolume, q);
        const double temperature = uaf(i, q, vars::aux::Tgas);

        // TODO(astrobarker): Profile; faster as hierarchical reduction?
        // This loop is over Saha species
        for (int e = eb.s; e <= eb.e; ++e) {
          const int z = species(e);
          const double x_e = basis_eval(phi, mass_fractions, i, e, q);

          const double A = z + neutron_number(e);
          const double nk = element_number_density(x_e, A, rho);

          // pull out element info
          const auto species_atomic_data =
              species_data(ion_data, species_offsets, z);
          auto ionization_fractions_e =
              Kokkos::subview(ionization_fractions, i, q, e, Kokkos::ALL);

          for (int s = 0; s <= z; ++s) {
            saha_factors(s) = saha_f(temperature, species_atomic_data(s));
          }

          double &zbar = zbars(i, q, e);

          saha_solve(ionization_fractions_e, z, saha_factors, rho, nk, zbar);
          std::println("Z Zbar T {} {:.5e} {:.5e}", z, zbar, temperature);
          n_e(i, q) += zbar * nk;
        }

        // loop over remaining species, assume complete ionization.
        for (std::size_t e = eb.e + 1; e < ncomps_all; ++e) {
          const int z = species(e);
          const double x_e = basis_eval(phi, mass_fractions, i, e, q);
          const double A = z + neutron_number(e);
          const double nk = element_number_density(x_e, A, rho);
          ionization_fractions(i, q, e, z) = 1.0;
          n_e(i, q) += z * nk;
        }
      });
}

template <Domain MeshDomain, eos::EOSInversion Inversion>
void solve_temperature_saha(const eos::EOS *eos, State *state,
                            const GridStructure &grid,
                            const basis::ModalBasis &basis) {
  static const auto &nnodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<MeshDomain>());
  static const IndexRange nb(nnodes + 2);

  auto ucf = state->u_cf();
  auto uaf = state->u_af();

  const auto *const comps = state->comps();
  auto *const ionization_states = state->ionization_state();
  auto ybar = ionization_states->ybar();
  const auto mass_fractions = state->mass_fractions();
  const auto species = comps->charge();
  const auto neutron_number = comps->neutron_number();
  auto n_e = comps->electron_number_density();
  auto number_density = comps->number_density();
  auto *species_indexer = comps->species_indexer();
  auto ionization_fractions = ionization_states->ionization_fractions();
  auto zbars = ionization_states->zbar();
  auto saha_factors = ionization_states->saha_factor();

  static const auto &ncomps_saha = ionization_states->ncomps();

  static const bool has_neuts = species_indexer->contains("neut");
  static const int start_elem = (has_neuts) ? 1 : 0;
  static const int end_elem = (has_neuts) ? ncomps_saha : ncomps_saha - 1;
  static const IndexRange eb_saha(std::make_pair(start_elem, end_elem));

  static const IndexRange eb(
      std::make_pair(start_elem, comps->n_species() - 1));

  const auto *const atomic_data = ionization_states->atomic_data();
  const auto ion_data = atomic_data->ion_data();
  const auto species_offsets = atomic_data->offsets();

  // set up root finder
  static root_finders::RootFinder<double,
                                  root_finders::RegulaFalsiAlgorithm<double>>
      solver({.abs_tol = 1.0e-12, .rel_tol = 1.0e-12, .max_iterations = 100});

  auto phi = basis.phi();
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "EOS :: T/Saha solve", DevExecSpace(), ib.s, ib.e,
      nb.s, nb.e, KOKKOS_LAMBDA(const int i, const int q) {
        const double rho =
            1.0 / basis::basis_eval(phi, ucf, i, vars::cons::SpecificVolume, q);

        double lambda[8];
        atom::paczynski_terms(state, i, q, lambda);

        const double temperature_guess = uaf(i, q, vars::aux::Tgas);
        auto target = [&](const double temperature, const double rho,
                          double *const lambda) {
          n_e(i, q) = 0.0;
          for (int e = eb_saha.s; e <= eb_saha.e; ++e) {
            const int z = species(e);
            const double x_e = basis::basis_eval(phi, mass_fractions, i, e, q);

            const double A = z + neutron_number(e);
            const double nk = atom::element_number_density(x_e, A, rho);

            // pull out element info
            const auto species_atomic_data =
                species_data(ion_data, species_offsets, z);
            auto ionization_fractions_e =
                Kokkos::subview(ionization_fractions, i, q, e, Kokkos::ALL);

            // As the iterative solve modifies the ionization fractions
            // directly, we have to reset it with each iteration. Also
            // precompute saha factors
            for (int s = 0; s <= z; ++s) {
              ionization_fractions(i, q, e, s) = 0.0;
              saha_factors(s) = saha_f(temperature, species_atomic_data(s));
            }

            double &zbar = zbars(i, q, e);

            saha_solve(ionization_fractions_e, z, saha_factors, rho, nk, zbar);
            n_e(i, q) += zbar * nk;
          }

          // TODO(astrobarker): [Saha] There must be a cleaner way to do this.
          for (int e = eb_saha.e + 1; e <= eb.e; ++e) {
            const int &z = species(e);
            const double x_e = basis::basis_eval(phi, mass_fractions, i, e, q);
            const double A = z + neutron_number(e);
            const double nk = element_number_density(x_e, A, rho);
            ionization_fractions(i, q, e, z) = 1.0;
            n_e(i, q) += z * nk;
          }

          ybar(i, q) = n_e(i, q) / number_density(i, q) / rho;
          fill_derived_ionization(&basis, mass_fractions, comps,
                                  ionization_states, eb, i, q);
          atom::paczynski_terms(state, i, q, lambda);
          if constexpr (Inversion == eos::EOSInversion::Pressure) {
            const double pressure = uaf(i, q, vars::aux::Pressure);
            return pressure_from_density_temperature(eos, rho, temperature,
                                                     lambda) -
                   pressure;
          } else {
            const double vel =
                basis::basis_eval(phi, ucf, i, vars::cons::Velocity, q);
            const double emt =
                basis::basis_eval(phi, ucf, i, vars::cons::Energy, q);
            const double sie = emt - 0.5 * vel * vel;
            return sie_from_density_temperature(eos, rho, temperature, lambda) -
                   sie;
          }
        };
        const double res =
            solver.solve(target, 500.0, 1.5e10, temperature_guess, rho, lambda);
        uaf(i, q, vars::aux::Tgas) = res;
      });
}

} // namespace athelas::atom
