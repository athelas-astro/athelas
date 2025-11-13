#pragma once

#include "atom/atom.hpp"
#include "composition/composition.hpp"
#include "constants.hpp"
#include "eos/eos_variant.hpp"
#include "error.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "loop_layout.hpp"
#include "polynomial_basis.hpp"
#include "state/state.hpp"

namespace athelas::atom {

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
auto ion_frac0(const double Zbar, const double temperature,
               const AthelasArray1D<const IonLevel> ion_datas, const double nh,
               const int min_state, const int max_state) -> double {

  double denominator = 0.0;
  for (int i = min_state; i < max_state; ++i) {
    double prod = 1.0;
    for (int j = min_state; j <= i; ++j) {
      prod *= (saha_f(temperature, ion_datas(j - 1))) / (Zbar * nh);
    }
    denominator += i * prod;
  }
  denominator += (min_state - 1.0);
  return Zbar / denominator;
}

KOKKOS_INLINE_FUNCTION
auto saha_target(const double Zbar, const double T,
                 const AthelasArray1D<const IonLevel> ion_datas,
                 const double nh, const int min_state, const int max_state)
    -> double {
  double result = Zbar;
  double numerator = 1.0;
  double denominator = 0.0;
  for (int i = min_state; i < max_state; ++i) {
    double prod = 1.0;
    for (int j = min_state; j <= i; ++j) {
      const double f = saha_f(T, ion_datas(j - 1));
      prod *= f / (Zbar * nh);
    }
    numerator += prod;
    denominator += i * prod;
  }
  denominator += (min_state - 1.0);

  result *= (numerator / denominator);
  result = 1.0 - result;
  return result;
}

KOKKOS_INLINE_FUNCTION
auto saha_d_target(const double Zbar, const double T,
                   const AthelasArray1D<const IonLevel> ion_datas,
                   const double nh, const int min_state, const int max_state)
    -> double {

  double product = 1.0;
  double sigma0 = 0.0;
  double sigma1 = 0.0;
  double sigma2 = 0.0;
  double sigma3 = 0.0;

  for (int i = min_state; i < max_state; ++i) {
    product *= saha_f(T, ion_datas(i - 1)) / (Zbar * nh);
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
auto saha_solve(AthelasArray1D<double> ionization_states, const int Z,
                const double temperature,
                const AthelasArray1D<const IonLevel> ion_datas, const double nk)
    -> double {

  using root_finders::RootFinder, root_finders::NewtonAlgorithm,
      root_finders::RelativeError;
  // Set up static root finder for Saha ionization
  // We keep tight tolerances here.
  // TODO(astrobarker): make tolerances runtime
  static RootFinder<double, NewtonAlgorithm<double>> solver(
      {.abs_tol = 1.0e-18, .rel_tol = 1.0e-16, .max_iterations = 100});
  static constexpr double ZBARTOL = 1.0e-15;
  static constexpr double ZBARTOLINV = 1.0e15;

  const int num_states = Z + 1;
  int min_state = 1;
  int max_state = num_states;

  const double Zbar_nk_inv = 1.0 / (Z * nk);

  for (int i = 0; i <= num_states - 1; ++i) {
    const double f_saha = saha_f(temperature, ion_datas(i));

    if (f_saha * Zbar_nk_inv > ZBARTOLINV) {
      min_state = i + 1;
      ionization_states(i) = 0.0;
    }
    if (f_saha * Zbar_nk_inv < ZBARTOL) {
      max_state = i;
      for (int j = i + 1; j < num_states; ++j) {
        ionization_states(j) = 0.0;
      }
      break;
    }
  }

  // Need  better solution for this.
  static constexpr double min_nk = 1.0e-40;
  if (nk < min_nk) {
    min_state = num_states;
  }

  double Zbar = 0;
  if (max_state == 0) {
    ionization_states(0) = 1.0; // neutral
    Zbar = 1.0e-16; // uncharged (but don't want division by 0)
  } else if (min_state == num_states) {
    ionization_states(Z) = 1.0; // full ionization
    Zbar = Z;
  } else if (min_state == max_state) {
    Zbar = min_state - 1.0;
    ionization_states(min_state - 1) = 1.0; // only one state possible
  } else { // iterative solve
    // I wonder if there is a smarter way to produce a guess -- T dependent?
    // Simpler ionization model to guess Zbar(T)?
    const double guess = 0.01 * Z;

    // we use an Anderson acclerated Newton Raphson iteration
    Zbar = solver.solve(saha_target, saha_d_target, guess, temperature,
                        ion_datas, nk, min_state, max_state);

    ionization_states(0) =
        ion_frac0(Zbar, temperature, ion_datas, nk, min_state, max_state);
    for (int i = 1; i <= Z; ++i) {
      ionization_states(i) =
          ionization_states(i - 1) *
          (saha_f(temperature, ion_datas(i - 1)) / (Zbar * nk));
    }
  }
  return Zbar;
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
  static const IndexRange ib(grid.domain<MeshDomain>());
  static const IndexRange nb(nNodes + 2);
  static const IndexRange eb(std::make_pair(start_elem, ncomps_saha - 1));

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Saha :: Zero ioniation fractions", DevExecSpace(),
      ib.s, ib.e, nb.s, nb.e, KOKKOS_LAMBDA(const int i, const int q) {
        for (std::size_t e = eb.s; e < ncomps_all; ++e) {
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

        n_e(i, q) = 0.0;
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

          const double zbar = saha_solve(ionization_fractions_e, z, temperature,
                                         species_atomic_data, nk);
          n_e(i, q) += zbar * nk;
        }

        // loop over remaining species, assume complete ionization.
        for (std::size_t e = eb.e + 1; e < ncomps_all; ++e) {
          const int z = species(e);
          const double x_e = basis_eval(phi, mass_fractions, i, e, q);
          const double A = z + neutron_number(e);
          const double nk = element_number_density(x_e, A, rho);
          // pull out element info
          auto ionization_fractions_e =
              Kokkos::subview(ionization_fractions, i, q, e, Kokkos::ALL);
          ionization_fractions_e(z) = 1.0;
          n_e(i, q) += z * nk;
        }
      });
}

} // namespace athelas::atom
