#pragma once

#include "Kokkos_Macros.hpp"
#include "atom/atom.hpp"
#include "basis/polynomial_basis.hpp"
#include "composition/compdata.hpp"
#include "composition/composition.hpp"
#include "constants.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "loop_layout.hpp"
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
auto ion_frac0(const double Zbar, const ScratchPad1D<double> saha_factors,
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
auto saha_target(const double Zbar, const ScratchPad1D<double> saha_factors,
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
auto saha_d_target(const double Zbar, const ScratchPad1D<double> saha_factors,
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
 * @param relaxed_tol If true, use relaxed tolerances for faster convergence
 *                     during outer temperature iteration
 */
KOKKOS_INLINE_FUNCTION
void saha_solve(AthelasArray1D<double> ionization_states, const int Z,
                const ScratchPad1D<double> saha_factors, const double rho,
                const double nk, double &zbar_old) {

  using root_finders::RootFinder, root_finders::NewtonAlgorithm,
      root_finders::AANewtonAlgorithm, root_finders::RegulaFalsiAlgorithm,
      root_finders::RelativeError;
  // Set up static root finder for Saha ionization
  // TODO(astrobarker): make tolerances runtime
  static RootFinder<double, NewtonAlgorithm<double>> solver(
      {.abs_tol = 1.0e-10, .rel_tol = 1.0e-10, .max_iterations = 16});
  static constexpr double ZBARTOL = 1.0e-15;
  static constexpr double ZBARTOLINV = 1.0e15;

  const int num_states = Z + 1;
  int min_state = 1;
  int max_state = num_states;

  const double Zbar_nk_inv = 1.0 / (Z * nk);

  for (int i = 1; i < num_states; ++i) {
    const double f_saha = saha_factors(i - 1);

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
void solve_saha_ionization(State &state, AthelasArray3D<double> ucf,
                           const GridStructure &grid, const eos::EOS &eos,
                           const basis::ModalBasis &fluid_basis) {
  using basis::basis_eval;

  const auto uaf = state.u_af();
  const auto *const comps = state.comps();
  auto *const ionization_states = state.ionization_state();
  const auto *const atomic_data = ionization_states->atomic_data();
  const auto mass_fractions = state.mass_fractions();
  const auto species = comps->charge();
  const auto neutron_number = comps->neutron_number();
  auto inv_atomic_mass = comps->inverse_atomic_mass();
  auto n_e = comps->electron_number_density();
  auto *species_indexer = comps->species_indexer();
  auto ionization_fractions = ionization_states->ionization_fractions();
  auto zbars = ionization_states->zbar();

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

  // Allocate scratch space
  static const size_t nstates_total = ionization_fractions.extent(3);
  const int scratch_level = 0; // 0 is actual scratch (tiny); 1 is HBM
  const size_t scratch_size = ScratchPad1D<double>::shmem_size(nstates_total);
  athelas::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Saha :: Solve ionization all",
      DevExecSpace(), scratch_size, scratch_level, ib.s, ib.e, nb.s, nb.e,
      KOKKOS_LAMBDA(athelas::team_mbr_t member, const int i, const int q) {
        ScratchPad1D<double> saha_factors(member.team_scratch(scratch_level),
                                          nstates_total);
        const double rho =
            1.0 / basis_eval(phi, ucf, i, vars::cons::SpecificVolume, q);
        const double temperature = uaf(i, q, vars::aux::Tgas);

        // TODO(astrobarker): Profile; faster as hierarchical reduction?
        // This loop is over Saha species
        athelas::par_for_inner(
            DEFAULT_INNER_LOOP_PATTERN, member, eb.s, eb.e, [&](const int e) {
              const int z = species(e);
              const double x_e = basis_eval(phi, mass_fractions, i, e, q);

              const double inv_A = inv_atomic_mass(e);
              const double nk = element_number_density(x_e, inv_A, rho);

              // pull out element info
              const auto species_atomic_data =
                  species_data(ion_data, species_offsets, z);
              auto ionization_fractions_e =
                  Kokkos::subview(ionization_fractions, i, q, e, Kokkos::ALL);

              for (int s = 0; s <= z; ++s) {
                saha_factors(s) = saha_f(temperature, species_atomic_data(s));
              }

              double &zbar = zbars(i, q, e);

              saha_solve(ionization_fractions_e, z, saha_factors, rho, nk,
                         zbar);
              n_e(i, q) += zbar * nk;
            });

        // loop over remaining species, assume complete ionization.
        athelas::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, eb.e + 1,
                               ncomps_all - 1, [&](const int e) {
                                 const int z = species(e);
                                 const double x_e =
                                     basis_eval(phi, mass_fractions, i, e, q);
                                 const double inv_A = inv_atomic_mass(e);
                                 const double nk =
                                     element_number_density(x_e, inv_A, rho);
                                 ionization_fractions(i, q, e, z) = 1.0;
                                 n_e(i, q) += z * nk;
                               });
      });
}

struct CoupledSolverContent {
  using AA1D = AthelasArray1D<double>;
  using AA2D = AthelasArray2D<double>;
  using AA3D = AthelasArray3D<double>;
  using AA4D = AthelasArray4D<double>;

  AA3D ucf;
  AA3D uaf;
  AA3D mass_fractions;
  AA3D phi;
  AA3D zbar;
  AA2D ye;
  AA2D n_e;
  AA2D number_density;
  AA2D ybar;
  AA2D abar;
  AA2D e_ion_corr;
  AA2D sigma1;
  AA2D sigma2;
  AA2D sigma3;

  AA4D ionization_fractions;
  ScratchPad1D<double> saha_factors;

  AthelasArray1D<int> species;
  AthelasArray1D<int> neutron_number;
  AthelasArray1D<double> inv_atomic_mass;
  AthelasArray1D<int> species_offsets;

  AthelasArray1D<IonLevel> ion_data;
  AthelasArray1D<double> sum_pots;

  IndexRange eb_saha;
  IndexRange eb;

  int i;
  int q;

  double target_var; // sie or pressure
};

KOKKOS_FUNCTION
template <eos::EOSInversion Inversion>
auto temperature_residual(const double temperature, const double rho,
                          const eos::EOS *eos,
                          const CoupledSolverContent &content) -> double {

  auto ucf = content.ucf;
  auto uaf = content.uaf;

  auto phi = content.phi;

  auto species = content.species;
  auto neutron_number = content.neutron_number;
  auto inv_atomic_mass = content.inv_atomic_mass;
  auto ye = content.ye;
  auto n_e = content.n_e;
  auto number_density = content.number_density;
  auto mass_fractions = content.mass_fractions;

  auto ion_data = content.ion_data;
  auto sum_pots = content.sum_pots;
  auto species_offsets = content.species_offsets;

  auto ybar = content.ybar;
  auto ionization_fractions = content.ionization_fractions;
  auto saha_factors = content.saha_factors;
  auto zbars = content.zbar;
  auto e_ion_corr = content.e_ion_corr;
  auto sigma1 = content.sigma1;
  auto sigma2 = content.sigma2;
  auto sigma3 = content.sigma3;

  const auto &eb = content.eb;
  const auto &eb_saha = content.eb_saha;

  const int &i = content.i;
  const int &q = content.q;

  double n_e_solve = 0.0;
  double sum_e_ion_corr = 0.0;
  double sum1 = 0.0;
  double sum2 = 0.0;
  double sum3 = 0.0;
  const double N = number_density(i, q);
  const auto abar = content.abar(i, q);

  // Loop over Saha species – solve ionization for the Saha subset
  for (int e = eb_saha.s; e <= eb_saha.e; ++e) {
    const int z = species(e);
    const double x_e = basis::basis_eval(phi, mass_fractions, i, e, q);

    const double inv_A = inv_atomic_mass(e);
    const double nk = atom::element_number_density(x_e, inv_A, rho);

    // species atomic data
    const auto species_atomic_data = species_data(ion_data, species_offsets, z);
    const auto species_sum_pot = species_data(sum_pots, species_offsets, z);

    auto ionization_fractions_e =
        Kokkos::subview(ionization_fractions, i, q, e, Kokkos::ALL);

    // reset ionization fractions, setup saha factors
    for (int s = 0; s <= z; ++s) {
      ionization_fractions(i, q, e, s) = 0.0;
      saha_factors(s) = saha_f(temperature, species_atomic_data(s));
    }

    double &zbar = zbars(i, q, e);

    saha_solve(ionization_fractions_e, z, saha_factors, rho, nk, zbar);

    n_e_solve += zbar * nk;

    const int nstates = z + 1;

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

    const double nu_k = abar * x_e * inv_A;
    const double term = nu_k * y_r * (1.0 - y_r);
    sum1 += term;
    sum2 += chi_r * term;
    sum3 += chi_r * chi_r * term;
    sum_e_ion_corr += nu_k * sum_ion_pot;
  }

  // Sum over non-Saha species
  for (int e = eb_saha.e + 1; e <= eb.e; ++e) {
    const int z = species(e);
    const double x_e = basis::basis_eval(phi, mass_fractions, i, e, q);
    const double inv_A = inv_atomic_mass(e);

    // species atomic data
    const auto species_atomic_data = species_data(ion_data, species_offsets, z);
    const auto species_sum_pot = species_data(sum_pots, species_offsets, z);

    auto ionization_fractions_e =
        Kokkos::subview(ionization_fractions, i, q, e, Kokkos::ALL);

    const double sum_ion_pot = species_sum_pot(z - 1);
    const double nu_k = abar * x_e * inv_A;

    sum_e_ion_corr += nu_k * sum_ion_pot;
  }

  ybar(i, q) = (n_e(i, q) + n_e_solve) / N / rho;
  sigma1(i, q) = sum1;
  sigma2(i, q) = sum2;
  sigma3(i, q) = sum3;
  e_ion_corr(i, q) = N * sum_e_ion_corr;

  // Fill lambda
  double lambda[8];
  lambda[0] = N;
  lambda[1] = ye(i, q);
  lambda[2] = ybar(i, q);
  lambda[3] = sigma1(i, q);
  lambda[4] = sigma2(i, q);
  lambda[5] = sigma3(i, q);
  lambda[6] = e_ion_corr(i, q);

  if constexpr (Inversion == eos::EOSInversion::Pressure) {
    const double inv_dfdt =
        1.0 / eos::Paczynski::dp_dt(temperature, rho, lambda);
    const double f =
        pressure_from_density_temperature(eos, rho, temperature, lambda) -
        content.target_var;
    return temperature - inv_dfdt * f;
  } else { // sie inversion
    const double inv_dfdt =
        1.0 / eos::Paczynski::dsie_dt(temperature, rho, lambda);
    const double f =
        sie_from_density_temperature(eos, rho, temperature, lambda) -
        content.target_var;
    return temperature - inv_dfdt * f;
  }
}

template <Domain MeshDomain, eos::EOSInversion Inversion>
void compute_temperature_with_saha(const eos::EOS *eos, State *state,
                                   AthelasArray3D<double> ucf,
                                   const GridStructure &grid,
                                   const basis::ModalBasis &basis) {
  using root_finders::RegulaFalsiAlgorithm, root_finders::FixedPointAlgorithm,
      root_finders::AAFixedPointAlgorithm;
  static const auto &nnodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<MeshDomain>());
  static const IndexRange nb(nnodes + 2);

  auto uaf = state->u_af();
  auto uaf_guess = state->u_af(-1); // get previous stage data when possible.

  const auto *const comps = state->comps();
  auto mass_fractions = state->mass_fractions();
  auto species = comps->charge();
  auto neutron_number = comps->neutron_number();
  auto inv_atomic_mass = comps->inverse_atomic_mass();
  auto ye = comps->ye();
  auto abar = comps->abar();
  auto n_e = comps->electron_number_density();
  auto number_density = comps->number_density();
  auto *species_indexer = comps->species_indexer();

  auto *const ionization_state = state->ionization_state();
  auto ybar = ionization_state->ybar();
  auto ionization_fractions = ionization_state->ionization_fractions();
  auto zbars = ionization_state->zbar();
  auto e_ion_corr = ionization_state->e_ion_corr();
  auto sigma1 = ionization_state->sigma1();
  auto sigma2 = ionization_state->sigma2();
  auto sigma3 = ionization_state->sigma3();

  static const auto &ncomps_saha = ionization_state->ncomps();

  static const bool has_neuts = species_indexer->contains("neut");
  static const int start_elem = (has_neuts) ? 1 : 0;
  static const int end_elem = (has_neuts) ? ncomps_saha : ncomps_saha - 1;
  static const IndexRange eb_saha(std::make_pair(start_elem, end_elem));

  static const IndexRange eb(
      std::make_pair(start_elem, comps->n_species() - 1));

  // atomic data
  const auto *const atomic_data = ionization_state->atomic_data();
  auto ion_data = atomic_data->ion_data();
  auto species_offsets = atomic_data->offsets();
  auto prefix_sum_pots = atomic_data->sum_pots();

  static root_finders::RootFinder<double, AAFixedPointAlgorithm<double>> solver(
      {.abs_tol = 1.0e-8, .rel_tol = 1.0e-8, .max_iterations = 32});

  auto phi = basis.phi();
  // Allocate scratch space
  static const std::size_t nstates_total =
      ionization_fractions.extent(3); // overkill
  const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
  const size_t scratch_size = ScratchPad1D<double>::shmem_size(nstates_total);

  athelas::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "EOS :: T/Saha solve", DevExecSpace(),
      scratch_size, scratch_level, ib.s, ib.e, nb.s, nb.e,
      KOKKOS_LAMBDA(athelas::team_mbr_t member, const int i, const int q) {
        n_e(i, q) = 0.0;

        ScratchPad1D<double> saha_factors(member.team_scratch(scratch_level),
                                          nstates_total);

        const double rho =
            1.0 / basis::basis_eval(phi, ucf, i, vars::cons::SpecificVolume, q);
        const double temperature_guess = uaf_guess(i, q, vars::aux::Tgas);

        double target_var = 0.0;
        if constexpr (Inversion == eos::EOSInversion::Pressure) {
          target_var = uaf(i, q, vars::aux::Pressure);

        } else {
          const double vel =
              basis::basis_eval(phi, ucf, i, vars::cons::Velocity, q);
          const double emt =
              basis::basis_eval(phi, ucf, i, vars::cons::Energy, q);
          target_var = emt - 0.5 * vel * vel;
        }

        // solver content
        const CoupledSolverContent content{ucf,
                                           uaf,
                                           mass_fractions,
                                           phi,
                                           zbars,
                                           ye,
                                           n_e,
                                           number_density,
                                           ybar,
                                           abar,
                                           e_ion_corr,
                                           sigma1,
                                           sigma2,
                                           sigma3,
                                           ionization_fractions,
                                           saha_factors,
                                           species,
                                           neutron_number,
                                           inv_atomic_mass,
                                           species_offsets,
                                           ion_data,
                                           prefix_sum_pots,
                                           eb_saha,
                                           eb,
                                           i,
                                           q,
                                           target_var};

        // TODO(astrobarker): [Saha] There must be a cleaner way to do this.
        athelas::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member,
                               eb_saha.e + 1, eb.e, [&](const int e) {
                                 const int &z = species(e);
                                 const double x_e = basis::basis_eval(
                                     phi, mass_fractions, i, e, q);
                                 const double inv_A = inv_atomic_mass(e);
                                 const double nk =
                                     element_number_density(x_e, inv_A, rho);
                                 n_e(i, q) += z * nk;
                               });

        const double res = solver.solve(temperature_residual<Inversion>,
                                        temperature_guess, rho, eos, content);
        uaf(i, q, vars::aux::Tgas) = res;
      });
}

} // namespace athelas::atom
