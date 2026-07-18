#pragma once

#include <cmath>

#include "Kokkos_Macros.hpp"
#include "atom/atom.hpp"
#include "basis/polynomial_basis.hpp"
#include "composition/compdata.hpp"
#include "composition/composition.hpp"
#include "constants.hpp"
#include "eos.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "loop_layout.hpp"
#include "solvers/root_finders.hpp"

namespace athelas::atom {

static constexpr double saha_min_temperature = 500.0; // K
// Upper end of the coupled temperature solve's bracket; matches the
// Paczynski temperature-inversion ceiling.
static constexpr double saha_max_temperature = 1.0e12; // K
static constexpr double NEUTRAL_ZBAR = 1.0e-16;

// NOTE: a lot of logic in here is 1-indexed.
// I need to change it.

KOKKOS_FUNCTION
template <SahaSolver SolverType>
auto saha_f(const double T, const IonLevel &ion_data) -> double {
  if constexpr (SolverType == SahaSolver::Linear) {
    const double prefix = 2.0 * (ion_data.g_upper / ion_data.g_lower) *
                          constants::k_saha * std::pow(T, 1.5);
    const double suffix = std::exp(-ion_data.chi / (constants::k_B * T));

    return prefix * suffix;
  } else {
    return (-ion_data.chi / (constants::k_B * T)) +
           std::log(2.0 * ion_data.g_upper / ion_data.g_lower) +
           constants::ln_k_saha + 1.5 * std::log(T);
  }
}

/**
 * @brief Compute neutral ionization fraction. Eq 8 of Zaghloul et al 2000.
 * @note Original linear form.
 */
KOKKOS_INLINE_FUNCTION
auto ion_frac0(const double zbar, const ScratchPad1D<double> saha_factors,
               const double nk, const int min_state, const int max_state)
    -> double {
  const double inv_zbar_nk = 1.0 / (zbar * nk);

  double denominator = 0.0;
  double prod = 1.0;
  for (int i = min_state; i < max_state; ++i) {
    prod *= inv_zbar_nk * saha_factors(i - 1);
    denominator += i * prod;
  }
  denominator += (min_state - 1.0);
  return zbar / denominator;
}

/**
 * @brief Compute neutral ionization fraction. Eq 8 of Zaghloul et al 2000.
 * @note modified log form
 */
KOKKOS_INLINE_FUNCTION
auto ion_frac0(const double lnz, const ScratchPad1D<double> saha_factors,
               AthelasArray1D<double> ln_i, const double ln_nk,
               const int min_state, const int max_state) -> double {
  const double xplnk = lnz + ln_nk;

  double lnD = -std::numeric_limits<double>::infinity();

  double Li = 0.0;
  for (int i = min_state; i < max_state; ++i) {
    const double &lni = ln_i(i);
    Li += saha_factors(i - 1);
    const double a_i = Li - i * xplnk;
    const double b_i = a_i + lni;

    lnD = std::max(lnD, b_i) + std::log1p(std::exp(-std::abs(lnD - b_i)));
  }
  return lnz - lnD;
}

/**
 * @brief Root finder target function for Saha solve
 * @note Log solver
 */
KOKKOS_INLINE_FUNCTION
auto saha_target_log(const double lnz, const ScratchPad1D<double> saha_factors,
                     AthelasArray1D<double> ln_i, const double ln_nk,
                     const int min_state, const int max_state)
    -> std::tuple<double, double> {
  const double xplnk = lnz + ln_nk;

  double lnN = 0.0;
  double lnD = -std::numeric_limits<double>::infinity();
  double lnE = -std::numeric_limits<double>::infinity();

  double Li = 0.0;
  for (int i = min_state; i < max_state; ++i) {
    const double &lni = ln_i(i);
    Li += saha_factors(i - 1);
    const double a_i = Li - i * xplnk;
    const double b_i = a_i + lni;
    const double c_i = a_i + 2.0 * lni;

    // Numerically stable update of e.g., lnN = ln(sum_i exp(a_i)) using
    // log-sum-exp
    lnN = std::max(lnN, a_i) + std::log1p(std::exp(-std::abs(lnN - a_i)));
    lnD = std::max(lnD, b_i) + std::log1p(std::exp(-std::abs(lnD - b_i)));
    lnE = std::max(lnE, c_i) + std::log1p(std::exp(-std::abs(lnE - c_i)));

    if (a_i < lnN - 20) {
      break;
    }
  }

  const double f = lnz + lnN - lnD;
  const double fprime = 1.0 - std::exp(lnD - lnN) + std::exp(lnE - lnD);

  return {f, fprime};
}

/**
 * @brief Root finder target function for Saha solve
 * @note Original linear solver
 */
KOKKOS_INLINE_FUNCTION
auto saha_target_linear(const double Zbar,
                        const ScratchPad1D<double> saha_factors,
                        const double nk, const int min_state,
                        const int max_state) -> std::tuple<double, double> {
  const double inv_zbar_nk = 1.0 / (Zbar * nk);
  double f = Zbar;
  double fp = 0.0;

  double prod = 1.0;
  double sum0 = 0.0;
  double sum1 = 0.0;
  double sum2 = 0.0;
  double sum3 = 0.0;
  for (int i = min_state; i < max_state; ++i) {
    prod *= saha_factors(i - 1) * inv_zbar_nk;
    sum0 += prod;
    sum1 += i * prod;
    sum2 += (i - min_state + 1.0) * prod;
    sum3 += i * (i - min_state + 1.0) * prod;
  }
  const double denominator = (min_state - 1.0 + sum1);
  const double numerator = 1.0 + sum0;

  const double denom = 1.0 / (min_state - 1.0 + sum1);
  fp = (sum2 - (1.0 + sum0) * (1.0 + sum3 * denom)) * denom;

  f *= (numerator / denominator);
  return {1.0 - f, fp};
}

/**
 * @brief Saha solve on a given cell using original Linear method
 * @return zbar
 */
KOKKOS_INLINE_FUNCTION
void saha_solve_linear(AthelasArray1D<double> ionization_states, const int Z,
                       const ScratchPad1D<double> saha_factors,
                       const double rho, const double nk, double &zbar_old) {

  using root_finders::RootFinder, root_finders::NewtonAlgorithm,
      root_finders::NewtonAlgorithmBundled, root_finders::AANewtonAlgorithm,
      root_finders::AANewtonAlgorithmBundled,
      root_finders::RegulaFalsiAlgorithm, root_finders::FixedPointAlgorithm,
      root_finders::RelativeError, root_finders::AbsoluteError;
  // Set up static root finder for Saha ionization
  // TODO(astrobarker): make tolerances runtime
  static RootFinder<double, AANewtonAlgorithmBundled<double>, AbsoluteError>
      solver({.abs_tol = 1.0e-10, .rel_tol = 1.0e-10, .max_iterations = 32});
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
    zbar_old = NEUTRAL_ZBAR; // uncharged (but don't want division by 0)
  } else if (min_state == num_states) {
    ionization_states(Z) = 1.0; // full ionization
    zbar_old = Z;
  } else if (min_state == max_state) {
    zbar_old = min_state - 1.0;
    ionization_states(min_state - 1) = 1.0; // only one state possible
  } else {
    const double guess = zbar_old;

    // we use a Newton Raphson iteration
    zbar_old = solver.solve(saha_target_linear, guess, saha_factors, nk,
                            min_state, max_state);
  }
}

/**
 * @brief Saha solve on a given cell using original Log method
 * @note This is currently the preferred method for large Z
 * @return zbar
 */
KOKKOS_INLINE_FUNCTION
void saha_solve_log(AthelasArray1D<double> ionization_states, const int Z,
                    const ScratchPad1D<double> saha_factors,
                    AthelasArray1D<double> ln_i, const double rho,
                    const double ln_nk, double &zbar_old) {

  using root_finders::RootFinder, root_finders::NewtonAlgorithm,
      root_finders::NewtonAlgorithmBundled, root_finders::AANewtonAlgorithm,
      root_finders::RegulaFalsiAlgorithm, root_finders::FixedPointAlgorithm,
      root_finders::RelativeError, root_finders::AbsoluteError;
  // Set up static root finder for Saha ionization
  // TODO(astrobarker): make tolerances runtime
  static RootFinder<double, NewtonAlgorithmBundled<double>, AbsoluteError>
      solver({.abs_tol = 1.0e-10, .rel_tol = 1.0e-10, .max_iterations = 32});

  const int num_states = Z + 1;
  static constexpr int min_state = 1;
  const int max_state = num_states;

  // iterative solve
  const double guess = std::log(zbar_old);

  // we use a Newton Raphson iteration
  zbar_old = solver.solve(saha_target_log, guess, saha_factors, ln_i, ln_nk,
                          min_state, max_state);
}

/**
 * @brief Functionality for saha ionization
 *
 * Word of warning: the code here is a gold medalist in index gymnastics.
 */
template <SahaSolver SolverType>
void solve_saha_ionization(StageData &stage_data, const Mesh &mesh) {
  using basis::basis_eval;

  auto evolved = stage_data.get_field("evolved");
  auto derived = stage_data.get_field("derived");
  const int idx_tau = stage_data.var_index("evolved", "specific_volume");
  const int idx_tgas = stage_data.var_index("derived", "gas_temperature");
  const auto *const comps = stage_data.comps();
  auto *const ionization_states = stage_data.ionization_state();
  const auto *const atomic_data = ionization_states->atomic_data();
  auto ln_i = ionization_states->ln_i();
  const auto mass_fractions = stage_data.mass_fractions("evolved");
  const auto species = comps->charge();
  const auto neutron_number = comps->neutron_number();
  auto inv_atomic_mass = comps->inverse_atomic_mass();
  auto n_e = comps->electron_number_density();
  auto *species_indexer = comps->species_indexer();
  auto ionization_fractions = ionization_states->ionization_fractions();
  auto zbars = ionization_states->zbar();

  const auto &basis = stage_data.basis();
  const auto phi = basis.phi();

  // pull out atomic data containers
  const auto ion_data = atomic_data->ion_data();
  const auto species_offsets = atomic_data->offsets();

  const auto &nNodes = mesh.n_nodes();
  assert(ionization_fractions.extent(2) <=
         static_cast<std::size_t>(std::numeric_limits<int>::max()));
  const auto &ncomps_saha = ionization_states->ncomps();
  const auto &ncomps_all = comps->n_species();

  static const bool has_neuts = species_indexer->contains("neut");
  static const int start_elem = (has_neuts) ? 1 : 0;
  static const int end_elem = (has_neuts) ? ncomps_saha : ncomps_saha - 1;
  const IndexRange ib(mesh.domain<Domain::Interior>());
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
        const double rho = 1.0 / basis_eval(phi, evolved, i, idx_tau, q);
        const double temperature = derived(i, q, idx_tgas);

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
                saha_factors(s) =
                    saha_f<SolverType>(temperature, species_atomic_data(s));
              }

              double &zbar = zbars(i, q, e);
              const double inv_zbar_nk = 1.0 / (zbar * nk);
              if constexpr (SolverType == SahaSolver::Log) {
                const double ln_nk = std::log(nk);

                saha_solve_log(ionization_fractions_e, z, saha_factors, ln_i,
                               rho, ln_nk, zbar);
                ionization_fractions_e(0) = std::exp(
                    ion_frac0(zbar, saha_factors, ln_i, ln_nk, 1, z + 1));
                zbar = std::exp(zbar);
                for (int i = 1; i <= z; ++i) {
                  ionization_fractions_e(i) = ionization_fractions_e(i - 1) *
                                              std::exp(saha_factors(i - 1)) *
                                              inv_zbar_nk;
                }
              } else {
                saha_solve_linear(ionization_fractions_e, z, saha_factors, rho,
                                  nk, zbar);
                ionization_fractions_e(0) =
                    ion_frac0(zbar, saha_factors, nk, 1, z + 1);
                for (int i = 1; i <= z; ++i) {
                  ionization_fractions_e(i) = ionization_fractions_e(i - 1) *
                                              saha_factors(i - 1) * inv_zbar_nk;
                }
              }
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

  AA3D evolved;
  AA3D derived;
  AA3D mass_fractions;
  AA3D mass_fractions_nodal;
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
  AthelasArray1D<double> ln_i;

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

KOKKOS_INLINE_FUNCTION
void set_low_temperature_ionization_state(const double temperature,
                                          const double rho,
                                          const CoupledSolverContent &content,
                                          double *lambda) {
  auto species = content.species;
  auto inv_atomic_mass = content.inv_atomic_mass;
  auto number_density = content.number_density;
  auto mass_fractions_nodal = content.mass_fractions_nodal;
  auto sum_pots = content.sum_pots;
  auto species_offsets = content.species_offsets;
  auto ye = content.ye;
  auto n_e = content.n_e;
  auto ybar = content.ybar;
  auto zbars = content.zbar;
  auto e_ion_corr = content.e_ion_corr;
  auto sigma1 = content.sigma1;
  auto sigma2 = content.sigma2;
  auto sigma3 = content.sigma3;
  auto ionization_fractions = content.ionization_fractions;

  const auto &eb = content.eb;
  const auto &eb_saha = content.eb_saha;
  const int &i = content.i;
  const int &q = content.q;

  double n_e_solve = 0.0;
  double sum_e_ion_corr = 0.0;
  const double N = number_density(i, q);
  const double abar = content.abar(i, q);

  for (int e = eb_saha.s; e <= eb_saha.e; ++e) {
    const int z = species(e);
    const double x_e = mass_fractions_nodal(i, q, e);
    const double inv_A = inv_atomic_mass(e);
    const double nk = atom::element_number_density(x_e, inv_A, rho);
    auto ionization_fractions_e =
        Kokkos::subview(ionization_fractions, i, q, e, Kokkos::ALL);
    for (int s = 0; s <= z; ++s) {
      ionization_fractions_e(s) = 0.0;
    }
    ionization_fractions_e(0) = 1.0;
    zbars(i, q, e) = NEUTRAL_ZBAR;
    n_e_solve += NEUTRAL_ZBAR * nk;
  }

  for (int e = eb_saha.e + 1; e <= eb.e; ++e) {
    const int z = species(e);
    const double x_e = mass_fractions_nodal(i, q, e);
    const double inv_A = inv_atomic_mass(e);
    const double nk = atom::element_number_density(x_e, inv_A, rho);
    const auto species_sum_pot = species_data(sum_pots, species_offsets, z);
    auto ionization_fractions_e =
        Kokkos::subview(ionization_fractions, i, q, e, Kokkos::ALL);

    for (int s = 0; s <= z; ++s) {
      ionization_fractions_e(s) = 0.0;
    }
    ionization_fractions_e(z) = 1.0;

    const double nu_k = abar * x_e * inv_A;
    n_e_solve += z * nk;
    sum_e_ion_corr += nu_k * species_sum_pot(z - 1);
  }

  n_e(i, q) = n_e_solve;
  ybar(i, q) = n_e_solve / N / rho;
  sigma1(i, q) = 0.0;
  sigma2(i, q) = 0.0;
  sigma3(i, q) = 0.0;
  e_ion_corr(i, q) = N * sum_e_ion_corr;

  lambda[eos::paczynski_lambda::number_density] = N;
  lambda[eos::paczynski_lambda::ye] = ye(i, q);
  lambda[eos::paczynski_lambda::ybar] = ybar(i, q);
  lambda[eos::paczynski_lambda::sigma1] = sigma1(i, q);
  lambda[eos::paczynski_lambda::sigma2] = sigma2(i, q);
  lambda[eos::paczynski_lambda::sigma3] = sigma3(i, q);
  lambda[eos::paczynski_lambda::e_ion_corr] = e_ion_corr(i, q);
  lambda[eos::EOS_LAMBDA_TEMPERATURE] =
      temperature > saha_min_temperature ? temperature : saha_min_temperature;
}

// Compute the coupled Saha ionization state at a trial temperature, writing
// the per-cell ionization views and the EOS lambda as side effects. At or
// below saha_min_temperature this is the floor state: Saha species neutral,
// non-Saha species fully ionized.
KOKKOS_FUNCTION
template <SahaSolver SolverType>
void compute_coupled_saha_state(const double temperature, const double rho,
                                const CoupledSolverContent &content,
                                double *lambda) {
  auto evolved = content.evolved;
  auto derived = content.derived;

  auto phi = content.phi;

  auto species = content.species;
  auto neutron_number = content.neutron_number;
  auto inv_atomic_mass = content.inv_atomic_mass;
  auto ye = content.ye;
  auto n_e = content.n_e;
  auto number_density = content.number_density;
  auto mass_fractions = content.mass_fractions;
  auto mass_fractions_nodal = content.mass_fractions_nodal;

  auto ion_data = content.ion_data;
  auto sum_pots = content.sum_pots;
  auto species_offsets = content.species_offsets;

  auto ybar = content.ybar;
  auto ionization_fractions = content.ionization_fractions;
  auto saha_factors = content.saha_factors;
  auto ln_i = content.ln_i;
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

  if (temperature <= saha_min_temperature) {
    set_low_temperature_ionization_state(temperature, rho, content, lambda);
    return;
  }

  // Loop over Saha species – solve ionization for the Saha subset
  for (int e = eb_saha.s; e <= eb_saha.e; ++e) {
    const int z = species(e);
    const double x_e = mass_fractions_nodal(i, q, e);

    const double inv_A = inv_atomic_mass(e);
    const double nk = atom::element_number_density(x_e, inv_A, rho);

    // species atomic data
    const auto species_atomic_data = species_data(ion_data, species_offsets, z);
    const auto species_sum_pot = species_data(sum_pots, species_offsets, z);

    auto ionization_fractions_e =
        Kokkos::subview(ionization_fractions, i, q, e, Kokkos::ALL);

    // reset ionization fractions, setup saha factors
    for (int s = 0; s <= z; ++s) {
      ionization_fractions_e(s) = 0.0;
      saha_factors(s) = saha_f<SolverType>(temperature, species_atomic_data(s));
    }

    double &zbar = zbars(i, q, e);
    if constexpr (SolverType == SahaSolver::Log) {
      const double ln_nk = std::log(nk);

      saha_solve_log(ionization_fractions_e, z, saha_factors, ln_i, rho, ln_nk,
                     zbar);
      if (zbar <= std::log(10.0 * NEUTRAL_ZBAR)) {
        zbar = NEUTRAL_ZBAR;
        ionization_fractions_e(0) = 1.0;
        n_e_solve += zbar * nk;
        continue;
      }
      ionization_fractions_e(0) =
          std::exp(ion_frac0(zbar, saha_factors, ln_i, ln_nk, 1, z + 1));
      zbar = std::exp(zbar);
    } else {
      saha_solve_linear(ionization_fractions_e, z, saha_factors, rho, nk, zbar);
      if (zbar <= 10 * NEUTRAL_ZBAR) {
        zbar = NEUTRAL_ZBAR;
        ionization_fractions_e(0) = 1.0;
        n_e_solve += zbar * nk;
        continue;
      }
      ionization_fractions_e(0) = ion_frac0(zbar, saha_factors, nk, 1, z + 1);
    }
    const int nstates = z + 1;
    n_e_solve += zbar * nk;
    const double inv_zbar_nk = 1.0 / (zbar * nk);
    double y_r = ionization_fractions_e(0);
    double chi_r = 0.0;
    double sum_ion_pot = 0.0;
    for (int s = 1; s < nstates; ++s) {
      if constexpr (SolverType == SahaSolver::Log) {
        ionization_fractions_e(s) = ionization_fractions_e(s - 1) *
                                    std::exp(saha_factors(s - 1)) * inv_zbar_nk;
      } else {
        ionization_fractions_e(s) =
            ionization_fractions_e(s - 1) * saha_factors(s - 1) * inv_zbar_nk;
      }
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
    const double x_e = mass_fractions_nodal(i, q, e);
    const double inv_A = inv_atomic_mass(e);
    const double nk = atom::element_number_density(x_e, inv_A, rho);

    // species atomic data
    const auto species_sum_pot = species_data(sum_pots, species_offsets, z);

    const double sum_ion_pot = species_sum_pot(z - 1);
    const double nu_k = abar * x_e * inv_A;
    n_e_solve += z * nk;

    sum_e_ion_corr += nu_k * sum_ion_pot;
  }

  n_e(i, q) = n_e_solve;
  ybar(i, q) = n_e_solve / N / rho;
  sigma1(i, q) = sum1;
  sigma2(i, q) = sum2;
  sigma3(i, q) = sum3;
  e_ion_corr(i, q) = N * sum_e_ion_corr;

  // Fill lambda
  lambda[eos::paczynski_lambda::number_density] = N;
  lambda[eos::paczynski_lambda::ye] = ye(i, q);
  lambda[eos::paczynski_lambda::ybar] = ybar(i, q);
  lambda[eos::paczynski_lambda::sigma1] = sigma1(i, q);
  lambda[eos::paczynski_lambda::sigma2] = sigma2(i, q);
  lambda[eos::paczynski_lambda::sigma3] = sigma3(i, q);
  lambda[eos::paczynski_lambda::e_ion_corr] = e_ion_corr(i, q);
  lambda[eos::EOS_LAMBDA_TEMPERATURE] = temperature;
}

// Root-form residual of the coupled temperature/ionization system:
// f(T) = model(rho, T, lambda(T)) - target, with lambda(T) the Saha state
// evaluated at the trial temperature. Thermal, ionization, and degeneracy
// contributions all grow with temperature, so f is monotone increasing and
// its root on a sign-changing bracket is the consistent temperature.
// Side effect: leaves the stored ionization state evaluated at `temperature`.
KOKKOS_FUNCTION
template <eos::EOSInversion Inversion, SahaSolver SolverType>
auto coupled_temperature_residual(const double temperature, const double rho,
                                  const eos::EOS &eos,
                                  const CoupledSolverContent &content)
    -> double {
  double lambda[eos::EOS_LAMBDA_SIZE];
  compute_coupled_saha_state<SolverType>(temperature, rho, content, lambda);
  if constexpr (Inversion == eos::EOSInversion::Pressure) {
    return pressure_from_density_temperature(eos, rho, temperature, lambda) -
           content.target_var;
  } else { // sie inversion
    return sie_from_density_temperature(eos, rho, temperature, lambda) -
           content.target_var;
  }
}

template <eos::EOSInversion Inversion, SahaSolver SolverType>
void compute_temperature_with_saha(StageData &stage_data, const Mesh &mesh);

} // namespace athelas::atom
