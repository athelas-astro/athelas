#pragma once

#include <tuple>

#include "Kokkos_Macros.hpp"
#include "solvers/root_finder_opts.hpp"
#include "solvers/root_finders.hpp"
#include "utils/riemann.hpp"

namespace athelas::radiation {

using root_finders::PhysicalScales, root_finders::RadHydroConvergence;

/**
 * @brief Radiation energy density from temperature
 */
KOKKOS_FORCEINLINE_FUNCTION
auto rad_energy(const double T) { return constants::a * T * T * T * T; }

/**
 * @brief radiation flux factor
 **/
KOKKOS_FORCEINLINE_FUNCTION
auto flux_factor(const double E, const double F) -> double {
  assert(E > 0.0 &&
         "Radiation :: flux_factor :: non positive definite energy density.");
  return std::abs(F) / (constants::c_cgs * E);
}

/**
 * @brief return std::tuple containing advective radiation flux
 */
KOKKOS_INLINE_FUNCTION
auto flux_rad(const double E, const double F, const double P, const double V)
    -> std::tuple<double, double> {
  return {F - E * V, constants::c_cgs * constants::c_cgs * P - F * V};
}

KOKKOS_INLINE_FUNCTION
auto eddington_factor(const double f) -> double {
  const double f2 = f * f;
  return (3.0 + 4.0 * f2) / (5.0 + 2.0 * std::sqrt(4.0 - 3.0 * f2));
}

KOKKOS_INLINE_FUNCTION
auto eddington_factor_prime(const double f) -> double {
  return 2.0 * f / (std::sqrt(4.0 - 3 * f * f));
}

/**
 * @brief Radiation 4 force for rad-matter interactions
 * Assumes kappa_e ~ kappa_p, kappa_F ~ kappa_r
 * D : Density
 * V : Velocity
 * T : Temperature
 * kappa_r : rosseland kappa
 * kappa_p : planck kappa
 * E : radiation energy density
 * F : radiation momentum density
 * Pr : radiation momentum closure
 **/
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto
radiation_four_force(const double D, const double V, const double T,
                     const double kappa_r, const double kappa_p, const double E,
                     const double F, const double Pr)
    -> std::tuple<double, double> {
  assert(D >= 0.0 &&
         "Radiation :: RadiationFourFource :: Non positive definite density.");
  assert(T > 0.0 &&
         "Radiation :: RadiationFourFource :: Non positive temperature.");
  assert(E > 0.0 && "Radiation :: RadiationFourFource :: Non positive "
                    "definite radiation energy density.");

  constexpr static double a = constants::a;
  constexpr static double c = constants::c_cgs;

  const double b = V / c;
  const double at4 = a * T * T * T * T;
  const double term1 = E - at4;
  const double Fc = F / c;

  // Krumholz et al. 2007 O(b^2)
  /*
  const double G0 =
      D * (kappa_p * term1 + (kappa_r - 2.0 * kappa_p) * b * Fc +
           0.5 * (2.0 * (kappa_p - kappa_r) * E + kappa_p * term1) * b * b +
           (kappa_p - kappa_r) * b * b * Pr);

  const double G =
      D * (kappa_r * Fc + kappa_p * term1 * b - kappa_r * b * (E + Pr) +
           0.5 * kappa_r * Fc * b * b + 2.0 * (kappa_r - kappa_p) * b * b * Fc);
  */

  // ala Skinner & Ostriker, simpler.
  const double cG0 = D * (c * kappa_p * term1 - kappa_r * b * F);
  const double G = D * (kappa_r * Fc - kappa_p * at4 * b - kappa_r * Pr * b);
  return {cG0, G};
}

/**
 * @brief factor of c scaling terms for radiation-matter sources
 **/
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION auto source_factor_rad()
    -> std::tuple<double, double> {
  constexpr static double c = constants::c_cgs;
  return {c, c * c};
}

/**
 * @brief M1 closure of Levermore 1984
 * TODO(astrobarker): It would be nice to make this easier to modify
 * Perhaps CRTP model
 */
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto compute_closure(const double E,
                                                          const double F)
    -> double {
  assert(E > 0.0 &&
         "Radiation :: compute_closure(radial) :: Non positive definite "
         "radiation energy density.");
  const double f = std::clamp(flux_factor(E, F), 0.0, 1.0);
  const double chi = eddington_factor(f);
  return chi * E;
}

[[nodiscard]] KOKKOS_INLINE_FUNCTION auto p_rad_perp(const double E,
                                                     const double F) -> double {
  assert(E > 0.0 && "Radiation :: p_rad_perp :: Non positive definite "
                    "radiation energy density.");
  const double f = std::clamp(flux_factor(E, F), 0.0, 1.0);
  const double chi = eddington_factor(f);
  return E * (1.0 - chi) * 0.5;
}

/**
 * @brief LLF numerical flux
 */
auto KOKKOS_FORCEINLINE_FUNCTION llf_flux(const double Fp, const double Fm,
                                          const double Up, const double Um,
                                          const double alpha) -> double {
  return 0.5 * std::fma(alpha, (Um - Up), (Fp + Fm));
}

/**
 * @brief eigenvalues of Jacobian for radiation solve
 * see 2013ApJS..206...21S (Skinner & Ostriker 2013) Eq 41a,b
 * and references therein
 **/
KOKKOS_INLINE_FUNCTION
auto lambda_hll(const double f, const int sign) -> double {
  constexpr static double c = constants::c_cgs;
  constexpr static double twothird = 2.0 / 3.0;

  const double f2 = f * f;
  const double sqrtterm = std::sqrt(4.0 - (3.0 * f2));
  auto res = c *
             (f + sign * std::sqrt((twothird * (4.0 - 3.0 * f2 - sqrtterm)) +
                                   (2.0 * (2.0 - f2 - sqrtterm)))) /
             sqrtterm;
  return res;
}

/**
 * @brief \lambda^{+/-} wavespeed
 * @note See Audit et al 2002
 */
KOKKOS_INLINE_FUNCTION
auto rad_lambda(const double f, const double chi, const double chi_prime,
                const int sign) -> double {
  return constants::c_cgs * 0.5 *
         (chi_prime + sign * std::sqrt(chi_prime * chi_prime -
                                       4.0 * chi_prime * f + 4.0 * chi));
}

/**
 * @brief Radiation wavespeed
 * @note See Audit et al 2002
 */
KOKKOS_INLINE_FUNCTION
auto rad_wavespeed(const double E_L, const double E_R, const double F_L,
                   const double F_R, const double vstar) -> double {
  const double f_l = flux_factor(E_L, F_L);
  const double f_r = flux_factor(E_R, F_R);
  const double chi_l = eddington_factor(f_l);
  const double chi_r = eddington_factor(f_r);
  const double chi_prime_l = eddington_factor_prime(f_l);
  const double chi_prime_r = eddington_factor_prime(f_r);
  // const double lam_l = rad_lambda(f_l, chi_l, chi_prime_l, +1);
  // const double lam_r = rad_lambda(f_r, chi_r, chi_prime_r, +1);
  // const double res = std::max(lam_l - vstar, lam_r - vstar);
  const double lam_lp = rad_lambda(f_l, chi_l, chi_prime_l, +1);
  const double lam_lm = rad_lambda(f_l, chi_l, chi_prime_l, -1);
  const double lam_rp = rad_lambda(f_r, chi_r, chi_prime_r, +1);
  const double lam_rm = rad_lambda(f_r, chi_r, chi_prime_r, -1);

  const double alpha =
      std::max({std::abs(lam_lp - vstar), std::abs(lam_lm - vstar),
                std::abs(lam_rp - vstar), std::abs(lam_rm - vstar)});
  return alpha;
}

/**
 * @brief HLL Riemann solver for radiation
 * see 2013ApJS..206...21S (Skinner & Ostriker 2013) Eq 39
 * and references & discussion therein
 **/
KOKKOS_INLINE_FUNCTION
auto numerical_flux_hll_rad(const double E_L, const double E_R,
                            const double F_L, const double F_R,
                            const double P_L, const double P_R,
                            const double vstar) -> std::tuple<double, double> {
  using namespace riemann;

  constexpr static double c2 = constants::c_cgs * constants::c_cgs;
  const double f_l = flux_factor(E_L, F_L);
  const double chi_l = eddington_factor(f_l);
  const double chi_prime_l = eddington_factor_prime(f_l);

  const double lam_lp = rad_lambda(f_l, chi_l, chi_prime_l, +1);
  const double lam_lm = rad_lambda(f_l, chi_l, chi_prime_l, -1);

  const double f_r = flux_factor(E_R, F_R);
  const double chi_r = eddington_factor(f_r);
  const double chi_prime_r = eddington_factor_prime(f_r);

  const double lam_rp = rad_lambda(f_r, chi_r, chi_prime_r, +1);
  const double lam_rm = rad_lambda(f_r, chi_r, chi_prime_r, -1);

  // --- Moving-mesh signal speeds ---
  const double s_l = std::min({lam_lm - vstar, lam_rm - vstar, 0.0});

  const double s_r = std::max({lam_lp - vstar, lam_rp - vstar, 0.0});

  const double flux_e =
      hll(E_L, E_R, F_L - 0 * vstar * E_L, F_R - 0 * vstar * E_R, s_l, s_r);
  const double flux_f = hll(F_L, F_R, c2 * P_L - 0 * vstar * F_L,
                            c2 * P_R - 0 * vstar * F_R, s_l, s_r);
  return {flux_e, flux_f};
}

} // namespace athelas::radiation
