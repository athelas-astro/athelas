#pragma once

#include <tuple>

#include "Kokkos_Macros.hpp"
#include "basic_types.hpp"
#include "eos/eos_variant.hpp"
#include "opacity/opac_variant.hpp"
#include "solvers/root_finder_opts.hpp"
#include "solvers/root_finders.hpp"
#include "utils.hpp"
#include "utils/riemann.hpp"

namespace athelas::radiation {

/**
 * @struct LLFRiemannState
 * @brief Holds the state for a Rusanov LLF Riemann solver.
 * @note Contains U, F, alpha.
 */
struct LLFRiemannState {
  double u;
  double f;
  double alpha;
};

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
  return std::clamp(std::abs(F) / (constants::c_cgs * E), 0.0, 1.0);
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
 * @brief M1 closure of Levermore 1984
 * @note These should be volumetric.
 * TODO(astrobarker): It would be nice to make this easier to modify
 * Perhaps CRTP model
 */
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto compute_closure(const double E,
                                                          const double F)
    -> double {
  assert(E > 0.0 &&
         "Radiation :: compute_closure(radial) :: Non positive definite "
         "radiation energy density.");
  const double f = flux_factor(E, F);
  const double chi = eddington_factor(f);
  return chi * E;
}

[[nodiscard]] KOKKOS_INLINE_FUNCTION auto p_rad_perp(const double E,
                                                     const double F) -> double {
  assert(E > 0.0 && "Radiation :: p_rad_perp :: Non positive definite "
                    "radiation energy density.");
  const double f = flux_factor(E, F);
  const double chi = eddington_factor(f);
  return E * (1.0 - chi) * 0.5;
}

/**
 * @brief LLF numerical flux
 */
auto KOKKOS_FORCEINLINE_FUNCTION llf_flux(const LLFRiemannState &left,
                                          const LLFRiemannState &right)
    -> double {
  // Weird check here, but to keep Riemann solvers APIs consistent we need
  // the shared wavespeed alpha in the struct.
  assert(left.alpha == right.alpha &&
         "llf_flux: left and right alphas must be identical!");
  return 0.5 * std::fma(left.alpha, (left.u - right.u), (right.f + left.f));
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
auto rad_lambda(const double f, const double sgn_F, const double chi,
                const double chi_prime, const int sign) -> double {
  return constants::c_cgs * 0.5 *
         (chi_prime * sgn_F +
          sign * std::sqrt(chi_prime * chi_prime - 4.0 * chi_prime * f +
                           4.0 * chi));
}

/**
 * @brief Radiation wavespeed
 * @note See Audit et al 2002
 */
KOKKOS_INLINE_FUNCTION
auto rad_wavespeed(const double E_L, const double E_R, const double F_L,
                   const double F_R, const double vstar) -> double {
  using math::utils::sgn;
  const double f_l = flux_factor(E_L, F_L);
  const double f_r = flux_factor(E_R, F_R);
  const double chi_l = eddington_factor(f_l);
  const double chi_r = eddington_factor(f_r);
  const double chi_prime_l = eddington_factor_prime(f_l);
  const double chi_prime_r = eddington_factor_prime(f_r);
  // const double lam_l = rad_lambda(f_l, chi_l, chi_prime_l, +1);
  // const double lam_r = rad_lambda(f_r, chi_r, chi_prime_r, +1);
  // const double res = std::max(lam_l - vstar, lam_r - vstar);
  const double sgn_F_L = sgn(F_L);
  const double sgn_F_R = sgn(F_R);
  const double lam_lp = rad_lambda(f_l, sgn_F_L, chi_l, chi_prime_l, +1);
  const double lam_lm = rad_lambda(f_l, sgn_F_L, chi_l, chi_prime_l, -1);
  const double lam_rp = rad_lambda(f_r, sgn_F_R, chi_r, chi_prime_r, +1);
  const double lam_rm = rad_lambda(f_r, sgn_F_R, chi_r, chi_prime_r, -1);

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
  using math::utils::sgn;

  constexpr static double c2 = constants::c_cgs * constants::c_cgs;
  const double f_l = flux_factor(E_L, F_L);
  const double chi_l = eddington_factor(f_l);
  const double chi_prime_l = eddington_factor_prime(f_l);

  const double sgn_F_L = sgn(F_L);
  const double sgn_F_R = sgn(F_R);
  const double lam_lp = rad_lambda(f_l, sgn_F_L, chi_l, chi_prime_l, +1);
  const double lam_lm = rad_lambda(f_l, sgn_F_L, chi_l, chi_prime_l, -1);

  const double f_r = flux_factor(E_R, F_R);
  const double chi_r = eddington_factor(f_r);
  const double chi_prime_r = eddington_factor_prime(f_r);

  const double lam_rp = rad_lambda(f_r, sgn_F_R, chi_r, chi_prime_r, +1);
  const double lam_rm = rad_lambda(f_r, sgn_F_R, chi_r, chi_prime_r, -1);

  // --- Moving-mesh signal speeds ---
  const double s_l = std::min({lam_lm - vstar, lam_rm - vstar, 0.0});

  const double s_r = std::max({lam_lp - vstar, lam_rp - vstar, 0.0});

  const double flux_e =
      hll(E_L, E_R, F_L - 0 * vstar * E_L, F_R - 0 * vstar * E_R, s_l, s_r);
  const double flux_f = hll(F_L, F_R, c2 * P_L - 0 * vstar * F_L,
                            c2 * P_R - 0 * vstar * F_R, s_l, s_r);
  return {flux_e, flux_f};
}

template <OpacityType Opac>
KOKKOS_INLINE_FUNCTION auto dkappa_dT(const Opacity &opac, const double rho,
                                      const double T, const double X,
                                      const double Z) -> double {
  constexpr double h_log = 1e-4;
  const double logT = std::log10(T);
  const double T_plus = std::pow(10.0, logT + h_log);

  double k_plus = 0.0;
  double k_base = 0.0;

  if constexpr (Opac == OpacityType::Planck) {
    k_plus = opac.planck_mean(rho, T_plus, X, Z, nullptr);
    k_base = opac.planck_mean(rho, T, X, Z, nullptr);
  } else if constexpr (Opac == OpacityType::Rosseland) {
    k_plus = opac.rosseland_mean(rho, T_plus, X, Z, nullptr);
    k_base = opac.rosseland_mean(rho, T, X, Z, nullptr);
  }

  const double dk_dlogT = (k_plus - k_base) / h_log;

  // d(kappa)/dT = d(kappa)/d(log10T) * d(log10T)/dT
  // d(log10T)/dT = 1 / (T * ln(10)) = log10(e) / T
  return dk_dlogT * std::numbers::log10e / T;
}

struct RadSourceInputs {
  double rho, e, v;
  // Specific radiation energy and flux (per unit mass), supplied directly
  // by the caller. Joint transport+source solves use these directly; the
  // legacy source-only newton_radhydro reconstructs them each Newton iter
  // from the conservation invariants (etot, m_tot below) and writes the
  // result into erad, frad.
  double erad, frad;
  double etot, m_tot; // conserved across a source-only step
  double X, Z;
  double dt_a_ii, dg_term;
  const eos::EOS *eos;
  const Opacity *opac;
};

struct RadHydroSources {
  double s_eg;
  double s_v;
  double s_er;
  double s_fr;

  KOKKOS_INLINE_FUNCTION
  RadHydroSources(const RadSourceInputs &in, double *lambda) {
    constexpr double c = constants::c_cgs;
    constexpr double inv_c = 1.0 / c;
    // Volumetric radiation variables. The struct holds specific (per-mass)
    // versions; we convert here.
    const double Er = in.erad * in.rho;
    const double Fr = in.frad * in.rho;

    const double temperature = eos::temperature_from_density_sie(
        *in.eos, in.rho, in.e - 0.5 * in.v * in.v, lambda);

    const double at3 = constants::a * temperature * temperature * temperature;
    const double at4 = at3 * temperature;

    const double kappa_p =
        in.opac->planck_mean(in.rho, temperature, in.X, in.Z, lambda);
    const double kappa_r =
        in.opac->rosseland_mean(in.rho, temperature, in.X, in.Z, lambda);

    const double Pr = compute_closure(Er, Fr);

    const double s_eg_local =
        in.rho * (-c * kappa_p * (at4 - Er) - kappa_r * in.v * Fr * inv_c) *
        in.dg_term;
    s_eg = s_eg_local;
    s_v = in.rho * inv_c *
          (kappa_r * Fr - kappa_p * in.v * at4 - kappa_r * in.v * Pr) *
          in.dg_term;

    s_er = -s_eg; // NOLINT
    s_fr = -c * c * s_v;
  }
};

/**
 * @brief Radhydro exchange source derivatives to v/c
 * NOTE: Derivatives are expected to be with respect to specific quantities.
 */
struct RadHydroSourceDerivatives {
  double dsedeg;
  double dsedv;
  double dseder;
  double dsedfr;
  double dsvdeg;
  double dsvdv;
  double dsvder;
  double dsvdfr;

  KOKKOS_INLINE_FUNCTION
  RadHydroSourceDerivatives(const RadSourceInputs &in, double *lambda) {
    constexpr double c = constants::c_cgs;
    constexpr double c2 = c * c;
    constexpr double inv_c = 1.0 / c;
    constexpr double inv_c2 = 1.0 / c2;

    const double Er = in.erad * in.rho;
    const double Fr = in.frad * in.rho;
    const double v = in.v;
    const double rho = in.rho;

    const double c_rho = c * in.rho;

    const double temperature = eos::temperature_from_density_sie(
        *in.eos, in.rho, in.e - 0.5 * in.v * in.v, lambda);

    // Numerical cv at fixed lambda. We deliberately avoid
    // eos::cv_from_density_temperature here: for EOSs that include
    // ionization state in lambda (Paczynski), the analytic dsie/dT can
    // include Saha-pathway terms that don't apply when lambda is held
    // fixed -- the regime the rest of this Jacobian assumes (no Saha
    // inside compute_rad_sources; only between Newton iters). A symmetric
    // FD on sie_from_density_temperature gives the partial derivative at
    // fixed lambda that matches what temperature_from_density_sie just
    // inverted. Two extra EOS evaluations per cell per Jacobian build.
    const double hT_cv = 1.0e-6 * temperature;
    const double sie_p = eos::sie_from_density_temperature(
        *in.eos, in.rho, temperature + hT_cv, lambda);
    const double sie_m = eos::sie_from_density_temperature(
        *in.eos, in.rho, temperature - hT_cv, lambda);
    const double cv = (sie_p - sie_m) / (2.0 * hT_cv);
    const double inv_cv = 1.0 / cv;

    const double at3 = constants::a * temperature * temperature * temperature;
    const double at4 = at3 * temperature;

    const double delta = at4 - Er;

    const double kappa_p =
        in.opac->planck_mean(in.rho, temperature, in.X, in.Z, lambda);
    const double kappa_r =
        in.opac->rosseland_mean(in.rho, temperature, in.X, in.Z, lambda);
    // Finite difference for opacity derivatives
    const auto *const opac = in.opac;
    const double dkappa_p_dT =
        dkappa_dT<OpacityType::Planck>(*opac, in.rho, temperature, in.X, in.Z);
    const double dkappa_r_dT = dkappa_dT<OpacityType::Rosseland>(
        *opac, in.rho, temperature, in.X, in.Z);

    const double f = flux_factor(Er, Fr);
    const double chi = eddington_factor(f);
    const double chi_prime = eddington_factor_prime(f);
    const double Pr = compute_closure(Er, Fr);
    const double dprder = rho * (chi - f * chi_prime);
    const double dprdfr = rho * inv_c * chi_prime * math::utils::sgn(Fr);

    // Energy source derivatives
    dsedeg = c_rho * inv_cv *
             (-4.0 * at3 * kappa_p - delta * dkappa_p_dT -
              v * Fr * dkappa_r_dT * inv_c2) *
             in.dg_term;
    dsedv = (c_rho * inv_cv *
                 (4 * at3 * in.v * kappa_p + delta * in.v * dkappa_p_dT +
                  Fr * v * v * dkappa_r_dT * inv_c2) -
             rho * Fr * kappa_r * inv_c) *
            in.dg_term;
    dseder = c_rho * rho * kappa_p * in.dg_term;
    dsedfr = -rho * rho * kappa_r * v * inv_c * in.dg_term;

    // Momentum source derivatives
    dsvdeg = rho * inv_c * inv_cv *
             (Fr * dkappa_r_dT - 4.0 * at3 * v * kappa_p -
              at4 * v * dkappa_p_dT - v * Pr * dkappa_r_dT) *
             in.dg_term;
    dsvdv = (rho * inv_c * inv_cv *
                 (-Fr * v * dkappa_r_dT + 4.0 * at3 * kappa_p * v * v +
                  at4 * v * v * dkappa_p_dT + v * v * Pr * dkappa_r_dT) -
             rho * Pr * kappa_r * inv_c - rho * at4 * kappa_p * inv_c) *
            in.dg_term;
    dsvder = -rho * kappa_r * v * inv_c * dprder * in.dg_term;
    dsvdfr = rho * (rho * kappa_r * inv_c - kappa_r * v * inv_c * dprdfr) *
             in.dg_term;
  }
};

} // namespace athelas::radiation
