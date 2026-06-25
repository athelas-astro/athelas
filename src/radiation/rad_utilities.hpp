#pragma once

#include <algorithm>
#include <cmath>
#include <tuple>

#include "Kokkos_Macros.hpp"
#include "basic_types.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/mesh.hpp"
#include "math/difference.hpp"
#include "opacity/opac_variant.hpp"
#include "solvers/root_finders.hpp"
#include "utils.hpp"
#include "utils/error.hpp"
#include "utils/riemann.hpp"

namespace athelas::radiation {

struct RadHydroSolverIonizationContent {
  double number_density{};
  double ye{};
  double ybar{};
  double sigma1{};
  double sigma2{};
  double sigma3{};
  double e_ion_corr{};
  double X{};
  double Z{};
};

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

struct RadBoundaryState {
  double E;
  double F;
};

struct RadNumericalFlux {
  double e;
  double f;
};

KOKKOS_FORCEINLINE_FUNCTION
auto ap_dissipation_factor(const double tau, const double coefficient = 1.0)
    -> double {
  assert(tau >= 0.0 && "ap_dissipation_factor: optical depth must be >= 0!");
  assert(coefficient >= 0.0 &&
         "ap_dissipation_factor: coefficient must be >= 0!");
  return 1.0 / (1.0 + coefficient * tau);
}

/**
 * @brief Optical depth kappa_R * rho * dr contributed by one cell side of a
 *        face, used to build the AP LLF dissipation factor.
 * @note `node` selects the cell-local node adjacent to the face (0 = left
 *       face, nNodes + 1 = right face). X/Z are read from the bulk composition
 *       when enabled, otherwise treated as zero. The opacity temperature is the
 *       (frozen) stage gas temperature from the derived field.
 * TODO(astrobarker): [optical depth] Is there a more "DG" way to do this?
 */
template <typename Opacity>
KOKKOS_FORCEINLINE_FUNCTION auto
cell_optical_depth(const Opacity &opac, const AthelasArray3D<double> derived,
                   const AthelasArray3D<double> bulk,
                   const bool composition_enabled, const int idx_tgas,
                   const int cell, const int node, const double rho,
                   const double dr) -> double {
  const double T = derived(cell, node, idx_tgas);
  double X = 0.0;
  double Z = 0.0;
  if (composition_enabled) {
    X = bulk(cell, node, 0);
    Z = bulk(cell, node, 2);
  }
  eos::EOSLambda lambda;
  lambda.data[eos::EOS_LAMBDA_TEMPERATURE] = T;
  const double kappa_R = opac.rosseland_mean(rho, T, X, Z, lambda.ptr());
  return kappa_R * rho * dr;
}

/**
 * @brief Combine left/right cell-side optical depths into a single face value
 *        for the AP LLF dissipation factor.
 * TODO(astrobarker): try a harmonic mean of the two sides here; it may behave
 * better than this arithmetic mean across sharp opacity jumps (e.g.
 * composition boundaries).
 */
KOKKOS_FORCEINLINE_FUNCTION
auto face_optical_depth(const double tau_left, const double tau_right)
    -> double {
  return 0.5 * (tau_left + tau_right);
}

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
                                          const LLFRiemannState &right,
                                          const double beta = 1.0) -> double {
  // Weird check here, but to keep Riemann solvers APIs consistent we need
  // the shared wavespeed alpha in the struct.
  assert(left.alpha == right.alpha &&
         "llf_flux: left and right alphas must be identical!");
  assert(beta >= 0.0 && beta <= 1.0 &&
         "llf_flux: AP dissipation factor must be in [0, 1]!");
  return 0.5 *
         std::fma(beta * left.alpha, (left.u - right.u), (right.f + left.f));
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
  const double discriminant =
      std::max(chi_prime * chi_prime - 4.0 * chi_prime * f + 4.0 * chi, 0.0);
  return constants::c_cgs * 0.5 *
         (chi_prime * sgn_F + sign * std::sqrt(discriminant));
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

KOKKOS_INLINE_FUNCTION
auto free_streaming_boundary_flux_rad(const Boundary side, const double E,
                                      const double F, const double vstar)
    -> std::tuple<double, double> {
  constexpr double c = constants::c_cgs;
  const double f = flux_factor(E, F);
  const double chi = eddington_factor(f);
  const double chi_prime = eddington_factor_prime(f);
  const double sgn_F = math::utils::sgn(F);
  const double lambda_m = rad_lambda(f, sgn_F, chi, chi_prime, -1) - vstar;
  const double lambda_p = rad_lambda(f, sgn_F, chi, chi_prime, +1) - vstar;

  const bool use_m =
      (side == Boundary::Interior) ? (lambda_m < 0.0) : (lambda_m > 0.0);
  const bool use_p =
      (side == Boundary::Interior) ? (lambda_p < 0.0) : (lambda_p > 0.0);

  if (use_m && use_p) {
    return flux_rad(E, F, chi * E, vstar);
  }
  if (!use_m && !use_p) {
    return {0.0, 0.0};
  }

  const double s_m = lambda_m + vstar;
  const double s_p = lambda_p + vstar;
  const double denom = s_p - s_m;
  if (std::abs(denom) <= 1.0e-14 * c) {
    return flux_rad(E, F, chi * E, vstar);
  }

  const double w_m = (s_p * E - F) / denom;
  const double w_p = (F - s_m * E) / denom;

  double flux_e = 0.0;
  double flux_f = 0.0;
  if (use_m) {
    flux_e += lambda_m * w_m;
    flux_f += lambda_m * w_m * s_m;
  }
  if (use_p) {
    flux_e += lambda_p * w_p;
    flux_f += lambda_p * w_p * s_p;
  }
  return {flux_e, flux_f};
}

KOKKOS_INLINE_FUNCTION
auto interior_boundary_flux_rad(const double E, const double F,
                                const double vstar)
    -> std::tuple<double, double> {
  return flux_rad(E, F, compute_closure(E, F), vstar);
}

KOKKOS_INLINE_FUNCTION
auto numerical_flux_llf_rad(const RadBoundaryState &left,
                            const RadBoundaryState &right, const double vstar,
                            const double beta = 1.0) -> RadNumericalFlux {
  constexpr double c2 = constants::c_cgs * constants::c_cgs;
  const double P_L = compute_closure(left.E, left.F);
  const double P_R = compute_closure(right.E, right.F);
  const double alpha = rad_wavespeed(left.E, right.E, left.F, right.F, vstar);

  const LLFRiemannState left_erad{
      .u = left.E, .f = left.F - vstar * left.E, .alpha = alpha};
  const LLFRiemannState right_erad{
      .u = right.E, .f = right.F - vstar * right.E, .alpha = alpha};

  const LLFRiemannState left_frad{
      .u = left.F, .f = c2 * P_L - vstar * left.F, .alpha = alpha};
  const LLFRiemannState right_frad{
      .u = right.F, .f = c2 * P_R - vstar * right.F, .alpha = alpha};

  return {.e = llf_flux(left_erad, right_erad, beta),
          .f = llf_flux(left_frad, right_frad, beta)};
}

KOKKOS_INLINE_FUNCTION
auto boundary_flux_rad(const Boundary side, const bc::BcType type,
                       const RadBoundaryState &interior,
                       const RadBoundaryState &exterior, const double vstar,
                       const double marshak_einc = 0.0, const double beta = 1.0)
    -> RadNumericalFlux {
  // Reflecting / Marshak set a ghost state and share the LLF call below; the
  // fallback (fluid-only types, excluded by runtime validation) keeps
  // boundary = exterior.
  RadBoundaryState boundary = exterior;
  switch (type) {
  case bc::BcType::InteriorFlux: {
    const auto [flux_e, flux_f] =
        interior_boundary_flux_rad(interior.E, interior.F, vstar);
    return {.e = flux_e, .f = flux_f};
  }
  case bc::BcType::FreeStreaming: {
    const auto [flux_e, flux_f] =
        free_streaming_boundary_flux_rad(side, interior.E, interior.F, vstar);
    return {.e = flux_e, .f = flux_f};
  }
  case bc::BcType::Periodic:
    return (side == Boundary::Interior)
               ? numerical_flux_llf_rad(exterior, interior, vstar, beta)
               : numerical_flux_llf_rad(interior, exterior, vstar, beta);
  case bc::BcType::Reflecting:
    boundary = {.E = interior.E, .F = -interior.F};
    break;
  case bc::BcType::Marshak: {
    constexpr double c = constants::c_cgs;
    boundary = {.E = marshak_einc,
                .F = 0.5 * c * marshak_einc -
                     0.5 * (c * interior.E + 2.0 * interior.F)};
    break;
  }
  case bc::BcType::Outflow:
  case bc::BcType::Surface:
  case bc::BcType::Null:
    break; // fluid-only / invalid for radiation — excluded by runtime
           // validation. No default: a new BcType must be handled explicitly.
  }

  return (side == Boundary::Interior)
             ? numerical_flux_llf_rad(boundary, interior, vstar, beta)
             : numerical_flux_llf_rad(interior, boundary, vstar, beta);
}

KOKKOS_INLINE_FUNCTION
auto numerical_flux_rad_with_boundary(
    const int face, const int inner_face, const int outer_face,
    const Kokkos::Array<bc::BoundaryConditionData, 2> &bcs,
    const RadBoundaryState &left, const RadBoundaryState &right,
    const double vstar, const double beta = 1.0) -> RadNumericalFlux {
  if (face == inner_face) {
    return boundary_flux_rad(Boundary::Interior, bcs[0].type, right, left,
                             vstar, bcs[0].marshak_incoming_energy, beta);
  }
  if (face == outer_face) {
    return boundary_flux_rad(Boundary::Exterior, bcs[1].type, left, right,
                             vstar, bcs[1].marshak_incoming_energy, beta);
  }

  return numerical_flux_llf_rad(left, right, vstar, beta);
}

KOKKOS_INLINE_FUNCTION
auto free_streaming_flux_jacobian(const Boundary side,
                                  const RadBoundaryState &interior,
                                  const double vstar, const double rho,
                                  const int flux_index, const int var_index)
    -> double {
  constexpr double c = constants::c_cgs;
  constexpr double eps = 1.0e-6;

  const double scale = (var_index == 0)
                           ? std::max(std::abs(interior.E), 1.0)
                           : std::max(std::abs(interior.F), c * interior.E);
  double h = eps * scale;

  if (var_index == 0) {
    h = std::min(h, 0.5 * interior.E);
  }

  const auto flux_component = [&](const double x) {
    RadBoundaryState state = interior;
    if (var_index == 0) {
      state.E = x;
    } else {
      state.F = x;
    }
    const auto flux =
        boundary_flux_rad(side, bc::BcType::FreeStreaming, state, state, vstar);
    return (flux_index == 0) ? flux.e : flux.f;
  };

  const double x = (var_index == 0) ? interior.E : interior.F;
  return rho * math::difference::finite_difference<DiffScheme::Central>(
                   h, flux_component, x);
}

// Jacobian of the radiation boundary flux (component flux_index) with respect
// to the per-mass radiation variable (var_index: 0 = E, 1 = F) that the
// implicit solve evolves. The *rho factor converts the volumetric-variable
// derivative to the specific (per-mass) one.
KOKKOS_INLINE_FUNCTION
auto boundary_flux_jacobian(const Boundary side, const bc::BcType type,
                            const RadBoundaryState &interior,
                            const double vstar, const double rho,
                            const int flux_index, const int var_index,
                            const double marshak_einc = 0.0,
                            const double beta = 1.0) -> double {
  if (type == bc::BcType::Periodic) {
    throw_athelas_error(
        "Periodic implicit radiation boundaries need cyclic coupling.");
  }

  if (type == bc::BcType::FreeStreaming || type == bc::BcType::Reflecting ||
      type == bc::BcType::Marshak) {
    if (type == bc::BcType::FreeStreaming) {
      return free_streaming_flux_jacobian(side, interior, vstar, rho,
                                          flux_index, var_index);
    }

    constexpr double eps = 1.0e-6;
    const double scale =
        (var_index == 0)
            ? std::max(std::abs(interior.E), 1.0)
            : std::max(std::abs(interior.F),
                       constants::c_cgs * std::max(interior.E, 1.0));
    double h = eps * scale;

    if (var_index == 0) {
      h = std::min(h, 0.5 * interior.E);
    }

    const auto flux_component = [&](const double x) {
      RadBoundaryState state = interior;
      if (var_index == 0) {
        state.E = x;
      } else {
        state.F = x;
      }
      const auto flux = boundary_flux_rad(side, type, state, state, vstar,
                                          marshak_einc, beta);
      return (flux_index == 0) ? flux.e : flux.f;
    };

    const double x = (var_index == 0) ? interior.E : interior.F;
    return rho * math::difference::finite_difference<DiffScheme::Central>(
                     h, flux_component, x);
  }

  constexpr double c = constants::c_cgs;
  constexpr double c2 = c * c;
  const double f = flux_factor(interior.E, interior.F);
  const double chi = eddington_factor(f);
  const double chi_prime = eddington_factor_prime(f);

  if (flux_index == 0 && var_index == 0) {
    return rho * (-vstar);
  }
  if (flux_index == 0 && var_index == 1) {
    return rho;
  }
  if (flux_index == 1 && var_index == 0) {
    return rho * c2 * (chi - f * chi_prime);
  }
  if (flux_index == 1 && var_index == 1) {
    return rho * (c * chi_prime * math::utils::sgn(interior.F) - vstar);
  }
  return 0.0;
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
  double dt_a_ii;
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

    s_eg = -c * kappa_p * (at4 - Er) - kappa_r * in.v * Fr * inv_c;
    s_v = inv_c * (kappa_r * Fr - kappa_p * in.v * at4 - kappa_r * in.v * Pr);

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
    double cv = (sie_p - sie_m) / (2.0 * hT_cv);
    if (!std::isfinite(cv) || cv <= 0.0) {
      cv = eos::cv_from_density_temperature(*in.eos, in.rho, temperature,
                                            lambda);
    }
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
    dsedeg = c * inv_cv *
             (-4.0 * at3 * kappa_p - delta * dkappa_p_dT -
              v * Fr * dkappa_r_dT * inv_c2);
    dsedv = c * inv_cv *
                (4 * at3 * in.v * kappa_p + delta * in.v * dkappa_p_dT +
                 Fr * v * v * dkappa_r_dT * inv_c2) -
            Fr * kappa_r * inv_c;
    dseder = c * rho * kappa_p;
    dsedfr = -rho * kappa_r * v * inv_c;

    // Momentum source derivatives
    dsvdeg = inv_c * inv_cv *
             (Fr * dkappa_r_dT - 4.0 * at3 * v * kappa_p -
              at4 * v * dkappa_p_dT - v * Pr * dkappa_r_dT);
    dsvdv = inv_c * inv_cv *
                (-Fr * v * dkappa_r_dT + 4.0 * at3 * kappa_p * v * v +
                 at4 * v * v * dkappa_p_dT + v * v * Pr * dkappa_r_dT) -
            Pr * kappa_r * inv_c - at4 * kappa_p * inv_c;
    dsvder = -kappa_r * v * inv_c * dprder;
    dsvdfr = rho * kappa_r * inv_c - kappa_r * v * inv_c * dprdfr;
  }
};

KOKKOS_INLINE_FUNCTION
auto compute_rad_sources(const RadSourceInputs &in, double *lambda)
    -> std::tuple<double, double> {
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

  const double se = -c * kappa_p * (at4 - Er) - kappa_r * in.v * Fr * inv_c;
  const double sv =
      inv_c * (kappa_r * Fr - kappa_p * in.v * at4 - kappa_r * in.v * Pr);
  return {se, sv};
}

template <DiffScheme Scheme = DiffScheme::Forward>
KOKKOS_INLINE_FUNCTION auto finite_diff_source(const RadSourceInputs &in,
                                               double *lambda)
    -> std::tuple<double, double, double, double> {
  constexpr double c2 = constants::c_cgs * constants::c_cgs;
  constexpr double h_base = (Scheme == DiffScheme::Central) ? 1.0e-6 : 1.0e-8;
  constexpr double tol = 1.0e-14;

  const double etot = in.etot;

  double dsede;
  double dsedv;
  double dsvde;
  double dsvdv;

  if constexpr (Scheme == DiffScheme::Forward ||
                Scheme == DiffScheme::Backward) {
    // --- Forward / Backward Scheme ---
    // In the forward scheme we check that e + h < etot.
    // If we cross that bounds we switch to a backwards difference.
    const auto [se0, sv0] = compute_rad_sources(in, lambda);

    // Energy
    {
      const double h_e = h_base * std::abs(in.e) + tol;
      const double side = (in.e + h_e > etot) ? -1.0 : 1.0;

      auto in_p = in;
      in_p.e += side * h_e;
      in_p.erad -= side * h_e;
      const auto [sep, svp] = compute_rad_sources(in_p, lambda);

      dsede = (sep - se0) / (side * h_e);
      dsvde = (svp - sv0) / (side * h_e);
    }
    // Velocity
    {
      const double h_v = 100.0 * h_base * std::abs(in.v) + tol;
      auto in_p = in;
      in_p.v += h_v;
      in_p.frad -= c2 * h_v;
      const auto [sep, svp] = compute_rad_sources(in_p, lambda);

      dsedv = (sep - se0) / h_v;
      dsvdv = (svp - sv0) / h_v;
    }

  } else {
    // --- Central Difference ---
    // Energy
    {
      const double h_e = h_base * std::abs(in.e) + tol;
      auto in_p = in;
      auto in_m = in;

      in_p.e += h_e;
      in_p.erad -= h_e;
      in_m.e -= h_e;
      in_m.erad += h_e;

      const auto [sep, svp] = compute_rad_sources(in_p, lambda);
      const auto [sem, svm] = compute_rad_sources(in_m, lambda);

      const double inv_2h = 0.5 / h_e;
      dsede = (sep - sem) * inv_2h;
      dsvde = (svp - svm) * inv_2h;
    }

    // Velocity
    {
      const double h_v = h_base * std::abs(in.v) + tol;
      auto in_p = in;
      auto in_m = in;

      in_p.v += h_v;
      in_p.frad -= c2 * h_v;
      in_m.v -= h_v;
      in_m.frad += c2 * h_v;

      const auto [sep, svp] = compute_rad_sources(in_p, lambda);
      const auto [sem, svm] = compute_rad_sources(in_m, lambda);

      const double inv_2h = 0.5 / h_v;
      dsedv = (sep - sem) * inv_2h;
      dsvdv = (svp - svm) * inv_2h;
    }
  }

  return {dsede, dsedv, dsvde, dsvdv};
}

} // namespace athelas::radiation
