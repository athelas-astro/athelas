/**
 * @file characteristic_decomposition.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Characteristic decomposition for limiter variables
 *
 * @details Implements characteristic decompositions for the 3-variable
 *          Lagrangian hydro system and the 2-variable M1 radiation system.
 */

#pragma once

#include <cassert>
#include <cmath>

#include "basic_types.hpp"
#include "eos/eos_variant.hpp"
#include "math/difference.hpp"
#include "math/linear_algebra.hpp"
#include "radiation/rad_utilities.hpp"
#include "utils/constants.hpp"

namespace athelas {

template <class M, class V>
KOKKOS_INLINE_FUNCTION void characteristic_mat_vec(const int n, const double a,
                                                   M A, V x, const double b,
                                                   V y) {
  for (int r = 0; r < n; ++r) {
    double sum = 0.0;
    for (int c = 0; c < n; ++c) {
      sum += A(r, c) * x(c);
    }
    y(r) = a * sum + b * y(r);
  }
}

// Characteristic limiting is defined for the 2-variable (radiation) and
// 3-variable (hydro) systems; larger systems fall back to component-wise
// limiting. These bound the per-cell scratch used while transforming a block
// of modal coefficients into / out of characteristic variables.
inline constexpr std::size_t max_characteristic_vars = 3;
inline constexpr std::size_t max_characteristic_modes = 6;

// Fixed-capacity (mode, variable) scratch block, device-callable. Holds one
// cell's modal coefficients in characteristic space during limiting. Indexed
// as block(k, v) for mode k and characteristic field v.
struct CharBlock {
  Kokkos::Array<double, max_characteristic_modes * max_characteristic_vars>
      data{};
  KOKKOS_INLINE_FUNCTION auto operator()(const int k, const int v) -> double & {
    return data[(static_cast<std::size_t>(k) * max_characteristic_vars) + v];
  }
  KOKKOS_INLINE_FUNCTION auto operator()(const int k, const int v) const
      -> double {
    return data[(static_cast<std::size_t>(k) * max_characteristic_vars) + v];
  }
};

// Project a block of modal coefficients into characteristic variables using the
// left eigenvectors R_inv:  w(k, m) = sum_v R_inv(m, v) u(k, v), for the first
// n_modes modes. Source and destination must not alias.
template <class MAT, class SRC, class DST>
KOKKOS_INLINE_FUNCTION void to_characteristic(const MAT &R_inv, const SRC &u,
                                              DST &w, const int n_modes,
                                              const int nvars) {
  for (int k = 0; k < n_modes; ++k) {
    for (int m = 0; m < nvars; ++m) {
      double sum = 0.0;
      for (int v = 0; v < nvars; ++v) {
        sum += R_inv(m, v) * u(k, v);
      }
      w(k, m) = sum;
    }
  }
}

// Reconstruct conserved modal coefficients from characteristic variables using
// the right eigenvectors R:  u(k, v) = sum_m R(v, m) w(k, m), for the first
// n_modes modes. Inverse of to_characteristic. Source and destination must not
// alias.
template <class MAT, class SRC, class DST>
KOKKOS_INLINE_FUNCTION void from_characteristic(const MAT &R, const SRC &w,
                                                DST &u, const int n_modes,
                                                const int nvars) {
  for (int k = 0; k < n_modes; ++k) {
    for (int v = 0; v < nvars; ++v) {
      double sum = 0.0;
      for (int m = 0; m < nvars; ++m) {
        sum += R(v, m) * w(k, m);
      }
      u(k, v) = sum;
    }
  }
}

// Temperature used to build the local thermodynamic derivatives. The limiter
// dispatch fills the reserved temperature slot with the cell-average gas
// temperature. If that slot is not populated, the characteristic decomposition
// was called incorrectly.
KOKKOS_INLINE_FUNCTION auto characteristic_temperature(const double *lambda)
    -> double {
  const double temp = lambda[eos::EOS_LAMBDA_TEMPERATURE];
  assert(temp > 0.0 &&
         "characteristic_temperature: EOS lambda temperature slot must be set");
  assert(
      std::isfinite(temp) &&
      "characteristic_temperature: EOS lambda temperature slot must be finite");
  return temp;
}

// dp/dE|_{tau,v} = (dp/dT)_rho / (de/dT)_rho at fixed composition/ionization
// state. This avoids finite-differencing pressure_from_conserved, which would
// require a fresh temperature root-find at E +/- dE and is fragile near EOS
// floors. dp/dT is differenced in temperature space; cv supplies de/dT.
KOKKOS_INLINE_FUNCTION auto
pressure_derivative_energy(const eos::EOS &eos, const double rho,
                           const double temp, const double *const lambda)
    -> double {
  const double cv = cv_from_density_temperature(eos, rho, temp, lambda);
  if (!(cv > 0.0) || !std::isfinite(cv)) {
    return NAN;
  }

  const double dT = std::max(1.0e-6 * std::abs(temp), 1.0e-20);
  const auto pressure_of_temperature = [&](const double t) {
    return pressure_from_density_temperature(eos, rho, t, lambda);
  };

  using math::difference::finite_difference;
  double dpdT = NAN;
  if (temp > 2.0 * dT) {
    dpdT = finite_difference<DiffScheme::Central>(dT, pressure_of_temperature,
                                                  temp);
  } else {
    dpdT = finite_difference<DiffScheme::Forward>(dT, pressure_of_temperature,
                                                  temp);
  }

  return dpdT / cv;
}

template <class T1, class T2>
KOKKOS_INLINE_FUNCTION void
compute_radiation_characteristic_decomposition(T1 U, T2 R, T2 R_inv) {
  using math::linalg::fill_identity;
  constexpr double c = constants::c_cgs;
  using radiation::eddington_factor;
  using radiation::eddington_factor_prime;
  using radiation::flux_factor;

  const double e_rad = U(0);
  const double f_rad = U(1);

  const double f = flux_factor(e_rad, f_rad);
  const double chi = eddington_factor(f);
  const double chi_prime = eddington_factor_prime(f);
  const double sgn_f = math::utils::sgn(f_rad);
  const double a = chi - f * chi_prime;
  const double b = c * chi_prime * sgn_f;
  const double discr = std::max(0.0, b * b + 4.0 * c * c * a);
  const double lam_m = 0.5 * (b - std::sqrt(discr));
  const double lam_p = 0.5 * (b + std::sqrt(discr));
  const double inv_denom = 1.0 / (lam_p - lam_m);

  // Guard against ill conditioned state.
  if (!(inv_denom > 0.0) || !std::isfinite(inv_denom)) {
    fill_identity(R, 2);
    fill_identity(R_inv, 2);
    return;
  }

  R(0, 0) = 1.0;
  R(1, 0) = lam_m;
  R(0, 1) = 1.0;
  R(1, 1) = lam_p;

  R_inv(0, 0) = lam_p * inv_denom;
  R_inv(0, 1) = -1.0 * inv_denom;
  R_inv(1, 0) = -lam_m * inv_denom;
  R_inv(1, 1) = 1.0 * inv_denom;
}

template <class T1, class T2, class EOS>
KOKKOS_INLINE_FUNCTION void
compute_hydro_characteristic_decomposition(T1 U, T2 R, T2 R_inv, EOS eos,
                                           const double *const lambda) {

  const double tau = U(0);
  const double v = U(1);
  const double rho = 1.0 / tau;

  const double temp = characteristic_temperature(lambda);
  const double p = pressure_from_density_temperature(eos, rho, temp, lambda);
  const double cs =
      sound_speed_from_density_temperature_pressure(eos, rho, temp, p, lambda);
  const double z = cs / tau;
  const double p_E = pressure_derivative_energy(eos, rho, temp, lambda);

  const double p_tau = p * p_E - z * z;
  const double entropy_energy = -p_tau / p_E;

  R(0, 0) = 1.0;
  R(1, 0) = 0.0;
  R(2, 0) = entropy_energy;

  R(0, 1) = 1.0;
  R(1, 1) = z;
  R(2, 1) = -p + z * v;

  R(0, 2) = 1.0;
  R(1, 2) = -z;
  R(2, 2) = -p - z * v;

  math::linalg::invert_3x3<math::linalg::InversionFallback::Identity>(R, R_inv);
}

// Dispatch by system size. `lambda` carries any EOS-specific state the hydro
// decomposition needs (e.g. the Paczynski ionization slots); it is unused by
// the M1 radiation path. For EOSs that ignore the lambda (ideal gas, Marshak)
// a default lambda is fine. Temperature is expected to be filled correctly.
template <class T1, class T2, class EOS>
KOKKOS_INLINE_FUNCTION void
compute_characteristic_decomposition(T1 U, T2 R, T2 R_inv, EOS eos,
                                     const double *const lambda) {
  using math::linalg::fill_identity;
  const int nvars = static_cast<int>(R.extent(0));
  if (nvars == 2) {
    compute_radiation_characteristic_decomposition(U, R, R_inv);
    return;
  }
  if (nvars == 3) {
    compute_hydro_characteristic_decomposition(U, R, R_inv, eos, lambda);
    return;
  }

  fill_identity(R, nvars);
  fill_identity(R_inv, nvars);
}

} // namespace athelas
