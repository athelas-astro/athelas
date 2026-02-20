/**
 * @file utilities.hpp
 * --------------
 *
 * @brief Useful utilities
 *
 * @details Provides
 *          - LINTERP
 *          - SGN
 *          - compute_internal_energy
 *          - to_lower
 */

#pragma once

#include <algorithm>
#include <cctype>

#include "Kokkos_Macros.hpp"

#include "basis/nodal_basis.hpp"
#include "basis/polynomial_basis.hpp"
#include "concepts/arithmetic.hpp"
#include "concepts/types.hpp"
#include "kokkos_types.hpp"

namespace athelas::utilities {
using basis::NodalBasis;

/**
 * @brief constexpr integer power function
 * requires a positive exponent
 */
template <Multipliable T>
constexpr T pow_int(T base, unsigned int exp) noexcept(noexcept(base * base)) {
  T result = T(1);
  while (exp > 0) {
    if (exp & 1) result = result * base; // multiply once
    base = base * base; // square
    exp >>= 1; // divide exponent by 2
  }
  return result;
}
/**
 * @brief simple linear interpolation to x
 *
 * Uses fused multiply add (std::fma) to reduce rounding errors when available.
 */
KOKKOS_FUNCTION
template <typename T>
constexpr auto LINTERP(T x0, T x1, T y0, T y1, T x) noexcept -> T {
  if (x0 == x1) {
    return y0;
  }
  const T t = (x - x0) / (x1 - x0);
  return std::fma(y1 - y0, t, y0);
}

KOKKOS_FUNCTION
template <VectorLike T>
auto find_closest_cell(T r_view, const double target_r, int num_cells) -> int {
  int left = 0;
  int right = num_cells - 1;

  while (left <= right) {
    const int mid = (left + right) / 2;

    const double cell_r = r_view[mid];

    if (cell_r < target_r) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  // Return closest cell (handle boundary cases)
  if (left >= num_cells) {
    return num_cells - 1;
  }
  if (right < 0) {
    return 0;
  }

  return (r_view[left] < target_r) ? left : left - 1;
}

// [[x]]_+ = -.5 * (x + |x|) is positive part of x
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto pos_part(const double x) noexcept
    -> double {
  return 0.5 * (x + std::abs(x));
}

template <typename T = double>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto EPS() {
  return 10 * std::numeric_limits<T>::epsilon();
}

template <typename T = double>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto LARGE() {
  return 0.1 * std::numeric_limits<T>::max();
}

template <typename T = double>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto SMALL() {
  return 10 * std::numeric_limits<T>::min();
}

KOKKOS_FORCEINLINE_FUNCTION
auto make_bounded(const double val, const double vmin, const double vmax)
    -> double {
  return std::min(std::max(val, vmin + EPS()), vmax * (1.0 - EPS()));
}

// Implements a typesafe SGN function
// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-SGN-in-c-c
template <typename T>
KOKKOS_INLINE_FUNCTION constexpr auto SGN(T val) -> int {
  return (T(0) < val) - (val < T(0));
}

template <typename A, typename B>
KOKKOS_INLINE_FUNCTION auto ratio(const A &a, const B &b) {
  const B sgn = b >= 0 ? 1 : -1;
  return a / (b + sgn * SMALL<B>());
}

// nodal specific internal energy
template <class T>
KOKKOS_INLINE_FUNCTION auto
compute_internal_energy(T U, const AthelasArray3D<double> phi, const int ix,
                        const int iN) -> double {
  using basis::basis_eval;
  const double Vel = basis_eval(phi, U, ix, 1, iN);
  const double EmT = basis_eval(phi, U, ix, 2, iN);

  return EmT - (0.5 * Vel * Vel);
}

// cell average specific internal energy
template <class T>
KOKKOS_INLINE_FUNCTION auto compute_internal_energy(T U, const int i,
                                                    const int q) -> double {
  return U(i, q, 2) - (0.5 * U(i, q, 1) * U(i, q, 1));
}

// cell average specific internal energy
template <class T>
KOKKOS_INLINE_FUNCTION auto compute_internal_energy(T U, const int ix)
    -> double {
  return U(ix, 0, 2) - (0.5 * U(ix, 0, 1) * U(ix, 0, 1));
}

// string to_lower function
// adapted from
// http://notfaq.wordpress.com/2007/08/04/cc-convert-string-to-upperlower-case/
template <class T>
KOKKOS_INLINE_FUNCTION auto to_lower(T data) -> T {
  std::transform(data.begin(), data.end(), data.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return data;
}

} // namespace athelas::utilities
