#pragma once

#include <cmath>
#include <limits>

#include "concepts/arithmetic.hpp"

#include "Kokkos_Macros.hpp"

namespace athelas::math::utils {

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

// [[x]]_+ = -.5 * (x + |x|) is positive part of x
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto pos_part(const double x) noexcept
    -> double {
  return 0.5 * (x + std::abs(x));
}

template <typename T = double>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto eps() {
  return 10 * std::numeric_limits<T>::epsilon();
}

template <typename T = double>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto large() {
  return 0.1 * std::numeric_limits<T>::max();
}

template <typename T = double>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto small() {
  return 10 * std::numeric_limits<T>::min();
}

// Implements a typesafe sgn function
// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename T>
KOKKOS_INLINE_FUNCTION constexpr auto sgn(T val) -> int {
  return (T(0) < val) - (val < T(0));
}

template <typename A, typename B>
KOKKOS_INLINE_FUNCTION auto ratio(const A &a, const B &b) {
  const B sgn = b >= 0 ? 1 : -1;
  return a / (b + sgn * small<B>());
}
} // namespace athelas::math::utils
