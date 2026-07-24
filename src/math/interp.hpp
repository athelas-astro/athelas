#pragma once

#include "concepts/types.hpp"

#include "Kokkos_MathematicalFunctions.hpp"

namespace athelas::math::interp {

/**
 * @brief simple linear interpolation to x
 *
 * Uses fused multiply add (std::fma) to reduce rounding errors when available.
 */
KOKKOS_FUNCTION
template <typename T>
constexpr auto linterp(T x0, T x1, T y0, T y1, T x) noexcept -> T {
  if (x0 == x1) {
    return y0;
  }
  const T t = (x - x0) / (x1 - x0);
  return Kokkos::fma(y1 - y0, t, y0);
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

/**
 * @brief Degree-(count-1) Lagrange interpolation.
 *
 * Value at `target` of the polynomial through values[j0 + a] sampled at
 * nodes[j0 + a], for a in [0, count) (e.g. count = 4 for a cubic).  The caller
 * keeps [j0, j0 + count) in range and the nodes distinct.
 */
KOKKOS_FUNCTION
template <VectorLike T>
auto lagrange_interp(const T &nodes, const T &values, const int j0,
                     const int count, const double target) -> double {
  double y = 0.0;
  for (int a = 0; a < count; ++a) {
    double term = values[j0 + a];
    for (int b = 0; b < count; ++b) {
      if (b != a) {
        term *= (target - nodes[j0 + b]) / (nodes[j0 + a] - nodes[j0 + b]);
      }
    }
    y += term;
  }
  return y;
}
} // namespace athelas::math::interp
