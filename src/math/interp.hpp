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
constexpr auto LINTERP(T x0, T x1, T y0, T y1, T x) noexcept -> T {
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
} // namespace athelas::math::interp
