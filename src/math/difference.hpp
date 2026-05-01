#pragma once

#include <functional>

#include "basic_types.hpp"

namespace athelas::math::difference {

/**
 * @brief Finite difference derivative.
 */
KOKKOS_FUNCTION
template <DiffScheme Scheme = DiffScheme::Forward, typename F, typename... Args>
constexpr auto finite_difference(double h, F &&f, double x, Args &&...args) {
  if constexpr (Scheme == DiffScheme::Forward) {
    return (std::invoke(std::forward<F>(f), x + h,
                        std::forward<Args>(args)...) -
            std::invoke(std::forward<F>(f), x, std::forward<Args>(args)...)) /
           h;
  } else if constexpr (Scheme == DiffScheme::Backward) {
    return (std::invoke(std::forward<F>(f), x, std::forward<Args>(args)...) -
            std::invoke(std::forward<F>(f), x - h,
                        std::forward<Args>(args)...)) /
           h;
  } else if constexpr (Scheme == DiffScheme::Central) {
    return (std::invoke(std::forward<F>(f), x + h,
                        std::forward<Args>(args)...) -
            std::invoke(std::forward<F>(f), x - h,
                        std::forward<Args>(args)...)) *
           (0.5 / h);
  }
}

} // namespace athelas::math::difference
