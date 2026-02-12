/**
 * @file slope_limiter_base.hpp
 * --------------
 *
 * @brief Base class for slope limiters.
 *
 * @details Defines the SlopeLimiterBase template class that serves
 *          as the foundation for all slope limiters implemented.
 *
 *          The class provides two key interface methods:
 *          - apply_slope_limiter: Applies the limiter to the solution
 *          - get_limited: Returns whether a cell was limited
 */

#pragma once

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"

namespace athelas {

template <class SlopeLimiter>
class SlopeLimiterBase {
 public:
  void apply_slope_limiter(AthelasArray3D<double> U, const GridStructure *grid,
                           const basis::NodalBasis *basis, const eos::EOS *eos,
                           const std::vector<int> &vars) const {
    return static_cast<SlopeLimiter const *>(this)->apply_slope_limiter(
        U, grid, basis, eos, vars);
  }
  [[nodiscard]] auto get_limited(const int ix) const -> int {
    return static_cast<SlopeLimiter const *>(this)->get_limited(ix);
  }
  [[nodiscard]] auto limited() const -> AthelasArray1D<int> {
    return static_cast<SlopeLimiter const *>(this)->limited();
  }
};

} // namespace athelas
