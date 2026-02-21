/**
 * @file opac_base.hpp
 * --------------
 *
 * @brief Base class for opacity models.
 *
 * @details Defines the OpacBase template class.
 *
 *          The class provides two interface methods:
 *          - planck_mean
 *          - rosseland_mean
 *
 *          The interface methods take density, temperature, and composition
 *          parameters to compute the appropriate mean opacity values.
 */

#pragma once

#include "Kokkos_Macros.hpp"
namespace athelas {

struct FloorValues {
  double rosseland;
  double planck;
};

template <class OPAC>
class OpacBase {
 public:
  KOKKOS_INLINE_FUNCTION auto planck_mean(const double rho, const double T,
                                          const double X, const double Z,
                                          double *lambda) const -> double {
    return static_cast<OPAC const *>(this)->planck_mean(rho, T, X, Z, lambda);
  }

  KOKKOS_INLINE_FUNCTION auto rosseland_mean(const double rho, const double T,
                                             const double X, const double Z,
                                             double *lambda) const -> double {
    return static_cast<OPAC const *>(this)->rosseland_mean(rho, T, X, Z,
                                                           lambda);
  }
};

} // namespace athelas
