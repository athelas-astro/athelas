#pragma once

#include <algorithm>
#include <cctype>

#include "Kokkos_Macros.hpp"

#include "basis/nodal_basis.hpp"
#include "basis/polynomial_basis.hpp"
#include "kokkos_types.hpp"

namespace athelas::utilities {
using basis::NodalBasis;

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
