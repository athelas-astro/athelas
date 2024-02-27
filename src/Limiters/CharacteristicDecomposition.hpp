#ifndef _CHARACTERISTICDECOMPOSITION_HPP_
#define _CHARACTERISTICDECOMPOSITION_HPP_

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"

void ComputeCharacteristicDecomposition( Kokkos::View<Real[3]> U,
                                         Kokkos::View<Real[3][3]> R,
                                         Kokkos::View<Real[3][3]> R_inv );

#endif // _CHARACTERISTICDECOMPOSITION_HPP_
