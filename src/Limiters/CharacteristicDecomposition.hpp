#ifndef CHARACTERISTICDECOMPOSITION_H
#define CHARACTERISTICDECOMPOSITION_H

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"

void ComputeCharacteristicDecomposition( Kokkos::View<Real[3]> U,
                                         Kokkos::View<Real[3][3]> R,
                                         Kokkos::View<Real[3][3]> R_inv );

#endif
