#ifndef EQUATIONOFSTATELIBRARY_H
#define EQUATIONOFSTATELIBRARY_H

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"
#include "PolynomialBasis.hpp"

Real ComputePressureFromConserved_IDEAL( const Real U0, const Real U1,
                                         const Real U2 );
Real ComputePressureFromPrimitive_IDEAL( const Real Tau, const Real V,
                                         const Real Em );
Real ComputeSoundSpeedFromPrimitive_IDEAL( const Real Tau, const Real V,
                                           const Real Em );
Real ComputeInternalEnergy( const Kokkos::View<Real ***> U, ModalBasis *Basis,
                            const UInt iX, const UInt iN );
Real ComputeInternalEnergy( const Kokkos::View<Real ***> U, const UInt iX );
Real ComputeEnthalpy( const Real Tau, const Real V, const Real Em_T );

#endif
