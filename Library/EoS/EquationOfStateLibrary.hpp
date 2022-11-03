#ifndef EQUATIONOFSTATELIBRARY_H
#define EQUATIONOFSTATELIBRARY_H

#include "Abstractions.hpp"

Real ComputePressureFromPrimitive_IDEAL( const Real Ev,
                                         const Real GAMMA = 1.4 );
Real ComputePressureFromConserved_IDEAL( const Real Tau, const Real V,
                                         const Real Em_T,
                                         const Real GAMMA = 1.4 );
Real ComputeSoundSpeedFromConserved_IDEAL( const Real Tau, const Real V,
                                           const Real Em_T,
                                           const Real GAMMA = 1.4 );
Real ComputeInternalEnergy( const Kokkos::View<Real ***> U, ModalBasis *Basis,
                            const UInt iX, const UInt iN );
Real ComputeInternalEnergy( const Kokkos::View<Real ***> U, const UInt iX );
Real ComputeEnthalpy( const Real Tau, const Real V, const Real Em_T, 
                      const Real GAMMA = 1.4  );

#endif
