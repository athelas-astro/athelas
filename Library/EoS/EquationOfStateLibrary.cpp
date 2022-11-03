/**
 * File     :  EquationOfStateLibrary_IDEAL.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Ideal equation of state routines
 **/

#include <math.h> /* sqrt */

#include "PolynomialBasis.hpp"
#include "EquationOfStateLibrary.hpp"

// Compute pressure assuming an ideal gas
Real ComputePressureFromPrimitive_IDEAL( const Real Ev, const Real GAMMA )
{
  return ( GAMMA - 1.0 ) * Ev;
}

Real ComputePressureFromConserved_IDEAL( const Real Tau, const Real V,
                                         const Real Em_T, const Real GAMMA )
{
  Real Em = Em_T - 0.5 * V * V;
  Real Ev = Em / Tau;
  Real P  = ( GAMMA - 1.0 ) * Ev;

  return P;
}

Real ComputeSoundSpeedFromConserved_IDEAL( const Real Tau, const Real V,
                                           const Real Em_T, const Real GAMMA )
{
  Real Em = Em_T - 0.5 * V * V;

  Real Cs = sqrt( GAMMA * ( GAMMA - 1.0 ) * Em );
  //  / ( D + GAMMA * Ev ) )
  return Cs;
}

// nodal specific internal energy
Real ComputeInternalEnergy( const Kokkos::View<Real ***> U, ModalBasis *Basis,
                            const UInt iX, const UInt iN )
{
  Real Vel = Basis->BasisEval( U, iX, 1, iN, false );
  Real EmT = Basis->BasisEval( U, iX, 2, iN, false );

  return EmT - 0.5 * Vel * Vel;
}

// cell average specific internal energy
Real ComputeInternalEnergy( const Kokkos::View<Real ***> U, const UInt iX )
{
  return U( 2, iX, 0 ) - 0.5 * U( 1, iX, 0 ) * U( 1, iX, 0 );
}

Real ComputeEnthalpy( const Real Tau, const Real V, const Real Em_T, 
                      const Real GAMMA  )
{
//TODO
}
