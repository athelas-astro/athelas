/**
 * File     :  EquationOfStateLibrary_IDEAL.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Ideal equation of state routines
 **/

#include <math.h> /* sqrt */

#include "Con2Prim.hpp"
#include "PolynomialBasis.hpp"
#include "EquationOfStateLibrary.hpp"

#define GAMMA 1.4

Real ComputePressureFromConserved_IDEAL( const Real U0, const Real U1,
                                         const Real U2 )
{
  Real P, v, em, tau;
  Con2Prim( U0, U1, U2, tau, v, em, P );
  return P;
}

Real ComputePressureFromPrimitive_IDEAL( const Real Tau, const Real V,
                                         const Real Em )
{
  Real Ev = Em / Tau;
  Real P  = ( GAMMA - 1.0 ) * Ev;

  return P;
}

Real ComputeSoundSpeedFromPrimitive_IDEAL( const Real Tau, const Real V,
                                           const Real Em, const Real P )
{
  const Real h = ComputeEnthalpy( Tau, V, Em + 0.5 * V * V );
  //const Real P = ComputePressureFromPrimitive_IDEAL( Tau, V, Em );

  Real Cs = sqrt( GAMMA * P * Tau / h );
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

//TODO This will need to be extended later. 
// Works for initialization when initializing the Newtonian variables.
Real ComputeEnthalpy( const Real Tau, const Real V, const Real Em_T  )
{
  Real Em = Em_T - 0.5 * V * V;
  Real P  = ComputePressureFromPrimitive_IDEAL( Tau, V, Em );

  return 1.0 + Em + P * Tau;
}
