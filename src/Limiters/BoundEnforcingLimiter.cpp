/**
 * File     :  BoundEnforcingLimiter.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Limit solution to maintain physicality
 * TODO: Need to give BEL much more thought
 * TODO: Can some functions here be simplified?
 *   ? If I pass U, iX, iCF, iN... why not just the value
 **/

#include <iostream>
#include <algorithm> // std::min, std::max
#include <cstdlib>   /* abs */

#include "Kokkos_Core.hpp"

#include "EoS.hpp"
#include "Error.hpp"
#include "Utilities.hpp"
#include "PolynomialBasis.hpp"
#include "BoundEnforcingLimiter.hpp"

void LimitDensity( Kokkos::View<Real ***> U, ModalBasis *Basis )
{
  const Real EPSILON = 1.0e-13; // maybe make this smarter
  const UInt order   = Basis->Get_Order( );

  if ( order == 1 ) return;

  Kokkos::parallel_for(
      "Limit Density", Kokkos::RangePolicy<>( 1, U.extent( 1 ) - 1 ),
      KOKKOS_LAMBDA( UInt iX ) {
        Real theta1 = 100000.0;
        Real nodal  = 0.0;
        Real frac   = 0.0;
        Real avg    = U( 0, iX, 0 );

        for ( UInt iN = 0; iN <= order; iN++ )
        {
          nodal  = Basis->BasisEval( U, iX, 0, iN, false );
          frac   = std::abs( ( avg - EPSILON ) / ( avg - nodal ) );
          theta1 = std::min( theta1, std::min( 1.0, frac ) );
        }

        for ( UInt k = 1; k < order; k++ )
        {
          U( 0, iX, k ) *= theta1;
        }
      } );
}

void LimitInternalEnergy( Kokkos::View<Real ***> U, ModalBasis *Basis, 
                          EOS *eos )
{
  const UInt order = Basis->Get_Order( );

  if ( order == 1 ) return;

  Kokkos::parallel_for(
      "Limit Internal Energy", Kokkos::RangePolicy<>( 1, U.extent( 1 ) - 1 ),
      KOKKOS_LAMBDA( UInt iX ) {
        Real theta2 = 10000000.0;
        Real nodal  = 0.0;
        Real temp   = 0.0;

        for ( UInt iN = 0; iN <= order + 1; iN++ )
        {
          nodal = eos->ComputeInternalEnergy( U, Basis, iX, iN );

          if ( nodal >= 0.0 )
          {
            temp = 1.0;
          }
          else
          {
            // TODO: Backtracing may be working okay...
            temp = Backtrace( U, Basis, eos, iX, iN );
            // TODO: This is hacked and Does Not Really Work
            // temp = Bisection( U, Basis, iX, iN ) / 2.0;
          }
          theta2 = std::min( theta2, temp );
        }

        for ( UInt k = 1; k < order; k++ )
        {
          U( 0, iX, k ) *= theta2;
          U( 1, iX, k ) *= theta2;
          U( 2, iX, k ) *= theta2;
        }
      } );
}

void ApplyBoundEnforcingLimiter( Kokkos::View<Real ***> U, ModalBasis *Basis, 
                                 EOS *eos )

{
  LimitDensity( U, Basis );
  LimitInternalEnergy( U, Basis, eos );
}

/* --- Utility Functions --- */

// ( 1 - theta ) U_bar + theta U_q
Real ComputeThetaState( const Kokkos::View<Real ***> U, ModalBasis *Basis,
                        const Real theta, const UInt iCF, const UInt iX,
                        const UInt iN )
{
  Real result = Basis->BasisEval( U, iX, iCF, iN, false );
  result -= U( iCF, iX, 0 );
  result *= theta;
  result += U( iCF, iX, 0 );
  return result;
}

Real TargetFunc( const Kokkos::View<Real ***> U, ModalBasis *Basis, EOS *eos,
                 const Real theta, const UInt iX, const UInt iN )
{
  const Real w  = std::min( 1.0e-13, eos->ComputeInternalEnergy( U, iX ) );
  const Real s1 = ComputeThetaState( U, Basis, theta, 1, iX, iN );
  const Real s2 = ComputeThetaState( U, Basis, theta, 2, iX, iN );

  Real e = s2 - 0.5 * s1 * s1;

  return e - w;
}

Real Bisection( const Kokkos::View<Real ***> U, ModalBasis *Basis, EOS *eos,
                const UInt iX, const UInt iN )
{
  const Real TOL       = 1e-10;
  const UInt MAX_ITERS = 100;

  // bisection bounds on theta
  Real a = 0.0;
  Real b = 1.0;
  Real c = 0.5;

  Real fa = 0.0; // f(a) etc
  Real fc = 0.0;

  UInt n = 0;
  while ( n <= MAX_ITERS )
  {
    c = ( a + b ) / 2.0;

    fa = TargetFunc( U, Basis, eos, a, iX, iN );
    fc = TargetFunc( U, Basis, eos, c, iX, iN );

    if ( std::abs( fc <= TOL / 10.0 ) || ( b - a ) / 2.0 < TOL )
    {
      return c;
    }

    // new interval
    if ( sgn( fc ) == sgn( fa ) )
    {
      a = c;
    }
    else
    {
      b = c;
    }

    n++;
  }

  std::printf( "Max Iters Reach In Bisection\n" );
  return c;
}

Real Backtrace( const View3D U, ModalBasis *Basis, EOS *eos,
                const UInt iX, const UInt iN )
{
  Real theta = 1.0;
  Real nodal = -1.0;

  while ( theta >= 0.01 && nodal < 0.0 )
  {
    nodal = TargetFunc( U, Basis, eos, theta, iX, iN );

    theta -= 0.05;
  }

  return theta;
}
