/**
 * File     :  RadUtilities.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Utility routines for radiation fields. Includes Riemann solvers.
 **/

#include <iostream>
#include <vector>

#include "Kokkos_Core.hpp"

#include "EoS.hpp"
#include "Constants.hpp"
#include "Error.hpp"
#include "PolynomialBasis.hpp"
#include "RadUtilities.hpp"
#include "Utilities.hpp"

Real Flux_Rad( Real E, Real F, Real P, Real V, UInt iRF ) {
  assert ( iRF == 0 || iRF == 1 );

  if ( iRF == 0 ) {
    return F - E * V;
  } else {
    return P - F * V;
  }
}

Real Source_Rad( Real D, Real V, Real T, Real X, Real kappa, 
                 Real E, Real F, Real Pr, UInt iRF ) {
  assert ( iRF == 0 || iRF == 1 );

  Real a = constants::a;
  Real c = constants::c_cgs;

  Real b = V / c;
  Real term1 = E - a * T*T*T*T - 2.0 * b * F;
  Real term2 = F - E * b - b * Pr;

  if ( iRF == 0 ) {
    return - ( D * kappa * term1 + D * X * b * term2 );
  } else {
    return - ( D * kappa * term1 * b + D * X * term2 );
  }
}

/**
 * Emissivity chi
 * TODO: actually implement this
 **/
Real ComputeEmissivity( const Real D, const Real V, const Real Em ) {
  return 1.0;
}

/**
 * Opacity kappa
 * TODO: actually implement this
 **/
Real ComputeOpacity( const Real D, const Real V, const Real Em ) {
  return 1.0;
}

/* pressure tensor closure */
// TODO: check Closure
Real ComputeClosure( const Real E, const Real F ) {
  const Real f = F / E;
  const Real chi = ( 3.0 + 4.0 * f * f ) 
    / ( 5.0 + 2.0 * std::sqrt( 4.0 - 3.0 * f * f ) );
  const Real T = ( 1.0 - chi ) / 2.0 + ( 3.0 * chi - 1.0) * sgn( F ) / 2.0; // TODO: Is this right?
  return E * T;
}
