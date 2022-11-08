/**
 * File     :  Can2Prim.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Conserved to Primitive recovery
 **/

#include <stdlib.h>     /* abs */

#include "Con2Prim.hpp"
#include "EquationOfStateLibrary.hpp"
#include "Error.hpp"
#include "FluidUtilities.hpp"

#define GAMMA 1.4

// TODO: ideal EoS hardcoded.
void Con2Prim( Real U0, Real U1, Real U2, Real &tau, Real &v, Real &Em, 
               Real &P )
{
  const Real a = GAMMA / ( GAMMA - 1.0 );
  const Real w = C2P_Newton( 0.1, a, U1, U2 + 0.0 );

  v = 2.0 * w / ( 1.0 + w * w );
  const Real W = LorentzFactor( v );
  Em = W * ( U2 - W + 1.0 ) / ( GAMMA * ( W * W - 1.0 ) + 1.0 );

  tau = U0 * W;

  P = ( GAMMA - 1.0 ) * Em / tau;
}

Real C2P_Newton( Real x0, const Real a, const Real xi, const Real eta )
{
  Real tol = 1e-12;
  Real eps = 1-10;

  UInt MAXITERS = 20;

  // define target polynomial
  auto Target = []( Real w, Real a, Real xi, Real eta ) {
    return (a-1.0) * xi * w*w*w*w - 2.0 * ( a * eta + 1.0 ) * w*w*w + 2.0 * ( a + 1.0 )*xi * w * w 
            - 2.0 * ( a * eta - 1.0 ) * w + ( a - 1.0 ) * xi;
  };

  // derivative
  auto dTarget = []( Real w, Real a, Real xi, Real eta ) {
    return 4.0 * (a-1.0) * xi * w*w*w - 6.0 * ( a * eta + 1.0 ) * w*w + 4.0 * ( a + 1.0 )*xi * w 
            - 2.0 * ( a * eta - 1.0 );
  };

  Real x1 = 0.0;
  UInt n = 0;
  while ( n <= MAXITERS )
  {
    const Real yprime = dTarget( x0, a, xi, eta );
    if ( std::abs( yprime ) < eps )
    {
      std::printf("Derivative approaching 0 in Con2Prim. Returning at iteration %d\n", n);
      return x0;
    }
    
    const Real y = Target( x0, a, xi, eta );
    x1 = x0 - y / yprime;

    if ( std::abs( x1 - x0 ) < tol )
    {
      return x1;
    }
    x0 = x1;
    n += 1;
  }
  
  throw Error(" ! Max Iters reached in Con2Prim!\n");
}

