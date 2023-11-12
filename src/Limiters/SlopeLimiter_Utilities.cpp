/**
 * File     :  SlopeLimiter_Utilities.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Hold utility functions for the slope limiter class
 *  to keep the class minimal.
 *
 * Contains:
 * ---------
 * sgn, minmod, minmodB
 **/

#include <iostream>
#include <limits>
#include <algorithm> // std::min, std::max
#include <cstdlib>   /* abs */

#include "Utilities.hpp"
#include "SlopeLimiter_Utilities.hpp"


/**
 *  Barth-Jespersen limiter
 *  Parameters:
 *  -----------
 *  U_v_*: left/right vertex values on target cell
 *  U_c_*: cell averages of target cell + neighbors
 *    [ left, target, right ]
 *  alpha: scaling coefficient for BJ limiter.
 *    alpha=1 is classical limiter, alpha=0 enforces constant solutions
 **/
Real BarthJespersen( Real U_v_L, Real U_v_R, Real U_c_L, Real U_c_T, Real U_c_R,
                     Real alpha )
{
  // Get U_min, U_max
  Real U_min_L = 10000000.0 * U_c_T;
  Real U_min_R = 10000000.0 * U_c_T;
  Real U_max_L = std::numeric_limits<Real>::epsilon( ) * U_c_T * 0.00001;
  Real U_max_R = std::numeric_limits<Real>::epsilon( ) * U_c_T * 0.00001;

  U_min_L = std::min( U_min_L, std::min( U_c_T, U_c_L ) );
  U_max_L = std::max( U_max_L, std::max( U_c_T, U_c_L ) );
  U_min_R = std::min( U_min_R, std::min( U_c_T, U_c_R ) );
  U_max_R = std::max( U_max_R, std::max( U_c_T, U_c_R ) );

  // loop of cell certices
  Real phi_L = 0.0;
  Real phi_R = 0.0;

  // left vertex
  if ( U_v_L - U_c_T + 1.0 > 1.0 )
  {
    phi_L = std::min( 1.0, alpha * ( U_max_L - U_c_T ) / ( U_v_L - U_c_T ) );
  }
  else if ( U_v_L - U_c_T + 1.0 < 1.0 )
  {
    phi_L = std::min( 1.0, alpha * ( U_min_L - U_c_T ) / ( U_v_L - U_c_T ) );
  }
  else
  {
    phi_L = 1.0;
  }

  // right vertex
  if ( U_v_R - U_c_T + 1.0 > 1.0 )
  {
    phi_R = std::min( 1.0, alpha * ( U_max_R - U_c_T ) / ( U_v_R - U_c_T ) );
  }
  else if ( U_v_R - U_c_T + 1.0 < 1.0 )
  {
    phi_R = std::min( 1.0, alpha * ( U_min_R - U_c_T ) / ( U_v_R - U_c_T ) );
  }
  else
  {
    phi_R = 1.0;
  }

  // return min of two values
  return std::min( phi_L, phi_R );
}
