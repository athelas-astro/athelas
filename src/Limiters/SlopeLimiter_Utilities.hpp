#ifndef _SLOPELIMITER_UTILITIES_HPP_
#define _SLOPELIMITER_UTILITIES_HPP_

#include "Abstractions.hpp"

// Standard minmod function
template <typename T>
constexpr Real minmod( T a, T b, T c ) {
  if ( sgn( a ) == sgn( b ) && sgn( b ) == sgn( c ) ) {
    return sgn( a ) * std::min( std::min( a, b ), c );
  } else {
    return 0.0;
  }
}

// TVB minmod function
template <typename T>
constexpr Real minmodB( T a, T b, T c, T dx, T M ) {
  if ( std::abs( a ) < M * dx * dx ) {
    return a;
  } else {
    return minmod( a, b, c );
  }
}

Real BarthJespersen( Real U_v_L, Real U_v_R, Real U_c_L, Real U_c_T, Real U_c_R,
                     Real alpha );
#endif // _SLOPELIMITER_UTILITIES_HPP_
