#ifndef UTILITIES_HPP_
#define UTILITIES_HPP_

#include <algorithm>
#include <cctype>
#include <string>

#include "abstractions.hpp"
#include "grid.hpp"
#include "polynomial_basis.hpp"

namespace utilities {

// Implements a typesafe sgn function
// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename T>
constexpr int sgn( T val ) {
  return ( T( 0 ) < val ) - ( val < T( 0 ) );
}

// nodal specific internal energy
template <class T>
Real ComputeInternalEnergy( T U, const ModalBasis *Basis, const int iX,
                            const int iN ) {
  const Real Vel = Basis->BasisEval( U, iX, 1, iN );
  const Real EmT = Basis->BasisEval( U, iX, 2, iN );

  return EmT - 0.5 * Vel * Vel;
}

// cell average specific internal energy
template <class T>
Real ComputeInternalEnergy( T U, const int iX ) {
  return U( 2, iX, 0 ) - 0.5 * U( 1, iX, 0 ) * U( 1, iX, 0 );
}

// string to_lower function
// adapted from
// http://notfaq.wordpress.com/2007/08/04/cc-convert-string-to-upperlower-case/
template <class T>
T to_lower( T data ) {
  std::transform( data.begin( ), data.end( ), data.begin( ),
                  []( unsigned char c ) { return std::tolower( c ); } );
  return data;
}

} // namespace utilities

#endif // UTILITIES_HPP_
