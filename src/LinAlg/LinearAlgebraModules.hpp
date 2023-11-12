#ifndef _LINEARALGEBRAMODULES_HPP_
#define _LINEARALGEBRAMODULES_HPP_

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"

// Fill identity matrix
template < class T >
constexpr void IdentityMatrix( T Mat, UInt n )
{
  for ( UInt i = 0; i < n; i++ )
    for ( UInt j = 0; j < n; j++ )
    {
      if ( i == j )
      {
        Mat( i, j ) = 1.0;
      }
      else
      {
        Mat( i, j ) = 0.0;
      }
    }
}

/**
 * Matrix vector multiplication
 **/
template < class M, class V >
constexpr void MatMul( Real alpha, M A, V x, Real beta, V y )
{
  // Calculate A*x=y
  for ( int i = 0; i < 3; i++ )
  {
    for ( int j = 0; j < 3; j++ )
    {
      y( i ) += ( A( i, j ) * x( j ) );
    }
  }
}
void Tri_Sym_Diag( int n, Real *d, Real *e, Real *array );
void InvertMatrix( Real *M, UInt n );

#endif // _LINEARALGEBRAMODULES_HPP_
