/**
 * File     :  QuadratureLibrary.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Functions necessary for computing quadrature rules
 **/

#include <iostream>
#include <math.h>

#include "Constants.hpp"
#include "LinearAlgebraModules.hpp"
#include "QuadratureLibrary.hpp"

namespace quadrature {
/**
 * Gauss-Legendre Quadrature
 **/

/**
 * Jacobi matrix for Legendre-Gauss quadrature rule
 *
 * Parameters:
 *
 *   int m      : number of quadrature nodes
 *   Real* aj : matrix diagonal    (output)
 *   Real* bj : matrix subdiagonal (output)
 *   Real z   : zero-th moment     (output)
 */
Real Jacobi_Matrix( int m, Real *aj, Real *bj ) {

  Real ab;
  Real zemu;
  Real abi;
  Real abj;

  ab   = 0.0;
  zemu = 2.0 / ( ab + 1.0 );
  for ( int i = 0; i < m; i++ ) {
    aj[i] = 0.0;
  }

  for ( int i = 1; i <= m; i++ ) {
    abi       = i + ab * ( i % 2 );
    abj       = 2 * i + ab;
    bj[i - 1] = sqrt( abi * abi / ( abj * abj - 1.0 ) );
  }

  return zemu;
}

/**
 * Compute Legendre-Gauss Quadrature
 **/
void LG_Quadrature( int m, Real *nodes, Real *weights ) {
  Real *aj = new Real[m];
  Real *bj = new Real[m];

  Real zemu;

  //  Get the Jacobi matrix and zero-th moment.
  zemu = Jacobi_Matrix( m, aj, bj );

  // Nodes and Weights
  for ( int i = 0; i < m; i++ ) {
    nodes[i] = aj[i];
  }

  weights[0] = sqrt( zemu );
  for ( int i = 1; i < m; i++ ) {
    weights[i] = 0.0;
  }

  // --- Diagonalize the Jacobi matrix. ---
  Tri_Sym_Diag( m, nodes, bj, weights ); // imtqlx

  for ( int i = 0; i < m; i++ ) {
    weights[i] = weights[i] * weights[i];

    // Shift to interval [-0.5, 0.5]
    weights[i] *= 0.5;
    nodes[i] *= 0.5;
  }

  delete[] aj;
  delete[] bj;
}

} // namespace quadrature
