/**
 * File     :  LinearAlgebraModules.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Linear algebra routines needed for quadrature,
 *  Calls LAPACK routine dstev
 **/

#include <iostream>
#include <vector>

#include <Kokkos_Core.hpp>

#include "Error.hpp"
#include "LinearAlgebraModules.hpp"
#include "lapacke.h"

/**
 * Use LAPACKE to diagonalize symmetric tridiagonal matrix with DSTEV
 *
 * Parameters:
 *
 *   n: matrix dimension
 *   d: diagonal array of matrix
 *   r: subdiagonal array (length n)
 *   array: product (Q*)z (in/output)
 **/
void Tri_Sym_Diag( int n, Real *d, Real *e, Real *array )
{

  // Parameters for LaPack
  lapack_int m, ldz, info, work_dim;
  m        = n;
  char job = 'V';
  ldz      = n;

  if ( n == 1 )
  {
    work_dim = 1;
  }
  else
  {
    work_dim = 2 * n - 2;
  }

  Real *ev   = new Real[n * n];
  Real *work = new Real[work_dim];

  info = LAPACKE_dstev( LAPACK_COL_MAJOR, job, m, d, e, ev, ldz );

  if ( info != 0 )
  {
    throw Error(
        " ! Issue occured in initializing quadrature in Tri_Sym_Diag." );
  }

  // Matrix multiply ev' * array. Only Array[0] is nonzero.
  Real k = array[0];
  for ( int i = 0; i < n; i++ )
  {
    array[i] = k * ev[n * i];
  }

  delete[] work;
  delete[] ev;
}

/**
 * Use LAPACKE to invert a matrix M using LU factorization.
 **/
void InvertMatrix( Real *M, UInt n )
{
  lapack_int info1, info2;

  int *IPIV = new int[n];

  info1 = LAPACKE_dgetrf( LAPACK_COL_MAJOR, n, n, M, n, IPIV );
  info2 = LAPACKE_dgetri( LAPACK_COL_MAJOR, n, M, n, IPIV );

  delete[] IPIV;

  if ( info1 != 0 || info2 != 0 )
  {
    throw Error( " ! Issue occured in matrix inversion." );
  }
}

