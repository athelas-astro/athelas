/**
 * File     :  CharacteristicDecomposition.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Compute matrices for characteristic decomposition
 **/

#include <iostream>
#include <math.h> /* sqrt */

#include "CharacteristicDecomposition.hpp"
#include "Error.hpp"

void ComputeCharacteristicDecomposition( Kokkos::View<Real[3]> U,
                                         Kokkos::View<Real[3][3]> R,
                                         Kokkos::View<Real[3][3]> R_inv ) {

  const Real Tau  = U( 0 );
  const Real V    = U( 1 );
  const Real Em_T = U( 2 );

  // const Real P  = ComputePressureFromConserved_IDEAL( Tau, V, Em_T );
  // const Real Cs = ComputeSoundSpeedFromConserved_IDEAL( Tau, V, Em_T );

  const Real Em = Em_T - 0.5 * V * V;

  const float GAMMA = 1.4;

  const Real k      = std::sqrt( GAMMA * ( GAMMA - 1.0 ) );
  const Real sqrt_e = std::sqrt( Em );
  const Real InvTau = 1.0 / Tau;

  /* --- Thermodynamic Derivatives of Pressure --- */

  // const Real P_Tau = - (GAMMA - 1.0) * Em / ( Tau * Tau );
  // const Real P_Em  = + (GAMMA - 1.0) / Tau;
  // const Real P_T_E = - Em / Tau; // ratio of derivatives

  // Eigenvalues are rho * Cs...
  // const Real lam = std::sqrt( P * P_Em - P_Tau );

  /*  --- Compute Matrix Elements --- */

  for ( int i = 0; i < 3; i++ )
    R( 0, i ) = 1.0;

  R( 1, 0 ) = +sqrt_e * k * InvTau;
  R( 1, 1 ) = +0.0;
  R( 1, 2 ) = -sqrt_e * k * InvTau;

  R( 2, 0 ) = ( Em + sqrt_e * k * V - Em * GAMMA ) * InvTau;
  R( 2, 1 ) = ( Em * InvTau );
  R( 2, 2 ) = ( Em - sqrt_e * k * V - Em * GAMMA ) * InvTau;

  R_inv( 0, 0 ) = 0.5;
  R_inv( 0, 1 ) = Tau * ( k * V + sqrt_e * GAMMA ) / ( 2.0 * Em * k );
  R_inv( 0, 2 ) = -Tau / ( 2.0 * Em );

  R_inv( 1, 0 ) = GAMMA - 1.0;
  R_inv( 1, 1 ) = -V * Tau / Em;
  R_inv( 1, 2 ) = Tau / Em;

  R_inv( 2, 0 ) = 0.5;
  R_inv( 2, 1 ) = Tau * ( k * V - sqrt_e * GAMMA ) / ( 2.0 * Em * k );
  R_inv( 2, 2 ) = -Tau / ( 2.0 * Em );

  for ( int i = 0; i < 3; i++ )
    for ( int j = 0; j < 3; j++ ) {
      R_inv( i, j ) /= GAMMA;
    }
}
