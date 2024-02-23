/**
 * File     :  EoS_Ideal.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : ideal equation of state routines
 **/

#include <math.h> /* sqrt */

#include "Abstractions.hpp"
#include "Constants.hpp"
#include "PolynomialBasis.hpp"
#include "EoS.hpp"

void IdealGas::PressureFromConserved( const Real Tau, const Real V,
                                      const Real Bm,   const Real EmT,
                                      Real &P ) const {
  const Real Em = EmT - 0.5 * V * V - 0.5 * Bm * Bm / Tau;
  const Real Ev = Em / Tau;
  P = ( gamma - 1.0 ) * Ev;
}

void IdealGas::SoundSpeedFromConserved( const Real Tau, const Real V, 
                                        const Real Bm,   const Real EmT, Real &Cs ) const {
  const Real Em = EmT - 0.5 * V * V - 0.5 * Bm * Bm / Tau;
  Cs = sqrt( gamma * ( gamma - 1.0 ) * Em );
}

void IdealGas::TemperatureFromTauPressureAbar( const Real Tau, 
                                               const Real P, const Real Abar, 
                                               Real &T ) const {
  T = ( P * Abar * Tau ) / ( constants::N_A * constants::k_B ); 
}

void IdealGas::TemperatureFromTauPressure( const Real Tau, 
                                           const Real P, Real &T ) const {
  const Real Abar = 1.0; 
  TemperatureFromTauPressureAbar( Tau, P, Abar, T );
}

void IdealGas::RadiationPressure( const Real T, Real &Prad ) const {
    Prad = constants::a * T * T * T * T;
}

// nodal specific internal energy
Real IdealGas::ComputeInternalEnergy( const View3D U, ModalBasis *Basis,
                                      const UInt iX, const UInt iN ) const
{
  const Real Tau = Basis->BasisEval( U, iX, 0, iN, false );
  const Real Vel = Basis->BasisEval( U, iX, 1, iN, false );
  const Real EmT = Basis->BasisEval( U, iX, 2, iN, false );
  const Real Bm  = Basis->BasisEval( U, iX, 3, iN, false );

  return EmT - 0.5 * Vel * Vel - 0.5 * Bm * Bm / Tau;
}

// cell average specific internal energy
Real IdealGas::ComputeInternalEnergy( const View3D U, const UInt iX ) const
{
  return U( 2, iX, 0 ) - 0.5 * U( 1, iX, 0 ) * U( 1, iX, 0 )
                       - 0.5 * U( 3, iX, 0 ) * U( 3, iX, 0 ) / U( 0, iX, 0 );
}
