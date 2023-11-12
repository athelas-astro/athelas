/**
 * File     :  FluidUtilities.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Utility routines for fluid fields. Includes Riemann solvers.
 **/

#include <iostream>
#include <vector>
#include <cstdlib>   /* abs */
#include <algorithm> // std::min, std::max
#include <math.h>    /* sqrt */

#include "Constants.hpp"
#include "Con2Prim.hpp"
#include "EoS.hpp"
#include "Error.hpp"
#include "Grid.hpp"
#include "PolynomialBasis.hpp"
#include "FluidUtilities.hpp"

/**
 * Compute the primitive quantities (density, momemtum, energy density)
 * from conserved quantities. Primitive quantities are stored at Gauss-Legendre
 * nodes.
 **/
void ComputePrimitiveFromConserved( View3D uCF,
                                    View3D uPF,
                                    ModalBasis *Basis, GridStructure *Grid )
{
  const UInt nNodes = Grid->Get_nNodes( );
  const UInt ilo    = Grid->Get_ilo( );
  const UInt ihi    = Grid->Get_ihi( );

  Real Tau = 0.0;
  Real Vel = 0.0;
  Real EmT = 0.0;

  for ( UInt iX = ilo; iX <= ihi; iX++ )
    for ( UInt iN = 0; iN < nNodes; iN++ )
    {
      // Density
      Tau              = Basis->BasisEval( uCF, 0, iX, iN + 1, false );
      uPF( 0, iX, iN ) = 1.0 / Tau;

      // Momentum
      Vel              = Basis->BasisEval( uCF, 1, iX, iN + 1, false );
      uPF( 1, iX, iN ) = uPF( 0, iX, iN ) * Vel;

      // Specific Total Energy
      EmT              = Basis->BasisEval( uCF, 2, iX, iN + 1, false );
      uPF( 2, iX, iN ) = EmT / Tau;
    }
}

/**
 * Return a component iCF of the flux vector.
 * TODO: Flux_Fluid needs streamlining
 **/
Real Flux_Fluid( const Real V, const Real P, const UInt iCF )
{
  if ( iCF == 0 )
  {
    return -V;
  }
  else if ( iCF == 1 )
  {
    return +P;
  }
  else if ( iCF == 2 )
  {
    return +P * V;
  }
  else
  { // Error case. Shouldn't ever trigger.
    throw Error( " ! Please input a valid iCF! (0,1,2). " );
    return -1.0; // just a formality.
  }
}

/**
 * Fluid radiation sources *
 * TODO: update for eos->
 **/
Real Source_Fluid_Rad( Real D, Real V, Real T, Real X, Real kappa,
                       Real E, Real F, Real Pr, UInt iCF ) {
  assert ( iCF == 0 || iCF == 1 || iCF == 2 );

  Real a = constants::a;
  Real c = constants::c_cgs;

  Real b = V / c;
  Real term1 = E - a * T*T*T*T - 2.0 * b * F;
  Real term2 = F - E * b - b * Pr;

  if ( iCF == 0 ) {
    return 0.0;
  } else if ( iCF == 1 ){
    return D * kappa * term1 * b + D * X * term2;
  } else {
    return c * ( D * kappa * term1 + D * X * b * term2 );
  }
}


/**
 * Gudonov style numerical flux. Constucts v* and p* states.
 **/
void NumericalFlux_Gudonov( const Real vL, const Real vR, const Real pL,
                            const Real pR, const Real zL, const Real zR,
                            Real &Flux_U, Real &Flux_P )
{
  const Real W_L = LorentzFactor( vL ); 
  const Real W_R = LorentzFactor( vR ); 
  const Real sigmaL = zL * zL / ( W_L * W_L * ( 1.0 - zL * zL ) );
  const Real sigmaR = zR * zR / ( W_R * W_R * ( 1.0 - zR * zR ) );
  const Real aL = sqrt( sigmaL * ( 1.0 - sigmaL ) ) / ( 1.0 + sigmaL );
  const Real aR = sqrt( sigmaR * ( 1.0 - sigmaR ) ) / ( 1.0 + sigmaR );
  Flux_U = ( pL - pR + zR * vR + zL * vL ) / ( zR + zL );
  Flux_P = ( zR * pL + zL * pR + zL * zR * ( vL - vR ) ) / ( zR + zL );
}


/**
// Simple, Yamada 1997 10.1086/303548
void NumericalFlux_Simple( const Real rhoL, const Real rhoR, const Real vL, 
                           const Real vR, const Real emL, const Real emR, 
                           const Real pL, const Real pR, 
                           const Real cL, const Real cR, 
                           Real &Flux_U, Real &Flux_P )
{
  const Real Gm = 1.4;
  const Real em = 0.5 * ( emL + emR );
  const Real rhom = 0.5 * ( rhoL + rhoR );
  const Real taum = 1.0 / rhom;
  const Real vm = 0.5 * ( vL + vR );
  const Real Pm = ComputePressureFromPrimitive_IDEAL( taum, vm, em );
  const Real Cm = ComputeSoundSpeedFromPrimitive_IDEAL( taum, vm, em, Pm );

  Flux_U = 0.5 * ( vR + vL ) - Cm * ( pR - pL ) / ( 2.0 * Gm * Pm );
  Flux_P = 0.5 * ( pR + pL ) - Gm * Pm * ( vR - vL ) / ( 2.0 * Cm );
}
88/


/**
 * Gudonov style numerical flux. Constucts v* and p* states.
 **/
void NumericalFlux_HLLC( Real vL, Real vR, Real pL, Real pR, Real cL, Real cR,
                         Real rhoL, Real rhoR, Real &Flux_U, Real &Flux_P )
{
  const Real W_L = LorentzFactor( vL ); 
  const Real W_R = LorentzFactor( vR ); 
  const Real sigmaL = cL * cL / ( W_L * W_L * ( 1.0 - cL * cL ) );
  const Real sigmaR = cR * cR / ( W_R * W_R * ( 1.0 - cR * cR ) );
  const Real aL = ( vL - sqrt( sigmaL * ( 1.0 - vL + sigmaL ) ) ) / ( 1.0 + sigmaL );
  const Real aR = ( vR - sqrt( sigmaR * ( 1.0 - vR + sigmaR ) ) ) / ( 1.0 + sigmaR );
  Flux_U  = ( rhoR * vR * ( aR - vR ) - rhoL * vL * ( aL - vL ) + pL - pR ) /
           ( rhoR * ( aR - vR ) - rhoL * ( aL - vL ) );
  Flux_P = rhoL * ( vL - aL ) * ( vL - Flux_U ) + pL;
}

// Compute Auxilliary

/**
 * Compute the fluid timestep.
 **/
Real ComputeTimestep_Fluid( const View3D U,
                            const GridStructure *Grid, EOS *eos, 
                            const Real CFL )
{

  const Real MIN_DT = 0.000000005;
  const Real MAX_DT = 0.1;

  const UInt &ilo = Grid->Get_ilo( );
  const UInt &ihi = Grid->Get_ihi( );

  Real dt = 0.0;
  Kokkos::parallel_reduce(
      "Compute Timestep", Kokkos::RangePolicy<>( ilo, ihi + 1 ),
      KOKKOS_LAMBDA( const int &iX, Real &lmin ) {
        // --- Compute Cell Averages ---
        //TODO: Don't need this con2prim... do it elsewhere once and pass in uPF
        Real tau, v, em, p;
        Con2Prim( U( 0, iX, 0 ), U(1, iX, 0), U(2, iX, 0), tau, v, em, p );

        Real dr = Grid->Get_Widths( iX );

        Real Cs = 0.0;
        eos->SoundSpeedFromConserved( tau, v, em, Cs );
        Real eigval = Cs;

        Real dt_old = std::abs( dr ) / std::abs( eigval );

        if ( dt_old < lmin ) lmin = dt_old;
      },
      Kokkos::Min<Real>( dt ) );

  dt = std::max( CFL * dt, MIN_DT );
  dt = std::min( dt, MAX_DT );

  // Triggers on NaN
  if ( dt != dt )
  {
    throw Error( " ! nan encountered in ComputeTimestep.\n" );
  }

  return dt;
}

Real LorentzFactor( const Real V )
{
  return 1.0 / sqrt( 1.0 - V*V );
}
