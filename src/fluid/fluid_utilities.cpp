/**
 * File     :  fluid_utilities.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Utility routines for fluid fields. Includes Riemann solvers.
 **/

#include <algorithm> // std::min, std::max
#include <cstdlib> /* abs */
#include <iostream>
#include <vector>

#include "constants.hpp"
#include "eos.hpp"
#include "error.hpp"
#include "fluid_utilities.hpp"
#include "grid.hpp"
#include "polynomial_basis.hpp"
#include "rad_utilities.hpp"

/**
 * Compute the primitive quantities (density, momemtum, energy density)
 * from conserved quantities. Primitive quantities are stored at Gauss-Legendre
 * nodes.
 **/
void ComputePrimitiveFromConserved( View3D<Real> uCF, View3D<Real> uPF,
                                    ModalBasis *Basis, GridStructure *Grid ) {
  const int nNodes = Grid->Get_nNodes( );
  const int ilo    = Grid->Get_ilo( );
  const int ihi    = Grid->Get_ihi( );

  Real Tau = 0.0;
  Real Vel = 0.0;
  Real EmT = 0.0;

  for ( int iX = ilo; iX <= ihi; iX++ )
    for ( int iN = 0; iN < nNodes; iN++ ) {
      // Density
      Tau              = Basis->BasisEval( uCF, 0, iX, iN + 1 );
      uPF( 0, iX, iN ) = 1.0 / Tau;

      // Momentum
      Vel              = Basis->BasisEval( uCF, 1, iX, iN + 1 );
      uPF( 1, iX, iN ) = uPF( 0, iX, iN ) * Vel;

      // Specific Total Energy
      EmT              = Basis->BasisEval( uCF, 2, iX, iN + 1 );
      uPF( 2, iX, iN ) = EmT / Tau;
    }
}

/**
 * Return a component iCF of the flux vector.
 * TODO: Flux_Fluid needs streamlining
 **/
Real Flux_Fluid( const Real V, const Real P, const int iCF ) {
  assert( iCF == 0 || iCF == 1 || iCF == 2 );
  assert( P > 0.0 && "Flux_Flux :: negative pressure" );
  if ( iCF == 0 ) {
    return -V;
  } else if ( iCF == 1 ) {
    return +P;
  } else if ( iCF == 2 ) {
    return +P * V;
  } else { // Error case. Shouldn't ever trigger.
    throw Error( " ! Please input a valid iCF! (0,1,2). " );
    return -1.0; // just a formality.
  }
}

/**
 * Fluid radiation sources. Kind of redundant with Rad_sources.
 * TODO: extend to O(b^2)
 **/
Real Source_Fluid_Rad( Real D, Real V, Real T, Real X, Real kappa, Real E,
                       Real F, Real Pr, int iCF ) {
  assert( iCF == 0 || iCF == 1 || iCF == 2 );
  if ( iCF == 0 ) return 0.0; // rad doesn't source mass

  const Real c = constants::c_cgs;

  Real G0, G;
  RadiationFourForce( D, V, T, kappa, E, F, Pr, G0, G );

  return ( iCF == 1 ) ? G : c * G0;
}
/**
 * Gudonov style numerical flux. Constucts v* and p* states.
 **/
void NumericalFlux_Gudonov( const Real vL, const Real vR, const Real pL,
                            const Real pR, const Real zL, const Real zR,
                            Real &Flux_U, Real &Flux_P ) {
  assert( pL > 0.0 && pR > 0.0 &&
          "NumericalFlux_Gudonov :: negative pressure" );
  Flux_U = ( pL - pR + zR * vR + zL * vL ) / ( zR + zL );
  Flux_P = ( zR * pL + zL * pR + zL * zR * ( vL - vR ) ) / ( zR + zL );
}

/**
 * Gudonov style numerical flux. Constucts v* and p* states.
 **/
void NumericalFlux_HLLC( Real vL, Real vR, Real pL, Real pR, Real cL, Real cR,
                         Real rhoL, Real rhoR, Real &Flux_U, Real &Flux_P ) {
  Real aL = vL - cL; // left wave speed estimate
  Real aR = vR + cR; // right wave speed estimate
  Flux_U  = ( rhoR * vR * ( aR - vR ) - rhoL * vL * ( aL - vL ) + pL - pR ) /
           ( rhoR * ( aR - vR ) - rhoL * ( aL - vL ) );
  Flux_P = rhoL * ( vL - aL ) * ( vL - Flux_U ) + pL;
}

// Compute Auxilliary

/**
 * Compute the fluid timestep.
 **/
Real ComputeTimestep_Fluid( const View3D<Real> U, const GridStructure *Grid,
                            EOS *eos, const Real CFL ) {

  const Real MIN_DT = 0.000000005;
  const Real MAX_DT = 1.0;

  const int &ilo = Grid->Get_ilo( );
  const int &ihi = Grid->Get_ihi( );

  Real dt = 0.0;
  Kokkos::parallel_reduce(
      "Compute Timestep", Kokkos::RangePolicy<>( ilo, ihi + 1 ),
      KOKKOS_LAMBDA( const int &iX, Real &lmin ) {
        // --- Compute Cell Averages ---
        Real tau_x  = U( 0, iX, 0 );
        Real vel_x  = U( 1, iX, 0 );
        Real eint_x = U( 2, iX, 0 );

        assert( tau_x > 0.0 && "Compute Timestep :: bad specific volume" );
        assert( eint_x > 0.0 && "Compute Timestep :: bad specific energy" );

        Real dr = Grid->Get_Widths( iX );

        auto lambda = nullptr;
        const Real Cs =
            eos->SoundSpeedFromConserved( tau_x, vel_x, eint_x, lambda );
        Real eigval = Cs;

        Real dt_old = std::abs( dr ) / std::abs( eigval );

        if ( dt_old < lmin ) lmin = dt_old;
      },
      Kokkos::Min<Real>( dt ) );

  dt = std::max( CFL * dt, MIN_DT );
  dt = std::min( dt, MAX_DT );

  // could be assertion?
  if ( std::isnan( dt ) ) {
    throw Error( " ! nan encountered in ComputeTimestep.\n" );
  }

  return dt;
}
