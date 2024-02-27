/**
 * File     :  Rad_Discretization.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : The main radiation spatial update routines go here.
 *  Compute divergence term.
 **/

#include <iostream>

#include "Kokkos_Core.hpp"

#include "BoundaryConditionsLibrary.hpp"
#include "Constants.hpp"
#include "EoS.hpp"
#include "Error.hpp"
#include "Grid.hpp"
#include "PolynomialBasis.hpp"
#include "RadUtilities.hpp"
#include "Rad_Discretization.hpp"

// Compute the divergence of the flux term for the update
void ComputeIncrement_Rad_Divergence( const View3D uCR, const View3D uCF,
                                      GridStructure &Grid,
                                      const ModalBasis *Basis, const EOS *eos,
                                      View3D dU, View3D Flux_q,
                                      View2D dFlux_num, View2D uCR_F_L,
                                      View2D uCR_F_R, View1D Flux_U,
                                      View1D Flux_P, View1D Flux_B ) {
  const auto &nNodes = Grid.Get_nNodes( );
  const auto &order  = Basis->Get_Order( );
  const auto &ilo    = Grid.Get_ilo( );
  const auto &ihi    = Grid.Get_ihi( );
  const int nvars    = uCR.extent( 0 );

  // Real rho_L, rho_R, P_L, P_R, Cs_L, Cs_R, lam_L, lam_R, P;

  // --- Interpolate Conserved Variable to Interfaces ---

  // Left/Right face states
  // TODO: Can this just be moved into the below kernel with a iCF loop?
  Kokkos::parallel_for(
      "Interface States; Rad",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { ilo, 0 }, { ihi + 2, nvars } ),
      KOKKOS_LAMBDA( const int iX, const int iCR ) {
        uCR_F_L( iCR, iX ) =
            Basis->BasisEval( uCR, iX - 1, iCR, nNodes + 1, false );
        uCR_F_R( iCR, iX ) = Basis->BasisEval( uCR, iX, iCR, 0, false );
      } );

  // --- Calc numerical flux at all faces
  Kokkos::parallel_for(
      "Numerical Fluxes; Rad", Kokkos::RangePolicy<>( ilo, ihi + 2 ),
      KOKKOS_LAMBDA( int iX ) {
        auto uCR_L = Kokkos::subview( uCR_F_L, Kokkos::ALL, iX );
        auto uCR_R = Kokkos::subview( uCR_F_R, Kokkos::ALL, iX );

        const Real tauR = Basis->BasisEval( uCF, iX, 0, 0, false );
        const Real tauL = Basis->BasisEval( uCF, iX - 1, 0, nNodes + 1, false );

        const Real Em_L = uCR_L( 0 ) / tauL;
        const Real Fm_L = uCR_L( 1 ) / tauL;
        const Real Em_R = uCR_R( 0 ) / tauR;
        const Real Fm_R = uCR_R( 1 ) / tauR;

        const Real P_L = ComputeClosure( Em_L, Fm_L );
        const Real P_R = ComputeClosure( Em_R, Fm_R );

        // --- Numerical Fluxes ---

        // Riemann Problem
        Real flux_e   = 0.0;
        Real flux_f   = 0.0;
        const Real vR = Basis->BasisEval( uCF, iX, 1, 0, false );
        const Real vL = Basis->BasisEval( uCF, iX - 1, 1, nNodes + 1, false );
        const Real c_cgs = constants::c_cgs;

        Real Fp = Flux_Rad( Em_R, Fm_R, vR, P_R, 0 );
        Real Fm = Flux_Rad( Em_L, Fm_L, vL, P_L, 0 );
        llf_flux( Fp, Fm, Em_R, Em_L, c_cgs, flux_e );

        Fp = Flux_Rad( Em_R, Fm_R, P_R, vR, 1 );
        Fm = Flux_Rad( Em_L, Fm_L, P_L, vL, 1 ); // WEIRD
        llf_flux( Fp, Fm, Fm_R, Fm_L, c_cgs, flux_f );

        dFlux_num( 0, iX ) = flux_e;
        dFlux_num( 1, iX ) = flux_f;
      } );

  // --- Surface Term ---
  Kokkos::parallel_for(
      "Surface Term; Rad",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                              { order, ihi + 1, nvars } ),
      KOKKOS_LAMBDA( const int k, const int iX, const int iCR ) {
        const auto &Poly_L   = Basis->Get_Phi( iX, 0, k );
        const auto &Poly_R   = Basis->Get_Phi( iX, nNodes + 1, k );
        const auto &X_L      = Grid.Get_LeftInterface( iX );
        const auto &X_R      = Grid.Get_LeftInterface( iX + 1 );
        const auto &SqrtGm_L = Grid.Get_SqrtGm( X_L );
        const auto &SqrtGm_R = Grid.Get_SqrtGm( X_R );

        dU( iCR, iX, k ) -= ( +dFlux_num( iCR, iX + 1 ) * Poly_R * SqrtGm_R -
                              dFlux_num( iCR, iX + 0 ) * Poly_L * SqrtGm_L );
      } );

  if ( order > 1 ) {
    // --- Compute Flux_q everywhere for the Volume term ---
    Kokkos::parallel_for(
        "Flux_q; Rad",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                                { nNodes, ihi + 1, nvars } ),
        KOKKOS_LAMBDA( const int iN, const int iX, const int iCR ) {
          const Real Tau = Basis->BasisEval( uCF, iX, 0, iN + 1, false );
          const auto P   = ComputeClosure(
              Basis->BasisEval( uCR, iX, 0, iN + 1, false ) / Tau,
              Basis->BasisEval( uCR, iX, 1, iN + 1, false ) / Tau );
          Flux_q( iCR, iX, iN ) =
              Flux_Rad( Basis->BasisEval( uCR, iX, 0, iN + 1, false ) / Tau,
                        Basis->BasisEval( uCR, iX, 1, iN + 1, false ) / Tau, P,
                        Basis->BasisEval( uCF, iX, 1, iN + 1, false ), iCR );
        } );

    // --- Volume Term ---
    // TODO: Make Flux_q a function?
    Kokkos::parallel_for(
        "Volume Term; Rad",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                                { order, ihi + 1, nvars } ),
        KOKKOS_LAMBDA( const int k, const int iX, const int iCR ) {
          Real local_sum = 0.0;
          for ( int iN = 0; iN < nNodes; iN++ ) {
            auto X = Grid.NodeCoordinate( iX, iN );
            local_sum += Grid.Get_Weights( iN ) * Flux_q( iCR, iX, iN ) *
                         Basis->Get_dPhi( iX, iN + 1, k ) *
                         Grid.Get_SqrtGm( X );
          }

          dU( iCR, iX, k ) += local_sum;
        } );
  }
}

/**
 * Compute rad increment from source terms
 **/
void ComputeIncrement_Rad_Source( const View3D uCR, const View3D uCF,
                                  GridStructure &Grid, const ModalBasis *Basis,
                                  const EOS *eos, View3D dU ) {
  const int nNodes = Grid.Get_nNodes( );
  const int order  = Basis->Get_Order( );
  const int ilo    = Grid.Get_ilo( );
  const int ihi    = Grid.Get_ihi( );
  const int nvars  = uCR.extent( 0 );

  Kokkos::parallel_for(
      "Rad::Source",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                              { order, ihi + 1, nvars } ),
      KOKKOS_LAMBDA( const int k, const int iX, const int iCR ) {
        Real local_sum = 0.0;
        for ( int iN = 0; iN < nNodes; iN++ ) {
          const Real D    = 1.0 / Basis->BasisEval( uCF, iX, 0, iN + 1, false );
          const Real V    = Basis->BasisEval( uCF, iX, 1, iN + 1, false );
          const Real Em_T = Basis->BasisEval( uCF, iX, 2, iN + 1, false );
          const Real Bm   = Basis->BasisEval( uCF, iX, 3, iN + 1, false );

          const Real Abar = 1.0; // TODO: update abar
          Real lambda[2]  = { Abar, 0.0 };
          const Real P = eos->PressureFromConserved( 1.0 / D, V, Bm, Em_T, lambda );
          const Real T = eos->TemperatureFromTauPressure( 1.0 / D, P, lambda );

          const Real kappa = ComputeOpacity( D, V, Em_T );
          const Real X     = ComputeEmissivity( D, V, Em_T );

          const Real E_r = Basis->BasisEval( uCR, iX, 0, iN + 1, false ) * D;
          const Real F_r = Basis->BasisEval( uCR, iX, 1, iN + 1, false ) * D;
          const Real P_r = ComputeClosure( E_r, F_r );

          const Real this_source =
              Source_Rad( D, V, T, X, kappa, E_r, F_r, P_r, iCR );

          local_sum += Grid.Get_Weights( iN ) *
                       Basis->Get_Phi( iX, iN + 1, k ) * this_source;
        }

        dU( iCR, iX, k ) += ( local_sum * Grid.Get_Widths( iX ) ) /
                            Basis->Get_MassMatrix( iX, k );
      } );
}

/** Compute dU for timestep update. e.g., U = U + dU * dt
 *
 * Parameters:
 * -----------
 * U                : Conserved variables
 * Grid             : Grid object
 * Basis            : Basis object
 * dU               : Update vector
 * Flux_q           : Nodal fluxes, for volume term
 * dFLux_num        : numerical surface flux
 * uCR_F_L, uCR_F_R : left/right face states
 * Flux_U, Flux_P   : Fluxes (from Riemann problem)
 * uCR_L, uCR_R     : holds interface data
 * BC               : (string) boundary condition type
 **/
void Compute_Increment_Explicit_Rad( const View3D uCR, const View3D uCF,
                                     GridStructure &Grid,
                                     const ModalBasis *Basis, const EOS *eos,
                                     View3D dU, View3D Flux_q, View2D dFlux_num,
                                     View2D uCR_F_L, View2D uCR_F_R,
                                     View1D Flux_U, View1D Flux_P,
                                     View1D Flux_B,
                                     const Options opts ) {

  const auto &order = Basis->Get_Order( );
  const auto &ilo   = Grid.Get_ilo( );
  const auto &ihi   = Grid.Get_ihi( );
  const int nvars   = uCR.extent( 0 );

  // --- Apply BC ---
  ApplyBC( uCR, &Grid, order, opts.BC );

  // --- Compute Increment for new solution ---

  // --- First: Zero out dU  ---
  Kokkos::parallel_for(
      "Rad::Zero dU",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                              { order, ihi + 1, nvars } ),
      KOKKOS_LAMBDA( const int k, const int iX, const int iCR ) {
        dU( iCR, iX, k ) = 0.0;
      } );

  Kokkos::parallel_for(
      ihi + 2, KOKKOS_LAMBDA( int iX ) { Flux_U( iX ) = 0.0; } );

  // --- Increment : Divergence ---
  ComputeIncrement_Rad_Divergence( uCR, uCF, Grid, Basis, eos, dU, Flux_q,
                                   dFlux_num, uCR_F_L, uCR_F_R, Flux_U,
                                   Flux_P, Flux_B );

  // --- Divide update by mass mastrix ---
  Kokkos::parallel_for(
      "Rad::Divide Update / Mass Matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                              { order, ihi + 1, nvars } ),
      KOKKOS_LAMBDA( const int k, const int iX, const int iCR ) {
        dU( iCR, iX, k ) /= ( Basis->Get_MassMatrix( iX, k ) );
      } );

  /* --- Increment Source Terms --- */
  ComputeIncrement_Rad_Source( uCR, uCF, Grid, Basis, eos, dU );
}
