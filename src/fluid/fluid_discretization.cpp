/**
 * File     :  fluid_discretization.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : The main fluid spatial update routines go here.
 *  Compute divergence term.
 **/

#include <iostream>

#include "Kokkos_Core.hpp"

#include "boundary_conditions.hpp"
#include "error.hpp"
#include "fluid_discretization.hpp"
#include "fluid_utilities.hpp"
#include "grid.hpp"
#include "polynomial_basis.hpp"
#include "rad_utilities.hpp"

// Compute the divergence of the flux term for the update
void ComputeIncrement_Fluid_Divergence(
    const View3D<Real> U, GridStructure &Grid, const ModalBasis *Basis,
    const EOS *eos, View3D<Real> dU, View3D<Real> Flux_q,
    View2D<Real> dFlux_num, View2D<Real> uCF_F_L, View2D<Real> uCF_F_R,
    View1D<Real> Flux_U, View1D<Real> Flux_P, const Options opts ) {
  const auto &nNodes = Grid.Get_nNodes( );
  const auto &order  = Basis->Get_Order( );
  const auto &ilo    = Grid.Get_ilo( );
  const auto &ihi    = Grid.Get_ihi( );
  const int nvars    = U.extent( 0 );

  // --- Interpolate Conserved Variable to Interfaces ---

  // Left/Right face states
  Kokkos::parallel_for(
      "Fluid :: Interface States",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { ilo, 0 }, { ihi + 2, nvars } ),
      KOKKOS_LAMBDA( const int iX, const int iCF ) {
        uCF_F_L( iCF, iX ) = Basis->BasisEval( U, iX - 1, iCF, nNodes + 1 );
        uCF_F_R( iCF, iX ) = Basis->BasisEval( U, iX, iCF, 0 );
      } );

  // --- Calc numerical flux at all faces
  Kokkos::parallel_for(
      "Fluid :: Numerical Fluxes", Kokkos::RangePolicy<>( ilo, ihi + 2 ),
      KOKKOS_LAMBDA( int iX ) {
        auto uCF_L = Kokkos::subview( uCF_F_L, Kokkos::ALL, iX );
        auto uCF_R = Kokkos::subview( uCF_F_R, Kokkos::ALL, iX );

        const Real rho_L = 1.0 / uCF_L( 0 );
        const Real rho_R = 1.0 / uCF_R( 0 );

        // Debug mode assertions.
        assert( rho_L > 0.0 && !std::isnan( rho_L ) &&
                "fluid_discretization :: Numerical Fluxes bad rho." );
        assert( rho_R > 0.0 && !std::isnan( rho_R ) &&
                "fluid_discretization :: Numerical Fluxes bad rho." );
        assert( uCF_L( 2 ) > 0.0 && !std::isnan( uCF_L( 2 ) ) &&
                "fluid_discretization :: Numerical Fluxes bad energy." );
        assert( uCF_R( 2 ) > 0.0 && !std::isnan( uCF_R( 2 ) ) &&
                "fluid_discretization :: Numerical Fluxes bad energy." );

        auto lambda      = nullptr;
        const Real P_L   = eos->PressureFromConserved( uCF_L( 0 ), uCF_L( 1 ),
                                                       uCF_L( 2 ), lambda );
        const Real Cs_L  = eos->SoundSpeedFromConserved( uCF_L( 0 ), uCF_L( 1 ),
                                                         uCF_L( 2 ), lambda );
        const Real lam_L = Cs_L * rho_L;

        const Real P_R   = eos->PressureFromConserved( uCF_R( 0 ), uCF_R( 1 ),
                                                       uCF_R( 2 ), lambda );
        const Real Cs_R  = eos->SoundSpeedFromConserved( uCF_R( 0 ), uCF_R( 1 ),
                                                         uCF_R( 2 ), lambda );
        const Real lam_R = Cs_R * rho_R;

        // --- Numerical Fluxes ---

        // Riemann Problem
        NumericalFlux_Gudonov( uCF_L( 1 ), uCF_R( 1 ), P_L, P_R, lam_L, lam_R,
                               Flux_U( iX ), Flux_P( iX ) );
        // NumericalFlux_HLLC( uCF_L( 1 ), uCF_R( 1 ), P_L, P_R, Cs_L, Cs_R,
        //  rho_L, rho_R, Flux_U( iX ), Flux_P( iX ) );

        // TODO: Clean This Up
        dFlux_num( 0, iX ) = -Flux_U( iX );
        dFlux_num( 1, iX ) = +Flux_P( iX );
        dFlux_num( 2, iX ) = +Flux_U( iX ) * Flux_P( iX );
      } );

  Flux_U( ihi + 2 ) = Flux_U( ihi + 1 );

  // --- Surface Term ---
  Kokkos::parallel_for(
      "Fluid :: Surface Term",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                              { order, ihi + 1, nvars } ),
      KOKKOS_LAMBDA( const int k, const int iX, const int iCF ) {
        const auto &Poly_L   = Basis->Get_Phi( iX, 0, k );
        const auto &Poly_R   = Basis->Get_Phi( iX, nNodes + 1, k );
        const auto &X_L      = Grid.Get_LeftInterface( iX );
        const auto &X_R      = Grid.Get_LeftInterface( iX + 1 );
        const auto &SqrtGm_L = Grid.Get_SqrtGm( X_L );
        const auto &SqrtGm_R = Grid.Get_SqrtGm( X_R );

        dU( iCF, iX, k ) -= ( +dFlux_num( iCF, iX + 1 ) * Poly_R * SqrtGm_R -
                              dFlux_num( iCF, iX + 0 ) * Poly_L * SqrtGm_L );
      } );

  if ( order > 1 ) {
    // --- Compute Flux_q everywhere for the Volume term ---
    Kokkos::parallel_for(
        "Fluid :: Flux_q",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                                { nNodes, ihi + 1, nvars } ),
        KOKKOS_LAMBDA( const int iN, const int iX, const int iCF ) {
          auto lambda  = nullptr;
          const Real P = eos->PressureFromConserved(
              Basis->BasisEval( U, iX, 0, iN + 1 ),
              Basis->BasisEval( U, iX, 1, iN + 1 ),
              Basis->BasisEval( U, iX, 2, iN + 1 ), lambda );
          Flux_q( iCF, iX, iN ) =
              Flux_Fluid( Basis->BasisEval( U, iX, 1, iN + 1 ), P, iCF );
        } );

    // --- Volume Term ---
    // TODO: Make Flux_q a function?
    Kokkos::parallel_for(
        "Fluid :: Volume Term",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                                { order, ihi + 1, nvars } ),
        KOKKOS_LAMBDA( const int k, const int iX, const int iCF ) {
          Real local_sum = 0.0;
          for ( int iN = 0; iN < nNodes; iN++ ) {
            auto X = Grid.NodeCoordinate( iX, iN );
            local_sum += Grid.Get_Weights( iN ) * Flux_q( iCF, iX, iN ) *
                         Basis->Get_dPhi( iX, iN + 1, k ) *
                         Grid.Get_SqrtGm( X );
          }

          dU( iCF, iX, k ) += local_sum;
        } );
  }
}

/**
 * Compute fluid increment from geometry in spherical symmetry
 **/
void ComputeIncrement_Fluid_Geometry( const View3D<Real> U, GridStructure &Grid,
                                      const ModalBasis *Basis, const EOS *eos,
                                      View3D<Real> dU ) {
  const int nNodes = Grid.Get_nNodes( );
  const int order  = Basis->Get_Order( );
  const int ilo    = Grid.Get_ilo( );
  const int ihi    = Grid.Get_ihi( );

  Kokkos::parallel_for(
      "Fluid :: Geometry Term",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { 0, ilo }, { order, ihi + 1 } ),
      KOKKOS_LAMBDA( const int k, const int iX ) {
        Real local_sum = 0.0;
        auto lambda    = nullptr;
        for ( int iN = 0; iN < nNodes; iN++ ) {
          const Real P = eos->PressureFromConserved(
              Basis->BasisEval( U, iX, 0, iN + 1 ),
              Basis->BasisEval( U, iX, 1, iN + 1 ),
              Basis->BasisEval( U, iX, 2, iN + 1 ), lambda );

          Real X = Grid.NodeCoordinate( iX, iN );

          local_sum +=
              Grid.Get_Weights( iN ) * P * Basis->Get_Phi( iX, iN + 1, k ) * X;
        }

        dU( 1, iX, k ) += ( 2.0 * local_sum * Grid.Get_Widths( iX ) ) /
                          Basis->Get_MassMatrix( iX, k );
      } );
}

/**
 * Compute fluid increment from radiation sources
 * TODO: Modify inputs?
 **/
void ComputeIncrement_Fluid_Rad( const View3D<Real> uCF, const View3D<Real> uCR,
                                 GridStructure &Grid, const ModalBasis *Basis,
                                 const EOS *eos, View3D<Real> dU ) {
  const int nNodes = Grid.Get_nNodes( );
  const int order  = Basis->Get_Order( );
  const int ilo    = Grid.Get_ilo( );
  const int ihi    = Grid.Get_ihi( );

  Kokkos::parallel_for(
      "Fluid :: Source Term; Rad",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>( { 0, ilo }, { order, ihi + 1 } ),
      KOKKOS_LAMBDA( const int k, const int iX ) {
        Real local_sum1 = 0.0;
        Real local_sum2 = 0.0;
        for ( int iN = 0; iN < nNodes; iN++ ) {
          const Real Tau = Basis->BasisEval( uCF, iX, 0, iN + 1 );
          const Real Vel = Basis->BasisEval( uCF, iX, 1, iN + 1 );
          const Real EmT = Basis->BasisEval( uCF, iX, 2, iN + 1 );

          const Real Er = Basis->BasisEval( uCR, iX, 0, iN + 1 ) / Tau;
          const Real Fr = Basis->BasisEval( uCR, iX, 1, iN + 1 ) / Tau;
          const Real Pr = ComputeClosure( Er, Fr );

          auto lambda  = nullptr;
          const Real P = eos->PressureFromConserved( Tau, Vel, EmT, lambda );

          const Real T = eos->TemperatureFromTauPressure( Tau, P, lambda );

          // TODO: kappa and chi will be updated here.
          const Real kappa = ComputeOpacity( Tau, Vel, EmT );

          const Real chi = ComputeEmissivity( Tau, Vel, EmT );

          local_sum1 +=
              Grid.Get_Weights( iN ) * Basis->Get_Phi( iX, iN + 1, k ) *
              Source_Fluid_Rad( Tau, Vel, T, chi, kappa, Er, Fr, Pr, 1 );
          local_sum2 +=
              Grid.Get_Weights( iN ) * Basis->Get_Phi( iX, iN + 1, k ) *
              Source_Fluid_Rad( Tau, Vel, T, chi, kappa, Er, Fr, Pr, 2 );
        }

        dU( 1, iX, k ) += ( local_sum1 * Grid.Get_Widths( iX ) ) /
                          Basis->Get_MassMatrix( iX, k );
        dU( 2, iX, k ) += ( local_sum2 * Grid.Get_Widths( iX ) ) /
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
 * uCF_F_L, uCF_F_R : left/right face states
 * Flux_U, Flux_P   : Fluxes (from Riemann problem)
 * uCF_L, uCF_R     : holds interface data
 * BC               : (string) boundary condition type
 **/
void Compute_Increment_Explicit( const View3D<Real> U, const View3D<Real> uCR,
                                 GridStructure &Grid, const ModalBasis *Basis,
                                 const EOS *eos, View3D<Real> dU,
                                 View3D<Real> Flux_q, View2D<Real> dFlux_num,
                                 View2D<Real> uCF_F_L, View2D<Real> uCF_F_R,
                                 View1D<Real> Flux_U, View1D<Real> Flux_P,
                                 const Options opts ) {

  const auto &order = Basis->Get_Order( );
  const auto &ilo   = Grid.Get_ilo( );
  const auto &ihi   = Grid.Get_ihi( );
  const int nvars   = U.extent( 0 );

  // --- Apply BC ---
  ApplyBC( U, &Grid, order, opts.BC );

  // --- Detect Shocks ---
  // TODO: Code up a shock detector...

  // --- Compute Increment for new solution ---

  // --- First: Zero out dU  ---
  Kokkos::parallel_for(
      "Zero dU",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                              { order, ihi + 1, nvars } ),
      KOKKOS_LAMBDA( const int k, const int iX, const int iCF ) {
        dU( iCF, iX, k ) = 0.0;
      } );

  Kokkos::parallel_for(
      ihi + 2, KOKKOS_LAMBDA( int iX ) { Flux_U( iX ) = 0.0; } );

  // --- Fluid Increment : Divergence ---
  ComputeIncrement_Fluid_Divergence( U, Grid, Basis, eos, dU, Flux_q, dFlux_num,
                                     uCF_F_L, uCF_F_R, Flux_U, Flux_P, opts );

  // --- Divide update by mass mastrix ---
  Kokkos::parallel_for(
      "Fluid::Divide Update / Mass Matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, ilo, 0 },
                                              { order, ihi + 1, nvars } ),
      KOKKOS_LAMBDA( const int k, const int iX, const int iCF ) {
        dU( iCF, iX, k ) /= ( Basis->Get_MassMatrix( iX, k ) );
      } );

  /* --- Increment from Geometry --- */
  if ( Grid.DoGeometry( ) ) {
    ComputeIncrement_Fluid_Geometry( U, Grid, Basis, eos, dU );
  }

  /* --- Increment Rad --- */
  if ( opts.do_rad ) {
    ComputeIncrement_Fluid_Rad( U, uCR, Grid, Basis, eos, dU );
  }
}
