#ifndef _TIMESTEPPER_HPP_
#define _TIMESTEPPER_HPP_

/**
 * File     :  Timestepper.hpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for SSPRK timestepping
 **/

#include "Abstractions.hpp"
#include "BoundEnforcingLimiter.hpp"
#include "EoS.hpp"
#include "PolynomialBasis.hpp"
#include "ProblemIn.hpp"
#include "state.hpp"

class TimeStepper {

 public:
  // TODO: Is it possible to initialize Grid_s from Grid directly?
  TimeStepper( ProblemIn *pin, GridStructure &Grid );

  void InitializeTimestepper( );

  /**
   * Update fluid solution with SSPRK methods
   **/
  template <typename T>
  void UpdateFluid( T ComputeIncrement, const Real dt, State *state,
                    GridStructure &Grid, const ModalBasis *Basis,
                    const EOS *eos, SlopeLimiter *S_Limiter,
                    const Options opts ) {

    const auto &order = Basis->Get_Order( );
    const auto &ihi   = Grid.Get_ihi( );

    auto U   = state->Get_uCF( );
    auto uCR = state->Get_uCR( );

    const int nvars = U.extent( 0 );

    unsigned short int i;

    Kokkos::parallel_for(
        "Timestepper 2",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                { order, ihi + 2, nvars } ),
        KOKKOS_LAMBDA( const int k, const int iX, const int iCF ) {
          U_s( 0, iCF, iX, k ) = U( iCF, iX, k );
        } );

    Grid_s[0] = Grid;
    // StageData holds left interface positions
    Kokkos::parallel_for(
        ihi + 2, KOKKOS_LAMBDA( int iX ) {
          StageData( 0, iX ) = Grid.Get_LeftInterface( iX );
          Flux_U( 0, iX )    = 0.0;
        } );

    for ( unsigned short int iS = 1; iS <= nStages; iS++ ) {
      i = iS - 1;
      // re-zero the summation variables `SumVar`
      Kokkos::parallel_for(
          "Timestepper 3",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                  { order, ihi + 2, nvars } ),
          KOKKOS_LAMBDA( const int k, const int iX, const int iCF ) {
            SumVar_U( iCF, iX, k ) = 0.0;
            StageData( iS, iX )    = 0.0;
            ;
          } );

      // --- Inner update loop ---

      for ( int j = 0; j < iS; j++ ) {
        auto Usj =
            Kokkos::subview( U_s, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
        auto dUsj =
            Kokkos::subview( dU_s, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
        auto Flux_Uj = Kokkos::subview( Flux_U, j, Kokkos::ALL );
        ComputeIncrement( Usj, uCR, Grid_s[j], Basis, eos, dUsj, Flux_q,
                          dFlux_num, uCF_F_L, uCF_F_R, Flux_Uj, Flux_P, Flux_B, opts );

        // inner sum
        Kokkos::parallel_for(
            "Timestepper 4",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                    { order, ihi + 2, nvars } ),
            KOKKOS_LAMBDA( const int k, const int iX, const int iCF ) {
              SumVar_U( iCF, iX, k ) += a_jk( i, j ) * Usj( iCF, iX, k ) +
                                        dt * b_jk( i, j ) * dUsj( iCF, iX, k );
            } );

        Kokkos::parallel_for(
            ihi + 2, KOKKOS_LAMBDA( int iX ) {
              StageData( iS, iX ) += a_jk( i, j ) * StageData( j, iX ) +
                                     dt * b_jk( i, j ) * Flux_Uj( iX );
            } );
      }
      // End inner loop

      Kokkos::parallel_for(
          "Timestepper 5",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                  { order, ihi + 2, nvars } ),
          KOKKOS_LAMBDA( const int k, const int iX, const int iCF ) {
            U_s( iS, iCF, iX, k ) = SumVar_U( iCF, iX, k );
          } );

      auto StageDataj = Kokkos::subview( StageData, iS, Kokkos::ALL );
      Grid_s[iS].UpdateGrid( StageDataj );

      // ! This may give poor performance. Why? ! But also helps with Sedov..
      //    auto Usj =
      //        Kokkos::subview( U_s, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL
      //        );
      //    S_Limiter->ApplySlopeLimiter( Usj, &Grid_s[iS], Basis );
      //    ApplyBoundEnforcingLimiter( Usj, Basis, eos );
    }

    Kokkos::parallel_for(
        "Timestepper Final",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                { order, ihi + 2, nvars } ),
        KOKKOS_LAMBDA( const int k, const int iX, const int iCF ) {
          U( iCF, iX, k ) = U_s( nStages, iCF, iX, k );
        } );

    Grid = Grid_s[nStages];
    S_Limiter->ApplySlopeLimiter( U, &Grid, Basis );
    ApplyBoundEnforcingLimiter( U, Basis, eos );
  }

  /**
   * Update radiation solution with SSPRK methods
   **/
  template <typename T>
  void UpdateRadiation( T ComputeIncrementRad, const Real dt, State *state,
                        GridStructure &Grid, const ModalBasis *Basis,
                        const EOS *eos, SlopeLimiter *S_Limiter,
                        const Options opts ) {

    const auto &order = Basis->Get_Order( );
    const auto &ihi   = Grid.Get_ihi( );

    auto uCF = state->Get_uCF( );
    auto uCR = state->Get_uCR( );

    const int nvars = uCR.extent( 0 );

    unsigned short int i;

    Kokkos::parallel_for(
        "Timestepper::Rad::1",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                { order, ihi + 2, nvars } ),
        KOKKOS_LAMBDA( const int k, const int iX, const int iCR ) {
          U_s( 0, iCR, iX, k )  = uCR( iCR, iX, k );
          dU_s( 0, iCR, iX, k ) = 0.0;
        } );

    // Grid_s[0] = Grid;

    for ( unsigned short int iS = 1; iS <= nStages; iS++ ) {
      i = iS - 1;
      // re-zero the summation variables `SumVar`
      Kokkos::parallel_for(
          "Timestepper::Rad::2",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                  { order, ihi + 2, nvars } ),
          KOKKOS_LAMBDA( const int k, const int iX, const int iCR ) {
            SumVar_U( iCR, iX, k ) = 0.0;
          } );

      // --- Inner update loop ---

      for ( int j = 0; j < iS; j++ ) {
        auto Usj =
            Kokkos::subview( U_s, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
        auto dUsj =
            Kokkos::subview( dU_s, j, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL );
        auto Flux_Uj = Kokkos::subview( Flux_U, j, Kokkos::ALL );
        ComputeIncrementRad( Usj, uCF, Grid, Basis, eos, dUsj, Flux_q,
                             dFlux_num, uCF_F_L, uCF_F_R, Flux_Uj, Flux_P,
                             Flux_B, opts );

        // inner sum
        Kokkos::parallel_for(
            "Timestepper::Rad::3",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                    { order, ihi + 2, nvars } ),
            KOKKOS_LAMBDA( const int k, const int iX, const int iCR ) {
              SumVar_U( iCR, iX, k ) += a_jk( i, j ) * Usj( iCR, iX, k ) +
                                        dt * b_jk( i, j ) * dUsj( iCR, iX, k );
            } );
      }
      // End inner loop

      Kokkos::parallel_for(
          "Timestepper::Rad::4",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                  { order, ihi + 2, nvars } ),
          KOKKOS_LAMBDA( const int k, const int iX, const int iCR ) {
            U_s( iS, iCR, iX, k ) = SumVar_U( iCR, iX, k );
          } );

      // ! This may give poor performance. Why? ! But also helps with Sedov..
      //    auto Usj =
      //        Kokkos::subview( U_s, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL
      //        );
      //    S_Limiter->ApplySlopeLimiter( Usj, &Grid_s[iS], Basis );
      //    ApplyBoundEnforcingLimiter( Usj, Basis, eos );
    }

    Kokkos::parallel_for(
        "Timestepper::Rad::Final",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>( { 0, 0, 0 },
                                                { order, ihi + 2, nvars } ),
        KOKKOS_LAMBDA( const int k, const int iX, const int iCR ) {
          uCR( iCR, iX, k ) = U_s( nStages, iCR, iX, k );
        } );

    // S_Limiter->ApplySlopeLimiter( uCR, &Grid, Basis );
    // ApplyBoundEnforcingLimiter( uCR, Basis, eos );
  }

 private:
  const int mSize;
  const int nStages;
  const int tOrder;
  const std::string BC;

  // SSP coefficients
  View2D a_jk;
  View2D b_jk;

  // Hold stage data
  Kokkos::View<Real ****> U_s;
  Kokkos::View<Real ****> dU_s;
  View3D SumVar_U;
  std::vector<GridStructure> Grid_s;

  // StageData Holds cell left interface positions
  View2D StageData;

  // Variables to pass to update step
  View3D Flux_q;

  View2D dFlux_num;
  View2D uCF_F_L;
  View2D uCF_F_R;

  View2D Flux_U;
  View1D Flux_P;
  View1D Flux_B;
};

#endif // _TIMESTEPPER_HPP_
