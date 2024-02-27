#ifndef _INITIALIZATION_HPP_
#define _INITIALIZATION_HPP_

#include "Abstractions.hpp"

#include <iostream>
#include <math.h> /* sin */
#include <string>

#include "Constants.hpp"
#include "Error.hpp"
#include "Grid.hpp"
#include "Initialization.hpp"
#include "equilibrium.hpp"
#include "state.hpp"

#define GAMMA 1.4

/**
 * Initialize the conserved Fields for various problems.
 * TODO: For now I initialize constant on each cell. Is there a better way?
 * TODO: iNodeX and order separation
 **/
void InitializeFields( State *state, GridStructure *Grid,
                       const std::string ProblemName ) {

  const int ilo    = Grid->Get_ilo( );
  const int ihi    = Grid->Get_ihi( );
  const int nNodes = Grid->Get_nNodes( );

  const int iCF_Tau = 0;
  const int iCF_V   = 1;
  const int iCF_E   = 2;
  const int iCF_Bm  = 3;

  const int iPF_D = 0;

  View3D uCF       = state->Get_uCF( );
  View3D uPF       = state->Get_uPF( );
  View3D uCR       = state->Get_uCR( );
  const int pOrder = state->Get_pOrder( );

  if ( ProblemName == "Sod" ) {
    const Real V0  = 0.0;
    const Real D_L = 1.0;
    const Real D_R = 0.125;
    const Real P_L = 1.0;
    const Real P_R = 0.1;

    Real X1 = 0.0;
    for ( int iX = ilo; iX <= ihi; iX++ )
      for ( int k = 0; k < pOrder; k++ )
        for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
          X1                    = Grid->Get_Centers( iX );
          uCF( iCF_Tau, iX, k ) = 0.0;
          uCF( iCF_V, iX, k )   = 0.0;
          uCF( iCF_E, iX, k )   = 0.0;
          uCR( 0, iX, k )       = 0.0;
          uCR( 1, iX, k )       = 0.0;

          if ( X1 <= 0.5 ) {
            if ( k == 0 ) {
              uCF( iCF_Tau, iX, 0 ) = 1.0 / D_L;
              uCF( iCF_V, iX, 0 )   = V0;
              uCF( iCF_E, iX, 0 )   = ( P_L / 0.4 ) * uCF( iCF_Tau, iX, 0 );
            }

            uPF( iPF_D, iX, iNodeX ) = D_L;
          } else { // right domain
            if ( k == 0 ) {
              uCF( iCF_Tau, iX, 0 ) = 1.0 / D_R;
              uCF( iCF_V, iX, 0 )   = V0;
              uCF( iCF_E, iX, 0 )   = ( P_R / 0.4 ) * uCF( iCF_Tau, iX, 0 );
            }

            uPF( iPF_D, iX, iNodeX ) = D_R;
          }
        }
    // Fill density in guard cells
    for ( int iX = 0; iX < ilo; iX++ )
      for ( int iN = 0; iN < nNodes; iN++ ) {
        uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
        uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
      }
  } else if ( ProblemName == "ShuOsher" ) {
    const Real V0  = 2.629369;
    const Real D_L = 3.857143;
    const Real P_L = 10.333333;
    const Real P_R = 1.0;

    Real X1 = 0.0;
    for ( int iX = ilo; iX <= ihi; iX++ )
      for ( int k = 0; k < pOrder; k++ )
        for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
          X1                    = Grid->Get_Centers( iX );
          uCF( iCF_Tau, iX, k ) = 0.0;
          uCF( iCF_V, iX, k )   = 0.0;
          uCF( iCF_E, iX, k )   = 0.0;

          if ( X1 <= -4.0 ) {
            if ( k == 0 ) {
              uCF( iCF_Tau, iX, 0 ) = 1.0 / D_L;
              uCF( iCF_V, iX, 0 )   = V0;
              uCF( iCF_E, iX, 0 ) =
                  ( P_L / 0.4 ) * uCF( iCF_Tau, iX, 0 ) + 0.5 * V0 * V0;
            }

            uPF( iPF_D, iX, iNodeX ) = D_L;
          } else { // right domain
            if ( k == 0 ) {
              uCF( iCF_Tau, iX, 0 ) =
                  1.0 / ( 1.0 + 0.2 * sin( 5.0 * constants::PI( ) * X1 ) );
              uCF( iCF_V, iX, 0 ) = 0.0;
              uCF( iCF_E, iX, 0 ) = ( P_R / 0.4 ) * uCF( iCF_Tau, iX, 0 );
            }

            uPF( iPF_D, iX, iNodeX ) =
                ( 1.0 + 0.2 * sin( 5.0 * constants::PI( ) * X1 ) );
          }
        }
    // Fill density in guard cells
    for ( int iX = 0; iX < ilo; iX++ )
      for ( int iN = 0; iN < nNodes; iN++ ) {
        uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
        uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
      }
  } else if ( ProblemName == "MovingContact" ) {
    // Moving Contact problem.
    const Real V0  = 0.1;
    const Real D_L = 1.4;
    const Real D_R = 1.0;
    const Real P_L = 1.0;
    const Real P_R = 1.0;

    Real X1 = 0.0;
    for ( int iX = ilo; iX <= ihi; iX++ )
      for ( int k = 0; k < pOrder; k++ )
        for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
          X1                    = Grid->Get_Centers( iX );
          uCF( iCF_Tau, iX, k ) = 0.0;
          uCF( iCF_V, iX, k )   = 0.0;
          uCF( iCF_E, iX, k )   = 0.0;

          if ( X1 <= 0.5 ) {
            if ( k == 0 ) {
              uCF( iCF_Tau, iX, 0 ) = 1.0 / D_L;
              uCF( iCF_V, iX, 0 )   = V0;
              uCF( iCF_E, iX, 0 ) =
                  ( P_L / 0.4 ) * uCF( iCF_Tau, iX, 0 ) + 0.5 * V0 * V0;
            }

            uPF( iPF_D, iX, iNodeX ) = D_L;
          } else {
            if ( k == 0 ) {
              uCF( iCF_Tau, iX, k ) = 1.0 / D_R;
              uCF( iCF_V, iX, k )   = V0;
              uCF( iCF_E, iX, k ) =
                  ( P_R / 0.4 ) * uCF( iCF_Tau, iX, k ) + 0.5 * V0 * V0;
            }

            uPF( iPF_D, iX, iNodeX ) = D_R;
          }
        }
    // Fill density in guard cells
    for ( int iX = 0; iX < ilo; iX++ )
      for ( int iN = 0; iN < nNodes; iN++ ) {
        uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
        uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
      }
  } else if ( ProblemName == "SmoothAdvection" ) {
    // Smooth advection problem
    const Real V0  = 1.0;
    const Real P0  = 0.01;
    const Real Amp = 1.0;

    Real X1 = 0.0;
    for ( int iX = ilo; iX <= ihi; iX++ )
      for ( int k = 0; k < pOrder; k++ )
        for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
          X1 = Grid->Get_Centers( iX );

          if ( k != 0 ) {
            uCF( iCF_Tau, iX, k ) = 0.0;
            uCF( iCF_V, iX, k )   = 0.0;
            uCF( iCF_E, iX, k )   = 0.0;
          } else {
            uCF( iCF_Tau, iX, k ) =
                1.0 / ( 2.0 + Amp * sin( 2.0 * constants::PI( ) * X1 ) );
            uCF( iCF_V, iX, k ) = V0;
            uCF( iCF_E, iX, k ) =
                ( P0 / 0.4 ) * uCF( iCF_Tau, iX, k ) + 0.5 * V0 * V0;
          }
          uPF( iPF_D, iX, iNodeX ) =
              ( 2.0 + Amp * sin( 2.0 * constants::PI( ) * X1 ) );
        }
    // Fill density in guard cells
    for ( int iX = 0; iX < ilo; iX++ )
      for ( int iN = 0; iN < nNodes; iN++ ) {
        uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
        uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
      }
  } else if ( ProblemName == "Sedov" ) {
    // Smooth advection problem
    const Real V0 = 0.0;
    const Real D0 = 1.0;
    const Real E0 = 0.3;

    const int origin = Grid->Get_nElements( ) / 2;

    const Real P0 = ( 5.0 / 3.0 - 1.0 ) * E0 / Grid->Get_Widths( origin );

    for ( int iX = ilo; iX <= ihi; iX++ )
      for ( int k = 0; k < pOrder; k++ )
        for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {

          if ( k != 0 ) {
            uCF( iCF_Tau, iX, k ) = 0.0;
            uCF( iCF_V, iX, k )   = 0.0;
            uCF( iCF_E, iX, k )   = 0.0;
          } else {
            uCF( iCF_Tau, iX, k ) = 1.0 / D0;
            uCF( iCF_V, iX, k )   = V0;
            if ( iX == origin - 1 || iX == origin ) {
              uCF( iCF_E, iX, k ) =
                  ( P0 / ( 5.0 / 3.0 - 1.0 ) ) * uCF( iCF_Tau, iX, k ) +
                  0.5 * V0 * V0;
            } else {
              uCF( iCF_E, iX, k ) =
                  ( 1.0e-6 / ( 5.0 / 3.0 - 1.0 ) ) * uCF( iCF_Tau, iX, k ) +
                  0.5 * V0 * V0;
            }
          }
          uPF( iPF_D, iX, iNodeX ) = D0;
        }
    // Fill density in guard cells
    for ( int iX = 0; iX < ilo; iX++ )
      for ( int iN = 0; iN < nNodes; iN++ ) {
        uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
        uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
      }
  } else if ( ProblemName == "Noh" ) {
    const Real V_L = 1.0;
    const Real D_L = 1.0;
    const Real P_L = 0.000001;
    const Real D_R = 1.0;
    const Real V_R = -1.0;
    const Real P_R = 0.000001;

    Real X1 = 0.0;
    for ( int iX = ilo; iX <= ihi; iX++ )
      for ( int k = 0; k < pOrder; k++ )
        for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
          X1                    = Grid->NodeCoordinate( iX, iNodeX );
          uCF( iCF_Tau, iX, k ) = 0.0;
          uCF( iCF_V, iX, k )   = 0.0;
          uCF( iCF_E, iX, k )   = 0.0;

          if ( X1 <= 0.5 ) {
            if ( k == 0 ) {
              uCF( iCF_Tau, iX, 0 ) = 1.0 / D_L;
              uCF( iCF_V, iX, 0 )   = V_L;
              uCF( iCF_E, iX, 0 ) =
                  ( P_L / ( GAMMA - 1.0 ) ) * uCF( iCF_Tau, iX, 0 ) +
                  0.5 * V_L * V_L;
            }

            uPF( iPF_D, iX, iNodeX ) = D_L;
          } else { // right domain
            if ( k == 0 ) {
              uCF( iCF_Tau, iX, 0 ) = 1.0 / D_R;
              uCF( iCF_V, iX, 0 )   = V_R;
              uCF( iCF_E, iX, 0 ) =
                  ( P_R / ( GAMMA - 1.0 ) ) * uCF( iCF_Tau, iX, 0 ) +
                  0.5 * V_R * V_R;
            }

            uPF( iPF_D, iX, iNodeX ) = D_R;
          }
        }
    // Fill density in guard cells
    for ( int iX = 0; iX < ilo; iX++ )
      for ( int iN = 0; iN < nNodes; iN++ ) {
        uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
        uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
      }
  } else if ( ProblemName == "ShocklessNoh" ) {
    const Real D   = 1.0;
    const Real E_M = 1.0;

    Real X1 = 0.0;
    for ( int iX = ilo; iX <= ihi; iX++ )
      for ( int k = 0; k < pOrder; k++ )
        for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
          X1                    = Grid->Get_Centers( iX );
          uCF( iCF_Tau, iX, k ) = 0.0;
          uCF( iCF_V, iX, k )   = 0.0;
          uCF( iCF_E, iX, k )   = 0.0;

          if ( k == 0 ) {
            uCF( iCF_Tau, iX, 0 ) = 1.0 / D;
            uCF( iCF_V, iX, 0 )   = -X1;
            uCF( iCF_E, iX, 0 ) =
                E_M + 0.5 * uCF( iCF_V, iX, 0 ) * uCF( iCF_V, iX, 0 );
          } else if ( k == 1 ) {
            uCF( iCF_Tau, iX, k ) = 0.0;
            uCF( iCF_V, iX, k )   = -Grid->Get_Widths( iX );
            uCF( iCF_E, iX, k )   = ( -X1 ) * ( -Grid->Get_Widths( iX ) );
          } else if ( k == 2 ) {
            uCF( iCF_Tau, iX, k ) = 0.0;
            uCF( iCF_V, iX, k )   = 0.0;
            uCF( iCF_E, iX, k )   = uCF( iCF_V, iX, 1 ) * uCF( iCF_V, iX, 1 );
          } else {
            uCF( iCF_Tau, iX, k ) = 0.0;
            uCF( iCF_V, iX, k )   = 0.0;
            uCF( iCF_E, iX, k )   = 0.0;
          }

          uPF( iPF_D, iX, iNodeX ) = D;
        }
    // Fill density in guard cells
    for ( int iX = 0; iX < ilo; iX++ )
      for ( int iN = 0; iN < nNodes; iN++ ) {
        uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
        uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
      }
  } else if ( ProblemName == "SmoothFlow" ) {

    Real X1  = 0.0;
    Real amp = 0.999999999999999999999999999999999995;
    for ( int iX = ilo; iX <= ihi; iX++ )
      for ( int k = 0; k < pOrder; k++ )
        for ( int iNodeX = 0; iNodeX < nNodes; iNodeX++ ) {
          X1                    = Grid->Get_Centers( iX );
          uCF( iCF_Tau, iX, k ) = 0.0;
          uCF( iCF_V, iX, k )   = 0.0;
          uCF( iCF_E, iX, k )   = 0.0;

          if ( k == 0 ) {
            Real D = ( 1.0 + amp * sin( constants::PI( ) * X1 ) );
            uCF( iCF_Tau, iX, 0 ) = 1.0 / D;
            uCF( iCF_V, iX, 0 )   = 0.0;
            uCF( iCF_E, iX, 0 )   = ( D * D * D / 2.0 ) * uCF( iCF_Tau, iX, 0 );
          } else if ( k == 1 ) {
            Real D  = ( 1.0 + amp * sin( constants::PI( ) * X1 ) );
            Real dD = ( amp * constants::PI( ) * cos( constants::PI( ) * X1 ) );
            uCF( iCF_Tau, iX, k ) =
                ( -1 / ( D * D ) ) * dD * Grid->Get_Widths( iX );
            uCF( iCF_V, iX, k ) = 0.0;
            uCF( iCF_E, iX, k ) =
                ( ( 2.0 / 2.0 ) * D ) * dD * Grid->Get_Widths( iX );
          } else if ( k == 2 ) {
            Real D   = ( 1.0 + amp * sin( constants::PI( ) * X1 ) );
            Real ddD = -( amp * constants::PI( ) * constants::PI( ) ) *
                       sin( constants::PI( ) * X1 );
            uCF( iCF_Tau, iX, k ) = ( 2.0 / ( D * D * D ) ) * ddD *
                                    Grid->Get_Widths( iX ) *
                                    Grid->Get_Widths( iX );
            uCF( iCF_V, iX, k ) = 0.0;
            uCF( iCF_E, iX, k ) = ( 2.0 / 2.0 ) * ddD * Grid->Get_Widths( iX ) *
                                  Grid->Get_Widths( iX );
          } else if ( k == 3 ) {
            Real D    = ( 1.0 + amp * sin( constants::PI( ) * X1 ) );
            Real dddD = -( amp * constants::PI( ) * constants::PI( ) *
                           constants::PI( ) ) *
                        cos( constants::PI( ) * X1 );
            uCF( iCF_Tau, iX, k ) =
                ( -6.0 / ( D * D * D * D ) ) * dddD * Grid->Get_Widths( iX ) *
                Grid->Get_Widths( iX ) * Grid->Get_Widths( iX );
            uCF( iCF_V, iX, k ) = 0.0;
            uCF( iCF_E, iX, k ) = 0.0;
          } else {
            uCF( iCF_Tau, iX, k ) = 0.0;
            uCF( iCF_V, iX, k )   = 0.0;
            uCF( iCF_E, iX, k )   = 0.0;
          }

          uPF( iPF_D, iX, iNodeX ) =
              ( 1.0 + amp * sin( constants::PI( ) * X1 ) );
        }
    // Fill density in guard cells
    for ( int iX = 0; iX < ilo; iX++ )
      for ( int iN = 0; iN < nNodes; iN++ ) {
        uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
        uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
      }
  }
  else if ( ProblemName == "ConstMagnetic" )
  {
    const Real B0 = 1.0;
    const Real V0 = 0.0;
    const Real D0 = 1.0;
    const Real P0 = 1.0;

    for ( unsigned int iX = ilo; iX <= ihi; iX++ )
      for ( unsigned int k = 0; k < pOrder; k++ )
        for ( unsigned int iNodeX = 0; iNodeX < nNodes; iNodeX++ )
        {

          uCF( iCF_Tau, iX, k ) = 0.0;
          uCF( iCF_V,   iX, k ) = 0.0;
          uCF( iCF_E,   iX, k ) = 0.0;
          uCF( iCF_Bm,  iX, k ) = 0.0;

          if ( k == 0 )
          {

            uCF( iCF_Tau, iX, 0 ) = 1.0 / D0;
            uCF( iCF_V,   iX, 0 ) = V0;
            uCF( iCF_E,   iX, 0 ) = ( P0 / 0.4 * D0 ) + 0.5 * B0 * B0 / D0;
            uCF( iCF_Bm,  iX, 0 ) = B0 / D0;

          }

         uPF( iPF_D, iX, iNodeX ) = D0;

         }
    // Fill density in guard cells
    for ( unsigned int iX = 0; iX < ilo; iX++ )
      for ( unsigned int iN = 0; iN < nNodes; iN++ )
      {
        uPF( 0, ilo - 1 - iX, iN ) = uPF( 0, ilo + iX, nNodes - iN - 1 );
        uPF( 0, ihi + 1 + iX, iN ) = uPF( 0, ihi - iX, nNodes - iN - 1 );
      }
  } else if ( ProblemName == "RadEquilibrium" ) {
    RadEquilibriumInit( uCF, uPF, uCR, Grid, pOrder );
  } else {
    throw Error( " ! Please choose a valid ProblemName" );
  }
}
#endif // _INITIALIZATION_HPP_
