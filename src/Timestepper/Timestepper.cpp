/**
 * File     :  Timestepper.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : SSPRK timestepping routines
 **/

#include <iostream>
#include <vector>

#include "Error.hpp"
#include "Fluid_Discretization.hpp"
#include "Grid.hpp"
#include "PolynomialBasis.hpp"
#include "SlopeLimiter.hpp"
#include "Timestepper.hpp"

/**
 * The constructor creates the necessary data structures for time evolution.
 * Lots of structures used in Fluid Discretization live here.
 **/
TimeStepper::TimeStepper( ProblemIn *pin, GridStructure &Grid )
    : mSize( Grid.Get_nElements( ) + 2 * Grid.Get_Guard( ) ),
      nStages( pin->nStages ), tOrder( pin->tOrder ), BC( pin->BC ),
      a_jk( "RK a_jk", nStages, nStages ), b_jk( "RK b_jk", nStages, nStages ),
      U_s( "U_s", nStages + 1, 4, mSize + 1, pin->pOrder ),
      dU_s( "dU_s", nStages + 1, 4, mSize + 1, pin->pOrder ),
      SumVar_U( "SumVar_U", 4, mSize + 1, pin->pOrder ),
      Grid_s( nStages + 1, GridStructure( pin ) ),
      StageData( "StageData", nStages + 1, mSize + 1 ),
      Flux_q( "Flux_q", 4, mSize + 1, Grid.Get_nNodes( ) ),
      dFlux_num( "Numerical Flux", 4, mSize + 1 ),
      uCF_F_L( "Face L", 4, mSize ), uCF_F_R( "Face R", 4, mSize ),
      Flux_U( "Flux_U", nStages + 1, mSize + 1 ),
      Flux_P( "Flux_P", mSize + 1 ), Flux_B( "Flux_B", mSize + 1 ) {

  // --- Call Initialization ---
  InitializeTimestepper( );
}

// Initialize arrays for timestepper
// TODO: Separate nStages from a tOrder
void TimeStepper::InitializeTimestepper( ) {

  if ( tOrder == 1 and nStages > 1 ) {
    throw Error( "\n \
      ! Issue in setting SSPRK coefficients.\n \
      Please enter an appropriate SSPRK temporal order and nStages\n \
      combination. We support first through fourth order timesteppers\n \
      using 1-3 stages for first-thrid order and 5 stages for second\n \
      through fourth order.\n === \n" );
  }
  if ( ( nStages != tOrder && nStages != 5 ) ) {
    throw Error( "\n \
      ! Issue in setting SSPRK coefficients.\n \
      Please enter an appropriate SSPRK temporal order and nStages\n \
      combination. We support first through fourth order timesteppers\n \
      using 1-3 stages for first-thrid order and 5 stages for second\n \
      through fourth order.\n === \n" );
  }
  if ( ( tOrder == 4 && nStages != 5 ) ) {
    throw Error( "\n \
      ! Issue in setting SSPRK coefficients.\n \
      Please enter an appropriate SSPRK temporal order and nStages\n \
      combination. We support first through fourth order timesteppers\n \
      using 1-3 stages for first-thrid order and 5 stages for second\n \
      through fourth order.\n === \n" );
  }
  if ( tOrder > 4 ) {
    throw Error( "\n ! Temporal torder > 4 not supported! \n" );
  }

  // Init to zero
  for ( int i = 0; i < nStages; i++ )
    for ( int j = 0; j < nStages; j++ ) {
      a_jk( i, j ) = 0.0;
      b_jk( i, j ) = 0.0;
    }

  if ( nStages < 5 ) {

    if ( tOrder == 1 ) {
      a_jk( 0, 0 ) = 1.0;
      b_jk( 0, 0 ) = 1.0;
    } else if ( tOrder == 2 ) {
      a_jk( 0, 0 ) = 1.0;
      a_jk( 1, 0 ) = 0.5;
      a_jk( 1, 1 ) = 0.5;

      b_jk( 0, 0 ) = 1.0;
      b_jk( 1, 0 ) = 0.0;
      b_jk( 1, 1 ) = 0.5;
    } else if ( tOrder == 3 ) {
      a_jk( 0, 0 ) = 1.0;
      a_jk( 1, 0 ) = 0.75;
      a_jk( 1, 1 ) = 0.25;
      a_jk( 2, 0 ) = 1.0 / 3.0;
      a_jk( 2, 1 ) = 0.0;
      a_jk( 2, 2 ) = 2.0 / 3.0;

      b_jk( 0, 0 ) = 1.0;
      b_jk( 1, 0 ) = 0.0;
      b_jk( 1, 1 ) = 0.25;
      b_jk( 2, 0 ) = 0.0;
      b_jk( 2, 1 ) = 0.0;
      b_jk( 2, 2 ) = 2.0 / 3.0;
    }
  } else if ( nStages == 5 ) {
    if ( tOrder == 1 ) {
      throw Error( "\n ! We do support a 1st order, 5 stage SSPRK "
                   "integrator. \n" );
    } else if ( tOrder == 2 ) {
      a_jk( 0, 0 ) = 1.0;
      a_jk( 4, 0 ) = 0.2;
      a_jk( 1, 1 ) = 1.0;
      a_jk( 2, 2 ) = 1.0;
      a_jk( 3, 3 ) = 1.0;
      a_jk( 4, 4 ) = 0.8;

      b_jk( 0, 0 ) = 0.25;
      b_jk( 1, 1 ) = 0.25;
      b_jk( 2, 2 ) = 0.25;
      b_jk( 3, 3 ) = 0.25;
      b_jk( 4, 4 ) = 0.20;
    } else if ( tOrder == 3 ) {
      a_jk( 0, 0 ) = 1.0;
      a_jk( 1, 0 ) = 0.0;
      a_jk( 2, 0 ) = 0.56656131914033;
      a_jk( 3, 0 ) = 0.09299483444413;
      a_jk( 4, 0 ) = 0.00736132260920;
      a_jk( 1, 1 ) = 1.0;
      a_jk( 3, 1 ) = 0.00002090369620;
      a_jk( 4, 1 ) = 0.20127980325145;
      a_jk( 2, 2 ) = 0.43343868085967;
      a_jk( 4, 2 ) = 0.00182955389682;
      a_jk( 3, 3 ) = 0.90698426185967;
      a_jk( 4, 4 ) = 0.78952932024253;

      b_jk( 0, 0 ) = 0.37726891511710;
      b_jk( 3, 0 ) = 0.00071997378654;
      b_jk( 4, 0 ) = 0.00277719819460;
      b_jk( 1, 1 ) = 0.37726891511710;
      b_jk( 4, 1 ) = 0.00001567934613;
      b_jk( 2, 2 ) = 0.16352294089771;
      b_jk( 3, 3 ) = 0.34217696850008;
      b_jk( 4, 4 ) = 0.29786487010104;
    } else if ( tOrder == 4 ) {
      // a_jk( 0, 0 ) = 1.0;
      // a_jk( 1, 0 ) = 0.44437049406734;
      // a_jk( 2, 0 ) = 0.62010185138540;
      // a_jk( 3, 0 ) = 0.17807995410773;
      // a_jk( 4, 0 ) = 0.00683325884039;
      // a_jk( 1, 1 ) = 0.55562950593266;
      // a_jk( 2, 2 ) = 0.37989814861460;
      // a_jk( 4, 2 ) = 0.51723167208978;
      // a_jk( 3, 3 ) = 0.82192004589227;
      // a_jk( 4, 3 ) = 0.12759831133288;
      // a_jk( 4, 4 ) = 0.34833675773694;

      // b_jk( 0, 0 ) = 0.39175222700392;
      // b_jk( 1, 1 ) = 0.36841059262959;
      // b_jk( 2, 2 ) = 0.25189177424738;
      // b_jk( 3, 3 ) = 0.54497475021237;
      // b_jk( 4, 3 ) = 0.08460416338212;
      // b_jk( 4, 4 ) = 0.22600748319395;
      a_jk( 0, 0 ) = 1.0;
      a_jk( 1, 0 ) = 0.444370493651235;
      a_jk( 2, 0 ) = 0.620101851488403;
      a_jk( 3, 0 ) = 0.178079954393132;
      a_jk( 4, 0 ) = 0.000000000000000;
      a_jk( 1, 1 ) = 0.555629506348765;
      a_jk( 2, 2 ) = 0.379898148511597;
      a_jk( 4, 2 ) = 0.517231671970585;
      a_jk( 3, 3 ) = 0.821920045606868;
      a_jk( 4, 3 ) = 0.096059710526147;
      a_jk( 4, 4 ) = 0.386708617503269;

      b_jk( 0, 0 ) = 0.391752226571890;
      b_jk( 1, 1 ) = 0.368410593050371;
      b_jk( 2, 2 ) = 0.251891774271694;
      b_jk( 3, 3 ) = 0.544974750228521;
      b_jk( 4, 3 ) = 0.063692468666290;
      b_jk( 4, 4 ) = 0.226007483236906;
    }
  }
}
