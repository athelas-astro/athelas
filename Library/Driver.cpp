/**
 * File    :  Driver.cpp
 * --------------
 *
 * Author  : Brandon L. Barker
 * Purpose : Main driver routine
 **/

#include <iostream>
#include <vector>
#include <string>

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"
#include "Grid.hpp"
#include "BoundaryConditionsLibrary.hpp"
#include "SlopeLimiter.hpp"
#include "Initialization.hpp"
#include "IOLibrary.hpp"
#include "Fluid_Discretization.hpp"
#include "FluidUtilities.hpp"
#include "Timestepper.hpp"
#include "Error.hpp"
#include "ProblemIn.hpp"
#include "Driver.hpp"

int main( int argc, char *argv[] )
{
  // Check cmd line args
  if ( argc < 2 ){ throw Error("No input file passed! Do: ./main IN_FILE"); }

  ProblemIn pin( argv[1] );


  /* --- Problem Parameters --- */
  const std::string ProblemName = pin.ProblemName;

  const UInt &nX      = pin.nElements;
  const UInt &order   = pin.pOrder;
  const UInt &nNodes  = pin.nNodes;
  const UInt &nStages = pin.nStages;
  const UInt &tOrder  = pin.tOrder;

  const UInt &nGuard = pin.nGhost;

  Real t           = 0.0;
  Real dt          = 0.0;
  const Real t_end = pin.t_end;

  bool Restart = pin.Restart;

  const std::string BC = pin.BC;

  const Real CFL = ComputeCFL( pin.CFL, order, nStages, tOrder );

  Kokkos::initialize( argc, argv );
  {

    // --- Create the Grid object ---
   GridStructure Grid( &pin );

   // --- Create the data structures ---
   Kokkos::View<Real ***> uCF( "uCF", 3, nX + 2 * nGuard, order );
   Kokkos::View<Real ***> uPF( "uPF", 3, nX + 2 * nGuard, nNodes );

   Kokkos::View<Real ***> uAF( "uAF", 3, nX + 2 * nGuard, order );

    if ( not Restart )
    {
      // --- Initialize fields ---
      InitializeFields( uCF, uPF, &Grid, order, ProblemName );

      ApplyBC_Fluid( uCF, &Grid, order, BC );
    }
    // WriteState( uCF, uPF, uAF, Grid, ProblemName, 0.0, order, 0 );

    // --- Datastructure for modal basis ---
    ModalBasis Basis( uPF, &Grid, order, nNodes, nX, nGuard );

    WriteBasis( &Basis, nGuard, Grid.Get_ihi( ), nNodes, order, ProblemName );

    // --- Initialize timestepper ---
    TimeStepper SSPRK( &pin, &Grid );


    SlopeLimiter S_Limiter( &Grid, &pin );

    // --- Limit the initial conditions ---
    S_Limiter.ApplySlopeLimiter( uCF, &Grid, &Basis );

    // -- print run parameters ---
    PrintSimulationParameters( &Grid, &pin, CFL );

    // --- Timer ---
    Kokkos::Timer timer;

    // --- Evolution loop ---
    UInt iStep   = 0;
    UInt i_print = 100;
    UInt i_write = -1;
    UInt i_out   = 1;
    std::cout << " ~ Step\tt\tdt" << std::endl;
    while ( t < t_end && iStep >= 0 )
    {

      dt = ComputeTimestep_Fluid( uCF, &Grid, CFL );

      if ( t + dt > t_end )
      {
        dt = t_end - t;
      }

      if ( iStep % i_print == 0 )
      {
        std::printf( " ~ %d \t %.5e \t %.5e\n", iStep, t, dt );
      }

      SSPRK.UpdateFluid( Compute_Increment_Explicit, dt, uCF, &Grid, &Basis,
                         &S_Limiter );

      t += dt;

      // Write state
      if ( iStep % i_write == 0 )
      {
        WriteState( uCF, uPF, uAF, &Grid, &S_Limiter, ProblemName, t, order,
                    i_out );
        i_out += 1;
      }

      iStep++;
    }

    // --- Finalize timer ---
    Real time = timer.seconds( );
    std::printf( " ~ Done! Elapsed time: %f seconds.\n", time );
    ApplyBC_Fluid( uCF, &Grid, order, BC );
    WriteState( uCF, uPF, uAF, &Grid, &S_Limiter, ProblemName, t, order, -1 );
  }
  Kokkos::finalize( );

  return 0;
}

/**
 * Pick number of quadrature points in order to evaluate polynomial of
 * at least order^2.
 * ! Broken for nNodes > order !
 **/
int NumNodes( UInt order )
{
  if ( order <= 4 )
  {
    return order;
  }
  else
  {
    return order + 1;
  }
}

/**
 * Compute the CFL timestep restriction.
 **/
Real ComputeCFL( Real CFL, UInt order, UInt nStages,
                 UInt tOrder )
{
  Real c = 1.0;

  if ( nStages == tOrder ) c = 1.0;
  if ( nStages != tOrder )
  {
    if ( tOrder == 2 ) c = 4.0;
    if ( tOrder == 3 ) c = 2.65062919294483;
    if ( tOrder == 4 ) c = 1.50818004975927;
  }

  return c * CFL / ( ( 2.0 * (order)-1.0 ) );
}
