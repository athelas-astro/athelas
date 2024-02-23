#ifndef TIMESTEPPER_H
#define TIMESTEPPER_H

/**
 * File     :  Timestepper.hpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for SSPRK timestepping
 **/

#include "Abstractions.hpp"
#include "ProblemIn.hpp"
#include "state.hpp"

typedef void ( *UpdateFunc )( const View3D, const View3D, 
                              GridStructure &,
                              ModalBasis *, EOS *eos, View3D,
                              View3D, View2D,
                              View2D, View2D,
                              View1D, View1D, View1D,
                              const Options opts );

class TimeStepper
{

 public:
  // TODO: Is it possible to initialize Grid_s from Grid directly?
  TimeStepper( ProblemIn *pin, GridStructure &Grid );

  void InitializeTimestepper( );

  void UpdateFluid( UpdateFunc ComputeIncrement, const Real dt,
                    State *state, GridStructure &Grid,
                    ModalBasis *Basis, EOS *eos, SlopeLimiter *S_Limiter, 
                    const Options opts );

 private:
  const UInt mSize;
  const UInt nStages;
  const UInt tOrder;
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

#endif
