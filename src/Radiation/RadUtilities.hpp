#ifndef _RADUTILITIES_HPP_
#define _RADUTILITIES_HPP_

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"

Real Flux_Rad( Real E, Real F, Real P, Real V, UInt iRF );
Real Source_Rad( Real D, Real V, Real T, Real X, Real kappa, 
                 Real E, Real F, Real Pr, UInt iRF );
Real ComputeEmissivity( const Real D, const Real V, const Real Em );
Real ComputeOpacity( const Real D, const Real V, const Real Em );
Real ComputeClosure( const Real E, const Real F );
//void NumericalFlux_HLL( Real vL, Real vR, Real pL, Real pR, Real cL, Real cR,
//                         Real rhoL, Real rhoR, Real &Flux_U, Real &Flux_P );

#endif
