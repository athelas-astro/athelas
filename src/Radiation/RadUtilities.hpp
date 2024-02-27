#ifndef _RADUTILITIES_HPP_
#define _RADUTILITIES_HPP_

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"

Real FluxFactor( const Real E, const Real F );
Real Flux_Rad( Real E, Real F, Real P, Real V, int iCR );
void RadiationFourForce( Real D, Real V, Real T, Real kappa, Real E, Real F,
                         Real Pr, Real &G0, Real &G );
Real Source_Rad( Real D, Real V, Real T, Real X, Real kappa, Real E, Real F,
                 Real Pr, int iCR );
Real ComputeEmissivity( const Real D, const Real V, const Real Em );
Real ComputeOpacity( const Real D, const Real V, const Real Em );
Real ComputeClosure( const Real E, const Real F );
Real Lambda_HLL( const Real f, const int sign );
void llf_flux( const Real Fp, const Real Fm, const Real Up, const Real Um,
               const Real alpha, Real &out );
void NumericalFlux_HLL_Rad( const Real E_L, const Real E_R, const Real F_L,
                            const Real F_R, const Real P_L, const Real P_R,
                            const Real V_L, const Real V_R, Real &Flux_E,
                            Real &Flux_F );

#endif // _RADUTILITIES_HPP_
