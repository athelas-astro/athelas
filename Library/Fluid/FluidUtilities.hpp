#ifndef FLUIDUTILITIES_H
#define FLUIDUTILITIES_H

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"
#include "Grid.hpp"

Real Flux_Fluid( const Real V, const Real P, const UInt iCF );
void NumericalFlux_Gudonov( const Real vL, const Real vR, const Real pL,
                            const Real pR, const Real zL, const Real zR,
                            Real &Flux_U, Real &Flux_P );
void NumericalFlux_Simple( const Real rhoL, const Real rhoR, const Real vL, 
                           const Real vR, const Real emL, const Real emR, 
                           const Real pL, const Real pR, 
                           const Real cL, const Real cR, 
                           Real &Flux_U, Real &Flux_P );
void NumericalFlux_HLLC( Real vL, Real vR, Real pL, Real pR, Real cL, Real cR,
                         Real rhoL, Real rhoR, Real &Flux_U, Real &Flux_P );
Real ComputeTimestep_Fluid( const Kokkos::View<Real ***> U,
                            const GridStructure *Grid, const Real CFL );
Real LorentzFactor( const Real V );

#endif
