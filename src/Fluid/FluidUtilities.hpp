#ifndef FLUIDUTILITIES_H
#define FLUIDUTILITIES_H

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"
#include "Grid.hpp"
#include "EoS.hpp"

void ComputePrimitiveFromConserved( View3D uCF,
                                    View3D uPF,
                                    ModalBasis *Basis, GridStructure *Grid );
Real Flux_Fluid( const Real V, const Real P, const UInt iCF );
Real Source_Fluid_Rad( Real D, Real V, Real T, Real X, Real kappa, 
                       Real E, Real F, Real Pr, UInt iCF );
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
Real LorentzFactor( const Real V );
Real ComputeTimestep_Fluid( const View3D U,
                            const GridStructure *Grid, EOS *eos, 
                            const Real CFL );

#endif
