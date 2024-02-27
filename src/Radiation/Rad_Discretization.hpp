#ifndef _RAD_DISCRETIZATION_HPP_
#define _RAD_DISCRETIZATION_HPP_

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"
#include "EoS.hpp"

void ComputeIncrement_Rad_Divergence( const View3D uCR, const View3D uCF,
                                      GridStructure &rid,
                                      const ModalBasis *Basis, const EOS *eos,
                                      View3D dU, View3D Flux_q,
                                      View2D dFlux_num, View2D uCF_F_L,
                                      View2D uCF_F_R, View1D Flux_U,
                                      View1D Flux_P, View1D Flux_B );

void Compute_Increment_Explicit_Rad( const View3D uCR, const View3D uCF,
                                     GridStructure &Grid,
                                     const ModalBasis *Basis, const EOS *eos,
                                     View3D dU, View3D Flux_q, View2D dFlux_num,
                                     View2D uCR_F_L, View2D uCR_F_R,
                                     View1D Flux_U, View1D Flux_P,
                                     View1D Flux_B,
                                     const Options opts );

void ComputeIncrement_Rad_Source( const View3D uCR, const View3D uCF,
                                  GridStructure &Grid, const ModalBasis *Basis,
                                  const EOS *eos, View3D dU );
#endif // _RAD_DISCRETIZATION_HPP_
