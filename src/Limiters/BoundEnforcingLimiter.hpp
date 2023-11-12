#ifndef _BOUNDENFORCINGLIMITER_HPP_
#define _BOUNDENFORCINGLIMITER_HPP_

#include "Abstractions.hpp"

void LimitDensity( View3D U, ModalBasis *Basis );
void LimitInternalEnergy( View3D U, ModalBasis *Basis, 
                          EOS *eos );
void ApplyBoundEnforcingLimiter( View3D U, ModalBasis *Basis, 
                                 EOS *eos );
Real ComputeThetaState( const View3D U, ModalBasis *Basis, 
                        EOS *eos, const Real theta, const UInt iCF, 
                        const UInt iX, const UInt iN );
Real TargetFunc( const View3D U, ModalBasis *Basis, EOS *eos,
                 const Real theta, const UInt iX, const UInt iN );
Real Bisection( const View3D U, ModalBasis *Basis, EOS *eos,
                const UInt iX, const UInt iN );
Real Backtrace( const View3D U, ModalBasis *Basis, EOS *eos,
                const UInt iX, const UInt iN );
#endif // _BOUNDENFORCINGLIMITER_HPP_
