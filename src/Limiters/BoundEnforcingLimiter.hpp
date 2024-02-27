#ifndef _BOUNDENFORCINGLIMITER_HPP_
#define _BOUNDENFORCINGLIMITER_HPP_

#include "Abstractions.hpp"
#include "EoS.hpp"
#include "PolynomialBasis.hpp"

void LimitDensity( View3D U, const ModalBasis *Basis );
void LimitInternalEnergy( View3D U, const ModalBasis *Basis, const EOS *eos );
void ApplyBoundEnforcingLimiter( View3D U, const ModalBasis *Basis,
                                 const EOS *eos );
Real ComputeThetaState( const View3D U, const ModalBasis *Basis, const EOS *eos,
                        const Real theta, const int iCF, const int iX,
                        const int iN );
Real TargetFunc( const View3D U, const ModalBasis *Basis, const EOS *eos,
                 const Real theta, const int iX, const int iN );
Real Bisection( const View3D U, ModalBasis *Basis, EOS *eos, const int iX,
                const int iN );
Real Backtrace( const View3D U, const ModalBasis *Basis, const EOS *eos,
                const int iX, const int iN );
#endif // _BOUNDENFORCINGLIMITER_HPP_
