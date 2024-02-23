#ifndef _EOS_HPP_
#define _EOS_HPP_

/**
 * Specific EoS classes here
 **/

#include <variant>

#include "Abstractions.hpp"
#include "EoS_Base.hpp"


class IdealGas : public EosBase<IdealGas> {
  public:
    IdealGas() = default;
    IdealGas( double gm ) : gamma(gm) {}

    void PressureFromConserved( const Real Tau, const Real V, const Real Bm,
                                const Real EmT, Real &P ) const;
    void SoundSpeedFromConserved( const Real Tau, const Real V, const Real Bm,
                                  const Real EmT, Real &Cs ) const;
    void TemperatureFromTauPressureAbar( const Real Tau, const Real P, 
        const Real Abar, Real &T ) const;
    void TemperatureFromTauPressure( const Real Tau, const Real P, 
        Real &T ) const;
    void RadiationPressure( const Real T, Real &Prad ) const;
    Real ComputeInternalEnergy( const View3D U, ModalBasis *Basis, 
        const UInt iX, const UInt iN ) const;
    Real ComputeInternalEnergy( const View3D U, const UInt iX ) const;

  private:
    Real gamma;
};

/* placeholder */
class Stellar : public EosBase<Stellar> {
  public:
    Stellar() = default;

    void PressureFromConserved( const Real Tau, const Real V, const Real EmT, 
        Real &P ) const;
    void SoundSpeedFromConserved( const Real Tau, const Real V, const Real EmT, 
        Real &Cs ) const;
    void TemperatureFromTauPressureAbar( const Real Tau, const Real P, 
        const Real Abar, Real &T ) const;
    void TemperatureFromTauPressure( const Real Tau, const Real P, 
        Real &T ) const;
    void RadiationPressure( const Real T, Real &Prad ) const;
    Real ComputeInternalEnergy( const View3D U, ModalBasis *Basis, 
        const UInt iX, const UInt iN ) const;
    Real ComputeInternalEnergy( const View3D U, const UInt iX ) const;

  private:
    Real gamma;
};

// TODO: adjust when we support more than one EOS
using EOS = IdealGas;

#endif // _EOS_HPP_
