#ifndef _EOS_HPP_
#define _EOS_HPP_

/**
 * Specific EoS classes here
 **/

#include <variant>

#include "abstractions.hpp"
#include "eos_base.hpp"
#include "error.hpp"

class IdealGas : public EosBase<IdealGas> {
 public:
  IdealGas( ) = default;
  IdealGas( double gm ) : gamma( gm ) {
    if ( gamma <= 0.0 ) throw Error( " ! Adiabatic gamma <= 0.0!" );
  }

  Real PressureFromConserved( const Real Tau, const Real V, const Real EmT,
                              Real *lambda ) const;
  Real SoundSpeedFromConserved( const Real Tau, const Real V, const Real EmT,
                                Real *lambda ) const;
  Real TemperatureFromTauPressureAbar( const Real Tau, const Real P,
                                       const Real Abar, Real *lambda ) const;
  Real TemperatureFromTauPressure( const Real Tau, const Real P,
                                   Real *lambda ) const;
  Real RadiationPressure( const Real T, Real *lambda ) const;

 private:
  Real gamma;
};

/* placeholder */
class Stellar : public EosBase<Stellar> {
 public:
  Stellar( ) = default;

  Real PressureFromConserved( const Real Tau, const Real V, const Real EmT,
                              Real *lambda ) const;
  Real SoundSpeedFromConserved( const Real Tau, const Real V, const Real EmT,
                                Real *lambda ) const;
  Real TemperatureFromTauPressureAbar( const Real Tau, const Real P,
                                       const Real Abar, Real *lambda ) const;
  Real TemperatureFromTauPressure( const Real Tau, const Real P,
                                   Real *lambda ) const;
  Real RadiationPressure( const Real T, Real *lambda ) const;

 private:
  Real gamma;
};

// TODO: adjust when we support more than one EOS
using EOS = IdealGas;

#endif // _EOS_HPP_
