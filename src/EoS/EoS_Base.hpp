#ifndef _EOS_BASE_HPP_
#define _EOS_BASE_HPP_

/**
 * define a base class using curiously recurring template pattern
 **/
#include "Abstractions.hpp"
#include "PolynomialBasis.hpp"

template < class EOS >
class EosBase {
  public:
    void PressureFromConserved( const Real Tau, const Real V, const Real Bm,
                                const Real EmT, Real &P ) const {
      static_cast<EOS const *>(this)->PressureFromConserved( Tau, V, Bm, EmT, P );
    }
    void Soun7dSpeedFromConserved( const Real Tau, const Real V, const Real Bm,
                                   const Real EmT, Real Cs ) const {
      static_cast<EOS const *>(this)->SoundSpeedFromConserved( 
              Tau, V, Bm, EmT, Cs );
    }
    void TemperatureFromTauPressureAbar( const Real Tau, const Real P, 
            const Real Abar, Real &T ) const {
        static_cast<EOS const *>(this)->TemperatureFromTauPressureAbar(
                Tau, P, Abar, T );
    }
    void TemperatureFromTauPressure( const Real Tau, const Real P, 
            Real &T ) const {
        static_cast<EOS const *>(this)->TemperatureFromPressureDensity( 
                Tau, P, T );
    }
    void RadiationPressure( const Real T, Real &Prad ) const {
        static_cast<EOS const *>(this)->RadiationPressure( T, Prad );
    }
    Real ComputeInternalEnergy( const View3D U, ModalBasis *Basis, 
            const UInt iX, const UInt iN ) const {
        return static_cast<EOS const *>(this)->ComputeInternalEnergy( U, Basis, 
                iX, iN );
    }
    Real ComputeInternalEnergy( const View3D U, const UInt iX ) const { 
        return static_cast<EOS const *>(this)->ComputeInternalEnergy( U, iX );
    }
};

#endif // _EOS_BASE_HPP_
