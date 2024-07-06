/**
 * File    :  error.hpp
 * --------------
 *
 * Author  : Brandon L. Barker
 * Purpose : Error throwing class, state checking
 **/

#ifndef ERROR_HPP_
#define ERROR_HPP_

#include <assert.h> /* assert */
#include <exception>
#include <stdexcept>

#include "constants.hpp"
#include "state.hpp"

class Error : public std::runtime_error {

 public:
  Error( const std::string &message ) : std::runtime_error( message ) {}
};

template <typename T>
void check_state( T state, const int ihi, const bool do_rad ) {

  auto uCR = state->Get_uCR( );
  auto uCF = state->Get_uCF( );

  const Real c = constants::c_cgs;

  Kokkos::parallel_for(
      "check_state", ihi, KOKKOS_LAMBDA( const int i ) {
        const int iX = i + 1; // hack
                              //
        const Real tau   = uCF( 0, iX, 0 ); // cell averages checked
        const Real vel   = uCF( 1, iX, 0 );
        const Real e_m   = uCF( 2, iX, 0 );
        const Real e_rad = uCR( 0, iX, 0 );
        const Real f_rad = uCR( 1, iX, 0 );

        assert( tau > 0.0 && " ! Check state :: Negative or zero density!" );
        assert( !std::isnan( tau ) &&
                " ! Check state :: Specific volume NaN!" );

        assert(
            std::fabs( vel ) < c &&
            " ! Check state :: Velocity reached or exceeded speed of light!" );
        assert( !std::isnan( vel ) && " ! Check state :: Velocity NaN!" );

        assert( e_m > 0.0 &&
                " Check state :: ! Negative or zero specific total energy!" );
        assert( !std::isnan( e_m ) &&
                " ! Check state :: Specific energy NaN!" );

        if ( do_rad ) {
          assert(
              e_rad > 0.0 &&
              " ! Check state :: Negative or zero radiation energy density!" );
          assert( !std::isnan( e_rad ) &&
                  " ! Check state :: Radiation energy NaN!" );

          // TODO: radiation flux bound
          // assert ( f_rad >= 0.0 && " Check state :: ! Negative or zero
          // radiation energy density!" );
          assert( !std::isnan( f_rad ) &&
                  " ! Check state :: Radiation flux NaN!" );
        }
      } );
}

#endif // ERROR_HPP_
