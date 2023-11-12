#ifndef _UTILITIES_HPP_
#define _UTILITIES_HPP_

// Implements a typesafe sgn function
// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename T>
constexpr int sgn( T val )
{
  return ( T( 0 ) < val ) - ( val < T( 0 ) );
}

#endif // _UTILITIES_HPP_
