#ifndef _QUADRATURELIBRARY_HPP_
#define _QUADRATURELIBRARY_HPP_

#include "Abstractions.hpp"

namespace quadrature {
Real Jacobi_Matrix( int m, Real *aj, Real *bj );
void LG_Quadrature( int m, Real *nodes, Real *weights );
} // namespace quadrature

#endif // _QUADRATURELIBRARY_HPP_
