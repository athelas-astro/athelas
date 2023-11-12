#ifndef QUADRATURELIBRARY_H
#define QUADRATURELIBRARY_H

#include "Abstractions.hpp"

namespace quadrature {
Real Jacobi_Matrix( int m, Real *aj, Real *bj );
void LG_Quadrature( int m, Real *nodes, Real *weights );
}


#endif
