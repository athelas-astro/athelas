#ifndef CON2PRIM_H
#define CON2PRIM_H

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"
#include "PolynomialBasis.hpp"

void Con2Prim( Real U0, Real U1, Real U2, Real &tau, Real &v, Real &Em, 
               Real &P );

Real C2P_Newton( Real x0, const Real a, const Real xi, const Real eta );

#endif
