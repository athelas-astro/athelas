#ifndef INITIALIZATION_H
#define INITIALIZATION_H

#include "Abstractions.hpp"

void InitializeFields( Kokkos::View<Real ***> uCF, Kokkos::View<Real ***> uPF,
                       GridStructure *Grid, const unsigned int pOrder,
                       const std::string ProblemName );
#endif
