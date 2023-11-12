#ifndef BOUNDARYCONDITIONSLIBRARY_H
#define BOUNDARYCONDITIONSLIBRARY_H

#include "Abstractions.hpp"

void ApplyBC( Kokkos::View<Real ***> uCF, GridStructure *Grid,
              const UInt order, const std::string BC );

#endif
