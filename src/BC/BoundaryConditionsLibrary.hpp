#ifndef _BOUNDARYCONDITIONSLIBRARY_HPP_
#define _BOUNDARYCONDITIONSLIBRARY_HPP_

#include "Abstractions.hpp"
#include "Grid.hpp"

void ApplyBC( Kokkos::View<Real ***> uCF, GridStructure *Grid, const int order,
              const std::string BC );

#endif // _BOUNDARYCONDITIONSLIBRARY_HPP_
