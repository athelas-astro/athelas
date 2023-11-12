#ifndef IOLIBRARY_H
#define IOLIBRARY_H

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"
#include "ProblemIn.hpp"

struct GridType
{
  Real r{ };
};

struct DataType
{
  Real x{ };
};

void WriteState( Kokkos::View<Real ***> uCF, Kokkos::View<Real ***> uPF,
                 GridStructure Grid, SlopeLimiter *SL, 
                 const std::string ProblemName, Real time,
                 UInt order, int i_write );

void PrintSimulationParameters( GridStructure Grid, ProblemIn *pin, const Real CFL );

void WriteBasis( ModalBasis *Basis, UInt ilo, UInt ihi, UInt nNodes, UInt order,
                 std::string ProblemName );

#endif
