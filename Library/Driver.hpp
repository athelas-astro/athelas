#ifndef DRIVER_H
#define DRIVER_H

#include "Abstractions.hpp"

int NumNodes( unsigned int order );

Real ComputeCFL( Real CFL, unsigned int order, unsigned int nStages,
                 unsigned int tOrder );
#endif
