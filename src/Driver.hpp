#ifndef _DRIVER_HPP_
#define _DRIVER_HPP_

#include "Abstractions.hpp"

int NumNodes( int order );

Real ComputeCFL( Real CFL, int order, int nStages, int tOrder );
#endif // _DRIVER_HPP_
