#ifndef _ABSTRACTIONS_HPP_
#define _ABSTRACTIONS_HPP_

#include "Kokkos_Core.hpp"

#include "Kokkos_Core.hpp"

using Real = double;
using UInt = unsigned int;
using View3D = Kokkos::View<Real ***>;
using View2D = Kokkos::View<Real **>;
using View1D = Kokkos::View<Real *>;

/* Where to put this? */
namespace PolyBasis {
  enum PolyBasis { Legendre, Taylor };
}

#endif // _ABSTRACTIONS_HPP_
