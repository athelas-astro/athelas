/**
 * @file gravity_potential.hpp
 * --------------
 *
 * @brief Newtonian gravitational potential per unit mass.
 *
 * @details Single source of truth for the potential phi_q used by both the
 *          diagnostic W_h = sum_q H_q phi_q
 * (analysis::total_gravitational_energy) and the limiter energy correction.
 * Both must use the identical phi so the reported potential energy and the
 * correction that cancels its limiter-induced change stay consistent.
 *
 *          phi is defined by the same relation as the gravity pressure,
 *          d phi / dr = -g, so that the energy source conserves W_h:
 *            - Spherical self-gravity: g = -G M / r^2  =>  phi = -G M / r.
 *            - Constant background:    g = -gval       =>  phi = +gval r.
 *          The energy source builds the gravity pressure from f = -g / A, so
 *          this is the potential it actually conserves.
 **/

#pragma once

#include "basic_types.hpp" // GravityModel, KOKKOS_INLINE_FUNCTION (Kokkos_Core)
#include "utils/constants.hpp"

namespace athelas::gravity {

KOKKOS_INLINE_FUNCTION
auto gravitational_potential(const GravityModel model, const double gval,
                             const double enclosed_mass, const double r)
    -> double {
  if (model == GravityModel::Spherical) {
    return -constants::G_GRAV * enclosed_mass / r;
  }
  return gval * r;
}

} // namespace athelas::gravity
