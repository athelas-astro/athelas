#pragma once

#include <cassert>
#include <tuple>

#include "Kokkos_Macros.hpp"

#include "bc/boundary_conditions_base.hpp"
#include "geometry/mesh.hpp"
#include "math/utils.hpp"
#include "utils/error.hpp"

namespace athelas::fluid {

/**
 * @struct FluidRiemannState
 * @brief Holds the state (i.e., left or right) for the fluid Riemann solver.
 * @note Contains specific volume, velocity, pressure, sound speed.
 */
struct FluidRiemannState {
  double tau;
  double v;
  double p;
  double cs;
};

struct FluidNumericalFlux {
  double u;
  double p;
};

KOKKOS_INLINE_FUNCTION
auto flux_fluid(const double V, const double P)
    -> std::tuple<double, double, double> {
  return {-V, P, P * V};
}

/**
 * Gudonov style numerical flux. Constructs v* and p* states.
 **/
KOKKOS_INLINE_FUNCTION
auto numerical_flux_gudonov(const FluidRiemannState &left,
                            const FluidRiemannState &right)
    -> std::tuple<double, double> {
  const double pL = left.p;
  const double pR = right.p;
  const double vL = left.v;
  const double vR = right.v;
  const double zL = left.cs / left.tau;
  const double zR = right.cs / right.tau;
  assert(pL > 0.0 && pR > 0.0 && "numerical_flux_gudonov :: negative pressure");
  const double Flux_U = (pL - pR + zR * vR + zL * vL) / (zR + zL);
  const double Flux_P = (zR * pL + zL * pR + zL * zR * (vL - vR)) / (zR + zL);
  return {Flux_U, Flux_P};
}
/**
 * Positivity preserving numerical flux. Constructs v* and p* states.
 * TODO(astrobarker): do I need tau_r_star if I construct p* with left?
 **/
KOKKOS_INLINE_FUNCTION
auto numerical_flux_gudonov_positivity(const FluidRiemannState &left,
                                       const FluidRiemannState &right)
    -> std::tuple<double, double> {
  using math::utils::pos_part;

  const double tauL = left.tau;
  const double tauR = right.tau;
  const double vL = left.v;
  const double vR = right.v;
  const double pL = left.p;
  const double pR = right.p;
  const double csL = left.cs;
  const double csR = right.cs;

  assert(pL > 0.0 && pR > 0.0 && "numerical_flux_gudonov :: negative pressure");

  const double pRmL = pR - pL; // [[p]]
  const double vRmL = vR - vL; // [[v]]
  /*
  const double zL   = std::max(
      std::max( std::sqrt( pos_part( pRmL ) / tauL ), -( vRmL ) / tauL ),
      csL / tauL );
  const double zR = std::max(
      std::max( std::sqrt( pos_part( -pR + pL ) / tauR ), -( vRmL ) / tauR ),
      csR / tauR );
  */
  const double zL = csL / tauL;
  const double zR = csR / tauR;
  const double z_sum = zL + zR;
  const double inv_z_sum = 1.0 / z_sum;
  const double zL2 = zL * zL;

  // get tau star states
  const double term1_l = tauL - (pRmL) / (zL2);
  const double term2_l = tauL + vRmL / zL;
  const double tau_l_star = (zL * term1_l + zR * term2_l) * inv_z_sum;

  /*
  const double term1_r = tauR + vRmL / zR;
  const double term2_r = tauR + pRmL / (zR * zR);
  const double tau_r_star = (zL * term1_r + zR * term2_r) / z_sum;
  */

  // vstar, pstar
  const double Flux_U = (-pRmL + zR * vR + zL * vL) * (inv_z_sum);
  const double Flux_P = pL - (zL2) * (tau_l_star - tauL);
  return {Flux_U, Flux_P};
}

KOKKOS_INLINE_FUNCTION
auto boundary_flux(const Boundary side, const bc::BcType type,
                   const FluidRiemannState &interior,
                   const FluidRiemannState &exterior,
                   const double surface_pressure = 0.0) -> FluidNumericalFlux {
  switch (type) {
  case bc::BcType::Outflow:
    return {.u = interior.v, .p = interior.p};
  case bc::BcType::Surface:
    return {.u = interior.v, .p = surface_pressure};
  case bc::BcType::Reflecting: {
    const FluidRiemannState wall{.tau = interior.tau,
                                 .v = -interior.v,
                                 .p = interior.p,
                                 .cs = interior.cs};
    const auto flux = (side == Boundary::Interior)
                          ? numerical_flux_gudonov_positivity(wall, interior)
                          : numerical_flux_gudonov_positivity(interior, wall);
    return {.u = 0.0, .p = std::get<1>(flux)};
  }
  case bc::BcType::Periodic: {
    const auto flux =
        (side == Boundary::Interior)
            ? numerical_flux_gudonov_positivity(exterior, interior)
            : numerical_flux_gudonov_positivity(interior, exterior);
    return {.u = std::get<0>(flux), .p = std::get<1>(flux)};
  }
  case bc::BcType::Marshak:
  case bc::BcType::InteriorFlux:
  case bc::BcType::FreeStreaming:
  case bc::BcType::Null:
    break; // radiation-only / invalid for fluid — excluded by runtime
           // validation. No default: a new BcType must be handled explicitly.
  }
  return {.u = 0.0, .p = 0.0};
}

KOKKOS_INLINE_FUNCTION
auto numerical_flux_fluid_with_boundary(
    const int face, const int inner_face, const int outer_face,
    const Kokkos::Array<bc::BoundaryConditionData, 2> &bcs,
    const FluidRiemannState &left, const FluidRiemannState &right)
    -> FluidNumericalFlux {
  if (face == inner_face) {
    return boundary_flux(Boundary::Interior, bcs[0].type, right, left,
                         bcs[0].surface_pressure);
  }
  if (face == outer_face) {
    return boundary_flux(Boundary::Exterior, bcs[1].type, left, right,
                         bcs[1].surface_pressure);
  }

  const auto [flux_u, flux_p] = numerical_flux_gudonov_positivity(left, right);
  return {.u = flux_u, .p = flux_p};
}

} // namespace athelas::fluid
