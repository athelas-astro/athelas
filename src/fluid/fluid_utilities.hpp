#pragma once

#include <cassert>
#include <tuple>

#include "Kokkos_Macros.hpp"

#include "math/utils.hpp"

namespace athelas::fluid {

KOKKOS_INLINE_FUNCTION
auto flux_fluid(const double V, const double P)
    -> std::tuple<double, double, double> {
  return {-V, P, P * V};
}

/**
 * Gudonov style numerical flux. Constructs v* and p* states.
 **/
KOKKOS_INLINE_FUNCTION
auto numerical_flux_gudonov(const double vL, const double vR, const double pL,
                            const double pR, const double zL, const double zR)
    -> std::tuple<double, double> {
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
auto numerical_flux_gudonov_positivity(const double tauL, const double tauR,
                                       const double vL, const double vR,
                                       const double pL, const double pR,
                                       const double csL, const double csR)
    -> std::tuple<double, double> {
  assert(pL > 0.0 && pR > 0.0 && "numerical_flux_gudonov :: negative pressure");
  using math::utils::pos_part;
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

/**
 * Gudonov style numerical flux. Constructs v* and p* states.
 **/
KOKKOS_INLINE_FUNCTION
void numerical_flux_hllc(double vL, double vR, double pL, double pR, double cL,
                         double cR, double rhoL, double rhoR, double &Flux_U,
                         double &Flux_P) {
  double const aL = vL - cL; // left wave speed estimate
  double const aR = vR + cR; // right wave speed estimate
  Flux_U = (rhoR * vR * (aR - vR) - rhoL * vL * (aL - vL) + pL - pR) /
           (rhoR * (aR - vR) - rhoL * (aL - vL));
  Flux_P = rhoL * (vL - aL) * (vL - Flux_U) + pL;
}
} // namespace athelas::fluid
