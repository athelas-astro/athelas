/**
 * @file characteristic_decomposition.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Hydro characteristic decomposition
 *
 * @details Implements a characteristic decomposition of the hydro variables.
 *          Currently this is only implemented ofr an ideal EOS.
 *          TODO(astrobarker): Write down for radiation.
 *
 * @note: Broken!
 */

#pragma once

#include <cmath>

#include "eos/eos_variant.hpp"

namespace athelas {

template <class T1, class T2, class EOS>
void compute_characteristic_decomposition(T1 U, T2 R, T2 R_inv, EOS eos) {

  const double tau = U(0);
  const double v = U(1);
  const double Em_T = U(2);

  auto lambda = nullptr;
  const double p = pressure_from_conserved(eos, tau, v, Em_T, lambda);

  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  const double v2 = v * v;
  const double cs = std::sqrt(p * gamma * tau);
  const double z = cs / tau;
  const double chi = p / (z);

  /* --- Thermodynamic Derivatives of pressure --- */

  // const double p_Tau = - (gamma - 1.0) * Em / ( tau * tau );
  // const double p_Em  = + (gamma - 1.0) / tau;
  // const double p_T_E = - Em / tau; // ratio of derivatives

  // Eigenvalues are 0, +/- Cs...
  // const double lam = std::sqrt( p * p_Em - p_Tau );

  /*  --- Compute Matrix Elements --- */
  const double R00 = gm1 / p;
  const double R01 = -1 / (z * (chi - v));
  const double R02 = -1 / (z * (chi + v));
  const double R10 = 0;
  const double R11 = gamma / (-cs + gamma * v);
  const double R12 = gamma / (cs + gamma * v);
  const double R20 = 1;
  const double R21 = 1;
  const double R22 = 1;
  const double R_inv00 = p / gamma;
  const double R_inv01 = -v / gamma;
  const double R_inv02 = 1.0 / gamma;
  const double R_inv10 = (1.0 / 2.0) * (-p + v * z) / gamma;
  const double R_inv11 = 0.5 * (-cs + v) / gamma + 0.5 * v2 * gm1 / cs;
  const double R_inv12 = 0.5 * (1.0 / gamma - v / cs) * gm1;
  const double R_inv20 = (1.0 / 2.0) * (-p - v * z) / gamma;
  const double R_inv21 = 0.5 * (cs + v) / gamma - 0.5 * v2 * gm1 / cs;
  const double R_inv22 = 0.5 * (1.0 / gamma + v / cs) * gm1;

  R(0, 0) = R00;
  R(0, 1) = R01;
  R(0, 2) = R02;

  R(1, 0) = R10;
  R(1, 1) = R11;
  R(1, 2) = R12;

  R(2, 0) = R20;
  R(2, 1) = R21;
  R(2, 2) = R22;

  R_inv(0, 0) = R_inv00;
  R_inv(0, 1) = R_inv01;
  R_inv(0, 2) = R_inv02;

  R_inv(1, 0) = R_inv10;
  R_inv(1, 1) = R_inv11;
  R_inv(1, 2) = R_inv12;

  R_inv(2, 0) = R_inv20;
  R_inv(2, 1) = R_inv21;
  R_inv(2, 2) = R_inv22;
}

} // namespace athelas
